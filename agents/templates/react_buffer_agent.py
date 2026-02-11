"""
React Agent with Exploration Buffer and Residual-Guided Decoding for ARC-AGI-3.

Adapted from R2 (Residual-Guided Decoding with Test-Time In-Context Learning).

Core ideas:
1. **Exploration Buffer**: Stores past game episodes (state-action trajectories)
   with embeddings for similarity retrieval.
2. **Residual Function**: Scores candidate actions by aggregating over buffer
   entries with cosine similarity (TRL Eq. 15-16).
3. **Temporal Credit Assignment**: Success credits later steps more, failure
   blames earlier steps more (TRL Eq. 22).
4. **Dual Buffer Pattern**: Long-term buffer (B_train) across retries +
   working memory (B_infer) per attempt.
5. **ReAct Loop**: Observe -> Retrieve from buffer -> Reason -> Act -> Learn.

On GAME_OVER the agent does NOT exit. Instead it stores the failed trajectory,
resets the game, and retries with accumulated experience informing the LLM
via buffer-augmented prompts and residual-guided action hints.

Tools available to the LLM:
- Game actions: RESET, ACTION1-ACTION6
- update_game_notes: Persist discovered knowledge to the arc_game_playing skill

Skills injected into system prompt:
- grid_perception: How to read structured perception data
- arc_game_playing: Game strategy, exploration, and the discovered knowledge DB

Usage:
    python main.py -a reactbuffer [-g GAME_ID]
"""

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..agent import Agent
from ..skills_loader import format_skills_for_prompt, load_local_skills
from .grid_perception import GridPerception, ObjectAnnotator
from .llm_agents import LLM

logger = logging.getLogger()

# Sections in arc_game_playing/SKILL.md the agent can update via tool
KNOWLEDGE_HEADERS = {
    "action_mappings": "### Action Mappings",
    "game_rules": "### Game Rules",
    "level_strategies": "### Level Strategies",
    "object_roles": "### Object Roles",
    "tips": "### Tips",
}


# =============================================================================
# State Encoding Helpers
# =============================================================================


def encode_frame_compact(frame_data: FrameData) -> str:
    """Encode a frame into a compact text representation for embedding.

    Produces a fingerprint like:
        state:PLAYING|score:2|g0:[1:40;5:120;10:542]
    that captures the essential grid structure without full grid data.
    """
    parts: list[str] = []
    parts.append(f"state:{frame_data.state.name}")
    parts.append(f"score:{getattr(frame_data, 'score', frame_data.levels_completed)}")

    if frame_data.frame:
        for grid_idx, grid in enumerate(frame_data.frame):
            value_counts: dict[int, int] = {}
            for row in grid:
                for val in row:
                    if val != 0:
                        value_counts[val] = value_counts.get(val, 0) + 1
            fingerprint = ";".join(
                f"{v}:{cnt}" for v, cnt in sorted(value_counts.items())
            )
            parts.append(f"g{grid_idx}:[{fingerprint}]")

    return "|".join(parts)


def compute_frame_diff(prev_frame: FrameData, curr_frame: FrameData) -> str:
    """Compute a compact diff between two consecutive frames.

    Returns a string like:
        score:1->2|state:PLAYING->PLAYING|g0:D42cells
    """
    diffs: list[str] = []

    prev_score = getattr(prev_frame, "score", prev_frame.levels_completed)
    curr_score = getattr(curr_frame, "score", curr_frame.levels_completed)
    if curr_score != prev_score:
        diffs.append(f"score:{prev_score}->{curr_score}")

    if prev_frame.state != curr_frame.state:
        diffs.append(f"state:{prev_frame.state.name}->{curr_frame.state.name}")

    if prev_frame.frame and curr_frame.frame:
        for g_idx in range(min(len(prev_frame.frame), len(curr_frame.frame))):
            prev_grid = prev_frame.frame[g_idx]
            curr_grid = curr_frame.frame[g_idx]
            changed = 0
            for r in range(min(len(prev_grid), len(curr_grid))):
                for c in range(min(len(prev_grid[r]), len(curr_grid[r]))):
                    if prev_grid[r][c] != curr_grid[r][c]:
                        changed += 1
            if changed > 0:
                diffs.append(f"g{g_idx}:D{changed}cells")

    return "|".join(diffs) if diffs else "no_change"


# =============================================================================
# Lightweight Embedding (no external dependencies beyond numpy)
# =============================================================================


class SimpleEmbedding:
    """TF-IDF bag-of-words embedding for fast local text similarity.

    Modeled after ``r2/src/alfworld/llm_agent.py::BagOfWordsEmbedding`` but
    simplified for the ARC-AGI-3 domain where texts are short state/action
    fingerprints rather than natural language.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._doc_count: int = 0
        self._word_doc_count: dict[str, int] = {}

    def encode(self, text: str) -> np.ndarray:
        """Encode *text* into a TF-IDF vector of length ``self.dim``."""
        words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
        words = [w for w in words if len(w) > 1]

        unique_words = set(words)
        self._doc_count += 1
        for w in unique_words:
            self._word_doc_count[w] = self._word_doc_count.get(w, 0) + 1
            if w not in self._vocab and len(self._vocab) < self.dim:
                self._vocab[w] = len(self._vocab)

        vec = np.zeros(self.dim, dtype=np.float32)
        word_counts: dict[str, int] = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        for w, cnt in word_counts.items():
            if w in self._vocab:
                tf = cnt / max(len(words), 1)
                idf = (
                    np.log(
                        (self._doc_count + 1)
                        / (self._word_doc_count.get(w, 1) + 1)
                    )
                    + 1
                )
                vec[self._vocab[w]] = tf * idf

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# =============================================================================
# Episode Data Structures
# =============================================================================


@dataclass
class StepRecord:
    """A single step in a game episode."""

    state_text: str  # compact state fingerprint
    action: str  # GameAction name
    frame_diff: str  # what changed after the action
    score_before: int
    score_after: int


@dataclass
class GameEpisode:
    """A complete game episode from RESET to WIN/GAME_OVER.

    Mirrors ``r2/src/alfworld/trajectory_buffer.py::Trajectory`` adapted
    for the ARC-AGI-3 discrete-grid domain.
    """

    game_id: str
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    final_score: int = 0
    total_actions: int = 0
    embedding: Optional[np.ndarray] = None
    step_embeddings: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "game_id": self.game_id,
            "steps": [asdict(s) for s in self.steps],
            "success": self.success,
            "final_score": self.final_score,
            "total_actions": self.total_actions,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "step_embeddings": [e.tolist() for e in self.step_embeddings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameEpisode":
        """Deserialize from a dict."""
        steps = [StepRecord(**s) for s in data.get("steps", [])]
        emb = data.get("embedding")
        step_embs = data.get("step_embeddings", [])
        return cls(
            game_id=data["game_id"],
            steps=steps,
            success=data.get("success", False),
            final_score=data.get("final_score", 0),
            total_actions=data.get("total_actions", 0),
            embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
            step_embeddings=[np.array(e, dtype=np.float32) for e in step_embs],
        )

    def summary(self, max_steps: int = 10) -> str:
        """Human-readable episode summary for LLM prompt injection."""
        outcome = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Episode ({outcome}, score={self.final_score}, "
            f"{self.total_actions} actions):"
        ]

        if len(self.steps) > max_steps:
            show = self.steps[: max_steps // 2] + self.steps[-(max_steps // 2) :]
        else:
            show = self.steps

        for i, step in enumerate(show):
            marker = ""
            if step.score_after > step.score_before:
                marker = " [SCORE+]"
            elif "no_change" in step.frame_diff:
                marker = " [NO EFFECT]"
            lines.append(f"  Step {i}: {step.action} -> {step.frame_diff}{marker}")

        return "\n".join(lines)


# =============================================================================
# Exploration Buffer
# =============================================================================


class EpisodeBuffer:
    """Buffer for storing and retrieving game episodes.

    Uses numpy cosine similarity instead of FAISS to avoid extra dependencies.
    Mirrors ``r2/src/alfworld/trajectory_buffer.py::TrajectoryBuffer`` with
    FIFO eviction and embedding-based retrieval.
    """

    def __init__(self, capacity: int = 100, emb_dim: int = 512):
        self.capacity = capacity
        self.emb_dim = emb_dim
        self.episodes: list[GameEpisode] = []

    def add(self, episode: GameEpisode) -> None:
        """Add an episode, evicting the oldest if at capacity (FIFO)."""
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: int = 5,
        success_only: bool = False,
        game_id: Optional[str] = None,
    ) -> list[tuple[GameEpisode, float]]:
        """Retrieve top-k similar episodes by cosine similarity.

        Args:
            query_emb: Query embedding vector.
            k: Number of episodes to return.
            success_only: Only return successful episodes.
            game_id: Filter to episodes from this game.

        Returns:
            List of ``(episode, similarity_score)`` tuples sorted descending.
        """
        candidates = self.episodes
        if success_only:
            candidates = [e for e in candidates if e.success]
        if game_id is not None:
            candidates = [e for e in candidates if e.game_id == game_id]

        if not candidates:
            return []

        results: list[tuple[GameEpisode, float]] = []
        for ep in candidates:
            if ep.embedding is not None:
                sim = float(np.dot(query_emb, ep.embedding))
                results.append((ep, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_size(self) -> int:
        return len(self.episodes)

    def get_success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def clear(self) -> None:
        self.episodes.clear()


# =============================================================================
# Residual Function  (TRL Eq. 15-16, 20, 22)
# =============================================================================


class ResidualFunction:
    """Score state-action pairs using buffer entries with temporal credit.

    Implements the multi-step residual from
    ``r2/src/alfworld/residual.py::ALFWorldResidualFunction`` adapted for
    the ARC-AGI-3 discrete-action domain:

        R_kappa(s_t, a; B) = sum_i  w_i * sum_{t'} phi(e(s_t,a), e_i,t') * c(tau_i, t')

    where:
        w_i   = exp(beta * R(tau_i)) / Z                          (Eq. 16)
        phi   = cosine_similarity                                  (Eq. 20)
        c     = R(tau_i) * omega(t'; T)                            (Eq. 22)
        omega = t'/T  for success,  1-t'/T  for failure
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _compute_weights(self, rewards: list[float]) -> np.ndarray:
        """Softmax weights by reward (Eq. 16)."""
        logits = np.array([self.beta * r for r in rewards], dtype=np.float64)
        logits -= logits.max()
        w = np.exp(logits)
        total = w.sum()
        if total > 0:
            w /= total
        return w

    @staticmethod
    def _temporal_credit(reward: float, step_idx: int, total_steps: int) -> float:
        """Per-step credit c(tau, t') with temporal weighting (Eq. 22).

        Success: later steps credited more  (omega = t'/T).
        Failure: earlier steps blamed more  (omega = 1 - t'/T).
        """
        if total_steps <= 0:
            return 0.0
        t = (step_idx + 1) / total_steps
        omega = t if reward > 0 else (1.0 - t)
        return reward * omega

    def score_action(
        self,
        sa_emb: np.ndarray,
        buffer: EpisodeBuffer,
        query_emb: np.ndarray,
        k: int = 5,
    ) -> float:
        """Compute R_kappa(s_t, a; B) for a single state-action embedding.

        Args:
            sa_emb: State-action embedding ``e(s_t, a)``.
            buffer: Episode buffer to retrieve from.
            query_emb: Episode-level query for retrieval.
            k: Number of entries to retrieve.

        Returns:
            Scalar residual score.
        """
        entries = buffer.retrieve(query_emb, k=k)
        if not entries:
            return 0.0

        rewards = [
            (1.0 if ep.success else -0.5) for ep, _ in entries
        ]
        weights = self._compute_weights(rewards)

        residual = 0.0
        for idx, (ep, task_sim) in enumerate(entries):
            if ep.step_embeddings:
                T = len(ep.step_embeddings)
                step_score = 0.0
                for s_idx, step_emb in enumerate(ep.step_embeddings):
                    sim = self._cosine_sim(sa_emb, step_emb)
                    credit = self._temporal_credit(
                        1.0 if ep.success else -0.5, s_idx, T
                    )
                    step_score += sim * credit
                residual += weights[idx] * step_score
            else:
                credit = 1.0 if ep.success else -0.5
                residual += weights[idx] * task_sim * credit

        return residual


# =============================================================================
# Loop Detection
# =============================================================================


@dataclass
class LoopDetection:
    """Result of loop/stuck analysis on the current action history."""

    is_looping: bool = False
    cycle_actions: list[str] = field(default_factory=list)
    cycle_length: int = 0
    cycle_repetitions: int = 0
    steps_since_score_change: int = 0
    is_stuck: bool = False
    no_effect_actions: dict[str, int] = field(default_factory=dict)
    recent_action_summary: str = ""


class LoopDetector:
    """Detects repetitive action cycles and score staleness.

    Runs after every step to provide the LLM with awareness of its own
    repetitive behavior — something it otherwise loses due to message
    truncation (MESSAGE_LIMIT).
    """

    STUCK_THRESHOLD: int = 8
    HISTORY_WINDOW: int = 15
    MIN_CYCLE_LENGTH: int = 2
    MAX_CYCLE_LENGTH: int = 8
    MIN_REPETITIONS: int = 2
    SEVERE_REPETITIONS: int = 3

    def reset(self) -> None:
        """Reset detector state (called on retry)."""
        pass  # Stateless — all analysis is done from step history

    def analyze(self, steps: list[StepRecord]) -> LoopDetection:
        """Run all loop/stuck checks and return a LoopDetection result."""
        if not steps:
            return LoopDetection()

        actions = [s.action for s in steps]
        cycle_actions, cycle_length, cycle_reps = self._detect_cycle(actions)
        steps_since = self._compute_steps_since_score_change(steps)
        no_effect = self._compute_no_effect_streaks(steps)

        # Build recent action summary (last HISTORY_WINDOW actions)
        window = actions[-self.HISTORY_WINDOW :]
        summary_parts: list[str] = []
        for i, act in enumerate(window):
            idx = len(actions) - len(window) + i
            step = steps[idx]
            markers = ""
            if step.score_after > step.score_before:
                markers = " [SCORE+]"
            elif "no_change" in step.frame_diff:
                markers = " [NO EFFECT]"
            summary_parts.append(f"Step {idx}: {act}{markers}")

        return LoopDetection(
            is_looping=cycle_reps >= self.MIN_REPETITIONS,
            cycle_actions=cycle_actions,
            cycle_length=cycle_length,
            cycle_repetitions=cycle_reps,
            steps_since_score_change=steps_since,
            is_stuck=steps_since >= self.STUCK_THRESHOLD,
            no_effect_actions=no_effect,
            recent_action_summary="\n".join(summary_parts),
        )

    def _detect_cycle(
        self, actions: list[str]
    ) -> tuple[list[str], int, int]:
        """Suffix-match cycle detection.

        For each candidate cycle length L (from MAX down to MIN), check if the
        last L*k actions consist of k repetitions of the same L-action pattern.

        Returns (cycle_actions, cycle_length, repetitions).
        """
        n = len(actions)
        best: tuple[list[str], int, int] = ([], 0, 0)

        for length in range(self.MAX_CYCLE_LENGTH, self.MIN_CYCLE_LENGTH - 1, -1):
            if length > n:
                continue
            # The candidate pattern is the last `length` actions
            pattern = actions[-length:]
            reps = 1
            # Check how many times this pattern repeats going backwards
            pos = n - length
            while pos >= length:
                segment = actions[pos - length : pos]
                if segment == pattern:
                    reps += 1
                    pos -= length
                else:
                    break

            if reps >= self.MIN_REPETITIONS and reps > best[2]:
                best = (pattern, length, reps)

        return best

    def _compute_steps_since_score_change(self, steps: list[StepRecord]) -> int:
        """Walk backwards to find the last step where the score changed."""
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].score_after != steps[i].score_before:
                return len(steps) - 1 - i
        return len(steps)

    def _compute_no_effect_streaks(
        self, steps: list[StepRecord]
    ) -> dict[str, int]:
        """Count consecutive no-effect uses per action at end of history.

        Returns {action_name: consecutive_no_effect_count} for actions whose
        most recent uses (at the tail of the history) had no effect.
        """
        result: dict[str, int] = {}
        seen_effective: set[str] = set()

        for step in reversed(steps[-self.HISTORY_WINDOW :]):
            if step.action in seen_effective:
                continue
            if "no_change" in step.frame_diff:
                result[step.action] = result.get(step.action, 0) + 1
            else:
                seen_effective.add(step.action)

        return result


# =============================================================================
# React Buffer Agent
# =============================================================================


class ReactBufferAgent(LLM, Agent):
    """React Agent with Exploration Buffer and Residual-Guided Decoding.

    Explores ARC-AGI-3 games by:
    1. Playing normally until WIN or GAME_OVER.
    2. On GAME_OVER: storing the failed trajectory and resetting (up to
       ``MAX_RETRIES`` times) instead of exiting.
    3. Injecting buffer-retrieved past experiences into the LLM prompt so
       the model can learn from prior failures/successes.
    4. Computing residual action-hints that quantify which actions are
       more similar to past successful steps vs. failed ones.

    Tools:
    - Game actions: RESET, ACTION1-ACTION6
    - update_game_notes: Persist discovered knowledge to arc_game_playing skill

    Skills (system prompt):
    - grid_perception: How to read structured perception data
    - arc_game_playing: Game strategy + discovered knowledge DB
    """

    MAX_ACTIONS: int = 300
    MAX_RETRIES: int = 5
    DO_OBSERVATION: bool = True
    MODEL: str = "o4-mini"
    REASONING_EFFORT: Optional[str] = "medium"
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 20

    # Buffer / residual configuration
    BUFFER_CAPACITY: int = 100
    NUM_RETRIEVE: int = 3
    RESIDUAL_BETA: float = 1.0
    EMB_DIM: int = 512

    # Only inject these two skills into the system prompt
    ALLOWED_SKILLS: tuple[str, ...] = ("grid_perception", "arc_game_playing")

    # Max update_game_notes calls per turn (prevent infinite loops)
    MAX_NOTES_PER_TURN: int = 3

    # API timeout in seconds
    API_TIMEOUT: int = 120
    API_RETRIES: int = 2

    # Save checkpoint every N steps (0 = only on episode end)
    CHECKPOINT_INTERVAL: int = 5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._embedder = SimpleEmbedding(dim=self.EMB_DIM)
        self.episode_buffer = EpisodeBuffer(
            capacity=self.BUFFER_CAPACITY, emb_dim=self.EMB_DIM
        )
        self.residual_fn = ResidualFunction(beta=self.RESIDUAL_BETA)

        # Grid perception engine (replaces raw grid dump with structured analysis)
        self.perception = GridPerception()

        # VLM object annotator — identifies objects + detects changes
        self._annotator = ObjectAnnotator(model=self._model)
        self._cached_entity_text: str = ""  # latest entity identification text
        self._entity_step: int = -1         # step when last identified entities
        self.ENTITY_INTERVAL: int = 8       # re-identify entities every N steps
        self._prev_bbox_image = None        # previous frame's bbox image for diff
        self._cached_change_text: str = ""  # latest change detection text

        # Current-episode tracking
        self._current_steps: list[StepRecord] = []
        self._current_step_embeddings: list[np.ndarray] = []
        self._retry_count: int = 0
        self._needs_reset: bool = False
        self._last_action_name: str = ""

        # Loop detection
        self._loop_detector = LoopDetector()
        self._loop_detection: Optional[LoopDetection] = None

        # Reasoning token tracking (same pattern as ReasoningLLM)
        self._last_reasoning_tokens: int = 0
        self._total_reasoning_tokens: int = 0
        self._last_response_content: str = ""

        # Skill file path (cached)
        self._skill_path: Optional[Path] = None

        # Auto-resume: load checkpoint if one exists for this game
        resumed = self._load_checkpoint()

        logger.info(
            f"ReactBufferAgent initialized: max_retries={self.MAX_RETRIES}, "
            f"buffer_capacity={self.BUFFER_CAPACITY}, beta={self.RESIDUAL_BETA}"
            f"{', RESUMED from checkpoint' if resumed else ''}"
        )

    # --------------------------------------------------------------------- #
    # Lifecycle overrides
    # --------------------------------------------------------------------- #

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Override to implement retry-on-failure logic.

        On GAME_OVER with retries remaining the agent stores the failed
        episode, clears its conversation, and will issue RESET on the next
        ``choose_action`` call.
        """
        if latest_frame.state == GameState.WIN:
            self._store_current_episode(frames, success=True)
            logger.info(
                f"WIN after {self._retry_count} retries. "
                f"Buffer: {self.episode_buffer.get_size()} episodes, "
                f"success rate {self.episode_buffer.get_success_rate():.0%}"
            )
            return True

        if latest_frame.state == GameState.GAME_OVER:
            self._store_current_episode(frames, success=False)
            if self._retry_count < self.MAX_RETRIES:
                self._prepare_retry()
                logger.info(
                    f"GAME_OVER -> retry {self._retry_count}/{self.MAX_RETRIES}. "
                    f"Buffer: {self.episode_buffer.get_size()} episodes"
                )
                return False
            logger.info(
                f"GAME_OVER -> max retries ({self.MAX_RETRIES}) exhausted"
            )
            return True

        return False

    def _prepare_retry(self) -> None:
        """Prepare state for a new attempt after GAME_OVER."""
        self._retry_count += 1
        self._needs_reset = True
        self._current_steps = []
        self._current_step_embeddings = []
        # Reset loop detector
        self._loop_detector.reset()
        self._loop_detection = None
        # Reset perception state but keep learned action effects
        saved_effects = dict(self.perception._action_effects)
        self.perception.reset()
        self.perception._action_effects = saved_effects
        # Reset VLM annotation caches for fresh start on retry
        self._cached_entity_text = ""
        self._entity_step = -1
        self._prev_bbox_image = None
        self._cached_change_text = ""
        # Invalidate skill cache so updated knowledge is re-read
        self._skills_system_prompt = None
        # Clear conversation so choose_action will issue RESET
        self.messages = []

    def _store_current_episode(
        self, frames: list[FrameData], success: bool
    ) -> None:
        """Finalize the current episode and add it to the buffer."""
        latest = frames[-1] if frames else None
        final_score = (
            getattr(latest, "score", latest.levels_completed)
            if latest
            else 0
        )

        # Episode-level embedding from game context
        ctx_parts = [
            f"game:{self.game_id}",
            f"score:{final_score}",
            f"actions:{len(self._current_steps)}",
        ]
        if self._current_steps:
            recent = " ".join(s.action for s in self._current_steps[-10:])
            ctx_parts.append(f"recent:{recent}")
        ctx_text = " ".join(ctx_parts)

        episode = GameEpisode(
            game_id=self.game_id,
            steps=list(self._current_steps),
            success=success,
            final_score=final_score,
            total_actions=len(self._current_steps),
            embedding=self._embedder.encode(ctx_text),
            step_embeddings=list(self._current_step_embeddings),
        )
        self.episode_buffer.add(episode)
        self._save_checkpoint()

    # --------------------------------------------------------------------- #
    # Checkpoint save / load (resume support)
    # --------------------------------------------------------------------- #

    def _checkpoint_dir(self) -> Path:
        """Return the checkpoint directory, creating it if needed."""
        base = os.environ.get("CHECKPOINT_DIR", "checkpoints")
        p = Path(base)
        if not p.is_absolute():
            p = Path.cwd() / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _checkpoint_path(self) -> Path:
        """Return the checkpoint file path for the current game."""
        return self._checkpoint_dir() / f"{self.game_id}.json"

    def _save_checkpoint(self) -> None:
        """Persist agent experience to disk for resume.

        Saved every CHECKPOINT_INTERVAL steps and on episode end.
        Includes both completed episodes and the current in-progress episode.
        """
        try:
            # Serialize current in-progress steps
            current_steps_data = [asdict(s) for s in self._current_steps]
            current_embs_data = [e.tolist() for e in self._current_step_embeddings]

            data: dict[str, Any] = {
                "game_id": self.game_id,
                "timestamp": time.time(),
                "retry_count": self._retry_count,
                "total_reasoning_tokens": self._total_reasoning_tokens,
                "token_counter": self.token_counter,
                # Completed episodes in buffer
                "episodes": [ep.to_dict() for ep in self.episode_buffer.episodes],
                # Current in-progress episode (not yet in buffer)
                "current_steps": current_steps_data,
                "current_step_embeddings": current_embs_data,
                # Embedder vocabulary state
                "embedder": {
                    "vocab": self._embedder._vocab,
                    "doc_count": self._embedder._doc_count,
                    "word_doc_count": self._embedder._word_doc_count,
                },
                # Learned action effects from perception
                # Structure: {action_name: [[dr, dc], [dr, dc], ...]}
                "action_effects": {
                    k: [list(t) for t in v]
                    for k, v in self.perception._action_effects.items()
                },
            }
            path = self._checkpoint_path()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(
                "Checkpoint saved: %s (%d episodes, %d current steps, retry=%d)",
                path.name,
                len(self.episode_buffer.episodes),
                len(self._current_steps),
                self._retry_count,
            )
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def _load_checkpoint(self) -> bool:
        """Load agent experience from disk if a checkpoint exists.

        Returns True if a checkpoint was loaded.
        """
        path = self._checkpoint_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read checkpoint %s: %s", path, e)
            return False

        if data.get("game_id") != self.game_id:
            logger.warning(
                "Checkpoint game_id mismatch: %s != %s",
                data.get("game_id"),
                self.game_id,
            )
            return False

        try:
            # Restore retry count
            self._retry_count = data.get("retry_count", 0)
            self._total_reasoning_tokens = data.get("total_reasoning_tokens", 0)
            self.token_counter = data.get("token_counter", 0)

            # Restore episode buffer
            episodes_data = data.get("episodes", [])
            for ep_data in episodes_data:
                ep = GameEpisode.from_dict(ep_data)
                self.episode_buffer.add(ep)

            # Restore embedder vocabulary
            emb_data = data.get("embedder", {})
            if emb_data:
                self._embedder._vocab = emb_data.get("vocab", {})
                self._embedder._doc_count = emb_data.get("doc_count", 0)
                self._embedder._word_doc_count = emb_data.get(
                    "word_doc_count", {}
                )

            # Restore perception action effects
            # Structure: {action_name: [[dr, dc], ...]} -> {action_name: [(dr, dc), ...]}
            effects_data = data.get("action_effects", {})
            for action_name, pairs in effects_data.items():
                self.perception._action_effects[action_name] = [
                    (int(p[0]), int(p[1])) for p in pairs
                ]

            # Recover interrupted in-progress episode as a failed attempt
            # (the game resets on restart, so the partial run is a "failure")
            current_steps_data = data.get("current_steps", [])
            if current_steps_data:
                steps = [StepRecord(**s) for s in current_steps_data]
                embs_data = data.get("current_step_embeddings", [])
                step_embs = [np.array(e, dtype=np.float32) for e in embs_data]

                # Build episode-level embedding
                ctx_parts = [
                    f"game:{self.game_id}",
                    f"actions:{len(steps)}",
                ]
                if steps:
                    recent = " ".join(s.action for s in steps[-10:])
                    ctx_parts.append(f"recent:{recent}")
                ctx_text = " ".join(ctx_parts)

                interrupted_ep = GameEpisode(
                    game_id=self.game_id,
                    steps=steps,
                    success=False,
                    final_score=steps[-1].score_after if steps else 0,
                    total_actions=len(steps),
                    embedding=self._embedder.encode(ctx_text),
                    step_embeddings=step_embs,
                )
                self.episode_buffer.add(interrupted_ep)
                self._retry_count += 1
                logger.info(
                    "Recovered interrupted episode: %d steps, "
                    "score=%d (treated as failed attempt)",
                    len(steps),
                    interrupted_ep.final_score,
                )

            logger.info(
                "Checkpoint loaded: %s (%d episodes, retry=%d, tokens=%d)",
                path.name,
                len(self.episode_buffer.episodes),
                self._retry_count,
                self.token_counter,
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore checkpoint %s: %s", path, e)
            return False

    # --------------------------------------------------------------------- #
    # Skills & system prompt (only grid_perception + arc_game_playing)
    # --------------------------------------------------------------------- #

    def _get_system_prompt_with_skills(self) -> str:
        """Load only the two allowed skills into the system prompt."""
        if self._skills_system_prompt is not None:
            return self._skills_system_prompt
        try:
            skills = load_local_skills(max_content_chars=12000)
            skills = [
                (n, c) for n, c in skills if n in self.ALLOWED_SKILLS
            ]
            self._skills_system_prompt = format_skills_for_prompt(
                skills,
                heading="You have the following skills. Follow them at every step:",
            )
        except Exception as e:
            logger.warning("Could not load skills: %s", e)
            self._skills_system_prompt = ""
        return self._skills_system_prompt

    def _messages_for_api(self) -> list[Any]:
        """System prompt with skills only (no scratchpad)."""
        system = self._get_system_prompt_with_skills().strip()
        if system:
            return [{"role": "system", "content": system}] + self.messages
        return self.messages

    def _get_skill_path(self) -> Path:
        """Resolve the path to arc_game_playing/SKILL.md."""
        if self._skill_path is not None:
            return self._skill_path
        skills_dir = os.environ.get("SKILLS_DIR", "agents/skills")
        p = Path(skills_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        self._skill_path = p / "arc_game_playing" / "SKILL.md"
        return self._skill_path

    # --------------------------------------------------------------------- #
    # Tool definitions
    # --------------------------------------------------------------------- #

    def build_functions(self) -> list[dict[str, Any]]:
        """Game actions + update_game_notes tool."""
        functions = super().build_functions()
        functions.append({
            "name": "update_game_notes",
            "description": (
                "Record discovered game knowledge to your persistent skill file. "
                "Use this to save confirmed action mappings, game rules, level "
                "strategies, object roles, or tips. This knowledge persists across "
                "retries and appears in your system prompt for future decisions. "
                "Call this BEFORE your game action when you have new discoveries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": list(KNOWLEDGE_HEADERS.keys()),
                        "description": "Which knowledge section to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The knowledge to record, in markdown format.",
                    },
                    "replace": {
                        "type": "string",
                        "enum": ["true", "false"],
                        "description": (
                            "Replace the entire section (true) or append to "
                            "existing content (false)."
                        ),
                    },
                },
                "required": ["section", "content", "replace"],
                "additionalProperties": False,
            },
        })
        return functions

    # --------------------------------------------------------------------- #
    # Action selection (full override with tool loop)
    # --------------------------------------------------------------------- #

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Buffer-augmented action selection with update_game_notes tool loop.

        Flow:
        1. Empty messages → RESET (first call or after retry).
        2. Push observation (tool response from last action).
        3. LLM generates free-text observation (if DO_OBSERVATION).
        4. LLM generates tool call. If update_game_notes, process and loop.
           If game action, return it.
        5. Record step into current episode for buffer.
        """
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            timeout=self.API_TIMEOUT,
        )

        tools = self.build_tools()

        if self._needs_reset:
            self._needs_reset = False

        prev_frame = frames[-2] if len(frames) > 1 else frames[-1]

        # --- Phase 1: RESET on first call / after retry ---
        if len(self.messages) == 0:
            user_prompt = self.build_user_prompt(latest_frame)
            self.push_message({"role": "user", "content": user_prompt})
            self.push_message({
                "role": "assistant",
                "tool_calls": [{
                    "id": self._latest_tool_call_id,
                    "type": "function",
                    "function": {
                        "name": GameAction.RESET.name,
                        "arguments": json.dumps({}),
                    },
                }],
            })
            action = GameAction.RESET
            self._record_step(action, prev_frame, latest_frame)
            return action

        # --- Phase 2: Push observation (tool response) ---
        function_response = self.build_func_resp_prompt(latest_frame)
        self.push_message({
            "role": "tool",
            "tool_call_id": self._latest_tool_call_id,
            "content": str(function_response),
        })

        # --- Phase 3: Observation (free-text reasoning) ---
        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            obs_content = ""
            for attempt in range(1, self.API_RETRIES + 1):
                try:
                    create_kwargs: dict[str, Any] = {
                        "model": self._model,
                        "messages": self._messages_for_api(),
                    }
                    if self.REASONING_EFFORT is not None:
                        create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                    response = client.chat.completions.create(**create_kwargs)
                    obs_content = response.choices[0].message.content or ""
                    self.track_tokens(response.usage.total_tokens, obs_content)
                    break
                except openai.APITimeoutError:
                    logger.warning(
                        "Observation API timeout (attempt %d/%d)",
                        attempt, self.API_RETRIES,
                    )
                    if attempt == self.API_RETRIES:
                        obs_content = "(observation timed out, skipping)"
                except openai.BadRequestError as e:
                    logger.info(f"Message dump: {self.messages}")
                    raise e
            self.push_message({"role": "assistant", "content": obs_content})
            logger.info(f"Assistant: {obs_content}")

        # --- Phase 3b: Filter tools on severe loops ---
        ld = self._loop_detection
        if (
            ld is not None
            and ld.is_looping
            and ld.cycle_repetitions >= LoopDetector.SEVERE_REPETITIONS
        ):
            cycle_set = set(ld.cycle_actions)
            filtered = [
                t for t in tools
                if t["function"]["name"] not in cycle_set
            ]
            # Safety: keep at least 3 tools (RESET + 2 actions)
            if len(filtered) >= 3:
                tools = filtered
                logger.info(
                    "Filtered cycling actions from tools: %s",
                    cycle_set,
                )

        # --- Phase 4: Action selection with tool loop ---
        user_prompt = self.build_user_prompt(latest_frame)
        self.push_message({"role": "user", "content": user_prompt})

        name = GameAction.ACTION5.name  # fallback
        arguments = None
        notes_calls = 0

        while True:
            logger.info("Sending to Assistant for action...")
            message = None
            for attempt in range(1, self.API_RETRIES + 1):
                try:
                    create_kwargs = {
                        "model": self._model,
                        "messages": self._messages_for_api(),
                        "tools": tools,
                        "tool_choice": "required",
                    }
                    if self.REASONING_EFFORT is not None:
                        create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                    response = client.chat.completions.create(**create_kwargs)
                    self.track_tokens(response.usage.total_tokens)
                    message = response.choices[0].message
                    break
                except openai.APITimeoutError:
                    logger.warning(
                        "Action API timeout (attempt %d/%d)",
                        attempt, self.API_RETRIES,
                    )
                except openai.BadRequestError as e:
                    logger.info(f"Message dump: {self.messages}")
                    raise e

            if message is None:
                # All retries timed out — fallback action
                logger.warning("All action API retries failed, using fallback")
                break

            if not message.tool_calls:
                # Model didn't call any tool — use fallback
                self.push_message(message)
                break

            tool_call = message.tool_calls[0]
            self._latest_tool_call_id = tool_call.id

            # Push the assistant message (contains tool_calls)
            self.push_message(message)

            # Error out any extra tool calls
            for tc in message.tool_calls[1:]:
                self.push_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: only one tool call at a time.",
                })

            if (
                tool_call.function.name == "update_game_notes"
                and notes_calls < self.MAX_NOTES_PER_TURN
            ):
                # Process game notes update, then loop for game action
                notes_calls += 1
                result = self._handle_game_notes_update(
                    tool_call.function.arguments
                )
                self.push_message({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                logger.info(
                    "update_game_notes call %d/%d: %s",
                    notes_calls, self.MAX_NOTES_PER_TURN, result,
                )
                continue
            else:
                # Game action (or max notes calls reached)
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                logger.debug(
                    f"Assistant: {name} ({tool_call.id}) {arguments}"
                )
                break

        # --- Parse action ---
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception:
                data = {}
        else:
            data = {}

        action = GameAction.from_name(name)
        action.set_data(data)

        # --- Record step ---
        self._record_step(action, prev_frame, latest_frame)

        self._last_action_name = action.name
        return action

    def _record_step(
        self,
        action: GameAction,
        prev_frame: FrameData,
        latest_frame: FrameData,
    ) -> None:
        """Record a step into the current episode and attach metadata."""
        state_text = encode_frame_compact(latest_frame)
        diff_text = compute_frame_diff(prev_frame, latest_frame)
        prev_score = getattr(prev_frame, "score", prev_frame.levels_completed)
        curr_score = getattr(
            latest_frame, "score", latest_frame.levels_completed
        )

        step = StepRecord(
            state_text=state_text,
            action=action.name,
            frame_diff=diff_text,
            score_before=prev_score,
            score_after=curr_score,
        )
        self._current_steps.append(step)

        # State-action embedding for residual scoring
        sa_text = f"{state_text} {action.name} {diff_text}"
        sa_emb = self._embedder.encode(sa_text)
        self._current_step_embeddings.append(sa_emb)

        # Loop detection
        self._loop_detection = self._loop_detector.analyze(self._current_steps)
        ld = self._loop_detection
        if ld.is_looping:
            logger.warning(
                "LOOP DETECTED: %s repeated %dx (cycle length %d)",
                " -> ".join(ld.cycle_actions),
                ld.cycle_repetitions,
                ld.cycle_length,
            )
        if ld.is_stuck:
            logger.warning(
                "STUCK: score unchanged for %d steps",
                ld.steps_since_score_change,
            )

        # Attach reasoning metadata
        action.reasoning = {
            "model": self._model,
            "agent_type": "react_buffer",
            "retry": self._retry_count,
            "buffer_size": self.episode_buffer.get_size(),
            "buffer_success_rate": round(
                self.episode_buffer.get_success_rate(), 3
            ),
            "step_in_episode": len(self._current_steps),
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "loop_detection": {
                "is_looping": ld.is_looping,
                "cycle_actions": ld.cycle_actions,
                "cycle_repetitions": ld.cycle_repetitions,
                "steps_since_score_change": ld.steps_since_score_change,
                "is_stuck": ld.is_stuck,
            },
            "game_context": {
                "score": curr_score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(self.frames) if hasattr(self, "frames") else 0,
            },
        }

        # Periodic checkpoint
        if (
            self.CHECKPOINT_INTERVAL > 0
            and len(self._current_steps) % self.CHECKPOINT_INTERVAL == 0
        ):
            self._save_checkpoint()

    # --------------------------------------------------------------------- #
    # update_game_notes tool handler
    # --------------------------------------------------------------------- #

    def _handle_game_notes_update(self, arguments_json: str) -> str:
        """Process an update_game_notes tool call — write to SKILL.md."""
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            return "Error: invalid JSON arguments."

        section = args.get("section", "")
        content = args.get("content", "").strip()
        replace = args.get("replace", "false") == "true"

        if section not in KNOWLEDGE_HEADERS:
            return (
                f"Error: unknown section '{section}'. "
                f"Valid: {list(KNOWLEDGE_HEADERS.keys())}"
            )
        if not content:
            return "Error: content is empty."

        skill_path = self._get_skill_path()
        if not skill_path.exists():
            return f"Error: skill file not found at {skill_path}"

        try:
            text = skill_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading skill file: {e}"

        header = KNOWLEDGE_HEADERS[section]

        # Find the section header
        header_idx = text.find(header)
        if header_idx < 0:
            # Section not found — append at end
            text = text.rstrip() + f"\n\n{header}\n{content}\n"
        else:
            # Find where the section content starts (after the header line)
            newline_after = text.find("\n", header_idx)
            if newline_after < 0:
                newline_after = len(text)
            content_start = newline_after + 1

            # Find the next section header (### or ##) or end of file
            next_header_idx = len(text)
            search_start = content_start
            for h in KNOWLEDGE_HEADERS.values():
                pos = text.find(h, search_start)
                if pos > 0 and pos < next_header_idx:
                    next_header_idx = pos
            # Also stop at any ## header
            pos = text.find("\n## ", search_start)
            if pos >= 0 and pos + 1 < next_header_idx:
                next_header_idx = pos + 1

            old_content = text[content_start:next_header_idx].strip()

            if replace:
                new_content = content
            else:
                # Append — skip placeholder text
                if old_content and old_content != "(no data yet)":
                    new_content = old_content + "\n" + content
                else:
                    new_content = content

            text = (
                text[:content_start]
                + new_content + "\n\n"
                + text[next_header_idx:]
            )

        try:
            skill_path.write_text(text, encoding="utf-8")
        except Exception as e:
            return f"Error writing skill file: {e}"

        # Invalidate system prompt cache so next API call picks up changes
        self._skills_system_prompt = None

        logger.info(
            "Game notes updated: section=%s, replace=%s, length=%d",
            section, replace, len(content),
        )
        return f"Updated '{section}'. Knowledge will inform your future decisions."

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Track tokens with reasoning-token bookkeeping."""
        super().track_tokens(tokens, message)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    # --------------------------------------------------------------------- #
    # Prompt construction
    # --------------------------------------------------------------------- #

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        """Observation prompt: perception data + buffer context + residual hints.

        Strategy is handled by the skills in the system prompt.
        This prompt provides the DATA the LLM needs to make decisions.
        """
        sections: list[str] = []

        # Game state
        score = getattr(latest_frame, "score", latest_frame.levels_completed)
        sections.append(
            f"# Game State: {latest_frame.state.name} | Score: {score}"
        )

        # Retry info
        if self._retry_count > 0:
            sections.append(
                f"# Attempt {self._retry_count + 1}/{self.MAX_RETRIES + 1} — "
                f"review past failures and try a DIFFERENT strategy."
            )

        # Structured perception (replaces raw grid)
        prev_frame_grids = None
        if len(self.frames) > 1 and self.frames[-2].frame:
            prev_frame_grids = self.frames[-2].frame

        if latest_frame.frame:
            perception_text = self.perception.describe_frame(
                frame=latest_frame.frame,
                prev_frame=prev_frame_grids,
                last_action=self._last_action_name or None,
            )
            sections.append(perception_text)

            # VLM: entity identification + change interpretation
            vlm_sections = self._get_vlm_analysis(
                latest_frame, prev_frame_grids
            )
            sections.extend(vlm_sections)

        # Past attempts from buffer
        buffer_ctx = self._build_buffer_context()
        if buffer_ctx:
            sections.append(buffer_ctx)

        # Residual action hints
        residual_hint = self._build_residual_hint(latest_frame)
        if residual_hint:
            sections.append(residual_hint)

        # Anti-repetition: action history, loop warning, stuck warning
        action_history = self._build_action_history()
        if action_history:
            sections.append(action_history)

        loop_warning = self._build_loop_warning()
        if loop_warning:
            sections.append(loop_warning)

        stuck_warning = self._build_stuck_warning()
        if stuck_warning:
            sections.append(stuck_warning)

        # Instruction
        sections.append(
            "# YOUR TURN\n"
            "Follow the arc_game_playing skill reasoning template.\n"
            "If you confirmed new knowledge, call update_game_notes first.\n"
            "Then call exactly one game action."
        )

        return "\n\n".join(sections)

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Minimal user prompt — strategy is in the skills system prompt."""
        return textwrap.dedent("""\
            You are playing an unknown ARC-AGI-3 grid game.
            Follow your grid_perception and arc_game_playing skills.

            Available tools:
            - update_game_notes: Save discovered knowledge (call BEFORE game action)
            - RESET, ACTION1-ACTION6: Game control

            You may call update_game_notes to record discoveries, then call exactly one game action.
        """)

    # --------------------------------------------------------------------- #
    # VLM & buffer helpers
    # --------------------------------------------------------------------- #

    def _get_vlm_analysis(
        self,
        latest_frame: FrameData,
        prev_frame_grids: Optional[list[list[list[int]]]],
    ) -> list[str]:
        """Run VLM entity identification + change detection.

        Returns a list of text sections to append to the prompt:
        1. Entity identification (periodic, every ``ENTITY_INTERVAL`` steps)
        2. Change detection (every action when a previous frame is available)
        """
        result: list[str] = []
        if not latest_frame.frame:
            return result

        step = len(self._current_steps)
        grid = latest_frame.frame[0]
        analysis = self.perception.analyze_grid(grid)
        curr_bbox = self.perception.render_grid(grid, analysis=analysis)

        if curr_bbox is None:
            return result

        # --- 1. Entity identification (periodic) ---
        need_identify = (
            self._entity_step < 0
            or (step - self._entity_step) >= self.ENTITY_INTERVAL
        )
        if need_identify:
            try:
                entities = self._annotator.annotate(curr_bbox, analysis)
                if entities:
                    self._cached_entity_text = ObjectAnnotator.format_entities(
                        entities
                    )
                    self._entity_step = step
                    logger.info(
                        "VLM entity identification at step %d: %d entities",
                        step,
                        len(entities),
                    )
            except Exception as exc:
                logger.warning(
                    "VLM entity identification failed at step %d: %s",
                    step,
                    exc,
                )

        if self._cached_entity_text:
            result.append(self._cached_entity_text)

        # --- 2. Change detection (per action, needs previous frame) ---
        if (
            self._prev_bbox_image is not None
            and prev_frame_grids is not None
            and self._last_action_name
        ):
            try:
                # Compute precise algorithmic diff
                change = self.perception.compute_diff(
                    prev_frame_grids[0], grid
                )

                frame_ann = self._annotator.annotate_change(
                    curr_bbox_image=curr_bbox,
                    analysis=analysis,
                    change=change,
                    action=self._last_action_name,
                )
                if frame_ann is not None:
                    self._cached_change_text = (
                        ObjectAnnotator.format_changes(frame_ann)
                    )
                    logger.info(
                        "VLM change detection at step %d: %d changes, "
                        "effect=%s",
                        step,
                        len(frame_ann.changes),
                        frame_ann.action_effect,
                    )
            except Exception as exc:
                logger.warning(
                    "VLM change detection failed at step %d: %s", step, exc
                )

        if self._cached_change_text:
            result.append(self._cached_change_text)

        # Save current bbox image for next step's diff
        self._prev_bbox_image = curr_bbox

        return result

    def _build_buffer_context(self) -> str:
        """Format past episodes for this game into an LLM-readable summary."""
        game_episodes = [
            ep
            for ep in self.episode_buffer.episodes
            if ep.game_id == self.game_id
        ]
        if not game_episodes:
            return ""

        lines: list[str] = [
            "\n# Past Attempts on This Game",
            f"({len(game_episodes)} episodes, "
            f"success rate: {self.episode_buffer.get_success_rate():.0%})",
        ]

        for i, ep in enumerate(game_episodes[-self.NUM_RETRIEVE :]):
            outcome = "SUCCESS" if ep.success else "FAILED"
            lines.append(
                f"\n## Attempt {i + 1} ({outcome}, score={ep.final_score}, "
                f"{ep.total_actions} actions)"
            )

            # Show important steps: first few, last few, score-changing
            important: list[str] = []
            for j, step in enumerate(ep.steps):
                is_early = j < 3
                is_late = j >= len(ep.steps) - 3
                is_score_up = step.score_after > step.score_before
                is_noop = "no_change" in step.frame_diff

                if is_early or is_late or is_score_up:
                    tag = ""
                    if is_score_up:
                        tag = " **[SCORE UP]**"
                    elif is_noop:
                        tag = " [NO EFFECT]"
                    important.append(
                        f"  Step {j}: {step.action} -> {step.frame_diff}{tag}"
                    )

            lines.extend(important)

            # Failure analysis
            if not ep.success and ep.steps:
                noop_count = sum(
                    1 for s in ep.steps if "no_change" in s.frame_diff
                )
                last_actions = [s.action for s in ep.steps[-5:]]
                lines.append(
                    f"  -> {noop_count}/{ep.total_actions} actions had no effect"
                )
                lines.append(
                    f"  -> Last actions before failure: "
                    f"{' -> '.join(last_actions)}"
                )

        return "\n".join(lines)

    def _build_residual_hint(self, latest_frame: FrameData) -> str:
        """Compute per-action residual scores and format as LLM hint."""
        if self.episode_buffer.get_size() == 0:
            return ""

        state_text = encode_frame_compact(latest_frame)
        query_emb = self._embedder.encode(state_text)

        action_names = [
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
        ]
        scores: dict[str, float] = {}

        for name in action_names:
            sa_emb = self._embedder.encode(f"{state_text} {name}")
            scores[name] = self.residual_fn.score_action(
                sa_emb, self.episode_buffer, query_emb, k=self.NUM_RETRIEVE
            )

        if all(abs(s) < 1e-6 for s in scores.values()):
            return ""

        sorted_actions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        lines: list[str] = [
            "\n# Residual Guidance (from past experience)",
            "Scores: positive = similar to past success, "
            "negative = similar to past failure:",
        ]
        for name, score in sorted_actions:
            if abs(score) >= 1e-6:
                sign = "+" if score > 0 else ""
                lines.append(f"  {name}: {sign}{score:.3f}")

        return "\n".join(lines)

    def _build_action_history(self) -> str:
        """Last ~15 actions with [NO EFFECT] / [SCORE+] markers.

        Gives the LLM visibility into its own recent actions — critical because
        MESSAGE_LIMIT truncation drops older messages from the conversation.
        """
        if not self._current_steps:
            return ""

        ld = self._loop_detection
        window = LoopDetector.HISTORY_WINDOW
        recent = self._current_steps[-window:]
        offset = len(self._current_steps) - len(recent)

        lines: list[str] = ["# Recent Action History"]

        if ld and ld.steps_since_score_change > 0:
            lines.append(
                f"(Score unchanged for {ld.steps_since_score_change} steps)"
            )

        for i, step in enumerate(recent):
            idx = offset + i
            markers = ""
            if step.score_after > step.score_before:
                markers = " [SCORE+]"
            elif "no_change" in step.frame_diff:
                markers = " [NO EFFECT]"
            lines.append(f"  Step {idx}: {step.action}{markers}")

        return "\n".join(lines)

    def _build_loop_warning(self) -> str:
        """WARNING/CRITICAL block when a cycle is detected."""
        ld = self._loop_detection
        if not ld or not ld.is_looping:
            return ""

        cycle_str = " -> ".join(ld.cycle_actions)

        if ld.cycle_repetitions >= LoopDetector.SEVERE_REPETITIONS:
            severity = "CRITICAL"
            instruction = (
                f"You have repeated the cycle [{cycle_str}] "
                f"{ld.cycle_repetitions} times. "
                "These actions are being REMOVED from your options. "
                "You MUST try a completely different approach — use RESET, "
                "ACTION5, ACTION6, or any action NOT in the cycle."
            )
        else:
            severity = "WARNING"
            instruction = (
                f"You have repeated the cycle [{cycle_str}] "
                f"{ld.cycle_repetitions} times. "
                "This pattern is NOT making progress. "
                "STOP repeating these actions and try something different."
            )

        return f"# {severity}: REPETITIVE LOOP DETECTED\n{instruction}"

    def _build_stuck_warning(self) -> str:
        """STUCK block when score unchanged for many steps."""
        ld = self._loop_detection
        if not ld or not ld.is_stuck:
            return ""

        lines: list[str] = [
            f"# STUCK: Score unchanged for {ld.steps_since_score_change} steps",
            "Your current strategy is not working. You need to change approach.",
        ]

        if ld.no_effect_actions:
            noop_parts = [
                f"{act} ({cnt}x no effect)"
                for act, cnt in sorted(
                    ld.no_effect_actions.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            lines.append(
                f"Actions with no recent effect: {', '.join(noop_parts)}"
            )
            lines.append(
                "AVOID these actions and try ones you haven't used recently."
            )

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Name override
    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        model = self._model.replace("/", "-").replace(":", "-")
        return (
            f"{self.game_id}.reactbuffer.{model}.{obs}"
            f".retry{self.MAX_RETRIES}"
        )
