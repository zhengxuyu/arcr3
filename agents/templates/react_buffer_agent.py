"""
React Agent with Exploration Buffer for ARC-AGI-3.

Core ideas:
1. **Exploration Buffer**: Stores past game episodes (state-action trajectories)
   with embeddings for similarity retrieval.
2. **ReAct Loop**: Observe -> Retrieve from buffer -> Reason -> Act -> Learn.

On GAME_OVER the agent does NOT exit. Instead it stores the failed trajectory,
resets the game, and retries with accumulated experience informing the LLM
via buffer-augmented prompts.

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
import subprocess
import sys
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
from .game_intelligence import (
    GridNavigator,
    InteractionEventType,
    InteractionTracker,
    ObjectInventory,
    ObjectStatus,
    TransformationDetector,
)
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


def _parse_cells_changed(diff_text: str) -> int:
    """Extract total cells changed from frame diff like 'g0:D42cells'."""
    total = 0
    for m in re.finditer(r'D(\d+)cells', diff_text):
        total += int(m.group(1))
    return total


# Threshold: energy bar changes 1-3 cells, player movement changes 25+ cells
MIN_CELLS_FOR_MOVEMENT = 8


def _did_player_move(
    prev_grid: list[list[int]],
    curr_grid: list[list[int]],
    player_colors: set[int],
) -> bool:
    """Check if player-colored cells shifted between frames.

    Uses a bounded region around the last known player position
    to avoid confusion from same-colored cells elsewhere in the grid
    (score indicators, decorations, etc.).
    """
    if not player_colors:
        return False

    # Count cells of player colors that CHANGED between frames.
    # If the player moved, many cells will differ (new position ≠ old).
    # If the player didn't move, very few cells of player colors will differ.
    changed = 0
    for r in range(len(prev_grid)):
        for c in range(len(prev_grid[0])):
            pv = prev_grid[r][c]
            cv = curr_grid[r][c]
            if pv != cv and (pv in player_colors or cv in player_colors):
                changed += 1

    # Player body is ~15 cells (3x5), head is ~10 cells (2x5).
    # When the player moves, ~25 player-colored cells change.
    # When the player doesn't move, 0 player-colored cells change.
    # Threshold at 5 to distinguish.
    moved = changed >= 5
    logger.debug("_did_player_move: colors=%s changed=%d moved=%s",
                 player_colors, changed, moved)
    return moved


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
    """React Agent with Exploration Buffer.

    Explores ARC-AGI-3 games by:
    1. Playing normally until WIN or GAME_OVER.
    2. On GAME_OVER: storing the failed trajectory and resetting (up to
       ``MAX_RETRIES`` times) instead of exiting.
    3. Injecting buffer-retrieved past experiences into the LLM prompt so
       the model can learn from prior failures/successes.

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
    # Inline VLM mode: "periodic" (default), "always", or "off"
    #   periodic: image in first INLINE_VLM_WARMUP steps, then every INLINE_VLM_INTERVAL steps
    #   always:   image every step
    #   off:      no image (pure text perception)
    INLINE_VLM: str = "periodic"
    INLINE_VLM_WARMUP: int = 10   # first N steps always include image
    INLINE_VLM_INTERVAL: int = 8  # after warmup, include image every N steps
    MODEL: str = "o4-mini"
    REASONING_EFFORT: Optional[str] = "medium"
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 20

    # Buffer configuration
    BUFFER_CAPACITY: int = 100
    NUM_RETRIEVE: int = 3
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
        # Timestamp must be set BEFORE super().__init__ because
        # start_recording() reads self.name which includes the timestamp.
        from datetime import datetime
        self._start_time_tag = datetime.now().strftime("%m%d_%H%M")

        super().__init__(*args, **kwargs)

        self._embedder = SimpleEmbedding(dim=self.EMB_DIM)
        self.episode_buffer = EpisodeBuffer(
            capacity=self.BUFFER_CAPACITY, emb_dim=self.EMB_DIM
        )

        # Inline VLM: env override (periodic|always|off, or legacy 0/1)
        _vlm_env = os.environ.get("INLINE_VLM", "").strip().lower()
        if _vlm_env in ("0", "false", "no", "off"):
            self.INLINE_VLM = "off"
        elif _vlm_env in ("1", "true", "yes", "always"):
            self.INLINE_VLM = "always"
        elif _vlm_env == "periodic":
            self.INLINE_VLM = "periodic"

        # Store current bbox image for inline VLM mode (Phase 4 multimodal message)
        self._current_bbox_image = None
        self._step_count_for_vlm = 0  # tracks steps within current level for periodic mode

        # Grid perception engine (replaces raw grid dump with structured analysis)
        self.perception = GridPerception()

        # VLM object annotator — identifies objects + detects changes
        # Uses VLM_MODEL env var if set, otherwise falls back to main model
        vlm_model = os.environ.get("VLM_MODEL", "") or self._model
        self._annotator = ObjectAnnotator(model=vlm_model)
        self._cached_entity_text: str = ""  # latest entity identification text
        self._entity_step: int = -1         # step when last identified entities
        self.ENTITY_INTERVAL: int = 8       # re-identify entities every N steps
        self._prev_bbox_image = None        # previous frame's bbox image for diff
        self._cached_change_text: str = ""  # latest change detection text

        # Game intelligence modules
        self._navigator = GridNavigator()
        self._interaction_tracker = InteractionTracker()
        self._inventory = ObjectInventory()
        self._transformation_detector = TransformationDetector()

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

        # Level replay: replay previously successful level action sequences
        self._level_replays: dict[int, list[str]] = {}
        self._replay_queue: list[str] = []
        self._replay_index: int = 0

        # Action plan queue (from execute_plan tool)
        self._action_plan: list[str] = []

        # Estimated player position during plan execution (updated after each
        # successful step using the action's direction delta).
        self._plan_estimated_pos: Optional[tuple[int, int]] = None
        self._last_detected_pos: Optional[tuple[int, int]] = None

        # Deferred navigator update: (player_pos_before, action_name)
        # Resolved at the START of the next choose_action when the result frame is available
        self._pending_nav_update: Optional[tuple[tuple[int, int], str]] = None

        # Reset skill file from template before each run
        self._reset_skill_from_template()

        # Auto-resume: load checkpoint if one exists for this game
        resumed = self._load_checkpoint()

        # Parse SKIP_REPLAY env var (comma-separated, controls which replays to skip)
        # Formats:  "0,1"         — skip levels 0,1 for ALL games
        #           "ls20:0"      — skip level 0 only for game ls20 (matches game_id prefix)
        #           "ls20:0,1"    — skip levels 0 and 1 for ls20, all other games unaffected
        skip_str = os.environ.get("SKIP_REPLAY", "").strip()
        # Global level skips (no game prefix)
        self._skip_replay_global: set[int] = set()
        # Per-game level skips: game_prefix -> set of levels
        self._skip_replay_game: dict[str, set[int]] = {}
        if skip_str:
            for entry in skip_str.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    game_part, level_part = entry.split(":", 1)
                    game_part = game_part.strip()
                    for lv in level_part.split(","):
                        lv = lv.strip()
                        if lv.isdigit():
                            self._skip_replay_game.setdefault(game_part, set()).add(int(lv))
                elif entry.isdigit():
                    self._skip_replay_global.add(int(entry))
            if self._skip_replay_global or self._skip_replay_game:
                logger.info(
                    "Replay skip config: global=%s, per-game=%s",
                    self._skip_replay_global or "none",
                    {k: v for k, v in self._skip_replay_game.items()} or "none",
                )

        # Load level replays and set up replay queue for level 0
        self._level_replays = self._load_level_replays()
        if 0 in self._level_replays and self._retry_count == 0 and not self._should_skip_replay(0):
            self._replay_queue = list(self._level_replays[0])
            self._replay_index = 0
            logger.info(
                "Replay loaded for level 0: %d actions",
                len(self._replay_queue),
            )

        logger.info(
            f"ReactBufferAgent initialized: max_retries={self.MAX_RETRIES}, "
            f"buffer_capacity={self.BUFFER_CAPACITY}"
            f", inline_vlm={self.INLINE_VLM}"
            f"{', RESUMED from checkpoint' if resumed else ''}"
            f", replays={list(self._level_replays.keys())}"
        )

    # --------------------------------------------------------------------- #
    # Lifecycle overrides
    # --------------------------------------------------------------------- #

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Override to implement retry-on-failure and multi-level logic.

        On WIN: if more levels remain, store the successful episode and
        prepare for the next level (return False to keep playing).
        On GAME_OVER with retries remaining: store failed episode, reset,
        and retry.
        """
        if latest_frame.state == GameState.WIN:
            self._store_current_episode(frames, success=True)
            win_levels = getattr(latest_frame, "win_levels", None)
            levels_done = latest_frame.levels_completed

            # Save replay for the just-completed level
            level_actions = [s.action for s in self._current_steps]
            self._save_level_replay(levels_done - 1, level_actions)

            # Auto-generate video snapshot
            self._generate_video(f"L{levels_done}_complete")

            # Multi-level: continue if more levels remain
            if win_levels and levels_done < win_levels:
                logger.info(
                    f"LEVEL {levels_done}/{win_levels} completed! "
                    f"Continuing to next level. "
                    f"Buffer: {self.episode_buffer.get_size()} episodes"
                )
                self._prepare_next_level()
                return False

            logger.info(
                f"WIN (all {levels_done} levels) after "
                f"{self._retry_count} retries. "
                f"Buffer: {self.episode_buffer.get_size()} episodes, "
                f"success rate {self.episode_buffer.get_success_rate():.0%}"
            )
            self._generate_video(f"L{levels_done}_complete")
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
        # Reset game intelligence modules (keep rules/classifications)
        self._navigator.reset()
        self._interaction_tracker.reset()
        self._inventory.reset(keep_classifications=True)
        self._transformation_detector.reset()
        # Reset replay and plan state — retries don't replay
        self._replay_queue = []
        self._replay_index = 0
        self._action_plan = []
        self._plan_estimated_pos = None
        self._pending_nav_update = None
        # Reset perception state but keep learned action effects
        saved_effects = dict(self.perception._action_effects)
        self.perception.reset()
        self.perception._action_effects = saved_effects
        # Reset VLM annotation caches and periodic counter for fresh start on retry
        self._step_count_for_vlm = 0
        self._cached_entity_text = ""
        self._entity_step = -1
        self._prev_bbox_image = None
        self._cached_change_text = ""
        self._current_bbox_image = None
        # Invalidate skill cache so updated knowledge is re-read
        self._skills_system_prompt = None
        # Clear conversation so choose_action will issue RESET
        self.messages = []

    def _prepare_next_level(self) -> None:
        """Prepare state for the next level after a level WIN.

        Unlike _prepare_retry, this preserves conversation context (the LLM
        keeps its learned knowledge) and does NOT issue RESET (the engine
        auto-transitions to the next level).  Retry count is also preserved.

        If a replay exists for the next level, sets up the replay queue
        and clears messages (LLM won't be called during replay).
        """
        # Determine the next level number from how many steps we just completed
        # levels_completed was already incremented by the engine
        next_level = len(self.episode_buffer.episodes)  # rough proxy
        # Better: count successful episodes for this game
        successful_levels = sum(
            1 for ep in self.episode_buffer.episodes
            if ep.game_id == self.game_id and ep.success
        )

        # Auto-generate level summary BEFORE resetting modules
        # (need their accumulated knowledge for the summary)
        self._generate_level_summary(level_num=successful_levels)

        self._current_steps = []
        self._current_step_embeddings = []
        # Reset loop detector for the new level
        self._loop_detector.reset()
        self._loop_detection = None
        # Reset game intelligence modules (keep rules/classifications for cross-level learning)
        self._navigator.reset()
        self._interaction_tracker.reset()
        self._inventory.reset(keep_classifications=True)
        self._transformation_detector.reset()
        # Reset perception — new level has a new grid layout
        saved_effects = dict(self.perception._action_effects)
        self.perception.reset()
        self.perception._action_effects = saved_effects
        # Reset VLM caches and periodic counter for fresh entity identification
        self._step_count_for_vlm = 0
        self._cached_entity_text = ""
        self._entity_step = -1
        self._prev_bbox_image = None
        self._cached_change_text = ""
        self._current_bbox_image = None
        # Invalidate skill cache so updated knowledge is re-read
        self._skills_system_prompt = None

        # Reset action plan and deferred navigator update
        self._action_plan = []
        self._plan_estimated_pos = None
        self._pending_nav_update = None

        # Check if we have a replay for the next level
        if successful_levels in self._level_replays and not self._should_skip_replay(successful_levels):
            self._replay_queue = list(
                self._level_replays[successful_levels]
            )
            self._replay_index = 0
            # Clear messages — not needed during replay
            self.messages = []
            logger.info(
                "Replay loaded for level %d: %d actions",
                successful_levels, len(self._replay_queue),
            )
        else:
            # No replay — keep messages for LLM context
            # Do NOT set _needs_reset — engine auto-transitions
            pass

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
    # Level replay save / load
    # --------------------------------------------------------------------- #

    def _replays_path(self) -> Path:
        """Return the replays file path for the current game."""
        return self._checkpoint_dir() / f"{self.game_id}.replays.json"

    def _generate_video(self, label: Any) -> None:
        """Generate an MP4 video from the current recording (background).

        Args:
            label: Suffix for the output filename (e.g. 1 -> .L1.mp4,
                   "L0_step10" -> .L0_step10.mp4).
        """
        if not hasattr(self, "recorder"):
            return
        recording_path = self.recorder.filename
        if not recording_path or not Path(recording_path).exists():
            return

        # Output: recordings/<game_id>.<model>.<start_time>.<label>.mp4
        rec_dir = Path(recording_path).parent
        model_tag = self._model.replace("/", "-").replace(":", "-")
        time_tag = getattr(self, "_start_time_tag", "0000_0000")
        out_name = f"{self.game_id}.{model_tag}.{time_tag}.{label}.mp4"
        out_path = rec_dir / out_name

        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "visualize_recording.py"
        if not script.exists():
            logger.warning("Video script not found: %s", script)
            return

        cmd = [
            sys.executable, str(script),
            str(recording_path),
            "-o", str(out_path),
            "--format", "mp4",
            "--fps", "3",
        ]
        try:
            # Run in background so it doesn't block gameplay
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Video generation started: %s", out_name)
        except Exception as e:
            logger.warning("Failed to start video generation: %s", e)

    def _save_level_replay(self, level_num: int, actions: list[str]) -> None:
        """Persist a successful level's action sequence for future replay."""
        try:
            # Load existing replays
            replays = self._load_level_replays()
            # Only save if we don't already have a replay, or this one is shorter
            if level_num not in replays or len(actions) < len(replays[level_num]):
                replays[level_num] = actions
                path = self._replays_path()
                # JSON keys must be strings
                data = {str(k): v for k, v in replays.items()}
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self._level_replays = replays
                logger.info(
                    "Saved replay for level %d: %d actions -> %s",
                    level_num, len(actions), path.name,
                )
        except Exception as e:
            logger.warning("Failed to save level replay: %s", e)

    def _load_level_replays(self) -> dict[int, list[str]]:
        """Load level replays from disk."""
        path = self._replays_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.warning("Failed to load level replays: %s", e)
            return {}

    def _should_skip_replay(self, level: int) -> bool:
        """Check if replay should be skipped for this game + level."""
        if level in self._skip_replay_global:
            return True
        # Check per-game rules: match if game_id starts with the configured prefix
        for game_prefix, levels in self._skip_replay_game.items():
            if self.game_id.startswith(game_prefix) and level in levels:
                return True
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

    def _reset_skill_from_template(self) -> None:
        """Copy skill template to active skills dir, resetting agent knowledge."""
        template = Path(__file__).resolve().parent / "skills" / "arc_game_playing" / "SKILL.md"
        if not template.exists():
            logger.warning("Skill template not found at %s", template)
            return
        dest = self._get_skill_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(template, dest)
        logger.info("Skill reset from template: %s -> %s", template, dest)

    # --------------------------------------------------------------------- #
    # Tool definitions
    # --------------------------------------------------------------------- #

    def build_functions(self) -> list[dict[str, Any]]:
        """Game actions + update_game_notes tool."""
        functions = super().build_functions()
        functions.append({
            "name": "execute_plan",
            "description": (
                "Execute a sequence of game actions efficiently WITHOUT pausing "
                "for observation between each step. Use this for navigation when "
                "you have a clear path (e.g., 'go DOWN 3 times then LEFT 2 times'). "
                "Execution stops early if: (1) an action has no effect (wall hit), "
                "(2) score changes (trigger collected / level complete), or "
                "(3) game state changes. You will receive observation of the "
                "final state after execution. This saves time and energy — "
                "ALWAYS prefer this over single actions when navigating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "string",
                        "description": (
                            "Comma-separated list of actions to execute in order. "
                            "E.g., 'ACTION2,ACTION2,ACTION2,ACTION3,ACTION3'. "
                            "Valid: ACTION1-ACTION6, RESET. Max 15 actions."
                        ),
                    },
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        })
        functions.append({
            "name": "navigate_to",
            "description": (
                "Navigate to a grid position using BFS pathfinding. "
                "The system will compute the shortest path avoiding known walls "
                "and execute it automatically. Use this instead of manual "
                "navigation when you know where you want to go. "
                "The path will stop early on wall hit or score change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "Target row position on the grid.",
                    },
                    "col": {
                        "type": "integer",
                        "description": "Target column position on the grid.",
                    },
                },
                "required": ["row", "col"],
                "additionalProperties": False,
            },
        })
        functions.append({
            "name": "update_game_notes",
            "description": (
                "Record discovered game knowledge to your persistent skill file. "
                "Use this to save confirmed action mappings, game rules, level "
                "strategies, object roles, or tips. This knowledge persists across "
                "retries and appears in your system prompt for future decisions. "
                "You can call this TOGETHER with a game action in the same turn — "
                "no need to call it separately. Only record concise confirmed facts."
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

        prev_frame = frames[-2] if len(frames) > 1 else frames[-1]

        # --- Resolve deferred navigator update from PREVIOUS action ---
        if self._pending_nav_update is not None:
            old_pos, old_action = self._pending_nav_update
            self._pending_nav_update = None

            # Detect actual player movement using player-colored cells
            player_colors = getattr(self.perception, "_player_color_votes", {})
            confirmed_colors = {c for c, v in player_colors.items() if v >= 1}
            prev_grid = prev_frame.frame[0] if prev_frame.frame else None
            curr_grid = latest_frame.frame[0] if latest_frame.frame else None

            if prev_grid is not None and curr_grid is not None and confirmed_colors:
                moved = _did_player_move(prev_grid, curr_grid, confirmed_colors)
            else:
                nav_diff = compute_frame_diff(prev_frame, latest_frame)
                nav_cells = _parse_cells_changed(nav_diff)
                moved = nav_cells > MIN_CELLS_FOR_MOVEMENT

            # Get current player position directly from grid
            new_pos = self._detect_player_from_grid()

            if not moved:
                self._navigator.update_from_action_result(
                    old_pos, old_action, moved=False, new_player_pos=None,
                )
            else:
                self._navigator.update_from_action_result(
                    old_pos, old_action, moved=True,
                    new_player_pos=new_pos,
                )

        # --- Phase 0: Level replay (skip LLM calls entirely) ---
        if self._replay_queue and self._replay_index < len(self._replay_queue):
            action_name = self._replay_queue[self._replay_index]
            self._replay_index += 1
            action = GameAction.from_name(action_name)

            self._record_step(action, prev_frame, latest_frame)
            self._last_action_name = action.name

            logger.info(
                "REPLAY step %d/%d: %s (level %d)",
                self._replay_index,
                len(self._replay_queue),
                action_name,
                latest_frame.levels_completed,
            )
            self._pending_nav_update = (self._get_player_position() or (0, 0), action.name)
            return action

        # Replay queue exhausted — clear it and switch to LLM mode
        if self._replay_queue:
            logger.info(
                "Replay exhausted (%d actions) — switching to LLM mode",
                len(self._replay_queue),
            )
            self._replay_queue = []
            self._replay_index = 0
            # Start LLM mode with fresh conversation
            self.messages = []

        # --- Phase 0b: Execute planned actions (from execute_plan tool) ---
        if self._action_plan:
            # Check if previous plan step hit a wall.
            # Use player-specific movement detection to avoid false positives
            # from grid transformations (color flashes etc.).
            should_interrupt = False
            interrupt_reason = ""

            player_colors = getattr(self.perception, "_player_color_votes", {})
            confirmed_colors = {
                c for c, v in player_colors.items() if v >= 1
            }
            prev_grid = prev_frame.frame[0] if prev_frame.frame else None
            curr_grid = latest_frame.frame[0] if latest_frame.frame else None

            if prev_grid is not None and curr_grid is not None and confirmed_colors:
                # Primary check: did the player actually move?
                moved = _did_player_move(prev_grid, curr_grid, confirmed_colors)
                if not moved:
                    should_interrupt = True
                    interrupt_reason = "Player did not move (wall hit)"
            else:
                # Fallback: use cell count when player colors unknown
                plan_diff = compute_frame_diff(prev_frame, latest_frame)
                plan_cells = _parse_cells_changed(plan_diff)
                if plan_cells <= MIN_CELLS_FOR_MOVEMENT:
                    should_interrupt = True
                    interrupt_reason = (
                        f"Previous action had no effect (only {plan_cells} cells changed — wall hit)"
                    )

            if not should_interrupt:
                # Check for score change (level transition etc.)
                prev_score = getattr(prev_frame, "score", prev_frame.levels_completed)
                curr_score = getattr(latest_frame, "score", latest_frame.levels_completed)
                if curr_score != prev_score:
                    should_interrupt = True
                    interrupt_reason = f"Score changed {prev_score}→{curr_score}!"

            if should_interrupt:
                skipped = len(self._action_plan)
                self._action_plan = []
                logger.info(
                    "Plan interrupted (%s), %d actions skipped",
                    interrupt_reason, skipped,
                )
                # Fall through to normal LLM mode
            else:
                action_name = self._action_plan.pop(0)
                action = GameAction.from_name(action_name)
                self._record_step(action, prev_frame, latest_frame)
                self._last_action_name = action.name
                logger.info(
                    "PLAN step: %s (%d remaining)",
                    action_name, len(self._action_plan),
                )
                # Get current player position directly from grid
                pos = self._get_player_position()
                if pos:
                    self._pending_nav_update = (pos, action.name)
                return action

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
                    self._record_llm_event(
                        "observation_request",
                        messages=[
                            self._serialize_message(m)
                            for m in self._messages_for_api()
                        ],
                    )
                    response = client.chat.completions.create(**create_kwargs)
                    obs_content = response.choices[0].message.content or ""
                    self._record_llm_event(
                        "observation_response",
                        content=obs_content,
                    )
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
        self._step_count_for_vlm += 1
        user_prompt = self.build_user_prompt(latest_frame)
        include_image = self._should_include_image() and self._current_bbox_image is not None
        if include_image:
            # Multimodal message: image + text in a single user turn
            data_url = ObjectAnnotator._encode_image(self._current_bbox_image)
            self.push_message({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                    {"type": "text", "text": user_prompt},
                ],
            })
        else:
            self.push_message({"role": "user", "content": user_prompt})

        name = GameAction.ACTION5.name  # fallback
        arguments = None
        notes_calls = 0

        while True:
            logger.info("Sending to Assistant for action...")
            message = None
            for attempt in range(1, self.API_RETRIES + 1):
                try:
                    # Some providers (e.g. SiliconFlow/Qwen) don't support
                    # tool_choice="required"; fall back to "auto" for them.
                    _tc = (
                        "auto" if "qwen" in self._model.lower()
                        else "required"
                    )
                    create_kwargs = {
                        "model": self._model,
                        "messages": self._messages_for_api(),
                        "tools": tools,
                        "tool_choice": _tc,
                    }
                    if self.REASONING_EFFORT is not None:
                        create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                    self._record_llm_event(
                        "action_request",
                        messages=[
                            self._serialize_message(m)
                            for m in self._messages_for_api()
                        ],
                        tools=[t["function"]["name"] for t in tools],
                    )
                    response = client.chat.completions.create(**create_kwargs)
                    self.track_tokens(response.usage.total_tokens)
                    message = response.choices[0].message
                    self._record_llm_event(
                        "action_response",
                        message=self._serialize_message(message),
                    )
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
                # Some models (Qwen, GLM) embed tool calls in text instead
                # of using the tool_calls field.  Try to extract one.
                _extracted = self._extract_tool_call_from_text(
                    message.content or ""
                )
                if _extracted:
                    name, arguments = _extracted
                    self.push_message(message)
                    logger.info(
                        "Extracted tool call from text: %s(%s)", name, arguments
                    )
                    break
                # Genuinely no tool call — use fallback
                self.push_message(message)
                break

            # Push the assistant message (contains all tool_calls)
            self.push_message(message)

            # --- Process ALL tool calls from this response ---
            # Separate notes from game actions so notes + action = 1 API call
            game_tc = None  # the game action / navigate / execute_plan tool call
            need_loop = False  # whether we need another API call

            for tc in message.tool_calls:
                if tc.function.name == "update_game_notes":
                    if notes_calls < self.MAX_NOTES_PER_TURN:
                        notes_calls += 1
                        result = self._handle_game_notes_update(
                            tc.function.arguments
                        )
                        self.push_message({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        logger.info(
                            "update_game_notes call %d/%d: %s",
                            notes_calls, self.MAX_NOTES_PER_TURN, result,
                        )
                    else:
                        self.push_message({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"Error: max {self.MAX_NOTES_PER_TURN} "
                                "update_game_notes calls per turn. "
                                "Now call a game action."
                            ),
                        })
                elif game_tc is None:
                    # First non-notes tool call = the game action
                    game_tc = tc
                else:
                    # Extra game action — error it
                    self.push_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Error: only one game action at a time.",
                    })

            # If model only called notes (no game action), loop for another API call
            if game_tc is None:
                need_loop = True

            if need_loop:
                continue

            # --- Process the game action tool call ---
            self._latest_tool_call_id = game_tc.id

            if game_tc.function.name == "navigate_to":
                # Parse target position and compute BFS path
                try:
                    args = json.loads(game_tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                target_r = args.get("row", 0)
                target_c = args.get("col", 0)

                # Sync action mappings from perception
                self._navigator.update_action_mappings(
                    self.perception._action_effects
                )

                # Get player position
                player_pos = self._get_player_position()
                if player_pos is None:
                    self.push_message({
                        "role": "tool",
                        "tool_call_id": game_tc.id,
                        "content": (
                            "Error: cannot determine player position. "
                            "Use single actions to explore first."
                        ),
                    })
                    continue

                path = self._navigator.navigate_to(
                    target_r, target_c, player_pos[0], player_pos[1]
                )
                if path is None or len(path) == 0:
                    if path is not None and len(path) == 0:
                        msg = "Already at target position."
                    else:
                        msg = (
                            f"No path found to ({target_r},{target_c}). "
                            "Try navigating manually or to a different target."
                        )
                    self.push_message({
                        "role": "tool",
                        "tool_call_id": game_tc.id,
                        "content": msg,
                    })
                    continue

                # Take first action, queue rest
                name = path[0]
                arguments = json.dumps({})
                self._action_plan = path[1:]
                logger.info(
                    "navigate_to(%d,%d): %d-step path from (%d,%d) -> %s",
                    target_r, target_c, len(path),
                    player_pos[0], player_pos[1],
                    ",".join(path),
                )
                break
            elif game_tc.function.name == "execute_plan":
                # Parse and validate the action plan
                try:
                    args = json.loads(game_tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                actions_str = args.get("actions", "")
                plan_actions = [
                    a.strip() for a in actions_str.split(",") if a.strip()
                ]
                valid_names = {
                    "RESET", "ACTION1", "ACTION2", "ACTION3",
                    "ACTION4", "ACTION5", "ACTION6",
                }
                invalid = [a for a in plan_actions if a not in valid_names]
                if invalid:
                    self.push_message({
                        "role": "tool",
                        "tool_call_id": game_tc.id,
                        "content": f"Error: invalid actions: {invalid}",
                    })
                    continue
                if not plan_actions:
                    self.push_message({
                        "role": "tool",
                        "tool_call_id": game_tc.id,
                        "content": "Error: no actions provided.",
                    })
                    continue
                # Cap at 15 actions
                plan_actions = plan_actions[:15]
                # Take first action now, queue the rest
                name = plan_actions[0]
                arguments = json.dumps({})
                self._action_plan = plan_actions[1:]
                logger.info(
                    "execute_plan: %d actions queued (%s)",
                    len(plan_actions),
                    ",".join(plan_actions),
                )
                break
            else:
                # Single game action (ACTION1-6, RESET)
                name = game_tc.function.name
                arguments = game_tc.function.arguments
                logger.debug(
                    f"Assistant: {name} ({game_tc.id}) {arguments}"
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
        player_pos = self._get_player_position()
        if player_pos:
            self._pending_nav_update = (player_pos, action.name)
        return action

    # ------------------------------------------------------------------ #
    #  Recording helpers
    # ------------------------------------------------------------------ #

    def push_message(self, message: Any) -> list[dict[str, Any]]:
        """Override to record tool calls and tool results to recording."""
        result = super().push_message(message)
        # Record tool-role messages (tool results) and assistant tool_calls
        if hasattr(self, "recorder") and not self.is_playback:
            msg = self._serialize_message(message)
            role = msg.get("role") if isinstance(msg, dict) else None
            if role == "tool":
                self._record_llm_event(
                    "tool_result",
                    tool_call_id=msg.get("tool_call_id"),
                    content=msg.get("content", "")[:500],
                )
            elif role == "assistant" and msg.get("tool_calls"):
                self._record_llm_event(
                    "tool_call",
                    tool_calls=msg["tool_calls"],
                )
        return result

    def _record_llm_event(self, event_type: str, **kwargs: Any) -> None:
        """Record an LLM interaction event to the recording file.

        event_type: 'observation_request', 'observation_response',
                    'action_request', 'action_response',
                    'tool_call', 'tool_result'
        """
        if not hasattr(self, "recorder") or self.is_playback:
            return
        data: dict[str, Any] = {"event": event_type}
        data.update(kwargs)
        self.recorder.record(data)

    def _serialize_message(self, msg: Any) -> Any:
        """Convert an OpenAI message object or dict to JSON-serializable form."""
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        return str(msg)

    # ------------------------------------------------------------------ #
    #  Qwen / GLM text-based tool call extraction
    # ------------------------------------------------------------------ #

    _TEXT_ACTION_RE = re.compile(
        r"(?:ACTION[1-6]|RESET|navigate_to|execute_plan)"
    )

    def _extract_tool_call_from_text(
        self, text: str
    ) -> Optional[tuple[str, str]]:
        """Try to parse a tool call embedded in plain text.

        Some models (Qwen, GLM) sometimes write tool calls in their text
        content instead of using the structured tool_calls field.
        Handles patterns like:
          - {"name": "ACTION1", "arguments": {}}
          - <tool_call>ACTION1</tool_call>
          - navigate_to(33, 21)

        Returns (name, arguments_json) or None.
        """
        # Pattern 1: JSON object with "name" key
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', text):
            fn_name = m.group(1)
            if self._TEXT_ACTION_RE.match(fn_name):
                # Try to extract arguments
                try:
                    obj = json.loads(m.group(0))
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        return fn_name, args
                    return fn_name, json.dumps(args)
                except json.JSONDecodeError:
                    return fn_name, "{}"

        # Pattern 2: <tool_call>ACTION1</tool_call>
        tc_match = re.search(r"<tool_call>\s*(\w+)\s*</tool_call>", text)
        if tc_match:
            fn_name = tc_match.group(1)
            if self._TEXT_ACTION_RE.match(fn_name):
                return fn_name, "{}"

        # Pattern 3: navigate_to(row, col) in text
        nav_match = re.search(r"navigate_to\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if nav_match:
            row, col = nav_match.group(1), nav_match.group(2)
            return "navigate_to", json.dumps({"row": int(row), "col": int(col)})

        # Pattern 4: bare ACTION name at end of text
        bare_match = re.search(r"\b(ACTION[1-6]|RESET)\b\s*$", text.strip())
        if bare_match:
            return bare_match.group(1), "{}"

        return None

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

        # --- Game intelligence module updates ---
        player_pos = self._get_player_position()

        # NOTE: Navigator wall/walkable updates are handled via the deferred
        # _pending_nav_update mechanism in choose_action (resolved at the START
        # of the next call when the result frame is available).

        # InteractionTracker: detect events
        prev_grid = None
        curr_grid = None
        if hasattr(self, "frames") and len(self.frames) > 1:
            if prev_frame.frame:
                prev_grid = prev_frame.frame[0]
            if latest_frame.frame:
                curr_grid = latest_frame.frame[0]

        if prev_grid is not None and curr_grid is not None:
            events = self._interaction_tracker.track_step(
                step_num=len(self._current_steps),
                prev_grid=prev_grid,
                curr_grid=curr_grid,
                player_pos=player_pos,
                score_before=prev_score,
                score_after=curr_score,
                action=action.name,
            )
            # Mark collected items in inventory
            collected_any = False
            for ev in events:
                if (
                    ev.event_type.value == "COLLECTED"
                    and ev.object_color is not None
                    and ev.position is not None
                ):
                    self._inventory.mark_collected(ev.object_color, ev.position)
                    collected_any = True

                    # Invalidate confirmed walls near the collected trigger —
                    # grid layout may have changed (doors opened, walls removed).
                    qr = (ev.position[0] // 5) * 5
                    qc = (ev.position[1] // 5) * 5
                    removed = self._navigator.memory.invalidate_near(
                        (qr, qc), radius=3
                    )
                    if removed > 0:
                        logger.info(
                            "Invalidated %d wall edges near collected "
                            "trigger at (%d,%d)",
                            removed, qr, qc,
                        )

            # Trigger collection changed the grid — refresh walkability
            # and interrupt any running plan (path is stale).
            if collected_any and curr_grid is not None:
                analysis = self.perception.analyze_grid(curr_grid)
                self._navigator.refresh_walkability(
                    curr_grid, analysis.bg_color
                )
                if self._action_plan:
                    skipped = len(self._action_plan)
                    self._action_plan = []
                    logger.info(
                        "Plan interrupted: trigger collected, "
                        "grid changed — %d actions skipped",
                        skipped,
                    )

            # Transformation detection (rotation, mirror, color map, shift)
            self._transformation_detector.analyze_step(
                step_num=len(self._current_steps),
                prev_grid=prev_grid,
                curr_grid=curr_grid,
                player_pos=player_pos,
                action=action.name,
            )

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

        # Attach reasoning metadata + LLM reasoning log
        reasoning_log = self._last_response_content or ""
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
            "reasoning_log": reasoning_log[:2000],
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

        # Periodic video snapshot every 10 total actions (overwrites previous)
        total_actions = self.action_counter if hasattr(self, "action_counter") else 0
        if total_actions > 0 and total_actions % 10 == 0:
            self._generate_video("progress")

    # --------------------------------------------------------------------- #
    # Player position helper
    # --------------------------------------------------------------------- #

    def _should_include_image(self) -> bool:
        """Decide whether to embed bbox image in the current step's message.

        Modes:
          - "always": every step
          - "periodic": first INLINE_VLM_WARMUP steps, then every INLINE_VLM_INTERVAL steps
          - "off": never
        """
        if self.INLINE_VLM == "always":
            return True
        if self.INLINE_VLM == "off":
            return False
        # periodic mode
        step = self._step_count_for_vlm
        if step < self.INLINE_VLM_WARMUP:
            return True
        return (step - self.INLINE_VLM_WARMUP) % self.INLINE_VLM_INTERVAL == 0

    def _get_player_position(self) -> Optional[tuple[int, int]]:
        """Extract current player position.

        During/after plan execution, compute position directly from the
        latest grid frame using known player colors (avoids stale
        perception._player_obj).

        Returns (row, col) or None if player not yet identified.
        """
        # Try direct grid detection first (works during plans)
        pos = self._detect_player_from_grid()
        if pos is not None:
            self._last_detected_pos = pos
            return pos

        # Fallback to perception
        player = self.perception._player_obj
        if player is not None:
            pos = (int(player.center_r), int(player.center_c))
            self._last_detected_pos = pos
            return pos
        return None

    # Rows reserved for UI at the bottom (energy bar, score display)
    _UI_BOTTOM_ROW = 52

    def _detect_player_from_grid(self) -> Optional[tuple[int, int]]:
        """Find player position directly from the latest frame grid.

        Uses confirmed player colors, excludes UI rows.  Instead of
        flood-fill clustering (which breaks when the player overlaps
        game patterns), uses a density-based search around the expected
        player area.

        Strategy:
        1. If we have a last-known position, search in a window around it.
        2. Otherwise, find the densest 10x10 window of player-colored cells
           in the playable area.
        """
        if not self.frames:
            return None
        latest = self.frames[-1]
        if not latest.frame:
            return None
        grid = latest.frame[0]

        player_votes = getattr(self.perception, "_player_color_votes", {})
        colors = {c for c, v in player_votes.items() if v >= 1}
        if not colors:
            return None

        H = min(len(grid), self._UI_BOTTOM_ROW)
        W = len(grid[0]) if len(grid) > 0 else 0

        # Get last known position as anchor (prefer most recent detection)
        anchor: Optional[tuple[int, int]] = getattr(
            self, "_last_detected_pos", None
        )
        if anchor is None:
            player_obj = self.perception._player_obj
            if player_obj is not None:
                anchor = (int(player_obj.center_r), int(player_obj.center_c))

        # If we have an anchor, search a 20x20 window around it
        if anchor is not None:
            ar, ac = anchor
            r_min = max(0, ar - 10)
            r_max = min(H, ar + 10)
            c_min = max(0, ac - 10)
            c_max = min(W, ac + 10)
            r_sum, c_sum, count = 0.0, 0.0, 0
            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    if grid[r][c] in colors:
                        r_sum += r
                        c_sum += c
                        count += 1
            if count >= 5:
                return (int(r_sum / count), int(c_sum / count))

        # No anchor or anchor search failed: sliding window approach
        # Find the 10x10 window with the most player-colored cells
        best_count = 0
        best_r, best_c = 0, 0
        step = 5  # slide by STEP_SIZE for efficiency
        for wr in range(0, H - 10, step):
            for wc in range(0, W - 10, step):
                count = 0
                for r in range(wr, min(wr + 10, H)):
                    for c in range(wc, min(wc + 10, W)):
                        if grid[r][c] in colors:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_r, best_c = wr, wc

        if best_count < 5:
            return None

        # Compute centroid within the best window
        r_sum, c_sum, total = 0.0, 0.0, 0
        for r in range(best_r, min(best_r + 10, H)):
            for c in range(best_c, min(best_c + 10, W)):
                if grid[r][c] in colors:
                    r_sum += r
                    c_sum += c
                    total += 1
        return (int(r_sum / total), int(c_sum / total))

    # --------------------------------------------------------------------- #
    # Level completion auto-summary
    # --------------------------------------------------------------------- #

    def _generate_level_summary(self, level_num: int) -> None:
        """Auto-generate a clean level summary and write it to SKILL.md.

        Called from _prepare_next_level() BEFORE modules are reset, so all
        accumulated knowledge (interaction rules, inventory classifications,
        events) is still available.
        """
        parts: list[str] = [f"## Level {level_num} Summary (auto-generated)"]

        # 1. Action mappings (from perception)
        effects = getattr(self.perception, "_action_effects", {})
        if effects:
            mapping_lines = []
            for action, direction in sorted(effects.items()):
                mapping_lines.append(f"- {action} = {direction}")
            parts.append("Action mappings: " + ", ".join(
                f"{a}={d}" for a, d in sorted(effects.items())
            ))

        # 2. Interaction rules (what objects do)
        rules = self._interaction_tracker.get_rules()
        if rules:
            parts.append("Interaction rules:")
            for rule in sorted(rules.values(), key=lambda r: r.confidence, reverse=True):
                parts.append(f"  {rule.describe()}")

        # 3. Key events — focus on SCORED and what happened right before
        scored_events = [
            e for e in self._interaction_tracker._events
            if e.event_type == InteractionEventType.SCORED
        ]
        collected_events = [
            e for e in self._interaction_tracker._events
            if e.event_type == InteractionEventType.COLLECTED
        ]
        if scored_events:
            se = scored_events[-1]  # last score event = level completion trigger
            parts.append(f"Level completion trigger: {se.description} at step {se.step}")
            # What was collected right before scoring?
            pre_score_collected = [
                e for e in collected_events if e.step <= se.step
            ]
            if pre_score_collected:
                colors_collected = set(e.object_color for e in pre_score_collected if e.object_color is not None)
                parts.append(f"Objects collected before scoring: colors {sorted(colors_collected)}")
                for e in pre_score_collected[-3:]:  # last 3
                    parts.append(f"  Step {e.step}: {e.description}")

        # 4. Object role classifications (from inventory)
        inv = self._inventory
        role_parts = []
        if inv._player_colors:
            role_parts.append(f"Player colors: {sorted(inv._player_colors)}")
        if inv._floor_colors:
            role_parts.append(f"Floor colors: {sorted(inv._floor_colors)}")
        if inv._wall_colors:
            role_parts.append(f"Wall colors: {sorted(inv._wall_colors)}")
        if inv._collected_colors:
            role_parts.append(f"Collectible colors: {sorted(inv._collected_colors)}")
        if role_parts:
            parts.append("Object classifications: " + "; ".join(role_parts))

        # 5. Toggle positions (if any)
        toggles = self._interaction_tracker._toggle_positions
        if toggles:
            parts.append(f"Toggle positions detected: {len(toggles)} — avoid revisiting!")

        # 5b. Grid transformations detected during this level
        if self._transformation_detector._history:
            parts.append("Grid transformations observed:")
            seen_types: set[str] = set()
            for tev in self._transformation_detector._history:
                ttype = tev.transform_type.value
                if ttype not in seen_types:
                    seen_types.add(ttype)
                    parts.append(f"  {tev.description}")
            if self._transformation_detector._learned_rules:
                parts.append("  Learned transform triggers:")
                for (act, pos), types in self._transformation_detector._learned_rules.items():
                    unique = sorted(set(t.value for t in types))
                    parts.append(f"    {act} near {pos} -> {', '.join(unique)}")

        # 6. Step count and efficiency
        total_steps = len(self._current_steps)
        parts.append(f"Completed in {total_steps} steps")

        # 7. Strategy recommendation for next level
        parts.append("")
        parts.append(f"## Strategy for Level {level_num + 1}")
        if collected_events and scored_events:
            parts.append("- Collect prerequisite objects FIRST, then navigate to exit/goal")
            colors_needed = set(e.object_color for e in collected_events if e.object_color is not None)
            if colors_needed:
                parts.append(f"- Look for objects of colors {sorted(colors_needed)} — they were prerequisites in Level {level_num}")
        elif scored_events and not collected_events:
            parts.append("- Navigate directly to exit/goal (no prerequisites needed)")
        parts.append("- Scan OBJECT INVENTORY for exit structures and prerequisite objects")
        parts.append("- Same game mechanics apply — reuse learned interaction rules")

        summary_text = "\n".join(parts)

        # Write to SKILL.md — replace level_strategies section with clean summary
        notes_json = json.dumps({
            "section": "level_strategies",
            "content": summary_text,
            "replace": "true",
        })
        result = self._handle_game_notes_update(notes_json)
        logger.info(
            "Auto-generated Level %d summary (%d chars): %s",
            level_num, len(summary_text), result,
        )

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
        """Observation prompt: perception data + buffer context.

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

            # Store bbox image for inline VLM mode (used in Phase 4 multimodal message)
            grid = latest_frame.frame[0]
            analysis = self.perception.analyze_grid(grid)
            bbox_img = self.perception.render_grid(grid, analysis=analysis)
            if bbox_img is not None:
                self._current_bbox_image = bbox_img

            if self.INLINE_VLM != "off":
                # Skip separate VLM calls — LLM sees image inline (always or periodic)
                pass
            else:
                # VLM: entity identification + change interpretation
                vlm_sections = self._get_vlm_analysis(
                    latest_frame, prev_frame_grids
                )
                sections.extend(vlm_sections)

            # --- Game intelligence module updates for prompt ---
            # (grid and analysis already computed above for bbox rendering)
            # Sync plan-estimated position now that perception has run
            player_pos = self._get_player_position()

            # Update navigator with action mappings and grid analysis
            self._navigator.update_action_mappings(
                self.perception._action_effects
            )
            if not self._navigator._initialized:
                self._navigator.infer_floor_and_walls(grid, analysis.bg_color)
            else:
                # Re-infer walkability every observation to catch grid
                # changes from trigger collection, level transitions, etc.
                self._navigator.refresh_walkability(grid, analysis.bg_color)

            # Record current player position as walkable (perception is properly updated here)
            if player_pos:
                qr = (player_pos[0] // 5) * 5
                qc = (player_pos[1] // 5) * 5
                self._navigator.memory.record_walkable((qr, qc))

            # Update inventory with current objects
            player_colors = set()
            if self.perception._player_obj is not None:
                player_colors.add(self.perception._player_obj.color)
            for c, count in self.perception._player_color_votes.items():
                if count >= 1:
                    player_colors.add(c)

            self._inventory.update_from_analysis(
                objects=analysis.objects,
                bg_color=analysis.bg_color,
                player_pos=player_pos,
                player_colors=player_colors,
                step=len(self._current_steps),
                interaction_rules=self._interaction_tracker.get_rules(),
            )
            self._inventory.update_distances(player_pos, self._navigator)

        # Game intelligence prompt sections
        rules_summary = self._interaction_tracker.get_rules_summary()
        if rules_summary:
            sections.append(rules_summary)
        toggle_warnings = self._interaction_tracker.get_toggle_warnings()
        if toggle_warnings:
            sections.append(toggle_warnings)
        recent_events = self._interaction_tracker.get_recent_events(3)
        if recent_events:
            sections.append(recent_events)
        transform_summary = self._transformation_detector.get_summary()
        if transform_summary:
            sections.append(transform_summary)

        inventory_summary = self._inventory.get_inventory_summary()
        if inventory_summary:
            sections.append(inventory_summary)

        navigator_status = self._navigator.get_status_summary()
        if navigator_status:
            sections.append(navigator_status)

        # Past attempts from buffer
        buffer_ctx = self._build_buffer_context()
        if buffer_ctx:
            sections.append(buffer_ctx)

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

        # Level transition hint — encourage applying cross-level learning
        levels_done = getattr(latest_frame, "levels_completed", 0)
        if levels_done > 0 and len(self._current_steps) < 5:
            sections.append(
                "# LEVEL TRANSITION\n"
                f"You just entered level {levels_done + 1}. "
                "The game mechanics from previous levels likely still apply.\n"
                "1. Check your game_notes for rules you discovered in previous levels\n"
                "2. Scan the grid for the SAME types of interactive objects\n"
                "3. Apply the same prerequisite/collection strategy that worked before\n"
                "4. Use navigate_to(row, col) for efficient pathfinding to targets\n"
                "5. Use update_game_notes to record any new discoveries"
            )

        # Instruction
        sections.append(
            "# YOUR TURN\n"
            "Follow the arc_game_playing skill reasoning template.\n"
            "If you confirmed new knowledge, call update_game_notes first.\n"
            "Then call navigate_to(row, col) to reach a target from OBJECT INVENTORY,\n"
            "or a single ACTION for exploration. Do NOT write navigate_to in text — "
            "you MUST call it as a tool/function call."
        )

        return "\n\n".join(sections)

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Minimal user prompt — strategy is in the skills system prompt."""
        base = textwrap.dedent("""\
            You are playing an unknown ARC-AGI-3 grid game.
            Follow your grid_perception and arc_game_playing skills.

            ## Tool priority (IMPORTANT — use the FIRST applicable tool):
            1. **navigate_to(row, col)** — ALWAYS use this to reach any target in OBJECT INVENTORY.
               It auto-computes BFS shortest path avoiding walls. Call it as a function/tool call.
               Example: to go to color 1 at (33,21), call the navigate_to tool with row=33, col=21.
            2. execute_plan("ACTION2,ACTION3,...") — only when you know an exact short sequence.
            3. Single ACTION1-ACTION6 — only for the first 3 steps to learn direction mappings.
            4. update_game_notes — save confirmed knowledge (call together with a game action).

            RULE: After direction mappings are learned, ALWAYS use navigate_to to reach targets.
            Never manually chain single actions to reach a known position — navigate_to does it better.
        """)
        if self._should_include_image() and self._current_bbox_image is not None:
            base += textwrap.dedent("""\

                The image above shows the current game grid with bounding boxes around detected objects.
                A white UP arrow in the top-right indicates orientation.

                Before choosing an action:
                1. Identify key entities in the image (player, walls, collectibles, doors, etc.)
                2. If you took an action last turn, assess its effect (did you move? hit a wall? collect something?)
                3. Then choose your next action using a tool call.
            """)
        return base

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
            "Your current strategy is not working. You MUST change approach:",
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

        # Suggest exploration when all inventory items are visited
        all_visited = (
            self._inventory._items
            and all(
                item.status
                in (ObjectStatus.VISITED, ObjectStatus.COLLECTED)
                for item in self._inventory._items
            )
        )
        if all_visited:
            lines.append(
                "All known objects were NOT interactive. "
                "Navigate to a completely different area of the grid "
                "(e.g., opposite corner) to discover new objects."
            )

        # Suggest exploration targets based on grid coverage
        player_pos = self._get_player_position()
        if player_pos and ld.steps_since_score_change >= 12:
            pr, pc = player_pos
            suggestions = []
            if pr > 30:
                suggestions.append("navigate_to(10, 30)")
            else:
                suggestions.append("navigate_to(50, 30)")
            if pc > 30:
                suggestions.append(f"navigate_to({pr}, 10)")
            else:
                suggestions.append(f"navigate_to({pr}, 50)")
            lines.append(f"Try exploring: {' or '.join(suggestions)}")

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Name override
    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        if self.INLINE_VLM == "always":
            obs = "inline-vlm"
        elif self.INLINE_VLM == "periodic":
            obs = "periodic-vlm"
        elif self.DO_OBSERVATION:
            obs = "with-observe"
        else:
            obs = "no-observe"
        model = self._model.replace("/", "-").replace(":", "-")
        ts = getattr(self, "_start_time_tag", "0000_0000")
        return (
            f"{self.game_id}.reactbuffer.{model}.{obs}"
            f".retry{self.MAX_RETRIES}.{ts}"
        )
