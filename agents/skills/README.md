# ARC game agent skills

Skills are loaded by the agent when `skills_dir` points to this directory (or when each skill is copied into `~/.claude/skills/`).

## Reasoning skills (use every step)

- **first_principles (第一性原理)**: Reason from minimal constraints and facts; build up from basics; avoid copying by analogy. Apply on every planning and analysis step.
- **critical_thinking (批判性思考)**: Question assumptions, consider alternatives, test hypotheses before believing. Apply on every observation and interpretation step.

These are **default reasoning habits** — the system prompt tells the agent to use them flexibly on all steps, not only when a task "matches."

## arc_scene_analysis

- **When to use**: Analyzing the game grid (e.g. after `get_game_state()`), segmenting objects, and writing a World Objects table to memory.
- **Defines**: Required World Objects table format (header, columns, bbox conventions), what to recognize, and optional `<SCENE_ANALYSIS>` block format.

## arc_agent_memory

- **When to use**: When you need to record or recall the ARC game agent's persistent knowledge (e.g. planning a step, after get_game_state or take_action).
- **Target**: The ARC game agent's long-term markdown memory file (one per game).
- **Defines**: How to read and update that memory (read_memory, update_memory): when to call, valid sections (Task Objectives, World Rules, World Objects, Observation Summary, Last Step Changes, TODO, Other Notes), append vs replace, batch updates, and coordination with arc_scene_analysis.

## arc_frame_diff

- **When to use**: After take_action, when you need to analyze object-level changes between the previous frame and the current frame (before vs after the action).
- **Target**: The two "frames"—previous scene (last World Objects table or last analysis) and current scene (get_game_state + get_connected_components).
- **Defines**: What counts as previous vs current frame, how to match objects across frames, how to classify changes (unchanged / moved / appeared / disappeared / state_changed), and how to summarize and persist (Last Step Changes, World Rules, Other Notes; SCENE_ANALYSIS or update_memory).

## arc_take_action

- **When to use**: Whenever you are about to call `take_action()`.
- **Rule**: **Look first, then act**. You must observe the scene and update World Objects in memory before each `take_action()`; after acting, observe again before the next action.
- **Defines**: The order of each cycle (observe → update World Objects → take_action → repeat), why the gate requires looking first, and coordination with arc_scene_analysis.

To enable for the agent, pass `skills_dir` when creating the agent, e.g.:

```python
from pathlib import Path
from agent.core.arc_agent import ArcGameAgent
from agent.tools.arc_tools import create_arc_game_tools

project_root = Path(__file__).resolve().parent.parent  # adjust if your script is elsewhere
skills_dir = project_root / "skills"
tools = create_arc_game_tools(game_id="ls20", seed=0)
agent = ArcGameAgent(game_id="ls20", seed=0, tools=tools, skills_dir=str(skills_dir))
```
