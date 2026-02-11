---
name: arc_game_playing
description: "Core strategy and reasoning framework for playing ARC-AGI-3 grid-based games. Covers the game loop (observe → hypothesize → act → learn), action semantics, common game patterns, exploration vs exploitation, and how to recover from failures. Apply at every step."
---

# ARC-AGI-3 游戏攻略 (Game Playing Strategy)

You are playing an **unknown** dynamic grid-based game. The rules, objectives, and mechanics are NOT told to you — you must **discover** them through observation and experimentation.

## Game Basics

- **Grid**: One or more 64x64 matrices, cell values 0-15 (each value = a color).
- **Actions**: RESET, ACTION1-ACTION6. Their meanings vary per game (could be directions, interactions, etc.).
- **States**: NOT_PLAYED → PLAYING → WIN or GAME_OVER.
- **Score**: Shown in observations. Score increase = progress. The goal is to reach WIN.
- **Turns are limited**: Every action costs a turn. Minimize wasted actions.

## Core Loop: Observe → Hypothesize → Act → Learn

Every step, follow this cycle:

### 1. Observe (read the perception data carefully)

- What objects exist? Where is the player? What changed since last frame?
- Did the last action have an effect, or was it a no-op (wall hit)?
- Is the score the same, higher, or did state change?

### 2. Hypothesize (form or update your mental model)

- **Early game** (steps 1-5): You know nothing. Your priority is to **map actions to effects**.
  - Try each of ACTION1-ACTION4 once and observe what happens.
  - Identify which action moves which direction (or does something else).
- **Mid game**: You should know the action mappings. Focus on:
  - What is the objective? (reach a door? collect items? solve a puzzle?)
  - What objects are interactive? (keys, switches, doors, enemies)
  - Are there hazards? (things that cause GAME_OVER)
- **Late game**: Execute your plan efficiently. Minimize unnecessary moves.

### 3. Act (choose ONE action with a clear reason)

- Never act randomly. State your reason: "I move RIGHT because the exit is to the right."
- If unsure, choose the action that gives the **most information** (exploration).
- Avoid repeating an action that just produced no effect.

### 4. Learn (update your understanding)

- Did the action match your hypothesis? If yes, increase confidence.
- Did something unexpected happen? **Revise** your model immediately.
- Log what you learned: "ACTION3 = LEFT, confirmed 3 times."

## Action Mapping Discovery

In the first few steps of any new game:

| Priority | Action  | Test it by...                                                       |
| -------- | ------- | ------------------------------------------------------------------- |
| 1st      | ACTION1 | Take it, observe if player moves UP                                 |
| 2nd      | ACTION2 | Take it, observe if player moves DOWN                               |
| 3rd      | ACTION3 | Take it, observe if player moves LEFT                               |
| 4th      | ACTION4 | Take it, observe if player moves RIGHT                              |
| 5th      | ACTION5 | Usually confirm/interact/spacebar — test near an interactive object |
| 6th      | ACTION6 | Click at (x,y) — try clicking on interactive objects                |

Common mappings (but verify!):

- ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right, ACTION5=Confirm/Space
- Some games swap Up/Down or use different schemes. **Always verify.**

## Common Game Patterns

### Navigation Games (most common)

- **Goal**: Move player from start to exit/door.
- **Pattern**: Walls block movement (no-op on collision). Floor is traversable.
- **Strategy**: Identify player → find exit → plan path → execute.
- **Watch for**: Keys (must collect before door opens), energy/health (depletes over time), switches (open walls/doors).

### Puzzle Games

- **Goal**: Transform the grid to match a target pattern.
- **Pattern**: Actions manipulate objects (rotate, shift, swap colors).
- **Strategy**: Understand what each action does to the grid, then plan the sequence.

### Collection Games

- **Goal**: Collect all items or reach a score threshold.
- **Pattern**: Items disappear when player touches them. Score increases.
- **Strategy**: Plan an efficient path through all collectibles.

### Multi-level Games

- **Goal**: Complete all levels (score tracks current level).
- **Pattern**: Score jumps = level cleared. Grid resets with new layout.
- **Strategy**: What worked in level 1 may not work in level 2. Re-discover rules each level.

## Exploration vs Exploitation

| Phase          | Strategy                         | When                                                              |
| -------------- | -------------------------------- | ----------------------------------------------------------------- |
| **Explore**    | Try new actions, visit new areas | Steps 1-10, or after GAME_OVER retry                              |
| **Exploit**    | Use known rules to make progress | When action mappings are confirmed and goal is clear              |
| **Re-explore** | Try something different          | When stuck (3+ no-ops in a row, or no score change for 10+ steps) |

## Handling NO EFFECT (Wall Hits)

When an action produces no change:

1. You hit a wall or boundary. **Do not repeat** the same action.
2. Try a perpendicular direction (if blocked going RIGHT, try UP or DOWN).
3. If blocked in all 4 directions, you may be trapped — try ACTION5/ACTION6 to interact.
4. Track which directions are blocked from the current position to avoid re-testing.

## Recovery from GAME_OVER

On retry:

1. **Review past episodes** in the buffer context — what went wrong?
2. **Do NOT repeat the same sequence** that failed.
3. **Identify the failure point**: Was it energy depletion? Wrong path? Hazard?
4. **Try a fundamentally different approach**:
   - If you went left last time, try going right first.
   - If you skipped collectibles, get them this time.
   - If you ran out of energy, prioritize energy pickups.

## Efficiency Rules

1. **Never repeat a no-op**: If ACTION1 had no effect, don't try ACTION1 again from the same position.
2. **Don't zig-zag**: Moving UP then DOWN returns you to the start — wasted 2 actions.
3. **Plan before acting**: Think 3-5 steps ahead based on the grid layout.
4. **Score is progress**: If score hasn't changed in 10+ actions, reassess your strategy.
5. **Fast fail**: If a hypothesis is wrong, abandon it quickly and try something new.

## Reasoning Template

Use this structure for your observation each step:

```
1. What happened: [describe the effect of the last action]
2. Current state: [player position, score, nearby objects]
3. Hypothesis: [what I think the game wants me to do]
4. Plan: [next 2-3 actions and why]
5. Next action: [specific action and reasoning]
```

## Using update_game_notes

You have a tool called `update_game_notes` that writes knowledge to this file.
**Use it** whenever you confirm something important — this knowledge persists
across retries and helps you avoid repeating mistakes.

When to call it:

- After confirming an action mapping (e.g. ACTION1 = UP) — write to `action_mappings`
- After discovering a game rule (e.g. "touching color 6 refills energy") — write to `game_rules`
- After clearing a level or understanding the objective — write to `level_strategies`
- After identifying what an object does (e.g. "color 11 border = exit door") — write to `object_roles`
- After learning a useful trick — write to `tips`

Call `update_game_notes` BEFORE your game action in the same turn.
You can call it multiple times per turn if you have several discoveries.

## Discovered Knowledge (auto-updated by agent)

### Action Mappings

- ACTION1 = UP (moves 5 cells per step) - CONFIRMED
- ACTION2 = DOWN (moves 5 cells per step) - CONFIRMED
- ACTION3 = LEFT (moves 5 cells per step) - CONFIRMED
- ACTION4 = RIGHT (moves 5 cells per step) - CONFIRMED
- Movement is 5 cells per action in all directions

### Game Rules

- Color 11 bar at bottom (row 62) is energy/turn counter: resets each level to ~84 cells, decreases by 2 per action
- Player is a paired entity: color 9 (3x5 rect) + color 12 (2x5 rect), moving together
- Color 12 is on top of color 9 (offset ~2 rows up)
- Movement is 5 cells per action in all directions
- Navigating player INTO a target structure (bordered box) scores a point and advances to next level
- Multi-level game: completing a level loads a new grid with new layout, full energy reset
- Two grids shown after level completion: Grid 0 shows completed state, Grid 1 shows new level
- Score = number of levels completed

### Level Strategies

## Level 1 (Score 0→1): COMPLETED

- Had to collect trigger item at (32,21), which revealed two structures
- Then navigate player to the top structure at (12,36) - a bordered box with color 0 border, color 5 inner border
- Moving INTO the structure scored the point

## Level 2 (Score 1→?): IN PROGRESS

- Player starts at (43,31) on Grid 1
- Color 5 square (7x7) at (42,16) with color 9 border at (42,16) and point at (43,17) — possible target structure
- Color 11 objects at (17,16) and (52,31) — possibly triggers/collectibles to collect first
- Color 0 border at (47,51) — another collectible/trigger
- Color 1 objects near (47,51) — blue dots
- Energy full: 84 cells
- Strategy: First check if there's a trigger to collect (like level 1), then navigate to the target structure
- The color 5 structure at (42,16) is close — try going LEFT to reach it

### Object Roles

- Color 0/1 small objects at (32,21) were a collectible/trigger - collecting them revealed two new structures:
  - Bottom-left structure: color 0 border (10x10) at center (58,6), with color 5 and color 9 interior - looks like a maze pattern
  - Top structure: color 0 border (7x7) at center (12,36), with color 5 border (5x5) at (12,36)
- Color 9 (player) is 3x5 rect, color 12 is 2x5 rect - they move together as player entity
- Color 11 bar at bottom row 62 = energy (decreases 2 cells per action)
- Color 5 structures: walls/borders
- Color 3: green floor/path area
- Color 8: small markers at bottom-right (rows 62, cols 56/60/62) - possibly score indicators

### Tips

(no data yet)
