---
name: arc_game_playing
description: "Core strategy and reasoning framework for playing ARC-AGI-3 grid-based games. Covers the game loop (observe → hypothesize → act → learn), action semantics, common game patterns, exploration vs exploitation, and how to recover from failures. Apply at every step."
---

# ARC-AGI-3 Game Playing Strategy

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
- Check INTERACTION RULES for learned cause-and-effect patterns.
- Check OBJECT INVENTORY for targets and their distances.

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
- **Use navigate_to(row, col) for efficient pathfinding** when you know where to go.

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

## Using navigate_to

You have a tool called `navigate_to(row, col)` that uses BFS pathfinding to compute
the shortest path to any grid position, avoiding known walls automatically.

**When to use navigate_to:**
- Going to a specific object you see in the OBJECT INVENTORY
- Returning to a previously visited location
- Reaching the goal after collecting all prerequisites
- Any time you know WHERE you want to go but not the exact action sequence

**How it works:**
1. You call `navigate_to(row, col)` with the target position
2. The system computes a BFS path on the step-quantized grid (5 cells/step)
3. Known walls are avoided; unknown cells are treated as walkable (optimistic)
4. The path executes automatically, stopping on wall hit or score change
5. After execution, you receive a new observation

**Prefer navigate_to over execute_plan** for reaching objects — it handles wall avoidance for you.

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

## Interactive Object Discovery

When you see small clusters of colored cells that are **distinct from walls and floor**:

1. **They are likely interactive.** Move your player directly onto them to test.
2. **If an object disappears when you overlap it**, you collected/activated it. Record this interaction pattern.
3. **If a resource bar changes** (refills, depletes) after interacting, that object type affects your resources. Note which objects are beneficial.
4. **Scan the ENTIRE grid** for all objects of the same type before heading to the goal — there may be multiple.

## Prerequisite Detection

Many games require you to complete prerequisites before reaching the goal:

1. **If you reach a goal/target and CANNOT enter**, this means there are prerequisites you haven't fulfilled.
2. **Common prerequisites**: collect all items of a certain type, activate all switches, clear all enemies.
3. **Test**: After being blocked at the goal, search the grid for any remaining interactive objects you haven't visited.
4. **Rule of thumb**: If object type X disappears when you touch it, assume ALL instances of type X must be collected before the goal opens.

## Level Transition Protocol

**CRITICAL**: When you enter a new level (score increased, grid changed), follow this checklist IMMEDIATELY:

1. **Read the Level Summary** below in "Discovered Knowledge > Level Strategies" — it contains auto-generated insights from the previous level including what triggered completion.
2. **Scan for trigger/prerequisite objects**: Look in the OBJECT INVENTORY for objects matching the same colors that were prerequisites in previous levels. If Level 1 required collecting color X objects, Level 2 almost certainly requires the same.
3. **Scan for exit/goal structures**: Look for bordered boxes, doors, or special structures matching what you learned. The exit pattern is usually the same across levels.
4. **Plan: prerequisites FIRST, exit LAST**: Collect all trigger objects before navigating to the exit.
5. **Reuse interaction rules**: The rules from previous levels still apply (same game mechanics).
6. **Do NOT wander aimlessly**: You already know the game mechanics. Execute efficiently.

## Cross-Level Learning

In multi-level games, the same game mechanics usually persist across levels:

1. **If collecting object type X was required in Level 1, it will likely be required in Level 2.**
2. **On entering a new level**: immediately scan for the same types of interactive objects you discovered before.
3. **Apply the same strategy**: collect all prerequisites first, then head to the goal.
4. **Adapt to new layouts**: The positions change but the mechanics stay the same.
5. **Use update_game_notes** to record confirmed mechanics — this knowledge persists across levels and retries.

## Resource Management

If the game has a resource bar (energy, health, time):

1. **Track the depletion rate** — how much resource each action costs.
2. **Identify refill sources** — which objects restore the resource when collected.
3. **Plan routes through refill sources** to extend your effective range.
4. **If you run out of resource and get GAME_OVER**, prioritize visiting refill sources earlier on the next attempt.

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
6. **Use navigate_to**: When you know a target position, use pathfinding instead of manual navigation.

## Reasoning Template

Use this structure for your observation each step:

```
1. What happened: [describe the effect of the last action]
2. Current state: [player position, score, nearby objects]
3. Inventory check: [what does the OBJECT INVENTORY say? any suggested targets?]
4. Hypothesis: [what I think the game wants me to do]
5. Plan: [next 2-3 actions and why]
6. Next action: [specific action/navigate_to call and reasoning]
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

- Level 1: Collect trigger item first, then navigate to the target structure (bordered box)
- Pattern: collect prerequisites -> navigate to goal
- Same pattern expected in subsequent levels

### Object Roles

- Color 9 + color 12 = player entity (move together)
- Color 11 bar at bottom row 62 = energy (decreases 2 cells per action, resets on level completion)
- Color 5 structures = walls/borders
- Color 3 = green floor/path area
- Color 8 small markers at bottom-right = score indicators
- Small colored objects (color 0/1 clusters) = collectible triggers
- Bordered boxes (color 0 border + color 5 inner) = goal/target structures

### Tips

- Always collect ALL trigger/prerequisite objects before heading to the goal
- Use navigate_to(row, col) for efficient pathfinding to known object positions
- Check OBJECT INVENTORY for suggested next targets and distances
- Check INTERACTION RULES for learned cause-and-effect patterns
- Energy refill objects (if any) should be prioritized when energy is low
