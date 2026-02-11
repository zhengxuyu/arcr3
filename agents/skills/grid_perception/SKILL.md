---
name: grid_perception
description: "Understand and use structured grid perception data. The system provides object segmentation, movement tracking, player identification, action-effect learning, and VLM-based entity/change interpretation instead of raw grid numbers. Use this skill to read and reason about the perception sections in every observation."
---

# Grid Perception — how to read your observations

Your observations do NOT contain raw 64x64 number grids. Instead, a **perception engine** analyzes each frame and provides you with structured data. This skill explains every section you may see and how to reason with it.

## 1. Grid Overview

```
## Grid 0 (64x64, bg=color 4)
Non-bg colors: c10:812, c3:340, c8:200, ...
Objects detected: 23
```

- **bg=color N**: The background / floor color. Ignore it when reasoning about objects.
- **Non-bg colors**: How many cells of each non-background color exist. A sudden change in counts between frames means something moved or was collected.
- **Objects detected**: Total connected components found (4-connected BFS per color).

## 2. Object Lists

Objects are grouped by size:

- **Large structures (>100 cells)**: Walls, borders, large floor areas.
- **Medium objects (10-100 cells)**: Doors, keys, switches, the player, collectibles.
- **Small objects (<10 cells)**: Dots, markers, single-cell items.

Each entry looks like:

```
#5 color 3: 48 cells, rect (6x8) center=(31,22) [PLAYER?]
```

- `#5` = object ID (stable within one frame, may shift between frames).
- `color 3` = grid cell value (0-15).
- `48 cells` = pixel count.
- `rect` = shape class: point / hline / vline / rect / border / irregular.
- `(6x8)` = bounding box height x width.
- `center=(31,22)` = center row, col.
- `[PLAYER?]` = the system's best guess that this is the player.

**Tip**: Object IDs can change between frames because segmentation re-runs every step. Use color + position + size to track identity across frames, not the raw ID number.

## 3. Frame Changes (after an action)

```
### Frame Changes (after ACTION1):
  42 cells changed (1.0% of grid)
  MOVEMENT: 16-cell object (31,22) -> (30,22) = UP (dr=-1, dc=0)
```

- **cells changed**: Total grid cells that differ from the previous frame.
- **MOVEMENT**: A cluster of same-color cells disappeared from one location and appeared nearby. Includes:
  - Direction: UP / DOWN / LEFT / RIGHT (aligned with the UP arrow in the image).
  - `dr, dc`: Row displacement (negative = up), column displacement (negative = left).
- **cells appeared / disappeared**: New objects entering or old objects leaving the grid.
- **"NO EFFECT"**: The action did nothing — you probably hit a wall or the action is invalid.

**Key rule**: If the grid is unchanged after an action, do NOT repeat that action in the same direction. You are blocked.

## 4. Learned Action Effects

```
### Learned Action Effects:
  ACTION1 -> UP (8/10 times, 2 no-ops)
  ACTION2 -> DOWN (9/9 times)
  ACTION3 -> LEFT (7/8 times, 1 no-ops)
  ACTION4 -> RIGHT (6/6 times)
```

This is accumulated from all previous actions in this episode. Use it to:
- Know which action maps to which direction.
- Estimate reliability (no-ops = wall hits).
- If an action has never been observed, try it once to learn what it does.

## 5. Player Tracking

```
### Player Tracking:
  Identified: #5, color 3 (seen 12x), 16 cells, rect (4x4), center=(30,22)
```

The system tracks the player as the object that consistently moves when you take actions. The more observations, the more confident the identification.

## 6. Zoomed View

```
### Zoomed View (player area, rows 24-39, cols 16-31):
    6789012345678901
 24 4444444444444444
 25 444444AA44444444
 ...
```

A hex-encoded crop around the player or movement area. Use it to see fine details:
- `0-9` = colors 0-9
- `A-F` = colors 10-15

## 7. VLM Object Identification

```
### Object Identification (from VLM):
  Main walls [#0, #1]: wall — large purple border structures (confidence: high)
  Player [#5]: player — small green 4x4 square (confidence: high)
  Exit door [#8, #9]: door/exit — dark green border with teal interior (confidence: medium)
```

A vision model periodically examines the rendered image to identify what each object IS. Use these role labels (wall, player, door, key, etc.) when reasoning about strategy.

## 8. Action Interpretation (from VLM)

```
### Action Interpretation (from VLM):
  [#5] player_moved: Player moved UP by 1 cell, now closer to the exit door
  => ACTION1 moved the player one step upward toward the exit.
```

After each action, the VLM interprets the algorithmic movement data in game terms. Use the `=>` summary line as a quick reference for what just happened.

## Strategy Guidelines

1. **Use movement data, not raw grids**: The perception tells you exactly what moved and where. Plan paths based on object positions and learned action mappings.
2. **Track the player**: The `[PLAYER?]` tag and Player Tracking section tell you where you are. Combine with object positions to navigate.
3. **Learn from no-ops**: If an action produces "NO EFFECT", you hit a wall or boundary. Try a different direction.
4. **Watch for score changes**: Score increases mean progress. Replicate the action sequence that led to score gains.
5. **Use VLM roles**: If the VLM identifies a "door" or "key", plan to interact with it. Doors usually need keys; switches change state.
6. **Avoid repeating failures**: If you've tried the same action 2-3 times with no effect, switch strategy entirely.
7. **Direction reference**: The rendered image has a white UP arrow in the top-right corner. UP in the image = UP in the grid = row index decreasing = ACTION1 (typically).
