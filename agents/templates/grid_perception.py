"""
Grid Perception Module for ARC-AGI-3.

Analyzes 64x64 grids (values 0-15) to extract structured information that
an LLM can actually reason about, replacing the raw number dump with:

1. **Object Segmentation**: Connected-component analysis with shape classification.
2. **Background Detection**: Identifies the dominant "floor" color.
3. **Movement Tracking**: Detects what moved between frames and in which direction.
4. **Player Identification**: Heuristic tracking of the controllable object.
5. **Action-Effect Learning**: Maps GameActions to observed movement directions.
6. **Zoomed Views**: Compact hex-encoded crops around regions of interest.

The key insight: ARC-AGI-3 games present grids where VLMs and simple connected-
component methods fail because objects can be multi-colored, small, or spatially
complex.  This module provides the LLM with a *semantic description* of the grid
instead of (or in addition to) raw numbers, dramatically improving reasoning.
"""

from __future__ import annotations  # noqa: I001

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Color palette for hex display and image rendering
# ============================================================================
HEX_CHARS = "0123456789ABCDEF"

# ARC-AGI color palette (RGB)
ARC_PALETTE = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),     # 1: blue
    (255, 65, 54),     # 2: red
    (46, 204, 64),     # 3: green
    (255, 220, 0),     # 4: yellow
    (170, 170, 170),   # 5: gray
    (240, 18, 190),    # 6: magenta
    (255, 133, 27),    # 7: orange
    (127, 219, 255),   # 8: light blue
    (135, 12, 37),     # 9: dark red
    (128, 0, 128),     # 10: purple
    (0, 128, 0),       # 11: dark green
    (0, 128, 128),     # 12: teal
    (128, 128, 0),     # 13: olive
    (192, 192, 192),   # 14: silver
    (255, 255, 255),   # 15: white
]

# Distinct bbox colors so overlapping boxes are distinguishable
BBOX_COLORS = [
    (255, 0, 0),       # red
    (0, 255, 0),       # green
    (0, 100, 255),     # blue
    (255, 255, 0),     # yellow
    (255, 0, 255),     # magenta
    (0, 255, 255),     # cyan
    (255, 128, 0),     # orange
    (128, 255, 0),     # lime
    (255, 0, 128),     # pink
    (0, 128, 255),     # sky
    (200, 200, 200),   # light gray
    (255, 180, 180),   # salmon
]


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class GridObject:
    """A detected object (connected component) in the grid."""

    obj_id: int
    color: int
    cells: list[tuple[int, int]]  # (row, col)
    bbox: tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    size: int
    width: int
    height: int
    center_r: float
    center_c: float
    shape: str  # "point", "hline", "vline", "rect", "border", "L", "irregular"
    fill_ratio: float  # cells / bbox_area
    is_at_edge: bool  # touches grid boundary


@dataclass
class MovedObject:
    """A detected movement between two consecutive frames."""

    colors: set[int]
    old_center: tuple[float, float]  # (row, col)
    new_center: tuple[float, float]
    dr: int  # row displacement (positive = down)
    dc: int  # col displacement (positive = right)
    size: int
    direction: str  # "UP", "DOWN", "LEFT", "RIGHT", "NONE", "DIAGONAL"


@dataclass
class GridAnalysis:
    """Full analysis result for one grid."""

    height: int
    width: int
    bg_color: int
    objects: list[GridObject]
    color_counts: dict[int, int]  # color -> total cell count

    @property
    def num_objects(self) -> int:
        return len(self.objects)


@dataclass
class FrameChange:
    """What changed between two consecutive grids."""

    total_changed: int
    change_ratio: float
    moved: list[MovedObject]
    num_appeared: int
    num_disappeared: int


# ============================================================================
# Main perception class
# ============================================================================


class GridPerception:
    """Perception engine for ARC-AGI-3 grids.

    Maintains state across frames to track the player and learn action effects.
    Create one instance per game (agent lifetime).
    """

    def __init__(self) -> None:
        # Player tracking
        self._player_obj: Optional[GridObject] = None
        self._player_color_votes: Counter = Counter()

        # Action-effect learning: action_name -> list of (dr, dc)
        self._action_effects: dict[str, list[tuple[int, int]]] = {}

    # ------------------------------------------------------------------ #
    #  Core analysis
    # ------------------------------------------------------------------ #

    def analyze_grid(self, grid: list[list[int]]) -> GridAnalysis:
        """Segment a single grid into objects with properties."""
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0

        # Color histogram
        color_counts: dict[int, int] = {}
        for row in grid:
            for v in row:
                color_counts[v] = color_counts.get(v, 0) + 1

        # Background = most common color
        bg = max(color_counts, key=color_counts.get) if color_counts else 0

        # Connected components via BFS (4-connected, per color)
        visited = [[False] * W for _ in range(H)]
        objects: list[GridObject] = []
        obj_id = 0

        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg and not visited[r][c]:
                    color = grid[r][c]
                    cells = self._bfs(grid, visited, r, c, color, H, W)
                    obj = self._build_object(obj_id, color, cells, H, W)
                    objects.append(obj)
                    obj_id += 1

        # Sort by size descending for easier reading
        objects.sort(key=lambda o: o.size, reverse=True)
        for i, obj in enumerate(objects):
            obj.obj_id = i

        return GridAnalysis(
            height=H, width=W, bg_color=bg,
            objects=objects, color_counts=color_counts,
        )

    def compute_diff(
        self, prev_grid: list[list[int]], curr_grid: list[list[int]]
    ) -> FrameChange:
        """Detect changes between two grids, including movement.

        Uses **per-color** tracking: for each non-background color, find
        cells where that color disappeared and where it appeared.  This
        catches objects moving over *any* surface, not just over the
        background color.
        """
        H = len(prev_grid)
        W = len(prev_grid[0]) if H > 0 else 0

        prev_bg = self._quick_bg(prev_grid, H, W)
        curr_bg = self._quick_bg(curr_grid, H, W)
        bg_colors = {prev_bg, curr_bg}

        # Per-color: cells where a color vanished / appeared
        # color -> list of (r, c)
        color_lost: dict[int, list[tuple[int, int]]] = {}
        color_gained: dict[int, list[tuple[int, int]]] = {}

        changed_total = 0
        num_appeared = 0   # bg -> non-bg (any surface)
        num_disappeared = 0  # non-bg -> bg

        for r in range(H):
            for c in range(W):
                pv, cv = prev_grid[r][c], curr_grid[r][c]
                if pv == cv:
                    continue
                changed_total += 1

                # Track per-color disappearance / appearance
                if pv not in bg_colors:
                    color_lost.setdefault(pv, []).append((r, c))
                if cv not in bg_colors:
                    color_gained.setdefault(cv, []).append((r, c))

                # Legacy counters
                if pv not in bg_colors and cv in bg_colors:
                    num_disappeared += 1
                elif pv in bg_colors and cv not in bg_colors:
                    num_appeared += 1

        total_cells = H * W
        ratio = changed_total / total_cells if total_cells > 0 else 0.0

        # Detect movements per color, then merge
        moved = self._detect_movements_by_color(color_lost, color_gained)

        return FrameChange(
            total_changed=changed_total,
            change_ratio=ratio,
            moved=moved,
            num_appeared=num_appeared,
            num_disappeared=num_disappeared,
        )

    # ------------------------------------------------------------------ #
    #  High-level description for LLM
    # ------------------------------------------------------------------ #

    def describe_frame(
        self,
        frame: list[list[list[int]]],
        prev_frame: Optional[list[list[list[int]]]] = None,
        last_action: Optional[str] = None,
    ) -> str:
        """Produce a structured LLM-readable description of the frame.

        This REPLACES the raw grid dump with a much more informative
        semantic analysis.

        Args:
            frame: List of grids (from FrameData.frame).
            prev_frame: Previous frame's grids (for diff).
            last_action: The action name that produced this frame.

        Returns:
            Multi-line structured text description.
        """
        parts: list[str] = []

        for g_idx, grid in enumerate(frame):
            analysis = self.analyze_grid(grid)
            parts.append(self._format_analysis(analysis, g_idx))

            # Frame diff
            if prev_frame and g_idx < len(prev_frame):
                change = self.compute_diff(prev_frame[g_idx], grid)
                if change.total_changed > 0:
                    diff_text = self._format_diff(change, last_action)
                    parts.append(diff_text)

                    # Learn action effects
                    if last_action and change.moved:
                        self._learn_action_effect(last_action, change.moved)

                    # Update player tracking
                    if change.moved:
                        self._update_player_tracking(change.moved, analysis)

                    # Zoomed view around changed area
                    zoom = self._zoomed_view(grid, change)
                    if zoom:
                        parts.append(zoom)
                else:
                    parts.append(
                        f"[Grid {g_idx} unchanged — last action had NO EFFECT]"
                    )

        # Action-effect summary
        effects = self._format_action_effects()
        if effects:
            parts.append(effects)

        # Player info
        player_info = self._format_player_info()
        if player_info:
            parts.append(player_info)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  BFS and object building
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bfs(
        grid: list[list[int]],
        visited: list[list[bool]],
        start_r: int,
        start_c: int,
        color: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int]]:
        """4-connected flood fill for a single color."""
        cells: list[tuple[int, int]] = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return cells

    @staticmethod
    def _build_object(
        obj_id: int,
        color: int,
        cells: list[tuple[int, int]],
        grid_h: int,
        grid_w: int,
    ) -> GridObject:
        """Build a GridObject from a set of cells."""
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        w = c_max - c_min + 1
        h = r_max - r_min + 1
        bbox_area = w * h
        fill = len(cells) / bbox_area if bbox_area > 0 else 0.0
        size = len(cells)

        # Shape classification
        if size == 1:
            shape = "point"
        elif h == 1 and w > 1:
            shape = "hline"
        elif w == 1 and h > 1:
            shape = "vline"
        elif fill > 0.85:
            shape = "rect"
        elif 0.3 < fill <= 0.85:
            border_cells = sum(
                1
                for r, c in cells
                if r in (r_min, r_max) or c in (c_min, c_max)
            )
            if border_cells / size > 0.75:
                shape = "border"
            else:
                shape = "irregular"
        else:
            shape = "irregular"

        is_at_edge = r_min == 0 or c_min == 0 or r_max >= grid_h - 1 or c_max >= grid_w - 1

        return GridObject(
            obj_id=obj_id,
            color=color,
            cells=cells,
            bbox=(r_min, c_min, r_max, c_max),
            size=size,
            width=w,
            height=h,
            center_r=sum(rs) / len(rs),
            center_c=sum(cs) / len(cs),
            shape=shape,
            fill_ratio=fill,
            is_at_edge=is_at_edge,
        )

    # ------------------------------------------------------------------ #
    #  Movement detection
    # ------------------------------------------------------------------ #

    def _detect_movements_by_color(
        self,
        color_lost: dict[int, list[tuple[int, int]]],
        color_gained: dict[int, list[tuple[int, int]]],
    ) -> list[MovedObject]:
        """Match per-color lost/gained cell clusters as object movements.

        For each color that has both lost and gained cells, cluster each
        set and match lost↔gained clusters by proximity and size to
        identify movement.
        """
        moved: list[MovedObject] = []

        # Process each color that appears in both lost and gained
        all_colors = set(color_lost.keys()) & set(color_gained.keys())
        for color in all_colors:
            lost_cells = color_lost[color]
            gained_cells = color_gained[color]

            lost_clusters = self._cluster_cells(lost_cells)
            gained_clusters = self._cluster_cells(gained_cells)

            used_gained: set[int] = set()

            for lc in lost_clusters:
                lc_center = self._centroid(lc)
                lc_size = len(lc)

                best_dist = float("inf")
                best_idx = -1

                for i, gc in enumerate(gained_clusters):
                    if i in used_gained:
                        continue
                    gc_size = len(gc)
                    if gc_size == 0:
                        continue
                    size_ratio = min(lc_size, gc_size) / max(lc_size, gc_size)
                    if size_ratio < 0.4:
                        continue
                    gc_center = self._centroid(gc)
                    dist = abs(gc_center[0] - lc_center[0]) + abs(
                        gc_center[1] - lc_center[1]
                    )
                    if dist < best_dist and dist < 20:
                        best_dist = dist
                        best_idx = i

                if best_idx >= 0:
                    used_gained.add(best_idx)
                    gc = gained_clusters[best_idx]
                    gc_center = self._centroid(gc)

                    dr = round(gc_center[0] - lc_center[0])
                    dc = round(gc_center[1] - lc_center[1])

                    if dr == 0 and dc == 0:
                        continue  # reshape, not movement

                    direction = self._direction_name(dr, dc)

                    moved.append(
                        MovedObject(
                            colors={color},
                            old_center=lc_center,
                            new_center=gc_center,
                            dr=dr,
                            dc=dc,
                            size=len(gc),
                            direction=direction,
                        )
                    )

        return moved

    @staticmethod
    def _cluster_cells(
        cells: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Cluster cells into groups of 8-connected neighbours."""
        if not cells:
            return []

        cell_set = set(cells)
        visited: set[tuple[int, int]] = set()
        clusters: list[list[tuple[int, int]]] = []

        for cell in cells:
            if cell in visited:
                continue
            cluster: list[tuple[int, int]] = []
            queue = deque([cell])
            visited.add(cell)
            while queue:
                r, c = queue.popleft()
                cluster.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r + dr, c + dc)
                        if nb in cell_set and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
            clusters.append(cluster)

        return clusters

    @staticmethod
    def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
        if not cells:
            return (0.0, 0.0)
        r = sum(c[0] for c in cells) / len(cells)
        c_ = sum(c[1] for c in cells) / len(cells)
        return (r, c_)

    @staticmethod
    def _direction_name(dr: int, dc: int) -> str:
        if dr == 0 and dc == 0:
            return "NONE"
        if abs(dr) > 0 and abs(dc) > 0:
            parts = []
            parts.append("UP" if dr < 0 else "DOWN")
            parts.append("LEFT" if dc < 0 else "RIGHT")
            return "-".join(parts)
        if dr < 0:
            return "UP"
        if dr > 0:
            return "DOWN"
        if dc < 0:
            return "LEFT"
        return "RIGHT"

    # ------------------------------------------------------------------ #
    #  Player tracking
    # ------------------------------------------------------------------ #

    def _update_player_tracking(
        self, movements: list[MovedObject], analysis: GridAnalysis
    ) -> None:
        """Update player identification based on observed movement."""
        if not movements:
            return

        # The player is the object that moves in response to actions.
        # Heuristic: the smallest moving cluster is most likely the player.
        smallest = min(movements, key=lambda m: m.size)

        # Find matching objects in the current analysis
        player_obj = None
        for obj in analysis.objects:
            dist = (
                abs(obj.center_r - smallest.new_center[0])
                + abs(obj.center_c - smallest.new_center[1])
            )
            if dist < 3:
                player_obj = obj
                self._player_obj = obj
                self._player_color_votes[obj.color] += 1
                break

        # Also vote for nearby objects that are likely part of the same
        # multi-color player entity (e.g. body=color9, head=color12).
        # Only consider objects whose color also moved this step, to avoid
        # voting for static game elements (patterns, decorations) near the
        # player.
        if player_obj is not None:
            moving_colors: set[int] = set()
            for m in movements:
                moving_colors.update(m.colors)
            for obj in analysis.objects:
                if obj is player_obj:
                    continue
                if obj.size >= 50:
                    continue
                if obj.color not in moving_colors:
                    continue
                ndist = (
                    abs(obj.center_r - player_obj.center_r)
                    + abs(obj.center_c - player_obj.center_c)
                )
                if ndist <= 5:  # within one step
                    self._player_color_votes[obj.color] += 1

    def _learn_action_effect(
        self, action: str, movements: list[MovedObject]
    ) -> None:
        """Record the observed displacement caused by an action."""
        if not movements:
            return
        # Use the smallest/player-like movement
        primary = min(movements, key=lambda m: m.size)
        if action not in self._action_effects:
            self._action_effects[action] = []
        self._action_effects[action].append((primary.dr, primary.dc))

    # ------------------------------------------------------------------ #
    #  Formatting helpers
    # ------------------------------------------------------------------ #

    def _format_analysis(self, analysis: GridAnalysis, grid_idx: int) -> str:
        """Format a GridAnalysis into LLM-readable text."""
        lines: list[str] = []
        lines.append(
            f"## Grid {grid_idx} ({analysis.height}x{analysis.width}, "
            f"bg=color {analysis.bg_color})"
        )

        # Color summary
        non_bg = {
            c: n
            for c, n in analysis.color_counts.items()
            if c != analysis.bg_color and n > 0
        }
        if non_bg:
            color_str = ", ".join(
                f"c{c}:{n}" for c, n in sorted(non_bg.items(), key=lambda x: -x[1])
            )
            lines.append(f"Non-bg colors: {color_str}")

        lines.append(f"Objects detected: {analysis.num_objects}")

        if not analysis.objects:
            return "\n".join(lines)

        # Group objects by role heuristic
        large: list[GridObject] = []  # > 100 cells — likely walls/borders
        medium: list[GridObject] = []  # 10-100
        small: list[GridObject] = []  # < 10

        for obj in analysis.objects:
            if obj.size > 100:
                large.append(obj)
            elif obj.size >= 10:
                medium.append(obj)
            else:
                small.append(obj)

        if large:
            lines.append("\nLarge structures (likely walls/borders):")
            for obj in large[:5]:
                player_tag = " [PLAYER?]" if obj is self._player_obj else ""
                lines.append(
                    f"  #{obj.obj_id} color {obj.color}: "
                    f"{obj.size} cells, {obj.shape} "
                    f"({obj.height}x{obj.width}) "
                    f"at ({obj.bbox[0]},{obj.bbox[1]})-"
                    f"({obj.bbox[2]},{obj.bbox[3]})"
                    f"{player_tag}"
                )

        if medium:
            lines.append("\nMedium objects:")
            for obj in medium[:8]:
                player_tag = " [PLAYER?]" if obj is self._player_obj else ""
                lines.append(
                    f"  #{obj.obj_id} color {obj.color}: "
                    f"{obj.size} cells, {obj.shape} "
                    f"({obj.height}x{obj.width}) "
                    f"center=({obj.center_r:.0f},{obj.center_c:.0f})"
                    f"{player_tag}"
                )

        if small:
            lines.append(f"\nSmall objects ({len(small)} total):")
            for obj in small[:10]:
                player_tag = " [PLAYER?]" if obj is self._player_obj else ""
                lines.append(
                    f"  #{obj.obj_id} color {obj.color}: "
                    f"{obj.size} cells, {obj.shape} "
                    f"at ({obj.center_r:.0f},{obj.center_c:.0f})"
                    f"{player_tag}"
                )
            if len(small) > 10:
                lines.append(f"  ... and {len(small) - 10} more small objects")

        # UI region detection: bottom 3 rows often have score/status
        bottom_objs = [
            obj
            for obj in analysis.objects
            if obj.bbox[2] >= analysis.height - 3
        ]
        if bottom_objs:
            lines.append(
                f"\nBottom-edge objects (possible UI): "
                f"{len(bottom_objs)} objects in last 3 rows"
            )

        return "\n".join(lines)

    def _format_diff(self, change: FrameChange, last_action: Optional[str]) -> str:
        """Format a FrameChange into LLM-readable text."""
        lines: list[str] = []
        pct = change.change_ratio * 100

        act_str = f" (after {last_action})" if last_action else ""
        lines.append(f"### Frame Changes{act_str}:")
        lines.append(
            f"  {change.total_changed} cells changed ({pct:.1f}% of grid)"
        )

        if change.moved:
            for m in change.moved:
                lines.append(
                    f"  MOVEMENT: {m.size}-cell object "
                    f"({m.old_center[0]:.0f},{m.old_center[1]:.0f}) -> "
                    f"({m.new_center[0]:.0f},{m.new_center[1]:.0f}) "
                    f"= {m.direction} (dr={m.dr}, dc={m.dc})"
                )

        if change.num_appeared > 0 and not change.moved:
            lines.append(f"  {change.num_appeared} cells appeared (new object?)")
        if change.num_disappeared > 0 and not change.moved:
            lines.append(
                f"  {change.num_disappeared} cells disappeared (object removed?)"
            )

        return "\n".join(lines)

    def _format_action_effects(self) -> str:
        """Summarize learned action-effect mappings."""
        if not self._action_effects:
            return ""

        lines: list[str] = ["### Learned Action Effects:"]
        for action, effects in sorted(self._action_effects.items()):
            if not effects:
                continue
            # Most common direction
            dir_counts: dict[str, int] = {}
            for dr, dc in effects:
                d = self._direction_name(dr, dc)
                dir_counts[d] = dir_counts.get(d, 0) + 1

            top_dir = max(dir_counts, key=dir_counts.get)
            count = dir_counts[top_dir]
            total = len(effects)
            noop = sum(1 for dr, dc in effects if dr == 0 and dc == 0)

            if noop == total:
                lines.append(f"  {action} -> NO MOVEMENT ({total} observations)")
            else:
                lines.append(
                    f"  {action} -> {top_dir} "
                    f"({count}/{total} times"
                    f"{f', {noop} no-ops' if noop else ''})"
                )

        return "\n".join(lines)

    def _format_player_info(self) -> str:
        """Summary of current player identification."""
        if self._player_obj is None:
            return ""

        obj = self._player_obj
        most_common = self._player_color_votes.most_common(1)
        color_info = (
            f"color {most_common[0][0]} (seen {most_common[0][1]}x)"
            if most_common
            else f"color {obj.color}"
        )

        return (
            f"### Player Tracking:\n"
            f"  Identified: #{obj.obj_id}, {color_info}, "
            f"{obj.size} cells, {obj.shape} "
            f"({obj.height}x{obj.width}), "
            f"center=({obj.center_r:.0f},{obj.center_c:.0f})"
        )

    # ------------------------------------------------------------------ #
    #  Zoomed view
    # ------------------------------------------------------------------ #

    def _zoomed_view(
        self,
        grid: list[list[int]],
        change: FrameChange,
        max_size: int = 16,
    ) -> str:
        """Produce a compact hex-encoded crop around the area of interest.

        Focuses on the player or the changed region.
        """
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0

        # Determine center of interest
        if self._player_obj is not None:
            cr = int(self._player_obj.center_r)
            cc = int(self._player_obj.center_c)
            label = "player area"
        elif change.moved:
            m = change.moved[0]
            cr = int(m.new_center[0])
            cc = int(m.new_center[1])
            label = "movement area"
        else:
            return ""

        # Compute crop bounds
        half = max_size // 2
        r_min = max(0, cr - half)
        r_max = min(H, cr + half)
        c_min = max(0, cc - half)
        c_max = min(W, cc + half)

        lines: list[str] = [
            f"### Zoomed View ({label}, "
            f"rows {r_min}-{r_max - 1}, cols {c_min}-{c_max - 1}):"
        ]

        # Column header
        col_header = "    " + "".join(
            str(c % 10) for c in range(c_min, c_max)
        )
        lines.append(col_header)

        for r in range(r_min, r_max):
            row_str = "".join(
                HEX_CHARS[grid[r][c]] for c in range(c_min, c_max)
            )
            lines.append(f" {r:2d} {row_str}")

        lines.append(
            "(hex: 0-9=colors 0-9, A=10, B=11, C=12, D=13, E=14, F=15)"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quick_bg(grid: list[list[int]], H: int, W: int) -> int:
        """Fast background detection via sampling."""
        counts: dict[int, int] = {}
        # Sample corners + edges + center for speed
        samples = []
        for r in (0, H // 4, H // 2, 3 * H // 4, H - 1):
            for c in (0, W // 4, W // 2, 3 * W // 4, W - 1):
                if 0 <= r < H and 0 <= c < W:
                    samples.append(grid[r][c])
        # Also sample full first/last rows
        if H > 0:
            samples.extend(grid[0])
            samples.extend(grid[H - 1])

        for v in samples:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get) if counts else 0

    def reset(self) -> None:
        """Reset perception state for a new game/episode."""
        self._player_obj = None
        self._player_color_votes.clear()
        self._action_effects.clear()

    # ------------------------------------------------------------------ #
    #  Image rendering with bounding boxes
    # ------------------------------------------------------------------ #

    def render_grid(
        self,
        grid: list[list[int]],
        analysis: Optional[GridAnalysis] = None,
        scale: int = 8,
        show_labels: bool = True,
        max_boxes: int = 30,
    ):
        """Render a grid as a PIL Image with detected object bounding boxes.

        Args:
            grid: 2-D list of color values (0-15).
            analysis: Pre-computed GridAnalysis (computed on the fly if None).
            scale: Pixel size per grid cell.
            show_labels: Draw object id / color labels next to boxes.
            max_boxes: Maximum bounding boxes to draw (largest first).

        Returns:
            ``PIL.Image.Image`` or *None* if Pillow is not installed.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None

        if analysis is None:
            analysis = self.analyze_grid(grid)

        H, W = analysis.height, analysis.width
        img = Image.new("RGB", (W * scale, H * scale))
        pixels = img.load()

        # Paint grid cells
        for r in range(H):
            for c in range(W):
                color = ARC_PALETTE[min(grid[r][c], 15)]
                for dy in range(scale):
                    for dx in range(scale):
                        pixels[c * scale + dx, r * scale + dy] = color

        draw = ImageDraw.Draw(img)

        # Try to get a small font for labels
        font = None
        if show_labels:
            for font_path in (
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/SFNSMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ):
                try:
                    font = ImageFont.truetype(font_path, max(scale, 10))
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

        # Draw bounding boxes
        objects_to_draw = analysis.objects[:max_boxes]
        for idx, obj in enumerate(objects_to_draw):
            bbox_color = BBOX_COLORS[idx % len(BBOX_COLORS)]

            r_min, c_min, r_max, c_max = obj.bbox
            x0 = c_min * scale
            y0 = r_min * scale
            x1 = (c_max + 1) * scale - 1
            y1 = (r_max + 1) * scale - 1

            # Draw 2-pixel wide rectangle
            draw.rectangle([x0, y0, x1, y1], outline=bbox_color, width=2)

            # Player gets a thicker box
            if obj is self._player_obj:
                draw.rectangle(
                    [x0 - 1, y0 - 1, x1 + 1, y1 + 1],
                    outline=(255, 255, 255),
                    width=1,
                )

            # Label
            if show_labels and font is not None:
                label = f"#{obj.obj_id} c{obj.color}"
                if obj is self._player_obj:
                    label += " P"
                # Position label above the box, or below if near top edge
                lx = x0
                ly = y0 - scale - 2
                if ly < 0:
                    ly = y1 + 2
                draw.text((lx, ly), label, fill=bbox_color, font=font)

        # -- Direction indicator: UP arrow + "UP" label in top-right corner --
        img_w, img_h = img.size
        arrow_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        margin = 4
        # Arrow dimensions
        arrow_h = min(5 * scale, img_h // 6)
        arrow_w = arrow_h // 2
        cx = img_w - margin - arrow_w  # center x of arrow
        top_y = margin
        bot_y = top_y + arrow_h
        # Draw outline first (black), then white on top for visibility
        for color_pass, width_pass in ((outline_color, 4), (arrow_color, 2)):
            # Shaft
            draw.line(
                [(cx, top_y + arrow_w), (cx, bot_y)],
                fill=color_pass, width=width_pass,
            )
            # Arrowhead (triangle)
            draw.polygon(
                [
                    (cx, top_y),                        # tip
                    (cx - arrow_w // 2, top_y + arrow_w),  # bottom-left
                    (cx + arrow_w // 2, top_y + arrow_w),  # bottom-right
                ],
                fill=color_pass,
            )
        # "UP" text label
        if font is not None:
            label_x = cx - arrow_w
            label_y = bot_y + 2
            draw.text((label_x + 1, label_y + 1), "UP", fill=outline_color, font=font)
            draw.text((label_x, label_y), "UP", fill=arrow_color, font=font)

        return img

    def render_frame(
        self,
        frame: list[list[list[int]]],
        scale: int = 8,
        show_labels: bool = True,
        max_boxes: int = 30,
    ):
        """Render all grids in a frame side-by-side with bounding boxes.

        Returns:
            ``PIL.Image.Image`` or *None* if Pillow is not installed.
        """
        try:
            from PIL import Image
        except ImportError:
            return None

        images = []
        for grid in frame:
            analysis = self.analyze_grid(grid)
            img = self.render_grid(
                grid, analysis=analysis, scale=scale,
                show_labels=show_labels, max_boxes=max_boxes,
            )
            if img is not None:
                images.append(img)

        if not images:
            return None
        if len(images) == 1:
            return images[0]

        # Side-by-side with 4px gap
        gap = 4
        total_w = sum(im.width for im in images) + gap * (len(images) - 1)
        max_h = max(im.height for im in images)
        combined = Image.new("RGB", (total_w, max_h), (40, 40, 40))
        x_offset = 0
        for im in images:
            combined.paste(im, (x_offset, 0))
            x_offset += im.width + gap

        return combined


# ============================================================================
# VLM-based Object Annotator
# ============================================================================

# --- Prompt: identify objects (sent periodically) ---
_IDENTIFY_SYSTEM = """\
You are a game-screen analyst for ARC-AGI-3 grid-based games.
You will receive an image of a 64x64 grid with colored bounding boxes
around detected objects. Each box is labeled "#id cN" (object index, color).

ORIENTATION: There is a white UP arrow with "UP" label in the top-right
corner of the image. Use it to determine directions — UP in the image
means UP in the game (row index decreasing).

Your job:
1. Group objects that visually form a SINGLE logical entity
   (e.g. a door = colored border + different-colored interior).
2. For each entity guess its most likely ROLE.
   Common roles: wall, floor/path, player, door/exit, key, collectible,
   switch/button, score_display, health/energy, obstacle, decoration, unknown.
3. Briefly describe the entity's visual APPEARANCE (shape, pattern, color).
4. State confidence: high / medium / low.

Respond ONLY with valid JSON (no markdown fences):
{
  "entities": [
    {
      "name": "<short name>",
      "object_ids": [<int>, ...],
      "role": "<one-line role description>",
      "appearance": "<brief visual description>",
      "confidence": "high|medium|low"
    }
  ]
}"""

_IDENTIFY_USER = """\
Detected objects:
{object_table}

Look at the image, group related segments into logical entities, and describe each.
Return JSON only."""

# --- Prompt: interpret changes (sent every action) ---
_CHANGE_SYSTEM = """\
You are a game-screen interpreter for ARC-AGI-3 grid-based games.
You will receive:
1. An image of the CURRENT frame with colored bounding boxes on objects.
   ORIENTATION: A white UP arrow with "UP" label is in the top-right
   corner — use it to confirm directions (UP = toward top of image).
2. A list of detected objects (id, color, size, position).
3. PRECISE algorithmic movement data showing exactly what moved, in which
   direction, and by how many cells. Trust these directions — they are
   computed from grid coordinates and aligned with the UP arrow.

The action taken was: {action}

Your job is NOT to detect changes (the algorithm already did that).
Your job is to INTERPRET what the changes MEAN in game terms:
- Which named entity (player, key, door, etc.) was affected?
- What is the game significance of this movement?
- Did the player approach / move away from any important object?
- Did anything appear, disappear, or get collected?
- Is the action useful or was it wasted (e.g. bumped into a wall)?

Respond ONLY with valid JSON (no markdown fences):
{{
  "interpretations": [
    {{
      "object_ids": [<int>, ...],
      "description": "<game-level interpretation of this change>",
      "change_type": "player_moved|object_moved|collected|door_opened|blocked|environment_shift|ui_update|unknown"
    }}
  ],
  "action_effect": "<one sentence: what {action} did in game terms>",
  "nothing_changed": <true|false>
}}"""

_CHANGE_USER = """\
Object table (current frame):
{object_table}

Algorithmic change detection (precise, trust these numbers):
{diff_details}

Based on the image and the algorithmic data above, interpret what these changes mean.
Return JSON only."""


@dataclass
class EntityAnnotation:
    """VLM-produced annotation for a logical game entity."""

    name: str
    object_ids: list[int]
    role: str
    appearance: str
    confidence: str  # "high" | "medium" | "low"


@dataclass
class ChangeAnnotation:
    """VLM-produced description of what changed between two frames."""

    object_ids: list[int]
    description: str
    change_type: str  # moved|appeared|disappeared|color_changed|shape_changed|no_change


@dataclass
class FrameAnnotation:
    """Combined result from the VLM annotator for one step."""

    entities: list[EntityAnnotation]
    changes: list[ChangeAnnotation]
    action_effect: str  # one-sentence summary of the action's effect
    nothing_changed: bool


class ObjectAnnotator:
    """Calls a vision LLM to identify objects and detect changes.

    Two modes:
    - ``annotate()``: Identify objects and their roles (periodic).
    - ``annotate_change()``: Compare before/after and describe changes (per action).

    Usage::

        annotator = ObjectAnnotator()
        # Periodic: identify objects
        entities = annotator.annotate(bbox_image, analysis)
        # Per-action: detect changes
        frame_ann = annotator.annotate_change(
            prev_bbox_image, curr_bbox_image, analysis, diff_summary, action
        )
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        import os

        import openai

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self._model = model or os.environ.get("VLM_MODEL", "") or "gpt-4o-mini"

    # ------------------------------------------------------------------ #
    #  Object identification (periodic)
    # ------------------------------------------------------------------ #

    def annotate(
        self,
        bbox_image,
        analysis: GridAnalysis,
    ) -> list[EntityAnnotation]:
        """Send the bbox image + object list to VLM and parse entity annotations.

        Returns:
            List of ``EntityAnnotation`` or empty list on failure.
        """
        if bbox_image is None:
            return []

        data_url = self._encode_image(bbox_image)
        object_table = self._build_object_table(analysis)
        user_text = _IDENTIFY_USER.format(object_table=object_table)

        messages = [
            {"role": "system", "content": _IDENTIFY_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        raw = self._call_llm(messages)
        if raw is None:
            return []
        return self._parse_entities(raw)

    # ------------------------------------------------------------------ #
    #  Change detection (per action)
    # ------------------------------------------------------------------ #

    def annotate_change(
        self,
        curr_bbox_image,
        analysis: GridAnalysis,
        change: FrameChange,
        action: str,
        prev_bbox_image=None,
    ) -> Optional[FrameAnnotation]:
        """Interpret algorithmic change data using the VLM.

        Instead of asking the VLM to *detect* changes (it can't do that
        reliably), we feed it the **precise algorithmic diff results** and
        ask it to *interpret* what they mean in game terms.

        Args:
            curr_bbox_image: PIL Image of current frame with bboxes.
            analysis: ``GridAnalysis`` for the current grid.
            change: ``FrameChange`` from ``compute_diff`` (precise data).
            action: The action name that was taken.
            prev_bbox_image: Unused, kept for backward compatibility.

        Returns:
            ``FrameAnnotation`` or None on failure.
        """
        if curr_bbox_image is None:
            return None

        # Build detailed algorithmic diff text, with object IDs resolved
        diff_lines: list[str] = []
        diff_lines.append(
            f"Total cells changed: {change.total_changed} "
            f"({change.change_ratio * 100:.1f}% of grid)"
        )
        # Match movements to objects, filter out background fills
        matched_moves: list[tuple[int, MovedObject]] = []
        for m in change.moved:
            obj_id = self._match_movement_to_object(m, analysis)
            if obj_id is not None:
                matched_moves.append((obj_id, m))
            # Skip unmatched — likely background/floor filling back

        if matched_moves:
            diff_lines.append(f"Object movements detected: {len(matched_moves)}")
            for obj_id, m in matched_moves:
                colors_str = ", ".join(f"color {c}" for c in m.colors) if m.colors else "unknown color"
                diff_lines.append(
                    f"  - object #{obj_id} ({colors_str}, {m.size} cells): "
                    f"moved {m.direction} by ({m.dr},{m.dc}) cells, "
                    f"from ({m.old_center[0]:.0f},{m.old_center[1]:.0f}) "
                    f"to ({m.new_center[0]:.0f},{m.new_center[1]:.0f})"
                )
        else:
            diff_lines.append("No object movements detected.")
        if change.num_appeared > 0:
            diff_lines.append(f"Cells newly occupied: {change.num_appeared}")
        if change.num_disappeared > 0:
            diff_lines.append(f"Cells vacated: {change.num_disappeared}")
        if change.total_changed == 0:
            diff_lines.append("NOTHING CHANGED — the action had no effect.")

        diff_details = "\n".join(diff_lines)

        curr_url = self._encode_image(curr_bbox_image)
        object_table = self._build_object_table(analysis)

        system_text = _CHANGE_SYSTEM.format(action=action)
        user_text = _CHANGE_USER.format(
            object_table=object_table, diff_details=diff_details
        )

        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": curr_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        raw = self._call_llm(messages)
        if raw is None:
            return None
        return self._parse_changes(raw)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_image(image) -> str:
        """Encode a PIL Image to a base64 data URL."""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _match_movement_to_object(
        m: MovedObject, analysis: GridAnalysis
    ) -> Optional[int]:
        """Match a detected movement to the closest GridObject by color + position.

        Returns the obj_id or None if no good match.
        """
        best_id: Optional[int] = None
        best_dist = float("inf")

        for obj in analysis.objects:
            # Must share at least one color
            if m.colors and obj.color not in m.colors:
                continue
            # Distance from object center to movement's new position
            dist = (
                abs(obj.center_r - m.new_center[0])
                + abs(obj.center_c - m.new_center[1])
            )
            # Size should be in the same ballpark
            if obj.size > 0:
                size_ratio = min(m.size, obj.size) / max(m.size, obj.size)
            else:
                size_ratio = 0
            if size_ratio < 0.3:
                continue
            if dist < best_dist:
                best_dist = dist
                best_id = obj.obj_id

        # Only accept if reasonably close
        if best_dist > 5:
            return None
        return best_id

    @staticmethod
    def _build_object_table(analysis: GridAnalysis) -> str:
        """Build a text table of detected objects."""
        lines: list[str] = []
        for obj in analysis.objects:
            lines.append(
                f"  #{obj.obj_id}: color={obj.color}, size={obj.size}, "
                f"shape={obj.shape}, bbox=({obj.bbox[0]},{obj.bbox[1]})-"
                f"({obj.bbox[2]},{obj.bbox[3]}), "
                f"center=({obj.center_r:.0f},{obj.center_c:.0f}), "
                f"edge={obj.is_at_edge}"
            )
        return "\n".join(lines) if lines else "(none)"

    def _call_llm(self, messages: list[dict]) -> Optional[str]:
        """Make a VLM API call, return raw text or None on failure."""
        import logging

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "ObjectAnnotator LLM call failed: %s", exc
            )
            return None

    @staticmethod
    def _strip_json(raw: str) -> dict:
        """Best-effort extraction of JSON from VLM response."""
        import json

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return {}
            return {}

    @staticmethod
    def _parse_entities(raw: str) -> list[EntityAnnotation]:
        """Parse entity identification response."""
        data = ObjectAnnotator._strip_json(raw)
        result: list[EntityAnnotation] = []
        for e in data.get("entities", []):
            result.append(
                EntityAnnotation(
                    name=str(e.get("name", "unknown")),
                    object_ids=list(e.get("object_ids", [])),
                    role=str(e.get("role", "")),
                    appearance=str(e.get("appearance", "")),
                    confidence=str(e.get("confidence", "low")),
                )
            )
        return result

    @staticmethod
    def _parse_changes(raw: str) -> Optional[FrameAnnotation]:
        """Parse change interpretation response."""
        data = ObjectAnnotator._strip_json(raw)
        if not data:
            return None

        changes: list[ChangeAnnotation] = []
        # Accept both "interpretations" (new) and "changes" (legacy) keys
        items = data.get("interpretations", data.get("changes", []))
        for c in items:
            changes.append(
                ChangeAnnotation(
                    object_ids=list(c.get("object_ids", [])),
                    description=str(c.get("description", "")),
                    change_type=str(c.get("change_type", "unknown")),
                )
            )

        return FrameAnnotation(
            entities=[],  # filled separately by annotate()
            changes=changes,
            action_effect=str(data.get("action_effect", "")),
            nothing_changed=bool(data.get("nothing_changed", False)),
        )

    # ------------------------------------------------------------------ #
    #  Formatting for prompt injection
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_entities(annotations: list[EntityAnnotation]) -> str:
        """Format entity annotations as LLM-readable text."""
        if not annotations:
            return ""
        lines: list[str] = ["### Object Identification (from VLM):"]
        for a in annotations:
            ids = ", ".join(f"#{i}" for i in a.object_ids)
            desc = f"{a.role}"
            if a.appearance:
                desc += f" — {a.appearance}"
            lines.append(
                f"  {a.name} [{ids}]: {desc} (confidence: {a.confidence})"
            )
        return "\n".join(lines)

    @staticmethod
    def format_changes(frame_ann: FrameAnnotation) -> str:
        """Format change interpretation as LLM-readable text."""
        if frame_ann.nothing_changed:
            return "### Action Interpretation: No effect — action was wasted."

        lines: list[str] = ["### Action Interpretation (from VLM):"]
        for c in frame_ann.changes:
            ids = ", ".join(f"#{i}" for i in c.object_ids)
            lines.append(f"  [{ids}] {c.change_type}: {c.description}")

        if frame_ann.action_effect:
            lines.append(f"  => {frame_ann.action_effect}")

        return "\n".join(lines)

    @staticmethod
    def format_annotations(annotations: list[EntityAnnotation]) -> str:
        """Legacy alias for format_entities."""
        return ObjectAnnotator.format_entities(annotations)
