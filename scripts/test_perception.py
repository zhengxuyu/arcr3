#!/usr/bin/env python3
"""
Test script for GridPerception accuracy.

Runs random actions in a game, generates GridPerception analysis at each step,
and saves results (raw grid as hex image + analysis text) to an output
directory for manual inspection.

Usage:
    # Run on all available games (first 3), 20 actions each:
    python scripts/test_perception.py

    # Run on a specific game:
    python scripts/test_perception.py -g GAME_ID

    # More actions per game:
    python scripts/test_perception.py -n 40

    # Custom output directory:
    python scripts/test_perception.py -o /tmp/perception_test

Output structure:
    perception_test_results/
    ├── GAME_ID_1/
    │   ├── step_000_RESET.txt          # perception analysis text
    │   ├── step_000_RESET_grid.txt     # raw grid as hex
    │   ├── step_001_ACTION3.txt
    │   ├── step_001_ACTION3_grid.txt
    │   ├── ...
    │   └── summary.txt                 # full-game summary
    └── GAME_ID_2/
        └── ...
"""

# ruff: noqa: E402
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.templates.grid_perception import GridPerception, ObjectAnnotator, FrameAnnotation

logger = logging.getLogger(__name__)

HEX_CHARS = "0123456789ABCDEF"

# ARC-AGI color palette (approximate RGB for PIL rendering)
ARC_COLORS = [
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


def grid_to_hex(grid: list[list[int]]) -> str:
    """Convert a grid to a compact hex representation."""
    lines = []
    # Column headers (every 5th)
    header = "    " + "".join(
        str(c % 10) if c % 5 == 0 else " " for c in range(len(grid[0]))
    )
    lines.append(header)

    for r, row in enumerate(grid):
        hex_row = "".join(HEX_CHARS[min(v, 15)] for v in row)
        lines.append(f"{r:3d} {hex_row}")

    lines.append(f"\nGrid size: {len(grid)}x{len(grid[0])}")
    lines.append("Hex key: 0-9=colors 0-9, A=10, B=11, C=12, D=13, E=14, F=15")
    return "\n".join(lines)


def grid_to_image(grid: list[list[int]], scale: int = 4) -> "PIL.Image.Image":
    """Convert a grid to a PIL Image for visual inspection."""
    try:
        from PIL import Image
    except ImportError:
        return None

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img = Image.new("RGB", (w * scale, h * scale))
    pixels = img.load()

    for r in range(h):
        for c in range(w):
            color = ARC_COLORS[min(grid[r][c], 15)]
            for dy in range(scale):
                for dx in range(scale):
                    pixels[c * scale + dx, r * scale + dy] = color

    return img


def convert_raw_frame(raw) -> FrameData:
    """Convert raw frame data to FrameData."""
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def run_perception_test(
    game_id: str,
    output_dir: Path,
    num_actions: int = 20,
    arcade: Arcade = None,
    annotator: ObjectAnnotator = None,
    annotate_interval: int = 5,
) -> dict:
    """Run random actions and save perception analysis for one game.

    Args:
        annotator: If provided, call VLM to annotate object roles every
                   *annotate_interval* steps (and on RESET frames).
        annotate_interval: How often (in steps) to run VLM annotation.

    Returns a summary dict with stats.
    """
    game_dir = output_dir / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    perception = GridPerception()

    # Create environment
    scorecard_id = arcade.open_scorecard(tags=["perception_test"])
    env = arcade.make(game_id, scorecard_id=scorecard_id)

    frames: list[FrameData] = []
    prev_frame: FrameData = None
    actions_taken: list[str] = []
    full_analysis_log: list[str] = []

    seed = int(time.time() * 1000) + hash(game_id) % 10000
    rng = random.Random(seed)

    step = 0
    game_over_count = 0

    for i in range(num_actions):
        # Choose action
        if not frames or frames[-1].state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            if frames and frames[-1].state == GameState.GAME_OVER:
                game_over_count += 1
                if game_over_count > 3:
                    logger.info(f"  Too many game overs, stopping")
                    break
        else:
            # Random action (no RESET)
            choices = [a for a in GameAction if a is not GameAction.RESET]
            action = rng.choice(choices)

        if action.is_complex():
            action.set_data({"x": rng.randint(0, 63), "y": rng.randint(0, 63)})

        # Execute action
        raw = env.step(action, data=action.action_data.model_dump(), reasoning={})
        frame = convert_raw_frame(raw)
        actions_taken.append(action.name)

        prev_grids = prev_frame.frame if prev_frame else None

        # Run perception
        if frame.frame:
            analysis_text = perception.describe_frame(
                frame=frame.frame,
                prev_frame=prev_grids,
                last_action=action.name if prev_frame else None,
            )
        else:
            analysis_text = "(no frame data)"

        # Build full step report
        report_lines = [
            f"=" * 70,
            f"Step {step}: {action.name}",
            f"State: {frame.state.name}",
            f"Score: {getattr(frame, 'score', frame.levels_completed)}",
            f"=" * 70,
            "",
            "--- PERCEPTION ANALYSIS ---",
            analysis_text,
            "",
        ]
        report = "\n".join(report_lines)
        full_analysis_log.append(report)

        # Save step analysis
        step_file = game_dir / f"step_{step:03d}_{action.name}.txt"
        step_file.write_text(report, encoding="utf-8")

        # Save raw grid as hex
        if frame.frame:
            grid_lines = []
            for g_idx, grid in enumerate(frame.frame):
                grid_lines.append(f"Grid {g_idx}:")
                grid_lines.append(grid_to_hex(grid))
                grid_lines.append("")
            grid_file = game_dir / f"step_{step:03d}_{action.name}_grid.txt"
            grid_file.write_text("\n".join(grid_lines), encoding="utf-8")

            # Save grid image with bounding boxes
            bbox_img = perception.render_frame(frame.frame, scale=8)
            if bbox_img:
                img_file = game_dir / f"step_{step:03d}_{action.name}_bbox.png"
                bbox_img.save(str(img_file))

            # Also save plain grid image (no boxes) for comparison
            for g_idx, grid in enumerate(frame.frame):
                img = grid_to_image(grid)
                if img:
                    img_file = game_dir / f"step_{step:03d}_{action.name}_g{g_idx}.png"
                    img.save(str(img_file))

            # --- VLM annotation (entity identification + change detection) ---
            if annotator is not None:
                for g_idx, grid in enumerate(frame.frame):
                    g_analysis = perception.analyze_grid(grid)
                    g_bbox_img = perception.render_grid(grid, analysis=g_analysis, scale=8)

                    ann_lines = [
                        f"Step {step} / Grid {g_idx} — VLM Analysis",
                        f"Action: {action.name}",
                        "",
                    ]

                    # Entity identification (periodic + on RESET)
                    should_identify = (
                        action is GameAction.RESET
                        or step % annotate_interval == 0
                    )
                    if should_identify:
                        entities = annotator.annotate(g_bbox_img, g_analysis)
                        entity_text = ObjectAnnotator.format_entities(entities)
                        ann_lines.append("--- ENTITY IDENTIFICATION ---")
                        ann_lines.append(entity_text if entity_text else "(no entities returned)")
                        ann_lines.append("")
                        for a in entities:
                            ann_lines.append(
                                f"  {a.name}: ids={a.object_ids}, "
                                f"role={a.role}, appearance={a.appearance}, "
                                f"conf={a.confidence}"
                            )
                        ann_lines.append("")
                        report += f"\n--- VLM ENTITIES (Grid {g_idx}) ---\n{entity_text}\n"

                    # Change interpretation (every step that has a previous frame)
                    prev_bbox_key = f"_prev_bbox_g{g_idx}"
                    prev_bbox = getattr(perception, prev_bbox_key, None)
                    if prev_bbox is not None and prev_grids is not None and g_idx < len(prev_grids):
                        change = perception.compute_diff(prev_grids[g_idx], grid)
                        frame_ann = annotator.annotate_change(
                            curr_bbox_image=g_bbox_img,
                            analysis=g_analysis,
                            change=change,
                            action=action.name,
                        )
                        if frame_ann is not None:
                            change_text = ObjectAnnotator.format_changes(frame_ann)
                            ann_lines.append("--- CHANGE DETECTION ---")
                            ann_lines.append(change_text)
                            ann_lines.append("")
                            if frame_ann.action_effect:
                                ann_lines.append(f"Action effect: {frame_ann.action_effect}")
                            ann_lines.append("")
                            report += f"\n--- VLM CHANGES (Grid {g_idx}) ---\n{change_text}\n"

                    # Store current bbox image for next step
                    setattr(perception, prev_bbox_key, g_bbox_img)

                    ann_file = game_dir / f"step_{step:03d}_{action.name}_g{g_idx}_annotations.txt"
                    ann_file.write_text("\n".join(ann_lines), encoding="utf-8")

                # Re-save the step report with VLM analysis appended
                step_file.write_text(report, encoding="utf-8")
                full_analysis_log[-1] = report

        prev_frame = frame
        frames.append(frame)
        step += 1

        if frame.state == GameState.WIN:
            logger.info(f"  WIN at step {step}!")
            break

    # Save full summary
    summary_lines = [
        f"Game: {game_id}",
        f"Total steps: {step}",
        f"Actions: {', '.join(actions_taken)}",
        f"Final state: {frames[-1].state.name if frames else 'N/A'}",
        f"Final score: {getattr(frames[-1], 'score', frames[-1].levels_completed) if frames else 0}",
        f"Game overs: {game_over_count}",
        "",
        "=" * 70,
        "LEARNED ACTION EFFECTS:",
        "=" * 70,
    ]
    for act, effects in sorted(perception._action_effects.items()):
        dirs = [perception._direction_name(dr, dc) for dr, dc in effects]
        summary_lines.append(f"  {act}: {dirs}")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("FULL ANALYSIS LOG:")
    summary_lines.append("=" * 70)
    summary_lines.extend(full_analysis_log)

    summary_file = game_dir / "summary.txt"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    # Close scorecard
    try:
        arcade.close_scorecard(scorecard_id)
    except Exception:
        pass

    result = {
        "game_id": game_id,
        "steps": step,
        "final_state": frames[-1].state.name if frames else "N/A",
        "game_overs": game_over_count,
        "action_effects_learned": len(perception._action_effects),
        "player_identified": perception._player_obj is not None,
        "output_dir": str(game_dir),
    }
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test GridPerception on ARC-AGI-3 games with random actions"
    )
    parser.add_argument(
        "-g", "--game",
        help="Specific game ID (comma-separated for multiple). Default: first 3 available.",
    )
    parser.add_argument(
        "-n", "--num-actions", type=int, default=20,
        help="Number of random actions per game (default: 20)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="perception_test_results",
        help="Output directory (default: perception_test_results)",
    )
    parser.add_argument(
        "--vlm", action="store_true",
        help="Enable VLM annotation: send bbox images to GPT-4o-mini to guess object roles",
    )
    parser.add_argument(
        "--vlm-model", type=str, default="gpt-4o-mini",
        help="Model for VLM annotation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--annotate-interval", type=int, default=5,
        help="Run VLM annotation every N steps (default: 5, also runs on RESET)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    arcade = Arcade()

    # Discover games
    import requests
    SCHEME = os.environ.get("SCHEME", "http")
    HOST = os.environ.get("HOST", "localhost")
    PORT = os.environ.get("PORT", 8001)
    if (SCHEME == "http" and str(PORT) == "80") or (SCHEME == "https" and str(PORT) == "443"):
        ROOT_URL = f"{SCHEME}://{HOST}"
    else:
        ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"

    HEADERS = {
        "X-API-Key": os.getenv("ARC_API_KEY", ""),
        "Accept": "application/json",
    }

    all_games = []
    try:
        r = requests.get(f"{ROOT_URL}/api/games", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            all_games = [g["game_id"] for g in r.json()]
    except Exception as e:
        logger.error(f"Failed to fetch games: {e}")

    if not all_games:
        logger.error("No games available. Is the API server running?")
        return

    # Select games
    if args.game:
        filters = args.game.split(",")
        games = [g for g in all_games if any(g.startswith(f) for f in filters)]
    else:
        games = all_games[:3]

    if not games:
        logger.error(f"No matching games found. Available: {all_games[:10]}")
        return

    # Create VLM annotator if requested
    annotator = None
    if args.vlm:
        logger.info(f"VLM annotation enabled: model={args.vlm_model}, interval={args.annotate_interval}")
        annotator = ObjectAnnotator(model=args.vlm_model)

    logger.info(f"Testing perception on {len(games)} game(s): {games}")
    logger.info(f"Output directory: {output_dir.resolve()}")

    results = []
    for game_id in games:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing game: {game_id}")
        logger.info(f"{'='*60}")

        try:
            result = run_perception_test(
                game_id=game_id,
                output_dir=output_dir,
                num_actions=args.num_actions,
                arcade=arcade,
                annotator=annotator,
                annotate_interval=args.annotate_interval,
            )
            results.append(result)
            logger.info(
                f"  Done: {result['steps']} steps, "
                f"state={result['final_state']}, "
                f"effects_learned={result['action_effects_learned']}, "
                f"player_found={result['player_identified']}"
            )
        except Exception as e:
            logger.error(f"  Error on {game_id}: {e}", exc_info=True)
            results.append({"game_id": game_id, "error": str(e)})

    # Save overall results
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary table
    print(f"\n{'='*70}")
    print("PERCEPTION TEST RESULTS")
    print(f"{'='*70}")
    print(f"{'Game':<30} {'Steps':>6} {'State':<12} {'Effects':>8} {'Player':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['game_id']:<30} ERROR: {r['error']}")
        else:
            print(
                f"{r['game_id']:<30} {r['steps']:>6} "
                f"{r['final_state']:<12} "
                f"{r['action_effects_learned']:>8} "
                f"{'Yes' if r['player_identified'] else 'No':>8}"
            )
    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"Review the step_*.txt files for perception accuracy.")


if __name__ == "__main__":
    main()
