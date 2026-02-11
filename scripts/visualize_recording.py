#!/usr/bin/env python3
"""
Visualize an ARC-AGI-3 recording file (.recording.jsonl) as animation (GIF or MP4).

Reads the JSONL, extracts frame data (grids), renders them with the same 16-color
palette, draws step/state/action/levels on each frame, and writes a GIF (default)
or MP4 video.

Usage:
  uv run python scripts/visualize_recording.py path/to/foo.recording.jsonl
  uv run python scripts/visualize_recording.py path/to/foo.recording.jsonl -o out.gif
  uv run python scripts/visualize_recording.py path/to/foo.recording.jsonl -o out.mp4 --format mp4

If no file is given, lists recordings in RECORDINGS_DIR (from .env / RECORDINGS_DIR).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Action id in recording is int 0..7 (same as GameAction enum value)
ACTION_NAMES = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]

# 16-color palette (RGBA) – same as agents/templates/multimodal.py
_PALETTE = [
    (0xFF, 0xFF, 0xFF, 0xFF),  # 0 White
    (0xCC, 0xCC, 0xCC, 0xFF),  # 1
    (0x99, 0x99, 0x99, 0xFF),  # 2
    (0x66, 0x66, 0x66, 0xFF),  # 3
    (0x33, 0x33, 0x33, 0xFF),  # 4
    (0x00, 0x00, 0x00, 0xFF),  # 5 Black
    (0xE5, 0x3A, 0xA3, 0xFF),  # 6
    (0xFF, 0x7B, 0xCC, 0xFF),  # 7
    (0xF9, 0x3C, 0x31, 0xFF),  # 8
    (0x1E, 0x93, 0xFF, 0xFF),  # 9
    (0x88, 0xD8, 0xF1, 0xFF),  # 10
    (0xFF, 0xDC, 0x00, 0xFF),  # 11
    (0xFF, 0x85, 0x1B, 0xFF),  # 12
    (0x92, 0x12, 0x31, 0xFF),  # 13
    (0x4F, 0xCC, 0x30, 0xFF),  # 14
    (0xA3, 0x56, 0xD6, 0xFF),  # 15
]

_SCALE = 4  # 64px -> 256px for easier viewing


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env.example")
        load_dotenv(root / ".env", override=True)
    except ImportError:
        pass


def grid_to_png_bytes(grid: list[list[int]], scale: int = _SCALE) -> bytes:
    """Turn a 2D grid (list[list[int]] 0–15) into PNG bytes."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("PIL/Pillow is required: uv sync (or pip install Pillow)")

    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return b""

    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(_PALETTE[idx & 15])

    img = Image.frombytes("RGBA", (w, h), bytes(raw))
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_action_name(action_input: dict | None) -> str:
    """Get action name from recording event's action_input (id is int 0-7)."""
    if not action_input or not isinstance(action_input, dict):
        return "—"
    aid = action_input.get("id")
    if isinstance(aid, int) and 0 <= aid < len(ACTION_NAMES):
        return ACTION_NAMES[aid]
    if isinstance(aid, dict) and "name" in aid:
        return str(aid["name"])
    return "—"


def frames_to_pil_image(frame_data: list[list[list[int]]], scale: int = _SCALE):
    """Render a list of grids into one PIL Image (RGB, for drawing text)."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("PIL/Pillow is required: uv sync (or pip install Pillow)")

    if not frame_data:
        return Image.new("RGB", (64 * scale, 64 * scale), (255, 255, 255))
    good = []
    ref_h, ref_w = len(frame_data[0]), len(frame_data[0][0]) if frame_data[0] else 0
    for block in frame_data:
        if len(block) == ref_h and len(block[0]) == ref_w:
            good.append(block)
    if not good:
        return Image.new("RGB", (ref_w * scale, ref_h * scale), (255, 255, 255))

    sep = 4
    total_w = ref_w * len(good) + sep * (len(good) - 1)
    img = Image.new("RGBA", (total_w * scale, ref_h * scale), (255, 255, 255, 255))
    for i, block in enumerate(good):
        raw = bytearray()
        for row in block:
            for idx in row:
                raw.extend(_PALETTE[idx & 15])
        part = Image.frombytes("RGBA", (ref_w, ref_h), bytes(raw))
        part = part.resize((ref_w * scale, ref_h * scale), Image.NEAREST)
        img.paste(part, (i * (ref_w * scale + sep * scale), 0))
    return img.convert("RGB")


def load_events(path: str) -> list[dict]:
    """Load JSONL file; return list of parsed events."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def is_frame_event(ev: dict) -> bool:
    """True if this event has frame data we can render."""
    data = ev.get("data") or {}
    return isinstance(data, dict) and "frame" in data and isinstance(data["frame"], list)


def _extract_steps(events: list[dict]) -> list[dict]:
    """Build list of step dicts: index, state, action, levels_completed, and PIL Image."""
    steps = []
    for ev in events:
        if not is_frame_event(ev):
            continue
        data = ev["data"]
        state = data.get("state", "?")
        action_name = get_action_name(data.get("action_input"))
        levels = data.get("levels_completed", "—")
        try:
            img = frames_to_pil_image(data["frame"])
        except Exception:
            continue
        steps.append(
            {
                "index": len(steps) + 1,
                "state": str(state),
                "action": str(action_name),
                "levels_completed": str(levels),
                "img": img,
            }
        )
    return steps


def _draw_overlay(img, step: dict) -> None:
    """Draw step/state/action/levels as text bar at bottom; ensure width fits text."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    w, h = img.size
    bar_h = 22
    text = f"Step {step['index']}  |  State: {step['state']}  |  Action: {step['action']}  |  Levels: {step['levels_completed']}"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
    # Ensure canvas is wide enough for full text (use dummy draw for bbox)
    dummy = Image.new("RGB", (1, 1))
    ddraw = ImageDraw.Draw(dummy)
    try:
        bbox = ddraw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
    except AttributeError:
        text_w = 520  # fallback min width
    min_width = max(w, text_w + 24)
    expanded = Image.new("RGB", (min_width, h + bar_h), (40, 40, 40))
    expanded.paste(img, (0, 0))
    draw = ImageDraw.Draw(expanded)
    draw.text((8, h + 4), text, fill=(220, 220, 220), font=font)
    step["img_with_overlay"] = expanded


def build_gif(events: list[dict], output_path: str, duration_ms: int = 400) -> None:
    """Write an animated GIF from recording frame events."""
    steps = _extract_steps(events)
    if not steps:
        raise ValueError("No frame steps to export")
    for s in steps:
        _draw_overlay(s["img"], s)
    frames = [s.get("img_with_overlay", s["img"]) for s in steps]
    frames = _normalize_frame_sizes(frames)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


OUTPUT_SIZE = (512, 512)


def _normalize_frame_sizes(frames: list, target_size: tuple[int, int] = OUTPUT_SIZE) -> list:
    """Resize all frames to target_size (default 512x512)."""
    try:
        from PIL import Image
    except ImportError:
        return frames
    if not frames:
        return frames
    out = []
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for f in frames:
        if f.size == target_size:
            out.append(f)
            continue
        out.append(f.resize(target_size, resample))
    return out


def build_mp4(events: list[dict], output_path: str, fps: float = 2.5) -> None:
    """Write an MP4 video from recording frame events (requires imageio)."""
    try:
        import imageio
    except ImportError:
        raise SystemExit("MP4 output requires: pip install imageio[ffmpeg] (and ffmpeg)")

    steps = _extract_steps(events)
    if not steps:
        raise ValueError("No frame steps to export")
    for s in steps:
        _draw_overlay(s["img"], s)
    frames = [s.get("img_with_overlay", s["img"]) for s in steps]
    frames = _normalize_frame_sizes(frames)
    import numpy as np

    arrs = [np.asarray(f) for f in frames]
    imageio.mimsave(output_path, arrs, fps=fps)


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Visualize an ARC-AGI-3 recording (.recording.jsonl) as GIF or MP4.",
    )
    parser.add_argument(
        "recording",
        nargs="?",
        help="Path to a .recording.jsonl file. If omitted, list recordings in RECORDINGS_DIR.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output path (default: same name as recording with .gif or .mp4)",
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="Output format: gif (default) or mp4 (requires imageio[ffmpeg])",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=400,
        help="GIF: duration per frame in ms (default 400)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.5,
        help="MP4: frames per second (default 2.5)",
    )
    args = parser.parse_args()

    if not args.recording:
        rec_dir = os.environ.get("RECORDINGS_DIR", "recordings")
        if not os.path.isdir(rec_dir):
            print(f"Recordings directory not found: {rec_dir}", file=sys.stderr)
            return 1
        files = sorted(f for f in os.listdir(rec_dir) if f.endswith(".recording.jsonl"))
        if not files:
            print(f"No .recording.jsonl files in {rec_dir}", file=sys.stderr)
            return 1
        print("Recordings (use one as argument):")
        for f in files:
            print(f"  {os.path.join(rec_dir, f)}")
        return 0

    path = args.recording
    if not os.path.isfile(path):
        rec_dir = os.environ.get("RECORDINGS_DIR", "recordings")
        alt = os.path.join(rec_dir, path)
        if os.path.isfile(alt):
            path = alt
        else:
            print(f"File not found: {args.recording}", file=sys.stderr)
            return 1

    events = load_events(path)
    frame_events = [e for e in events if is_frame_event(e)]
    if not frame_events:
        print("No frame data found in this recording.", file=sys.stderr)
        return 1

    ext = ".mp4" if args.format == "mp4" else ".gif"
    out = args.output or path.rsplit(".", 1)[0] + ext
    if not out.endswith(ext):
        out += ext

    if args.format == "mp4":
        build_mp4(events, out, fps=args.fps)
    else:
        build_gif(events, out, duration_ms=args.duration)
    print(f"Wrote {len(frame_events)} steps to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
