"""
Persistent scratchpad for the agent. Content survives process restarts and game resets.

Set SCRATCHPAD_PATH in .env to a file path (default: agent_scratchpad.md in current working directory).
The agent has full control: each time it can output a complete new version in ```scratchpad ... ```
and we overwrite the file with that content.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def get_scratchpad_path() -> Path:
    """Return the scratchpad file path (from env or default)."""
    env_path = os.environ.get("SCRATCHPAD_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    return Path.cwd() / "agent_scratchpad.md"


def read_scratchpad() -> str:
    """Read current scratchpad content. Returns empty string if file missing or unreadable."""
    path = get_scratchpad_path()
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
        logger.debug("Scratchpad path (file not yet created): %s", path.resolve())
    except Exception as e:
        logger.warning("Could not read scratchpad %s: %s", path, e)
    return ""


def write_scratchpad(content: str) -> None:
    """Overwrite the scratchpad file with the given content (full edit by the model)."""
    path = get_scratchpad_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if content else "", encoding="utf-8")
        logger.info("Scratchpad written to: %s (%d bytes)", path.resolve(), len(content or ""))
    except Exception as e:
        logger.warning("Could not write scratchpad %s: %s", path, e)


def parse_scratchpad_block_from_text(text: str) -> str | None:
    """
    If the model's text contains a block ```scratchpad ... ```, return that content (full new version).
    Otherwise return None.
    """
    if not text or "scratchpad" not in text.lower():
        return None
    m = re.search(r"```\s*scratchpad\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None
