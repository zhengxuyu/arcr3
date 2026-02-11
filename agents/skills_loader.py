"""
Load local skills from a directory (e.g. SKILL.md files) for injection into the LLM agent.

Set SKILLS_DIR in .env to a path; default is "skills" in the current working directory.
Also checks common locations like .cursor/skills-cursor and ~/.cursor/skills if SKILLS_DIR is unset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_FILENAMES = ("SKILL.md", "skill.md", "README.md")


def _get_skills_dirs() -> list[Path]:
    """Return list of directories to search for skills (first existing wins for loading)."""
    cwd = Path.cwd()
    env_dir = os.environ.get("SKILLS_DIR", "").strip()
    dirs: list[Path] = []
    if env_dir:
        p = Path(env_dir).expanduser()
        if not p.is_absolute():
            p = cwd / p
        dirs.append(p)
    # Defaults: project skills/, .cursor/skills-cursor, ~/.cursor/skills
    dirs.append(cwd / "skills")
    dirs.append(cwd / ".cursor" / "skills-cursor")
    dirs.append(Path.home() / ".cursor" / "skills-cursor")
    dirs.append(Path.home() / ".cursor" / "skills")
    return dirs


def _read_skill_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("Could not read skill %s: %s", path, e)
        return ""


def load_local_skills(max_content_chars: int = 0) -> list[tuple[str, str]]:
    """
    Discover and load skills from the first existing SKILLS_DIR.

    Scans for subdirectories containing SKILL.md (or skill.md, README.md), reads each file,
    and returns a list of (skill_name, content). skill_name is the directory name.

    If max_content_chars > 0, truncate each skill's content to that length (with a note).
    """
    for base in _get_skills_dirs():
        if not base.is_dir():
            continue
        result: list[tuple[str, str]] = []
        try:
            for sub in sorted(base.iterdir()):
                if not sub.is_dir():
                    continue
                for fname in SKILL_FILENAMES:
                    f = sub / fname
                    if f.is_file():
                        content = _read_skill_file(f)
                        if content:
                            if max_content_chars > 0 and len(content) > max_content_chars:
                                content = content[:max_content_chars] + "\n\n... [truncated]"
                            result.append((sub.name, content))
                        break
            if result:
                logger.info("Loaded %d skill(s) from %s", len(result), base)
                return result
        except Exception as e:
            logger.warning("Error scanning skills dir %s: %s", base, e)
    return []


def format_skills_for_prompt(
    skills: list[tuple[str, str]], heading: str = "Available skills (follow when relevant):"
) -> str:
    """Format a list of (name, content) skills into one string for the system prompt."""
    if not skills:
        return ""
    parts = [heading, ""]
    for name, content in skills:
        parts.append(f"## {name}")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).strip()
