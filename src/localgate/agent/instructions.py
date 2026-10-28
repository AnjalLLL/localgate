"""Standing instructions the agent reads every turn, like Claude Code's CLAUDE.md.

Two layers, both optional: a per-user ``LOCALGATE.md`` in the config directory
(personal preferences that apply everywhere) and a per-project file at the
project root (build commands, conventions, things to avoid). The project file is
the first of ``INSTRUCTION_FILES`` that exists, so a repo that already carries an
``AGENTS.md`` or ``CLAUDE.md`` works without a copy.
"""

from __future__ import annotations

from pathlib import Path

from localgate import paths

INSTRUCTION_FILES: tuple[str, ...] = ("LOCALGATE.md", "AGENTS.md", "CLAUDE.md")
#: Each file is cut to this many characters, so a huge file can't eat the context.
MAX_INSTRUCTION_CHARS = 8_000

INIT_PROMPT = (
    "Create a LOCALGATE.md file at the project root: standing instructions for a coding "
    "agent working in this project. First explore with list_directory, read_file and "
    "search_files — look at manifests (package.json, pyproject.toml, Makefile, ...), the "
    "README and the main source files. Then write LOCALGATE.md with these sections, each "
    "short and specific to this project:\n"
    "## Overview — what the project is, in two or three sentences\n"
    "## Commands — how to install, run, build, test and lint (only commands that exist)\n"
    "## Layout — the important directories and files and what they hold\n"
    "## Conventions — code style, naming and patterns the existing code follows\n"
    "Do not invent commands or files you did not see. If LOCALGATE.md already exists, "
    "read it and improve it instead of starting over."
)


def project_instructions_file(root: Path) -> Path | None:
    """The project's instructions file, or ``None`` when it has none."""
    for name in INSTRUCTION_FILES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def user_instructions_file() -> Path:
    return paths.config_dir() / "LOCALGATE.md"


def _read(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) > MAX_INSTRUCTION_CHARS:
        text = text[:MAX_INSTRUCTION_CHARS] + "\n[... truncated ...]"
    return text


def load_instructions(root: Path) -> str:
    """Both layers formatted for the system prompt, or ``""`` when neither exists."""
    sections: list[str] = []
    user_text = _read(user_instructions_file())
    if user_text:
        sections.append(f"User instructions (~/.config/localgate/LOCALGATE.md):\n{user_text}")
    project_file = project_instructions_file(root)
    project_text = _read(project_file)
    if project_file is not None and project_text:
        sections.append(f"Project instructions ({project_file.name}):\n{project_text}")
    if not sections:
        return ""
    return "\n\nFollow these standing instructions; they override your defaults.\n\n" + "\n\n".join(
        sections
    )
