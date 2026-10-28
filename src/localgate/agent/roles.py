"""Named sub-agent roles for `delegate_task`.

A role is a reusable specialist: a description the orchestrating model reads to
decide when to use it, the tools it gets, an optional model, and a system prompt.
On local hardware the per-role model is the point — exploration can run on a
small, fast model while edits stay on the strongest one available.

Roles come from three places, later ones overriding earlier ones by name:

1. the built-ins below;
2. ``<config dir>/agents/*.md`` (user-wide);
3. ``<project>/.localgate/agents/*.md`` (per project).

Each file is Markdown with a small front-matter block::

    ---
    name: tester
    description: Writes and runs focused tests for a change.
    tools: read_file, search_files, create_file, edit_file, run_command
    model: qwen2.5-coder:7b
    max_turns: 12
    ---
    You write minimal, focused tests ...

Only ``key: value`` lines are understood (no nested YAML), which keeps the format
dependency-free and identical on every platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from localgate.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentRole:
    name: str
    description: str
    tools: tuple[str, ...] = field(default_factory=tuple)
    model: str | None = None
    max_turns: int = 10
    prompt: str = ""


BUILTIN_ROLES: dict[str, AgentRole] = {
    "explorer": AgentRole(
        name="explorer",
        description="Read-only search of the project: finds code, call sites and patterns.",
        tools=("read_file", "list_directory", "search_files", "git_status", "git_diff"),
        prompt=(
            "Explore the project to answer the question. Do not change files. Reply with a "
            "concise answer that cites exact file paths and line numbers."
        ),
    ),
    "researcher": AgentRole(
        name="researcher",
        description="Looks things up on the web (docs, APIs, errors) and summarizes with sources.",
        tools=("web_search", "fetch_url", "read_file", "search_files"),
        prompt=(
            "Research the question on the web. Prefer official documentation. Reply with a "
            "short answer and the URLs you relied on."
        ),
    ),
    "reviewer": AgentRole(
        name="reviewer",
        description="Reviews the current uncommitted changes for bugs and risky edits.",
        tools=("git_status", "git_diff", "read_file", "search_files"),
        prompt=(
            "Review the uncommitted changes. Report concrete bugs with file and line, most "
            "severe first. Do not change files."
        ),
    ),
}


def _parse_role(text: str, fallback_name: str) -> AgentRole | None:
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().lower()] = value.strip()
            body = parts[2]
    name = meta.get("name") or fallback_name
    description = meta.get("description", "")
    if not name or not description:
        return None
    tools = tuple(t.strip() for t in meta.get("tools", "").split(",") if t.strip())
    try:
        max_turns = max(1, int(meta.get("max_turns", "10")))
    except ValueError:
        max_turns = 10
    return AgentRole(
        name=name,
        description=description,
        tools=tools,
        model=meta.get("model") or None,
        max_turns=max_turns,
        prompt=body.strip(),
    )


def load_roles(*directories: Path) -> dict[str, AgentRole]:
    """Built-in roles overlaid with every ``*.md`` role in ``directories``, in order."""
    roles = dict(BUILTIN_ROLES)
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                role = _parse_role(path.read_text(encoding="utf-8"), path.stem)
            except OSError as exc:
                logger.warning("agent_role_unreadable", path=str(path), error=str(exc))
                continue
            if role is None:
                logger.warning("agent_role_invalid", path=str(path))
                continue
            roles[role.name] = role
    return roles
