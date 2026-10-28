"""Custom slash commands: a Markdown file per command, like Claude Code's.

``.localgate/commands/review.md`` in a project (or ``commands/review.md`` in the
user config directory) becomes ``/review``. The file body is the prompt sent to
the agent; ``$ARGUMENTS`` is replaced with whatever follows the command name, or
the arguments are appended when the body doesn't mention them. An optional
front-matter block supplies the one-line description shown in ``/help``::

    ---
    description: Review the working-tree diff for bugs
    ---
    Review `git diff` for correctness bugs. Focus on: $ARGUMENTS
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from localgate import paths

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class CustomCommand:
    name: str
    description: str
    template: str
    source: Path

    def expand(self, arguments: str) -> str:
        if "$ARGUMENTS" in self.template:
            return self.template.replace("$ARGUMENTS", arguments)
        return f"{self.template}\n\n{arguments}" if arguments else self.template


def _parse(path: Path) -> CustomCommand | None:
    name = path.stem.lower()
    if not _NAME_RE.match(name):
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    description = ""
    if text.startswith("---\n"):
        header, sep, body = text[4:].partition("\n---")
        if sep:
            text = body.lstrip("-").lstrip("\n")
            for line in header.splitlines():
                key, _, value = line.partition(":")
                if key.strip().lower() == "description":
                    description = value.strip()
    template = text.strip()
    if not template:
        return None
    if not description:
        description = template.splitlines()[0][:60]
    return CustomCommand(name, description, template, path)


def command_dirs(root: Path) -> tuple[Path, Path]:
    """User-level first, so a project command of the same name wins."""
    return paths.config_dir() / "commands", root / ".localgate" / "commands"


def load_commands(root: Path) -> dict[str, CustomCommand]:
    commands: dict[str, CustomCommand] = {}
    for directory in command_dirs(root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            command = _parse(path)
            if command is not None:
                commands[command.name] = command
    return commands
