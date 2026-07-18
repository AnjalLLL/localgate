"""A minimal `.gitignore`-style pattern matcher for `.gitignore` and
`.localgateignore`.

This is a deliberate subset, not a full implementation: `*`, `**`, `?`, a leading
`/` to anchor at the root, and a trailing `/` for directory-only patterns are all
supported; negation (`!pattern`) is not — a negation line is simply skipped, which
means it's ignored, and everything it would have un-ignored stays covered by
whatever pattern matched it. That is the safe direction to be wrong in for a tool
whose whole job is keeping secrets out of the model's hands: under-ignoring would
leak a file, over-ignoring only costs the model a read it didn't get to make.

The `.git` directory is always excluded, regardless of what's in either ignore
file — there is no legitimate reason for the agent to read `.git/config` or
`.git/objects`.
"""

from __future__ import annotations

import re
from pathlib import Path

_ALWAYS_IGNORED_DIRS = frozenset({".git"})


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    # A trailing "/" restricts a real gitignore pattern to directories, but either
    # way a match on a directory also covers everything under it — so the two
    # cases only differ in a distinction (file vs. directory) this matcher
    # doesn't track. Both are treated the same: match the entry itself, or
    # anything nested below it.
    body = pattern.rstrip("/")
    anchored = body.startswith("/")
    body = body.lstrip("/")

    # Escape everything, then re-expand the glob metacharacters we support.
    # A NUL placeholder for "**" survives re.escape and single-"*" expansion
    # untouched, since NUL never appears in a real pattern.
    escaped = re.escape(body).replace(r"\*\*", "\0").replace(r"\*", "[^/]*")
    escaped = escaped.replace(r"\?", "[^/]").replace("\0", ".*")

    prefix = "^" if anchored else "(^|.*/)"
    return re.compile(prefix + escaped + "(/.*)?$")


def load_patterns(root: Path) -> list[re.Pattern[str]]:
    """Read `.gitignore` and `.localgateignore` from ``root``, if present."""
    patterns: list[re.Pattern[str]] = []
    for filename in (".gitignore", ".localgateignore"):
        ignore_file = root / filename
        if not ignore_file.is_file():
            continue
        for raw_line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            patterns.append(_pattern_to_regex(line))
    return patterns


def is_ignored(root: Path, path: Path, patterns: list[re.Pattern[str]]) -> bool:
    """Whether ``path`` (absolute, inside ``root``) is excluded from the agent's view."""
    relative = path.relative_to(root)
    if _ALWAYS_IGNORED_DIRS.intersection(relative.parts):
        return True
    rel_posix = relative.as_posix()
    return any(pattern.match(rel_posix) for pattern in patterns)
