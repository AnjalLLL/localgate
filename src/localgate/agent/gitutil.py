"""Minimal git plumbing for the coding agent's safety net.

Shells out to the ``git`` binary rather than adding a dependency like GitPython —
these are a handful of read-only or trivially-reversible commands, not a reason to
carry a whole library. Every write here (`commit_all`, `checkout_file`,
`reset_hard_last`) exists to make an agent's mistake a five-second `git` command
away from undone, per CODING_AGENT_PLAN.md's git-aware safety net.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Commit messages the agent makes are prefixed with this, so `/undo` can refuse
#: to `reset --hard` a commit a human made themselves.
AGENT_COMMIT_PREFIX = "localgate-agent:"


class GitError(RuntimeError):
    """A git command exited non-zero, or git isn't installed."""


def _run(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed"
        raise GitError(message)
    return result.stdout


def is_repo(root: Path) -> bool:
    try:
        return _run(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except GitError:
        return False


def is_dirty(root: Path) -> bool:
    return bool(_run(root, "status", "--porcelain").strip())


def status(root: Path) -> str:
    return _run(root, "status", "--porcelain")


def diff(root: Path, path: str | None = None) -> str:
    args = ["diff"] if path is None else ["diff", "--", path]
    return _run(root, *args)


def last_commit_message(root: Path) -> str | None:
    try:
        return _run(root, "log", "-1", "--format=%s").strip()
    except GitError:
        return None  # no commits yet


def commit_all(root: Path, message: str) -> bool:
    """Stage and commit everything. Returns False if there was nothing to commit."""
    _run(root, "add", "-A")
    try:
        _run(root, "commit", "-m", message)
    except GitError as exc:
        if "nothing to commit" in str(exc):
            return False
        raise
    return True


def checkout_file(root: Path, path: str) -> None:
    """Restore ``path`` to its state in the last commit, discarding local edits."""
    _run(root, "checkout", "--", path)


def reset_hard_last(root: Path) -> None:
    """Drop the last commit and its changes entirely. Irreversible past this point."""
    _run(root, "reset", "--hard", "HEAD~1")


def file_is_untracked(root: Path, path: str) -> bool:
    """True if ``path`` has never been committed (a plain `checkout --` can't undo it)."""
    porcelain = _run(root, "status", "--porcelain", "--", path)
    return porcelain.startswith("??")


def undo_file(root: Path, path: str) -> str:
    """Best-effort undo of the most recent write to ``path``.

    A brand-new file has nothing to check out back to, so undoing it means
    deleting it; anything else reverts to the last commit.
    """
    if file_is_untracked(root, path):
        target = root / path
        if target.exists():
            target.unlink()
        return f"Deleted newly created {path} (it was never committed)."
    checkout_file(root, path)
    return f"Reverted {path} to its last committed version."
