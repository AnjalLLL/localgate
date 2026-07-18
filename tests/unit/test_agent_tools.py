"""Path-escape protection is the one thing agent/tools.py must never get wrong —
these tests exist to prove `../` and absolute-path escapes are actually blocked,
not just intended to be.
"""

import subprocess

import pytest

from localgate.agent.tools import (
    IgnoredPathError,
    PathEscapeError,
    execute_tool_call,
    git_diff,
    git_status,
    list_directory,
    read_file,
    resolve_within,
    search_files,
    write_file,
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# demo\n")
    return tmp_path


def test_resolve_within_accepts_a_nested_relative_path(project):
    resolved = resolve_within(project, "src/app.py")
    assert resolved == (project / "src" / "app.py").resolve()


def test_resolve_within_rejects_parent_traversal(project):
    with pytest.raises(PathEscapeError):
        resolve_within(project, "../outside.txt")


def test_resolve_within_rejects_nested_parent_traversal(project):
    with pytest.raises(PathEscapeError):
        resolve_within(project, "src/../../outside.txt")


def test_resolve_within_rejects_absolute_paths_outside_root(project, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    with pytest.raises(PathEscapeError):
        resolve_within(project, str(other))


def test_read_file_returns_contents(project):
    assert read_file(project, "src/app.py") == "print('hi')\n"


def test_read_file_missing_raises(project):
    with pytest.raises(FileNotFoundError):
        read_file(project, "src/missing.py")


def test_write_file_creates_parent_dirs(project):
    write_file(project, "new/dir/out.txt", "hello")
    assert (project / "new" / "dir" / "out.txt").read_text() == "hello"


def test_write_file_overwrites_existing(project):
    write_file(project, "README.md", "# replaced\n")
    assert (project / "README.md").read_text() == "# replaced\n"


def test_list_directory_lists_root_by_default(project):
    entries = list_directory(project, ".")
    assert "README.md" in entries
    assert "src/" in entries


def test_execute_tool_call_read_file_success(project):
    result = execute_tool_call(project, "call-1", "read_file", {"path": "README.md"})
    assert result.content == "# demo\n"
    assert not result.is_error


def test_execute_tool_call_path_escape_is_reported_not_raised(project):
    result = execute_tool_call(project, "call-1", "read_file", {"path": "../outside.txt"})
    assert result.is_error
    assert "resolves outside" in result.content


def test_execute_tool_call_unknown_tool_is_reported_not_raised(project):
    result = execute_tool_call(project, "call-1", "delete_everything", {})
    assert result.is_error
    assert "Unknown tool" in result.content


# ------------------------------------------------------------------------- ignore


def test_read_file_refuses_a_gitignored_path(project):
    (project / ".gitignore").write_text("*.secret\n")
    (project / "creds.secret").write_text("shh")
    with pytest.raises(IgnoredPathError):
        read_file(project, "creds.secret")


def test_write_file_refuses_a_localgateignored_path(project):
    (project / ".localgateignore").write_text(".env\n")
    with pytest.raises(IgnoredPathError):
        write_file(project, ".env", "SECRET=1")
    assert not (project / ".env").exists()


def test_list_directory_hides_ignored_entries(project):
    (project / ".gitignore").write_text("*.secret\n")
    (project / "creds.secret").write_text("shh")
    entries = list_directory(project, ".")
    assert "creds.secret" not in entries
    assert "README.md" in entries


def test_read_file_inside_dot_git_is_always_refused(project):
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("[core]")
    with pytest.raises(IgnoredPathError):
        read_file(project, ".git/config")


def test_execute_tool_call_reports_ignored_paths_not_raises(project):
    (project / ".gitignore").write_text("*.secret\n")
    (project / "creds.secret").write_text("shh")
    result = execute_tool_call(project, "call-1", "read_file", {"path": "creds.secret"})
    assert result.is_error
    assert "excluded by" in result.content


# -------------------------------------------------------------------- search_files


def test_search_files_finds_matching_lines(project):
    matches = search_files(project, "print")
    assert matches == ["src/app.py:1: print('hi')"]


def test_search_files_respects_gitignore(project):
    (project / ".gitignore").write_text("*.secret\n")
    (project / "creds.secret").write_text("print('leak')\n")
    matches = search_files(project, "print")
    assert not any("creds.secret" in m for m in matches)


def test_search_files_skips_binary_files(project):
    (project / "binary.dat").write_bytes(b"\x00\x01\x02print\x03")
    matches = search_files(project, "print")
    assert not any("binary.dat" in m for m in matches)


def test_search_files_returns_no_matches_message_via_execute(project):
    result = execute_tool_call(project, "c1", "search_files", {"pattern": "nope-not-here"})
    assert result.content == "No matches."
    assert not result.is_error


# --------------------------------------------------------------------- git tools


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_git_status_outside_a_repo(project):
    assert git_status(project) == "Not a git repository."


def test_git_status_inside_a_clean_repo(project):
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "initial")
    assert git_status(project) == "Working tree clean."


def test_git_status_shows_a_modified_file(project):
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "initial")
    (project / "README.md").write_text("changed\n")
    assert "README.md" in git_status(project)


def test_git_diff_outside_a_repo(project):
    assert git_diff(project) == "Not a git repository."


def test_git_diff_shows_pending_changes(project):
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "initial")
    (project / "README.md").write_text("changed\n")
    diff = git_diff(project, "README.md")
    assert "+changed" in diff
