"""git-backed safety net: dirty-tree detection, commit, and the two `/undo` paths
(delete an untracked file vs. check out a tracked one back to its last commit).
"""

import subprocess

import pytest

from localgate.agent import gitutil


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("original\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def test_is_repo_true_inside_a_repo(repo):
    assert gitutil.is_repo(repo) is True


def test_is_repo_false_outside_a_repo(tmp_path_factory):
    plain = tmp_path_factory.mktemp("not-a-repo")
    assert gitutil.is_repo(plain) is False


def test_is_dirty_false_on_a_clean_checkout(repo):
    assert gitutil.is_dirty(repo) is False


def test_is_dirty_true_after_an_edit(repo):
    (repo / "tracked.py").write_text("changed\n")
    assert gitutil.is_dirty(repo) is True


def test_commit_all_commits_pending_changes(repo):
    (repo / "tracked.py").write_text("changed\n")
    committed = gitutil.commit_all(repo, "localgate-agent: test change")
    assert committed is True
    assert gitutil.is_dirty(repo) is False
    assert gitutil.last_commit_message(repo) == "localgate-agent: test change"


def test_commit_all_returns_false_when_nothing_changed(repo):
    assert gitutil.commit_all(repo, "localgate-agent: nothing") is False


def test_undo_file_deletes_a_brand_new_untracked_file(repo):
    (repo / "new_file.py").write_text("brand new\n")
    message = gitutil.undo_file(repo, "new_file.py")
    assert not (repo / "new_file.py").exists()
    assert "Deleted" in message


def test_undo_file_reverts_a_tracked_file_to_last_commit(repo):
    (repo / "tracked.py").write_text("changed\n")
    message = gitutil.undo_file(repo, "tracked.py")
    assert (repo / "tracked.py").read_text() == "original\n"
    assert "Reverted" in message


def test_reset_hard_last_drops_the_last_commit(repo):
    (repo / "tracked.py").write_text("changed\n")
    gitutil.commit_all(repo, "localgate-agent: change")
    gitutil.reset_hard_last(repo)
    assert (repo / "tracked.py").read_text() == "original\n"
    assert gitutil.last_commit_message(repo) == "initial"


def test_last_commit_message_none_with_no_commits(tmp_path):
    _git(tmp_path, "init", "-q")
    assert gitutil.last_commit_message(tmp_path) is None


def test_diff_shows_pending_changes(repo):
    (repo / "tracked.py").write_text("changed\n")
    output = gitutil.diff(repo, "tracked.py")
    assert "-original" in output
    assert "+changed" in output
