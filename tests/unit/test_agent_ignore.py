from localgate.agent.ignore import is_ignored, load_patterns


def test_git_directory_is_always_ignored_even_without_any_ignore_file(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("")
    patterns = load_patterns(tmp_path)
    assert is_ignored(tmp_path, tmp_path / ".git" / "config", patterns)


def test_gitignore_simple_filename_pattern(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "debug.log").write_text("")
    (tmp_path / "app.py").write_text("")
    patterns = load_patterns(tmp_path)
    assert is_ignored(tmp_path, tmp_path / "debug.log", patterns)
    assert not is_ignored(tmp_path, tmp_path / "app.py", patterns)


def test_gitignore_directory_pattern_covers_nested_files(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n")
    nested = tmp_path / "node_modules" / "pkg" / "index.js"
    nested.parent.mkdir(parents=True)
    nested.write_text("")
    patterns = load_patterns(tmp_path)
    assert is_ignored(tmp_path, nested, patterns)


def test_gitignore_anchored_pattern_only_matches_at_root(tmp_path):
    (tmp_path / ".gitignore").write_text("/build\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.txt").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "build").mkdir()
    (tmp_path / "src" / "build" / "out.txt").write_text("")
    patterns = load_patterns(tmp_path)
    assert is_ignored(tmp_path, tmp_path / "build" / "out.txt", patterns)
    assert not is_ignored(tmp_path, tmp_path / "src" / "build" / "out.txt", patterns)


def test_localgateignore_is_read_alongside_gitignore(tmp_path):
    (tmp_path / ".localgateignore").write_text(".env\n")
    (tmp_path / ".env").write_text("SECRET=1")
    patterns = load_patterns(tmp_path)
    assert is_ignored(tmp_path, tmp_path / ".env", patterns)


def test_negation_lines_are_skipped_not_honored(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n")
    (tmp_path / "keep.log").write_text("")
    patterns = load_patterns(tmp_path)
    # keep.log still matches *.log; the negation line itself contributes nothing
    assert is_ignored(tmp_path, tmp_path / "keep.log", patterns)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    (tmp_path / ".gitignore").write_text("# a comment\n\n*.log\n")
    (tmp_path / "app.py").write_text("")
    patterns = load_patterns(tmp_path)
    assert not is_ignored(tmp_path, tmp_path / "app.py", patterns)


def test_no_ignore_files_means_nothing_is_ignored_except_git(tmp_path):
    (tmp_path / "app.py").write_text("")
    patterns = load_patterns(tmp_path)
    assert patterns == []
    assert not is_ignored(tmp_path, tmp_path / "app.py", patterns)
