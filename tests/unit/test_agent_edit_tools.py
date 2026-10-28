"""edit_file, paged read_file, argument validation and newline preservation —
the tools behave the same on every OS."""

from localgate.agent.tools import (
    READ_PAGE_LINES,
    TOOL_SCHEMAS,
    execute_tool_call,
    validate_arguments,
)


def run(root, name, **arguments):
    return execute_tool_call(root, "id", name, arguments)


def test_edit_file_replaces_a_unique_snippet(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    result = run(tmp_path, "edit_file", path="a.py", old_string="y = 2", new_string="y = 3")
    assert not result.is_error
    assert (tmp_path / "a.py").read_text() == "x = 1\ny = 3\n"


def test_edit_file_refuses_an_ambiguous_snippet_unless_replace_all(tmp_path):
    (tmp_path / "a.py").write_text("v = 0\nv = 0\n")
    result = run(tmp_path, "edit_file", path="a.py", old_string="v = 0", new_string="v = 1")
    assert result.is_error and "2 times" in result.content
    result = run(
        tmp_path, "edit_file", path="a.py", old_string="v = 0", new_string="v = 1", replace_all=True
    )
    assert not result.is_error
    assert (tmp_path / "a.py").read_text() == "v = 1\nv = 1\n"


def test_edit_file_missing_snippet_points_at_a_similar_line(tmp_path):
    (tmp_path / "a.py").write_text("def main():\n    return  1\n")
    result = run(
        tmp_path, "edit_file", path="a.py", old_string="    return 1", new_string="    return 2"
    )
    assert result.is_error
    assert "not found" in result.content


def test_edit_file_keeps_crlf_line_endings(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"one\r\ntwo\r\n")
    run(tmp_path, "edit_file", path="w.txt", old_string="two\n", new_string="2\nthree\n")
    assert (tmp_path / "w.txt").read_bytes() == b"one\r\n2\r\nthree\r\n"


def test_write_file_keeps_crlf_and_writes_lf_exactly_for_new_files(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"old\r\n")
    run(tmp_path, "write_file", path="w.txt", content="new\nlines\n")
    assert (tmp_path / "w.txt").read_bytes() == b"new\r\nlines\r\n"
    run(tmp_path, "create_file", path="n.txt", content="a\nb\n")
    assert (tmp_path / "n.txt").read_bytes() == b"a\nb\n"


def test_edit_file_cannot_escape_the_project(tmp_path):
    result = run(tmp_path, "edit_file", path="../x.py", old_string="a", new_string="b")
    assert result.is_error


def test_read_file_pages_long_files(tmp_path):
    lines = "".join(f"line {i}\n" for i in range(1, READ_PAGE_LINES + 51))
    (tmp_path / "big.txt").write_text(lines)
    first = run(tmp_path, "read_file", path="big.txt").content
    assert first.startswith("line 1\n")
    assert f"offset={READ_PAGE_LINES + 1}" in first
    second = run(tmp_path, "read_file", path="big.txt", offset=READ_PAGE_LINES + 1).content
    assert second.startswith(f"line {READ_PAGE_LINES + 1}\n")
    assert "for more" not in second


def test_read_file_returns_short_files_unchanged(tmp_path):
    (tmp_path / "s.txt").write_text("a\nb\n")
    assert run(tmp_path, "read_file", path="s.txt").content == "a\nb\n"


def test_validation_reports_missing_and_mistyped_arguments(tmp_path):
    result = run(tmp_path, "edit_file", path="a.py", old_string="x")
    assert result.is_error and "missing required 'new_string'" in result.content
    result = run(tmp_path, "read_file", path="a.py", offset="ten")
    assert result.is_error and "'offset' must be integer" in result.content


def test_validate_arguments_rejects_booleans_for_integers():
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "read_file")
    assert validate_arguments(schema, {"path": "a", "limit": True}) is not None
    assert validate_arguments(schema, {"path": "a", "limit": 5}) is None


def test_ambiguous_edit_lists_each_match_with_context(tmp_path):
    (tmp_path / "c.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a + b\n", newline="\n"
    )
    result = run(tmp_path, "edit_file", path="c.py", old_string="return a + b", new_string="x")
    assert result.is_error
    assert "line 2" in result.content and "line 6" in result.content
    assert "def sub(a, b):" in result.content
    assert (tmp_path / "c.py").read_text().count("return a + b") == 2


def test_ambiguous_edit_offers_a_complete_corrected_call(tmp_path):
    import json as _json  # noqa: PLC0415

    (tmp_path / "c.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a + b\n", newline="\n"
    )
    error = run(
        tmp_path, "edit_file", path="c.py", old_string="return a + b", new_string="return a - b"
    )
    last = error.content.splitlines()[-1]
    decoder = _json.JSONDecoder()
    old_part = last.split("use old_string ", 1)[1]
    old, end = decoder.raw_decode(old_part)
    new, _ = decoder.raw_decode(old_part[end:].split("new_string ", 1)[1])
    assert old.startswith("def sub(a, b):") and new == old.replace("a + b", "a - b")
    assert not run(tmp_path, "edit_file", path="c.py", old_string=old, new_string=new).is_error
    text = (tmp_path / "c.py").read_text()
    assert "def add(a, b):\n    return a + b" in text and "def sub(a, b):\n    return a - b" in text


def test_read_file_on_a_directory_points_to_list_directory(tmp_path):
    (tmp_path / "tests").mkdir()
    result = run(tmp_path, "read_file", path="tests")
    assert result.is_error and "list_directory" in result.content


def test_a_single_escaped_newline_inside_code_is_left_alone():
    from localgate.agent.tools import unescape_text_arguments  # noqa: PLC0415

    args = {"content": 'print("a\\nb")'}
    assert unescape_text_arguments(args) == args
