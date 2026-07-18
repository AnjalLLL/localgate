"""Golden-file-style tests for diff line classification: a known before/after pair
must always render the same (line, style) sequence.
"""

from localgate.agent.render import diff_lines


def test_added_lines_are_styled_green():
    lines = diff_lines("a.py", "one\n", "one\ntwo\n")
    added = [text for text, style in lines if style == "green"]
    assert added == ["+two"]


def test_removed_lines_are_styled_red():
    lines = diff_lines("a.py", "one\ntwo\n", "one\n")
    removed = [text for text, style in lines if style == "red"]
    assert removed == ["-two"]


def test_hunk_headers_are_styled_cyan():
    lines = diff_lines("a.py", "one\n", "two\n")
    headers = [text for text, style in lines if style == "cyan"]
    assert len(headers) == 1
    assert headers[0].startswith("@@")


def test_file_headers_are_styled_dim():
    lines = diff_lines("a.py", "one\n", "two\n")
    dim = [text for text, style in lines if style == "dim"]
    assert any(text.startswith("---") for text in dim)
    assert any(text.startswith("+++") for text in dim)


def test_identical_content_produces_no_diff_lines():
    assert diff_lines("a.py", "same\n", "same\n") == []


def test_full_replacement_shows_removals_and_additions():
    lines = diff_lines("a.py", "old\n", "new\n")
    texts = [text for text, _ in lines]
    assert "-old" in texts
    assert "+new" in texts
