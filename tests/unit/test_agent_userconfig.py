"""`UserConfig` — persisted `localgate code` preferences (theme, default_model,
auto_approve, max_turns) round-tripped through the hand-rolled TOML subset.
"""

from localgate.agent.userconfig import UserConfig


def test_load_missing_file_returns_defaults(tmp_path):
    config = UserConfig.load(tmp_path / "missing.toml")
    assert config == UserConfig()


def test_save_then_load_round_trips_all_fields(tmp_path):
    path = tmp_path / "config.toml"
    original = UserConfig(
        theme="light", default_model="qwen2.5-coder:7b", auto_approve=True, max_turns=30
    )
    original.save(path)
    assert UserConfig.load(path) == original


def test_save_then_load_round_trips_none_default_model(tmp_path):
    path = tmp_path / "config.toml"
    UserConfig().save(path)
    assert UserConfig.load(path) == UserConfig()


def test_load_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\nsome_future_key = "x"\n', encoding="utf-8")
    config = UserConfig.load(path)
    assert config.theme == "dark"


def test_load_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '# a comment\n\ntheme = "light"\n\n# another\nmax_turns = 5\n', encoding="utf-8"
    )
    config = UserConfig.load(path)
    assert config.theme == "light"
    assert config.max_turns == 5


def test_with_override_returns_a_new_instance(tmp_path):
    original = UserConfig()
    changed = original.with_override(theme="light")
    assert changed.theme == "light"
    assert original.theme == "dark"


def test_dump_is_readable_key_value_toml(tmp_path):
    path = tmp_path / "config.toml"
    UserConfig(theme="dark", auto_approve=True, max_turns=20).save(path)
    text = path.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert "auto_approve = true" in text
    assert "max_turns = 20" in text
