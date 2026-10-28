"""The CLI.

The CLI talks to the database directly rather than to a running server, because
`keys create` has to work before you have a key and `db upgrade` has to work when
the server won't start. These tests drive it the same way, against a temp database.
"""

import json

import pytest
from typer.testing import CliRunner

from localgate.cli import app

runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point the CLI at a scratch database, and away from the developer's real one.

    Both matter: LOCALGATE_DATABASE_URL for .env, and the cwd for the
    localgate.config.json that an admin-established database would live in.
    """
    monkeypatch.setenv("LOCALGATE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/cli.db")
    monkeypatch.setenv("LOCALGATE_BACKEND_TYPE", "fake")
    monkeypatch.setenv("LOCALGATE_ADMIN_KEY", "cli-admin-key")
    monkeypatch.chdir(tmp_path)


def test_version(env):
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_version_flag(env):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_version_flag_matches_the_version_subcommand(env):
    flag_result = runner.invoke(app, ["--version"])
    subcommand_result = runner.invoke(app, ["version"])
    assert flag_result.stdout == subcommand_result.stdout


def test_backends_lists_the_installed_adapters(env):
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0
    assert "ollama" in result.stdout
    assert "vllm" in result.stdout


def test_db_upgrade_then_current(env):
    upgrade = runner.invoke(app, ["db", "upgrade"])
    assert upgrade.exit_code == 0
    assert "up to date" in upgrade.stdout

    current = runner.invoke(app, ["db", "current"])
    assert current.exit_code == 0
    assert "0009" in current.stdout


def test_db_current_on_an_unmigrated_database_says_what_to_do(env):
    result = runner.invoke(app, ["db", "current"])
    assert result.exit_code == 1
    assert "localgate db upgrade" in result.stdout


def test_keys_create_prints_the_key_once_and_says_so(env):
    result = runner.invoke(app, ["keys", "create", "--name", "my-app", "--rate-limit", "30"])
    assert result.exit_code == 0
    assert "lg_" in result.stdout
    assert "30/min" in result.stdout
    assert "cannot be shown again" in result.stdout


def test_keys_create_works_on_a_fresh_database_without_db_upgrade_first(env):
    """Creating your first key is how you start — it must not fail because nobody has
    run `db upgrade` yet."""
    result = runner.invoke(app, ["keys", "create", "--name", "first"])
    assert result.exit_code == 0
    assert "lg_" in result.stdout


def test_keys_list_shows_the_created_key(env):
    runner.invoke(app, ["keys", "create", "--name", "listed-app"])

    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "listed-app" in result.stdout
    assert "active" in result.stdout


def test_keys_list_on_an_empty_database_explains_the_next_step(env):
    runner.invoke(app, ["db", "upgrade"])

    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "localgate keys create" in result.stdout


def test_keys_revoke(env):
    import json

    assert runner.invoke(app, ["keys", "create", "--name", "doomed"]).exit_code == 0
    key_id = json.loads(runner.invoke(app, ["keys", "list", "--json"]).stdout)[0]["id"]

    assert runner.invoke(app, ["keys", "revoke", key_id]).exit_code == 0

    after = json.loads(runner.invoke(app, ["keys", "list", "--json"]).stdout)
    assert after[0]["revoked"] is True


def test_revoking_an_unknown_key_fails_loudly(env):
    runner.invoke(app, ["db", "upgrade"])

    result = runner.invoke(app, ["keys", "revoke", "no-such-key"])
    assert result.exit_code == 1


def test_running_against_an_unmigrated_database_says_how_to_fix_it(env):
    """The most likely failure on a fresh install deserves a sentence, not a
    SQLAlchemy traceback."""
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 1
    assert "localgate db upgrade" in result.output
    assert "Traceback" not in result.output


def test_health_reports_backend_and_database(env):
    runner.invoke(app, ["db", "upgrade"])

    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "backend" in result.stdout
    assert "database" in result.stdout


def test_health_exits_nonzero_when_the_backend_is_unreachable(env, monkeypatch):
    """The exit code is the point: `localgate health` belongs in a deploy script."""
    monkeypatch.setenv("LOCALGATE_BACKEND_TYPE", "ollama")
    monkeypatch.setenv("LOCALGATE_BACKEND_URL", "http://127.0.0.1:1")  # nothing listens here
    runner.invoke(app, ["db", "upgrade"])

    result = runner.invoke(app, ["health"])
    assert result.exit_code == 1
    assert "unreachable" in result.stdout


def test_a_bad_config_is_a_readable_message_not_a_traceback(env, monkeypatch):
    monkeypatch.setenv("LOCALGATE_ENVIRONMENT", "production")
    monkeypatch.setenv("LOCALGATE_ADMIN_KEY", "change-me-in-production")

    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 2
    assert "Configuration error" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------- admin key


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A pip install's first run: no LOCALGATE_ADMIN_KEY anywhere."""
    monkeypatch.setenv("LOCALGATE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/cli.db")
    monkeypatch.setenv("LOCALGATE_BACKEND_TYPE", "fake")
    monkeypatch.chdir(tmp_path)


def test_first_serve_generates_and_saves_a_private_admin_key(fresh, monkeypatch):
    from localgate import paths
    from localgate.config import INSECURE_ADMIN_KEY

    started: list[str] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: started.append(app.state.settings.admin_key)
    )
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert "generated an admin key" in result.stdout
    saved = paths.user_env_file().read_text()
    (key,) = [
        line.split("=", 1)[1]
        for line in saved.splitlines()
        if line.startswith("LOCALGATE_ADMIN_KEY=")
    ]
    assert key in result.stdout
    assert started == [key] and key != INSECURE_ADMIN_KEY

    # Second run: the saved key is reused, not replaced or printed again.
    result = runner.invoke(app, ["serve"])
    assert "generated an admin key" not in result.stdout
    assert started == [key, key]


def test_serve_leaves_an_explicitly_configured_key_alone(fresh, monkeypatch):
    from localgate import paths

    monkeypatch.setenv("LOCALGATE_ADMIN_KEY", "chosen-by-the-operator")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    result = runner.invoke(app, ["serve"])
    assert "generated an admin key" not in result.stdout
    assert not paths.user_env_file().exists()


def test_admin_key_shows_the_key_and_where_it_comes_from(fresh, monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    runner.invoke(app, ["serve"])
    result = runner.invoke(app, ["admin-key"])
    assert result.exit_code == 0
    assert result.stdout.strip() and "localgate.env" in result.stderr


def test_admin_key_before_any_setup_says_how_to_create_one(fresh):
    result = runner.invoke(app, ["admin-key"])
    assert result.exit_code == 1
    assert "localgate serve" in result.stdout


def test_admin_key_rotate_replaces_the_saved_key(fresh, monkeypatch):
    from localgate import paths

    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    runner.invoke(app, ["serve"])
    before = paths.user_env_file().read_text()
    result = runner.invoke(app, ["admin-key", "--rotate"])
    assert result.exit_code == 0
    after = paths.user_env_file().read_text()
    assert before != after and after.count("LOCALGATE_ADMIN_KEY=") == 1


def test_admin_key_rotate_refuses_when_the_key_is_set_elsewhere(fresh, monkeypatch):
    monkeypatch.setenv("LOCALGATE_ADMIN_KEY", "from-the-environment")
    result = runner.invoke(app, ["admin-key", "--rotate"])
    assert result.exit_code == 2
    assert "environment variable" in result.stderr


# ---------------------------------------------------------------- accounts


def test_users_create_list_passwd_delete(env):
    created = runner.invoke(
        app,
        ["users", "create", "--email", "Ana@Example.com"],
        input="correct horse\ncorrect horse\n",
    )
    assert created.exit_code == 0, created.output
    assert "ana@example.com" in runner.invoke(app, ["users", "list"]).stdout

    duplicate = runner.invoke(
        app, ["users", "create", "--email", "ana@example.com"], input="another one\nanother one\n"
    )
    assert duplicate.exit_code == 1

    changed = runner.invoke(
        app, ["users", "passwd", "ana@example.com"], input="new password\nnew password\n"
    )
    assert changed.exit_code == 0
    assert runner.invoke(app, ["users", "delete", "ana@example.com"]).exit_code == 0
    assert "No accounts yet" in runner.invoke(app, ["users", "list"]).stdout


def test_users_create_rejects_a_short_password(env):
    result = runner.invoke(
        app,
        ["users", "create", "--email", "ana@example.com"],
        input="short\nshort\ncorrect horse\ncorrect horse\n",
    )
    assert result.exit_code == 0
    assert "at least 8 characters" in result.stdout


# ---------------------------------------------------------------- key limits


def test_keys_create_and_update_set_limits(env):
    created = runner.invoke(
        app,
        [
            "keys", "create", "--name", "phone", "--models", "llama3,qwen2.5-coder:7b",
            "--max-tokens", "256", "--quota", "100000", "--quota-period", "day",
            "--expires", "30d", "--no-store-conversations",
        ],
    )  # fmt: skip
    assert created.exit_code == 0, created.output
    assert "models llama3,qwen2.5-coder:7b" in created.stdout
    assert "100,000 tokens/day" in created.stdout
    assert "conversations not stored" in created.stdout

    rows = json.loads(runner.invoke(app, ["keys", "list", "--json"]).stdout)
    key_id = rows[0]["id"]
    updated = runner.invoke(
        app,
        ["keys", "update", key_id, "--models", "any", "--max-tokens", "0", "--expires", "never"],
    )
    assert updated.exit_code == 0, updated.output
    assert "models" not in updated.stdout and "max 256" not in updated.stdout
    assert "100,000 tokens/day" in updated.stdout  # untouched


def test_keys_update_with_nothing_to_change_says_so(env):
    runner.invoke(app, ["keys", "create", "--name", "x"])
    key_id = json.loads(runner.invoke(app, ["keys", "list", "--json"]).stdout)[0]["id"]
    result = runner.invoke(app, ["keys", "update", key_id])
    assert result.exit_code == 2


def test_bad_expiry_is_rejected(env):
    result = runner.invoke(app, ["keys", "create", "--name", "x", "--expires", "soonish"])
    assert result.exit_code != 0


# ---------------------------------------------------------------- sharing


def test_serve_share_listens_on_the_network_and_prints_addresses(fresh, monkeypatch):
    started: list[dict] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: started.append(kwargs))
    monkeypatch.setattr(
        "localgate.cli.share_urls",
        lambda port, public_url=None: [
            {"label": "Local network", "url": f"http://192.168.1.20:{port}"}
        ],
    )
    result = runner.invoke(app, ["serve", "--share"])
    assert result.exit_code == 0, result.output
    assert started[0]["host"] == "0.0.0.0"
    assert "http://192.168.1.20:8000/v1" in result.stdout
    assert "/dashboard/#/share" in result.stdout


def test_serve_refuses_the_network_with_the_placeholder_key(fresh, monkeypatch):
    from localgate.config import INSECURE_ADMIN_KEY

    monkeypatch.setenv("LOCALGATE_ADMIN_KEY", INSECURE_ADMIN_KEY)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: pytest.fail("must not start"))
    result = runner.invoke(app, ["serve", "--host", "192.168.1.20"])
    assert result.exit_code == 2
    assert "placeholder admin key" in result.stderr


def test_compose_deploy_reaches_the_gateway_from_caddy(env, tmp_path):
    out = tmp_path / "deploy"
    result = runner.invoke(
        app,
        ["deploy", "--domain", "gw.example.com", "--target", "compose", "--output-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "reverse_proxy localgate:8000" in (out / "Caddyfile").read_text()
    assert "LOCALGATE_HOST=0.0.0.0" in (out / "localgate.env").read_text()
    assert "--proxy-headers" in (out / "docker-compose.yml").read_text()
