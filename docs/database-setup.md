# Database Setup

localgate runs on SQLite with zero configuration, and on Postgres (including Neon) by
changing one environment variable. The schema and every query are identical on both.

## SQLite (the default)

Nothing to do. The database is a file, created on first run:

```bash
LOCALGATE_DATABASE_URL=sqlite+aiosqlite:///./localgate.db
```

**When SQLite stops being the right answer.** Memory retrieval scans every chunk in a
session and scores it in Python. That is comfortably fine into the low thousands of
chunks per session and stops being fine well before ten thousand. `GET /health` warns
you that you're in this mode. The other limit is concurrency: SQLite serializes writers,
so a busy multi-user gateway will contend on writes.

## PostgreSQL

```bash
LOCALGATE_DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/localgate
```

Then:

```bash
uv run localgate db upgrade
```

A ready-made local Postgres is in `examples/docker-compose.postgres.yml`.

### The driver must be async

This is the single most common mistake:

```bash
# WRONG — selects a synchronous driver localgate does not install
LOCALGATE_DATABASE_URL=postgresql://user:pass@host/db

# RIGHT
LOCALGATE_DATABASE_URL=postgresql+asyncpg://user:pass@host/db
```

The gateway is async from top to bottom. A plain `postgresql://` URL makes SQLAlchemy
reach for psycopg2, which isn't installed — so you get `ModuleNotFoundError` rather than
anything that sounds like a configuration problem. `PUT /admin/config/database-url`
detects exactly this case and tells you to add `+asyncpg`.

## Neon

Neon is standard Postgres, so it works — with two wrinkles that are worth knowing before
they cost you an afternoon.

### 1. `sslmode` is spelled `ssl` for asyncpg

The connection string Neon gives you to copy uses the psycopg spelling:

```bash
# Neon's copy-paste string — asyncpg rejects this with a bare TypeError
postgresql://user:pass@ep-xxx.neon.tech/db?sslmode=require

# What asyncpg wants
postgresql+asyncpg://user:pass@ep-xxx.neon.tech/db?ssl=require
```

Same meaning, different keyword. `PUT /admin/config/database-url` catches this and says
so explicitly, because the raw error (`TypeError: connect() got an unexpected keyword
argument 'sslmode'`) gives no clue that the fix is a one-word rename.

### 2. The pooled endpoint and prepared statements

If your hostname contains `-pooler`, you are talking to PgBouncer in transaction-pooling
mode. PgBouncer swaps the underlying connection between queries, which invalidates
asyncpg's per-connection prepared-statement cache — so queries start failing
*unpredictably*, in a way that looks like a flaky network rather than a config problem.

localgate handles this for you: for any `postgresql+asyncpg` URL it disables the
statement cache and uses `NullPool`, letting PgBouncer own the pooling instead of
stacking a second pool on top of it. This costs nothing against a direct connection, so
it is applied unconditionally rather than by sniffing the hostname (see
`db/engine.py::make_engine`).

Either endpoint works. The pooled one is the better choice for a gateway that opens many
short-lived connections.

## Establishing a database through the admin UI

`PUT /admin/config/database-url` (and the dashboard's Database tab) does something the
env var can't: it **proves the connection works before saving it**.

```bash
curl -X PUT http://localhost:8000/admin/config/database-url \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"database_url": "postgresql+asyncpg://user:pass@host/localgate?ssl=require"}'
```

On success the URL is written to `localgate.config.json` and **takes precedence over
`LOCALGATE_DATABASE_URL`** on the next start. That precedence is the whole point: a URL
that has been connection-tested outranks a string someone typed into a file. This is
what "established" means, as opposed to merely "configured".

A restart is required. Swapping the database live would mean rebuilding the engine and
re-running migrations underneath in-flight requests, and there's no version of that which
is safe. The response says `"restart_required": true` rather than pretending otherwise.

Once you restart, **all** of localgate's data — keys, usage, conversation history, memory
chunks — lives in the new database. There is one engine behind every table, so this is
never a partial move.

## Migrations

The schema is versioned with Alembic. There is no `create_all` path: if the schema could
be built two ways, the migration history and the ORM models would drift, and the person
who found out would be someone upgrading a database that had never been migrated.

```bash
localgate db upgrade    # apply pending migrations (safe to re-run; no-ops when current)
localgate db current    # print the revision the database is at
localgate db init       # alias for upgrade, for a fresh database
```

The server also runs pending migrations on startup, so a normal deploy needs no extra step.

### Upgrading from localgate 0.1–0.2

Those versions created their schema with `create_all` and left no `alembic_version`
behind. Running migrations against such a database naively would try to
`CREATE TABLE api_keys` a second time and fail on startup.

localgate detects this — tables present, no version stamp — and adopts the database by
stamping it at revision `0001` (which reproduces exactly what `create_all` built), then
migrating it forward. **Your existing keys, usage and conversations are preserved.** No
manual step is needed; this is covered by
`tests/integration/test_migrations.py::test_pre_migrations_database_is_adopted_without_losing_data`.

The one thing that cannot be recovered is `key_prefix` for keys created before the
upgrade — the raw key is gone by design, so old keys show a blank prefix in listings.
They keep working.

## Moving between databases

`GET /admin/export` returns everything as JSON. Point localgate at the new database, run
`db upgrade`, and reload. Nobody should feel locked into a self-hosted tool.

## Vector search and pgvector

Embeddings are stored as JSON float arrays and similarity is computed in Python, so the
schema is identical on SQLite and Postgres and neither needs an extension installed. That
tradeoff — portability now, a scan later — is recorded in
[docs/decisions/0002](decisions/0002-json-embeddings-over-pgvector.md), which also
describes the pgvector upgrade path for Postgres users who outgrow it. The repository
method signature is designed not to change when they do.
