# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.1] — 2026-10-29

### Fixed

- **The coding agent no longer loops on "a file must be read before write_file".** What the
  session had read was forgotten at every new message, so a file read in one turn was refused
  again in the next; small models echoed the refusal and stopped, turn after turn. The session
  now remembers each file's content hash across turns: a file it has seen and nobody has
  changed since can be rewritten directly, while one changed in between (e.g. by you) must
  still be read again. Paths are compared normalised (`./src/a.html` = `src/a.html`).

## [1.1.0] — 2026-10-29 — share with one command, from anywhere

`localgate serve` now shares with your network by default, the dashboard can give the gateway
an internet address with one switch, each key chooses local network, internet or both, and a
device connects by running `localgate code` and answering two questions.

**Upgrading:** migration `0010` runs automatically. `localgate serve` now listens on your network
by default — use `--local` (or set `LOCALGATE_HOST=127.0.0.1`) to keep the old behaviour.

### Added

- **Sharing needs no flag:** `localgate serve` shares with your network by default;
  `--local` keeps it to this machine (an explicit `LOCALGATE_HOST` is still respected).
- **Internet access, built in:** `localgate serve --public`, or the **Internet access**
  switch on Dashboard → Share, opens a free Cloudflare quick tunnel — an
  `https://….trycloudflare.com` address that works from any network, no account or router
  setup. `cloudflared` is used if installed, otherwise downloaded once from Cloudflare's
  release and checked against its published SHA-256. Behind the tunnel the real visitor
  address (`CF-Connecting-IP`) is used for every trust decision.
- **Per-key network access** (migration `0010`): a key works on the **local network**,
  over the **internet**, or **both** — `403 network_not_allowed` elsewhere. Set it on the
  Share page, the key editor, `--network local|internet|both`, or the API; the Share page
  hands an internet-only key the internet address.
- **Pick a key's models with checkboxes** on the Share page and in the key editor.
- LAN discovery is built in (zeroconf is now a regular dependency).

- **Connecting a device takes one command.** `localgate code` on a machine without a model
  server asks where models should run; choosing a shared gateway asks for its address
  (`192.168.1.20` is enough — gateways announced on the network are offered by number) and
  the API key, checks them, saves them, and starts the session on the gateway. Inside a
  session, `/connect` moves to a gateway and `/disconnect` back to this machine; the banner
  shows which gateway is in use. `localgate logout` forgets it.
- Connection errors say what to do — e.g. "refused to connect" now explains that the GPU
  machine must run `localgate serve --share`.

### Changed

- `localgate login --url` accepts a bare address (`192.168.1.20`, `gpu-box:9000`).
- On a gateway, `localgate code` keeps your default model when the gateway has it (`llama3`
  matches `llama3:latest`), else picks one your key may use, preferring tool-capable models.

## [1.0.0] — 2026-10-29 — share your GPU

localgate was built for one idea: a machine with a good GPU runs the models, and every
other device — laptops, phones, apps, teammates — uses them through it with its own API
key. 1.0 makes that real. `localgate serve --share` puts the gateway on the network, the
dashboard's Share page hands a device a limited key with a QR code, a fair queue shares the
GPU, the admin side stays on your own network, and several GPUs or machines can sit behind
one gateway.

**Upgrading from 0.x:** migrations `0008` (key limits) and `0009` (audit log) run
automatically. Two defaults change — see *Changed*: the dashboard, `/admin`, the API docs
and `/metrics` now answer only private networks, and `/health` shows details only to them.
Behind a reverse proxy, start with `--proxy-headers --forwarded-allow-ips <proxy>`.

### Added

- **`localgate serve --share`**: listens on the network and prints the LAN and Tailscale
  addresses devices should use. `LOCALGATE_PUBLIC_URL` names a tunnel or domain address.
- **Per-key limits**: allowed models, a `max_tokens` cap, a token quota per UTC day or
  month, an expiry date, requests at once, and whether conversations are stored. Set them
  when creating a key or change them later (dashboard edit dialog, `localgate keys update`,
  `PATCH /admin/keys/{id}` — a partial update where `null` clears a limit). Refusals are
  `403 model_not_allowed`, `429 quota_exceeded` + `Retry-After`, `401 key_expired`.
  `/v1/models` lists only the models a key may use; `/v1/usage` and `localgate whoami`
  show a key's limits and usage this period.
- **Fair GPU queue**: `LOCALGATE_MAX_CONCURRENT_REQUESTS` generations per backend at once;
  the rest wait, round-robin across keys, up to `LOCALGATE_QUEUE_TIMEOUT`, then get
  `503` + `Retry-After`. Queue size is in the admin overview and Prometheus.
- **Dashboard Share page**: create a device's key with limits and get its base URL,
  connect command, snippets and a **QR code** that opens the new public **`/connect`**
  page (the key travels in the URL fragment, never sent to a server).
- **`localgate service install | uninstall | status | logs`**: run the gateway at login and
  restart it if it stops — launchd on macOS, a systemd user unit on Linux, a logon task on
  Windows. `--dry-run` shows what would be installed.
- **Several GPUs or machines** (`LOCALGATE_BACKENDS`): each request goes to a backend
  serving its model, least busy first, with failover when one is unreachable; each backend
  has its own queue; `/v1/models`, `/health` and the dashboard cover all of them.
- **Shared limits across workers** (`LOCALGATE_REDIS_URL`, `localgate[redis]`): rate limits
  and per-key concurrency hold across worker processes.
- **LAN discovery** (`localgate[discovery]`): `serve --share` announces the gateway over
  mDNS; `localgate discover` lists gateways; `localgate login --url <name>` connects by name.
- **Admin audit log**: key create / change / reveal / revoke, sign-ins (including
  failures), accounts and settings changes, with who and from where — dashboard Activity
  page, `GET /admin/audit`, `localgate audit`.
- **Remote coding agent backend**: `localgate code --remote` now talks to the gateway with a
  backend that keeps Ollama options and JSON tool mode and reads each model's context
  window and tool support from the gateway.
- `localgate doctor` reports network exposure: bind address, admin networks, public URL,
  queue settings.

### Changed

- **Admin surfaces are private by default**: the dashboard, `/admin`, `/docs`, `/redoc`,
  `/openapi.json` and `/metrics` answer only `LOCALGATE_ADMIN_ALLOWED_NETWORKS` (loopback,
  private LANs, Tailscale, IPv6 ULA); other clients get `403 untrusted_network`. The `/v1`
  API is unaffected. `*` restores the old behaviour.
- **`/health` is minimal for untrusted clients**: `{status, version}`. Trusted networks and
  signed-in admins still get backend, database, queue and per-backend details.
- `/v1/models` entries carry `context_window`, `capabilities`, `parameter_size` and
  `quantization` (OpenAI clients ignore unknown fields).
- The gateway refuses to listen on *any* non-loopback address with the placeholder admin
  key (it used to check only `0.0.0.0`, and missed `serve --host`).
- CLI commands other than `serve` log only warnings (no migration chatter).
- `Development Status :: 5 - Production/Stable`.

### Fixed

- **Context window through the gateway**: requests carried no `num_ctx`, so Ollama ran at
  its small default and silently dropped the start of long prompts — including every
  `localgate code --remote` session. Now `LOCALGATE_DEFAULT_NUM_CTX` (8192, capped at the
  model's maximum) is sent unless the client sets one.
- **Disconnected clients**: a dropped stream now closes the upstream request (the GPU stops
  generating), frees its queue slot and still bills the tokens produced; a non-streaming
  request is cancelled when its client goes away.
- **The coding agent on small context windows** (e.g. llama3, 8K): the reply reserve now
  scales with the window, so the agent no longer compacts after every step; it can never
  write a compaction placeholder (`<500 chars elided>`) into a file; and repeated full
  rewrites of one file in a task are stopped instead of asking for approval again and
  again. Half-escaped HTML/CSS (literal `\n`) is repaired.
- `localgate deploy --target compose`: the gateway now listens inside its container and
  Caddy proxies to it on the compose network (it pointed at Caddy's own localhost), with
  forwarded client addresses trusted.

## [0.14.0] — 2026-10-28 — accounts and a live input box

Sign in to the dashboard with an email and password instead of copying a key, and keep typing
while the agent works: the input box stays on screen, queues your next message, and still
switches the write mode.

### Added

- **Email and password sign-in for the dashboard.** The first `localgate serve` in a terminal asks
  you to create a login (email + password); `localgate users create | list | passwd | delete`
  manage accounts. Passwords are stored only as salted scrypt hashes (standard library, no new
  dependency) in a new `users` table (migration `0007`). The admin key still signs in and still
  authenticates scripts. Repeated failed sign-ins from one address are throttled (429).
- **The input box stays on screen while the agent works**: a spinner with elapsed time, a line to
  type your next message into (Enter queues it; queued messages run in order after the task),
  and the status line with the write mode — Shift+Tab switches it mid-task. Esc clears a
  half-typed line, then interrupts.

### Changed

- The dashboard and landing page default to the **sunrise** palette (a palette you picked is kept).
- The landing page no longer links the API reference, and explains the email sign-in.

## [0.13.0] — 2026-10-28 — onboarding, landing page and agent output

Getting started from `pip install` no longer needs a `.env`: the first `localgate serve` makes a
private admin key, and `http://localhost:8000` explains the rest. `localgate code` gets a compact
input box, full themes (new default: sunrise), and stops dumping code into the chat — while making
sure code the model writes actually reaches your files.

### Added

- **A landing page at `/`** instead of a JSON 404: what localgate is, live backend/database/admin-key
  status, and a step-by-step guide (model server → admin key → API keys → calling it like OpenAI →
  coding agent → using it from another machine). API clients asking for JSON get a short pointer
  document.
- **`localgate admin-key`** prints the admin key and where it's set; `--rotate` replaces it.
- **`sunrise` terminal theme** (the dashboard's amber palette), now the default for `localgate code`.

### Fixed

- **Code the model wrote into chat now reaches the files.** Blocks labelled by a comment on their
  first line (`<!-- index.html -->`, `/* styles.css */`, `// app.js`, `# main.py`) are applied as
  writes, like blocks with the filename above them already were. A label naming a file that doesn't
  exist (`src/index.html`) goes to the one project file with that name instead of creating a copy.
- If a reply still holds file-sized code in the project's languages and nothing was changed, the
  model is sent back once to apply it — whatever words the request used.
- Removed the "Window too small..." lines in short terminals (see the input box change below).

### Changed

- **Less code in the chat:** code blocks over 12 lines are collapsed to one line, both while the
  reply streams and in the final answer; `/last` shows the full reply. Write previews (diffs and new
  files) stop at 40 lines with a count of the rest, and the diff header shows `+added -removed`.
- **Every turn says whether files changed:** `✓ wrote …`, `no files changed`, or a warning when the
  reply contains code that wasn't applied.
- **First `localgate serve` creates a private admin key** when none is configured, saves it to the
  per-user env file and prints it once, instead of running with the public placeholder.
  `localgate init` prints the key too. A key you set yourself (env var or `.env`) is never touched.
- `localgate login` explains that it's only for a remote gateway and where its API key comes from.
- **Themes now cover the whole terminal UI:** the prompt, input frame, completion menu, toolbar,
  Markdown answers, tables and spinner, not just a few text colors. `/theme` shows a preview of each.
- **A compact input box:** the prompt sits between two rules with the status line right under it.
  The blank space prompt_toolkit used to reserve for the completion menu (and the toolbar pinned to
  the bottom of the screen) is gone; space is made only while typing a `/command` or `@path`.

## [0.12.0] — 2026-10-28 — the Claude Code-style CLI release

`localgate code` now works like a modern coding-agent CLI: tab completion, `@file`
attachments, `!` shell commands, Esc to interrupt, streaming replies, project instructions and
custom commands. It also fixes the two problems that made it look broken: turns that
silently wrote nothing, and turns that seemed to hang.

### Added

- **Autocomplete:** `/` opens a menu of commands (built-in and custom) with descriptions; `@`
  completes project file paths, skipping anything in `.gitignore`/`.localgateignore`.
- **`@file` attachments:** `fix the navbar in @index.html` sends the file with the message, and
  it counts as read, so the agent can edit it straight away.
- **Prompt history and multi-line input:** ↑/↓ recall earlier prompts (kept across sessions),
  Ctrl+R searches them; end a line with `\` or press Esc then Enter for a new line.
- **`!` shell mode:** `!npm test` runs a command in your own shell with live output; the agent
  sees the output with your next message.
- **Esc to interrupt:** Esc or Ctrl+C stops the running turn (or `!` command) and keeps the session
  open. Half-finished tool calls are dropped from history, and sandboxed commands are killed.
- **`LOCALGATE.md` and `/init`:** standing project instructions the agent reads every turn
  (`AGENTS.md` or `CLAUDE.md` also work), plus a per-user `~/.config/localgate/LOCALGATE.md`.
  `/init` has the agent explore the project and write one.
- **Custom slash commands:** `.localgate/commands/<name>.md` (or `~/.config/localgate/commands/`)
  becomes `/<name>`, with `$ARGUMENTS` filled in; `/commands` lists them.
- **Streaming replies:** the answer appears as the model writes it, then is replaced by the tool
  step or the formatted final answer.

### Changed

- **A cleaner REPL:** startup banner with version, model, mode, files, tools and instructions
  file; `/model` shows an aligned table (params, quant, size, tool-calling data); final answers
  render as Markdown; each tool step is one `● tool args` line; a summary line after every turn
  (files written, tool calls, time); a `❯` prompt with model, mode and context use in the toolbar;
  the to-do list shows as a ☐/◐/☑ checklist.
- `localgate code` only logs warnings (no more structlog debug or Alembic INFO lines in the REPL)
  unless `LOCALGATE_LOG_LEVEL=DEBUG`.

### Fixed

- **Turns that seemed to hang:** `run_command` and `fetch_url` confirmations were drawn over by the
  spinner, so the agent sat waiting for a `y` the user couldn't see.
- **Turns that wrote nothing:** a tool call with invalid JSON (often an unescaped `"` in file
  content) was taken as the final answer. The model is now told the parse error and retries.
- Requests phrased as "make …", "improve …", "enhance …", "redesign …" and similar are now
  treated as change requests, so a reply that only describes a plan is sent back to do the work.

## [0.11.0] — 2026-10-28 — the agent & benchmark release

The biggest update so far. `localgate code` was rebuilt around **measured** reliability on
small local models. The new `localgate bench agent` (11 tasks checked on disk) drove the work:
with qwen2.5-coder:7b on Ollama, the first cut of the rebuilt agent scored **5/11 (45%)**; after
fixing what each failure trace showed, it scores **10–11 of 11 (91–100%)** across repeated runs.
Alongside it:
context management that keeps the original goal, web search with citations, sandboxed commands,
sub-agent roles, local voice, viewable API keys, and a redesigned dashboard.

**Upgrading:** migrations `0005` and `0006` run automatically on start. Voice needs the new extra:
`pip install 'localgate[voice]'`.


### Added

- **Reliable context on Ollama:** the agent sends `num_ctx` through Ollama's native `/api/chat`
  (translated to and from the OpenAI shape), so long sessions are no longer silently truncated
  from the front. Configure with `LOCALGATE_AGENT_NUM_CTX`.
- **Session ledger and compaction:** the original goal, current request, files read and changed,
  decisions, constraints, to-dos and last error are pinned after the system prompt and stored per
  owner and session (migration `0005`). History is compacted against the budget cheapest step
  first, keeping user messages word for word. New REPL commands: `/compact [focus]` and
  `/ledger`. `/context` now shows a breakdown.
- **Tools:** `edit_file` (exact-snippet replacement), paged `read_file`, `todo_write`,
  `fetch_url` (public hosts only, robots.txt respected) and `run_command`, which runs only inside
  an OS sandbox (macOS sandbox-exec, Linux bubblewrap, or Docker/Podman on any OS, including
  Windows). Also SEARCH/REPLACE-block parsing, argument validation against tool schemas, parallel
  read-only calls, and a grammar-constrained JSON tool mode for Ollama models without native tools.
- **Sub-agent roles:** built-in `explorer`, `researcher` and `reviewer`, plus Markdown-defined
  roles in `~/.config/localgate/agents/` and `.localgate/agents/`, each with its own tools and
  optional model. `/agents` lists them.
- **Search provider chain:** `searxng`, `openserp`, `brave`, `tavily`, `duckduckgo` with ordered
  fallback and a short-lived cache.
- **Web search in chat:** `"localgate": {"web_search": true}` on `/v1/chat/completions` returns an
  answer with numbered `localgate.citations`.
- **`localgate bench agent`:** a tool-calling benchmark whose results are checked on disk.
- **Logging:** durable, redacted logs of every agent tool call, recording owner, run and model.
- **Dashboard rebuild:** cookie sign-in (the admin key is no longer stored in the browser) with CSRF
  protection and a strict CSP. Pages: overview, chat with web search, keys, usage, sessions with
  the ledger and a memory inspector, agent runs, models, and settings with a search test.
- **CI:** tests now run on macOS and Windows as well as Linux.
- **Viewable API keys:** new keys keep an AES-GCM-encrypted copy (key derived from the admin key,
  migration `0006`), so an admin can view them again with **Show key** in the dashboard or
  `POST /admin/keys/{id}/reveal`. Changing the admin key makes the old copies unreadable. Nothing
  is stored while the default admin key is in use. Keys created before this can't be shown. To
  disable, set `LOCALGATE_API_KEY_REVEAL_ENABLED=false`.
- **Local voice** (`pip install 'localgate[voice]'`): Piper text-to-speech and faster-whisper
  speech-to-text.
  - **API:** `POST /v1/audio/speech`, `POST /v1/audio/transcriptions` (OpenAI-compatible) and
    `GET /v1/audio/voices`.
  - **Dashboard:** microphone dictation and spoken replies, falling back to browser speech.
  - **CLI:** `localgate voice status|download|say|listen` and `localgate code --voice` with
    `/voice` and `/listen`.
  - **Compatibility:** works around espeak-ng's path-length limit and faster-whisper's
    incompatibility with PyAV 19.
- **Dashboard UX pass** (reviewed with screenshots in headless Chrome, light and dark, desktop and
  phone):
  - **Layout:** cards in a row no longer stagger, and the page spacing is consistent.
  - **Icons:** SVG navigation icons and a favicon.
  - **Times:** relative times with the full timestamp on hover. UTC timestamps without an offset
    were shown hours off outside UTC, which is now fixed.
  - **Chat:** a welcome screen with suggestions, an API-key banner, the web-search hint only when
    it's on, an auto-growing composer, an SVG mic button, and removable conversations.
  - **Models and sessions:** embedding-only models are labeled and left out of the chat picker,
    and sessions show API key names.
- **Dashboard redesign:** chat-first, and the console is reorganised.
  - **Chat:** a reading-first layout with a greeting and centered composer on new chats, a reading
    column, serif answers with full Markdown (headings, nested lists, quotes, links), code blocks
    with copy buttons, and copy, read-aloud and retry under each answer.
  - **Composer:** model picker, Web search and Voice toggles, dictation and send/stop all live
    inside it.
  - **Context:** the recent turns are sent with each message, so the model follows the thread.
  - **Sidebar:** New chat and Recents (with per-conversation URLs), plus a Console section. API
    keys are set inline on first use.
  - **Agent runs:** each task is shown as an expandable timeline of tool steps with status,
    duration, files changed and errors.
  - **Look:** Localgate's own "Jade & Ink" theme (jade accent on cool paper, or bright jade on
    ink in dark mode, with amber as the second chart colour), a new gate logo, and bundled
    open-license fonts (Inter, Source Serif 4, JetBrains Mono; see `FONTS-LICENSE.txt`).
- **Settings redesign:** sections with a sticky side navigation.
  - **Appearance:** a colour-mode switch and a theme picker.
  - **Chat:** API key with show/hide and **Test key**, which shows that key's usage, plus switches
    for default model, web search and voice.
  - **Server:** gateway configuration as readable cards (inference, memory, coding agent, web,
    voice, security); `/admin/config` now also reports agent, search, voice and security
    settings, never the keys themselves.
  - **Database and web search:** connection and provider tests.
  - **Data & account:** export, clear Recents, and sign out.
- **Six colour themes:** Jade (default), Lagoon, Iris, Rose, Sunrise and Graphite, each with
  light and dark variants. Neutrals take a subtle tint from the accent.
- **README:** a new demo video and GIF (a real `localgate code` session plus a dashboard tour),
  and page-by-page screenshots of the new design, replacing the old images.
- **Agent fixes from recording real sessions:**
  - `create_file` on an existing file is refused (pointing to `edit_file`) instead of prompting
    to overwrite.
  - Double-escaped file contents (`\n` text instead of newlines) are repaired.
  - `read_file` on a folder points to `list_directory`.
  - A verbatim repeat of a read-only call or command with no file changed since returns the
    earlier result instead of re-running it.
  - A task that asks for a change but ends with none attempted gets one nudge naming the file.
  - Tool results echoed back as an answer count as no answer.
  - `todo_write` is now opt-in (`LOCALGATE_AGENT_TODO_TOOL`).
  - Duplicate tool-call lines in the terminal are gone, and "files written" no longer repeats
    a file.
- **Chat history:** opening a conversation that belongs to a different API key now shows it
  read-only through the admin session, instead of "no stored history".

### Changed

- New system prompt that no longer forces one JSON tool call per reply. The loop no longer
  injects "next step" or "you have read enough" turns. Stuck-loop detection now looks for repeated
  identical calls or consecutive failures. Large tool results are capped; assistant prose is no
  longer cut at 2000 characters.
- Sub-agents now inherit search, extra tools, timeouts (`tool_timeout_delegate` is now applied)
  and tool logging.
- File writes keep an existing file's CRLF/LF line endings and write identical bytes on every OS.
- Tuned against `localgate bench agent` with qwen2.5-coder:7b. The score went from 5/11 to 9–11/11
  across repeated runs; the changes below are what got it there.
  - **Read rules:** files shown in the first prompt count as read. `edit_file` no longer needs a
    prior read, because its exact match proves the model knows the content. A refused full rewrite
    or delete now reads the file for the model and asks it to retry.
  - **Edit feedback:** an ambiguous `edit_file` returns a complete, unique old/new pair for each
    match. After an edit that removes a name, the result lists where that name is still used.
  - **Syntax checks:** Python and JSON files are checked after every write; a newly introduced
    syntax error is reported to the model.
  - **Unfinished work:** one corrective turn is added when the model claims success after failed
    edits, leaves a file broken, describes a next step instead of doing it, names a file it didn't
    change, or replies with nothing.
  - **Temperature:** the agent now samples at 0.2 by default (`LOCALGATE_AGENT_TEMPERATURE`)
    instead of Ollama's 0.8.

### Fixed

- `/resume` lists a project's sessions newest first even when two share a timestamp, which
  happens routinely on Windows because of its coarse clock.

## [0.10.8] — 2026-09-20

### Fixed

- Normalize empty assistant tool-call content before sending conversation history to Ollama,
  preventing `invalid message content type: <nil>` errors after model switches or tool calls.

## [0.10.7] — 2026-09-20

### Fixed

- `localgate code` now opens the interactive REPL even when its configured model lacks tool support,
  with a warning and access to `/model` for switching. One-shot tasks still fail fast with a clear
  tool-capability error.

## [0.10.6] — 2026-09-20

### Fixed

- Stop the progress spinner before dirty-tree, write, search, delegation, and MCP confirmations so
  interactive prompts remain visible and the CLI no longer appears stuck while waiting for input.

## [0.10.5] — 2026-09-20

### Security

- Confine every file operation to the selected workspace, reject symlink traversal, and apply ignore
  rules consistently to search, status, and diff output.
- Validate mutation targets before previewing existing content and use atomic file replacement.
- Scope conversations, summaries, prompt-cache entries, and retrieved memory by both API key and
  session ID.

### Fixed

- Timed writes now pass through write tracking, preserving summaries, undo, and auto-commit behavior.
- Auto-commit includes only exact agent-written paths and excludes files dirty before the session.
- `/undo` restores pre-agent checkpoints instead of discarding existing human edits or resetting Git.
- Mutation requests that repeatedly produce no tool calls now fail clearly instead of reporting false
  success. Models without advertised tool capability are rejected before a CLI session starts.

### Added

- Explicit `create_file`, `update_file`, and `delete_file` tools, with same-task read requirements for
  updates, deletes, and legacy full-file overwrites.
- Destructive confirmation for overwrites and deletes even in auto mode, plus confirmation for
  externally supplied MCP tools.
- Compound owner/session indexes and regression coverage for cross-key session collisions.

## [0.8.0] — 2026-07-26

This release makes localgate properly installable and self-hostable for anyone who isn't
in the repo directory, and adds a remote-client path so you can hand someone an API key
and they can use your gateway from their own CLI.

### Added

- **`localgate init`** — idempotent first-run setup: creates `~/.config/localgate/` and
  `~/.local/share/localgate/` at 0700, generates a secure admin key into
  `~/.config/localgate/localgate.env` at 0600, and runs database migrations.
  `--force` regenerates the key; `--dry-run` previews without touching the filesystem.
- **`localgate doctor`** — diagnoses the installation: config/data paths, which env files
  loaded, the resolved database URL and which of the three sources won, secret file
  permissions, admin-key placeholder status, and database connectivity.
- **`localgate login --url <url> --api-key <key>`** — verifies a key against a remote
  gateway (via `/v1/usage` — never calls the backend, so a 502 from a down Ollama can't
  block login), does a version handshake, and saves credentials at 0600.
- **`localgate whoami`** — shows your usage stats from the stored gateway credentials.
- **`localgate code --remote/--local`** — routes inference through a logged-in gateway.
  Remote mode defaults to `--no-memory` (the gateway handles memory server-side; running
  both would embed every turn twice into two separate databases).
- **`localgate keys update <id> --rate-limit N`** — change a key's rate limit without
  reissuing it. The error message in `api/deps.py` has been advertising this command since
  0.6.0; now it exists.
- **`localgate db set-url <url>`** — test a database URL and save it as the active database,
  the same connection test the admin dashboard uses — so DB setup isn't dashboard-only.
- **`localgate deploy --domain <domain> --target compose|systemd`** — generates a Caddyfile
  (Caddy handles ACME/TLS automatically) plus either a Docker Compose file or a systemd
  unit, with a fresh admin key printed once and written to a 0600 env file. By default only
  `/v1/*` and `/health*` are exposed; `--expose-admin` adds the dashboard/admin routes.
- **`--proxy-headers` / `--forwarded-allow-ips`** on `localgate serve` — passed through to
  uvicorn so client IP and scheme are correct behind a reverse proxy.

### Changed

- **Default `host` is now `127.0.0.1`** (was `0.0.0.0`). Local-first software defaults to
  loopback; binding to the network is an explicit deployment choice.
- **`create_app` refuses to start** when `host == "0.0.0.0"` and the admin key is still the
  placeholder. A warning was not a control.
- **Default database is now `~/.local/share/localgate/localgate.db`** — stable across
  directories. Before this, `sqlite+aiosqlite:///./localgate.db` resolved relative to CWD,
  giving a different database per directory. `LOCALGATE_DATABASE_URL` and
  `localgate db set-url` still override this.
- **`localgate.config.json`** (the admin-dashboard-established database config) now lives at
  `~/.config/localgate/database.json` (0600) instead of CWD. The legacy CWD file is
  recognised with a one-time warning.
- **`app.state.active_database_url`** replaces the previous `app.state.settings.database_url`
  mutation, so `Settings` stays immutable after construction.
- Dockerfile now sets `LOCALGATE_DATABASE_URL=sqlite+aiosqlite:////data/localgate.db` so the
  database lands in the declared `/data` volume rather than `/app` (which is root-owned and
  caused a permissions error on first boot with a plain `docker run`).

### Fixed

- **HTTP 429 from the gateway** now shows a clear rate-limit message with the retry window,
  instead of the misleading "model doesn't support tool calling" text.

## [0.7.1] — 2026-07-19

### Fixed

- **`localgate code`/`localgate code "task"` crashed with a raw traceback** whenever the
  backend rejected a request — most commonly a 400 from Ollama when the configured model
  doesn't support tool calling (e.g. the default `llama3`). It now shows the backend's own
  error message and, on a 400, a hint to switch models (`/model qwen2.5-coder:7b` in the
  REPL, `--model` for a single-shot run). In the REPL, a rejected turn no longer ends the
  session — it stays open so you can just switch models and continue.

## [0.7.0] — 2026-07-19

A coding agent, backed by whatever model localgate is already pointed at.

### Added

- **`localgate code`** — reads and edits files in a project directory, backed directly by
  the configured inference backend (no API key needed, same pattern as `localgate health`).
  Run it with no argument for an interactive REPL (`/exit`, `/clear`, `/model`, `/undo`), or
  `localgate code "task"` for a single-shot run.
- Tools: `read_file`, `write_file`, `list_directory`, `search_files` (grep-like), `git_status`,
  `git_diff`. All confined to the project root; `.gitignore` and `.localgateignore` keep
  secrets and generated directories out of the model's reach. No shell/`run_command` tool —
  deliberately.
- A git-aware safety net: a one-time warning before writing into a dirty tree (`--force` to
  skip), optional `--auto-commit` (tagged `localgate-agent:`), and `/undo` to revert the last
  write or the last agent commit.
- Colored diffs before every write, streamed responses, and a spinner while waiting on the
  model — via `rich`.
- Session memory: conversation history and recalled context persist per project
  (`.localgate/session_id`), reusing the same RAG memory tables the HTTP API uses.
  `--no-memory` to opt out for a single run.
- A fallback parser for models that print a tool call as plain-text JSON instead of using
  structured `tool_calls` (observed with `qwen2.5-coder` via Ollama) — same execution path as
  a real tool call, so confirmation/diff/memory logic isn't duplicated.
- Shell completion: `localgate --install-completion`.

## [0.6.0] — 2026-07-14

The release where the advertised features actually work.

### Fixed

- **The vLLM, llama.cpp and generic OpenAI backends did not work at all.** They declared
  `InferenceBackend` subclasses that implemented none of its abstract methods, so
  `get_backend("vllm")` raised `TypeError` at startup. Three of the four advertised backends
  were unusable. They now share a real OpenAI-compatible HTTP implementation.
- **Malformed chat requests returned 500 instead of 422.** The handler read `body["messages"]`
  off a raw dict, so a missing field was a crash rather than a client error. Every request
  body is now validated by Pydantic.
- **The admin key was compared with `==`**, which short-circuits at the first differing byte
  and leaks the secret's prefix length through timing. Now `hmac.compare_digest`.
- **The rate limiter never forgot a key it had seen**, leaking one dict entry per key
  forever. Expired windows are now swept.
- **Errors did not use the OpenAI envelope**, so the OpenAI SDK could not parse them. All
  errors — including validation failures and mid-stream backend failures — now return
  `{"error": {"message", "type", "code"}}`.
- **A mid-stream backend failure left clients hanging.** It now reports the error inside the
  stream and still terminates with `[DONE]`.

### Added

- **Migrations.** The schema is versioned with Alembic; `create_all` is gone. Databases from
  0.1–0.2 (tables, but no `alembic_version`) are detected and adopted automatically without
  data loss.
- **A real CLI.** `keys create/list/revoke/usage`, `db upgrade/current/init`, `health`,
  `backends` — the commands the README had been advertising all along.
- **`/v1/models`, `/v1/embeddings`, `/v1/completions`** — previously empty files. The legacy
  completions route is implemented by translating to chat, so it works against backends that
  never implemented it.
- **Rolling conversation summarization.** Past a threshold, older turns are incrementally
  summarized and injected alongside retrieved chunks.
- **Model aliasing.** `LOCALGATE_MODEL_ALIASES='{"fast":"phi4-mini"}'` — swap models without
  touching clients.
- **Prompt caching** (opt-in). Identical prompts skip inference. Off by default because it
  makes sampling deterministic.
- **Plugin system.** Backends are discovered via the `localgate.backends` entry point. The
  built-ins use the same mechanism, so there's no privileged path a plugin can't take.
- **Production hardening.** Structured JSON logs with correlation IDs, Prometheus `/metrics`,
  a liveness/readiness split, graceful shutdown, and fail-fast config validation — production
  now refuses to start with the placeholder admin key.
- **`GET /admin/export`** — every row as JSON, so nobody is locked in.
- **Retrieval score logging and `MEMORY_MIN_SCORE`** — the floor that stops irrelevant chunks
  from filling the context window with noise.
- **`key_prefix`** on keys, so a key can be identified in a listing.
- **`PATCH /admin/keys/{id}`** — change a rate limit without reissuing the key.
- **Documentation.** All seven docs were one-line stubs; they are now written, along with
  three ADRs covering the decisions that shaped the codebase.

### Changed

- `GET /v1/conversations/{id}` now returns an object (messages, summary, chunk count) rather
  than a bare array.
- `POST /admin/keys` returns 201, not 200.
- Usage records now carry latency and a cached flag.

### Security

- SECURITY.md claimed keys were bcrypted; they are SHA-256, which is the *correct* choice for
  high-entropy random keys verified on every request. The document now says so and explains
  why ([ADR 0003](docs/decisions/0003-sha256-for-api-keys.md)).

## [0.2.0] — 2026-07-13

### Added
- Auth, token accounting, RAG memory, dashboard UI, established-database config.
- Test suite with a deterministic `FakeBackend`.

## [0.1.0] — 2026-07-13

### Added
- Initial scaffolding: FastAPI app, Ollama backend, `/v1/chat/completions`, config, CI.
