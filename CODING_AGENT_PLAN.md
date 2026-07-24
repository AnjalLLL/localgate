# localgate code — Plan to Reach Parity with Gemini CLI / Claude Code / aider

## 0. Where things stand today

`localgate code "<task>"` exists as a working prototype:

- `src/localgate/agent/tools.py` — `read_file`, `write_file`, `list_directory`, with path-escape protection (tested, verified to actually block `../` and absolute-path escapes)
- `src/localgate/agent/loop.py` — a single-threaded loop: call `/v1/chat/completions` with `tools`, execute any `tool_calls`, feed results back, repeat until the model returns plain text
- `src/localgate/cli.py` — wires it up as a Typer subcommand, with a y/N prompt before every write

This is enough to prove the concept. It is **not** yet a tool someone would reach for daily. The gaps below are what separate a working loop from a CLI people actually adopt.

---

## 1. What "parity with Gemini CLI / Claude Code" actually means

Break this into four separate problems — they're easy to conflate but need different work:

| Problem | What it means | Current state |
|---|---|---|
| **Invocation ergonomics** | Typing one short word starts an interactive session, like `gemini` or `claude` | Only a single-shot subcommand (`localgate code "task"`) exists — no REPL |
| **Installability** | `pip install localgate` / `pipx install localgate` / a one-line curl script just works, from any machine | Not published anywhere; only runnable from a cloned repo via `uv run` |
| **Terminal UX** | Streaming text, colorized diffs before writes, a visible "thinking" indicator, syntax highlighting | None of this exists — output is flat `print()` calls |
| **Agent capability** | Multi-file awareness, git-awareness, searching across a codebase, not just single read/write calls | Only 3 tools, no search, no git integration, no persistent session |

Each row below becomes its own phase.

---

## 2. Phase-by-phase plan

### Phase 1 — Interactive REPL mode (invocation ergonomics)
**Goal:** `localgate code` with no argument drops into a persistent chat session in the current directory, like `gemini` or `claude` do. `localgate code "one-off task"` keeps working as a single-shot mode for scripting.

- [ ] Add a REPL loop: prompt `> `, read a line, run one full agent turn, print the result, loop back to `> ` — conversation history persists across turns in the same process (not just within one task)
- [ ] Support in-REPL slash-commands: `/exit`, `/clear` (reset conversation), `/model <name>` (switch model mid-session), `/undo` (see Phase 6)
- [ ] Ctrl+C during a tool-call loop should cancel that turn cleanly, not kill the whole session
- [ ] Decide and document: is the entry point `localgate code` (subcommand) or a standalone binary (e.g. `lg`)? **Recommendation:** keep `localgate code` — it's discoverable via `localgate --help`, and a second top-level binary is an extra install-and-PATH problem for no real benefit at this stage.

**Definition of done:** running `localgate code` with no args starts a REPL; you can have a multi-turn conversation about the same codebase without re-invoking the command each time.

---

### Phase 2 — Terminal UX (the part that makes it feel like a real tool)
**Goal:** output looks and feels like Claude Code / aider, not like debug logs.

- [ ] Swap raw `print()` for [`rich`](https://github.com/Textualize/rich) (or `textual` if you want a full TUI later) — spinners while waiting on the model, colored role labels, markdown rendering of the assistant's final text
- [ ] **Stream** the assistant's text response token-by-token instead of waiting for the full completion (localgate's `/v1/chat/completions` already supports `stream: true` — the CLI just isn't using it yet)
- [ ] Before every `write_file`, render an actual **colored diff** (old content vs new) instead of just printing byte counts — use `difflib` for the diff, `rich`'s syntax highlighting for the file content
- [ ] Show a live-updating status line during multi-tool-call turns ("Reading app.py... Writing app.py... Done") instead of scrolling raw print statements

**Definition of done:** a stranger watching over your shoulder would assume this is a polished tool, not a debug script.

---

### Phase 3 — Expand the tool set safely
**Goal:** the agent can actually work across a real project, not just poke at one file at a time.

- [ ] `search_files(pattern, path=".")` — grep-like search across the project (respecting `.gitignore`, see below), so the model can find relevant code without you telling it exact filenames
- [ ] `git_diff()` / `git_status()` — read-only git awareness, so the model knows what's already changed
- [ ] **Explicitly punt on `run_command`/shell execution** until the rest is solid. If you do add it later: require confirmation on every call (no exceptions), and consider a real sandbox (container, restricted subprocess) rather than a bare `subprocess.run`. This is the single riskiest thing on this whole list — don't add it casually.
- [ ] Respect a `.localgateignore` file (same syntax as `.gitignore`) so secrets, `.env`, `node_modules`, etc. are never readable by the model even if it asks — layer this on top of the existing path-escape check, don't replace it

**Definition of done:** the agent can find, read, and reason about code it wasn't explicitly pointed at, without ever touching ignored/sensitive files.

---

### Phase 4 — Git-aware safety net (this is what makes auto-approve safe)
**Goal:** every write is trivially reversible, so `--auto-approve` stops being scary.

- [ ] Before the first write in a session, check `git status` — if the working tree is dirty, warn ("uncommitted changes exist — writes may be hard to distinguish from your own edits") and require explicit `--force` or a y/N confirmation to proceed
- [ ] After each write (or each turn), optionally auto-commit with a generated message (`localgate-agent: <short summary>`) — gated behind a flag, off by default
- [ ] `/undo` in the REPL (Phase 1) = `git checkout -- <file>` for the last-written file, or `git reset --hard HEAD~1` if auto-commit is on

**Definition of done:** a bad agent turn is a 5-second git command away from being undone, not a manual diff-and-revert exercise.

---

### Phase 5 — Session & memory integration (localgate's actual differentiator)
**Goal:** use the RAG memory layer you already built, instead of the agent starting cold every time.

- [ ] Route the REPL's conversation through the same `X-Session-Id` mechanism `chat.py` already supports, so a coding session's history is retrievable later via `GET /v1/conversations/{session_id}` — you get this almost for free, it already exists
- [ ] Persist a `session_id` per project (e.g. a `.localgate/session_id` file in the project directory) so re-running `localgate code` in the same repo resumes the same memory context automatically
- [ ] Longer term: chunk+embed *file contents* (not just chat turns) into the memory layer, so the model can "recall" relevant code from earlier in a long session even after it's scrolled out of the immediate context window — this is the same mechanism as `memory/chunker.py` + `memory/embedder.py`, just applied to file reads as well as chat turns

**Definition of done:** re-opening the CLI in a project you worked on yesterday, the agent already has useful context about what you were doing, without you re-explaining it.

---

### Phase 6 — Distribution & installation
**Goal:** someone who isn't you can install this in one command.

- [ ] Publish to PyPI (`pyproject.toml` already has the `[project.scripts]` entry point wired — this is mostly a `uv build && twine upload` / CI workflow away, and `.github/workflows/release.yml` already exists as a starting point)
- [ ] Once published: `pipx install localgate` or `uv tool install localgate` should just work and put `localgate` on PATH
- [ ] Add shell completion — Typer generates this for free (`localgate --install-completion`), just needs to be documented
- [ ] Optional, once stable: a one-line install script (`curl -fsSL .../install.sh | sh`) for people who don't already have Python tooling set up — mirrors how Ollama itself is installed

**Definition of done:** the install instructions in the README are literally one command, and work on a machine that has never seen this repo before.

---

### Phase 7 — Testing
**Goal:** confidence that changes to the loop don't silently break tool-calling.

- [ ] A fake/mock HTTP transport (similar to `backends/fake.py`) that returns scripted multi-turn tool-call conversations, so the loop logic (not just the tool functions, which are already tested) gets real test coverage without needing a live Ollama + tool-calling model
- [ ] Golden-file tests for diff rendering (Phase 2) — a known before/after pair should always render the same diff output
- [ ] A manual test matrix: which locally-available models actually support tool calling reliably through Ollama (llama3.1, qwen2.5-coder, mistral-nemo, etc.) — document this in `docs/`, since it's the single most common "why doesn't this work" support question you'll get

**Definition of done:** `uv run pytest` covers the agent loop's branching logic (tool call → execute → feed back → repeat vs. plain-text → stop), not just the tool functions in isolation.

---

## 3. Suggested build order

Not everything above needs to happen before this is usable. Recommended sequence:

1. **Phase 1** (REPL) + **Phase 2** (streaming + diffs) together — this is what makes it feel like a real CLI day-to-day, and they're both purely about the existing loop's presentation, not new capabilities.
2. **Phase 4** (git safety net) — do this before expanding tools or enabling auto-approve more widely; it's the thing that makes everything after it lower-stakes.
3. **Phase 3** (search + git-read tools, `.localgateignore`) — now that writes are safe to undo, it's safe to let the agent explore more of the codebase.
4. **Phase 7** (testing) — do this alongside Phases 1–3, not after; a mock-transport test harness will make every subsequent phase faster to build correctly.
5. **Phase 5** (memory integration) — the most novel/differentiated piece, but also the one most worth getting right rather than rushed — do it once the loop itself is stable.
6. **Phase 6** (distribution) — last, once the tool is actually good. Publishing something half-working is worse than not publishing yet.

---

## 4. Open decisions worth making explicitly before building (not just discovering by accident)

- **Entry point naming**: `localgate code` vs. a separate top-level binary — recommendation above is to keep it as a subcommand.
- **Shell/`run_command` tool**: include it or not? This plan deliberately defers it — worth a deliberate yes/no rather than adding it reflexively because "other tools have it."
- **Auto-commit by default or opt-in?** Recommendation: opt-in. Silently commuting on someone's behalf, even for safety, is a surprising default.
- **How much of this lives in the `localgate` package itself vs. becomes a separate `localgate-code` package?** As this grows, it may deserve its own release cadence separate from the core gateway. Not urgent, but worth revisiting once Phases 1–4 are done.
