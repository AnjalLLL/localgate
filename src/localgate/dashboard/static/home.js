// The landing page at `/`: live status from /health, this server's own URL in the
// examples, copy buttons and the Python/curl tabs. No framework, no build step.

const origin = window.location.origin;
for (const el of document.querySelectorAll("[data-origin]")) el.textContent = origin;

function setStatus(id, state, text) {
  const row = document.getElementById(id);
  if (!row) return;
  row.dataset.state = state;
  row.querySelector(".value").textContent = text;
}

function showHint(text) {
  const hint = document.getElementById("status-hint");
  hint.textContent = text;
  hint.hidden = false;
}

async function loadStatus() {
  let body;
  try {
    // /health answers 503 with the same JSON body when something is down.
    const res = await fetch("/health", { headers: { Accept: "application/json" } });
    body = await res.json();
  } catch {
    setStatus("status-backend", "bad", "unknown");
    setStatus("status-database", "bad", "unknown");
    setStatus("status-admin", "bad", "unknown");
    showHint("Couldn't read /health from this server.");
    return;
  }

  const version = `v${body.version}`;
  document.getElementById("version").textContent = version;
  document.getElementById("foot-version").textContent = version;

  // From outside the owner's trusted networks /health only says whether the
  // gateway is up — so show just that rather than three "unknown" rows.
  if (!body.backend) {
    const up = body.status === "ok";
    setStatus("status-backend", up ? "ok" : "bad", up ? "ready" : "degraded");
    for (const id of ["status-database", "status-admin"]) document.getElementById(id).hidden = true;
    document.querySelector("#status-backend .label").textContent = "Gateway";
    return;
  }

  const backend = body.backend || {};
  if (backend.reachable) {
    setStatus("status-backend", "ok", `${backend.type} connected`);
  } else {
    setStatus("status-backend", "bad", `${backend.type} unreachable`);
    showHint(
      `localgate can't reach ${backend.type} at ${backend.url}. Start it (for Ollama: \`ollama serve\`), ` +
        "or set LOCALGATE_BACKEND_URL to where it runs.",
    );
  }

  const database = body.database || {};
  setStatus(
    "status-database",
    database.connected ? "ok" : "bad",
    database.connected ? `${database.dialect.split("+")[0]} connected` : "not connected",
  );

  const placeholder = (body.warnings || []).some((w) => /admin key/i.test(w));
  setStatus(
    "status-admin",
    placeholder ? "warn" : "ok",
    placeholder ? "default placeholder" : "set",
  );
  if (placeholder) {
    showHint(
      "The admin key is still the public placeholder. Remove LOCALGATE_ADMIN_KEY from your .env " +
        "and restart `localgate serve` to generate a private one, or run `localgate admin-key --rotate`.",
    );
  }
}

function addCopyButtons() {
  for (const pre of document.querySelectorAll("pre")) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code").innerText);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select & copy";
      }
      setTimeout(() => (button.textContent = "Copy"), 1500);
    });
    pre.appendChild(button);
  }
}

function setUpTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      for (const other of tabs) {
        const selected = other === tab;
        other.setAttribute("aria-selected", String(selected));
        document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
      }
    });
  }
}

addCopyButtons();
setUpTabs();
loadStatus();
