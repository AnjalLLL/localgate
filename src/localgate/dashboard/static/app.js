// Localgate admin panel. No build step, no framework, no CDN.
//
// Security rules this file keeps (the server's CSP enforces the rest):
// * Untrusted text (model replies, key names, errors, web pages) is only ever set with
//   textContent / text nodes — never assigned as HTML.
// * The admin key is sent once to POST /admin/session and never stored; the session is
//   an httpOnly cookie. Every state-changing admin request carries X-Localgate-Csrf.
// * Only the *API key* used for chat may be stored, and only if the viewer opts in.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const nf = new Intl.NumberFormat();
const compactNf = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const compact = (n) => (Math.abs(n) >= 1000 ? compactNf.format(n) : String(n));
const SVG_NS = "http://www.w3.org/2000/svg";

/** Build a DOM element. Strings among children become text nodes (never HTML). */
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key in el && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

/** Server timestamps are UTC; SQLite returns them without an offset, which a
 * browser would otherwise read as local time (hours off outside UTC). */
const parseTime = (iso) => new Date(/(?:[zZ]|[+-]\d\d:?\d\d)$/.test(iso) ? iso : `${iso}Z`);
const fmtDate = (iso) => (iso ? parseTime(iso).toLocaleString() : "—");
const shortId = (id, n = 8) => (id && id.length > n + 4 ? `${id.slice(0, n)}…` : id || "—");
const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
/** "3 minutes ago" for recent times, a short date for older ones. */
function ago(iso) {
  if (!iso) return "—";
  const then = parseTime(iso);
  const seconds = (then.getTime() - Date.now()) / 1000;
  const abs = Math.abs(seconds);
  if (abs < 45) return "just now";
  for (const [unit, size] of [["minute", 60], ["hour", 3600], ["day", 86400]]) {
    const next = unit === "minute" ? 3600 : unit === "hour" ? 86400 : 7 * 86400;
    if (abs < next) return rtf.format(Math.round(seconds / size), unit);
  }
  return then.toLocaleDateString(undefined, { day: "numeric", month: "short", year: then.getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
}
/** A relative time with the full timestamp on hover. */
const when = (iso) => h("span", { class: "time", title: iso ? fmtDate(iso) : "", text: ago(iso) });

// ---------------------------------------------------------------- storage (never throws)

const storage = {
  get(area, key) {
    try { return window[area].getItem(key); } catch { return null; }
  },
  set(area, key, value) {
    try { window[area].setItem(key, value); } catch { /* private mode / blocked */ }
  },
  remove(area, key) {
    try { window[area].removeItem(key); } catch { /* ignore */ }
  },
};

const prefs = {
  get apiKey() {
    return storage.get("sessionStorage", "lg_api_key") || storage.get("localStorage", "lg_api_key") || "";
  },
  get rememberApiKey() { return storage.get("localStorage", "lg_api_key") !== null; },
  setApiKey(value, remember) {
    storage.remove("localStorage", "lg_api_key");
    storage.remove("sessionStorage", "lg_api_key");
    if (!value) return;
    storage.set(remember ? "localStorage" : "sessionStorage", "lg_api_key", value);
  },
  get sessions() {
    try { return JSON.parse(storage.get("localStorage", "lg_sessions") || "[]"); } catch { return []; }
  },
  set sessions(list) { storage.set("localStorage", "lg_sessions", JSON.stringify(list.slice(0, 60))); },
  get activeSession() { return storage.get("localStorage", "lg_active") || ""; },
  set activeSession(id) { storage.set("localStorage", "lg_active", id); },
  get model() { return storage.get("localStorage", "lg_model") || ""; },
  set model(v) { storage.set("localStorage", "lg_model", v); },
  get speak() { return storage.get("localStorage", "lg_speak") === "1"; },
  set speak(v) { storage.set("localStorage", "lg_speak", v ? "1" : "0"); },
  get webSearch() { return storage.get("localStorage", "lg_web") === "1"; },
  set webSearch(v) { storage.set("localStorage", "lg_web", v ? "1" : "0"); },
  get palette() { return storage.get("localStorage", "lg_palette") || "sunrise"; },
  set palette(v) { storage.set("localStorage", "lg_palette", v); },
  get theme() { return storage.get("localStorage", "lg_theme") || "system"; },
  set theme(v) { storage.set("localStorage", "lg_theme", v); },
};

// ---------------------------------------------------------------- feedback

function toast(message, kind = "") {
  const el = h("div", { class: `toast ${kind}`, role: kind === "bad" ? "alert" : "status", text: message });
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "bad" ? 6000 : 3500);
}

/** In-page confirmation (a <dialog>), never window.confirm. Resolves true on confirm. */
function confirmAction(title, body, confirmLabel = "Confirm") {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-body").textContent = body;
  $("#confirm-ok").textContent = confirmLabel;
  return new Promise((resolve) => {
    const done = () => {
      dialog.removeEventListener("close", done);
      resolve(dialog.returnValue === "ok");
    };
    dialog.returnValue = "";
    dialog.addEventListener("close", done);
    if (typeof dialog.showModal === "function") dialog.showModal();
    else resolve(false);
  });
}

const loading = (text = "Loading…") => h("div", { class: "loading", text });
const empty = (text) => h("div", { class: "empty", text });
const errorState = (err) => h("div", { class: "error-state", role: "alert", text: err?.message || String(err) });

function downloadBlob(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- API

class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

function errorMessage(payload, resp) {
  return payload?.error?.message || payload?.detail || `${resp.status} ${resp.statusText}`;
}

/** Admin API call, authenticated by the session cookie. */
async function api(path, { method = "GET", body } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET") headers["X-Localgate-Csrf"] = "1";
  const resp = await fetch(path, {
    method,
    headers,
    credentials: "same-origin",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (resp.status === 401 && path.startsWith("/admin")) {
    showLogin("Your session has expired. Sign in again.");
    throw new ApiError("Not signed in.", 401);
  }
  if (!resp.ok) throw new ApiError(errorMessage(await resp.json().catch(() => null), resp), resp.status);
  if (resp.status === 204) return null;
  return resp.json();
}

/** API-key call for the chat surface (/v1/*). */
async function apiWithKey(path, init = {}) {
  const key = prefs.apiKey;
  if (!key) throw new ApiError("Set an API key for chat first (Chat → API key, or Settings).", 0);
  const resp = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}`, ...(init.headers || {}) },
  });
  if (!resp.ok) throw new ApiError(errorMessage(await resp.json().catch(() => null), resp), resp.status);
  return resp;
}

// ---------------------------------------------------------------- theme

const THEMES = ["system", "light", "dark"];
const THEME_ICONS = { system: "monitor", light: "sun", dark: "moon" };

const PALETTES = [
  { id: "jade", name: "Jade", light: ["#0d7a62", "#c98d1f"], dark: ["#3ad2a5", "#f0b54f"] },
  { id: "lagoon", name: "Lagoon", light: ["#0b63a5", "#e0664f"], dark: ["#5cb8f2", "#ff8f78"] },
  { id: "iris", name: "Iris", light: ["#5b3fd0", "#d4881c"], dark: ["#a594ff", "#f3b24d"] },
  { id: "rose", name: "Rose", light: ["#b8325a", "#2b8a8a"], dark: ["#ff7fa3", "#5fd0c8"] },
  { id: "sunrise", name: "Sunrise", light: ["#a8590e", "#2f6fb5"], dark: ["#ffb454", "#78b1f0"] },
  { id: "graphite", name: "Graphite", light: ["#22272b", "#6b7b88"], dark: ["#e8ebed", "#9aaab6"] },
];

function applyPalette(id) {
  const known = PALETTES.some((p) => p.id === id) ? id : "sunrise";
  document.documentElement.dataset.palette = known;
}

function applyTheme(theme) {
  applyPalette(prefs.palette);
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  const btn = $("#theme-toggle");
  if (btn) {
    btn.replaceChildren(icon(THEME_ICONS[theme]));
    btn.setAttribute("aria-label", `Theme: ${theme}. Click to change.`);
    btn.title = `Theme: ${theme}`;
  }
}

// ---------------------------------------------------------------- auth screens

// Sign in with an account (email + password) by default; the admin key stays
// available for whoever runs the server, and is the only option until an account exists.
let loginWithAdminKey = false;

function setLoginMode(adminKey) {
  loginWithAdminKey = adminKey;
  $("#login-account").hidden = adminKey;
  $("#login-admin").hidden = !adminKey;
  $("#login-email").required = !adminKey;
  $("#login-password").required = !adminKey;
  $("#login-key").required = adminKey;
  $("#login-mode").textContent = adminKey ? "Sign in with email instead" : "Use the admin key instead";
  (adminKey ? $("#login-key") : $("#login-email")).focus();
}

async function showLogin(message = "") {
  $("#app").hidden = true;
  $("#login").hidden = false;
  $("#login-error").textContent = message;
  $("#login-key").value = "";
  $("#login-password").value = "";
  let accounts = true;
  try {
    accounts = (await (await fetch("/admin/session/methods")).json()).accounts;
  } catch { /* assume accounts; the admin key option is still there */ }
  $("#login-intro").replaceChildren(
    ...(accounts
      ? ["Your local models, memory and agents in one place."]
      : ["No dashboard account yet. Create one on the server with ", h("code", { text: "localgate users create" }), ", or sign in with the admin key."]),
  );
  setLoginMode(!accounts);
}

function showApp() {
  $("#login").hidden = true;
  $("#app").hidden = false;
  route();
}

async function checkSession() {
  try {
    const resp = await fetch("/admin/session", { credentials: "same-origin" });
    if (resp.ok) {
      const who = await resp.json().catch(() => ({}));
      $("#sign-out").title = who.email ? `Sign out (${who.email})` : "Sign out";
      showApp();
    } else showLogin();
  } catch {
    showLogin("Couldn't reach the server.");
  }
}

async function signIn(event) {
  event.preventDefault();
  const button = $("#login-submit");
  button.disabled = true;
  $("#login-error").textContent = "";
  try {
    const resp = await fetch("/admin/session", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        loginWithAdminKey
          ? { admin_key: $("#login-key").value }
          : { email: $("#login-email").value, password: $("#login-password").value },
      ),
    });
    if (!resp.ok) throw new Error(errorMessage(await resp.json().catch(() => null), resp));
    $("#login-key").value = "";
    $("#login-password").value = "";
    checkSession();
  } catch (err) {
    $("#login-error").textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

async function signOut() {
  try { await api("/admin/session", { method: "DELETE" }); } catch { /* already signed out */ }
  showLogin("Signed out.");
}

// ---------------------------------------------------------------- health

async function pollHealth() {
  const dot = $("#health-dot");
  const text = $("#health-text");
  try {
    // /health answers 503 when a dependency is down; the body says which one.
    const health = await (await fetch("/health", { credentials: "same-origin" })).json();
    const backendOk = health.backend?.reachable;
    const dbOk = health.database?.connected;
    $("#version").textContent = health.version ? `v${health.version}` : "";
    dot.className = `dot ${backendOk && dbOk ? "ok" : "bad"}`;
    if (!backendOk) text.textContent = `${health.backend?.type ?? "backend"} unreachable`;
    else if (!dbOk) text.textContent = "database unreachable";
    else text.textContent = `${health.backend.type} · ${String(health.database.dialect).split("+")[0]}`;
  } catch {
    dot.className = "dot bad";
    text.textContent = "server unreachable";
  }
}

// ---------------------------------------------------------------- shared widgets

function pageHead(title, subtitle, ...actions) {
  return h(
    "div",
    { class: "page-head" },
    h("div", {}, h("h1", { text: title }), subtitle ? h("p", { text: subtitle }) : null),
    actions.length ? h("div", { class: "actions" }, actions) : null,
  );
}

function card(title, body, ...headExtras) {
  return h(
    "section",
    { class: "card" },
    title ? h("div", { class: "card-head" }, h("h2", { text: title }), headExtras) : null,
    body,
  );
}

function stat(label, value, sub) {
  return h(
    "div",
    { class: "card stat" },
    h("div", { class: "label", text: label }),
    h("div", { class: "value", text: value }),
    sub ? h("div", { class: "sub", text: sub }) : null,
  );
}

/** A table from column specs: {label, value(row) -> string|Node, num?}. */
function table(columns, rows, { emptyText = "Nothing here yet.", onRowClick, rowClass } = {}) {
  if (!rows.length) return empty(emptyText);
  const thead = h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", scope: "col", text: c.label }))));
  const tbody = h(
    "tbody",
    {},
    rows.map((row) => {
      const tr = h(
        "tr",
        { class: [onRowClick ? "clickable" : "", rowClass ? rowClass(row) : ""].join(" ").trim() },
        columns.map((c) => {
          const v = c.value(row);
          return h("td", { class: [c.num ? "num" : "", c.class || ""].join(" ").trim() }, v instanceof Node ? v : String(v ?? "—"));
        }),
      );
      if (onRowClick) {
        tr.tabIndex = 0;
        tr.addEventListener("click", () => onRowClick(row, tr));
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRowClick(row, tr); }
        });
      }
      return tr;
    }),
  );
  return h("div", { class: "table-wrap" }, h("table", {}, thead, tbody));
}

const badge = (text, kind = "") => h("span", { class: `badge ${kind}`, text });

function rateClass(rate) {
  if (rate >= 0.8) return "";
  return rate >= 0.5 ? "mid" : "low";
}

function meterRow(label, rate, detail) {
  const fill = h("span", { class: rateClass(rate) });
  fill.style.width = `${Math.max(0, Math.min(1, rate)) * 100}%`;
  return h(
    "div",
    { class: "meter-row" },
    h("span", { class: "truncate mono", title: label, text: label }),
    h("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(rate * 100)), "aria-label": `${label} success rate` }, fill),
    h("span", { class: "num small", text: detail }),
  );
}

/** Stacked prompt/completion bars per day, with an accessible label and hover tooltip. */
function tokenChart(daily) {
  const wrap = h("div", {});
  if (!daily.length) return empty("No usage in this period yet.");
  const legend = h(
    "div",
    { class: "legend" },
    h("span", {}, h("span", { class: "swatch s1" }), "Prompt"),
    h("span", {}, h("span", { class: "swatch s2" }), "Completion"),
  );
  const total = daily.reduce((n, d) => n + d.prompt_tokens + d.completion_tokens, 0);
  const chart = svg("svg", { class: "chart", role: "img", "aria-label": `Tokens per day over ${daily.length} days, ${nf.format(total)} in total` });
  wrap.append(chart, legend);

  const draw = () => {
    chart.textContent = "";
    const W = chart.clientWidth || 800;
    const H = 220;
    const pad = { top: 12, right: 8, bottom: 24, left: 46 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const max = Math.max(...daily.map((d) => d.prompt_tokens + d.completion_tokens), 1);
    const step = 10 ** Math.floor(Math.log10(max));
    const top = Math.ceil(max / step) * step;
    const y = (v) => pad.top + plotH - (v / top) * plotH;

    for (let i = 0; i <= 2; i++) {
      const value = (top / 2) * i;
      chart.append(
        svg("line", { class: "grid-line", x1: pad.left, x2: W - pad.right, y1: y(value), y2: y(value) }),
        Object.assign(svg("text", { class: "axis-text", x: pad.left - 8, y: y(value) + 3, "text-anchor": "end" }), { textContent: compact(value) }),
      );
    }
    const slot = plotW / daily.length;
    const barW = Math.max(4, Math.min(32, slot - 6));
    const tip = $("#tooltip");
    daily.forEach((d, i) => {
      const x = pad.left + slot * i + (slot - barW) / 2;
      const sum = d.prompt_tokens + d.completion_tokens;
      const label = new Date(`${d.date}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
      const completionH = Math.max(0, y(0) - y(d.completion_tokens));
      const promptH = Math.max(0, y(d.completion_tokens) - y(sum) - (completionH > 0 ? 2 : 0));
      if (completionH > 0) chart.append(svg("rect", { x, y: y(d.completion_tokens), width: barW, height: completionH, rx: 2, fill: "var(--series-2)" }));
      if (promptH > 0) chart.append(svg("rect", { x, y: y(sum), width: barW, height: promptH, rx: 3, fill: "var(--series-1)" }));
      // A full-height transparent hit area so thin bars are still easy to hover.
      const hit = svg("rect", { class: "bar-hit", x: pad.left + slot * i, y: pad.top, width: slot, height: plotH, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => {
        tip.textContent = "";
        append(tip, [
          h("div", { class: "t-head", text: label }),
          h("div", { class: "t-row" }, h("span", {}, h("span", { class: "swatch s1" }), "Prompt"), h("span", { text: nf.format(d.prompt_tokens) })),
          h("div", { class: "t-row" }, h("span", {}, h("span", { class: "swatch s2" }), "Completion"), h("span", { text: nf.format(d.completion_tokens) })),
          h("div", { class: "t-row" }, h("span", { text: "Requests" }), h("span", { text: nf.format(d.requests ?? 0) })),
        ]);
        tip.hidden = false;
        tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 12)}px`;
        tip.style.top = `${e.clientY - 12}px`;
      });
      hit.addEventListener("mouseleave", () => { tip.hidden = true; });
      chart.append(hit);
      if (i === 0 || i === daily.length - 1 || i === Math.floor(daily.length / 2)) {
        const anchor = i === 0 ? "start" : i === daily.length - 1 ? "end" : "middle";
        chart.append(Object.assign(svg("text", { class: "axis-text", x: x + barW / 2, y: H - 6, "text-anchor": anchor }), { textContent: label }));
      }
    });
  };
  requestAnimationFrame(draw);
  const observer = new ResizeObserver(() => draw());
  observer.observe(chart);
  return wrap;
}

// ---------------------------------------------------------------- icons

const ICON_PATHS = {
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  retry: '<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/>',
  speaker: '<path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/>',
  mic: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8"/>',
  stopSq: '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" stroke="none"/>',
  arrowUp: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  key: '<circle cx="7.5" cy="15.5" r="4.5"/><path d="m10.7 12.3 9.8-9.8M17 6l3 3M14 9l2 2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  code: '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
  pen: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5V21h16"/>',
  bulb: '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13 7 4 4"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  terminal: '<path d="m4 17 6-6-6-6M12 19h8"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  git: '<circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="8" r="2.5"/><path d="M6 8.5v7M18 10.5c0 4-6 3-10.5 6"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
  plug: '<path d="M12 22v-5M9 8V2M15 8V2M18 8v4a6 6 0 0 1-12 0V8z"/>',
  gate: '<path d="M4 21V10a8 8 0 0 1 16 0v11"/><path d="M9 21v-7a3 3 0 0 1 6 0v7"/><path d="M3 21h18"/>',
  tool: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z"/>',
  chevron: '<path d="m9 18 6-6-6-6"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="M10.6 5.1A10.7 10.7 0 0 1 12 5c6.5 0 10 7 10 7a17.6 17.6 0 0 1-2.4 3.3M6.6 6.6C3.9 8.4 2 12 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2M2 2l20 20"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  monitor: '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>',
};

/** An inline SVG icon (static path data only — never interpolated user text). */
function icon(name, size) {
  const el = document.createElementNS(SVG_NS, "svg");
  el.setAttribute("viewBox", "0 0 24 24");
  el.setAttribute("fill", "none");
  el.setAttribute("stroke", "currentColor");
  el.setAttribute("stroke-width", "1.9");
  el.setAttribute("stroke-linecap", "round");
  el.setAttribute("stroke-linejoin", "round");
  el.setAttribute("aria-hidden", "true");
  if (size) { el.setAttribute("width", size); el.setAttribute("height", size); }
  const doc = new DOMParser().parseFromString(`<svg xmlns="${SVG_NS}">${ICON_PATHS[name] || ""}</svg>`, "image/svg+xml");
  for (const node of [...doc.documentElement.childNodes]) el.appendChild(document.importNode(node, true));
  return el;
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    if (button) {
      const original = [...button.childNodes];
      button.replaceChildren(icon("check"), ...(button.classList.contains("copy-btn") ? ["Copied"] : []));
      setTimeout(() => button.replaceChildren(...original), 1400);
    } else toast("Copied.");
  } catch {
    toast("Copy failed — select the text and copy it manually.", "bad");
  }
}

// ---------------------------------------------------------------- markdown rendering

/** Inline markdown → DOM: `code`, **bold**, *italic*, [links](https://…). Text only, never HTML. */
function renderInline(text, into) {
  const re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\s][^*\n]*\*)|(\[[^\]\n]+\]\((https?:\/\/[^)\s]+)\))/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) into.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = m[0];
    if (m[1]) into.appendChild(h("code", { text: tok.slice(1, -1) }));
    else if (m[2] || m[3]) into.appendChild(h("strong", { text: tok.slice(2, -2) }));
    else if (m[4]) into.appendChild(h("em", { text: tok.slice(1, -1) }));
    else if (m[5]) {
      const label = tok.slice(1, tok.indexOf("]("));
      into.appendChild(h("a", { href: m[6], target: "_blank", rel: "noopener noreferrer", text: label }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) into.appendChild(document.createTextNode(text.slice(last)));
  return into;
}

function codeBlock(code, lang) {
  const copy = h("button", { class: "copy-btn", type: "button", "aria-label": "Copy code", onclick: (e) => copyText(code, e.currentTarget) }, icon("copy"), "Copy");
  const codeEl = h("code", { text: code });
  if (lang) codeEl.dataset.lang = lang;
  return h("div", { class: "codeblock" }, h("div", { class: "codeblock-head" }, h("span", { text: lang || "code" }), copy), h("pre", {}, codeEl));
}

const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

/** A list starting at ``lines[start]``: items at the same indent, more-indented items
 * nested under the previous one, and the source's own numbering kept (so "1. … 2. …"
 * separated by sub-bullets doesn't restart at 1). Returns [element, next index]. */
function parseList(lines, start) {
  const first = lines[start].match(LIST_RE);
  const indent = first[1].length;
  const ordered = /\d/.test(first[2]);
  const list = h(ordered ? "ol" : "ul", {});
  if (ordered && parseInt(first[2], 10) > 1) list.start = parseInt(first[2], 10);
  let i = start;
  let lastItem = null;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // A blank line ends the list unless the list continues right after it.
      const next = lines[i + 1]?.match(LIST_RE);
      if (next && next[1].length >= indent) { i += 1; continue; }
      break;
    }
    const item = line.match(LIST_RE);
    if (item && item[1].length > indent && lastItem) {
      const [sub, next] = parseList(lines, i);
      lastItem.appendChild(sub);
      i = next;
      continue;
    }
    if (!item || item[1].length < indent || /\d/.test(item[2]) !== ordered) {
      // An indented continuation line belongs to the current item.
      if (!item && lastItem && /^\s{2,}\S/.test(line)) { lastItem.appendChild(document.createTextNode(" ")); renderInline(line.trim(), lastItem); i += 1; continue; }
      break;
    }
    lastItem = renderInline(item[3], h("li", {}));
    list.appendChild(lastItem);
    i += 1;
  }
  return [list, i];
}

/** Render model output safely: code fences, headings, lists, quotes, rules, paragraphs. */
function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  let para = [];
  const flushPara = () => {
    if (!para.length) return;
    const p = h("p", {});
    para.forEach((line, k) => {
      if (k > 0) p.appendChild(document.createElement("br"));
      renderInline(line, p);
    });
    frag.appendChild(p);
    para = [];
  };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]{0,20})\s*$/);
    if (fence) {
      flushPara();
      const body = [];
      i += 1;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { body.push(lines[i]); i += 1; }
      frag.appendChild(codeBlock(body.join("\n"), fence[1]));
      i += 1;
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushPara();
      const level = Math.min(4, heading[1].length + 1);
      frag.appendChild(renderInline(heading[2], h(`h${level}`, {})));
      i += 1;
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushPara(); frag.appendChild(h("hr", {})); i += 1; continue; }
    const listMatch = line.match(LIST_RE);
    if (listMatch) {
      flushPara();
      const [list, next] = parseList(lines, i);
      frag.appendChild(list);
      i = next;
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flushPara();
      const quote = h("blockquote", {});
      const qlines = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { qlines.push(lines[i].replace(/^\s*>\s?/, "")); i += 1; }
      quote.appendChild(renderInline(qlines.join(" "), h("p", {})));
      frag.appendChild(quote);
      continue;
    }
    if (!line.trim()) { flushPara(); i += 1; continue; }
    para.push(line);
    i += 1;
  }
  flushPara();
  return frag;
}

function citationChips(citations) {
  if (!citations?.length) return null;
  return h(
    "div",
    { class: "citations", "aria-label": "Sources" },
    citations.map((c) => {
      let href = "";
      try {
        const url = new URL(c.url);
        if (url.protocol === "http:" || url.protocol === "https:") href = url.href;
      } catch { /* not a URL: render without a link */ }
      const inner = [h("span", { class: "n", text: `${c.index}` }), h("span", { class: "t", text: c.title || c.url })];
      return href
        ? h("a", { class: "citation", href, target: "_blank", rel: "noopener noreferrer", title: c.url }, inner)
        : h("span", { class: "citation" }, inner);
    }),
  );
}

// ================================================================ views

async function viewOverview(root) {
  root.append(pageHead("Overview", "Gateway activity at a glance."));
  const body = h("div", {}, loading());
  root.append(body);
  const [overview, usage] = await Promise.all([api("/admin/overview"), api("/admin/usage?days=14")]);
  const daily = overview.daily || usage.daily || [];
  const prompt = daily.reduce((n, d) => n + d.prompt_tokens, 0);
  const completion = daily.reduce((n, d) => n + d.completion_tokens, 0);
  const rate = overview.agent?.success_rate;
  body.textContent = "";
  body.append(
    h(
      "div",
      { class: "grid stats" },
      stat("Requests", nf.format(overview.usage.request_count), "all time"),
      stat("Total tokens", compact(overview.usage.total_tokens), `avg ${nf.format(Math.round(overview.usage.avg_latency_ms))} ms latency`),
      stat("Prompt / completion", `${compact(prompt)} / ${compact(completion)}`, "last 14 days"),
      stat("Active keys", String(overview.keys.active), `${overview.keys.total} total`),
      stat("Agent tool calls", rate === null || rate === undefined ? "—" : `${Math.round(rate * 100)}%`, `${nf.format(overview.agent.recent_tool_calls)} recent calls succeeded`),
      stat("Backend", overview.backend.healthy ? "Healthy" : "Unreachable", overview.backend.type),
    ),
    h(
      "div",
      { class: "grid split" },
      card("Tokens per day", h("div", { class: "card-body" }, tokenChart(daily))),
      card(
        "By model",
        table(
          [
            { label: "Model", value: (m) => h("span", { class: "mono", text: m.model }) },
            { label: "Requests", value: (m) => nf.format(m.request_count), num: true },
            { label: "Tokens", value: (m) => compact(m.total_tokens), num: true },
          ],
          usage.by_model || [],
          { emptyText: "No requests yet." },
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------- chat

let chatAbort = null;
let currentChatId = "";

/** Recent conversations in the sidebar. Stored per browser; the history itself lives
 * on the server, scoped to the API key that created it. */
function renderHistory() {
  const list = $("#history");
  if (!list) return;
  list.textContent = "";
  const sessions = prefs.sessions;
  if (!sessions.length) {
    list.append(h("div", { class: "hist-empty", text: "Your conversations will appear here." }));
    return;
  }
  sessions.forEach((s) => {
    list.append(
      h(
        "div",
        { class: "hist-row", role: "listitem", "aria-current": s.id === currentChatId ? "true" : "false" },
        h("a", { class: "hist-item", href: `#/chat/${encodeURIComponent(s.id)}`, title: s.label, text: s.label }),
        h("button", {
          class: "hist-remove",
          type: "button",
          title: "Remove from Recents (stored history is kept)",
          "aria-label": `Remove ${s.label} from Recents`,
          text: "×",
          onclick: () => {
            prefs.sessions = prefs.sessions.filter((x) => x.id !== s.id);
            if (currentChatId === s.id) location.hash = "#/chat/new";
            else renderHistory();
          },
        }),
      ),
    );
  });
}

function greetingText() {
  const hour = new Date().getHours();
  if (hour < 5) return "Burning the midnight oil";
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

const STARTERS = [
  { icon: "code", text: "Write code", prompt: "Write a Python function that " },
  { icon: "bulb", text: "Explain a concept", prompt: "Explain, simply, how " },
  { icon: "pen", text: "Help me write", prompt: "Help me write a short note about " },
  { icon: "globe", text: "Search the web", prompt: "What's the latest news about ", web: true },
  { icon: "book", text: "Summarize", prompt: "Summarize the key points of " },
];

async function viewChat(root, chatId) {
  root.className = "chat-page";
  const isNew = !chatId || chatId === "new";
  currentChatId = isNew ? "" : chatId;
  if (!isNew) prefs.activeSession = chatId;
  renderHistory();

  /** The conversation as shown, oldest first: {role, content}. Also the context sent
   * with each request, so the model sees the thread even before memory recalls it. */
  const turns = [];

  // ---- header
  const title = h("div", { class: "chat-title" });
  const setTitle = () => {
    const entry = prefs.sessions.find((s) => s.id === currentChatId);
    title.textContent = entry ? entry.label : "";
  };
  const keyBtn = h("button", { class: "icon-btn ghost", type: "button", title: "API key for chat", "aria-label": "API key for chat" }, icon("key"));
  const head = h("div", { class: "chat-head" }, title, keyBtn);

  // ---- API key (inline setup when missing, a popover to change it)
  const keyForm = () => {
    const input = h("input", { type: "password", placeholder: "lg_…", value: prefs.apiKey, "aria-label": "API key" });
    const remember = h("input", { type: "checkbox", checked: prefs.rememberApiKey });
    const save = (e) => {
      e?.preventDefault();
      prefs.setApiKey(input.value.trim(), remember.checked);
      toast(input.value.trim() ? "API key saved." : "API key cleared.");
      refreshKeyState();
    };
    return h(
      "form",
      { onsubmit: save },
      h("div", { class: "field-row" }, h("label", { class: "field" }, h("span", { text: "API key" }), input), h("button", { class: "btn primary", type: "submit", text: "Save" })),
      h("label", { class: "check small" }, remember, "Remember on this device (otherwise this tab only)"),
    );
  };
  const keySetup = h(
    "div",
    { class: "key-setup" },
    h("h2", { text: "Connect an API key" }),
    h("p", {}, "Chat uses a gateway API key, like any client would. ", h("a", { href: "#/keys", text: "Create one in API keys" }), " and paste it here."),
    keyForm(),
  );
  let popover = null;
  keyBtn.addEventListener("click", () => {
    if (popover) { popover.remove(); popover = null; return; }
    popover = h("div", { class: "popover" }, h("h2", { text: "API key for chat" }), h("p", { class: "small muted", text: "Requests and memory are scoped to this key." }), keyForm());
    head.append(popover);
    popover.querySelector("input")?.focus();
  });

  // ---- thread
  const threadInner = h("div", { class: "thread-inner" });
  const thread = h("div", { class: "thread" }, threadInner);
  const scrollDown = () => { thread.scrollTop = thread.scrollHeight; };

  // ---- composer
  const prompt = h("textarea", { rows: 1, placeholder: "How can I help you today?", "aria-label": "Message" });
  const autosize = () => { prompt.style.height = "auto"; prompt.style.height = `${Math.min(prompt.scrollHeight, 240)}px`; };
  const webPill = h("button", { class: "pill", type: "button", "aria-pressed": String(prefs.webSearch), title: "Search the web and cite sources" }, icon("globe"), h("span", { class: "pill-text", text: "Web search" }));
  const speakPill = h("button", { class: "pill", type: "button", "aria-pressed": String(prefs.speak), title: "Read replies aloud (local Piper voice)" }, icon("speaker"), h("span", { class: "pill-text", text: "Voice" }));
  const micBtn = h("button", { class: "round-btn", type: "button", title: "Dictate", "aria-label": "Dictate a message" }, icon("mic"));
  const modelSelect = h("select", { class: "model-pick", "aria-label": "Model" }, h("option", { value: "", text: "Default model" }));
  const sendBtn = h("button", { class: "send-btn", type: "submit", "aria-label": "Send message", disabled: true }, icon("arrowUp"));
  const form = h(
    "form",
    { class: "composer", onsubmit: (e) => { e.preventDefault(); if (chatAbort) chatAbort.abort(); else send(prompt.value); } },
    prompt,
    h("div", { class: "composer-bar" }, webPill, speakPill, h("span", { class: "spacer" }), modelSelect, micBtn, sendBtn),
  );
  const composerWrap = h("div", { class: "composer-wrap" }, form, h("p", { class: "fineprint", text: "Runs on your own hardware. Models can make mistakes — check important answers." }));

  const setStreaming = (on) => {
    sendBtn.replaceChildren(icon(on ? "stopSq" : "arrowUp"));
    sendBtn.setAttribute("aria-label", on ? "Stop" : "Send message");
    sendBtn.disabled = !on && !prompt.value.trim();
  };
  prompt.addEventListener("input", () => { autosize(); if (!chatAbort) sendBtn.disabled = !prompt.value.trim(); });
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!chatAbort && prompt.value.trim()) form.requestSubmit(); }
  });
  const togglePill = (pill, key) => {
    const on = pill.getAttribute("aria-pressed") !== "true";
    pill.setAttribute("aria-pressed", String(on));
    prefs[key] = on;
    if (key === "speak" && !on) stopSpeaking();
  };
  webPill.addEventListener("click", () => togglePill(webPill, "webSearch"));
  speakPill.addEventListener("click", () => togglePill(speakPill, "speak"));
  modelSelect.addEventListener("change", () => { prefs.model = modelSelect.value; });

  // ---- layout: centered greeting for a new chat, thread + docked composer otherwise
  const starters = h(
    "div",
    { class: "starters" },
    STARTERS.map((s) =>
      h("button", {
        class: "starter",
        type: "button",
        onclick: () => {
          prompt.value = s.prompt;
          if (s.web && webPill.getAttribute("aria-pressed") !== "true") togglePill(webPill, "webSearch");
          prompt.focus();
          autosize();
          sendBtn.disabled = false;
        },
      }, icon(s.icon), s.text),
    ),
  );
  const refreshKeyState = () => {
    keySetup.hidden = Boolean(prefs.apiKey);
    if (popover && prefs.apiKey) { popover.remove(); popover = null; }
  };
  const layout = () => {
    root.textContent = "";
    if (!turns.length) {
      root.append(
        head,
        h(
          "div",
          { class: "chat-empty" },
          h("div", { class: "greeting" }, h("span", { class: "logo", "aria-hidden": "true" }, icon("gate")), greetingText()),
          keySetup,
          composerWrap,
          starters,
        ),
      );
    } else {
      root.append(head, thread, keySetup, composerWrap);
      keySetup.style.margin = "0 auto 12px";
    }
    setTitle();
    refreshKeyState();
  };

  // ---- rendering turns
  const userTurn = (content) => h("div", { class: "turn user" }, h("div", { class: "bubble", text: content }));
  const assistantTurn = (content, { citations, error, onRetry } = {}) => {
    const answer = h("div", { class: "answer" });
    if (error) answer.textContent = content;
    else answer.append(renderMarkdown(content));
    const actions = h(
      "div",
      { class: "msg-actions" },
      h("button", { class: "act-btn", type: "button", title: "Copy", "aria-label": "Copy reply", onclick: (e) => copyText(content, e.currentTarget) }, icon("copy")),
      h("button", { class: "act-btn", type: "button", title: "Read aloud", "aria-label": "Read reply aloud", onclick: () => speakText(content) }, icon("speaker")),
      onRetry ? h("button", { class: "act-btn", type: "button", title: "Retry", "aria-label": "Retry", onclick: onRetry }, icon("retry")) : null,
    );
    return h("div", { class: `turn assistant${error ? " error" : ""}` }, answer, citationChips(citations), error ? null : actions, error && onRetry ? h("div", { class: "msg-actions" }, h("button", { class: "btn sm", type: "button", text: "Try again", onclick: onRetry })) : null);
  };
  const renderTurns = () => {
    threadInner.textContent = "";
    turns.forEach((t, idx) => {
      if (t.note) threadInner.append(h("div", { class: "summary-note", text: t.note }));
      else if (t.role === "user") threadInner.append(userTurn(t.content));
      else threadInner.append(assistantTurn(t.content, { citations: t.citations, onRetry: idx === turns.length - 1 ? retryLast : null }));
    });
  };

  // ---- voice: local Piper/Whisper on the server, browser speech as a fallback
  let speaking = null;
  function stopSpeaking() {
    if (speaking) { speaking.pause(); URL.revokeObjectURL(speaking.src); speaking = null; }
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }
  async function speakText(text) {
    const clean = text.replace(/```[\s\S]*?```/g, " (code omitted) ").replace(/[*_#>`]/g, "").trim().slice(0, 1500);
    if (!clean) return;
    stopSpeaking();
    try {
      const resp = await apiWithKey("/v1/audio/speech", { method: "POST", body: JSON.stringify({ input: clean }) });
      const audio = new Audio(URL.createObjectURL(await resp.blob()));
      speaking = audio;
      audio.addEventListener("ended", () => { URL.revokeObjectURL(audio.src); if (speaking === audio) speaking = null; });
      await audio.play();
    } catch (err) {
      if (window.speechSynthesis && (err.status === 501 || err.status === 0)) window.speechSynthesis.speak(new SpeechSynthesisUtterance(clean));
      else toast(`Couldn't read the reply aloud: ${err.message}`, "bad");
    }
  }
  let recorder = null;
  const setRecording = (on) => {
    micBtn.classList.toggle("recording", on);
    micBtn.replaceChildren(icon(on ? "stopSq" : "mic"));
    micBtn.setAttribute("aria-label", on ? "Stop dictating" : "Dictate a message");
  };
  micBtn.addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") { recorder.stop(); return; }
    if (!prefs.apiKey) { toast("Add an API key first.", "bad"); return; }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { toast("This browser can't record audio here (it needs https or localhost).", "bad"); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch (err) { toast(`Microphone unavailable: ${err.message}`, "bad"); return; }
    stopSpeaking();
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.addEventListener("dataavailable", (e) => { if (e.data.size) chunks.push(e.data); });
    recorder.addEventListener("stop", async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      recorder = null;
      setRecording(false);
      if (!blob.size) return;
      micBtn.disabled = true;
      micBtn.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }));
      try {
        const resp = await apiWithKey("/v1/audio/transcriptions", { method: "POST", body: blob, headers: { "Content-Type": blob.type || "audio/webm" } });
        const text = ((await resp.json()).text || "").trim();
        if (text) send(prompt.value ? `${prompt.value} ${text}` : text);
        else toast("Didn't catch that — try again.", "bad");
      } catch (err) {
        toast(err.status === 501 ? "Voice isn't installed on the server: pip install 'localgate[voice]'" : `Transcription failed: ${err.message}`, "bad");
      } finally {
        micBtn.disabled = false;
        micBtn.replaceChildren(icon("mic"));
      }
    });
    recorder.start();
    setRecording(true);
  });

  // ---- sending
  function retryLast() {
    if (chatAbort) return;
    const lastUser = [...turns].reverse().find((t) => t.role === "user");
    if (!lastUser) return;
    while (turns.length && turns[turns.length - 1].role !== "user") turns.pop();
    turns.pop();
    send(lastUser.content);
  }

  async function send(raw) {
    const text = String(raw || "").trim();
    if (!text || chatAbort) return;
    if (!prefs.apiKey) { refreshKeyState(); toast("Connect an API key first.", "bad"); return; }
    if (!currentChatId) {
      currentChatId = crypto.randomUUID();
      prefs.sessions = [{ id: currentChatId, label: text.slice(0, 60) }, ...prefs.sessions.filter((s) => s.id !== currentChatId)];
      prefs.activeSession = currentChatId;
      history.replaceState(null, "", `#/chat/${encodeURIComponent(currentChatId)}`);
      renderHistory();
    }
    const context = turns.filter((t) => !t.note).slice(-12).map((t) => ({ role: t.role, content: t.content }));
    turns.push({ role: "user", content: text });
    if (turns.length === 1) layout();
    renderTurns();
    prompt.value = "";
    autosize();

    const answer = h("div", { class: "answer" }, h("div", { class: "thinking", "aria-label": "Thinking" }, h("span"), h("span"), h("span")));
    const live = h("div", { class: "turn assistant" }, answer);
    threadInner.append(live);
    scrollDown();
    chatAbort = new AbortController();
    setStreaming(true);

    let reply = "";
    let citations = null;
    let searchError = null;
    let pending = false;
    const paint = () => {
      pending = false;
      answer.replaceChildren(renderMarkdown(reply || " "));
      answer.classList.add("typing");
      const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
      if (nearBottom) scrollDown();
    };
    try {
      const body = {
        model: modelSelect.value || undefined,
        stream: true,
        messages: [...context, { role: "user", content: text }],
      };
      if (webPill.getAttribute("aria-pressed") === "true") body.localgate = { web_search: true };
      const resp = await apiWithKey("/v1/chat/completions", {
        method: "POST",
        headers: { "X-Session-Id": currentChatId },
        body: JSON.stringify(body),
        signal: chatAbort.signal,
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          let chunk;
          try { chunk = JSON.parse(data); } catch { continue; }
          if (chunk.error) throw new Error(chunk.error.message || "The backend failed mid-stream.");
          if (chunk.localgate) {
            citations = chunk.localgate.citations || null;
            searchError = chunk.localgate.web_search_error || null;
          }
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            reply += delta;
            if (!pending) { pending = true; requestAnimationFrame(paint); }
          }
        }
      }
      turns.push({ role: "assistant", content: reply || "(no reply)", citations });
      renderTurns();
      if (searchError) threadInner.append(h("div", { class: "summary-note", text: `Web search unavailable: ${searchError}` }));
      if (prefs.speak && reply) speakText(reply);
    } catch (err) {
      if (err.name === "AbortError") {
        if (reply) turns.push({ role: "assistant", content: `${reply}\n\n*(stopped)*` });
        renderTurns();
      } else {
        live.replaceWith(assistantTurn(err.message, { error: true, onRetry: retryLast }));
      }
    } finally {
      chatAbort = null;
      setStreaming(false);
      scrollDown();
      prompt.focus();
    }
  }

  // ---- open an existing conversation
  layout();
  if (!isNew) {
    threadInner.append(loading("Loading conversation…"));
    root.textContent = "";
    root.append(head, thread, keySetup, composerWrap);
    setTitle();
    refreshKeyState();
    const load = (data, note) => {
      if (note) turns.push({ note });
      if (data.summary) turns.push({ note: `Earlier in this conversation (summarized): ${data.summary}` });
      (data.messages || []).forEach((m) => turns.push({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
      if (!turns.some((t) => !t.note)) { turns.length = 0; layout(); return; }
      if (!prefs.sessions.some((x) => x.id === chatId)) {
        const firstUser = turns.find((t) => t.role === "user");
        prefs.sessions = [{ id: chatId, label: (firstUser?.content || "Conversation").slice(0, 60) }, ...prefs.sessions];
        renderHistory();
      }
      layout();
      renderTurns();
      requestAnimationFrame(scrollDown);
    };
    try {
      if (!prefs.apiKey) throw Object.assign(new Error("no key"), { status: 0 });
      const resp = await apiWithKey(`/v1/conversations/${encodeURIComponent(chatId)}`);
      load(await resp.json());
    } catch (err) {
      if (err.status !== 404 && err.status !== 0 && err.status !== 401) {
        threadInner.replaceChildren(errorState(err));
      } else {
        // History belongs to the API key that created it; a signed-in admin can still read it.
        try {
          const owned = (await api("/admin/sessions?limit=1000")).filter((s) => s.session_id === chatId);
          if (!owned.length) { turns.length = 0; layout(); }
          else {
            const entry = owned[0];
            const data = await api(`/admin/sessions/${encodeURIComponent(entry.api_key_id)}/${encodeURIComponent(chatId)}`);
            load(data, `Read-only view: this conversation belongs to another API key (${entry.api_key_id.slice(0, 8)}…). Messages you send now are stored under your current key.`);
          }
        } catch (err2) {
          threadInner.replaceChildren(errorState(err2));
        }
      }
    }
  }
  prompt.focus();

  // ---- models (chat-capable only)
  try {
    const models = await api("/admin/models");
    models.filter((m) => m.can_chat !== false).forEach((m) => {
      modelSelect.append(h("option", { value: m.name, text: m.name }));
    });
    if (prefs.model && models.some((m) => m.name === prefs.model)) modelSelect.value = prefs.model;
  } catch { /* keep the default option */ }
}

// ---------------------------------------------------------------- keys

// ---------------------------------------------------------------- key limits

// Model names for the "Models" suggestions — fetched once per page load.
let modelNamesPromise = null;
function modelNames() {
  modelNamesPromise ||= api("/admin/models")
    .then((rows) => rows.filter((m) => m.can_chat !== false).map((m) => m.name))
    .catch(() => []);
  return modelNamesPromise;
}

const NETWORK_CHOICES = [
  ["both", "Both", "on your network and over the internet"],
  ["local", "Local network", "this network and Tailscale only"],
  ["internet", "Internet", "only through the internet address"],
];

function field(label, control, hint) {
  return h("label", { class: "field" }, h("span", { text: label }), control, hint ? h("small", { class: "field-hint", text: hint }) : null);
}

/** The optional per-key limits, shared by the create form, the edit dialog and
 *  the Share page. `values()` returns them in the /admin/keys shape (null = no limit). */
function limitsFields(initial = {}) {
  // Models: tick the ones this key may use, or "Any model".
  const chosen = new Set(initial.allowed_models || []);
  const anyModel = h("input", { type: "checkbox" });
  anyModel.checked = chosen.size === 0;
  const picks = new Map();
  const modelGroup = h("div", { class: "model-picks", role: "group", "aria-label": "Models this key may use" }, h("span", { class: "muted small", text: "Loading models…" }));
  anyModel.addEventListener("change", () => {
    if (anyModel.checked) for (const box of picks.values()) box.checked = false;
  });
  modelNames().then((names) => {
    const all = [...new Set([...chosen, ...names])];
    modelGroup.replaceChildren(
      h("label", { class: "pick" }, anyModel, h("span", { text: "Any model" })),
      ...all.map((name) => {
        const box = h("input", { type: "checkbox" });
        box.checked = chosen.has(name);
        box.addEventListener("change", () => { if (box.checked) anyModel.checked = false; });
        picks.set(name, box);
        return h("label", { class: "pick" }, box, h("span", { class: "mono", text: name }));
      }),
    );
    if (!all.length) modelGroup.append(h("span", { class: "muted small", text: "No models found — the key will allow any model." }));
  });
  const network = h(
    "div",
    { class: "segmented", role: "radiogroup", "aria-label": "Where this key works" },
    NETWORK_CHOICES.map(([value, label, hint]) => {
      const radio = h("input", { type: "radio", name: `net-${Math.random().toString(36).slice(2)}`, value });
      radio.checked = (initial.network_access || "both") === value;
      return h("label", { class: "seg", title: hint }, radio, h("span", { text: label }));
    }),
  );
  // Radios in one group share a name; give them the first one's.
  const radios = [...network.querySelectorAll("input")];
  for (const radio of radios) radio.name = radios[0].name;
  const maxTokens = h("input", { type: "number", min: "1", placeholder: "no cap", value: initial.max_tokens ?? "" });
  const quota = h("input", { type: "number", min: "1", placeholder: "unlimited", value: initial.token_quota ?? "" });
  const period = h("select", {}, h("option", { value: "month", text: "per month" }), h("option", { value: "day", text: "per day" }));
  period.value = initial.quota_period || "month";
  const expires = h("input", { type: "date", value: initial.expires_at ? initial.expires_at.slice(0, 10) : "" });
  const concurrent = h("input", { type: "number", min: "1", placeholder: "no limit", value: initial.max_concurrent ?? "" });
  const store = h("input", { type: "checkbox" });
  store.checked = initial.store_conversations !== false;
  const element = h(
    "div",
    { class: "limits" },
    field("Models", modelGroup),
    field("Works on", network, NETWORK_CHOICES.map(([, label, hint]) => `${label}: ${hint}`).join(" · ")),
    h("div", { class: "field-row" }, field("Max reply tokens", maxTokens), field("Requests at once", concurrent)),
    h("div", { class: "field-row" }, field("Token quota", quota), field("Quota period", period)),
    h("div", { class: "field-row" }, field("Expires", expires)),
    h("label", { class: "check" }, store, h("span", { text: "Store conversations for memory (you can read them in Sessions)" })),
  );
  const num = (input) => (input.value === "" ? null : Number(input.value));
  return {
    element,
    values: () => {
      const names = [...picks].filter(([, box]) => box.checked).map(([name]) => name);
      return {
        allowed_models: anyModel.checked || !names.length ? null : names,
        network_access: radios.find((r) => r.checked)?.value || "both",
        max_tokens: num(maxTokens),
        token_quota: num(quota),
        quota_period: period.value,
        // The whole chosen day counts: the key stops at the end of it, UTC.
        expires_at: expires.value ? new Date(`${expires.value}T23:59:59Z`).toISOString() : null,
        max_concurrent: num(concurrent),
        store_conversations: store.checked,
      };
    },
  };
}

/** One-line summary of a key's limits for tables. */
function limitsSummary(k) {
  const parts = [`${k.rate_limit_per_min}/min`];
  if (k.allowed_models?.length) parts.push(k.allowed_models.length === 1 ? k.allowed_models[0] : `${k.allowed_models.length} models`);
  if (k.max_tokens) parts.push(`≤${compact(k.max_tokens)} tokens/reply`);
  if (k.token_quota) parts.push(`${compact(k.tokens_this_period || 0)} / ${compact(k.token_quota)} per ${k.quota_period}`);
  if (k.max_concurrent) parts.push(`${k.max_concurrent} at once`);
  if (k.expires_at) parts.push(`${k.expired ? "expired" : "until"} ${fmtDate(k.expires_at).split(",")[0]}`);
  if (k.store_conversations === false) parts.push("no history");
  if (k.network_access && k.network_access !== "both") parts.push(k.network_access === "local" ? "local network only" : "internet only");
  return parts.join(" · ");
}

async function viewKeys(root) {
  const listBody = h("div", {}, loading());
  const usagePanel = h("div", { class: "card-body" }, h("p", { class: "muted", text: "Select a key to see its usage." }));
  const reveal = h("div", {});
  const nameInput = h("input", { type: "text", placeholder: "my-app", required: true, maxlength: "100" });
  const limitInput = h("input", { type: "number", min: "1", value: "60" });
  const createLimits = limitsFields();

  const createForm = h(
    "form",
    {
      class: "card-body",
      onsubmit: async (e) => {
        e.preventDefault();
        const name = nameInput.value.trim();
        if (!name) return;
        try {
          const key = await api("/admin/keys", { method: "POST", body: { name, rate_limit_per_min: Number(limitInput.value) || undefined, ...createLimits.values() } });
          nameInput.value = "";
          showReveal(key);
          toast("Key created.", "ok");
          load();
        } catch (err) { toast(err.message, "bad"); }
      },
    },
    h(
      "div",
      { class: "field-row" },
      h("label", { class: "field" }, h("span", { text: "Name" }), nameInput),
      h("label", { class: "field" }, h("span", { text: "Requests / min" }), limitInput),
      h("button", { class: "btn primary", type: "submit", text: "Create key" }),
    ),
    h("details", { class: "more-limits" }, h("summary", { text: "More limits — models, quota, expiry…" }), createLimits.element),
    reveal,
  );

  function showReveal(key, note) {
    reveal.textContent = "";
    const copyBtn = h("button", {
      class: "btn sm",
      type: "button",
      text: "Copy",
      onclick: async () => {
        try { await navigator.clipboard.writeText(key.api_key); toast("Copied."); } catch { toast("Copy failed — select the key and copy it manually.", "bad"); }
      },
    });
    const useBtn = h("button", {
      class: "btn sm",
      type: "button",
      text: "Use for chat",
      onclick: () => { prefs.setApiKey(key.api_key, false); toast("This tab now chats with the new key."); },
    });
    reveal.append(
      h(
        "div",
        { class: "secret", role: "status" },
        h("code", { text: key.api_key }),
        h("p", {
          class: "small muted",
          text: note || "Copy it now. While signed in as admin you can show it again later with “Show key” (keys created with the default admin key can't be shown again).",
        }),
        h("div", { class: "actions" }, copyBtn, useBtn),
      ),
    );
  }

  async function showUsage(k) {
    usagePanel.textContent = "";
    usagePanel.append(loading());
    try {
      const u = await api(`/admin/usage/${encodeURIComponent(k.id)}`);
      usagePanel.textContent = "";
      usagePanel.append(
        h("h3", { text: k.name }),
        h(
          "dl",
          { class: "kv" },
          [
            ["Requests", nf.format(u.request_count)],
            ["Prompt tokens", nf.format(u.prompt_tokens)],
            ["Completion tokens", nf.format(u.completion_tokens)],
            ["Total tokens", nf.format(u.total_tokens)],
            ["Average latency", `${nf.format(u.avg_latency_ms)} ms`],
          ].map(([dt, dd]) => [h("dt", { text: dt }), h("dd", { text: dd })]),
        ),
      );
    } catch (err) {
      usagePanel.textContent = "";
      usagePanel.append(errorState(err));
    }
  }

  function editLimits(k) {
    const limits = limitsFields(k);
    const rate = h("input", { type: "number", min: "1", value: String(k.rate_limit_per_min) });
    const dialog = h(
      "dialog",
      { class: "dialog wide", "aria-label": `Limits for ${k.name}` },
      h(
        "form",
        { method: "dialog" },
        h("h2", { text: `Limits for “${k.name}”` }),
        h("p", { class: "muted small", text: "Changes apply on the next request — the key itself stays the same." }),
        h("div", { class: "field-row" }, field("Requests / min", rate)),
        limits.element,
        h("div", { class: "dialog-actions" }, h("button", { value: "cancel", class: "btn ghost", text: "Cancel" }), h("button", { value: "ok", class: "btn primary", text: "Save limits" })),
      ),
    );
    dialog.addEventListener("close", async () => {
      if (dialog.returnValue === "ok") {
        try {
          await api(`/admin/keys/${encodeURIComponent(k.id)}`, { method: "PATCH", body: { rate_limit_per_min: Number(rate.value) || k.rate_limit_per_min, ...limits.values() } });
          toast("Limits updated.", "ok");
          load();
        } catch (err) { toast(err.message, "bad"); }
      }
      dialog.remove();
    });
    document.body.append(dialog);
    dialog.showModal();
  }

  async function revealKey(k) {
    try {
      const data = await api(`/admin/keys/${encodeURIComponent(k.id)}/reveal`, { method: "POST" });
      showReveal({ ...k, api_key: data.api_key }, `Full key for “${k.name}”. Treat it like a password.`);
      reveal.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { toast(err.message, "bad"); }
  }

  async function revoke(k) {
    const ok = await confirmAction("Revoke this key?", `“${k.name}” stops working immediately. This cannot be undone.`, "Revoke key");
    if (!ok) return;
    try {
      await api(`/admin/keys/${encodeURIComponent(k.id)}`, { method: "DELETE" });
      toast("Key revoked.", "ok");
      load();
    } catch (err) { toast(err.message, "bad"); }
  }

  async function load() {
    try {
      const keys = await api("/admin/keys");
      listBody.textContent = "";
      listBody.append(
        table(
          [
            { label: "Name", value: (k) => h("span", {}, k.name, h("span", { class: "cell-sub mono", text: k.key_prefix ? `${k.key_prefix}…` : "issued before v0.6" })) },
            { label: "Status", value: (k) => (k.revoked ? badge("revoked", "bad") : k.expired ? badge("expired", "warn") : badge("active", "ok")) },
            {
              label: "Limits",
              value: (k) => {
                const span = h("span", { class: "limits-cell" }, limitsSummary(k), " ");
                if (!k.revoked) {
                  span.append(
                    h("button", {
                      class: "link-btn small",
                      type: "button",
                      text: "edit",
                      "aria-label": `Edit limits for ${k.name}`,
                      onclick: (e) => { e.stopPropagation(); editLimits(k); },
                    }),
                  );
                }
                return span;
              },
            },
            { label: "Created", value: (k) => when(k.created_at) },
            { label: "Last used", value: (k) => (k.last_used_at ? when(k.last_used_at) : "never") },
            {
              label: "",
              value: (k) =>
                k.revoked
                  ? ""
                  : h(
                      "span",
                      { class: "actions" },
                      k.revealable
                        ? h("button", { class: "link-btn", type: "button", text: "Show key", "aria-label": `Show the full key for ${k.name}`, onclick: (e) => { e.stopPropagation(); revealKey(k); } })
                        : h("span", { class: "cell-sub", title: "Created before keys could be revealed, or with the default admin key. Create a new key if you've lost it.", text: "not viewable" }),
                      h("button", { class: "link-btn danger", type: "button", text: "Revoke", "aria-label": `Revoke ${k.name}`, onclick: (e) => { e.stopPropagation(); revoke(k); } }),
                    ),
            },
          ],
          keys,
          {
            emptyText: "No keys yet. Create one below.",
            onRowClick: (k, tr) => {
              $$("tr.selected", listBody).forEach((r) => r.classList.remove("selected"));
              tr.classList.add("selected");
              showUsage(k);
            },
          },
        ),
      );
    } catch (err) {
      listBody.textContent = "";
      listBody.append(errorState(err));
    }
  }

  root.append(
    pageHead("API keys", "Issue, limit and revoke the keys clients and devices use."),
    h("div", { class: "grid split" }, card("Keys", listBody), card("Usage for selected key", usagePanel)),
    card("Create a key", createForm),
  );
  await load();
}

// ---------------------------------------------------------------- usage

async function viewUsage(root) {
  const daysSelect = h(
    "select",
    { "aria-label": "Period" },
    [7, 14, 30, 90].map((d) => h("option", { value: String(d), text: `Last ${d} days`, selected: d === 14 })),
  );
  const exportBtn = h("button", { class: "btn", type: "button", text: "Export CSV", disabled: true });
  const body = h("div", {}, loading());
  let current = null;

  const csvCell = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  exportBtn.addEventListener("click", () => {
    if (!current) return;
    const rows = [["section", "name", "requests", "prompt_tokens", "completion_tokens", "total_tokens"]];
    current.daily.forEach((d) => rows.push(["day", d.date, d.requests, d.prompt_tokens, d.completion_tokens, d.prompt_tokens + d.completion_tokens]));
    current.by_key.forEach((k) => rows.push(["key", k.name, k.request_count, "", "", k.total_tokens]));
    current.by_model.forEach((m) => rows.push(["model", m.model, m.request_count, "", "", m.total_tokens]));
    downloadBlob(`localgate-usage-${daysSelect.value}d.csv`, rows.map((r) => r.map(csvCell).join(",")).join("\n"), "text/csv");
  });

  async function load() {
    body.textContent = "";
    body.append(loading());
    try {
      current = await api(`/admin/usage?days=${daysSelect.value}`);
      exportBtn.disabled = false;
      const t = current.totals;
      body.textContent = "";
      body.append(
        h(
          "div",
          { class: "grid stats" },
          stat("Requests", nf.format(t.request_count), "all time"),
          stat("Tokens", compact(t.total_tokens), "all time"),
          stat("Average latency", `${nf.format(Math.round(t.avg_latency_ms))} ms`, "per request"),
          stat("In period", nf.format(current.daily.reduce((n, d) => n + (d.requests || 0), 0)), `requests, last ${daysSelect.value} days`),
        ),
        card("Tokens per day", h("div", { class: "card-body" }, tokenChart(current.daily))),
        h(
          "div",
          { class: "grid two" },
          card(
            "By key",
            table(
              [
                { label: "Key", value: (k) => h("span", {}, k.name, k.revoked ? h("span", { class: "cell-sub", text: "revoked" }) : null) },
                { label: "Requests", value: (k) => nf.format(k.request_count), num: true },
                { label: "Tokens", value: (k) => nf.format(k.total_tokens), num: true },
              ],
              current.by_key,
              { emptyText: "No keys." },
            ),
          ),
          card(
            "By model",
            table(
              [
                { label: "Model", value: (m) => h("span", { class: "mono", text: m.model }) },
                { label: "Requests", value: (m) => nf.format(m.request_count), num: true },
                { label: "Tokens", value: (m) => nf.format(m.total_tokens), num: true },
              ],
              current.by_model,
              { emptyText: "No requests yet." },
            ),
          ),
        ),
        card(
          "Daily",
          table(
            [
              { label: "Date", value: (d) => d.date },
              { label: "Requests", value: (d) => nf.format(d.requests ?? 0), num: true },
              { label: "Prompt", value: (d) => nf.format(d.prompt_tokens), num: true },
              { label: "Completion", value: (d) => nf.format(d.completion_tokens), num: true },
            ],
            [...current.daily].reverse(),
            { emptyText: "No usage in this period." },
          ),
        ),
      );
    } catch (err) {
      body.textContent = "";
      body.append(errorState(err));
    }
  }
  daysSelect.addEventListener("change", load);
  root.append(pageHead("Usage", "Token accounting across every key and model.", daysSelect, exportBtn), body);
  await load();
}

// ---------------------------------------------------------------- sessions

function renderLedger(ledger) {
  if (!ledger || !Object.values(ledger).some((v) => (Array.isArray(v) ? v.length : v && (typeof v !== "object" || Object.keys(v).length)))) {
    return empty("No agent ledger for this session.");
  }
  const list = (title, items) =>
    items?.length ? h("div", { class: "ledger-item" }, h("h4", { text: title }), h("ul", {}, items.map((i) => h("li", { text: i })))) : null;
  const files = Object.entries(ledger.files_changed || {});
  return h(
    "div",
    { class: "ledger" },
    ledger.goal ? h("div", { class: "ledger-item" }, h("h4", { text: "Original goal" }), h("div", { class: "goal", text: ledger.goal })) : null,
    ledger.current_task && ledger.current_task !== ledger.goal
      ? h("div", { class: "ledger-item" }, h("h4", { text: "Current task" }), h("div", { class: "goal", text: ledger.current_task }))
      : null,
    list("Decisions", ledger.decisions),
    list("Constraints", ledger.constraints),
    list("Open to-dos", ledger.open_todos),
    files.length
      ? h(
          "div",
          { class: "ledger-item" },
          h("h4", { text: "Files changed" }),
          h("div", { class: "files" }, files.map(([path, op]) => badge(`${path} · ${op}`, op === "deleted" ? "bad" : op === "created" ? "ok" : "accent"))),
        )
      : null,
    ledger.files_read?.length
      ? h("div", { class: "ledger-item" }, h("h4", { text: "Files read" }), h("div", { class: "files" }, ledger.files_read.map((p) => badge(p))))
      : null,
    ledger.last_error ? h("div", { class: "ledger-item" }, h("h4", { text: "Last tool error" }), h("div", { class: "callout bad", text: ledger.last_error })) : null,
    ledger.notes ? h("div", { class: "ledger-item" }, h("h4", { text: "Compacted history" }), h("p", { text: ledger.notes })) : null,
  );
}

async function viewSessions(root) {
  const listBody = h("div", {}, loading());
  const detail = h("div", {}, card("Session", empty("Select a session to inspect its history, summary and ledger.")));
  const ownerInput = h("input", { type: "text", placeholder: "api key id", required: true });
  const sessionInput = h("input", { type: "text", placeholder: "session id", required: true });
  const queryInput = h("input", { type: "search", placeholder: "What would memory recall for…", required: true });
  const results = h("div", {});

  const inspector = h(
    "form",
    {
      class: "card-body",
      onsubmit: async (e) => {
        e.preventDefault();
        results.textContent = "";
        results.append(loading("Embedding and ranking…"));
        try {
          const data = await api("/admin/memory/search", {
            method: "POST",
            body: { api_key_id: ownerInput.value.trim(), session_id: sessionInput.value.trim(), query: queryInput.value.trim() },
          });
          results.textContent = "";
          if (!data.results.length) { results.append(empty("Nothing stored for this session.")); return; }
          let markerPlaced = false;
          data.results.forEach((r) => {
            if (!markerPlaced && r.score < data.min_score) {
              markerPlaced = true;
              results.append(h("div", { class: "threshold" }, h("span", { text: `min_score ${data.min_score} — below this is not injected` })));
            }
            results.append(
              h(
                "div",
                { class: `score-row${r.score < data.min_score ? " below" : ""}` },
                h("span", { class: "mono", text: r.score.toFixed(3) }),
                h("span", { text: r.content }),
              ),
            );
          });
        } catch (err) {
          results.textContent = "";
          results.append(errorState(err));
        }
      },
    },
    h(
      "div",
      { class: "field-row" },
      h("label", { class: "field" }, h("span", { text: "Owner (API key id)" }), ownerInput),
      h("label", { class: "field" }, h("span", { text: "Session id" }), sessionInput),
    ),
    h("div", { class: "field-row" }, h("label", { class: "field" }, h("span", { text: "Query" }), queryInput), h("button", { class: "btn primary", type: "submit", text: "Search memory" })),
    results,
  );

  async function open(s, tr) {
    $$("tr.selected", listBody).forEach((r) => r.classList.remove("selected"));
    tr?.classList.add("selected");
    ownerInput.value = s.api_key_id;
    sessionInput.value = s.session_id;
    detail.textContent = "";
    detail.append(card("Session", loading()));
    try {
      const d = await api(`/admin/sessions/${encodeURIComponent(s.api_key_id)}/${encodeURIComponent(s.session_id)}`);
      detail.textContent = "";
      detail.append(
        h(
          "div",
          { class: "grid two" },
          card(
            "Messages",
            h(
              "div",
              { class: "card-body" },
              d.summary ? h("div", { class: "summary-note", text: `Rolling summary: ${d.summary}` }) : null,
              h("p", { class: "muted small", text: `${d.messages.length} message(s) · ${d.memory_chunks} memory chunk(s)` }),
              d.messages.length
                ? h("div", { class: "message-log" }, d.messages.map((m) => h("div", { class: `log-msg ${m.role}` }, h("span", { class: "role", text: `${m.role} · ${ago(m.created_at)}`, title: fmtDate(m.created_at) }), m.content)))
                : empty("No stored messages."),
            ),
            h("span", { class: "mono small muted", text: shortId(d.session_id) }),
          ),
          card("Agent session ledger", h("div", { class: "card-body" }, renderLedger(d.ledger))),
        ),
      );
    } catch (err) {
      detail.textContent = "";
      detail.append(card("Session", errorState(err)));
    }
  }

  root.append(
    pageHead("Memory & sessions", "Every stored conversation, its rolling summary and agent ledger — and what memory would recall."),
    card("Recent sessions", listBody),
    detail,
    card("Memory inspector", inspector),
  );

  try {
    const [sessions, keys] = await Promise.all([api("/admin/sessions?limit=200"), api("/admin/keys").catch(() => [])]);
    const keyNames = Object.fromEntries(keys.map((k) => [k.id, k.name]));
    listBody.textContent = "";
    listBody.append(
      table(
        [
          { label: "Session", value: (s) => h("span", { class: "mono", title: s.session_id, text: shortId(s.session_id, 24) }) },
          {
            label: "Owner key",
            value: (s) =>
              keyNames[s.api_key_id]
                ? h("span", { title: s.api_key_id }, keyNames[s.api_key_id], h("span", { class: "cell-sub mono", text: shortId(s.api_key_id) }))
                : h("span", { class: "mono", title: s.api_key_id, text: s.api_key_id === "local-agent" ? "localgate code" : shortId(s.api_key_id) }),
          },
          { label: "Messages", value: (s) => nf.format(s.message_count), num: true },
          { label: "Last activity", value: (s) => when(s.last_message_at) },
        ],
        sessions,
        { emptyText: "No conversations stored yet.", onRowClick: open },
      ),
    );
  } catch (err) {
    listBody.textContent = "";
    listBody.append(errorState(err));
  }
}

// ---------------------------------------------------------------- agent runs

async function viewAgent(root) {
  const body = h("div", {}, loading());
  root.append(pageHead("Agent runs", "Every task `localgate code` worked on, step by step — what it read, changed and ran, and where it struggled."), body);
  const data = await api("/admin/tool-calls?limit=1000");
  body.textContent = "";
  if (!data.recent.length) {
    body.append(card("No agent activity yet", empty("Run `localgate code` in a project and each task will appear here as a timeline of tool calls.")));
    return;
  }

  const TOOL_ICONS = {
    read_file: "file", list_directory: "folder", search_files: "search", git_status: "git", git_diff: "git",
    edit_file: "edit", write_file: "edit", create_file: "plus", update_file: "edit", delete_file: "file",
    run_command: "terminal", web_search: "globe", fetch_url: "globe", delegate_task: "users", todo_write: "list",
  };
  const argsSummary = (args) =>
    Object.entries(args || {})
      .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join("  ");

  // Group calls into runs (one user task each), oldest step first.
  const runsById = new Map();
  for (const call of data.recent) {
    const key = call.run_id || `${call.session_id}:${call.created_at.slice(0, 16)}`;
    if (!runsById.has(key)) runsById.set(key, []);
    runsById.get(key).push(call);
  }
  const runs = [...runsById.entries()].map(([id, calls]) => {
    const steps = [...calls].sort((a, b) => a.created_at.localeCompare(b.created_at));
    const failed = steps.filter((s) => !s.success).length;
    return {
      id,
      steps,
      failed,
      timeouts: steps.filter((s) => s.timed_out).length,
      model: steps[0].model || "unknown model",
      session: steps[0].session_id,
      started: steps[0].created_at,
      duration: steps.reduce((n, s) => n + (s.duration_ms || 0), 0),
      changed: [...new Set(steps.filter((s) => s.success && /write|create|update|edit|delete/.test(s.tool_name)).map((s) => s.arguments?.path).filter(Boolean))],
    };
  });

  const modelFilter = h("select", { "aria-label": "Filter by model" }, h("option", { value: "", text: "All models" }), data.by_model.map((m) => h("option", { value: m.name, text: m.name })));
  const failuresOnly = h("input", { type: "checkbox" });
  const runsHost = h("div", { class: "runs" });

  const stepEl = (s) =>
    h(
      "li",
      { class: `step${s.timed_out ? " timeout" : s.success ? "" : " failed"}` },
      h("div", { class: "step-ico" }, icon(TOOL_ICONS[s.tool_name] || (s.tool_name.startsWith("mcp__") ? "plug" : "tool"))),
      h(
        "div",
        { class: "step-body" },
        h("div", { class: "step-name", text: s.tool_name }),
        argsSummary(s.arguments) ? h("div", { class: "step-args", title: argsSummary(s.arguments), text: argsSummary(s.arguments) }) : null,
        s.error ? h("div", { class: "step-error", text: s.error }) : null,
      ),
      h("div", { class: "step-side" }, s.timed_out ? badge("timeout", "warn") : s.success ? null : badge("failed", "bad"), h("div", { text: `${nf.format(s.duration_ms)} ms` })),
    );

  const runEl = (r, open) => {
    const status = r.failed === 0 ? "" : r.failed === r.steps.length ? "bad" : "warn";
    const details = h(
      "details",
      { class: "run", open },
      h(
        "summary",
        { class: "run-head" },
        h("span", { class: `run-status ${status}`, title: r.failed ? `${r.failed} failed step(s)` : "all steps succeeded" }),
        h(
          "div",
          {},
          h("div", { class: "run-title", text: `${r.steps.length} step${r.steps.length === 1 ? "" : "s"}${r.changed.length ? ` · changed ${r.changed.join(", ")}` : ""}` }),
          h(
            "div",
            { class: "run-meta" },
            h("span", { class: "mono", text: r.model }),
            when(r.started),
            h("span", { text: `${nf.format(r.duration)} ms in tools` }),
            r.failed ? h("span", { text: `${r.failed} failed` }) : null,
          ),
        ),
        h("span", { class: "chev" }, icon("chevron", 16)),
      ),
      h("ol", { class: "steps" }, r.steps.map(stepEl)),
    );
    return details;
  };

  const renderRuns = () => {
    const shown = runs.filter((r) => (!modelFilter.value || r.model === modelFilter.value) && (!failuresOnly.checked || r.failed));
    runsHost.textContent = "";
    if (!shown.length) runsHost.append(empty("No runs match these filters."));
    shown.slice(0, 40).forEach((r, i) => runsHost.append(runEl(r, i === 0)));
  };
  [modelFilter, failuresOnly].forEach((el) => el.addEventListener("change", renderRuns));

  const total = data.recent.length;
  const ok = data.recent.filter((r) => r.success).length;
  body.append(
    h(
      "div",
      { class: "grid stats" },
      stat("Runs", nf.format(runs.length), `${nf.format(total)} tool calls`),
      stat("Tool success", `${Math.round((ok / total) * 100)}%`, `${nf.format(total - ok)} failed`),
      stat("Clean runs", nf.format(runs.filter((r) => !r.failed).length), "no failed step"),
      stat("Models", String(data.by_model.length), "seen in recent runs"),
    ),
    h(
      "div",
      { class: "grid two" },
      card("Reliability by model", h("div", { class: "card-body" }, data.by_model.map((m) => meterRow(m.name, m.success_rate, `${m.ok}/${m.calls}`)))),
      card("Reliability by tool", h("div", { class: "card-body" }, data.by_tool.map((t) => meterRow(t.name, t.success_rate, `${t.ok}/${t.calls}`)))),
    ),
    h("div", { class: "page-head" }, h("h2", { text: "Recent runs" }), h("div", { class: "filters" }, modelFilter, h("label", { class: "check" }, failuresOnly, "Only runs with failures"))),
    runsHost,
  );
  renderRuns();
}

// ---------------------------------------------------------------- models

async function viewModels(root) {
  const body = h("div", {}, loading("Asking the backend…"));
  root.append(pageHead("Models", "What the inference backend serves, and what each model can do."), card("Available models", body));
  const models = await api("/admin/models");
  body.textContent = "";
  body.append(
    table(
      [
        { label: "Model", value: (m) => h("span", { class: "mono", text: m.name }) },
        // Only worth a column when several machines serve models (LOCALGATE_BACKENDS).
        ...(new Set(models.map((m) => m.backend)).size > 1 ? [{ label: "Backend", value: (m) => m.backend }] : []),
        { label: "Parameters", value: (m) => m.parameter_size || "—" },
        { label: "Quantization", value: (m) => m.quantization || "—" },
        { label: "Size", value: (m) => m.size_human || "—", num: true },
        {
          label: "Tool calling",
          value: (m) =>
            m.can_chat === false
              ? badge("embedding only", "")
              : m.tool_support === true
                ? badge("native", "ok")
                : m.tool_support === false
                  ? badge("JSON mode", "warn")
                  : badge("unknown"),
        },
        { label: "Context window", value: (m) => (m.context_window ? `${nf.format(m.context_window)} tokens` : "—"), num: true },
      ],
      models,
      { emptyText: "The backend reports no models. Pull one, e.g. `ollama pull qwen2.5-coder:7b`." },
    ),
  );
}

// ---------------------------------------------------------------- settings

const isDarkNow = () =>
  document.documentElement.dataset.theme === "dark" ||
  (!document.documentElement.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);

/** One settings row: label + description on the left, a control on the right. */
function sRow(label, description, ...controls) {
  return h(
    "div",
    { class: "s-row" },
    h("div", { class: "s-label" }, h("strong", { text: label }), description ? h("span", { text: description }) : null),
    h("div", { class: "s-control" }, controls),
  );
}

function sSwitch(checked, onChange, label) {
  const input = h("input", { type: "checkbox", role: "switch", checked, "aria-label": label });
  input.addEventListener("change", () => onChange(input.checked));
  return h("label", { class: "switch" }, input, h("span", {}));
}

function sSection(id, iconName, title, description, ...cards) {
  return h("section", { class: "s-section", id: `settings-${id}`, dataset: { section: id, icon: iconName } },
    h("header", {}, h("h2", { text: title }), description ? h("p", { text: description }) : null),
    cards,
  );
}

const sCard = (title, iconName, ...rows) =>
  h("div", { class: "s-card" }, title ? h("div", { class: "s-card-title" }, iconName ? icon(iconName) : null, title) : null, rows);

const yesNo = (v, yes = "On", no = "Off") => badge(v ? yes : no, v ? "ok" : "");

async function viewSettings(root) {
  root.append(pageHead("Settings", "How this dashboard looks and chats, and how the gateway is configured."));

  // ---- Appearance
  const modes = [["system", "monitor", "System"], ["light", "sun", "Light"], ["dark", "moon", "Dark"]];
  const segmented = h("div", { class: "segmented", role: "group", "aria-label": "Colour mode" });
  const renderModes = () => {
    segmented.replaceChildren(
      ...modes.map(([id, ic, label]) =>
        h("button", {
          type: "button",
          "aria-pressed": String(prefs.theme === id),
          onclick: () => { prefs.theme = id; applyTheme(id); renderModes(); renderPalettes(); },
        }, icon(ic), label),
      ),
    );
  };
  const paletteGrid = h("div", { class: "palettes", role: "group", "aria-label": "Theme" });
  const renderPalettes = () => {
    const dark = isDarkNow();
    paletteGrid.replaceChildren(
      ...PALETTES.map((pal) => {
        const [accent, second] = dark ? pal.dark : pal.light;
        const base = dark ? "#121516" : "#f6f6f4";
        const side = dark ? "#0b0d0e" : "#ececea";
        const ink = dark ? "#eceeee" : "#15181b";
        const preview = h("div", { class: "palette-preview", "aria-hidden": "true" });
        const sideEl = h("div", { class: "pv-side" }, h("i"), h("i"), h("i"));
        const mainEl = h("div", { class: "pv-main" }, h("div", { class: "pv-bubble" }), h("div", { class: "pv-line" }), h("div", { class: "pv-line" }), h("div", { class: "pv-btn" }));
        sideEl.style.background = `color-mix(in srgb, ${accent} 5%, ${side})`;
        mainEl.style.background = `color-mix(in srgb, ${accent} 4%, ${base})`;
        sideEl.querySelectorAll("i").forEach((el, i) => { el.style.background = i === 0 ? accent : ink; el.style.width = `${[80, 60, 70][i]}%`; });
        mainEl.querySelector(".pv-bubble").style.background = `color-mix(in srgb, ${accent} 22%, ${base})`;
        const lines = mainEl.querySelectorAll(".pv-line");
        lines.forEach((el) => { el.style.background = ink; });
        lines[lines.length - 1].style.background = second;
        mainEl.querySelector(".pv-btn").style.background = accent;
        preview.append(sideEl, mainEl);
        return h("button", {
          class: "palette",
          type: "button",
          "aria-pressed": String(prefs.palette === pal.id),
          "aria-label": `${pal.name} theme`,
          onclick: () => { prefs.palette = pal.id; applyPalette(pal.id); renderPalettes(); },
        }, preview, h("div", { class: "palette-name" }, pal.name, h("span", { class: "tick" }, icon("check"))));
      }),
    );
  };
  renderModes();
  renderPalettes();
  const appearance = sSection(
    "appearance", "sun", "Appearance", "Pick a colour mode and a theme. Saved in this browser.",
    sCard(null, null,
      sRow("Colour mode", "Follow your system, or force light or dark.", segmented),
      h("div", { class: "s-row stack-row" }, h("div", { class: "s-label" }, h("strong", { text: "Theme" }), h("span", { text: "Accent colour; backgrounds and borders take a subtle matching tint." })), paletteGrid),
    ),
  );

  // ---- Chat
  const keyInput = h("input", { type: "password", value: prefs.apiKey, placeholder: "lg_…", autocomplete: "off", "aria-label": "API key for chat" });
  const eye = h("button", { class: "icon-btn", type: "button", title: "Show key", "aria-label": "Show key" }, icon("eye"));
  eye.addEventListener("click", () => {
    const hidden = keyInput.type === "password";
    keyInput.type = hidden ? "text" : "password";
    eye.replaceChildren(icon(hidden ? "eyeOff" : "eye"));
    eye.title = hidden ? "Hide key" : "Show key";
    eye.setAttribute("aria-label", eye.title);
  });
  const keyStatus = h("div", { class: "key-status", role: "status" });
  const rememberSwitch = sSwitch(prefs.rememberApiKey, (on) => { if (prefs.apiKey) prefs.setApiKey(prefs.apiKey, on); toast(on ? "Key remembered on this device." : "Key kept for this tab only."); }, "Remember API key");
  const saveKey = () => {
    prefs.setApiKey(keyInput.value.trim(), rememberSwitch.querySelector("input").checked);
    toast(keyInput.value.trim() ? "API key saved." : "API key removed.");
  };
  const testKey = async () => {
    saveKey();
    keyStatus.replaceChildren(h("span", { class: "spinner" }), "Checking…");
    try {
      const resp = await apiWithKey("/v1/usage");
      const u = await resp.json();
      keyStatus.replaceChildren(badge("valid", "ok"), `${nf.format(u.request_count ?? u.requests ?? 0)} requests · ${nf.format(u.total_tokens ?? 0)} tokens so far`);
    } catch (err) {
      keyStatus.replaceChildren(badge("not valid", "bad"), err.message);
    }
  };
  const modelSelect = h("select", { "aria-label": "Default chat model" }, h("option", { value: "", text: "Server default" }));
  modelSelect.addEventListener("change", () => { prefs.model = modelSelect.value; toast("Default model saved."); });
  api("/admin/models").then((models) => {
    models.filter((m) => m.can_chat !== false).forEach((m) => modelSelect.append(h("option", { value: m.name, text: m.name })));
    modelSelect.value = prefs.model;
  }).catch(() => {});
  const chat = sSection(
    "chat", "pen", "Chat", "Defaults for the chat page in this browser.",
    sCard("API key", "key",
      h("div", { class: "s-row stack-row" },
        h("div", { class: "s-label" }, h("strong", { text: "Gateway API key" }), h("span", {}, "Chat calls the gateway like any client. Create keys under ", h("a", { href: "#/keys", text: "API keys" }), ".")),
        h("div", { class: "input-group" }, keyInput, eye, h("button", { class: "btn", type: "button", text: "Save", onclick: saveKey }), h("button", { class: "btn primary", type: "button", text: "Test key", onclick: testKey })),
        keyStatus,
      ),
      sRow("Remember on this device", "Off keeps the key only until this tab closes.", rememberSwitch),
    ),
    sCard("Defaults", "list",
      sRow("Default model", "Used for new messages unless you pick another in the composer.", modelSelect),
      sRow("Web search", "Search the web and cite sources by default.", sSwitch(prefs.webSearch, (on) => { prefs.webSearch = on; }, "Web search by default")),
      sRow("Read replies aloud", "Speak each answer with the server's local Piper voice.", sSwitch(prefs.speak, (on) => { prefs.speak = on; }, "Read replies aloud")),
    ),
  );

  // ---- Server (read-only, from /admin/config)
  const serverHost = h("div", {}, loading());
  const server = sSection("server", "monitor", "Server", "How the gateway is configured. Change these with LOCALGATE_* environment variables and restart.", serverHost);

  // ---- Database
  const dbInput = h("input", { type: "text", placeholder: "postgresql+asyncpg://user:pass@host/dbname", autocomplete: "off", spellcheck: "false", "aria-label": "Database URL" });
  const dbResult = h("div", { class: "result-box", role: "status" });
  const dbButton = h("button", { class: "btn primary", type: "submit", text: "Test & save" });
  const dbForm = h("form", {
    class: "s-row stack-row",
    onsubmit: async (e) => {
      e.preventDefault();
      const url = dbInput.value.trim();
      if (!url) return;
      dbButton.disabled = true;
      dbResult.replaceChildren(h("div", { class: "callout neutral", text: "Testing the connection…" }));
      try {
        const result = await api("/admin/config/database-url", { method: "PUT", body: { database_url: url } });
        dbResult.replaceChildren(h("div", { class: "callout ok", text: result.message }));
        dbInput.value = "";
        loadConfig();
      } catch (err) {
        dbResult.replaceChildren(h("div", { class: "callout bad", text: err.message }));
      } finally {
        dbButton.disabled = false;
      }
    },
  },
  h("div", { class: "s-label" }, h("strong", { text: "Switch database" }), h("span", { text: "SQLite, local Postgres or Neon. The connection is tested before anything is saved; restart to apply." })),
  h("div", { class: "input-group" }, dbInput, dbButton),
  dbResult);
  const dbCurrent = h("div", { class: "s-value", text: "…" });
  const database = sSection("database", "folder", "Database", "Where conversations, memory, keys and usage are stored.",
    sCard(null, null, sRow("Current database", "Passwords are hidden.", dbCurrent), dbForm));

  // ---- Web search
  const searchInput = h("input", { type: "search", value: "python asyncio gather", required: true, "aria-label": "Test query" });
  const searchResult = h("div", { class: "result-box", role: "status" });
  const searchForm = h("form", {
    class: "s-row stack-row",
    onsubmit: async (e) => {
      e.preventDefault();
      searchResult.replaceChildren(loading("Searching…"));
      try {
        const r = await api("/admin/search/test", { method: "POST", body: { query: searchInput.value.trim() } });
        searchResult.replaceChildren(
          h("div", { class: "actions" },
            badge(r.providers.join(" → ") || "no providers", "accent"),
            r.leaves_machine ? badge("query leaves this machine", "warn") : badge("self-hosted", "ok"),
            r.elapsed_ms !== undefined ? badge(`${nf.format(r.elapsed_ms)} ms`) : null),
        );
        if (r.error) searchResult.append(h("div", { class: "callout bad", text: r.error }));
        if (r.results.length) {
          searchResult.append(h("ol", {}, r.results.map((x) => {
            let href = "";
            try { const u = new URL(x.url); if (u.protocol.startsWith("http")) href = u.href; } catch { /* no link */ }
            return h("li", {}, href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: x.title }) : x.title, h("div", { class: "muted small", text: x.snippet }));
          })));
        } else if (!r.error) searchResult.append(empty("No results."));
      } catch (err) {
        searchResult.replaceChildren(errorState(err));
      }
    },
  },
  h("div", { class: "s-label" }, h("strong", { text: "Test the provider chain" }), h("span", { text: "Runs LOCALGATE_SEARCH_PROVIDER once (e.g. searxng,duckduckgo) and shows what came back." })),
  h("div", { class: "input-group" }, searchInput, h("button", { class: "btn", type: "submit", text: "Run search" })),
  searchResult);
  const web = sSection("web", "globe", "Web search", "Used by chat's Web search toggle and the agent's web_search tool.", sCard(null, null, searchForm));

  // ---- Data & account
  const data = sSection("data", "book", "Data & account", null,
    h("div", { class: "s-card danger-zone" },
      sRow("Export everything", "Keys (without secrets), usage and conversations as JSON.",
        h("button", { class: "btn", type: "button", text: "Download JSON", onclick: async () => {
          try {
            const exported = await api("/admin/export");
            downloadBlob(`localgate-export-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(exported, null, 2), "application/json");
          } catch (err) { toast(err.message, "bad"); }
        } })),
      sRow("Clear Recents", "Forget the conversation list in this browser. Stored history stays on the server.",
        h("button", { class: "btn", type: "button", text: "Clear", onclick: async () => {
          if (!(await confirmAction("Clear Recents?", "The list in this browser is emptied; conversations stay stored on the server.", "Clear"))) return;
          prefs.sessions = [];
          renderHistory();
          toast("Recents cleared.");
        } })),
      sRow("Sign out", "End this admin session in this browser.", h("button", { class: "btn danger", type: "button", text: "Sign out", onclick: signOut })),
    ),
  );

  const sections = [appearance, chat, server, database, web, data];
  const nav = h("nav", { class: "settings-nav", "aria-label": "Settings sections" },
    sections.map((sec) => h("a", {
      href: `#settings-${sec.dataset.section}`,
      dataset: { target: sec.id },
      onclick: (e) => { e.preventDefault(); sec.scrollIntoView({ behavior: "smooth", block: "start" }); },
    }, icon(sec.dataset.icon), sec.querySelector("h2").textContent)));
  root.append(h("div", { class: "settings" }, nav, h("div", { class: "settings-body" }, sections)));

  // Highlight the section in view.
  const links = $$("a", nav);
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((a) => a.classList.toggle("active", a.dataset.target === entry.target.id));
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  sections.forEach((sec) => observer.observe(sec));
  links[0].classList.add("active");

  async function loadConfig() {
    try {
      const c = await api("/admin/config");
      dbCurrent.textContent = c.database?.url || "—";
      const row = (label, desc, value) => sRow(label, desc, value instanceof Node ? value : h("div", { class: "s-value", text: String(value ?? "—") }));
      serverHost.replaceChildren(...[
        c.security?.using_default_admin_key
          ? h("div", { class: "callout warn", text: "This server uses the default admin key. Set LOCALGATE_ADMIN_KEY before exposing it — and before creating keys you want to be able to view again." })
          : null,
        sCard("Inference", "plug",
          row("Backend", c.backend?.url, h("div", { class: "actions" }, badge(c.backend?.type || "—", "accent"))),
          row("Default model", "Used when a request names no model.", c.backend?.default_model),
          row("Request timeout", null, `${c.backend?.timeout ?? "—"} s`),
          row("Model aliases", null, Object.entries(c.backend?.model_aliases || {}).map(([k, v]) => `${k} → ${v}`).join(", ") || "none"),
        ),
        sCard("Memory", "book",
          row("Memory", "Store conversations, embed them and recall relevant turns.", yesNo(c.memory?.enabled)),
          row("Embedding model", null, c.memory?.embedding_model),
          row("Recall", "Chunks injected per request · minimum similarity", `${c.memory?.max_retrieved_chunks} chunks · ≥ ${c.memory?.min_score}`),
          row("Summarise after", null, `${c.memory?.summarize_after_messages} messages`),
          row("Prompt cache", null, yesNo(c.cache?.enabled)),
        ),
        c.agent ? sCard("Coding agent", "terminal",
          row("Context window", "Requested from Ollama (num_ctx), capped at the model's maximum.", `${nf.format(c.agent.context_window)} tokens`),
          row("Temperature", null, c.agent.temperature),
          row("Command sandbox", "run_command only runs inside it.", h("div", { class: "actions" }, badge(c.agent.sandbox, c.agent.sandbox === "off" ? "" : "ok"))),
          row("Tool-call log", "Shown in Agent runs.", yesNo(c.agent.tool_logging)),
        ) : null,
        c.search ? sCard("Web", "globe",
          row("Search providers", "Tried in order.", c.search.providers),
          row("Chat web search", null, yesNo(c.search.chat_web_search, "Allowed", "Disabled")),
          row("Sources per answer", null, c.search.max_sources),
          row("Fetch private hosts", "Off blocks loopback and private-network addresses.", yesNo(c.search.fetch_private_hosts, "Allowed", "Blocked")),
        ) : null,
        c.voice ? sCard("Voice", "speaker",
          row("Voice", "Piper speech and Whisper transcription (localgate[voice]).", yesNo(c.voice.enabled)),
          row("Piper voice", null, c.voice.tts_voice),
          row("Whisper model", null, c.voice.stt_model),
        ) : null,
        sCard("Security", "key",
          row("Admin key", null, c.security?.using_default_admin_key ? badge("default — change it", "warn") : badge("custom", "ok")),
          row("Viewable API keys", "Admins can reveal keys created after this was enabled.", yesNo(c.security?.api_key_reveal, "On", "Off")),
          row("Environment", null, c.environment),
        ),
      ].filter(Boolean));
    } catch (err) {
      serverHost.replaceChildren(errorState(err));
    }
  }
  await loadConfig();
}

// ================================================================ router

// ---------------------------------------------------------------- share

/** Give another device or person access: a key with limits, plus everything
 *  they need to use it — base URL, a connect link, a QR code and snippets. */
async function viewShare(root) {
  const addressBody = h("div", { class: "card-body" }, loading());
  const internetBody = h("div", { class: "card-body" });
  const result = h("div", {});
  let urls = [];
  let internet = { enabled: false, starting: false, url: null, error: null };
  let polling = null;

  async function loadAddresses() {
    try {
      const share = await api("/admin/share");
      urls = share.urls;
      internet = share.internet || internet;
    } catch (err) {
      addressBody.replaceChildren(errorState(err));
      return;
    }
    renderInternet();
    if (internet.starting && !polling) polling = setTimeout(() => { polling = null; loadAddresses(); }, 1500);
    addressBody.textContent = "";
    if (!urls.length) {
      addressBody.append(
        h("p", { class: "muted", text: "No local network address found — is this machine on a network? Devices elsewhere can still connect: turn on Internet access above." }),
      );
    } else {
      addressBody.append(
        h(
          "ul",
          { class: "share-urls" },
          urls.map((u) =>
            h(
              "li",
              {},
              h("span", { class: "share-label", text: u.label }),
              h("code", { text: `${u.url}/v1` }),
              h("button", { class: "btn sm", type: "button", text: "Copy", onclick: (e) => copyText(`${u.url}/v1`, e.currentTarget) }),
            ),
          ),
        ),
        h("p", { class: "small muted", text: "Devices can't connect? Check the gateway wasn't started with --local, and that the device is on this network. Devices elsewhere: turn on Internet access above." }),
      );
    }
    urlSelect.replaceChildren(
      ...(urls.length ? urls : [{ label: "This browser's address", url: location.origin }]).map((u) => h("option", { value: u.url, text: `${u.label} — ${u.url}` })),
    );
  }

  function renderInternet() {
    const toggle = h("input", { type: "checkbox", "aria-label": "Internet access" });
    toggle.checked = internet.enabled || internet.starting;
    toggle.disabled = internet.starting;
    toggle.addEventListener("change", async () => {
      toggle.disabled = true;
      try {
        internet = await api("/admin/share/internet", { method: "POST", body: { enabled: toggle.checked } });
      } catch (err) { toast(err.message, "bad"); }
      await loadAddresses();
    });
    let status;
    if (internet.starting) status = h("p", { class: "muted", text: "Opening an internet address… the first time this downloads Cloudflare's tunnel program (about 40 MB)." });
    else if (internet.enabled && internet.url)
      status = h("div", {}, h("p", { text: "On — devices on any network can connect at:" }), h("div", { class: "actions" }, h("code", { text: `${internet.url}/v1` }), h("button", { class: "btn sm", type: "button", text: "Copy", onclick: (e) => copyText(`${internet.url}/v1`, e.currentTarget) })), h("p", { class: "small muted", text: "HTTPS through a free Cloudflare tunnel. The address changes each time the gateway restarts." }));
    else status = h("p", { class: "muted", text: internet.error || "Off — only devices on your network (and Tailscale) can connect." });
    internetBody.replaceChildren(
      h("div", { class: "s-row" }, h("div", { class: "s-label" }, h("strong", { text: "Internet access" }), h("span", { text: "Let devices on other networks connect, with no setup." })), h("label", { class: "switch" }, toggle, h("span", {}))),
      status,
    );
  }

  /** The address a key should be given, by where it's allowed to work. */
  function baseFor(key, chosen) {
    const internetUrl = urls.find((u) => u.label === "Internet")?.url;
    const localUrl = urls.find((u) => u.label !== "Internet" && u.label !== "Public")?.url;
    if (key.network_access === "internet") return internetUrl || null;
    if (key.network_access === "local") return localUrl || location.origin;
    return chosen;
  }

  const nameInput = h("input", { type: "text", placeholder: "Ana's laptop", required: true, maxlength: "100" });
  const urlSelect = h("select", {});
  const limits = limitsFields({ store_conversations: true });
  const form = h(
    "form",
    {
      class: "card-body",
      onsubmit: async (e) => {
        e.preventDefault();
        const name = nameInput.value.trim();
        if (!name) return;
        try {
          const values = limits.values();
          if (values.network_access === "internet" && !internet.enabled) {
            toast("Turn on Internet access first — this key only works through the internet address.", "bad");
            return;
          }
          const key = await api("/admin/keys", { method: "POST", body: { name, ...values } });
          nameInput.value = "";
          await showAccess(key, baseFor(key, urlSelect.value || location.origin));
          toast("Access created.", "ok");
        } catch (err) { toast(err.message, "bad"); }
      },
    },
    h("div", { class: "field-row" }, field("Device or person", nameInput), field("They connect to", urlSelect)),
    h("details", { class: "more-limits", open: true }, h("summary", { text: "Limits for this key" }), limits.element),
    h("div", { class: "actions" }, h("button", { class: "btn primary", type: "submit", text: "Create access" })),
  );

  async function qrFor(text) {
    const resp = await fetch("/admin/share/qr", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Localgate-Csrf": "1" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) throw new Error("Couldn't make the QR code.");
    // A data: URL — the dashboard's CSP allows images only from itself or data:.
    const svgText = await resp.text();
    return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgText)))}`;
  }

  async function showAccess(key, base) {
    const connect = `${base}/connect#url=${encodeURIComponent(base)}&key=${encodeURIComponent(key.api_key)}`;
    const qr = h("img", { class: "qr", alt: "QR code: opens the connect page with this key", width: "220", height: "220" });
    qrFor(connect).then((src) => { qr.src = src; }).catch(() => qr.replaceWith(h("p", { class: "muted small", text: "QR code unavailable — share the link instead." })));
    result.replaceChildren(
      card(
        `Access for “${key.name}”`,
        h(
          "div",
          { class: "card-body share-result" },
          h(
            "div",
            { class: "share-grid" },
            h(
              "div",
              {},
              h("p", { class: "muted small", text: "Scan with their phone, or send them the link. It opens a page with everything they need." }),
              qr,
            ),
            h(
              "div",
              { class: "share-creds" },
              h("dl", { class: "kv" }, [
                [h("dt", { text: "Base URL" }), h("dd", {}, h("code", { text: `${base}/v1` }))],
                [h("dt", { text: "API key" }), h("dd", {}, h("code", { class: "secret-inline", text: key.api_key }))],
                [h("dt", { text: "Limits" }), h("dd", { text: limitsSummary(key) })],
              ]),
              h(
                "div",
                { class: "actions" },
                h("button", { class: "btn sm", type: "button", text: "Copy connect link", onclick: (e) => copyText(connect, e.currentTarget) }),
                h("button", { class: "btn sm", type: "button", text: "Copy key", onclick: (e) => copyText(key.api_key, e.currentTarget) }),
              ),
              h("p", { class: "small muted", text: "The key is shown in full here once. You can show it again later under API keys." }),
            ),
          ),
          codeBlock(`localgate login --url ${base} --api-key ${key.api_key}\nlocalgate code --remote`, "shell"),
          codeBlock(
            `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}/v1", api_key="${key.api_key}")`,
            "python",
          ),
        ),
      ),
    );
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  root.append(
    pageHead("Share", "Let another device or person use this machine's models — each with their own key and limits."),
    card("Who can connect", internetBody),
    card("Where devices reach this gateway", addressBody),
    card("Add a device", form),
    result,
  );
  await loadAddresses();
}

// ---------------------------------------------------------------- activity

const ACTION_LABELS = {
  "key.created": "created key",
  "key.updated": "changed limits of key",
  "key.revealed": "viewed key",
  "key.revoked": "revoked key",
  sign_in: "signed in",
  "sign_in.failed": "failed sign-in",
  "user.created": "created account",
  "user.deleted": "deleted account",
  "user.password_changed": "changed password of",
  "config.database_url": "changed database to",
};

/** Who did what to the gateway: keys, sign-ins, accounts, settings. */
async function viewActivity(root) {
  const body = h("div", {}, loading());
  root.append(pageHead("Activity", "Admin actions on this gateway — who did what, and from where."), card("Recent actions", body));
  try {
    const events = await api("/admin/audit?limit=200");
    body.replaceChildren(
      table(
        [
          { label: "When", value: (e) => when(e.at) },
          { label: "Who", value: (e) => h("span", { class: e.actor === "admin-key" || e.actor === "cli" ? "muted" : "", text: e.actor }) },
          {
            label: "What",
            value: (e) =>
              h(
                "span",
                {},
                e.action === "sign_in.failed" ? badge("failed sign-in", "bad") : ACTION_LABELS[e.action] || e.action,
                e.target && e.action !== "sign_in.failed" ? h("span", { class: "mono", text: ` ${e.details?.name || e.target}` }) : null,
              ),
          },
          { label: "From", value: (e) => h("span", { class: "mono cell-sub", text: e.ip || "this machine" }) },
        ],
        events,
        { emptyText: "No admin actions yet." },
      ),
    );
  } catch (err) {
    body.replaceChildren(errorState(err));
  }
}

const ROUTES = {
  overview: viewOverview,
  chat: viewChat,
  keys: viewKeys,
  share: viewShare,
  usage: viewUsage,
  sessions: viewSessions,
  agent: viewAgent,
  models: viewModels,
  settings: viewSettings,
  activity: viewActivity,
};

let routeToken = 0;

async function route() {
  if ($("#app").hidden) return;
  const [name, arg] = (location.hash.replace(/^#\/?/, "") || "chat").split("/");
  const view = ROUTES[name] ? name : "chat";
  chatAbort?.abort();
  $$(".console a").forEach((a) => {
    if (a.dataset.route === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (view !== "chat") { currentChatId = ""; renderHistory(); }
  closeMenu();
  const content = $("#content");
  const token = ++routeToken;
  const page = h("div", { class: "page" });
  content.textContent = "";
  content.append(page);
  $("#tooltip").hidden = true;
  document.title = view === "chat" ? "Localgate" : `${$(`.console a[data-route="${view}"]`)?.textContent || view} · Localgate`;
  try {
    await ROUTES[view](page, arg ? decodeURIComponent(arg) : "");
  } catch (err) {
    if (token !== routeToken || err.status === 401) return;
    page.querySelector(".loading")?.replaceWith(errorState(err));
    if (!page.querySelector(".error-state")) page.append(errorState(err));
  }
}

function closeMenu() {
  $("#sidebar").classList.remove("open");
  $("#scrim").hidden = true;
  $("#menu-toggle").setAttribute("aria-expanded", "false");
}

function init() {
  applyTheme(prefs.theme);
  $("#theme-toggle").addEventListener("click", () => {
    const next = THEMES[(THEMES.indexOf(prefs.theme) + 1) % THEMES.length];
    prefs.theme = next;
    applyTheme(next);
  });
  $("#menu-toggle").addEventListener("click", () => {
    const open = $("#sidebar").classList.toggle("open");
    $("#scrim").hidden = !open;
    $("#menu-toggle").setAttribute("aria-expanded", String(open));
  });
  $("#scrim").addEventListener("click", closeMenu);
  $("#history").addEventListener("click", (e) => { if (e.target.closest("a")) closeMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
  $("#login-form").addEventListener("submit", signIn);
  $("#login-mode").addEventListener("click", () => setLoginMode(!loginWithAdminKey));
  $("#sign-out").addEventListener("click", signOut);
  window.addEventListener("hashchange", route);
  renderHistory();
  pollHealth();
  setInterval(pollHealth, 15000);
  checkSession();
}

init();
