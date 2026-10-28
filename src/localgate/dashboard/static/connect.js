// /connect — the page a Share-page QR code opens. The gateway address and API key
// arrive in the URL fragment (#url=…&key=…), which browsers never send to a
// server, and are shown here with ready-to-paste examples. No framework, no build.

const params = new URLSearchParams(window.location.hash.slice(1));
const baseUrl = (params.get("url") || window.location.origin).replace(/\/+$/, "");
const apiKey = params.get("key") || "";

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

/** Fill an example box — its <code>, so the Copy button beside it survives. */
function setExample(id, text) {
  document.querySelector(`#${id} code`).textContent = text;
}

function examples(model = "llama3") {
  const api = `${baseUrl}/v1`;
  setExample(
    "ex-python",
    `from openai import OpenAI

client = OpenAI(base_url="${api}", api_key="${apiKey}")
reply = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)`,
  );
  setExample(
    "ex-curl",
    `curl ${api}/models \\
  -H "Authorization: Bearer ${apiKey}"`,
  );
  setExample(
    "ex-cli",
    `pip install localgate
localgate login --url ${baseUrl} --api-key ${apiKey}
localgate whoami          # your limits and usage
localgate code --remote   # the coding agent, on this gateway's models`,
  );
}

function addCopyButtons() {
  for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", () => copy(document.getElementById(button.dataset.copy).textContent, button));
  }
  for (const pre of document.querySelectorAll("pre")) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.addEventListener("click", () => copy(pre.querySelector("code").textContent, button));
    pre.appendChild(button);
  }
}

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch {
    button.textContent = "Select & copy";
  }
  setTimeout(() => (button.textContent = "Copy"), 1500);
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

async function testConnection() {
  const result = document.getElementById("test-result");
  result.textContent = "Checking…";
  try {
    // Same gateway that served this page, so the CSP allows it.
    const resp = await fetch("/v1/models", { headers: { Authorization: `Bearer ${apiKey}` } });
    const body = await resp.json().catch(() => ({}));
    if (resp.ok) {
      const models = (body.data || []).map((m) => m.id);
      if (!models.length) {
        result.textContent = "Connected, but no models are available to this key right now — ask the owner.";
      } else {
        result.textContent = `Connected — you can use ${models.slice(0, 4).join(", ")}${models.length > 4 ? ` and ${models.length - 4} more` : ""}.`;
        examples(models[0]);
      }
    } else {
      result.textContent = body.error?.message || `The gateway answered ${resp.status}.`;
    }
  } catch {
    result.textContent = "Couldn't reach the gateway from this device.";
  }
}

if (!apiKey) {
  document.getElementById("missing").hidden = false;
} else {
  document.getElementById("details").hidden = false;
  setText("base-url", `${baseUrl}/v1`);
  setText("api-key", apiKey);
  examples();
  addCopyButtons();
  setUpTabs();
  document.getElementById("test").addEventListener("click", testConnection);
  // Don't leave the key in the address bar or browser history once it's shown.
  history.replaceState(null, "", window.location.pathname);
}
