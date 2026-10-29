# Sharing your GPU

One machine with a good GPU runs the models; laptops, phones, apps and teammates use
them through localgate, each with its own API key and limits. This guide goes from
"it runs on my machine" to "my devices use it every day".

```
 laptop / phone / app ──(API key lg_…)──▶  GPU machine: localgate serve  ──▶  Ollama ──▶ GPU
                                             keys · limits · fair queue · usage · dashboard
```

## 1. Start sharing

On the GPU machine:

```bash
ollama serve                      # or llama.cpp / vLLM — see LOCALGATE_BACKEND_*
localgate serve --share
```

`--share` listens on the network and prints the addresses other devices can use:

```
Sharing this machine's models
  Local network  http://192.168.1.20:8000/v1
  Give each device its own key: http://127.0.0.1:8000/dashboard/#/share
```

Without `--share` the gateway stays on this machine only (`127.0.0.1`).

## 2. Give a device access

Open **Dashboard → Share** on the GPU machine, name the device, set its limits and press
**Create access**. You get the base URL, the key, a one-line connect command, code
snippets — and a QR code. Scanned on a phone (or sent as a link) it opens `/connect`, a
page with everything the device needs. The key travels in the link's `#fragment`, which
browsers never send to a server, and the page removes it from the address bar once shown.

From a terminal instead:

```bash
localgate keys create --name ana-laptop --models qwen2.5-coder:7b \
  --max-tokens 2048 --quota 2000000 --quota-period month --expires 90d
```

On the other device, any OpenAI-compatible client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.20:8000/v1", api_key="lg_...")
```

or the localgate coding agent. Install it and just run it:

```bash
pip install localgate
localgate code
```

With no model server on that device, it asks where models should run — pick **a shared
gateway**, type the address (`192.168.1.20` is enough; gateways announced on the network
are listed by number) and the API key. The connection is checked, saved, and used from
then on. Inside a session, `/connect` switches to a gateway and `/disconnect` back to the
device's own models. From scripts: `localgate login --url 192.168.1.20 --api-key lg_...`,
`localgate whoami` (your limits and usage), `localgate logout`.

### Same network, or not?

`serve --share` works for devices on the **same network** (home or office Wi-Fi, LAN).
For a device somewhere else, use Tailscale or a tunnel — see section 5.

If a device gets **"refused to connect"** / **"Nothing answered"**: the gateway is
running without `--share` (it then accepts connections from its own machine only),
isn't running, or uses another port. Check on the GPU machine that `serve --share`
printed a *Local network* address, and use exactly that address and port.

## 3. Limits per key

| Limit | What happens when it's hit |
|---|---|
| Requests / minute | `429 rate_limit_exceeded` + `Retry-After` |
| Allowed models | `403 model_not_allowed`; `/v1/models` lists only what the key may use |
| Max reply tokens | `max_tokens` is capped (and filled in when missing) |
| Token quota per day / month (UTC) | `429 quota_exceeded` + `Retry-After` until the period resets |
| Expiry | `401 key_expired` |
| Requests at once | extra requests from that key wait in the queue |
| Store conversations | off: no history or memory is kept for the key — only usage |

Change any of them later without reissuing the key: Dashboard → API keys → **edit**, or
`localgate keys update <id> --quota 0` (0 removes a numeric limit, `--models any` removes
the model list, `--expires never` removes expiry).

## 4. The GPU queue

A GPU runs a few generations at once; more just makes everyone slower until requests time
out. localgate runs `LOCALGATE_MAX_CONCURRENT_REQUESTS` (default 2) per backend and queues
the rest **fairly**: slots go round-robin across keys, so one device sending 50 requests
can't starve another sending one. A request that waits longer than
`LOCALGATE_QUEUE_TIMEOUT` (60 s) gets `503` with a `Retry-After` estimate instead of
hanging.

Match the limit to what Ollama runs in parallel:

```bash
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 ollama serve
LOCALGATE_MAX_CONCURRENT_REQUESTS=2 localgate serve --share
```

Every request also carries a context window (`LOCALGATE_DEFAULT_NUM_CTX`, 8192, capped at
the model's maximum) so Ollama doesn't fall back to its small default and silently drop
the start of long prompts. A client that sends its own `options.num_ctx` wins.

A client that disconnects stops its generation: the GPU doesn't keep working for nobody,
and the tokens produced so far are still counted.

## 5. Beyond your own network

`--share` speaks plain HTTP, which is fine on a home or office network. For anything else:

- **Tailscale (recommended).** Install it on the GPU machine and the devices; use the
  machine's Tailscale address (`100.x.y.z`, shown by `serve --share`). Encrypted, private,
  no ports opened, and the dashboard stays reachable because Tailscale's range is trusted
  by default.
- **Cloudflare Tunnel.** `cloudflared tunnel --url http://localhost:8000` gives an HTTPS
  address without opening ports. Set `LOCALGATE_PUBLIC_URL` to it so the Share page uses
  it.
- **Your own domain.** `localgate deploy --domain gw.example.com` generates Docker
  Compose + Caddy (automatic HTTPS) or a systemd unit. Only `/v1` and `/health` are
  exposed publicly.

## 6. What the internet can't reach

The dashboard, `/admin`, the API docs and `/metrics` answer only clients on
`LOCALGATE_ADMIN_ALLOWED_NETWORKS` — by default this machine, private LANs and Tailscale.
From anywhere else they return `403`, even with the admin key, and `/health` reports only
whether the gateway is up. The `/v1` API works from anywhere the port is reachable, with
a key. Behind a reverse proxy, run `localgate serve --proxy-headers --forwarded-allow-ips
<proxy-address>` so localgate sees real client addresses (the `deploy` output does this).

Admin actions — keys created, changed, revealed or revoked, sign-ins, accounts — are
recorded: Dashboard → **Activity**, or `localgate audit`.

## 7. Keep it running

```bash
localgate service install --share     # start at login, restart if it stops
localgate service status
localgate service logs
localgate service uninstall
```

macOS uses a launchd agent, Linux a systemd user unit (`loginctl enable-linger $USER`
keeps it running while logged out), Windows a scheduled task at logon.
`localgate service install --dry-run` shows exactly what would be installed.

## 8. More than one GPU or machine

List them all and localgate routes each request to a backend that serves its model,
least busy first, failing over when one is unreachable:

```bash
LOCALGATE_BACKENDS='[
  {"name": "big-gpu", "type": "ollama", "url": "http://10.0.0.5:11434", "models": ["llama3:70b"], "max_concurrent": 4},
  {"name": "desk",    "type": "ollama", "url": "http://localhost:11434", "max_concurrent": 2}
]' localgate serve --share
```

A backend without `models` is asked for its list (refreshed every minute). Each has its
own queue. `/v1/models` lists everything; `/health` and the dashboard show each backend.

## 9. Several worker processes

One `localgate serve` process is the right choice on a GPU box. If you run several
(`--workers N`), set `LOCALGATE_REDIS_URL` (and `pip install 'localgate[redis]'`) so rate
limits and per-key concurrency hold across them. The GPU queue itself is per process.

## 10. Privacy

For keys that store conversations, the gateway keeps each exchange (for memory) and the
owner can read it under Sessions. Turn **Store conversations** off for a key that
shouldn't leave a trace; usage is still counted. Tell the people you share with which it is.
