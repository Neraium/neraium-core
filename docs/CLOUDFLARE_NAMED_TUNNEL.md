# Named Cloudflare Tunnel → local FastAPI (port 7860)

Use this instead of a **quick tunnel** (`cloudflared tunnel --url …`) when you want a **stable hostname** on your own zone (e.g. `neraium.example.com`).

**Prerequisites**

- A domain whose **DNS is on Cloudflare** (zone active in the same account you log in with).
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/) installed and on `PATH`.

---

## 1. Log in to Cloudflare (one machine)

```bash
cloudflared tunnel login
```

A browser opens. Pick the **zone** (domain) you will attach the hostname to. This stores a **cert** under your user profile, e.g.:

- Windows: `%USERPROFILE%\.cloudflared\cert.pem`

---

## 2. Create a **named** tunnel

Pick a name (e.g. `neraium-local`):

```bash
cloudflared tunnel create neraium-local
```

**Save:**

- The **Tunnel ID** (UUID) printed in the output.
- The **credentials file** path — typically:

`%USERPROFILE%\.cloudflared\<TUNNEL_UUID>.json`

List tunnels anytime:

```bash
cloudflared tunnel list
```

---

## 3. Route DNS (hostname → tunnel)

Choose the **public hostname** that should hit your laptop (e.g. `neraium.yourdomain.com`). The zone must be on Cloudflare.

**Automatic DNS record (recommended):**

```bash
cloudflared tunnel route dns neraium-local neraium.yourdomain.com
```

This creates the appropriate **CNAME** in Cloudflare DNS for that tunnel.

**Manual alternative:** In the Cloudflare dashboard → **DNS** → add a **CNAME** from `neraium` (or your chosen name) to **`<TUNNEL_ID>.cfargotunnel.com`** (exact target format is shown in Zero Trust → Networks → Tunnels → your tunnel).

---

## 4. Tunnel config (`ingress` → `http://localhost:7860`)

**Option A — user config (default for `cloudflared tunnel run`):**

Edit:

`%USERPROFILE%\.cloudflared\config.yml`

**Option B — project file:**

Copy `cloudflared/config.yml.example` in this repo to a real path (do **not** commit secrets), fill `tunnel`, `credentials-file`, and `hostname`.

Minimal example:

```yaml
tunnel: <YOUR_TUNNEL_UUID>
credentials-file: C:\Users\<you>\.cloudflared\<YOUR_TUNNEL_UUID>.json

ingress:
  - hostname: neraium.yourdomain.com
    service: http://127.0.0.1:7860
  - service: http_status:404
```

Use `127.0.0.1` instead of `localhost` if you want to avoid IPv6 quirks.

---

## 5. Run the tunnel (expose local FastAPI)

**Start your app first** (example):

```bash
python run_demo.py --port 7860
```

**Then** run the tunnel (pick one).

**If config is in the default user path** (`%USERPROFILE%\.cloudflared\config.yml`):

```bash
cloudflared tunnel run neraium-local
```

**If you use a custom config file:**

```bash
cloudflared tunnel --config C:\path\to\config.yml run neraium-local
```

You should see the tunnel connect; **`https://neraium.yourdomain.com`** (your hostname) will proxy to **`http://127.0.0.1:7860`** while both processes run.

---

## 6. Quick checklist

| Step | Command / action |
|------|------------------|
| Login | `cloudflared tunnel login` |
| Create | `cloudflared tunnel create neraium-local` |
| DNS | `cloudflared tunnel route dns neraium-local neraium.yourdomain.com` |
| Config | Set `ingress` → `http://127.0.0.1:7860` |
| Run app | `python run_demo.py --port 7860` |
| Run tunnel | `cloudflared tunnel run neraium-local` |

---

## Troubleshooting

- **`502` / connection refused:** FastAPI not listening on `7860`, or bind address is not `127.0.0.1` / `0.0.0.0`. For `0.0.0.0`, `http://127.0.0.1:7860` in ingress is still correct.
- **Wrong host header:** If the app checks `Host`, you may need Cloudflare **Origin Rules** or app config — most FastAPI setups work without changes.
- **Certificate / login errors:** Re-run `cloudflared tunnel login` and select the correct zone.

Official reference: [Cloudflare Tunnel (cloudflared) guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/).
