# Share a live demo (external link)

The MVP web UI and API are **one server** (`python run_demo.py`). There is **no fixed public URL** in the repo—you expose your machine **while the process runs** using a tunnel.

## 1. Start the stack

```bash
python run_demo.py
```

Default: **http://127.0.0.1:7860/** (same port for UI + API).

## 2. Get a link others can open (tunnel)

### Option A — Cloudflare Tunnel (recommended; free, no account for quick tunnels)

1. Install **cloudflared** (one-time):  
   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
2. Run:

```bash
python run_demo.py --share
```

3. Watch the terminal: when the tunnel is up, a line like  
   `https://something-random.trycloudflare.com`  
   appears. The script also prints a **“SHARE THIS URL”** block with:
   - **Public base** — the tunnel root  
   - **Demo deep link** — same base + `/dashboard?demo=1&prepare=1`  
     (enables Demo Mode and **auto-creates** the three scripted demo runs if the DB has no runs yet)

**Important:** The link works **only while** `python run_demo.py --share` is running and your network allows outbound connections.

### Option B — ngrok

1. Install **ngrok**: https://ngrok.com/download  
2. Run `ngrok config add-authtoken <token>` once (free tier).  
3. Run:

```bash
python run_demo.py --share
```

If ngrok is on `PATH`, it starts automatically; check the terminal for an `https://…ngrok-free.app` (or similar) URL.

## 3. Windows (PowerShell)

Same commands from the repo root:

```powershell
cd C:\path\to\neraium-core-1
python run_demo.py --share
```

Install cloudflared for Windows and ensure `cloudflared.exe` is on `PATH`, **or** install ngrok and add it to `PATH`.

## 4. What to send to guests

Send the **demo deep link** from the printed banner, for example:

```
https://<your-tunnel-host>/dashboard?demo=1&prepare=1
```

Guests get:

- Demo Mode on  
- Automatic preparation of the three demo runs (stable / watch / escalation) when there are no runs yet  
- Redirect to the focus run detail  

## 5. Long-lived / production links

For a URL that does **not** depend on your laptop session, deploy to a host (VM, Docker, cloud) and point DNS or a reverse proxy at it. See **[CUSTOMER_DEPLOYMENT.md](CUSTOMER_DEPLOYMENT.md)**.
