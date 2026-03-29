# Share a live reference replay (external link)

The MVP web UI and API are **one server** (`python run_demo.py`). There is **no fixed public URL** in the repo—you expose your machine **while the process runs** using a tunnel.

## 1. Start the stack

```bash
python run_demo.py
```

Default: **http://127.0.0.1:7860/** (same port for UI + API).

## 2. Get a link others can open (tunnel)

### Option A — ngrok (recommended)

1. Install **ngrok**: https://ngrok.com/download  
2. Run `ngrok config add-authtoken <token>` once (free tier).  
3. Run:

```bash
python run_demo.py --share
```

If ngrok is on `PATH`, it starts automatically; check the terminal for an `https://…ngrok-free.app` (or similar) URL.

### Option B — another HTTPS tunnel provider

If you use another tunnel provider, expose local port `7860` and share the resulting HTTPS URL.

Then run:

```bash
python run_demo.py
```

The app is available at `http://127.0.0.1:7860` locally and through your configured public tunnel URL.
**Important:** The link works **only while** your tunnel and local server are running.

## 3. Windows (PowerShell)

Same commands from the repo root:

```powershell
cd C:\path\to\neraium-core-1
python run_demo.py --share
```

Install ngrok for Windows and ensure `ngrok.exe` is on `PATH` (or run a different tunnel provider explicitly).

## 4. What to send to guests

Send the **reference replay deep link** from the printed banner, for example:

```
https://<your-tunnel-host>/dashboard?replay=1&prepare=1
```

Guests get:

- Replay mode on for historical validation  
- Automatic preparation of the three reference runs (stable / watch / escalation) when there are no runs yet  
- Redirect to the focus run detail  

## 5. Long-lived / production links

For a URL that does **not** depend on your laptop session, deploy to a host (VM, Docker, cloud) and point DNS or a reverse proxy at it. See **[CUSTOMER_DEPLOYMENT.md](CUSTOMER_DEPLOYMENT.md)**.
