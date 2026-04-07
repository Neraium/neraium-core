# AWS App Runner source-repository deployment (FastAPI)

This guide deploys Neraium from GitHub to AWS App Runner using the repository config file (`apprunner.yaml`) at repo root.

## Readiness audit summary

- **Source directory assumption:** deploy from repository root (`/`) because `apprunner.yaml` and `pyproject.toml` are at root.
- **Dependency installation path:** single runtime-phase install via `pip3 install .` (reads canonical dependencies from `pyproject.toml`).
- **FastAPI entrypoint:** `apps.api.main:app`.
- **Static assets:** served from `apps/api/static` through router + `/web` static mount.
- **Health endpoint:** `GET /health` returns JSON `200`.
- **Requirements layout:** root `requirements.txt` is a compatibility shim that delegates to `pyproject.toml` via `.` install.
- **Platform scope:** AWS App Runner (Railway is intentionally not part of the deployment path for this repo).

## App Runner console steps (source code repository)

1. Open **AWS App Runner** → **Create service**.
2. Choose **Source code repository**.
3. Connect/select your GitHub repository.
4. Set **Branch** to your deployment branch (for example `main`).
5. Set **Source directory** to `/` (repository root).
6. In **Deployment settings / Configuration**, choose to use the repository configuration file (`apprunner.yaml`).
7. Create/deploy service.

## App paths after deploy

- API docs (if enabled): `/docs`
- Health: `/health`
- Primary operational UI: `/dashboard` (also `/pilot`, `/operations`)
- Operator compatibility routes: `/operator`, `/operator/workflow` (redirect to `/dashboard`)
- Historical replay routes: `/demo`, `/demo/full` (redirect with replay mode enabled)
- Web static files mount: `/web/*`

## Why GitHub pushes may not update App Runner

If your App Runner service is connected to GitHub but new commits do **not** deploy,
it is usually one of these operational issues (outside app code):

1. **Automatic deployment is disabled** in the App Runner service settings.
2. **Branch mismatch**: App Runner watches one branch, while changes land in another.
3. **Stale GitHub connection/webhook**: App Runner can lose webhook delivery after
   repo/org permission changes or GitHub App authorization updates.

A practical hardening step is to trigger deployments from GitHub Actions on each
push to `main`.

### Included fallback workflow (recommended)

This repository now includes `.github/workflows/aws-apprunner-redeploy.yml`.
On every push to `main` (or manual dispatch), it calls:

- `aws apprunner start-deployment --service-arn <your-service-arn>`

That forces App Runner to pull the latest watched branch commit even if webhook
notifications were missed.

### Required GitHub configuration

Set these in your GitHub repo before enabling the workflow:

- **Repository variable:** `AWS_REGION` (for example `us-east-1`)
- **Repository variable:** `AWS_APP_RUNNER_SERVICE_ARN`
- **Repository secret:** `AWS_GITHUB_DEPLOY_ROLE_ARN` (IAM role trusted by GitHub OIDC)

IAM role needs at minimum:

- `apprunner:StartDeployment` on the target service ARN.

## Common failure points

1. **Wrong source directory**
   - Symptom: App Runner ignores `apprunner.yaml` or cannot find dependencies.
   - Fix: ensure source directory is `/`.

2. **Config file name mismatch**
   - Symptom: App Runner uses console defaults instead of repo-defined commands.
   - Fix: file must be exactly `apprunner.yaml` (not `.yml`).

3. **Dependency install omitted for Python 3.11 flow**
   - Symptom: startup fails with import errors.
   - Fix: keep the explicit `pip3 install .` pre-run command in `apprunner.yaml`.

4. **Wrong app command**
   - Symptom: service starts but immediately exits or returns 502.
   - Fix: use `uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers`.

5. **Static web asset issues**
   - Symptom: `/dashboard` (or `/operator` redirect) 404 or missing JS/CSS.
   - Fix: confirm `apps/api/static/*` exists in the deployed branch and source directory is root.

## Fast checklist when "the website still shows the old version"

Use this order to isolate where the stale version is coming from:

1. **Confirm the live service is your latest commit**
   - In App Runner, open your service **Deployments** tab and verify the latest
     deployment references the same Git commit you just pushed.
   - If it does not, run **Deploy** (or call `StartDeployment`) to force a pull.

2. **Verify branch and source directory**
   - Ensure App Runner is watching the branch you actually updated.
   - Ensure source directory is `/` so `apprunner.yaml` and static assets are used.

3. **Check you are opening the correct domain**
   - Confirm you are using the current App Runner default URL or your active custom
     domain, not an old environment URL.
   - If both `www` and apex are configured, verify which one your DNS points to.

4. **Bypass browser cache / service worker**
   - Hard refresh (`Ctrl+Shift+R` on Windows/Linux, `Cmd+Shift+R` on macOS).
   - Open an incognito/private window.
   - In browser devtools → Application, unregister any old service worker and clear
     site data for the domain if stale assets persist.

5. **Validate static asset freshness directly**
   - Open a JS asset URL directly (for example `/web/modules/boot.js`) and confirm
     the response contains your latest changes.
   - If this file is old, the issue is deployment/source branch; if this file is new
     but UI is old, the issue is browser cache or service worker.
## Railway decommissioning note

This repository is intentionally configured for AWS deployment workflows only.
If a historical Railway service exists, treat it as decommissioned and keep Railway
GitHub integration disabled to prevent accidental redeploys.
