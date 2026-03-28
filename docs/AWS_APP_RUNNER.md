# AWS App Runner source-repository deployment (FastAPI)

This guide deploys Neraium from GitHub to AWS App Runner using the repository config file (`apprunner.yaml`) at repo root.

## Readiness audit summary

- **Source directory assumption:** deploy from repository root (`/`) because `apprunner.yaml` and root `requirements.txt` are at root.
- **Dependency installation path:** explicit `pip3 install -r requirements.txt` in both App Runner build and run pre-run phase.
- **FastAPI entrypoint:** `apps.api.main:app`.
- **Static assets:** served from `apps/api/static` through router + `/web` static mount.
- **Health endpoint:** `GET /health` returns JSON `200`.
- **Requirements layout:** root `requirements.txt` includes `-r apps/api/requirements.txt` so installs are unambiguous from repo root.

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
- Operator UI: `/operator`
- Operator workflow: `/operator/workflow`
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
   - Fix: keep explicit pip install commands in `apprunner.yaml`.

4. **Wrong app command**
   - Symptom: service starts but immediately exits or returns 502.
   - Fix: use `uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers`.

5. **Static/operator file issues**
   - Symptom: `/operator` 404 or missing JS/CSS.
   - Fix: confirm `apps/api/static/*` exists in the deployed branch and source directory is root.
