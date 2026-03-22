function qs(sel) {
  return document.querySelector(sel);
}

function qsa(sel) {
  return Array.from(document.querySelectorAll(sel));
}

function apiUrl(path, params) {
  const u = new URL(path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && String(v).length > 0) {
        u.searchParams.set(k, String(v));
      }
    });
  }
  return u.toString();
}

async function fetchJson(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body && body.detail ? String(body.detail) : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

function toPretty(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : "-";
  return String(v);
}

function getPathInfo() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length === 0) return { page: "home" };
  if (parts[0] === "runs" && parts[1]) return { page: "run", runId: parts[1] };
  if (parts[0] === "results" && parts[1]) return { page: "result", resultId: parts[1] };
  return { page: "home" };
}

async function ensureActiveRun() {
  const active = await fetchJson("/runs/active");
  if (active.run) return active.run;
  const created = await fetchJson("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "Default Run",
      config: { source: "mvp-web-default" },
      activate: true,
    }),
  });
  return created.run;
}

function currentRunIdFromRouteOrStorage(pathInfo) {
  if (pathInfo.page === "run") return pathInfo.runId;
  return window.localStorage.getItem("active_run_id") || "";
}

function setStatus(msg, isErr) {
  const el = qs("#status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = isErr ? "status status-error" : "status";
}

function phaseFromResult(r) {
  if (!r) return "-";
  return r.phase || r.state || r.interpreted_state || "-";
}

function trendFromResult(r) {
  if (!r) return "-";
  return r.trend || "-";
}

function structuralDriftFromResult(r) {
  if (!r) return null;
  const v = r.structural_drift_score;
  return typeof v === "number" ? v : null;
}

function compositeInstabilityFromResult(r) {
  if (!r) return null;
  if (typeof r.latest_instability === "number") return r.latest_instability;
  const analytics = r.experimental_analytics;
  if (analytics && typeof analytics.composite_instability === "number") {
    return analytics.composite_instability;
  }
  return null;
}

function renderDashboard(latest) {
  qs("#metric-structural-drift").textContent = toPretty(structuralDriftFromResult(latest));
  qs("#metric-composite-instability").textContent = toPretty(compositeInstabilityFromResult(latest));
  qs("#metric-phase").textContent = toPretty(phaseFromResult(latest));
  qs("#metric-trend").textContent = toPretty(trendFromResult(latest));
  qs("#metric-risk").textContent = toPretty(latest ? latest.risk_level : null);
  qs("#operator-message").textContent = latest && latest.operator_message
    ? String(latest.operator_message)
    : "No operator message available yet.";
}

function renderHistory(results, runId) {
  const tbody = qs("#history-body");
  tbody.innerHTML = "";
  (results || []).forEach((r) => {
    const tr = document.createElement("tr");
    const rid = r.result_id;
    tr.innerHTML = `
      <td>${toPretty(rid)}</td>
      <td>${toPretty(r.timestamp || r.persisted_at)}</td>
      <td>${toPretty(phaseFromResult(r))}</td>
      <td>${toPretty(structuralDriftFromResult(r))}</td>
      <td>${toPretty(compositeInstabilityFromResult(r))}</td>
      <td>${toPretty(r.risk_level)}</td>
      <td><a href="/results/${encodeURIComponent(rid)}${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}">view</a></td>
    `;
    tbody.appendChild(tr);
  });
}

function exportData(format, runId) {
  const url = apiUrl("/results/export", { format, run_id: runId, limit: 500 });
  window.location.href = url;
}

function parseCsvText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read CSV file"));
    reader.readAsText(file, "utf-8");
  });
}

async function submitRunForm(existingRunId) {
  const name = String(qs("#run-name").value || "").trim();
  const baselineWindow = Number(qs("#cfg-baseline").value);
  const recentWindow = Number(qs("#cfg-recent").value);
  const activate = qs("#cfg-activate").checked;
  if (!name) throw new Error("Run name is required");
  const payload = {
    name,
    config: {
      baseline_window: baselineWindow,
      recent_window: recentWindow,
    },
    activate,
  };
  if (existingRunId) {
    const u = `/runs/${encodeURIComponent(existingRunId)}`;
    const body = {
      name: payload.name,
      config: payload.config,
    };
    return fetchJson(u, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  return fetchJson("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function submitCsvUpload(runId) {
  const input = qs("#csv-file");
  const file = input && input.files ? input.files[0] : null;
  if (!file) throw new Error("Choose a CSV file first");
  const csvText = await parseCsvText(file);
  return fetchJson(`/ingest/csv?run_id=${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ csv_text: csvText }),
  });
}

function setRunIdUi(runId) {
  qs("#active-run-id").textContent = runId ? runId : "-";
  if (runId) window.localStorage.setItem("active_run_id", runId);
}

async function loadRunSelector(currentRunId) {
  const runsResp = await fetchJson("/runs?limit=100");
  const sel = qs("#run-selector");
  sel.innerHTML = "";
  (runsResp.runs || []).forEach((run) => {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    const activeTag = run.is_active ? " [active]" : "";
    opt.textContent = `${run.name} (${run.run_id})${activeTag}`;
    if (run.run_id === currentRunId) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadLatestAndRecent(runId) {
  const latestEnv = await fetchJson(apiUrl("/results/latest", { run_id: runId }));
  const recentEnv = await fetchJson(apiUrl("/results/recent", { run_id: runId, limit: 50 }));
  const latest = latestEnv.latest || (recentEnv.results && recentEnv.results[0]) || null;
  renderDashboard(latest);
  renderHistory(recentEnv.results || [], runId);
}

async function loadResultDetail(pathInfo) {
  const card = qs("#result-detail");
  if (pathInfo.page !== "result") {
    card.style.display = "none";
    return;
  }
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run_id") || "";
  const data = await fetchJson(apiUrl(`/results/${encodeURIComponent(pathInfo.resultId)}`, { run_id: runId }));
  card.style.display = "block";
  qs("#result-detail-id").textContent = String(data.result_id || pathInfo.resultId);
  qs("#result-detail-run").textContent = toPretty(data.run_id);
  qs("#result-detail-ts").textContent = toPretty(data.timestamp || data.persisted_at);
  qs("#result-detail-pre").textContent = JSON.stringify(data, null, 2);
}

async function init() {
  const pathInfo = getPathInfo();
  try {
    setStatus("Loading...");
    const runIdFromRouteOrStorage = currentRunIdFromRouteOrStorage(pathInfo);
    let activeRun = null;
    if (runIdFromRouteOrStorage) {
      const runResp = await fetchJson(`/runs/${encodeURIComponent(runIdFromRouteOrStorage)}`);
      activeRun = runResp.run;
    } else {
      activeRun = await ensureActiveRun();
    }
    const runId = activeRun ? activeRun.run_id : "";
    setRunIdUi(runId);
    qs("#run-name").value = activeRun && activeRun.name ? activeRun.name : "";
    const cfg = (activeRun && activeRun.config) || {};
    if (cfg.baseline_window) qs("#cfg-baseline").value = String(cfg.baseline_window);
    if (cfg.recent_window) qs("#cfg-recent").value = String(cfg.recent_window);

    await loadRunSelector(runId);
    await loadLatestAndRecent(runId);
    await loadResultDetail(pathInfo);

    qs("#run-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        setStatus("Saving run...");
        const existing = qs("#edit-existing").checked ? runId : "";
        const out = await submitRunForm(existing);
        const run = out.run;
        if (run && run.run_id) {
          setRunIdUi(run.run_id);
          if (run.is_active) {
            await loadLatestAndRecent(run.run_id);
          }
          await loadRunSelector(run.run_id);
        }
        setStatus("Run saved");
      } catch (err) {
        setStatus(String(err.message || err), true);
      }
    });

    qs("#btn-activate-run").addEventListener("click", async () => {
      const selected = qs("#run-selector").value;
      if (!selected) return;
      try {
        setStatus("Activating run...");
        const out = await fetchJson(`/runs/${encodeURIComponent(selected)}/activate`, { method: "POST" });
        const run = out.run;
        setRunIdUi(run.run_id);
        await loadRunSelector(run.run_id);
        await loadLatestAndRecent(run.run_id);
        setStatus("Run activated");
      } catch (err) {
        setStatus(String(err.message || err), true);
      }
    });

    qs("#csv-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const rid = qs("#active-run-id").textContent.trim();
      if (!rid || rid === "-") {
        setStatus("No active run available", true);
        return;
      }
      try {
        setStatus("Uploading CSV...");
        const out = await submitCsvUpload(rid);
        const latest = out.latest || (out.results && out.results[0]) || null;
        renderDashboard(latest);
        await loadLatestAndRecent(rid);
        setStatus(`CSV ingested (${out.count} rows processed)`);
      } catch (err) {
        setStatus(String(err.message || err), true);
      }
    });

    qsa("[data-export-format]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rid = qs("#active-run-id").textContent.trim();
        const fmt = btn.getAttribute("data-export-format");
        exportData(fmt, rid && rid !== "-" ? rid : "");
      });
    });

    setStatus("Ready");
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

init();
