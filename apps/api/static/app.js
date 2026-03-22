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

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
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

function getRoute() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length === 0 || parts[0] === "dashboard") return { page: "dashboard" };
  if (parts[0] === "upload") return { page: "upload" };
  if (parts[0] === "app" && parts[1] === "runs" && parts[2]) return { page: "run-detail", runId: parts[2] };
  if (parts[0] === "app" && parts[1] === "runs") return { page: "runs" };
  if (parts[0] === "app" && parts[1] === "results" && parts[2]) return { page: "result-detail", resultId: parts[2] };
  return { page: "dashboard" };
}

const state = {
  activeRun: null,
  runs: [],
  dashboardRecent: [],
  runRecent: [],
  charts: {
    drift: null,
    composite: null,
  },
};

function setLoading(isLoading, message = "Loading...") {
  const el = qs("#globalLoading");
  if (!el) return;
  if (isLoading) {
    el.classList.remove("hidden");
    el.textContent = message;
  } else {
    el.classList.add("hidden");
    el.textContent = "Loading...";
  }
}

function setStatus(message = "", isError = false) {
  const el = qs("#globalStatus");
  if (!el) return;
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    el.className = "status hidden";
    return;
  }
  el.className = `status ${isError ? "error" : "ok"}`;
  el.textContent = message;
  el.classList.remove("hidden");
}

function setPage(page) {
  const titles = {
    dashboard: ["Dashboard", "Live summary of the active run"],
    runs: ["Runs", "Create, inspect, and activate runs"],
    upload: ["Upload", "Upload telemetry CSV into the active run"],
    "run-detail": ["Run Detail", "Deep inspection of run outputs"],
    "result-detail": ["Result Detail", "Focused view for a single result"],
  };
  qsa(".page").forEach((p) => p.classList.add("hidden"));
  const pageEl = qs(`#page-${page}`);
  if (pageEl) pageEl.classList.remove("hidden");
  const [title, subtitle] = titles[page] || ["Neraium", ""];
  qs("#pageTitle").textContent = title;
  qs("#pageSubtitle").textContent = subtitle;
  qsa(".nav a").forEach((a) => a.classList.remove("active"));
  if (page === "dashboard") qs('[data-nav="dashboard"]')?.classList.add("active");
  if (page === "runs" || page === "run-detail") qs('[data-nav="runs"]')?.classList.add("active");
  if (page === "upload") qs('[data-nav="upload"]')?.classList.add("active");
}

async function ensureActiveRun() {
  const active = await fetchJson("/runs/active");
  if (active.run) return active.run;
  const created = await fetchJson("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "Default Run",
      config: { source: "product-ui-default" },
      activate: true,
    }),
  });
  return created.run;
}

function updateActiveRunHeader(run) {
  state.activeRun = run || null;
  qs("#activeRunName").textContent = run?.name || "No active run";
  qs("#activeRunId").textContent = run?.run_id || "-";
  if (run?.run_id) {
    window.localStorage.setItem("active_run_id", run.run_id);
  }
}

async function loadRuns() {
  const runsEnv = await fetchJson("/runs?limit=200");
  state.runs = runsEnv.runs || [];
  if (runsEnv.active_run) {
    updateActiveRunHeader(runsEnv.active_run);
  }
  renderRunsList();
}

function renderRunsList() {
  const tbody = qs("#runsBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  state.runs.forEach((run) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(run.name || "")}</td>
      <td class="mono">${escapeHtml(run.run_id || "")}</td>
      <td>${run.is_active ? "active" : escapeHtml(run.status || "open")}</td>
      <td>${escapeHtml(run.created_at || "")}</td>
      <td class="row">
        <a href="/app/runs/${encodeURIComponent(run.run_id)}">Open</a>
        ${run.is_active ? "" : `<button class="small secondary" data-activate-run="${escapeHtml(run.run_id)}" type="button">Activate</button>`}
      </td>
    `;
    tbody.appendChild(tr);
  });
  qsa("[data-activate-run]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const runId = btn.getAttribute("data-activate-run");
      if (!runId) return;
      try {
        setLoading(true, "Activating run...");
        const out = await fetchJson(`/runs/${encodeURIComponent(runId)}/activate`, { method: "POST" });
        updateActiveRunHeader(out.run);
        await loadRuns();
        setStatus(`Activated run ${out.run.name}`);
      } catch (err) {
        setStatus(String(err.message || err), true);
      } finally {
        setLoading(false);
      }
    });
  });
}

function renderDashboardMetrics(latest) {
  qs("#metricDrift").textContent = toPretty(structuralDriftFromResult(latest));
  qs("#metricComposite").textContent = toPretty(compositeInstabilityFromResult(latest));
  qs("#metricPhase").textContent = toPretty(phaseFromResult(latest));
  qs("#metricTrend").textContent = toPretty(trendFromResult(latest));
  qs("#metricRisk").textContent = toPretty(latest?.risk_level);
  qs("#metricState").textContent = toPretty(latest?.state || latest?.interpreted_state);
  qs("#metricOperatorMessage").textContent = latest?.operator_message || "No operator message yet.";
}

function renderDashboardRecent(results) {
  const tbody = qs("#dashboardRecentBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  (results || []).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${toPretty(r.result_id)}</td>
      <td>${toPretty(r.timestamp || r.persisted_at)}</td>
      <td>${toPretty(phaseFromResult(r))}</td>
      <td>${toPretty(r.risk_level)}</td>
      <td>${toPretty(structuralDriftFromResult(r))}</td>
      <td>${toPretty(compositeInstabilityFromResult(r))}</td>
      <td><a href="/app/results/${encodeURIComponent(r.result_id)}?run_id=${encodeURIComponent(state.activeRun?.run_id || "")}">View</a></td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadDashboard() {
  const runId = state.activeRun?.run_id || "";
  const recentEnv = await fetchJson(apiUrl("/results/recent", { run_id: runId, limit: 50 }));
  const latest = (recentEnv.results && recentEnv.results[0]) || null;
  state.dashboardRecent = recentEnv.results || [];
  renderDashboardMetrics(latest);
  renderDashboardRecent(state.dashboardRecent);
}

function exportData(format, runId) {
  const url = apiUrl("/results/export", { format, run_id: runId || "", limit: 500 });
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

async function uploadCsvToActiveRun() {
  const fileInput = qs("#csvFileInput");
  const file = fileInput?.files?.[0];
  if (!file) throw new Error("Choose a CSV file first");
  const runId = state.activeRun?.run_id;
  if (!runId) throw new Error("No active run found");
  const csvText = await parseCsvText(file);
  return fetchJson(`/ingest/csv?run_id=${encodeURIComponent(runId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ csv_text: csvText }),
  });
}

async function createRunFromForm() {
  const name = String(qs("#runNameInput").value || "").trim();
  const configRaw = String(qs("#runConfigInput").value || "").trim();
  const activate = Boolean(qs("#runActivateInput").checked);
  if (!name) throw new Error("Run name is required");
  let config = {};
  if (configRaw) {
    try {
      const parsed = JSON.parse(configRaw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        config = parsed;
      } else {
        throw new Error("Run config must be a JSON object");
      }
    } catch (err) {
      throw new Error(`Invalid run config JSON: ${err.message || err}`);
    }
  }
  const out = await fetchJson("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config, activate }),
  });
  return out.run;
}

function destroyCharts() {
  if (state.charts.drift) {
    state.charts.drift.destroy();
    state.charts.drift = null;
  }
  if (state.charts.composite) {
    state.charts.composite.destroy();
    state.charts.composite = null;
  }
}

function renderRunDetailCharts(results) {
  destroyCharts();
  const labels = results.map((r) => String(r.timestamp || r.persisted_at || r.result_id || ""));
  const driftValues = results.map((r) => structuralDriftFromResult(r) ?? 0);
  const compositeValues = results.map((r) => compositeInstabilityFromResult(r) ?? 0);

  const driftCtx = qs("#driftChart");
  const compCtx = qs("#compositeChart");
  if (driftCtx && window.Chart) {
    state.charts.drift = new window.Chart(driftCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "structural_drift_score",
            data: driftValues,
            borderColor: "#60a5fa",
            backgroundColor: "rgba(96, 165, 250, 0.2)",
            fill: true,
            tension: 0.2,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } },
        },
      },
    });
  }
  if (compCtx && window.Chart) {
    state.charts.composite = new window.Chart(compCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "composite_instability",
            data: compositeValues,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.2)",
            fill: true,
            tension: 0.2,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } },
        },
      },
    });
  }
}

function renderPhaseTimeline(results) {
  const el = qs("#phaseTimeline");
  if (!el) return;
  el.innerHTML = "";
  results.forEach((r) => {
    const phase = phaseFromResult(r);
    const item = document.createElement("div");
    item.className = "timeline-item";
    item.innerHTML = `
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-phase">${escapeHtml(String(phase))}</div>
        <div class="timeline-meta">${escapeHtml(String(r.timestamp || r.persisted_at || ""))}</div>
      </div>
    `;
    el.appendChild(item);
  });
}

function renderOperatorMessages(results) {
  const el = qs("#operatorMessagesList");
  if (!el) return;
  el.innerHTML = "";
  const msgs = results
    .map((r) => ({ id: r.result_id, ts: r.timestamp || r.persisted_at, msg: r.operator_message || "" }))
    .filter((x) => x.msg);
  if (msgs.length === 0) {
    const li = document.createElement("li");
    li.className = "message-item";
    li.textContent = "No operator messages available for this run.";
    el.appendChild(li);
    return;
  }
  msgs.slice(0, 20).forEach((x) => {
    const li = document.createElement("li");
    li.className = "message-item";
    li.innerHTML = `
      <div class="msg-head">#${escapeHtml(String(x.id))} · ${escapeHtml(String(x.ts || ""))}</div>
      <div>${escapeHtml(String(x.msg))}</div>
    `;
    el.appendChild(li);
  });
}

function renderRunResultsTable(results) {
  const tbody = qs("#runResultsBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  results.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${toPretty(r.result_id)}</td>
      <td>${toPretty(r.timestamp || r.persisted_at)}</td>
      <td>${toPretty(r.state)}</td>
      <td>${toPretty(phaseFromResult(r))}</td>
      <td>${toPretty(trendFromResult(r))}</td>
      <td>${toPretty(r.risk_level)}</td>
      <td>${toPretty(structuralDriftFromResult(r))}</td>
      <td>${toPretty(compositeInstabilityFromResult(r))}</td>
      <td>${toPretty(r.operator_message)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadRunDetail(runId) {
  const runRes = await fetchJson(`/runs/${encodeURIComponent(runId)}`);
  const run = runRes.run;
  qs("#runDetailTitle").textContent = `Run: ${run.name}`;
  qs("#runDetailMeta").textContent = `${run.run_id} · status=${run.status} · created=${run.created_at}`;
  const recentEnv = await fetchJson(apiUrl("/results/recent", { run_id: runId, limit: 200 }));
  const results = (recentEnv.results || []).slice().reverse();
  state.runRecent = recentEnv.results || [];
  renderRunDetailCharts(results);
  renderPhaseTimeline(results);
  renderOperatorMessages(results);
  renderRunResultsTable(results);

  qs("#runDetailExportJsonBtn").onclick = () => exportData("json", runId);
  qs("#runDetailExportCsvBtn").onclick = () => exportData("csv", runId);
}

async function loadResultDetail(resultId) {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run_id") || state.activeRun?.run_id || "";
  const env = await fetchJson(apiUrl(`/results/${encodeURIComponent(resultId)}`, { run_id: runId }));
  const r = env.result;
  const grid = qs("#resultDetailGrid");
  grid.innerHTML = "";
  const keys = [
    ["result_id", r.result_id],
    ["run_id", r.run_id],
    ["timestamp", r.timestamp || r.persisted_at],
    ["state", r.state],
    ["phase", phaseFromResult(r)],
    ["trend", trendFromResult(r)],
    ["risk_level", r.risk_level],
    ["structural_drift_score", structuralDriftFromResult(r)],
    ["composite_instability", compositeInstabilityFromResult(r)],
    ["operator_message", r.operator_message],
  ];
  keys.forEach(([k, v]) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<h3>${escapeHtml(String(k))}</h3><p class="metric-value">${escapeHtml(toPretty(v))}</p>`;
    grid.appendChild(card);
  });
}

async function wireEvents() {
  qs("#refreshBtn")?.addEventListener("click", async () => {
    try {
      setLoading(true, "Refreshing...");
      const route = getRoute();
      await loadRuns();
      if (route.page === "dashboard") await loadDashboard();
      if (route.page === "runs") renderRunsList();
      if (route.page === "run-detail") await loadRunDetail(route.runId);
      if (route.page === "result-detail") await loadResultDetail(route.resultId);
      setStatus("Refreshed");
    } catch (err) {
      setStatus(String(err.message || err), true);
    } finally {
      setLoading(false);
    }
  });

  qs("#runCreateForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      setLoading(true, "Creating run...");
      const run = await createRunFromForm();
      if (run.is_active) updateActiveRunHeader(run);
      qs("#runNameInput").value = "";
      qs("#runConfigInput").value = "";
      qs("#runActivateInput").checked = true;
      await loadRuns();
      setStatus(`Run created: ${run.name}`);
    } catch (err) {
      setStatus(String(err.message || err), true);
    } finally {
      setLoading(false);
    }
  });

  qs("#csvUploadForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      setLoading(true, "Uploading CSV...");
      const out = await uploadCsvToActiveRun();
      await loadDashboard();
      setStatus(`CSV ingested (${out.count} rows processed)`);
    } catch (err) {
      setStatus(String(err.message || err), true);
    } finally {
      setLoading(false);
    }
  });

  qs("#exportJsonBtn")?.addEventListener("click", () => exportData("json", state.activeRun?.run_id || ""));
  qs("#exportCsvBtn")?.addEventListener("click", () => exportData("csv", state.activeRun?.run_id || ""));
}

async function init() {
  const route = getRoute();
  const routeToPage = {
    dashboard: "dashboard",
    runs: "runs",
    upload: "upload",
    "run-detail": "run-detail",
    "result-detail": "result-detail",
  };
  setPage(routeToPage[route.page] || "dashboard");
  try {
    setLoading(true, "Initializing...");
    const activeRun = await ensureActiveRun();
    updateActiveRunHeader(activeRun);
    await loadRuns();
    if (route.page === "dashboard") await loadDashboard();
    if (route.page === "runs") renderRunsList();
    if (route.page === "run-detail") await loadRunDetail(route.runId);
    if (route.page === "result-detail") await loadResultDetail(route.resultId);
    await wireEvents();
    setStatus("");
  } catch (err) {
    setStatus(String(err.message || err), true);
  } finally {
    setLoading(false);
  }
}

init();
