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

function ensureChartJsLoaded() {
  if (window.Chart) return Promise.resolve();
  if (chartJsLoadPromise) return chartJsLoadPromise;
  chartJsLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js";
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Chart runtime failed to load."));
    document.head.appendChild(script);
  }).catch((err) => {
    chartJsLoadPromise = null;
    throw err;
  });
  return chartJsLoadPromise;
}

function renderRunDetailCharts(results) {
  if (!window.Chart) return;
  const labels = results.map((r) => String(r.timestamp || r.persisted_at || r.result_id || ""));
  const driftValues = results.map((r) => structuralDriftFromResult(r) ?? 0);
  const compositeValues = results.map((r) => compositeInstabilityFromResult(r) ?? 0);
  const pointRadius = labels.map(() => 0);
  const pointHoverRadius = labels.map(() => 0);
  const sharedOptions = buildTrendChartOptions();

  const driftCtx = qs("#driftChart");
  const compCtx = qs("#compositeChart");
  if (driftCtx && window.Chart) {
    if (state.charts.drift && state.charts.drift.canvas === driftCtx) {
      state.charts.drift.data.labels = labels;
      state.charts.drift.data.datasets[0].data = driftValues;
      state.charts.drift.update("none");
    } else {
      if (state.charts.drift) {
        state.charts.drift.destroy();
        state.charts.drift = null;
      }
      state.charts.drift = new window.Chart(driftCtx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "structural_drift_score",
              data: driftValues,
              borderColor: "#79abff",
              backgroundColor: "rgba(106, 156, 250, 0.24)",
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius,
              pointHoverRadius,
            },
          ],
        },
        options: sharedOptions,
      });
    }
  }
  if (compCtx && window.Chart) {
    if (state.charts.composite && state.charts.composite.canvas === compCtx) {
      state.charts.composite.data.labels = labels;
      state.charts.composite.data.datasets[0].data = compositeValues;
      state.charts.composite.update("none");
    } else {
      if (state.charts.composite) {
        state.charts.composite.destroy();
        state.charts.composite = null;
      }
      state.charts.composite = new window.Chart(compCtx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "composite_instability",
              data: compositeValues,
              borderColor: "#ffbf56",
              backgroundColor: "rgba(242, 179, 74, 0.2)",
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius: labels.map(() => 0),
              pointHoverRadius: labels.map(() => 0),
            },
          ],
        },
        options: sharedOptions,
      });
    }
  }
  setTrendPlaybackCursorMarker(-1, labels.length);
}

function clearRunDetailObserver() {
  if (state.ui.runDetailObserver) {
    state.ui.runDetailObserver.disconnect();
    state.ui.runDetailObserver = null;
  }
}

async function hydrateRunDetailSection(section, runId) {
  if (section === "overview" || state.ui.runDetailHydratedSections[section]) return;
  state.ui.runDetailHydratedSections[section] = true;
  if (section === "trends") {
    await ensureChartJsLoaded();
    renderRunDetailFromState();
    return;
  }
  if (section === "results") {
    await loadRunDetailBackgroundHistory(runId);
    renderRunDetailFromState();
    return;
  }
  if (section === "geometry") {
    await loadRunGeometry(runId, state.runRecent[0]?.result_id ?? null);
    renderRunDetailFromState();
  }
}

function setTrendPlaybackCursorMarker(idx, lengthHint = 0) {
  const charts = [state.charts.drift, state.charts.composite];
  charts.forEach((chart) => {
    if (!chart?.data?.datasets?.[0]) return;
    const len = chart.data.labels?.length || lengthHint || 0;
    const radius = Array.from({ length: len }, (_v, i) => (i === idx ? 3 : 0));
    const hover = Array.from({ length: len }, (_v, i) => (i === idx ? 4 : 0));
    chart.data.datasets[0].pointRadius = radius;
    chart.data.datasets[0].pointHoverRadius = hover;
    chart.update("none");
  });
}

function renderPhaseTimeline(results) {
  const el = qs("#phaseTimeline");
  if (!el) return;
  el.innerHTML = "";
  results.forEach((r, idx) => {
    const phase = phaseFromResult(r);
    const prev = idx > 0 ? results[idx - 1] : null;
    const severity = transitionSeverity(prev, r);
    const transition = transitionLabel(prev, r);
    const prevRisk = normalizeRiskLevel(prev?.risk_level);
    const nextRisk = normalizeRiskLevel(r.risk_level);
    const prevDrift = structuralDriftFromResult(prev);
    const nextDrift = structuralDriftFromResult(r);
    const driftJump =
      typeof prevDrift === "number" && typeof nextDrift === "number"
        ? Math.abs(nextDrift - prevDrift)
        : 0;
    const keyEvent = (nextRisk === "HIGH" && prevRisk !== "HIGH") || driftJump >= 0.14;
    const item = document.createElement("div");
    item.className = `timeline-item timeline-${severity}${keyEvent ? " timeline-key-event" : ""}`;
    item.innerHTML = `
      <div class="timeline-dot timeline-dot-${severity}"></div>
      <div class="timeline-content">
        <div class="timeline-primary">
          <span class="timeline-state">${escapeHtml(String(r.state || r.interpreted_state || "-"))}</span>
          <span>${riskBadgeHtml(r.risk_level)}</span>
          <span class="timeline-trend-chip">${escapeHtml(String(trendFromResult(r) || "-"))}</span>
          ${keyEvent ? '<span class="demo-event-pill">Key event</span>' : ""}
        </div>
        <div class="timeline-phase">${phaseBadgeHtml(phase)}</div>
        <div class="timeline-transition">${escapeHtml(transition)}</div>
        <div class="timeline-meta">${escapeHtml(String(r.timestamp || r.persisted_at || ""))}</div>
      </div>
    `;
    el.appendChild(item);
  });
}

function renderOperatorMessages(results, opts = {}) {
  const el = qs("#operatorMessagesList");
  if (!el) return;
  el.innerHTML = "";
  const emphasize = Boolean(opts.emphasize);
  const msgs = results
    .slice()
    .reverse()
    .map((r, idx, arr) => {
      const prev = idx + 1 < arr.length ? arr[idx + 1] : null;
      return {
        id: r.result_id,
        ts: r.timestamp || r.persisted_at,
        severity: transitionSeverity(prev, r),
        msg: demoFriendlyOperatorMessage(r, prev),
      };
    })
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
    li.className = `message-item ${emphasize ? `message-item-${x.severity}` : ""}`.trim();
    li.innerHTML = `
      <div class="msg-head">#${escapeHtml(String(x.id))} · ${escapeHtml(String(x.ts || ""))}</div>
      <div>${escapeHtml(String(x.msg))}</div>
    `;
    el.appendChild(li);
  });
}

function filterRunResults(results) {
  const q = state.runDetailView.search.trim().toLowerCase();
  if (!q) return results.slice();
  return results.filter((r) => {
    const fields = [
      r.result_id,
      r.timestamp,
      r.persisted_at,
      r.state,
      phaseFromResult(r),
      trendFromResult(r),
      r.risk_level,
      r.operator_message,
    ];
    return fields.some((value) => String(value || "").toLowerCase().includes(q));
  });
}

function sortRunResults(results) {
  const mode = state.runDetailView.sort;
  const items = results.slice();
  const riskRank = { LOW: 1, MEDIUM: 2, HIGH: 3 };
  if (mode === "timestamp_asc") return items.sort((a, b) => parseTime(a.timestamp || a.persisted_at) - parseTime(b.timestamp || b.persisted_at));
  if (mode === "drift_desc") return items.sort((a, b) => (structuralDriftFromResult(b) ?? -Infinity) - (structuralDriftFromResult(a) ?? -Infinity));
  if (mode === "drift_asc") return items.sort((a, b) => (structuralDriftFromResult(a) ?? Infinity) - (structuralDriftFromResult(b) ?? Infinity));
  if (mode === "composite_desc") return items.sort((a, b) => (compositeInstabilityFromResult(b) ?? -Infinity) - (compositeInstabilityFromResult(a) ?? -Infinity));
  if (mode === "composite_asc") return items.sort((a, b) => (compositeInstabilityFromResult(a) ?? Infinity) - (compositeInstabilityFromResult(b) ?? Infinity));
  if (mode === "risk_desc") {
    return items.sort((a, b) => (riskRank[normalizeRiskLevel(b.risk_level)] || 0) - (riskRank[normalizeRiskLevel(a.risk_level)] || 0));
  }
  if (mode === "risk_asc") {
    return items.sort((a, b) => (riskRank[normalizeRiskLevel(a.risk_level)] || 0) - (riskRank[normalizeRiskLevel(b.risk_level)] || 0));
  }
  return items.sort((a, b) => parseTime(b.timestamp || b.persisted_at) - parseTime(a.timestamp || a.persisted_at));
}

function renderRunResultsTable(results, opts = {}) {
  const tbody = qs("#runResultsBody");
  const empty = qs("#runResultsEmpty");
  if (!tbody) return;
  tbody.innerHTML = "";
  const latestId = opts.latestResultId != null ? String(opts.latestResultId) : "";
  if (empty) {
    if (results.length === 0) {
      empty.classList.remove("hidden");
      const p = empty.querySelector("p");
      if (p) {
        const uiTruth = buildFrontendUiState();
        p.textContent = uiTruth.mode === "validation"
          ? "No results match these filters — clear filters or widen the time range."
          : "No results match these filters.";
      }
    } else empty.classList.add("hidden");
  }
  results.forEach((r, idx) => {
    const prev = idx + 1 < results.length ? results[idx + 1] : null;
    const transition = transitionLabel(prev, r);
    const severity = transitionSeverity(prev, r);
    const stateText = String(r.state || r.interpreted_state || "-");
    const riskLvl = normalizeRiskLevel(r.risk_level);
    const tr = document.createElement("tr");
    const isLatest = latestId && String(r.result_id) === latestId;
    tr.className = `result-row result-row-${severity} result-row-risk-${riskLvl}${isLatest ? " result-row-latest" : ""}`;
    tr.setAttribute("data-risk-level", riskLvl);
    tr.innerHTML = `
      <td>${toPretty(r.result_id)}</td>
      <td>${toPretty(r.timestamp || r.persisted_at)}</td>
      <td><span class="state-pill state-pill-${stateTone(stateText)}">${escapeHtml(stateText)}</span></td>
      <td>${phaseBadgeHtml(phaseFromResult(r))}</td>
      <td><span class="trend-pill">${escapeHtml(String(trendFromResult(r) || "-"))}</span></td>
      <td>${riskBadgeHtml(r.risk_level)}</td>
      <td><span class="row-transition row-transition-${severity}">${escapeHtml(transition)}</span></td>
      <td>${toPretty(structuralDriftFromResult(r))}</td>
      <td>${toPretty(compositeInstabilityFromResult(r))}</td>
      <td>
        <div>${toPretty(r.operator_message)}</div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function setRangeButtonState(rangeValue) {
  qsa("#runRangeControls [data-range]").forEach((btn) => {
    if (btn.getAttribute("data-range") === String(rangeValue)) btn.classList.add("active");
    else btn.classList.remove("active");
  });
}

function currentRangeSlice(resultsChronological) {
  const range = state.runDetailView.range;
  if (range === "all") return resultsChronological.slice();
  const n = Number.parseInt(range, 10);
  if (!Number.isFinite(n) || n <= 0) return resultsChronological.slice();
  return resultsChronological.slice(-n);
}

function buildStructuralFlowTimeline(resultsChronological) {
  return (resultsChronological || []).map((r, idx) => ({
    idx,
    resultId: String(r?.result_id ?? ""),
    timestamp: String(r?.timestamp || r?.persisted_at || ""),
    driftN: Math.max(0, Math.min(1, Number(structuralDriftFromResult(r) ?? 0))),
    instN: Math.max(0, Math.min(1, Number(compositeInstabilityFromResult(r) ?? 0))),
  }));
}

function setFlowModeButtonState() {
  qsa("[data-flow-mode]").forEach((btn) => {
    const active = btn.getAttribute("data-flow-mode") === state.runDetailView.flowPlaybackMode;
    btn.classList.toggle("active", active);
  });
}

function syncStructuralFlowTimeline(points, latestResultId = null) {
  const g = state.geometry3d;
  const tf = g.temporalFlow || (g.temporalFlow = {});
  tf.points = Array.isArray(points) ? points : [];
  tf.latestResultId = latestResultId != null ? String(latestResultId) : null;
  if (!Number.isFinite(tf.localTimeSec) || tf.localTimeSec < 0) tf.localTimeSec = 0;
  if (state.runDetailView.flowPlaybackMode === "live") {
    tf.localTimeSec = Math.max(0, tf.points.length - 1);
  } else {
    tf.localTimeSec = Math.min(Math.max(0, tf.localTimeSec), Math.max(0, tf.points.length - 1));
  }
}

function renderRunDetailFromState(opts = {}) {
  const activeSection = String(state.runDetailView.section || "overview");
  const trendsHydrated = Boolean(state.ui.runDetailHydratedSections.trends);
  const resultsHydrated = Boolean(state.ui.runDetailHydratedSections.results);
  const geometryHydrated = Boolean(state.ui.runDetailHydratedSections.geometry);
  ["overview", "trends", "geometry", "results"].forEach((section) => {
    qs(`#analysis-${section}`)?.classList.toggle("hidden", activeSection !== section);
  });
  qsa("#runDetailSectionTabs [data-run-section]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-run-section") === activeSection);
  });
  qs("#runTrendsDeferredHint")?.classList.toggle("hidden", trendsHydrated);
  qs("#runResultsDeferredHint")?.classList.toggle("hidden", resultsHydrated);
  qs("#runGeometryDeferredHint")?.classList.toggle("hidden", geometryHydrated);
  setFlowModeButtonState();
  const hasResults = state.runRecent.length > 0;
  const latestResult = hasResults ? state.runRecent[0] : null;
  setConnectionStatus(getOperationalBadgeDisplay(buildFrontendUiState(latestResult)));
  renderRunDetailHeaderContext(state.activeRun, latestResult);
  const runDetailEmpty = qs("#runDetailEmpty");
  const geomPanel = qs(".geometry-panel");
  if (runDetailEmpty) {
    if (hasResults) runDetailEmpty.classList.add("hidden");
    else runDetailEmpty.classList.remove("hidden");
  }
  if (geomPanel) {
    geomPanel.classList.remove("hidden");
  }
  if (!hasResults) {
    destroyCharts();
    disposeGeometryRenderer();
    clearGeometryModelsPanel();
    state.runGeometry = null;
    renderRunResultsTable([]);
    return;
  }

  const chronologicalFull = state.runRecent.slice().reverse();
  const chronological = chronologicalFull;
  const ranged = currentRangeSlice(chronological);
  const flowTimeline = buildStructuralFlowTimeline(ranged);
  const latest = chronological.length ? chronological[chronological.length - 1] : state.runRecent[0];
  if (activeSection === "trends" && trendsHydrated) {
    renderRunDetailCharts(ranged);
  }
  syncStructuralFlowTimeline(flowTimeline, latest?.result_id);
  if (activeSection === "results" && resultsHydrated) {
    const filtered = filterRunResults(ranged);
    const sorted = sortRunResults(filtered);
    renderRunResultsTable(sorted, { latestResultId: latest?.result_id });
  } else if (activeSection === "results") {
    renderRunResultsTable([], { latestResultId: latest?.result_id });
  }
}


async function loadRunDetailBackgroundHistory(runId) {
  if (state.ui.runDetailBackgroundHistoryLoaded || state.ui.runDetailBackgroundHistoryPending) return;
  state.ui.runDetailBackgroundHistoryPending = true;
  try {
    const fullEnv = await fetchRecentResults({ run_id: runId, limit: RUN_DETAIL_BACKGROUND_LIMIT });
    const fullResults = Array.isArray(fullEnv?.results) ? fullEnv.results : [];
    if (fullResults.length > state.runRecent.length) {
      state.runRecent = fullResults;
      renderRunDetailFromState({ deferHeavy: true });
    }
    state.ui.runDetailBackgroundHistoryLoaded = true;
  } finally {
    state.ui.runDetailBackgroundHistoryPending = false;
  }
}

function setRunDetailEmptyMessage(primary, secondary) {
  const empty = qs("#runDetailEmpty");
  if (!empty) return;
  const lines = empty.querySelectorAll("p");
  if (lines[0]) lines[0].textContent = primary;
  if (lines[1]) lines[1].textContent = secondary;
}

function setRunDetailDemoProgress({ visible = false, phase = "Preparing demo telemetry", current = 0, total = 0, text = "" } = {}) {
  const panel = qs("#runDetailDemoProgress");
  const phaseEl = qs("#runDetailDemoProgressPhase");
  const countEl = qs("#runDetailDemoProgressCount");
  const fillEl = qs("#runDetailDemoProgressFill");
  const textEl = qs("#runDetailDemoProgressText");
  if (panel) panel.classList.toggle("hidden", !visible);
  if (phaseEl) phaseEl.textContent = phase;
  if (countEl) countEl.textContent = `${current}/${total}`;
  const pct = total > 0 ? Math.max(0, Math.min(100, (current / total) * 100)) : 0;
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (textEl) textEl.textContent = text || `${phase}… (${current}/${total})`;
}

function clearDemoJobIdParam() {
  try {
    const next = new URL(window.location.href);
    if (!next.searchParams.has("demo_job_id")) return;
    next.searchParams.delete("demo_job_id");
    window.history.replaceState({}, "", `${next.pathname}${next.search}${next.hash}`);
  } catch (_err) {
    // no-op
  }
}

function startGrowOpDemoMonitor(runId, runConfig = {}) {
  const params = new URLSearchParams(window.location.search);
  const jobId = String(params.get("demo_job_id") || "").trim();
  const isGrowOpRun = String(runConfig?.source || "").toLowerCase() === "grow-op-demo";
  if (!runId || (!jobId && !isGrowOpRun)) return;
  let attempt = 0;
  const maxAttempts = 180;
  setRunDetailEmptyMessage("Guided demo is loading telemetry now.", "No separate script is required. Keep this tab open.");
  setRunDetailDemoProgress({ visible: true, phase: "Connecting", current: 0, total: 0, text: "Requesting demo stream status…" });
  const poll = async () => {
    attempt += 1;
    try {
      const statusEnv = await fetchJson(
        apiUrl("/demo/grow-op/status", tenantScopeParams({ run_id: runId, ...(jobId ? { job_id: jobId } : {}) }))
      );
      const job = statusEnv?.job || {};
      const stateLabel = String(job?.status || statusEnv?.status || "running").toLowerCase();
      if (stateLabel === "error") {
        const msg = String(job?.error || "Grow-op demo seeding failed.");
        setStatus(msg, true, true);
        setRunDetailEmptyMessage("Guided demo failed to load telemetry.", "Open status for details, then retry launch guided demo.");
        setRunDetailDemoProgress({ visible: true, phase: "Failed", current: 0, total: 0, text: msg });
        return;
      }
      const processed = Math.max(0, Number(job?.processed || 0));
      const totalFrames = Math.max(processed, Number(job?.total_frames || 0));
      const progressPct = Number(job?.progress || 0);
      const inferredCurrent = totalFrames > 0 ? processed : Math.round(Math.max(0, Math.min(100, progressPct)));
      const inferredTotal = totalFrames > 0 ? totalFrames : 100;
      const phase = stateLabel === "complete" || stateLabel === "ready" ? "Finalizing" : "Streaming telemetry";
      setRunDetailDemoProgress({
        visible: true,
        phase,
        current: inferredCurrent,
        total: inferredTotal,
        text: `Telemetry ingest ${Math.max(0, Math.min(100, Math.round(progressPct)))}%`,
      });
      const recentEnv = await fetchRecentResults({ run_id: runId, limit: RUN_DETAIL_INITIAL_LIMIT });
      state.runRecent = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
      renderRunDetailFromState({ deferHeavy: true });
      if (state.runRecent.length > 0) {
        setStatus("Guided demo loaded.", false, true);
        clearDemoJobIdParam();
        setRunDetailDemoProgress({ visible: false });
        return;
      }
      if (stateLabel === "complete" || stateLabel === "ready") {
        setRunDetailEmptyMessage(
          "Guided demo is finalizing telemetry.",
          "No separate script is required. Keep this tab open while final frames are indexed."
        );
      }
      if (attempt < maxAttempts) {
        window.setTimeout(poll, 850);
      } else {
        setStatus("Guided demo is still processing. No separate script is needed—this page will update when telemetry arrives.", false, false);
        setRunDetailDemoProgress({ visible: true, phase: "Still processing", current: inferredCurrent, total: inferredTotal, text: "Still indexing telemetry frames…" });
      }
    } catch (_err) {
      if (attempt < maxAttempts) {
        window.setTimeout(poll, 1000);
      } else {
        setStatus("Guided demo status checks timed out. No separate script is required; try Refresh once and keep this tab open.", true, true);
        setRunDetailDemoProgress({ visible: true, phase: "Status timeout", current: 0, total: 0, text: "Could not read progress from server." });
      }
    }
  };
  window.setTimeout(poll, 600);
}

async function loadRunDetail(runId) {
  const runReq = fetchJson(apiUrl(`/runs/${encodeURIComponent(runId)}`, tenantScopeParams()));
  const recentReq = fetchRecentResults({ run_id: runId, limit: RUN_DETAIL_INITIAL_LIMIT });
  state.runRecent = [];
  renderRunDetailFromState();
  const [runRes, recentEnv] = await Promise.all([runReq, recentReq]);
  const run = runRes.run;
  const title = qs("#runDetailTitle");
  const meta = qs("#runDetailMeta");
  if (title) title.textContent = `Run analysis · ${run.name}`;
  if (meta) meta.textContent = `${run.run_id} · created ${run.created_at}`;
  state.runRecent = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
  collectKnownSites(state.runRecent);
  renderTenantControls();
  setRangeButtonState(state.runDetailView.range);
  state.runDetailView.section = "overview";
  state.runDetailView.runId = runId;
  state.ui.runDetailHydratedSections = { overview: true, trends: false, geometry: false, results: false };
  renderRunDetailFromState();
  state.ui.runDetailBackgroundHistoryLoaded = false;
  startGrowOpDemoMonitor(runId, run.config || {});

  const exportJson = qs("#runDetailExportJsonBtn");
  const exportCsv = qs("#runDetailExportCsvBtn");
  if (exportJson) exportJson.onclick = () => exportData("json", runId);
  if (exportCsv) exportCsv.onclick = () => exportData("csv", runId);
}

function wireRunDetailEvents() {
  if (qs("#analysis-executive")?.dataset.wired === "1") return;
  const marker = qs("#analysis-executive");
  if (marker) marker.dataset.wired = "1";
  const flowSpeedSelect = qs("#flowPlaybackSpeedSelect");
  if (flowSpeedSelect) flowSpeedSelect.value = String(state.runDetailView.flowPlaybackSpeed || 1);
  const flowHistoryToggle = qs("#flowHistoryToggle");
  if (flowHistoryToggle) flowHistoryToggle.checked = Boolean(state.runDetailView.flowHistoryEnabled);
  qsa("[data-flow-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.runDetailView.flowPlaybackMode = String(btn.getAttribute("data-flow-mode") || "live");
      const tf = state.geometry3d.temporalFlow || (state.geometry3d.temporalFlow = {});
      if (state.runDetailView.flowPlaybackMode === "replay") {
        tf.localTimeSec = 0;
        tf.historyCentroid = [];
        tf.historyDriftTip = [];
        tf.historyContact = [];
        tf.contactPersistence = 0;
      }
      setFlowModeButtonState();
    });
  });
  qs("#flowPlaybackSpeedSelect")?.addEventListener("change", (e) => {
    const next = Number.parseFloat(String(e.target?.value || "1"));
    state.runDetailView.flowPlaybackSpeed = Number.isFinite(next) && next > 0 ? next : 1;
  });
  qs("#flowHistoryToggle")?.addEventListener("change", (e) => {
    state.runDetailView.flowHistoryEnabled = Boolean(e.target?.checked);
  });
  qs("#runResultsSearchInput")?.addEventListener("input", (e) => {
    state.runDetailView.search = String(e.target.value || "");
    renderRunDetailFromState();
  });
  qs("#runResultsSortSelect")?.addEventListener("change", (e) => {
    state.runDetailView.sort = String(e.target.value || "timestamp_desc");
    renderRunDetailFromState();
  });
  qsa("#runRangeControls [data-range]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const range = String(btn.getAttribute("data-range") || "200");
      state.runDetailView.range = range;
      setRangeButtonState(range);
      renderRunDetailFromState();
    });
  });
  qsa("#runDetailSectionTabs [data-run-section]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const section = String(btn.getAttribute("data-run-section") || "overview");
      state.runDetailView.section = section;
      renderRunDetailFromState();
      try {
        await hydrateRunDetailSection(section, state.runDetailView.runId || state.activeRun?.run_id || "");
      } catch (err) {
        setStatus(String(err.message || err), true, true);
      }
    });
  });
}

async function loadResultDetail(resultId) {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run_id") || state.activeRun?.run_id || "";
  const env = await fetchJson(
    apiUrl(`/results/${encodeURIComponent(resultId)}`, tenantScopeParams({ run_id: runId }))
  );
  const r = env.result;
  const grid = qs("#resultDetailGrid");
  if (!grid) return;
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
  renderRiskExplanation(r, {
    panelSelector: "#resultRiskExplanationPanel",
    titleSelector: "#resultRiskExplanationTitle",
    bodySelector: "#resultRiskExplanationText",
    badgeSelector: "#resultRiskExplanationBadge",
  });
}
