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

function scheduleDeferredRunDetailPaint() {
  if (state.ui.runDetailDeferredPaint) {
    window.cancelAnimationFrame(state.ui.runDetailDeferredPaint);
    state.ui.runDetailDeferredPaint = null;
  }
  state.ui.runDetailDeferredPaint = window.requestAnimationFrame(() => {
    renderRunDetailFromState({ deferHeavy: false });
    state.ui.runDetailDeferredPaint = null;
  });
}

function clearRunDetailObserver() {
  if (state.ui.runDetailObserver) {
    state.ui.runDetailObserver.disconnect();
    state.ui.runDetailObserver = null;
  }
}

function setupRunDetailProgressiveHydration(runId) {
  clearRunDetailObserver();
  state.ui.runDetailHydratedSections = {};
  if (!("IntersectionObserver" in window)) {
    scheduleDeferredRunDetailPaint();
    scheduleHeavyWork(() => {
      loadRunGeometry(runId, state.runRecent[0]?.result_id ?? null).catch(() => {
        setGeometrySurfaceState("Structural view is unavailable for the current snapshot.", "error");
      });
    });
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const id = String(entry.target.id || "");
      if (id === "analysis-trends" && !state.ui.runDetailHydratedSections.trends) {
        state.ui.runDetailHydratedSections.trends = true;
        ensureChartJsLoaded()
          .then(() => scheduleDeferredRunDetailPaint())
          .catch((err) => setStatus(String(err.message || err), true, true));
      }
      if (id === "analysis-results" && !state.ui.runDetailHydratedSections.results) {
        state.ui.runDetailHydratedSections.results = true;
        scheduleDeferredRunDetailPaint();
        scheduleHeavyWork(() => {
          loadRunDetailBackgroundHistory(runId).catch(() => {
            /* best effort */
          });
        });
      }
      if (id === "analysis-geometry" && !state.ui.runDetailHydratedSections.geometry) {
        state.ui.runDetailHydratedSections.geometry = true;
        const resultId = state.runRecent[0]?.result_id ?? null;
        scheduleHeavyWork(() => {
          loadRunGeometry(runId, resultId).catch(() => {
            setGeometrySurfaceState("Structural view is unavailable for the current snapshot.", "error");
          });
        });
      }
    });
  }, { rootMargin: window.matchMedia("(max-width: 740px)").matches ? "80px 0px" : "180px 0px" });
  state.ui.runDetailObserver = observer;
  ["analysis-trends", "analysis-results", "analysis-geometry"].forEach((id) => {
    const el = qs(`#${id}`);
    if (el) observer.observe(el);
  });
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

function renderRunCurrentStateGauges(latest) {
  const card = qs("#runCurrentStateCard");
  if (!card) return;
  if (!latest) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  const drift = structuralDriftFromResult(latest);
  const inst = compositeInstabilityFromResult(latest);
  const risk = normalizeRiskLevel(latest.risk_level);
  const health = healthScoreFromSignals(latest);
  const driftFill = qs("#gaugeDriftFill");
  const instFill = qs("#gaugeInstFill");
  const riskFill = qs("#gaugeRiskFill");
  const healthFill = qs("#gaugeHealthFill");
  const toPct = (val, cap) => {
    if (typeof val !== "number" || !Number.isFinite(val)) return 0;
    return Math.max(0, Math.min(100, (val / cap) * 100));
  };
  if (driftFill) driftFill.style.width = `${toPct(drift, 2.5)}%`;
  if (instFill) instFill.style.width = `${toPct(inst, 2.5)}%`;
  if (riskFill) {
    const rp = risk === "HIGH" ? 92 : risk === "MEDIUM" ? 58 : risk === "LOW" ? 24 : 38;
    riskFill.style.width = `${rp}%`;
  }
  if (healthFill) healthFill.style.width = `${typeof health === "number" ? Math.max(0, Math.min(100, health)) : 0}%`;
  const dv = qs("#gaugeDriftVal");
  const iv = qs("#gaugeInstVal");
  const rv = qs("#gaugeRiskVal");
  const hv = qs("#gaugeHealthVal");
  if (dv) dv.textContent = typeof drift === "number" ? drift.toFixed(2) : "—";
  if (iv) iv.textContent = typeof inst === "number" ? inst.toFixed(2) : "—";
  if (rv) rv.textContent = risk;
  if (hv) hv.textContent = typeof health === "number" ? String(health) : "—";
  card.setAttribute("data-risk", risk);
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
  const deferHeavy = Boolean(opts.deferHeavy);
  const trendsHydrated = !deferHeavy || Boolean(state.ui.runDetailHydratedSections.trends);
  const resultsHydrated = !deferHeavy || Boolean(state.ui.runDetailHydratedSections.results);
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
    setDemoPlaybackUI();
    renderRunResultsTable([]);
    renderRiskExplanation(null, {
      panelSelector: "#runRiskExplanationPanel",
      titleSelector: "#runRiskExplanationTitle",
      bodySelector: "#runRiskExplanationText",
      badgeSelector: "#runRiskExplanationBadge",
    });
    return;
  }

  const chronologicalFull = state.runRecent.slice().reverse();
  const activeRunId = String(state.activeRun?.run_id || "");
  if (state.demo.enabled) {
    if (state.demo.activeRunId !== activeRunId) {
      stopDemoPlayback();
      state.demo.activeRunId = activeRunId;
      state.demo.cursor = chronologicalFull.length;
    }
    if (!Number.isFinite(state.demo.cursor) || state.demo.cursor <= 0) {
      state.demo.cursor = chronologicalFull.length;
    }
    state.demo.cursor = Math.max(1, Math.min(state.demo.cursor, chronologicalFull.length));
    state.demo.keyEvents = extractDemoKeyEvents(chronologicalFull);
  } else {
    state.demo.activeRunId = activeRunId;
    state.demo.cursor = chronologicalFull.length;
    state.demo.keyEvents = [];
  }
  const chronological = state.demo.enabled
    ? chronologicalFull.slice(0, Math.max(1, Number(state.demo.cursor || chronologicalFull.length)))
    : chronologicalFull;
  const ranged = currentRangeSlice(chronological);
  const flowTimeline = buildStructuralFlowTimeline(ranged);
  const latest = chronological.length ? chronological[chronological.length - 1] : state.runRecent[0];
  setDemoPlaybackUI();
  if (trendsHydrated) {
    renderRunDetailCharts(ranged);
  }
  syncStructuralFlowTimeline(flowTimeline, latest?.result_id);
  if (resultsHydrated) {
    const filtered = filterRunResults(ranged);
    const sorted = sortRunResults(filtered);
    renderRunResultsTable(sorted, { latestResultId: latest?.result_id });
  } else {
    renderRunResultsTable([], { latestResultId: latest?.result_id });
  }
  renderRiskExplanation(latest, {
    panelSelector: "#runRiskExplanationPanel",
    titleSelector: "#runRiskExplanationTitle",
    bodySelector: "#runRiskExplanationText",
    badgeSelector: "#runRiskExplanationBadge",
  }, chronologicalFull);
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

async function loadRunDetail(runId) {
  if (runId) {
    state.demo.seedRunId = String(runId);
    beginReplayStatusMonitoring(runId);
  }
  const runRes = await fetchJson(apiUrl(`/runs/${encodeURIComponent(runId)}`, tenantScopeParams()));
  const run = runRes.run;
  const title = qs("#runDetailTitle");
  const meta = qs("#runDetailMeta");
  if (title) title.textContent = `Run analysis · ${run.name}`;
  const runTruth = buildFrontendUiState(null);
  const workspaceLabel = runTruth.mode === "validation" ? "validation replay workspace" : "pilot telemetry workspace";
  if (meta) meta.textContent = `Run ${run.run_id} · ${workspaceLabel} · created ${run.created_at}`;
  const recentEnv = await fetchRecentResults({ run_id: runId, limit: RUN_DETAIL_INITIAL_LIMIT });
  state.runRecent = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
  if (state.demo.enabled) {
    if (state.demo.activeRunId !== runId) {
      stopDemoPlayback();
      state.demo.activeRunId = runId;
      state.demo.cursor = state.runRecent.length;
    } else if (!Number.isFinite(state.demo.cursor) || state.demo.cursor <= 0 || state.demo.cursor > state.runRecent.length) {
      state.demo.cursor = state.runRecent.length;
    }
    state.demo.keyEvents = extractDemoKeyEvents(state.runRecent.slice().reverse());
  } else {
    state.demo.activeRunId = runId;
    state.demo.cursor = state.runRecent.length;
    state.demo.keyEvents = [];
  }
  collectKnownSites(state.runRecent);
  renderTenantControls();
  setRangeButtonState(state.runDetailView.range);
  renderRunDetailFromState({ deferHeavy: true });
  setupRunDetailProgressiveHydration(runId);
  let autoplayHandled = false;
  try {
    if (new URLSearchParams(window.location.search).get("autoplay") === "1" && state.demo.enabled && state.runRecent.length > 1) {
      replayDemoTimeline();
      autoplayHandled = true;
      const u = new URL(window.location.href);
      u.searchParams.delete("autoplay");
      window.history.replaceState({}, "", u.pathname + u.search + u.hash);
    }
  } catch (_e) {
    // ignore
  }
  if (!autoplayHandled) {
    maybeAutoStartDemoPlayback();
  }
  state.ui.runDetailBackgroundHistoryLoaded = false;

  const exportJson = qs("#runDetailExportJsonBtn");
  const exportCsv = qs("#runDetailExportCsvBtn");
  if (exportJson) exportJson.onclick = () => exportData("json", runId);
  if (exportCsv) exportCsv.onclick = () => exportData("csv", runId);
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
