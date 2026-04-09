function clearDemoReplayPollTimer() {
  if (state.demo.replay.pollTimer) {
    window.clearTimeout(state.demo.replay.pollTimer);
    state.demo.replay.pollTimer = null;
  }
}

function setDemoUiState(nextState, reason = "") {
  const normalized = DEMO_UI_STATES[nextState] || DEMO_UI_STATES.idle;
  const prev = state.demo.replay.uiState || DEMO_UI_STATES.idle;
  if (prev === normalized) return;
  console.info("[demo] ui-state transition", {
    from: prev,
    to: normalized,
    run_id: state.demo.replay.runId || state.demo.seedRunId || state.activeRun?.run_id || "",
    reason: String(reason || ""),
  });
  state.demo.replay.uiState = normalized;
}

function normalizeReplayUiState({
  runStatus = "",
  hasTelemetry = false,
  explicitFailed = false,
  explicitCompleted = false,
} = {}) {
  if (explicitFailed) return DEMO_UI_STATES.failed;
  if (explicitCompleted) return DEMO_UI_STATES.completed;
  const normalizedRunStatus = String(runStatus || "").trim().toLowerCase();
  if (hasTelemetry) return DEMO_UI_STATES.running;
  if (!normalizedRunStatus) return DEMO_UI_STATES.starting;
  if (["active", "running", "live", "streaming", "monitoring"].includes(normalizedRunStatus)) return DEMO_UI_STATES.running;
  if (["starting", "pending", "queued", "initializing", "created", "open"].includes(normalizedRunStatus)) {
    return DEMO_UI_STATES.starting;
  }
  if (["offline", "no_data", "idle", "waiting", "paused"].includes(normalizedRunStatus)) return DEMO_UI_STATES.offline;
  if (["completed", "complete", "done", "finished"].includes(normalizedRunStatus)) return DEMO_UI_STATES.completed;
  if (["failed", "error", "aborted", "cancelled"].includes(normalizedRunStatus)) return DEMO_UI_STATES.failed;
  return DEMO_UI_STATES.offline;
}

function scheduleReplayStatusPoll(runId, delayMs) {
  clearDemoReplayPollTimer();
  state.demo.replay.pollTimer = window.setTimeout(() => {
    pollReplayStatus(runId).catch((err) => {
      console.warn("[demo] replay poll tick failed", { run_id: runId, error: String(err?.message || err) });
    });
  }, Math.max(120, Number(delayMs || DEMO_REPLAY_INITIAL_POLL_MS)));
}

async function getCmapssReplayStatus(runId) {
  return fetchJson(apiUrl("/demo/cmapss/status", tenantScopeParams({ run_id: runId })));
}

async function pollReplayStatus(runId) {
  const replayRunId = String(runId || state.demo.replay.runId || state.demo.seedRunId || "");
  if (!replayRunId) return;
  if (state.demo.replay.runId && state.demo.replay.runId !== replayRunId) return;
  const startedMs = Number(state.demo.replay.startingSinceMs || Date.now());
  try {
    const status = await getCmapssReplayStatus(replayRunId);
    const recentEnv = await fetchRecentResults({ run_id: replayRunId, limit: 5 }).catch(() => ({ results: [] }));
    const results = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
    const processedAtLaunch = Number(state.demo.replay.launchProcessed || 0);
    const processedByStatus = Number(status?.frames_processed || 0);
    const framesRequested = Number(status?.frames_requested || 0);
    const hasProcessedFrames = processedAtLaunch > 0 || processedByStatus > 0;
    const replayStatus = String(status?.status || "").toLowerCase();
    const runStatus = String(status?.run_status || "");
    state.demo.replay.pollFailures = 0;
    state.demo.replay.pollBackoffMs = DEMO_REPLAY_INITIAL_POLL_MS;
    if (replayStatus === "failed") {
      setDemoUiState(DEMO_UI_STATES.failed, `status=${replayStatus}`);
      state.demo.replay.errorMessage = String(status?.error_message || "Replay backend reported a failure.");
      setStatus(`Replay launch failed: ${state.demo.replay.errorMessage} Tap retry.`, true, true);
      return;
    }
    if (replayStatus === "ready" || results.length > 0) {
      setDemoUiState(DEMO_UI_STATES.running, `status=${replayStatus} results=${results.length}`);
      state.demo.replay.errorMessage = "";
      setStatus("");
      if (state.activeRun?.run_id === replayRunId && results.length > 0) {
        state.runRecent = results;
        renderRunDetailFromState();
      }
      return;
    }
    if (replayStatus === "empty") {
      setDemoUiState(DEMO_UI_STATES.completed, "replay-completed-empty");
      state.demo.replay.errorMessage = "Replay completed but generated no analysis outputs.";
      setStatus("Replay completed, but no analysis was materialized. Increase frames or verify runtime health.", true, true);
      return;
    }
    const elapsed = Date.now() - startedMs;
    if (replayStatus === "starting" || replayStatus === "ingesting" || hasProcessedFrames) {
      const progressText = hasProcessedFrames
        ? `Replay launch succeeded (${processedByStatus || processedAtLaunch}/${framesRequested || "?"} frames processed). Materializing analysis…`
        : "Replay launch acknowledged. Initializing NASA CMAPSS ingest…";
      setDemoUiState(DEMO_UI_STATES.starting, `status=${replayStatus || "pending"} run_status=${runStatus}`);
      setStatus(progressText, false, true);
      state.demo.replay.pollBackoffMs = Math.min(DEMO_REPLAY_MAX_POLL_MS, state.demo.replay.pollBackoffMs * 2);
      scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
      return;
    }
    if (elapsed >= DEMO_REPLAY_RESULTS_MATERIALIZATION_TIMEOUT_MS) {
      setDemoUiState(DEMO_UI_STATES.interrupted, "starting-timeout");
      state.demo.replay.errorMessage = "Replay launch succeeded, but analysis did not materialize before timeout.";
      setStatus(`${state.demo.replay.errorMessage} Verify runtime health and retry.`, true, true);
      return;
    }
    setDemoUiState(DEMO_UI_STATES.starting, `fallback-run-status=${runStatus || "-"}`);
    scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
  } catch (err) {
    const apiStatus = Number(err?.apiError?.status || 0);
    const body = err?.apiError?.body || {};
    const runtimeUnavailable = apiStatus === 503 && String(body?.type || "").includes("core_runtime_unavailable");
    const message = String(err?.message || err || "Replay status check failed");
    const notFound = message.includes("status=404");
    state.demo.replay.pollFailures += 1;
    if (runtimeUnavailable) {
      setDemoUiState(DEMO_UI_STATES.failed, "runtime-unavailable");
      state.demo.replay.errorMessage = String(body?.message || "Analysis engine is unavailable in degraded runtime mode.");
      setStatus(`${state.demo.replay.errorMessage} ${String(body?.actionable_detail || "")}`.trim(), true, true);
      return;
    }
    const transient = notFound || state.demo.replay.pollFailures <= DEMO_REPLAY_MAX_TRANSIENT_ERRORS;
    console.warn("[demo] replay poll error", {
      run_id: replayRunId,
      poll_failures: state.demo.replay.pollFailures,
      transient,
      error: message,
    });
    if (transient) {
      setDemoUiState(DEMO_UI_STATES.starting, notFound ? "run-not-yet-visible" : "transient-error");
      state.demo.replay.pollBackoffMs = Math.min(DEMO_REPLAY_MAX_POLL_MS, state.demo.replay.pollBackoffMs * 2);
      scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
      setStatus("Replay status temporarily unavailable. Retrying status check…", false, true);
      return;
    }
    setDemoUiState(DEMO_UI_STATES.interrupted, "persistent-poll-error");
    state.demo.replay.errorMessage = `Replay monitoring failed after repeated retries: ${message}`;
    setStatus(`${state.demo.replay.errorMessage} Tap retry.`, true, true);
  }
}

function beginReplayStatusMonitoring(runId) {
  const replayRunId = String(runId || "").trim();
  if (!replayRunId) return;
  clearDemoReplayPollTimer();
  state.demo.seedRunId = replayRunId;
  state.demo.replay.runId = replayRunId;
  state.demo.replay.pollFailures = 0;
  state.demo.replay.pollBackoffMs = DEMO_REPLAY_INITIAL_POLL_MS;
  state.demo.replay.startingSinceMs = Date.now();
  state.demo.replay.errorMessage = "";
  setDemoUiState(DEMO_UI_STATES.starting, "launch");
  console.info("[demo] replay monitoring started", { run_id: replayRunId });
  pollReplayStatus(replayRunId).catch((err) => {
    console.warn("[demo] first replay status fetch failed", { run_id: replayRunId, error: String(err?.message || err) });
  });
}



async function seedDemoData() {
  if (state.demo.replay.launchInFlight && state.demo.replay.launchPromise) {
    return state.demo.replay.launchPromise;
  }
  state.demo.replay.launchInFlight = true;
  state.demo.preparing = true;
  setDemoButtonsDisabled(true);
  renderTenantControls();
  const launchPromise = (async () => {
    setStatus("Preparing historical validation replay (secondary workflow)...", false);
    state.demo.replay.errorMessage = "";
    state.demo.replay.launchProcessed = 0;
    setDemoUiState(DEMO_UI_STATES.starting, "launch-begin");
    setDemoProgress({
      visible: true,
      phase: "Preparing historical validation replay",
      current: 0,
      total: 3,
      text: "Processing NASA CMAPSS dataset...",
    });
    setLoading(true, "Preparing historical validation replay...");
    setDemoProgress({
      visible: true,
      phase: "Processing NASA CMAPSS dataset",
      current: 1,
      total: 3,
      text: "Running NASA CMAPSS FD004 scenario...",
    });
    const out = await startCmapssDemo(customerIdValue(state.tenant.customerId), { max_frames: CMAPSS_REPLAY_DEFAULT_MAX_FRAMES });
    const resolvedRunId = String(out?.run_id || "");
    if (!resolvedRunId) throw new Error("NASA reference replay did not return a run ID.");
    state.demo.replay.launchProcessed = Number(out?.processed || 0);
    if (out?.launch_succeeded === false && state.demo.replay.launchProcessed <= 0) {
      throw new Error("Replay launch completed but produced zero processed frames. Verify runtime health and try a larger frame window.");
    }
    state.demo.seedRunId = resolvedRunId;
    state.demo.replay.runId = resolvedRunId;
    beginReplayStatusMonitoring(resolvedRunId);
    setLoading(true, "Running NASA CMAPSS FD004 scenario...");
    setStatus("Building structural state...", false);
    setDemoProgress({
      visible: true,
      phase: "Building structural state",
      current: 2,
      total: 3,
      text: "Predictive structural behavior is being calculated...",
    });
    await loadRuns();
    const resolvedRun = state.runs.find((r) => String(r.run_id || "") === resolvedRunId) || null;
    if (resolvedRun) updateActiveRunHeader(resolvedRun);
    setDemoProgress({ visible: true, phase: "Ready", current: 3, total: 3, text: "NASA CMAPSS FD004 run ready." });
    window.setTimeout(() => setDemoProgress({ visible: false }), 900);
    setStatus(`Reference replay ready: NASA CMAPSS FD004 (${Number(out?.processed || 0)} frames processed).`, false, true);
    return {
      count: Number(out?.processed || 0),
      processed: Number(out?.processed || 0),
      run_id: resolvedRunId,
    };
  })().catch((err) => {
    setDemoUiState(DEMO_UI_STATES.failed, "launch-failure");
    setDemoProgress({ visible: false });
    const apiStatus = Number(err?.apiError?.status || 0);
    const apiBody = err?.apiError?.body || {};
    if (apiStatus === 503 && String(apiBody?.type || "").includes("core_runtime_unavailable")) {
      throw new Error(`${String(apiBody?.message || "Analysis engine unavailable.")} ${String(apiBody?.actionable_detail || "")}`.trim());
    }
    throw new Error(String(err?.message || err || "Demo failed — retry"));
  }).finally(() => {
    state.demo.replay.launchInFlight = false;
    state.demo.replay.launchPromise = null;
    state.demo.preparing = false;
    setDemoButtonsDisabled(false);
    renderTenantControls();
  });
  state.demo.replay.launchPromise = launchPromise;
  return launchPromise;
}

function wireValidationEvents() {
  const btn = qs("#seedDemoBtn");
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", async () => {
    try {
      setLoading(true, "Preparing historical validation replay...");
      const out = await seedDemoData();
      const cid = encodeURIComponent(customerIdValue(state.tenant.customerId));
      window.location.href = `/app/runs/${encodeURIComponent(out.run_id)}?customer_id=${cid}&replay=1&from=validation`;
    } catch (err) {
      setStatus(String(err.message || err), true, true);
    } finally {
      setLoading(false);
    }
  });
  wireHistoricalReplayEvents();
}

async function handleValidationStartupBehavior({
  demoQuery = {},
  refreshCurrentPage: refreshPage,
} = {}) {
  wireValidationEvents();
  const shouldAutoPrepare = Boolean(demoQuery?.shouldAutoPrepare);
  if (!shouldAutoPrepare || state.demo.preparing || state.runs.length !== 0) {
    return false;
  }
  try {
    setLoading(true, "Preparing replay runs (shared link)…");
    await toggleDemoMode(true);
    const focusRun = await prepareDemoRuns({ mode: "all" });
    if (typeof refreshPage === "function") {
      await refreshPage();
    }
    if (focusRun?.run_id) {
      const cid = encodeURIComponent(customerIdValue(state.tenant.customerId));
      window.location.href = `/app/runs/${encodeURIComponent(focusRun.run_id)}?customer_id=${cid}&replay=1&autoplay=1&from=validation`;
      return true;
    }
    setStatus("Reference replay runs ready — pick a run from the list.", false, true);
    return true;
  } catch (err) {
    setStatus(String(err.message || err), true, true);
    return true;
  } finally {
    setLoading(false);
  }
}

async function loadValidationPage() {
  const demoQs = applyDemoQueryParams();
  readDemoModeFromStorage();
  applyDemoUiShell();
  renderTenantControls();
  await hydrateHistoricalReplayControls();
  await renderHistoricalReplayFromRun(state.activeRun?.run_id || "");
  await handleValidationStartupBehavior({
    demoQuery: demoQs,
    refreshCurrentPage,
  });
}

function setHistoricalReplayStatus(message, isError = false) {
  const el = qs("#historicalReplayStatus");
  if (!el) return;
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    el.classList.remove("status-error");
    return;
  }
  el.classList.remove("hidden");
  el.textContent = String(message);
  el.classList.toggle("status-error", Boolean(isError));
}

async function fetchHistoricalCsvOptions() {
  const out = await fetchJson(apiUrl("/demo/historical/csv-options", tenantScopeParams()));
  return Array.isArray(out?.sources) ? out.sources : [];
}

async function fetchHistoricalCsvSource(sourceKey) {
  return fetchJson(apiUrl("/demo/historical/csv-source", tenantScopeParams({ source_key: sourceKey })));
}

function selectedValidationRunId() {
  const sel = qs("#historicalReplayRunSelect");
  return String(sel?.value || "").trim();
}

function selectedHistoricalSourceKey() {
  const sel = qs("#historicalCsvSelect");
  return String(sel?.value || "").trim();
}

async function hydrateHistoricalReplayControls() {
  const runSel = qs("#historicalReplayRunSelect");
  if (runSel) {
    const runs = Array.isArray(state.runs) ? state.runs.slice() : [];
    runSel.innerHTML = runs
      .map((run) => {
        const runId = String(run.run_id || "");
        const selected = runId && runId === String(state.activeRun?.run_id || "") ? " selected" : "";
        return `<option value="${escapeHtml(runId)}"${selected}>${escapeHtml(run.name || runId)} · ${escapeHtml(runId)}</option>`;
      })
      .join("");
    if (!runSel.value && state.activeRun?.run_id) runSel.value = state.activeRun.run_id;
  }
  const sourceSel = qs("#historicalCsvSelect");
  if (sourceSel) {
    const sources = await fetchHistoricalCsvOptions().catch(() => []);
    if (sources.length === 0) {
      sourceSel.innerHTML = '<option value="">No bundled historical CSV sources found</option>';
    } else {
      sourceSel.innerHTML = sources
        .map((item, idx) => `<option value="${escapeHtml(item.key || "")}"${idx === 0 ? " selected" : ""}>${escapeHtml(item.label || item.key || "")}</option>`)
        .join("");
    }
  }
}

function updateHistoricalFileLabel(file) {
  const label = qs("#historicalCsvFileLabel");
  if (!label) return;
  if (!file) {
    label.textContent = "No file selected.";
    return;
  }
  label.textContent = `${file.name} (${formatBytes(Number(file.size || 0))})`;
}

function buildHistoricalReplayRunName() {
  const stamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  return `Historical Replay ${stamp} UTC`;
}

async function createHistoricalReplayRun() {
  const out = await fetchJson(apiUrl("/runs", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: buildHistoricalReplayRunName(),
      config: { source: "historical-replay-proof" },
      activate: true,
    }),
  });
  return out?.run || null;
}

function buildReplayFileFromSource(out) {
  const csvText = String(out?.csv_text || "");
  const fallbackName = `${String(out?.key || "historical")}.csv`;
  const filename = String(out?.path || fallbackName).split("/").slice(-1)[0] || fallbackName;
  return new File([csvText], filename, { type: "text/csv" });
}

function toReplayPoint(result) {
  const drift = structuralDriftFromResult(result);
  const composite = compositeInstabilityFromResult(result);
  const timestamp = String(result?.timestamp || result?.persisted_at || result?.created_at || "");
  return {
    timestamp,
    drift: typeof drift === "number" && Number.isFinite(drift) ? drift : null,
    composite: typeof composite === "number" && Number.isFinite(composite) ? composite : null,
    phase: String(phaseFromResult(result) || "-"),
    trend: String(trendFromResult(result) || "-"),
    risk: normalizeRiskLevel(result?.risk_level),
    state: String(result?.state || result?.interpreted_state || "-"),
    message: String(result?.operator_message || result?.message || "").trim(),
  };
}

function summarizeHistoricalReplay(results) {
  const chronological = (Array.isArray(results) ? results.slice() : []).sort((a, b) => parseTime(a.timestamp || a.persisted_at || a.created_at) - parseTime(b.timestamp || b.persisted_at || b.created_at));
  const points = chronological.map(toReplayPoint);
  const divergenceIdx = points.findIndex((p) => (typeof p.drift === "number" && p.drift >= 0.65) || p.risk === "HIGH" || p.risk === "MEDIUM");
  const divergencePoint = divergenceIdx >= 0 ? points[divergenceIdx] : null;
  const first = points[0] || null;
  const latest = points[points.length - 1] || null;
  const transitions = [];
  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const next = points[i];
    const transition = transitionLabel(
      { state: prev.state, interpreted_state: prev.state, risk_level: prev.risk, trend: prev.trend },
      { state: next.state, interpreted_state: next.state, risk_level: next.risk, trend: next.trend },
    );
    const driftJump =
      typeof next.drift === "number" && typeof prev.drift === "number" && Math.abs(next.drift - prev.drift) >= 0.25;
    if (transition !== "No major transition" || driftJump) {
      transitions.push({
        timestamp: next.timestamp,
        transition,
        drift: next.drift,
        severity: driftJump ? "watch" : transitionSeverity(
          { state: prev.state, interpreted_state: prev.state, risk_level: prev.risk, trend: prev.trend },
          { state: next.state, interpreted_state: next.state, risk_level: next.risk, trend: next.trend },
        ),
      });
    }
  }
  return {
    points,
    divergencePoint,
    transitions: transitions.slice(-8),
    first,
    latest,
  };
}

function renderHistoricalReplayResults(runId, results) {
  const root = qs("#historicalReplayResults");
  if (!root) return;
  const summary = summarizeHistoricalReplay(results);
  if (!summary.points.length) {
    root.classList.add("hidden");
    return;
  }
  root.classList.remove("hidden");
  const summaryGrid = qs("#historicalReplaySummary");
  const context = qs("#historicalContext");
  const driftList = qs("#historicalDriftTimeline");
  const transList = qs("#historicalTransitions");
  const explanationEl = qs("#historicalExplanation");
  const operatorEl = qs("#historicalOperatorInterpretation");
  const latest = summary.latest;
  const divergenceTime = summary.divergencePoint?.timestamp || "No material divergence detected in replay window.";
  const structuralShift = latest
    ? `${latest.phase} phase · ${latest.trend} trend · risk ${latest.risk}`
    : "No structural shift available.";
  const operatorNotice = summary.transitions[summary.transitions.length - 1]
    ? `Track transition "${summary.transitions[summary.transitions.length - 1].transition}" around ${summary.transitions[summary.transitions.length - 1].timestamp}.`
    : "Monitor drift and risk progression for first watch-level shift.";

  if (summaryGrid) {
    summaryGrid.innerHTML = `
      <article class="summary-item"><p>Material divergence began</p><strong>${escapeHtml(divergenceTime)}</strong></article>
      <article class="summary-item"><p>Structural change</p><strong>${escapeHtml(structuralShift)}</strong></article>
      <article class="summary-item"><p>Operator should notice</p><strong>${escapeHtml(operatorNotice)}</strong></article>
    `;
  }
  if (context) {
    context.innerHTML = `
      <article class="summary-item"><p>Run</p><strong>${escapeHtml(runId)}</strong></article>
      <article class="summary-item"><p>Asset</p><strong>${escapeHtml(String(latest?.state || "-"))}</strong></article>
      <article class="summary-item"><p>Window</p><strong>${escapeHtml(String(summary.first?.timestamp || "-"))} → ${escapeHtml(String(summary.latest?.timestamp || "-"))}</strong></article>
      <article class="summary-item"><p>Samples</p><strong>${escapeHtml(String(summary.points.length))}</strong></article>
    `;
  }
  if (driftList) {
    driftList.innerHTML = summary.points.slice(-24).map((point) => `<li><span>${escapeHtml(point.timestamp || "-")}</span><strong>Drift ${point.drift == null ? "-" : point.drift.toFixed(2)} · Composite ${point.composite == null ? "-" : point.composite.toFixed(2)} · Risk ${escapeHtml(point.risk)}</strong></li>`).join("");
  }
  if (transList) {
    if (!summary.transitions.length) {
      transList.innerHTML = "<li>No major transition recorded yet.</li>";
    } else {
      transList.innerHTML = summary.transitions.map((t) => `<li data-severity="${escapeHtml(t.severity)}"><span>${escapeHtml(t.timestamp || "-")}</span><strong>${escapeHtml(t.transition)}</strong></li>`).join("");
    }
  }
  if (explanationEl) {
    const message = latest?.message || "Replay complete. Structural changes are grounded in current run outputs.";
    explanationEl.textContent = message;
  }
  if (operatorEl) {
    operatorEl.textContent = `Latest interpretation: ${structuralShift}. ${operatorNotice}`;
  }
}

async function renderHistoricalReplayFromRun(runId) {
  const resolvedRun = String(runId || "").trim();
  if (!resolvedRun) return;
  const env = await fetchRecentResults({ run_id: resolvedRun, limit: 500 }).catch(() => ({ results: [] }));
  const results = Array.isArray(env?.results) ? env.results : [];
  renderHistoricalReplayResults(resolvedRun, results);
}

async function runHistoricalReplayFlow() {
  const fileInput = qs("#historicalCsvFileInput");
  const selectedFile = fileInput?.files?.[0] || null;
  const selectedSource = selectedHistoricalSourceKey();
  let file = selectedFile;
  if (!file && selectedSource) {
    const source = await fetchHistoricalCsvSource(selectedSource);
    file = buildReplayFileFromSource(source);
  }
  if (!file) throw new Error("Choose a CSV upload or load a bundled historical CSV source.");
  let runId = selectedValidationRunId();
  if (!runId) {
    const run = await createHistoricalReplayRun();
    runId = String(run?.run_id || "");
  }
  if (!runId) throw new Error("Unable to resolve replay run.");
  await runCsvPreviewForFile(file);
  const mapping = state.uploadCsv.mapping || null;
  const started = await uploadCsvFileWithProgress(file, runId, mapping);
  const jobId = String(started?.job_id || "");
  if (!jobId) throw new Error("Replay ingest did not return a job ID.");
  const job = await waitForIngestJob(jobId);
  await loadRuns();
  await hydrateHistoricalReplayControls();
  const runSel = qs("#historicalReplayRunSelect");
  if (runSel) runSel.value = runId;
  await renderHistoricalReplayFromRun(runId);
  return { runId, job };
}

function wireHistoricalReplayEvents() {
  const runCreateBtn = qs("#historicalReplayCreateRunBtn");
  const startBtn = qs("#historicalReplayStartBtn");
  const loadSourceBtn = qs("#loadHistoricalCsvBtn");
  const fileInput = qs("#historicalCsvFileInput");
  const runSel = qs("#historicalReplayRunSelect");

  if (runCreateBtn && runCreateBtn.dataset.wired !== "1") {
    runCreateBtn.dataset.wired = "1";
    runCreateBtn.addEventListener("click", async () => {
      try {
        setLoading(true, "Creating replay run...");
        const run = await createHistoricalReplayRun();
        await loadRuns();
        await hydrateHistoricalReplayControls();
        if (runSel && run?.run_id) runSel.value = run.run_id;
        setHistoricalReplayStatus(`Created replay run ${String(run?.run_id || "")}.`, false);
      } catch (err) {
        setHistoricalReplayStatus(String(err?.message || err), true);
      } finally {
        setLoading(false);
      }
    });
  }
  if (fileInput && fileInput.dataset.wired !== "1") {
    fileInput.dataset.wired = "1";
    fileInput.addEventListener("change", () => updateHistoricalFileLabel(fileInput.files?.[0] || null));
  }
  if (loadSourceBtn && loadSourceBtn.dataset.wired !== "1") {
    loadSourceBtn.dataset.wired = "1";
    loadSourceBtn.addEventListener("click", async () => {
      try {
        setLoading(true, "Loading bundled historical CSV...");
        const sourceKey = selectedHistoricalSourceKey();
        if (!sourceKey) throw new Error("Select a historical CSV source first.");
        const out = await fetchHistoricalCsvSource(sourceKey);
        const file = buildReplayFileFromSource(out);
        if (fileInput) {
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;
        }
        updateHistoricalFileLabel(file);
        setHistoricalReplayStatus(`Loaded ${String(out?.label || sourceKey)}.`, false);
      } catch (err) {
        setHistoricalReplayStatus(String(err?.message || err), true);
      } finally {
        setLoading(false);
      }
    });
  }
  if (startBtn && startBtn.dataset.wired !== "1") {
    startBtn.dataset.wired = "1";
    startBtn.addEventListener("click", async () => {
      try {
        setLoading(true, "Running historical replay...");
        setHistoricalReplayStatus("Starting historical replay ingest...", false);
        const out = await runHistoricalReplayFlow();
        const status = String(out?.job?.status || "");
        const processed = Number(out?.job?.rows_processed || 0);
        setHistoricalReplayStatus(`Replay ${status || "completed"} for ${out.runId} (${processed} rows processed).`, status === "failed");
      } catch (err) {
        setHistoricalReplayStatus(String(err?.message || err), true);
      } finally {
        setLoading(false);
      }
    });
  }
  if (runSel && runSel.dataset.wired !== "1") {
    runSel.dataset.wired = "1";
    runSel.addEventListener("change", async () => {
      await renderHistoricalReplayFromRun(selectedValidationRunId());
    });
  }
}
