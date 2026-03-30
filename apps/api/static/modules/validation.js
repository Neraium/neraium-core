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

async function pollReplayStatus(runId) {
  const replayRunId = String(runId || state.demo.replay.runId || state.demo.seedRunId || "");
  if (!replayRunId) return;
  if (state.demo.replay.runId && state.demo.replay.runId !== replayRunId) return;
  const startedMs = Number(state.demo.replay.startingSinceMs || Date.now());
  try {
    const runRes = await fetchJson(apiUrl(`/runs/${encodeURIComponent(replayRunId)}`, tenantScopeParams()));
    const run = runRes?.run || null;
    const recentEnv = await fetchRecentResults({ run_id: replayRunId, limit: 5 });
    const results = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
    const processedAtLaunch = Number(state.demo.replay.launchProcessed || 0);
    const hasProcessedFrames = processedAtLaunch > 0;
    const uiState = normalizeReplayUiState({
      runStatus: String(run?.status || ""),
      hasTelemetry: results.length > 0 || hasProcessedFrames,
    });
    state.demo.replay.pollFailures = 0;
    state.demo.replay.pollBackoffMs = DEMO_REPLAY_INITIAL_POLL_MS;
    setDemoUiState(uiState, `run.status=${String(run?.status || "-")} results=${results.length}`);
    if (uiState === DEMO_UI_STATES.running) {
      state.demo.replay.errorMessage = "";
      setStatus("");
      if (state.activeRun?.run_id === replayRunId && results.length > 0) {
        state.runRecent = results;
        renderRunDetailFromState();
      }
      return;
    }
    const elapsed = Date.now() - startedMs;
    if (uiState === DEMO_UI_STATES.starting && elapsed < DEMO_REPLAY_STARTING_TIMEOUT_MS) {
      scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
      return;
    }
    if (uiState === DEMO_UI_STATES.offline) {
      state.demo.replay.pollBackoffMs = Math.min(DEMO_REPLAY_MAX_POLL_MS, state.demo.replay.pollBackoffMs * 2);
      scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
      return;
    }
    if (uiState === DEMO_UI_STATES.starting) {
      if (hasProcessedFrames) {
        setDemoUiState(DEMO_UI_STATES.offline, "starting-timeout-processed-launch");
        state.demo.replay.pollBackoffMs = Math.min(DEMO_REPLAY_MAX_POLL_MS, state.demo.replay.pollBackoffMs * 2);
        scheduleReplayStatusPoll(replayRunId, state.demo.replay.pollBackoffMs);
        return;
      }
      setDemoUiState(DEMO_UI_STATES.interrupted, "starting-timeout");
      state.demo.replay.errorMessage = "Replay launch timed out while waiting for live telemetry.";
      setStatus(`${state.demo.replay.errorMessage} Tap retry.`, true, true);
      return;
    }
    if (uiState === DEMO_UI_STATES.failed) {
      state.demo.replay.errorMessage = "Replay backend reported a failed run state.";
      setStatus(`${state.demo.replay.errorMessage} Tap retry.`, true, true);
      return;
    }
  } catch (err) {
    const message = String(err?.message || err || "Replay status check failed");
    const notFound = message.includes("status=404");
    state.demo.replay.pollFailures += 1;
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
      return;
    }
    setDemoUiState(DEMO_UI_STATES.interrupted, "persistent-poll-error");
    state.demo.replay.errorMessage = `Replay monitoring interrupted: ${message}`;
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
    const out = await startCmapssDemo(customerIdValue(state.tenant.customerId), { max_frames: 120 });
    const resolvedRunId = String(out?.run_id || "");
    if (!resolvedRunId) throw new Error("NASA reference replay did not return a run ID.");
    state.demo.replay.launchProcessed = Number(out?.processed || 0);
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
  await handleValidationStartupBehavior({
    demoQuery: demoQs,
    refreshCurrentPage,
  });
}
