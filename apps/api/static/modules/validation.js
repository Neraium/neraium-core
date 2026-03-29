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
    const uiState = normalizeReplayUiState({
      runStatus: String(run?.status || ""),
      hasTelemetry: results.length > 0,
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


