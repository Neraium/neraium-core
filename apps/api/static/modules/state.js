(function attachStateModule(globalObj) {
  function isReplayMode(page, demoEnabled) {
    const p = String(page || "").toLowerCase();
    return p === "validation" || Boolean(demoEnabled);
  }

  function normalizeReplayUiState(value) {
    const raw = String(value || "").toLowerCase();
    if (["starting", "running", "offline", "interrupted", "failed", "completed"].includes(raw)) return raw;
    return "idle";
  }

  function toTimestampMs(value) {
    const ms = Date.parse(String(value || ""));
    return Number.isFinite(ms) ? ms : 0;
  }

  function deriveFrontendState(input = {}) {
    const page = String(input.page || "dashboard").toLowerCase();
    const replayMode = isReplayMode(page, input.demoEnabled);
    const replayUiState = normalizeReplayUiState(input.replayUiState);
    const hasLatest = Boolean(input.hasLatest);
    const hasTelemetrySeries = Boolean(input.hasTelemetrySeries);
    const hasTelemetry = hasLatest || hasTelemetrySeries;
    const hasActiveRun = Boolean(input.hasActiveRun);
    const alertActive = Boolean(input.alertActive);
    const degradedRuntime = Boolean(input.degradedRuntime);
    const analysisInterrupted = Boolean(input.analysisInterrupted);
    const replayInterrupted = replayMode && (replayUiState === "interrupted" || replayUiState === "failed");
    const replayActive = replayMode && (replayUiState === "starting" || replayUiState === "running");

    let statusKey = "no_data";
    if (analysisInterrupted) statusKey = "analysis_interrupted";
    else if (replayInterrupted) statusKey = "replay_interrupted";
    else if (replayMode) {
      if (replayActive || hasTelemetry) statusKey = "replay_active";
      else statusKey = "validation_ready";
    } else if (alertActive) statusKey = "alert_active";
    else if (hasTelemetry) statusKey = "active_monitoring";
    else if (hasActiveRun) statusKey = "waiting_for_telemetry";

    return {
      page,
      mode: replayMode ? "validation" : "pilot",
      replayMode,
      hasTelemetry,
      hasActiveRun,
      replayActive,
      replayInterrupted,
      analysisInterrupted,
      alertActive,
      degradedRuntime,
      statusKey,
      latestTimestampMs: toTimestampMs(input.latestTimestamp),
    };
  }

  function getRunModeDisplay(state) {
    if (!state || state.mode === "validation") {
      return "Validation reference workflow";
    }
    return "Pilot telemetry monitoring";
  }

  function getAnalysisStatusDisplay(state) {
    if (!state) return "No data";
    switch (state.statusKey) {
      case "analysis_interrupted":
        return "Analysis interrupted";
      case "replay_interrupted":
        return "Replay interrupted";
      case "replay_active":
        return "Historical replay active";
      case "validation_ready":
        return "Ready for historical replay";
      case "alert_active":
        return "Alert active";
      case "active_monitoring":
        return "Live monitoring active";
      case "waiting_for_telemetry":
        return "Waiting for telemetry";
      default:
        return "No telemetry in active run";
    }
  }

  function getOperationalBadgeDisplay(state) {
    if (!state) return "NO DATA INGESTED";
    switch (state.statusKey) {
      case "analysis_interrupted":
        return "ANALYSIS INTERRUPTED";
      case "replay_interrupted":
        return "REPLAY INTERRUPTED";
      case "replay_active":
        return "HISTORICAL VALIDATION";
      case "validation_ready":
        return "VALIDATION RUN";
      case "alert_active":
        return "ALERT ACTIVE";
      case "active_monitoring":
        return "ACTIVE MONITORING";
      case "waiting_for_telemetry":
        return "WAITING FOR TELEMETRY";
      default:
        return "NO DATA INGESTED";
    }
  }

  function getLastUpdateDisplay(state, latestTimestamp, nowMs = Date.now()) {
    const tsMs = toTimestampMs(latestTimestamp) || Number(state?.latestTimestampMs || 0);
    if (!state) return "No data yet";
    if (!tsMs) {
      return state.mode === "validation"
        ? "No replay frame timestamp yet"
        : "No telemetry timestamp yet for this active run.";
    }
    if (state.mode === "validation") return String(latestTimestamp);
    const ageMs = Math.max(0, nowMs - tsMs);
    const mins = Math.floor(ageMs / 60000);
    if (mins < 2) return "Fresh telemetry (updated just now)";
    if (mins < 60) return `Freshness: ${mins} minute${mins === 1 ? "" : "s"} ago`;
    const hours = Math.floor(mins / 60);
    return `Stale telemetry — last upload ${hours} hour${hours === 1 ? "" : "s"} ago`;
  }

  function getErrorDisplayContext(state, message = "") {
    const text = String(message || "").toLowerCase();
    if (text.includes("geometry") || text.includes("structural view")) return "geometry";
    if (text.includes("ingest") || text.includes("upload") || text.includes("csv")) return "ingest";
    if (state?.mode === "validation" || text.includes("replay") || text.includes("demo") || text.includes("validation")) {
      return "replay";
    }
    return "analysis";
  }

  globalObj.NeraiumState = {
    deriveFrontendState,
    getRunModeDisplay,
    getAnalysisStatusDisplay,
    getLastUpdateDisplay,
    getOperationalBadgeDisplay,
    getErrorDisplayContext,
  };
})(window);
