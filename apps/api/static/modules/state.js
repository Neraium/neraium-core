(function attachStateModule(globalObj) {
  const STATUS = Object.freeze({
    NO_DATA_INGESTED: "NO DATA INGESTED",
    WAITING_FOR_TELEMETRY: "WAITING FOR TELEMETRY",
    ACTIVE_MONITORING: "ACTIVE MONITORING",
    ALERT_ACTIVE: "ALERT ACTIVE",
    VALIDATION_READY: "VALIDATION READY",
    HISTORICAL_VALIDATION: "HISTORICAL VALIDATION",
    REPLAY_INTERRUPTED: "REPLAY INTERRUPTED",
    ANALYSIS_INTERRUPTED: "ANALYSIS INTERRUPTED",
  });

  function normalizeMode(value) {
    const v = String(value || "").toLowerCase();
    if (v === "validation" || v === "pilot") return v;
    return "";
  }

  function deriveMode(input = {}) {
    const page = String(input.page || "").toLowerCase();
    const backendMode = normalizeMode(input.backendMode);
    const routeMode = normalizeMode(input.routeMode);
    const requestedMode = normalizeMode(input.requestedMode);
    if (backendMode) return backendMode;
    if (routeMode) return routeMode;
    if (requestedMode) return requestedMode;
    if (page === "validation") return "validation";
    if (page === "dashboard" || page === "upload" || page === "runs" || page === "run-detail" || page === "result-detail") {
      return "pilot";
    }
    return Boolean(input.validationContext) ? "validation" : "pilot";
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
    const mode = deriveMode(input);
    const replayMode = mode === "validation";
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
      if (replayActive || hasTelemetry) statusKey = "historical_validation";
      else statusKey = "validation_ready";
    } else if (alertActive) statusKey = "alert_active";
    else if (hasTelemetry) statusKey = "active_monitoring";
    else if (hasActiveRun) statusKey = "waiting_for_telemetry";

    return {
      page,
      mode,
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
    if (!state) return STATUS.NO_DATA_INGESTED;
    switch (state.statusKey) {
      case "analysis_interrupted":
        return STATUS.ANALYSIS_INTERRUPTED;
      case "replay_interrupted":
        return STATUS.REPLAY_INTERRUPTED;
      case "historical_validation":
        return STATUS.HISTORICAL_VALIDATION;
      case "validation_ready":
        return STATUS.VALIDATION_READY;
      case "alert_active":
        return STATUS.ALERT_ACTIVE;
      case "active_monitoring":
        return STATUS.ACTIVE_MONITORING;
      case "waiting_for_telemetry":
        return STATUS.WAITING_FOR_TELEMETRY;
      default:
        return STATUS.NO_DATA_INGESTED;
    }
  }

  function getOperationalBadgeDisplay(state) {
    return getAnalysisStatusDisplay(state);
  }

  function getLastUpdateDisplay(state, latestTimestamp, nowMs = Date.now()) {
    const tsMs = toTimestampMs(latestTimestamp) || Number(state?.latestTimestampMs || 0);
    if (!state) return "No data yet";
    if (!tsMs) {
      return state.mode === "validation"
        ? "No historical validation timestamp yet."
        : "No telemetry timestamp yet.";
    }
    if (state.mode === "validation") return `Validation frame time: ${String(latestTimestamp)}`;
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
    STATUS,
    deriveFrontendState,
    getRunModeDisplay,
    getAnalysisStatusDisplay,
    getLastUpdateDisplay,
    getOperationalBadgeDisplay,
    getErrorDisplayContext,
  };
})(window);
