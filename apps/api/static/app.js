const { qs, qsa, debounce, animateNumberText, friendlyErrorMessage, toPretty, formatBytes, escapeHtml } = window.NeraiumUI;
const { apiUrl, fetchJson } = window.NeraiumApi;
const { deriveFrontendState, getRunModeDisplay, getAnalysisStatusDisplay, getLastUpdateDisplay, getOperationalBadgeDisplay, getErrorDisplayContext } = window.NeraiumState;

async function fetchRecentResults(params) {
  const recentParams = tenantScopeParams({ ...(params || {}) });
  const cacheKey = JSON.stringify(recentParams);
  const now = Date.now();
  const cached = state?.ui?.recentResultsCache?.get(cacheKey);
  if (cached && now - cached.ts < 2000) {
    return cached.value;
  }
  if (state?.ui?.recentResultsInflight?.has(cacheKey)) {
    return state.ui.recentResultsInflight.get(cacheKey);
  }
  const request = fetchJson(apiUrl("/results/recent", recentParams)).then((env) => {
    const normalized =
      env && Array.isArray(env.results)
        ? env
        : env && Array.isArray(env.runs)
          ? { latest: null, count: env.runs.length, results: [] }
          : { latest: null, count: 0, results: [] };
    if (state?.ui?.recentResultsCache) {
      state.ui.recentResultsCache.set(cacheKey, { ts: Date.now(), value: normalized });
    }
    return normalized;
  }).finally(() => {
    state?.ui?.recentResultsInflight?.delete(cacheKey);
  });
  state?.ui?.recentResultsInflight?.set(cacheKey, request);
  const env = await request;
  if (env && Array.isArray(env.results)) return env;
  if (env && Array.isArray(env.runs)) return { latest: null, count: env.runs.length, results: [] };
  return { latest: null, count: 0, results: [] };
}





function buildFrontendUiState(latest = null, overrides = {}) {
  const route = getRoute();
  const page = String(route?.page || "dashboard").toLowerCase();
  const requestedMode = page === "validation" ? "validation" : "pilot";
  const alertStatus = state.dashboardCurrentAlertStatus || (latest && latest.alert_status) || null;
  const alertState = String(alertStatus?.state || alertStatus?.alert_state || "").toUpperCase();
  const alertActive = Boolean(alertStatus?.alert_active) || alertState === "ESCALATED" || alertState === "PENDING_ALERT";
  const replayUiState = state.demo?.replay?.uiState || "idle";
  return deriveFrontendState({
    page,
    routeMode: requestedMode,
    requestedMode,
    validationContext: Boolean(state.demo.enabled) && page === "validation",
    replayUiState,
    hasLatest: Boolean(latest),
    hasTelemetrySeries: Array.isArray(state.dashboardRecent) && state.dashboardRecent.length > 0,
    hasActiveRun: Boolean(state.activeRun?.run_id),
    alertActive,
    degradedRuntime: Boolean(state.runtimeDegraded),
    analysisInterrupted: Boolean(overrides.analysisInterrupted),
    latestTimestamp: latest?.timestamp || latest?.persisted_at || latest?.created_at || "",
  });
}
function phaseFromResult(r) {
  if (!r) return "-";
  return r.phase || r.state || r.interpreted_state || "-";
}

function trendFromResult(r) {
  if (!r) return "-";
  return r.trend || "-";
}

function stateTone(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("alert") || text.includes("unstable")) return "critical";
  if (text.includes("watch") || text.includes("shift") || text.includes("drift")) return "watch";
  if (text.includes("stable") || text.includes("nominal")) return "stable";
  return "unknown";
}

function transitionLabel(prevResult, nextResult) {
  if (!prevResult || !nextResult) return "Start";
  const prevState = String(prevResult.state || prevResult.interpreted_state || "-").toUpperCase();
  const nextState = String(nextResult.state || nextResult.interpreted_state || "-").toUpperCase();
  if (prevState !== nextState) return `${prevState} -> ${nextState}`;
  const prevRisk = normalizeRiskLevel(prevResult.risk_level);
  const nextRisk = normalizeRiskLevel(nextResult.risk_level);
  if (prevRisk !== nextRisk) return `Risk ${prevRisk} -> ${nextRisk}`;
  const prevTrend = String(trendFromResult(prevResult)).toUpperCase();
  const nextTrend = String(trendFromResult(nextResult)).toUpperCase();
  if (prevTrend !== nextTrend) return `Trend ${prevTrend} -> ${nextTrend}`;
  return "No major transition";
}

function transitionSeverity(prevResult, nextResult) {
  if (!prevResult || !nextResult) return "normal";
  const prevStateTone = stateTone(prevResult.state || prevResult.interpreted_state);
  const nextStateTone = stateTone(nextResult.state || nextResult.interpreted_state);
  const prevRisk = normalizeRiskLevel(prevResult.risk_level);
  const nextRisk = normalizeRiskLevel(nextResult.risk_level);
  if ((prevStateTone !== "critical" && nextStateTone === "critical") || (prevRisk !== "HIGH" && nextRisk === "HIGH")) {
    return "critical";
  }
  if (prevStateTone !== nextStateTone || prevRisk !== nextRisk) return "watch";
  return "normal";
}

function structuralDriftFromResult(r) {
  if (!r) return null;
  const v = r.structural_drift_score;
  return typeof v === "number" ? v : null;
}

function compositeInstabilityFromResult(r) {
  if (!r) return null;
  if (typeof r.latest_instability === "number" && Number.isFinite(r.latest_instability)) return r.latest_instability;
  if (typeof r.composite_instability === "number" && Number.isFinite(r.composite_instability)) {
    return r.composite_instability;
  }
  if (typeof r.instability === "number" && Number.isFinite(r.instability)) return r.instability;
  const analytics = r.experimental_analytics;
  if (analytics && typeof analytics.composite_instability === "number" && Number.isFinite(analytics.composite_instability)) {
    return analytics.composite_instability;
  }
  if (typeof r.system_health === "number" && Number.isFinite(r.system_health)) {
    return Math.max(0, Math.min(1, 1 - (r.system_health / 100)));
  }
  return null;
}

function summarizeRiskDrivers(result) {
  const risk = normalizeRiskLevel(result?.risk_level);
  const drift = structuralDriftFromResult(result);
  const instability = compositeInstabilityFromResult(result);
  const structuralAvailable = Boolean(result?.structural_analysis_available);
  const skippedReason = String(result?.skipped_reason || "").trim();
  const phase = String(phaseFromResult(result) || "-");
  const trend = String(trendFromResult(result) || "-");

  const hasDrift = typeof drift === "number" && Number.isFinite(drift);
  const hasInstability = typeof instability === "number" && Number.isFinite(instability);
  const highSignal = (hasDrift && drift >= 0.65) || (hasInstability && instability >= 0.65);
  const mediumSignal =
    (!highSignal && hasDrift && drift >= 0.35) || (!highSignal && hasInstability && instability >= 0.35);

  let summary = "Risk is currently unknown because core structural signals are incomplete.";
  if (risk === "HIGH") {
    summary = highSignal
      ? "Risk is HIGH because drift/instability indicate strong structural stress."
      : "Risk is HIGH due to unstable structural behavior and elevated state interpretation.";
  } else if (risk === "MEDIUM") {
    summary = mediumSignal
      ? "Risk is MEDIUM because drift/instability are elevated but not critical."
      : "Risk is MEDIUM due to watch-level structural movement.";
  } else if (risk === "LOW") {
    summary = "Risk is LOW because drift and instability remain within stable bounds.";
  }

  const driftText = hasDrift ? drift.toFixed(2) : "unavailable";
  const instabilityText = hasInstability ? instability.toFixed(2) : "unavailable";
  const structureText = structuralAvailable
    ? "Structural relationship analysis is available."
    : skippedReason
      ? `Structural relationship analysis is limited (${skippedReason}).`
      : "Structural relationship analysis is limited due to insufficient relationship signal.";
  return {
    risk,
    text: `${summary} Drift is ${driftText}, instability is ${instabilityText}, and phase is ${phase} (${trend} trend). ${structureText}`,
  };
}

function buildSeriesRiskInsight(latest, chronological) {
  if (!latest || !Array.isArray(chronological) || chronological.length < 4) return "";
  const drifts = chronological.map(structuralDriftFromResult).filter((x) => typeof x === "number" && Number.isFinite(x));
  const inst = chronological.map(compositeInstabilityFromResult).filter((x) => typeof x === "number" && Number.isFinite(x));
  const parts = [];
  if (drifts.length >= 4) {
    const hi = drifts.filter((d) => d >= 1.5).length;
    const early = drifts.slice(0, Math.ceil(drifts.length / 3));
    const late = drifts.slice(-Math.ceil(drifts.length / 3));
    const eAvg = early.reduce((a, b) => a + b, 0) / early.length;
    const lAvg = late.reduce((a, b) => a + b, 0) / late.length;
    if (hi >= 15) {
      parts.push(`Drift stayed above 1.5 on ${hi} consecutive snapshots — sustained stress drove the assessment.`);
    } else if (lAvg > eAvg + 0.12) {
      parts.push(`Structural drift trended upward (${eAvg.toFixed(2)} → ${lAvg.toFixed(2)}) across the visible window.`);
    }
  }
  if (inst.length >= 4) {
    const early = inst.slice(0, Math.ceil(inst.length / 3));
    const late = inst.slice(-Math.ceil(inst.length / 3));
    const eAvg = early.reduce((a, b) => a + b, 0) / early.length;
    const lAvg = late.reduce((a, b) => a + b, 0) / late.length;
    if (lAvg > eAvg + 0.15) {
      parts.push(`Composite instability accelerated late in the series (${eAvg.toFixed(2)} → ${lAvg.toFixed(2)}).`);
    }
  }
  return parts.length ? ` ${parts.join(" ")}` : "";
}

function renderRiskExplanation(result, opts = {}, chronological = null) {
  const titleEl = qs(opts.titleSelector || "#riskExplainTitle");
  const bodyEl = qs(opts.bodySelector || "#riskExplainBody");
  const panelEl = qs(opts.panelSelector || "#riskExplainPanel");
  const badgeEl = qs(opts.badgeSelector || "#riskExplainBadge");
  if (!titleEl || !bodyEl || !panelEl) return;

  if (!result) {
    panelEl.classList.remove("hidden");
    panelEl.setAttribute("data-risk", "UNKNOWN");
    titleEl.textContent = "Why this risk level";
    bodyEl.textContent = "No structural result available yet. Upload telemetry or refresh the active run to generate a risk explanation.";
    if (badgeEl) badgeEl.innerHTML = riskBadgeHtml("UNKNOWN");
    return;
  }

  const explanation = summarizeRiskDrivers(result);
  let bodyText = explanation.text;
  const uiTruth = buildFrontendUiState(result);
  if (uiTruth.mode === "validation" && Array.isArray(chronological) && chronological.length >= 3) {
    bodyText += buildSeriesRiskInsight(result, chronological);
  }
  panelEl.classList.remove("hidden");
  panelEl.setAttribute("data-risk", explanation.risk);
  titleEl.textContent = `Why risk is ${explanation.risk}`;
  bodyEl.textContent = bodyText;
  if (badgeEl) badgeEl.innerHTML = riskBadgeHtml(explanation.risk);
}

function parseTime(value) {
  const ms = Date.parse(String(value || ""));
  return Number.isFinite(ms) ? ms : 0;
}

function normalizeRiskLevel(value) {
  const risk = String(value || "UNKNOWN").trim().toUpperCase();
  if (risk === "LOW" || risk === "MEDIUM" || risk === "HIGH") {
    return risk;
  }
  return "UNKNOWN";
}

function riskRankNumber(value) {
  const r = normalizeRiskLevel(value);
  if (r === "HIGH") return 3;
  if (r === "MEDIUM") return 2;
  if (r === "LOW") return 1;
  return 0;
}

function healthScoreFromSignals(result) {
  if (!result) return null;
  const risk = normalizeRiskLevel(result.risk_level);
  const drift = structuralDriftFromResult(result);
  const inst = compositeInstabilityFromResult(result);
  let score = 90;
  if (risk === "HIGH") score -= 44;
  else if (risk === "MEDIUM") score -= 24;
  else if (risk === "UNKNOWN") score -= 10;
  if (typeof drift === "number" && Number.isFinite(drift)) {
    score -= Math.min(30, drift * 34);
  }
  if (typeof inst === "number" && Number.isFinite(inst)) {
    score -= Math.min(28, inst * 32);
  }
  return Math.max(0, Math.min(100, Math.round(score)));
}

function setHealthRingScore(score) {
  const arc = qs("#dashboardHealthArc");
  if (!arc) return;
  const safe = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  const r = 52;
  const c = 2 * Math.PI * r;
  arc.style.strokeDasharray = `${c}`;
  arc.style.strokeDashoffset = `${c * (1 - safe / 100)}`;
  arc.setAttribute("data-score", String(safe));
}


function exportData(format, runId) {
  const url = apiUrl("/results/export/download", tenantScopeParams({ format, run_id: runId || "", limit: 500 }));
  window.location.href = url;
}
