const { qs, qsa, debounce, animateNumberText, friendlyErrorMessage, toPretty, formatBytes, escapeHtml } = window.NeraiumUI;
const { apiUrl, fetchJson } = window.NeraiumApi;
const { deriveFrontendState, getRunModeDisplay, getAnalysisStatusDisplay, getLastUpdateDisplay, getOperationalBadgeDisplay, getErrorDisplayContext } = window.NeraiumState;

async function fetchRecentResults(params) {
  const recentParams = tenantScopeParams({ ...(params || {}), compact: 1 });
  const cacheKey = JSON.stringify(recentParams);
  const now = Date.now();
  const cached = state?.ui?.recentResultsCache?.get(cacheKey);
  if (cached && now - cached.ts < 2000) {
    return cached.value;
  }
  if (state?.ui?.recentResultsInflight?.has(cacheKey)) {
    return state.ui.recentResultsInflight.get(cacheKey);
  }
  const request = fetchJson(apiUrl("/runs", recentParams)).then((env) => {
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
  if (typeof r.latest_instability === "number") return r.latest_instability;
  const analytics = r.experimental_analytics;
  if (analytics && typeof analytics.composite_instability === "number") {
    return analytics.composite_instability;
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

function commandStateFromRisk(value) {
  const risk = normalizeRiskLevel(value);
  if (risk === "HIGH") return "INTERVENE";
  if (risk === "MEDIUM") return "WATCH";
  return "SYSTEM NORMAL";
}

function zoneToneFromRisk(value) {
  const risk = normalizeRiskLevel(value);
  if (risk === "HIGH") return "intervene";
  if (risk === "MEDIUM") return "watch";
  return "normal";
}

function titleCaseFromSnake(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function resolveZoneName(row, index) {
  return titleCaseFromSnake(row?.name || row?.zone_name || row?.zone_id || row?.asset_id || `Zone ${index + 1}`);
}

function resolveZoneRisk(row, fallbackRisk) {
  return normalizeRiskLevel(row?.risk_level || row?.risk || row?.status || fallbackRisk || "LOW");
}

function applySmoothedTone(key, observedTone) {
  state.ui.zoneToneMemory = state.ui.zoneToneMemory || {};
  const now = Date.now();
  const entry = state.ui.zoneToneMemory[key] || { current: observedTone, candidate: observedTone, since: now };
  if (observedTone !== entry.candidate) {
    entry.candidate = observedTone;
    entry.since = now;
  }
  if (entry.current !== entry.candidate && now - entry.since >= 2500) {
    entry.current = entry.candidate;
  }
  state.ui.zoneToneMemory[key] = entry;
  return entry.current;
}

function renderZoneGrid(zones = []) {
  const zoneGrid = qs("#zoneGrid");
  if (!zoneGrid) return;
  zoneGrid.innerHTML = "";
  zones.forEach((zone, index) => {
    const zoneEl = document.createElement("article");
    zoneEl.className = "operator-zone-box";
    const smoothedTone = applySmoothedTone(resolveZoneName(zone, index), zoneToneFromRisk(zone.risk));
    zoneEl.setAttribute("data-tone", smoothedTone);
    zoneEl.setAttribute("role", "listitem");
    zoneEl.textContent = zone.name;
    zoneGrid.appendChild(zoneEl);
  });
}

function renderCommandLayer(latest, zones) {
  const commandEl = qs("#commandStatusText");
  const eventPanel = qs("#eventPanel");
  const eventHeadline = qs("#eventHeadline");
  const eventIssue = qs("#eventIssue");
  const eventLocation = qs("#eventLocation");
  const eventProgression = qs("#eventProgression");
  const topZone = zones.find((z) => z.risk === "HIGH") || zones.find((z) => z.risk === "MEDIUM") || zones[0] || { name: "ZONE", risk: normalizeRiskLevel(latest?.risk_level) };
  const command = commandStateFromRisk(topZone.risk);
  if (commandEl) {
    commandEl.textContent = command === "SYSTEM NORMAL" ? "SYSTEM NORMAL" : `${command} ${topZone.name.toUpperCase()}`;
    commandEl.setAttribute("data-tone", zoneToneFromRisk(topZone.risk));
  }
  if (eventPanel) {
    const showEvent = command === "INTERVENE";
    eventPanel.classList.toggle("hidden", !showEvent);
    if (showEvent) {
      if (eventHeadline) eventHeadline.textContent = `INTERVENE ${topZone.name.toUpperCase()}`;
      if (eventIssue) eventIssue.textContent = `Issue: ${String(latest?.phase || latest?.state || "Critical shift").slice(0, 48)}`;
      if (eventLocation) eventLocation.textContent = `Location: ${topZone.name}`;
      if (eventProgression) eventProgression.textContent = `Progression: ${String(latest?.trend || "increasing").toLowerCase()}`;
    }
  }
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

// Load dashboard with analysis results applied to data
async function loadDashboard() {
  try {
    wireDetailsLayerToggle();
    const activeRun = state.activeRun?.run_id;
    const recentResults = await fetchRecentResults({ run_id: activeRun, limit: 50 });
    state.dashboardRecent = Array.isArray(recentResults?.results) ? recentResults.results : [];

    const latest = state.dashboardRecent?.[0] || null;
    state.dashboardCurrentAlertStatus = latest?.alert_status || null;

    const zonesResponse = await fetchJson(apiUrl("/zones", tenantScopeParams())).catch(() => []);
    const zoneRows = Array.isArray(zonesResponse) ? zonesResponse : [];
    const fallbackZones = zoneRows.length
      ? zoneRows
      : (state.dashboardRecent || []).slice(0, 9).map((row, index) => ({
          zone_name: row.asset_id || `Zone ${index + 1}`,
          risk_level: row.risk_level || "LOW",
        }));
    const zones = fallbackZones.map((row, index) => ({
      name: resolveZoneName(row, index),
      risk: resolveZoneRisk(row, latest?.risk_level),
    }));
    renderZoneGrid(zones);
    renderCommandLayer(latest, zones);

    const healthCaptionEl = qs("#dashboardHealthCaption");
    if (healthCaptionEl) healthCaptionEl.textContent = latest?.timestamp ? `Updated ${latest.timestamp}` : "No telemetry timestamp yet.";

    const trendEl = qs("#snapshotTrend");
    if (trendEl) trendEl.textContent = (latest?.trend || "-").toUpperCase();
    const stateEl = qs("#snapshotState");
    if (stateEl) stateEl.textContent = latest?.state || latest?.interpreted_state || "Unknown";
    const recommendationEl = qs("#snapshotRecommendation");
    if (recommendationEl) recommendationEl.textContent = latest?.operator_message || "No event narrative available.";

    // Render metrics
    const trendMetricEl = qs("#metricTrend");
    const phaseEl = qs("#metricPhaseBadge");
    const riskBadgeEl = qs("#metricRiskBadge");
    const confidenceEl = qs("#metricStateConfidence");

    if (trendMetricEl) trendMetricEl.textContent = latest?.trend || "-";
    if (phaseEl) phaseEl.innerHTML = latest ? phaseBadgeHtml(phaseFromResult(latest)) : "-";
    if (riskBadgeEl) riskBadgeEl.innerHTML = riskBadgeHtml(normalizeRiskLevel(latest?.risk_level));
    if (confidenceEl) {
      const conf = latest?.confidence || latest?.state_confidence || 0;
      confidenceEl.textContent = typeof conf === "number" ? `${Math.round(conf)}%` : "--%";
    }

    // Render risk explanation
    renderRiskExplanation(latest, {}, state.dashboardRecent);

  } catch (err) {
    console.error("Dashboard load error:", err);
    setStatus(String(err.message || err), true, true);
  }
}

function wireDetailsLayerToggle() {
  const btn = qs("#viewDetailsBtn");
  const panel = qs("#detailsLayer");
  if (!btn || !panel || btn.dataset.wired) return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", () => {
    const nowHidden = panel.classList.toggle("hidden");
    btn.textContent = nowHidden ? "View Details" : "Hide Details";
  });
}

// Load validation/replay page with demo scenario
async function loadValidationPage() {
  try {
    const demoBtn = qs("#seedDemoBtn");
    if (demoBtn && !demoBtn.dataset.wired) {
      demoBtn.dataset.wired = "1";
      demoBtn.addEventListener("click", async () => {
        try {
          setStatus("Starting validation replay...", false, false);
          const demoPanel = qs("#demoProgressPanel");
          if (demoPanel) demoPanel.classList.remove("hidden");

          // Fetch demo scenario
          const demoRes = await fetchJson(apiUrl("/demo/scenario", tenantScopeParams()));
          const results = Array.isArray(demoRes?.results) ? demoRes.results : [];

          state.dashboardRecent = results;
          state.demo = {
            enabled: true,
            replay: {
              uiState: "completed",
              totalFrames: results.length,
              currentFrame: results.length
            }
          };

          // Update progress panel
          const phaseEl = qs("#demoProgressPhase");
          const countEl = qs("#demoProgressCount");
          const textEl = qs("#demoProgressText");

          if (phaseEl) phaseEl.textContent = "Validation replay completed";
          if (countEl) countEl.textContent = `${results.length}/${results.length}`;
          if (textEl) textEl.textContent = "Replay analysis complete. Review results in Investigation run.";

          setStatus("Replay completed. Opening run investigation...", false, false);
          setTimeout(() => {
            window.location.href = "/app/runs";
          }, 2000);

        } catch (err) {
          const { friendlyErrorMessage } = window.NeraiumUI || {};
          const msg = friendlyErrorMessage ? friendlyErrorMessage(err, "replay") : String(err.message || err);
          setStatus(msg, true, true);
        }
      });
    }
  } catch (err) {
    console.error("Validation page load error:", err);
  }
}
