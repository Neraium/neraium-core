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
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(2) : "-";
  return String(v);
}

function formatBytes(bytes) {
  const n = Number(bytes || 0);
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[i]}`;
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

function transitionArrow(prevResult, nextResult) {
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

function renderRiskExplanation(result, opts = {}) {
  const titleEl = qs(opts.titleSelector || "#riskExplainTitle");
  const bodyEl = qs(opts.bodySelector || "#riskExplainBody");
  const panelEl = qs(opts.panelSelector || "#riskExplainPanel");
  const badgeEl = qs(opts.badgeSelector || "#riskExplainBadge");
  if (!titleEl || !bodyEl || !panelEl) return;

  if (!result) {
    panelEl.classList.remove("hidden");
    panelEl.setAttribute("data-risk", "UNKNOWN");
    titleEl.textContent = "Why this risk level";
    bodyEl.textContent = "No result available yet to explain risk.";
    if (badgeEl) badgeEl.innerHTML = riskBadgeHtml("UNKNOWN");
    return;
  }

  const explanation = summarizeRiskDrivers(result);
  panelEl.classList.remove("hidden");
  panelEl.setAttribute("data-risk", explanation.risk);
  titleEl.textContent = `Why risk is ${explanation.risk}`;
  bodyEl.textContent = explanation.text;
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

function dashboardChronologicalResults() {
  return (state.dashboardRecent || []).slice().reverse();
}

function sparkPointAnomaly(prev, curr) {
  if (!curr) return false;
  const pd = structuralDriftFromResult(prev);
  const cd = structuralDriftFromResult(curr);
  const pi = compositeInstabilityFromResult(prev);
  const ci = compositeInstabilityFromResult(curr);
  const dJump = typeof pd === "number" && typeof cd === "number" ? Math.abs(cd - pd) : 0;
  const iJump = typeof pi === "number" && typeof ci === "number" ? Math.abs(ci - pi) : 0;
  if (dJump >= 0.12 || iJump >= 0.12) return true;
  if (prev && riskRankNumber(curr.risk_level) > riskRankNumber(prev.risk_level)) return true;
  return false;
}

function renderDashboardSparkline(series) {
  const canvas = qs("#dashboardSparkline");
  const tooltip = qs("#dashboardSparklineTooltip");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 640;
  const cssH = 140;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 8, r: 12, t: 18, b: 22 };
  const innerW = cssW - pad.l - pad.r;
  const innerH = cssH - pad.t - pad.b;
  const items = Array.isArray(series) ? series : [];
  if (items.length === 0) {
    ctx.fillStyle = "rgba(140, 164, 206, 0.55)";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillText("Trend appears after ingest.", pad.l, pad.t + 24);
    if (tooltip) {
      tooltip.classList.add("hidden");
      tooltip.textContent = "";
    }
    return;
  }

  const driftVals = items.map((r) => structuralDriftFromResult(r)).map((v) => (typeof v === "number" ? v : 0));
  const compVals = items.map((r) => compositeInstabilityFromResult(r)).map((v) => (typeof v === "number" ? v : 0));
  const maxY = Math.max(0.08, ...driftVals, ...compVals, 1);
  const n = items.length;
  const step = n <= 1 ? 0 : innerW / (n - 1);

  ctx.strokeStyle = "rgba(80, 110, 160, 0.35)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g += 1) {
    const y = pad.t + innerH * (g / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + innerW, y);
    ctx.stroke();
  }

  function xAt(i) {
    return pad.l + step * i;
  }
  function yAt(v) {
    return pad.t + innerH - (Math.min(maxY, Math.max(0, v)) / maxY) * innerH;
  }

  function drawLine(vals, color, fill) {
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = xAt(i);
      const y = yAt(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (fill) {
      ctx.lineTo(xAt(n - 1), pad.t + innerH);
      ctx.lineTo(pad.l, pad.t + innerH);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
  }

  drawLine(
    driftVals,
    "rgba(121, 171, 255, 0.95)",
    "rgba(106, 156, 250, 0.12)",
  );

  drawLine(compVals, "rgba(255, 191, 86, 0.92)", null);

  const lastIdx = n - 1;
  items.forEach((r, i) => {
    const prev = i > 0 ? items[i - 1] : null;
    const anomaly = sparkPointAnomaly(prev, r);
    const cx = xAt(i);
    const cyD = yAt(driftVals[i]);
    const cyC = yAt(compVals[i]);
    const isHover = state.dashboardSparkline.hoveredIndex === i;
    const radius = i === lastIdx ? (anomaly ? 5 : 4) : anomaly || isHover ? 4 : 0;
    if (radius > 0) {
      ctx.beginPath();
      ctx.arc(cx, cyD, radius, 0, Math.PI * 2);
      ctx.fillStyle = anomaly ? "rgba(255, 120, 120, 0.95)" : "rgba(186, 210, 255, 0.95)";
      ctx.fill();
      if (i === lastIdx) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    ctx.beginPath();
    ctx.arc(cx, cyC, i === lastIdx ? 3 : 0, 0, Math.PI * 2);
    if (i === lastIdx) {
      ctx.fillStyle = "rgba(255, 207, 120, 0.95)";
      ctx.fill();
    }
  });

  ctx.fillStyle = "rgba(180, 198, 230, 0.85)";
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.fillText("drift", pad.l, 12);
  ctx.fillStyle = "rgba(255, 201, 120, 0.9)";
  ctx.fillText("composite", pad.l + 52, 12);

  canvas.dataset.sparkMeta = JSON.stringify(
    items.map((r, i) => ({
      i,
      x: xAt(i),
      drift: driftVals[i],
      comp: compVals[i],
      ts: String(r.timestamp || r.persisted_at || ""),
      risk: normalizeRiskLevel(r.risk_level),
    })),
  );
}

function bindDashboardSparklineInteractions() {
  const canvas = qs("#dashboardSparkline");
  if (!canvas || canvas.dataset.sparkBound === "1") return;
  canvas.dataset.sparkBound = "1";
  const tooltip = qs("#dashboardSparklineTooltip");

  function metaList() {
    try {
      return JSON.parse(canvas.dataset.sparkMeta || "[]");
    } catch (_e) {
      return [];
    }
  }

  function showTip(idx, clientX, clientY) {
    const list = metaList();
    const row = list.find((m) => m.i === idx);
    if (!row || !tooltip) return;
    tooltip.classList.remove("hidden");
    tooltip.innerHTML = `<strong>${escapeHtml(row.ts)}</strong><br/>Drift ${row.drift.toFixed(2)} · Composite ${row.comp.toFixed(
      2,
    )}<br/><span class="spark-tip-risk">${escapeHtml(row.risk)} risk</span>`;
    const wrap = canvas.parentElement;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const tipW = 140;
    const tipH = 52;
    tooltip.style.left = `${Math.min(rect.width - tipW - 8, Math.max(8, x + 12))}px`;
    tooltip.style.top = `${Math.min(rect.height - tipH - 8, Math.max(8, y - 36))}px`;
  }

  function pickIndex(clientX) {
    const list = metaList();
    if (!list.length) return -1;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    let best = -1;
    let bestDist = 22;
    list.forEach((m) => {
      const d = Math.abs(m.x - x);
      if (d < bestDist) {
        bestDist = d;
        best = m.i;
      }
    });
    return best;
  }

  function clearHover() {
    state.dashboardSparkline.hoveredIndex = null;
    if (tooltip) tooltip.classList.add("hidden");
    renderDashboardSparkline(dashboardChronologicalResults());
  }

  function updateFromClient(clientX, clientY) {
    const list = metaList();
    if (!list.length) {
      clearHover();
      return;
    }
    const best = pickIndex(clientX);
    if (best < 0) {
      clearHover();
      return;
    }
    state.dashboardSparkline.hoveredIndex = best;
    renderDashboardSparkline(dashboardChronologicalResults());
    showTip(best, clientX, clientY);
  }

  canvas.addEventListener("pointerdown", (evt) => {
    if (evt.pointerType === "touch" || evt.pointerType === "pen") {
      try {
        canvas.setPointerCapture(evt.pointerId);
      } catch (_e) {
        /* ignore */
      }
    }
    updateFromClient(evt.clientX, evt.clientY);
  });

  canvas.addEventListener("pointermove", (evt) => {
    updateFromClient(evt.clientX, evt.clientY);
  });

  canvas.addEventListener("pointerup", (evt) => {
    if (evt.pointerType === "touch" || evt.pointerType === "pen") {
      try {
        canvas.releasePointerCapture(evt.pointerId);
      } catch (_e) {
        /* ignore */
      }
      clearHover();
    }
  });

  canvas.addEventListener("pointerleave", (evt) => {
    if (evt.pointerType === "mouse") {
      clearHover();
    }
  });

  canvas.addEventListener("pointercancel", () => {
    clearHover();
  });
}

function renderDashboardHero(latest, prev) {
  const scoreEl = qs("#dashboardHealthScore");
  const narrative = qs("#dashboardNarrativeStrip");
  const countEl = qs("#dashboardSnapshotCount");
  const alertCountEl = qs("#dashboardAlertCount");
  const alertSummaryEl = qs("#dashboardAlertSummary");
  const link = qs("#dashboardOpenRunLink");
  const alertTile = qs("#dashboardAlertTile");

  const score = healthScoreFromSignals(latest);
  if (scoreEl) scoreEl.textContent = latest ? String(score) : "-";
  setHealthRingScore(latest ? score : 0);
  if (narrative) {
    const line = buildDemoNarrative(latest, prev);
    narrative.textContent = line.message;
    narrative.setAttribute("data-severity", line.severity || "normal");
  }
  const chron = dashboardChronologicalResults();
  if (countEl) countEl.textContent = String(chron.length);

  const alerts = state.dashboardAlerts || [];
  if (alertCountEl) alertCountEl.textContent = String(alerts.length);
  if (alertSummaryEl) {
    const first = alerts[0];
    alertSummaryEl.textContent = first
      ? String(first.message || first.type || "Alert").slice(0, 120)
      : "No open alerts for this run.";
  }
  if (alertTile) {
    const critical = alerts.some((a) => String(a.severity || "").toLowerCase() === "critical");
    const high = alerts.some((a) => {
      const s = String(a.severity || "").toLowerCase();
      return s === "high" || s === "critical";
    });
    alertTile.classList.toggle("insight-tile-alert-critical", critical);
    alertTile.classList.toggle("insight-tile-alert-watch", !critical && high);
  }

  const runId = state.activeRun?.run_id || "";
  if (link) {
    if (runId) {
      link.href = `/app/runs/${encodeURIComponent(runId)}?customer_id=${encodeURIComponent(customerIdValue(state.tenant.customerId))}`;
      link.classList.remove("disabled");
    } else {
      link.href = "/app/runs";
      link.classList.add("disabled");
    }
  }
}

function riskBadgeHtml(value) {
  const risk = normalizeRiskLevel(value);
  const classMap = {
    LOW: "badge-risk-low",
    MEDIUM: "badge-risk-medium",
    HIGH: "badge-risk-high",
    UNKNOWN: "badge-risk-unknown",
  };
  return `<span class="badge ${classMap[risk]}">${escapeHtml(risk)}</span>`;
}

function phaseBadgeHtml(value) {
  const raw = String(value || "-").trim();
  const phase = raw || "-";
  const s = phase.toLowerCase();
  let klass = "badge-phase";
  if (s.includes("stable") || s === "nominal_structure") {
    klass = "badge-phase-stable";
  } else if (s.includes("watch") || s.includes("drift") || s.includes("regime_shift")) {
    klass = "badge-phase-watch";
  } else if (s.includes("alert") || s.includes("unstable") || s.includes("instability")) {
    klass = "badge-phase-alert";
  }
  return `<span class="badge ${klass}">${escapeHtml(phase)}</span>`;
}

function buildDemoNarrative(result, prevResult) {
  if (!result) {
    return {
      message: "No structural signal yet. Seed or upload data to begin monitoring.",
      severity: "normal",
    };
  }
  const risk = normalizeRiskLevel(result.risk_level);
  const drift = structuralDriftFromResult(result);
  const instability = compositeInstabilityFromResult(result);
  const transition = transitionLabel(prevResult, result);
  const severity = transitionSeverity(prevResult, result);
  const driftText = typeof drift === "number" ? drift.toFixed(2) : "n/a";
  const instabilityText = typeof instability === "number" ? instability.toFixed(2) : "n/a";
  let message = `System ${String(result.state || result.interpreted_state || "UNKNOWN").toUpperCase()}, risk ${risk}.`;
  if (severity === "critical") {
    message = `Immediate attention: ${transition}. Drift ${driftText}, instability ${instabilityText}.`;
  } else if (severity === "watch") {
    message = `Change detected: ${transition}. Drift ${driftText}, instability ${instabilityText}.`;
  } else if (risk === "HIGH") {
    message = `High-risk condition sustained. Drift ${driftText}, instability ${instabilityText}.`;
  } else if (risk === "MEDIUM") {
    message = `Watch condition. Drift ${driftText}, instability ${instabilityText}.`;
  } else if (risk === "LOW") {
    message = `Stable operating envelope. Drift ${driftText}, instability ${instabilityText}.`;
  }
  return { message, severity };
}

function demoFriendlyOperatorMessage(result, prevResult) {
  const narrative = buildDemoNarrative(result, prevResult);
  const raw = String(result?.operator_message || "").trim();
  if (!raw) return narrative.message;
  if (raw.toLowerCase().startsWith(narrative.message.toLowerCase())) return raw;
  return `${narrative.message} ${raw}`;
}

function renderRunSignals(latest, prev) {
  const strip = qs("#runSignalSeparation");
  const stateEl = qs("#runSignalState");
  const riskEl = qs("#runSignalRisk");
  const trendEl = qs("#runSignalTrend");
  if (!strip || !stateEl || !riskEl || !trendEl) return;
  if (!latest) {
    strip.classList.add("hidden");
    return;
  }
  strip.classList.remove("hidden");
  const stateText = String(latest.state || latest.interpreted_state || "-");
  stateEl.textContent = stateText;
  riskEl.innerHTML = riskBadgeHtml(latest.risk_level);
  trendEl.textContent = String(trendFromResult(latest) || "-");
  const sev = transitionSeverity(prev, latest);
  strip.setAttribute("data-severity", sev);
}

function renderRunTransitionStrip(prev, latest) {
  const strip = qs("#runTransitionStrip");
  if (!strip) return;
  if (!latest) {
    strip.className = "timeline-transition-strip hidden";
    strip.textContent = "";
    return;
  }
  const label = transitionLabel(prev, latest);
  const sev = transitionSeverity(prev, latest);
  strip.className = `timeline-transition-strip ${sev === "critical" ? "high" : sev === "watch" ? "watch" : ""}`;
  strip.innerHTML = `<strong>${sev === "critical" ? "Key transition" : "Latest transition"}:</strong> ${escapeHtml(label)}`;
  strip.classList.remove("hidden");
}

function stopDemoPlayback() {
  if (state.demo.timer) {
    window.clearTimeout(state.demo.timer);
    state.demo.timer = null;
  }
  state.demo.isPlaying = false;
}

function extractDemoKeyEvents(results) {
  if (!Array.isArray(results) || results.length === 0) return [];
  const events = [];
  for (let i = 0; i < results.length; i += 1) {
    const current = results[i];
    const prev = i > 0 ? results[i - 1] : null;
    const sev = transitionSeverity(prev, current);
    const transition = transitionLabel(prev, current);
    const prevDrift = structuralDriftFromResult(prev);
    const nextDrift = structuralDriftFromResult(current);
    const driftJump =
      typeof prevDrift === "number" && typeof nextDrift === "number"
        ? Math.abs(nextDrift - prevDrift)
        : 0;
    const isSpike = normalizeRiskLevel(current.risk_level) === "HIGH" && normalizeRiskLevel(prev?.risk_level) !== "HIGH";
    const isDriftEvent = driftJump >= 0.14;
    if (sev === "critical" || isSpike || isDriftEvent) {
      events.push({
        index: i + 1,
        ts: String(current.timestamp || current.persisted_at || ""),
        severity: sev === "normal" ? (isSpike ? "critical" : "watch") : sev,
        text: isDriftEvent ? `${transition} · drift jump ${driftJump.toFixed(2)}` : transition,
      });
    }
  }
  return events.slice(0, 8);
}

function renderDemoKeyEvents(events = state.demo.keyEvents || []) {
  const panel = qs("#demoKeyEventsPanel");
  const list = qs("#demoKeyEventsList");
  if (!panel || !list) return;
  const show = !!state.demo.enabled && Array.isArray(events) && events.length > 0;
  if (!show) {
    panel.classList.add("hidden");
    list.innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  const cursor = Math.max(1, Number(state.demo.cursor || 1));
  list.innerHTML = events
    .map((event) => {
      const reached = event.index <= cursor;
      return `<li class="demo-key-event ${escapeHtml(event.severity)} ${reached ? "reached" : ""}">
        <div class="msg-head">${escapeHtml(event.ts)}</div>
        <div>${escapeHtml(event.text)}</div>
      </li>`;
    })
    .join("");
}

function setDemoPlaybackUI() {
  const panel = qs("#demoPlaybackPanel");
  const progress = qs("#demoPlaybackProgress");
  const playPauseBtn = qs("#demoPlayPauseBtn");
  const replayBtn = qs("#demoReplayBtn");
  if (!panel || !progress || !playPauseBtn || !replayBtn) return;
  const route = getRoute();
  const total = state.runRecent.length;
  const show = state.demo.enabled && route.page === "run-detail" && total > 0;
  if (!show) {
    panel.classList.add("hidden");
    progress.textContent = state.demo.enabled ? "Open a run to start playback" : "Demo Mode off";
    playPauseBtn.textContent = "Play timeline";
    replayBtn.disabled = true;
    renderDemoKeyEvents([]);
    return;
  }
  panel.classList.remove("hidden");
  const cursor = Math.max(1, Math.min(total, Number(state.demo.cursor || total)));
  progress.textContent = `Snapshot ${cursor}/${total}`;
  playPauseBtn.textContent = state.demo.isPlaying ? "Pause timeline" : "Play timeline";
  replayBtn.disabled = total <= 1;
  renderDemoKeyEvents();
}

function maybeAutoStartDemoPlayback() {
  if (!state.demo.enabled) return;
  if (state.demo.isPlaying) return;
  if (state.runRecent.length < 2) return;
  const route = getRoute();
  if (route.page !== "run-detail") return;
  state.demo.cursor = 1;
  state.demo.isPlaying = true;
  applyDemoSnapshot();
  scheduleDemoTick();
}

function applyDemoSnapshot() {
  if (!state.demo.enabled) return;
  if (!state.runRecent.length) {
    stopDemoPlayback();
    return;
  }
  const total = state.runRecent.length;
  if (!Number.isFinite(state.demo.cursor) || state.demo.cursor <= 0) {
    state.demo.cursor = 1;
  }
  if (state.demo.cursor > total) {
    state.demo.cursor = total;
    stopDemoPlayback();
  }
  renderRunDetailFromState();
  const route = getRoute();
  if (route.page === "run-detail" && route.runId && state.runRecent.length > 0) {
    const chronologicalFull = state.runRecent.slice().reverse();
    const chronological = chronologicalFull.slice(0, Math.max(1, Number(state.demo.cursor || chronologicalFull.length)));
    const latest = chronological.length ? chronological[chronological.length - 1] : null;
    const resultId = latest?.result_id ?? null;
    loadRunGeometry(route.runId, resultId);
  }
}

function scheduleDemoTick() {
  if (!state.demo.isPlaying) return;
  if (state.demo.timer) {
    window.clearTimeout(state.demo.timer);
    state.demo.timer = null;
  }
  state.demo.timer = window.setTimeout(() => {
    if (!state.demo.isPlaying) return;
    const total = state.runRecent.length;
    if (state.demo.cursor >= total) {
      stopDemoPlayback();
      setDemoPlaybackUI();
      return;
    }
    state.demo.cursor += 1;
    applyDemoSnapshot();
    scheduleDemoTick();
  }, DEMO_PLAYBACK_INTERVAL_MS);
}

function toggleDemoPlayback(forcePlay = null) {
  if (!state.demo.enabled) {
    setStatus("Enable Demo Mode first.", true, true);
    return;
  }
  if (!state.runRecent.length) {
    setStatus("No run results available for playback.", true, true);
    return;
  }
  const shouldPlay = forcePlay === null ? !state.demo.isPlaying : !!forcePlay;
  if (!shouldPlay) {
    stopDemoPlayback();
    setDemoPlaybackUI();
    return;
  }
  if (state.demo.cursor >= state.runRecent.length) {
    state.demo.cursor = 1;
  }
  state.demo.isPlaying = true;
  applyDemoSnapshot();
  scheduleDemoTick();
}

function replayDemoTimeline() {
  if (!state.demo.enabled || !state.runRecent.length) return;
  state.demo.cursor = 1;
  state.demo.isPlaying = true;
  applyDemoSnapshot();
  scheduleDemoTick();
}

async function toggleDemoMode(enabled) {
  state.demo.enabled = !!enabled;
  persistDemoMode();
  if (!state.demo.enabled) {
    stopDemoPlayback();
    state.demo.cursor = state.runRecent.length || 0;
  } else if (state.runRecent.length > 0) {
    state.demo.cursor = state.runRecent.length;
    state.demo.keyEvents = extractDemoKeyEvents(state.runRecent.slice().reverse());
    state.demo.activeRunId = state.activeRun?.run_id || state.demo.activeRunId;
  }
  renderTenantControls();
  renderRunDetailFromState();
  if (state.demo.enabled) {
    maybeAutoStartDemoPlayback();
  }
}

function buildDemoScenarioItems({ profile, siteId, assetId, minutes = 120 }) {
  const now = Date.now();
  const out = [];
  for (let i = 0; i < minutes; i += 1) {
    const p = i / Math.max(1, minutes - 1);
    const t = new Date(now - (minutes - i) * 60_000).toISOString();
    let driftLift = 0.12;
    let vibSpike = 0.2;
    if (profile === "watch") {
      driftLift = 0.35 + p * 0.35;
      vibSpike = 0.55 + p * 0.45;
    } else if (profile === "critical") {
      driftLift = 0.25 + p * 0.85;
      vibSpike = 0.8 + p * 1.8;
    }
    const waveA = Math.sin(i / 6);
    const waveB = Math.cos(i / 8);
    out.push({
      timestamp: t,
      site_id: siteId,
      asset_id: assetId,
      sensor_values: {
        pressure: 44 + waveA * (1 + driftLift * 0.6) + p * (0.8 + driftLift),
        flow: 28 + waveB * (1 + driftLift * 0.4) - p * (0.3 + driftLift * 0.3),
        vibration: 6 + Math.sin(i / 3.2) * (1 + vibSpike) + driftLift * 2.2,
        temperature: 61 + Math.cos(i / 4.8) * (1 + driftLift * 0.5) + p * (0.5 + driftLift * 0.8),
      },
    });
  }
  return out;
}

async function prepareDemoRuns() {
  if (state.demo.preparing) return null;
  state.demo.preparing = true;
  renderTenantControls();
  try {
    const suffix = new Date().toISOString().slice(11, 16).replace(":", "");
    const scenarios = [
      { name: `Demo Stable ${suffix}`, profile: "stable", siteId: "north-yard", assetId: "compressor-A" },
      { name: `Demo Watch ${suffix}`, profile: "watch", siteId: "north-yard", assetId: "compressor-B" },
      { name: `Demo Escalation ${suffix}`, profile: "critical", siteId: "south-yard", assetId: "compressor-C" },
    ];
    const created = [];
    for (const scenario of scenarios) {
      const runEnv = await fetchJson(apiUrl("/runs", tenantScopeParams()), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: scenario.name,
          config: { source: "demo-mode", scenario: scenario.profile },
          activate: false,
        }),
      });
      const run = runEnv.run;
      created.push(run);
      const items = buildDemoScenarioItems(scenario);
      await fetchJson(apiUrl("/ingest/batch", tenantScopeParams({ run_id: run.run_id })), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: items.map((item) => ({
            ...item,
            customer_id: customerIdValue(state.tenant.customerId),
          })),
        }),
      });
    }
    const focusRun = created[created.length - 1] || null;
    if (focusRun?.run_id) {
      await fetchJson(apiUrl(`/runs/${encodeURIComponent(focusRun.run_id)}/activate`, tenantScopeParams()), {
        method: "POST",
      });
      state.demo.activeRunId = focusRun.run_id;
    }
    state.demo.prepared = true;
    return focusRun;
  } finally {
    state.demo.preparing = false;
    renderTenantControls();
  }
}

function getRoute() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length === 0 || parts[0] === "dashboard") return { page: "dashboard" };
  if (parts[0] === "upload") return { page: "upload" };
  if (parts[0] === "demo" && parts[1] === "sii") return { page: "demo-sii" };
  if (parts[0] === "app" && parts[1] === "runs" && parts[2]) return { page: "run-detail", runId: parts[2] };
  if (parts[0] === "app" && parts[1] === "runs") return { page: "runs" };
  if (parts[0] === "app" && parts[1] === "results" && parts[2]) return { page: "result-detail", resultId: parts[2] };
  return { page: "dashboard" };
}

const state = {
  activeRun: null,
  runs: [],
  dashboardRecent: [],
  dashboardAlerts: [],
  runRecent: [],
  runGeometry: null,
  uploadFile: null,
  uploadJob: {
    id: null,
    pollTimer: null,
    active: false,
  },
  tenant: {
    customerId: "default-customer",
    siteId: "",
    knownSites: [],
  },
  runsView: {
    search: "",
    status: "all",
    sort: "created_desc",
  },
  runDetailView: {
    search: "",
    sort: "timestamp_desc",
    range: "200",
  },
  charts: {
    drift: null,
    composite: null,
  },
  geometry3d: {
    renderer: null,
    scene: null,
    camera: null,
    controls: null,
    edgeGroup: null,
    raycaster: null,
    pointer: null,
    nodeGroup: null,
    nodeMeshById: {},
    nodeDataById: {},
    nodeLabelById: {},
    nodeGlowById: {},
    unstablePulseById: {},
    selectedId: null,
    hoveredId: null,
    frameId: null,
    resizeObserver: null,
    interactionEnabled: false,
    cleanupPointer: null,
    cleanupResize: null,
  },
  demo: {
    enabled: false,
    prepared: false,
    preparing: false,
    isPlaying: false,
    timer: null,
    cursor: 0,
    keyEvents: [],
    activeRunId: "",
  },
  dashboardSparkline: {
    hoveredIndex: null,
  },
};

const TENANT_STORAGE_KEY = "neraium_customer_id";
const DEMO_MODE_STORAGE_KEY = "neraium_demo_mode";
const DEMO_PLAYBACK_INTERVAL_MS = 850;

function customerIdValue(value) {
  const text = String(value || "").trim();
  return text || "default-customer";
}

function siteIdValue(value) {
  return String(value || "").trim();
}

function readTenantFromStorage() {
  try {
    const stored = window.localStorage.getItem(TENANT_STORAGE_KEY);
    state.tenant.customerId = customerIdValue(stored);
  } catch (_err) {
    state.tenant.customerId = "default-customer";
  }
}

function readDemoModeFromStorage() {
  try {
    const raw = String(window.localStorage.getItem(DEMO_MODE_STORAGE_KEY) || "").trim().toLowerCase();
    state.demo.enabled = raw === "1" || raw === "true" || raw === "on";
  } catch (_err) {
    state.demo.enabled = false;
  }
}

function persistDemoMode() {
  try {
    window.localStorage.setItem(DEMO_MODE_STORAGE_KEY, state.demo.enabled ? "1" : "0");
  } catch (_err) {
    // no-op
  }
}

function tenantScopeParams(extra = {}) {
  const out = {
    customer_id: customerIdValue(state.tenant.customerId),
  };
  const siteId = siteIdValue(state.tenant.siteId);
  if (siteId) {
    out.site_id = siteId;
  }
  return { ...out, ...extra };
}

function routeScopeFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const out = {};
  const customer = params.get("customer_id");
  const site = params.get("site_id");
  if (customer) out.customer_id = customerIdValue(customer);
  if (site) out.site_id = siteIdValue(site);
  return out;
}

function collectKnownSites(results) {
  const known = new Set(state.tenant.knownSites);
  (results || []).forEach((row) => {
    const site = siteIdValue(row?.site_id);
    if (site) known.add(site);
  });
  state.tenant.knownSites = Array.from(known).sort((a, b) => a.localeCompare(b));
}

function renderTenantControls() {
  const customerInput = qs("#customerFilterInput");
  const siteInput = qs("#siteFilterInput");
  const siteList = qs("#knownSitesList");
  const demoToggle = qs("#demoModeToggle");
  const prepareDemoBtn = qs("#prepareDemoBtn");
  if (customerInput) customerInput.value = customerIdValue(state.tenant.customerId);
  if (siteInput) siteInput.value = siteIdValue(state.tenant.siteId);
  if (demoToggle) demoToggle.checked = !!state.demo.enabled;
  if (prepareDemoBtn) {
    prepareDemoBtn.disabled = state.demo.preparing;
    prepareDemoBtn.textContent = state.demo.preparing ? "Creating…" : "Create Demo Run";
  }
  if (siteList) {
    siteList.innerHTML = state.tenant.knownSites
      .map((site) => `<option value="${escapeHtml(site)}"></option>`)
      .join("");
  }
}

async function applyTenantFromControls() {
  const customerInput = qs("#customerFilterInput");
  const siteInput = qs("#siteFilterInput");
  const customer = customerIdValue(customerInput?.value);
  const site = siteIdValue(siteInput?.value);
  state.tenant.customerId = customer;
  state.tenant.siteId = site;
  try {
    window.localStorage.setItem(TENANT_STORAGE_KEY, customer);
  } catch (_err) {
    // no-op
  }
  await refreshCurrentPage();
}

const chartTheme = {
  tickColor: "#9cb0ce",
  gridColor: "rgba(153, 178, 217, 0.12)",
  legendColor: "#d8e5ff",
  tooltipBg: "rgba(6, 12, 23, 0.92)",
  tooltipBorder: "#3d5786",
};

function buildTrendChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    scales: {
      x: {
        ticks: { color: chartTheme.tickColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 6 },
        grid: { color: chartTheme.gridColor },
      },
      y: {
        ticks: { color: chartTheme.tickColor },
        grid: { color: chartTheme.gridColor },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: chartTheme.legendColor,
          usePointStyle: true,
          pointStyle: "circle",
          boxHeight: 7,
        },
      },
      tooltip: {
        backgroundColor: chartTheme.tooltipBg,
        borderColor: chartTheme.tooltipBorder,
        borderWidth: 1,
        titleColor: "#ecf3ff",
        bodyColor: "#cfddf8",
        displayColors: false,
      },
    },
  };
}

function disposeGeometryRenderer() {
  const g = state.geometry3d;
  if (g.frameId) {
    window.cancelAnimationFrame(g.frameId);
    g.frameId = null;
  }
  if (g.resizeObserver) {
    try {
      g.resizeObserver.disconnect();
    } catch (_err) {
      // no-op
    }
    g.resizeObserver = null;
  }
  if (typeof g.cleanupResize === "function") {
    g.cleanupResize();
    g.cleanupResize = null;
  }
  if (typeof g.cleanupPointer === "function") {
    g.cleanupPointer();
    g.cleanupPointer = null;
  }
  if (g.controls) {
    g.controls.dispose();
    g.controls = null;
  }
  if (g.renderer) {
    g.renderer.dispose();
    if (g.renderer.domElement && g.renderer.domElement.parentElement) {
      g.renderer.domElement.parentElement.removeChild(g.renderer.domElement);
    }
    g.renderer = null;
  }
  g.scene = null;
  g.camera = null;
  g.edgeGroup = null;
  g.raycaster = null;
  g.pointer = null;
  g.nodeGroup = null;
  g.nodeMeshById = {};
  g.nodeDataById = {};
  g.nodeLabelById = {};
  g.nodeGlowById = {};
  g.unstablePulseById = {};
  g.selectedId = null;
  g.hoveredId = null;
  g.interactionEnabled = false;
}

function getThreeNamespace() {
  return window.__THREE_ESM || window.THREE;
}

function getOrbitControlsConstructor() {
  if (typeof window.__OrbitControls === "function") {
    return window.__OrbitControls;
  }
  if (window.THREE && window.THREE.OrbitControls) {
    return window.THREE.OrbitControls;
  }
  if (window.OrbitControls) {
    return window.OrbitControls;
  }
  return null;
}

function fitGeometryCamera(camera, controls, nodeGroup, three) {
  if (!camera || !controls || !nodeGroup || !three) return;
  const box = new three.Box3().setFromObject(nodeGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new three.Vector3());
  const size = box.getSize(new three.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.08);
  const dist = Math.max(1.25, maxDim * 2.2);
  controls.target.copy(center);
  camera.near = Math.max(0.008, dist / 500);
  camera.far = Math.max(80, dist * 45);
  camera.updateProjectionMatrix();
  const offset = new three.Vector3(1.2, 0.75, 1.45).normalize().multiplyScalar(dist);
  camera.position.copy(center.clone().add(offset));
  camera.lookAt(center);
}

function refreshGeometryEdges() {
  const g = state.geometry3d;
  const payload = state.runGeometry;
  if (!g.edgeGroup || !payload?.edges?.length) return;
  let idx = 0;
  payload.edges.forEach((edge) => {
    const s = payload.nodes.find((n) => String(n.id) === String(edge.source));
    const t = payload.nodes.find((n) => String(n.id) === String(edge.target));
    if (!s || !t) return;
    const p1 = geometryPositionForNode(s);
    const p2 = geometryPositionForNode(t);
    const line = g.edgeGroup.children[idx];
    idx += 1;
    if (line?.geometry?.attributes?.position) {
      const arr = line.geometry.attributes.position.array;
      arr[0] = p1.x;
      arr[1] = p1.y;
      arr[2] = p1.z;
      arr[3] = p2.x;
      arr[4] = p2.y;
      arr[5] = p2.z;
      line.geometry.attributes.position.needsUpdate = true;
    }
  });
}

function geometryViewportDimensions() {
  const host = qs("#geometryViewport");
  if (!host) return null;
  const rect = host.getBoundingClientRect();
  const w = Math.floor(rect.width || host.clientWidth || 0);
  const h = Math.floor(rect.height || host.clientHeight || 0);
  return { host, width: w, height: h };
}

function ensureGeometryViewportReady(timeoutMs = 1600) {
  return new Promise((resolve) => {
    const start = window.performance.now();
    function tick() {
      const dims = geometryViewportDimensions();
      if (dims && dims.width > 40 && dims.height > 40) {
        resolve(dims);
        return;
      }
      if (window.performance.now() - start > timeoutMs) {
        resolve(dims);
        return;
      }
      window.requestAnimationFrame(tick);
    }
    tick();
  });
}

function setGeometrySurfaceState(message, level = "info") {
  const fallback = qs("#geometryFallback");
  const canvasWrap = qs("#geometryCanvasWrap");
  const viewport = qs("#geometryViewport");
  if (fallback) {
    fallback.textContent = String(message || "");
    const tone = level === "error" ? "error" : level === "warn" ? "warn" : "info";
    fallback.className = `empty-state geometry-fallback geometry-fallback--${tone}`;
    fallback.classList.remove("hidden");
  }
  if (canvasWrap) {
    canvasWrap.classList.remove("hidden");
  }
  if (viewport) {
    viewport.classList.remove("ready");
  }
}

function showGeometryCanvas() {
  const fallback = qs("#geometryFallback");
  const canvasWrap = qs("#geometryCanvasWrap");
  const viewport = qs("#geometryViewport");
  if (fallback) {
    fallback.classList.add("hidden");
    fallback.textContent = "";
  }
  if (viewport) {
    viewport.classList.add("ready");
  }
}

function riskColorHex(riskLevel) {
  const risk = normalizeRiskLevel(riskLevel);
  if (risk === "HIGH") return 0xe46060;
  if (risk === "MEDIUM") return 0xffbf56;
  if (risk === "LOW") return 0x68d497;
  return 0x8aa6cf;
}

function nodeStateColorHex(nodeState) {
  const s = String(nodeState || "").toLowerCase();
  if (s === "critical") return 0xff7a7a;
  if (s === "watch") return 0xffca72;
  return 0x7fdaac;
}

function edgeColorHex(edge, metrics) {
  const risk = normalizeRiskLevel(metrics?.risk_level);
  if (risk === "HIGH" && Math.abs(Number(edge.delta || 0)) > 0.18) {
    return 0xff8a8a;
  }
  if (risk === "MEDIUM" && Math.abs(Number(edge.delta || 0)) > 0.12) {
    return 0xffca72;
  }
  return edge.type === "negative" ? 0x8aa4d0 : 0x9ebcf0;
}

function ensureThreeLibs() {
  const three = getThreeNamespace();
  const controlsCtor = getOrbitControlsConstructor();
  if (!three || !controlsCtor) {
    throw new Error("Three.js did not load (check network or import map).");
  }
  return { three, controlsCtor };
}

function renderGeometryLegend(payload) {
  const legend = qs("#geometryLegend");
  if (!legend) return;
  if (!payload || !payload.available) {
    legend.innerHTML = "";
    return;
  }
  const unstableCount = Number(payload?.summary?.unstable_nodes_current || 0);
  const changedEdges = Number(payload?.summary?.changed_edges_current || 0);
  legend.innerHTML = `
    <span class="geom-pill">Nodes: ${Number(payload?.nodes?.length || 0)}</span>
    <span class="geom-pill">Edges: ${Number(payload?.edges?.length || 0)}</span>
    <span class="geom-pill geom-pill-unstable">Unstable nodes: ${unstableCount}</span>
    <span class="geom-pill">Changed edges: ${changedEdges}</span>
  `;
}

function geometryPositionForNode(node) {
  const source = node?.position_current || node?.position || {};
  return {
    x: Number(source?.x || 0),
    y: Number(source?.y || 0),
    z: Number(source?.z || 0),
  };
}

function geometryStructureSummary(payload) {
  if (!payload || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) return "-";
  const totalNodes = payload.nodes.length;
  const totalEdges = payload.edges.length;
  const unstableNodes = payload.nodes.filter((n) => Boolean(n.is_unstable)).length;
  const strongEdges = payload.edges.filter((e) => Math.abs(Number(e.magnitude || 0)) >= 0.6).length;
  const density = totalNodes > 1 ? (2 * totalEdges) / (totalNodes * (totalNodes - 1)) : 0;
  return `${totalNodes} nodes · ${totalEdges} edges · unstable ${unstableNodes} · strong links ${strongEdges} · density ${toPretty(
    density
  )}`;
}

function clearGeometryModelsPanel() {
  const panel = qs("#geometryModelsPanel");
  const graphDl = qs("#geometryGraphMetricsDl");
  const graphEmpty = qs("#geometryGraphMetricsEmpty");
  const sysDl = qs("#geometrySystemStateDl");
  const sysEmpty = qs("#geometrySystemStateEmpty");
  if (panel) panel.classList.add("hidden");
  if (graphDl) graphDl.innerHTML = "";
  if (sysDl) sysDl.innerHTML = "";
  if (graphEmpty) graphEmpty.classList.add("hidden");
  if (sysEmpty) sysEmpty.classList.add("hidden");
}

function renderGeometryModelsPanel(payload) {
  const panel = qs("#geometryModelsPanel");
  const graphDl = qs("#geometryGraphMetricsDl");
  const graphEmpty = qs("#geometryGraphMetricsEmpty");
  const sysDl = qs("#geometrySystemStateDl");
  const sysEmpty = qs("#geometrySystemStateEmpty");
  if (!panel || !graphDl || !graphEmpty || !sysDl || !sysEmpty) return;

  const ga = payload && payload.graph_analytics;
  const ss = payload && payload.system_state;
  const hasGraph =
    ga &&
    typeof ga === "object" &&
    (Boolean(ga.correlation_graph) || Boolean(ga.causal_graph));
  const hasSys = ss && typeof ss === "object" && Object.keys(ss).length > 0;

  if (!hasGraph && !hasSys) {
    clearGeometryModelsPanel();
    return;
  }

  panel.classList.remove("hidden");

  if (!hasGraph) {
    graphEmpty.classList.remove("hidden");
    graphDl.innerHTML = "";
  } else {
    graphEmpty.classList.add("hidden");
    const rows = [];
    const cg = ga.correlation_graph;
    if (cg && typeof cg === "object") {
      rows.push(
        ["Correlation graph · mean degree", toPretty(cg.mean_degree)],
        ["Correlation graph · density", toPretty(cg.density)],
        ["Correlation graph · clustering", toPretty(cg.clustering)],
        ["Correlation graph · connectivity", toPretty(cg.connectivity)],
        ["Correlation graph · mean |corr| off-diag", toPretty(cg.mean_absolute_connectivity)]
      );
    }
    const cag = ga.causal_graph;
    if (cag && typeof cag === "object") {
      const labels = Array.isArray(cag.dominant_source_labels) ? cag.dominant_source_labels.filter(Boolean) : [];
      const labelText = labels.length ? labels.join(", ") : "-";
      rows.push(
        ["Causal graph · density", toPretty(cag.density)],
        ["Causal graph · asymmetry", toPretty(cag.asymmetry)],
        ["Causal graph · dominant sources", escapeHtml(labelText)]
      );
    }
    graphDl.innerHTML = rows
      .map(([dt, dd]) => `<div><dt>${escapeHtml(String(dt))}</dt><dd>${escapeHtml(String(dd))}</dd></div>`)
      .join("");
  }

  if (!hasSys) {
    sysEmpty.classList.remove("hidden");
    sysDl.innerHTML = "";
  } else {
    sysEmpty.classList.add("hidden");
    const rm = ss.regime_memory && typeof ss.regime_memory === "object" ? ss.regime_memory : {};
    const rs = ss.regime_signature && typeof ss.regime_signature === "object" ? ss.regime_signature : {};
    const nr = ss.nearest_regime && typeof ss.nearest_regime === "object" ? ss.nearest_regime : null;
    const sysRows = [
      ["Phase", toPretty(ss.phase)],
      ["Interpreted state", toPretty(ss.interpreted_state)],
      ["Regime name", toPretty(ss.regime_name)],
      ["Confidence", toPretty(ss.confidence)],
      ["Structural analysis", ss.structural_analysis_available ? "available" : "limited"],
      ["Regime library size", toPretty(rs.library_size != null ? rs.library_size : rm.library_size)],
      ["Regime memory entries", toPretty(rm.baseline_count)],
      ["Assigned regime (signature)", toPretty(rs.assigned_name)],
      ["Regime distance", ss.regime_distance != null && ss.regime_distance !== undefined ? toPretty(ss.regime_distance) : "-"],
    ];
    if (nr && nr.name != null) {
      sysRows.push(["Nearest regime (library)", `${String(nr.name)} · ${toPretty(nr.distance)}`]);
    }
    sysDl.innerHTML = sysRows
      .map(([dt, dd]) => `<div><dt>${escapeHtml(String(dt))}</dt><dd>${escapeHtml(String(dd))}</dd></div>`)
      .join("");
  }
}

function applyGeometryDisplayMode() {
  const g = state.geometry3d;
  Object.entries(g.nodeMeshById || {}).forEach(([nodeId, mesh]) => {
    const node = g.nodeDataById[nodeId];
    if (!node || !mesh) return;
    const pos = geometryPositionForNode(node);
    if (!mesh.userData) mesh.userData = {};
    const three = getThreeNamespace();
    if (!three?.Vector3) return;
    mesh.userData.basePos = new three.Vector3(pos.x, pos.y, pos.z);
    mesh.position.set(pos.x, pos.y, pos.z);
    const label = g.nodeLabelById[nodeId];
    if (label) {
      const radius = Number(mesh.userData.radius || 0.05);
      label.position.set(pos.x, pos.y + radius + 0.08, pos.z);
    }
    const halo = g.nodeGlowById[nodeId];
    if (halo) halo.position.set(pos.x, pos.y, pos.z);
  });
  refreshGeometryEdges();
  const note = qs("#geometryProjectionNote");
  const payload = state.runGeometry;
  if (note && payload) {
    let text = String(payload.projection?.note || "").replace(/\s*\[view:\s*[^\]]+\]\s*$/i, "").trim();
    if (!text) {
      text = "Positions reflect the live structural field for this snapshot.";
    }
    note.textContent = text;
  }
  const summary = qs("#geometryStructureSummary");
  if (summary) summary.textContent = geometryStructureSummary(state.runGeometry);
  if (g.camera && g.controls && g.nodeGroup && getThreeNamespace()) {
    fitGeometryCamera(g.camera, g.controls, g.nodeGroup, getThreeNamespace());
  }
}

function updateGeometryDetails(nodeId = null) {
  const payload = state.runGeometry;
  const g = state.geometry3d;
  const nodeLabel = qs("#geometryNodeLabel");
  const nodeStress = qs("#geometryNodeStress");
  const nodeMagnitude = qs("#geometryNodeMagnitude");
  const nodeState = qs("#geometryNodeState");
  const metricState = qs("#geometryState");
  const metricRisk = qs("#geometryRisk");
  const metricDrift = qs("#geometryDrift");
  const metricComposite = qs("#geometryComposite");

  if (metricState) metricState.textContent = toPretty(payload?.metrics?.state);
  if (metricRisk) metricRisk.textContent = toPretty(payload?.metrics?.risk_level);
  if (metricDrift) metricDrift.textContent = toPretty(payload?.metrics?.structural_drift_score);
  if (metricComposite) metricComposite.textContent = toPretty(payload?.metrics?.composite_instability);

  const setNodeFields = (label, stress, magnitude, stateText) => {
    if (nodeLabel) nodeLabel.textContent = label;
    if (nodeStress) nodeStress.textContent = stress;
    if (nodeMagnitude) nodeMagnitude.textContent = magnitude;
    if (nodeState) nodeState.textContent = stateText;
  };

  if (!payload || !payload.available) {
    setNodeFields("-", "-", "-", "-");
    return;
  }
  if (!nodeId) {
    setNodeFields("None selected", "-", "-", "-");
    return;
  }
  const node = g.nodeDataById[nodeId];
  if (!node) {
    setNodeFields("None selected", "-", "-", "-");
    return;
  }
  const stressVal = node.stress;
  const magVal = node.magnitude;
  const stressText =
    stressVal != null && stressVal !== "" && Number.isFinite(Number(stressVal)) ? toPretty(Number(stressVal)) : "—";
  const magText =
    magVal != null && magVal !== "" && Number.isFinite(Number(magVal)) ? toPretty(Number(magVal)) : "—";
  setNodeFields(
    String(node.label || node.id || "-"),
    stressText,
    magText,
    `${String(node.state || "-")}${node.is_unstable ? " (UNSTABLE)" : ""}`,
  );
}

function resetNodeMeshVisual(mesh) {
  if (!mesh || !mesh.material) return;
  mesh.scale.setScalar(1);
  if (mesh.material.emissive) {
    mesh.material.emissive.setHex(0x0d1424);
    mesh.material.emissiveIntensity = 0.35;
  }
}

function setGeometrySelection(nextId = null) {
  const g = state.geometry3d;
  if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
    resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
  }
  g.hoveredId = null;
  const previous = g.selectedId;
  if (previous && g.nodeMeshById[previous]) {
    resetNodeMeshVisual(g.nodeMeshById[previous]);
  }
  g.selectedId = nextId;
  if (nextId && g.nodeMeshById[nextId]) {
    const mesh = g.nodeMeshById[nextId];
    mesh.scale.setScalar(1.2);
    if (mesh.material && mesh.material.emissive) {
      mesh.material.emissive.setHex(0x33557f);
      mesh.material.emissiveIntensity = 0.6;
    }
  }
  updateGeometryDetails(g.selectedId);
}

function buildGeometryLegend(payload) {
  const details = qs("#geometryDetails");
  if (!details) return;
  if (!payload || !payload.available) {
    details.setAttribute("data-geometry-risk", "UNKNOWN");
    renderGeometryLegend(payload);
    return;
  }
  details.setAttribute("data-geometry-risk", normalizeRiskLevel(payload.metrics?.risk_level));
  renderGeometryLegend(payload);
}

function createNodeLabelSprite(three, text, colorHex = 0xd9e6ff) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  canvas.width = 256;
  canvas.height = 64;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(8, 16, 30, 0.72)";
  ctx.strokeStyle = "rgba(86, 119, 176, 0.86)";
  ctx.lineWidth = 2;
  const x = 4;
  const y = 8;
  const w = 248;
  const h = 48;
  const r = 8;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, "0")}`;
  ctx.font = "600 22px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(text || ""), canvas.width / 2, canvas.height / 2);
  const texture = new three.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new three.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new three.Sprite(material);
  sprite.scale.set(0.58, 0.145, 1);
  sprite.userData.pickMesh = false;
  return sprite;
}

function renderGeometryScene(payload, viewportDims) {
  const canvasHost = qs("#geometryViewport");
  const viewport = qs("#geometryViewport");
  if (!canvasHost || !viewport || !viewportDims) return;
  const { three, controlsCtor } = ensureThreeLibs();
  disposeGeometryRenderer();
  state.runGeometry = payload;
  buildGeometryLegend(payload);
  if (!payload || !payload.available) {
    disposeGeometryRenderer();
    setGeometrySurfaceState(payload?.reason || "Geometry unavailable.", "warn");
    updateGeometryDetails(null);
    return;
  }
  if (!Array.isArray(payload.nodes) || payload.nodes.length === 0) {
    disposeGeometryRenderer();
    setGeometrySurfaceState("No geometry nodes available in this result.", "warn");
    updateGeometryDetails(null);
    return;
  }

  if (typeof console !== "undefined" && console.debug) {
    console.debug("[geometry3d] build scene", {
      nodes: payload.nodes.length,
      edges: Array.isArray(payload.edges) ? payload.edges.length : 0,
      sampleNode: payload.nodes[0],
    });
  }

  showGeometryCanvas();
  const width = Math.max(240, viewportDims.width || viewport.clientWidth || 240);
  const height = Math.max(240, viewportDims.height || viewport.clientHeight || 360);
  const renderer = new three.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  if (three.SRGBColorSpace) {
    renderer.outputColorSpace = three.SRGBColorSpace;
  } else if (typeof renderer.outputEncoding !== "undefined" && typeof three.sRGBEncoding !== "undefined") {
    renderer.outputEncoding = three.sRGBEncoding;
  }
  renderer.domElement.className = "geometry-renderer-canvas";
  canvasHost.appendChild(renderer.domElement);

  const scene = new three.Scene();
  scene.background = null;
  scene.fog = new three.FogExp2(0x070d19, 0.07);

  const camera = new three.PerspectiveCamera(48, width / height, 0.01, 60);
  camera.position.set(0, 0.45, 3.05);
  scene.add(camera);

  const hemi = new three.HemisphereLight(0xa7c9ff, 0x0e1a30, 0.74);
  scene.add(hemi);
  const key = new three.DirectionalLight(0x8db5ff, 0.82);
  key.position.set(2.4, 3.6, 2.2);
  scene.add(key);
  const rim = new three.PointLight(0x5f8fde, 0.66, 8);
  rim.position.set(-2.4, -0.8, -1.4);
  scene.add(rim);

  const controls = new controlsCtor(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.7;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.7;
  controls.minDistance = 1.3;
  controls.maxDistance = 7.2;
  controls.target.set(0, 0, 0);

  const g = state.geometry3d;
  g.renderer = renderer;
  g.scene = scene;
  g.camera = camera;
  g.controls = controls;
  g.raycaster = new three.Raycaster();
  g.pointer = new three.Vector2();
  g.nodeMeshById = {};
  g.nodeDataById = {};
  g.nodeLabelById = {};
  g.nodeGlowById = {};
  g.unstablePulseById = {};
  g.selectedId = null;
  g.hoveredId = null;
  g.interactionEnabled = true;

  const riskColor = riskColorHex(payload.metrics?.risk_level);
  const glowMaterial = new three.MeshBasicMaterial({
    color: riskColor,
    transparent: true,
    opacity: 0.08,
    blending: three.AdditiveBlending,
    depthWrite: false,
  });
  const glow = new three.Mesh(new three.SphereGeometry(2.35, 32, 32), glowMaterial);
  scene.add(glow);

  const edgeGroup = new three.Group();
  const drift = Number(payload.metrics?.structural_drift_score || 0);
  const instability = Number(payload.metrics?.composite_instability || 0);
  const stressScale = Math.max(0, Math.min(1.2, 0.35 * drift + 0.45 * instability));
  const jitterAmp = 0.05 + stressScale * 0.07;

  payload.edges.forEach((edge) => {
    const source = payload.nodes.find((n) => n.id === edge.source);
    const target = payload.nodes.find((n) => n.id === edge.target);
    if (!source || !target) return;
    const p1 = new three.Vector3(Number(source.position.x), Number(source.position.y), Number(source.position.z));
    const p2 = new three.Vector3(Number(target.position.x), Number(target.position.y), Number(target.position.z));
    const edgeGeom = new three.BufferGeometry().setFromPoints([p1, p2]);
    const edgeMat = new three.LineBasicMaterial({
      color: edgeColorHex(edge, payload.metrics),
      transparent: true,
      opacity: 0.18 + 0.5 * Math.min(1, Math.abs(Number(edge.magnitude || 0))),
    });
    edgeGroup.add(new three.Line(edgeGeom, edgeMat));
  });
  scene.add(edgeGroup);
  g.edgeGroup = edgeGroup;

  const nodeGroup = new three.Group();
  payload.nodes.forEach((node) => {
    const magnitude = Math.max(0, Number(node.magnitude || 0));
    const radius = 0.045 + magnitude * 0.09;
    const geom = new three.SphereGeometry(radius, 24, 24);
    const mat = new three.MeshStandardMaterial({
      color: nodeStateColorHex(node.state),
      roughness: 0.35,
      metalness: 0.2,
      emissive: 0x0d1424,
      emissiveIntensity: 0.35,
    });
    const mesh = new three.Mesh(geom, mat);
    const px = Number(node.position?.x || 0);
    const py = Number(node.position?.y || 0);
    const pz = Number(node.position?.z || 0);
    mesh.position.set(px, py, pz);
    mesh.userData.nodeId = String(node.id);
    mesh.userData.pickMesh = true;
    mesh.userData.radius = radius;
    mesh.userData.basePos = new three.Vector3(px, py, pz);
    mesh.userData.jitterSeed = Math.random() * Math.PI * 2;
    nodeGroup.add(mesh);
    g.nodeMeshById[String(node.id)] = mesh;
    g.nodeDataById[String(node.id)] = node;

    if (node.is_unstable) {
      const unstableHalo = new three.Mesh(
        new three.SphereGeometry(radius * 2.05, 16, 16),
        new three.MeshBasicMaterial({
          color: 0xff7878,
          transparent: true,
          opacity: 0.24,
          blending: three.AdditiveBlending,
          depthWrite: false,
        })
      );
      unstableHalo.position.set(px, py, pz);
      unstableHalo.userData.nodeId = String(node.id);
      unstableHalo.userData.pickMesh = false;
      nodeGroup.add(unstableHalo);
      g.nodeGlowById[String(node.id)] = unstableHalo;
      g.unstablePulseById[String(node.id)] = 0.65 + Math.random() * 0.7;
    }

    if (payload.nodes.length <= 24) {
      const label = createNodeLabelSprite(three, node.label || node.id);
      if (label) {
        label.position.set(px, py + radius + 0.08, pz);
        nodeGroup.add(label);
        g.nodeLabelById[String(node.id)] = label;
      }
    }
  });
  scene.add(nodeGroup);
  g.nodeGroup = nodeGroup;
  applyGeometryDisplayMode();
  updateGeometryDetails(null);

  const grid = new three.GridHelper(5.8, 14, 0x2f476b, 0x182741);
  grid.position.y = -0.78;
  grid.material.opacity = 0.17;
  grid.material.transparent = true;
  scene.add(grid);

  function onResize() {
    if (!g.renderer || !g.camera) return;
    const dims = geometryViewportDimensions();
    const w = Math.max(240, dims?.width || viewport.clientWidth || 240);
    const h = Math.max(240, dims?.height || viewport.clientHeight || 360);
    g.renderer.setSize(w, h);
    g.camera.aspect = w / h;
    g.camera.updateProjectionMatrix();
  }

  if (window.ResizeObserver) {
    const ro = new window.ResizeObserver(() => onResize());
    ro.observe(canvasHost);
    g.resizeObserver = ro;
  } else {
    const handler = () => onResize();
    window.addEventListener("resize", handler);
    g.cleanupResize = () => window.removeEventListener("resize", handler);
  }

  let pointerDown = null;

  function updateHoverFromEvent(evt) {
    if (!g.interactionEnabled || !g.raycaster || !g.camera || !g.renderer || !g.scene || !g.nodeGroup) return;
    if (g.selectedId) return;
    if (pointerDown) return;
    g.scene.updateMatrixWorld(true);
    const rect = g.renderer.domElement.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    g.pointer.x = ((evt.clientX - rect.left) / w) * 2 - 1;
    g.pointer.y = -((evt.clientY - rect.top) / h) * 2 + 1;
    g.raycaster.setFromCamera(g.pointer, g.camera);
    const hits = g.raycaster.intersectObjects([g.nodeGroup], true);
    const hit = hits.find((h) => h.object && h.object.userData && h.object.userData.pickMesh === true);
    const nextHover = hit && hit.object.userData.nodeId ? String(hit.object.userData.nodeId) : null;
    if (nextHover === g.hoveredId) return;
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
    }
    g.hoveredId = nextHover;
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      const hm = g.nodeMeshById[g.hoveredId];
      if (hm.material && hm.material.emissive) {
        hm.material.emissive.setHex(0x2a4a66);
        hm.material.emissiveIntensity = 0.48;
      }
    }
    const tip = g.hoveredId && g.nodeDataById[g.hoveredId];
    g.renderer.domElement.title = tip ? String(tip.label || tip.id || g.hoveredId || "") : "";
  }

  function clearHover() {
    if (g.hoveredId && g.nodeMeshById[g.hoveredId]) {
      resetNodeMeshVisual(g.nodeMeshById[g.hoveredId]);
    }
    g.hoveredId = null;
    if (g.renderer && g.renderer.domElement) {
      g.renderer.domElement.title = "";
    }
  }

  function pickNodeFromEvent(evt) {
    if (!g.interactionEnabled || !g.raycaster || !g.camera || !g.renderer || !g.scene) return;
    if (!g.nodeGroup) return;
    g.scene.updateMatrixWorld(true);
    const rect = g.renderer.domElement.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    g.pointer.x = ((evt.clientX - rect.left) / w) * 2 - 1;
    g.pointer.y = -((evt.clientY - rect.top) / h) * 2 + 1;
    g.raycaster.setFromCamera(g.pointer, g.camera);
    const hits = g.raycaster.intersectObjects([g.nodeGroup], true);
    if (typeof console !== "undefined" && console.debug) {
      console.debug("[geometry3d] pick ray", {
        pointer: [g.pointer.x, g.pointer.y],
        hitCount: hits.length,
        firstPick: hits[0]?.object?.userData,
      });
    }
    const hit = hits.find((h) => h.object && h.object.userData && h.object.userData.pickMesh === true);
    const id = hit && hit.object.userData.nodeId ? String(hit.object.userData.nodeId) : null;
    if (typeof console !== "undefined" && console.debug) {
      console.debug("[geometry3d] pick result", { selectedId: id });
    }
    setGeometrySelection(id);
  }

  const onPointerDown = (evt) => {
    if (evt.button !== 0) return;
    pointerDown = { x: evt.clientX, y: evt.clientY };
  };
  const onPointerUp = (evt) => {
    if (evt.button !== 0 || !pointerDown) return;
    const dx = evt.clientX - pointerDown.x;
    const dy = evt.clientY - pointerDown.y;
    pointerDown = null;
    if (Math.hypot(dx, dy) > 10) return;
    pickNodeFromEvent(evt);
  };
  const onPointerMove = (evt) => {
    updateHoverFromEvent(evt);
  };
  const onPointerLeave = () => {
    clearHover();
    pointerDown = null;
  };

  const el = renderer.domElement;
  el.addEventListener("pointerdown", onPointerDown);
  el.addEventListener("pointerup", onPointerUp);
  el.addEventListener("pointermove", onPointerMove);
  el.addEventListener("pointerleave", onPointerLeave);
  g.cleanupPointer = () => {
    el.removeEventListener("pointerdown", onPointerDown);
    el.removeEventListener("pointerup", onPointerUp);
    el.removeEventListener("pointermove", onPointerMove);
    el.removeEventListener("pointerleave", onPointerLeave);
  };

  let t = 0;
  function animate() {
    g.frameId = window.requestAnimationFrame(animate);
    t += 0.016;
    if (nodeGroup && payload.metrics) {
      nodeGroup.children.forEach((obj) => {
        if (!obj.userData || !obj.userData.basePos || !obj.userData.nodeId) return;
        const nodeId = String(obj.userData.nodeId);
        const nodeData = g.nodeDataById[nodeId];
        if (!nodeData) return;
        const base = obj.userData.basePos;
        const localStress = Number(nodeData.stress || 0);
        const amp = jitterAmp * (0.3 + localStress);
        const phase = obj.userData.jitterSeed || 0;
        obj.position.x = base.x + Math.sin(t * 1.7 + phase) * amp * 0.12;
        obj.position.y = base.y + Math.cos(t * 1.3 + phase) * amp * 0.1;
        obj.position.z = base.z + Math.sin(t * 1.1 + phase) * amp * 0.12;
      });
      Object.entries(g.nodeGlowById || {}).forEach(([nodeId, halo]) => {
        const mesh = g.nodeMeshById[nodeId];
        if (!halo || !mesh) return;
        const k = Number(g.unstablePulseById[nodeId] || 1);
        const pulse = 1 + 0.12 * Math.sin(t * 2.4 + k);
        halo.scale.setScalar(pulse);
        halo.position.copy(mesh.position);
      });
    }
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

async function loadRunGeometry(runId, resultId = null) {
  const statusEl = qs("#geometry3dStatus");
  if (statusEl) {
    statusEl.textContent = "Loading geometry…";
    statusEl.className = "geometry-status info";
    statusEl.classList.remove("hidden");
  }
  const params = tenantScopeParams(resultId != null ? { result_id: resultId } : {});
  let payload;
  try {
    payload = await fetchJson(
      apiUrl(`/runs/${encodeURIComponent(runId)}/geometry`, params)
    );
  } catch (_err) {
    clearGeometryModelsPanel();
    if (statusEl) statusEl.classList.add("hidden");
    throw _err;
  }
  state.runGeometry = payload;
  renderGeometryModelsPanel(payload);
  const projectionNote =
    payload?.projection?.note ||
    "Structural projection metadata was not returned for this result.";
  const fallback = qs("#geometryFallback");
  const summary = qs("#geometryStructureSummary");
  if (fallback) {
    fallback.setAttribute("title", projectionNote);
    if (!payload?.available) {
      fallback.textContent = payload?.reason || "No relationship graph available for this snapshot.";
    }
  }
  if (summary) summary.textContent = geometryStructureSummary(payload);
  try {
    const dims = await ensureGeometryViewportReady();
    if (!dims || dims.width <= 40 || dims.height <= 40) {
      setGeometrySurfaceState("Viewport has no size yet. Resize the window or refresh.", "info");
      updateGeometryDetails(null);
      if (statusEl) statusEl.classList.add("hidden");
      return;
    }
    renderGeometryScene(payload, dims);
    if (statusEl) statusEl.classList.add("hidden");
  } catch (err) {
    setGeometrySurfaceState(`3D view failed: ${String(err.message || err)}`, "error");
    updateGeometryDetails(null);
    if (statusEl) statusEl.classList.add("hidden");
  }
}

function createToast(message, type = "success") {
  const container = qs("#toastContainer");
  if (!container || !message) return;
  const toast = document.createElement("div");
  const safeType = type === "error" ? "error" : "success";
  toast.className = `toast ${safeType}`;
  toast.textContent = String(message);
  container.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, 3200);
}

function setLoading(isLoading, message = "Loading...") {
  const overlay = qs("#loadingOverlay");
  const text = qs("#loadingMessage");
  if (!overlay || !text) return;
  if (isLoading) {
    text.textContent = String(message || "Loading...");
    overlay.classList.remove("hidden");
  } else {
    text.textContent = "Loading...";
    overlay.classList.add("hidden");
  }
}

function setStatus(message = "", isError = false, showToast = false) {
  const el = qs("#globalStatus");
  if (!el) return;
  if (!message) {
    el.className = "status hidden";
    el.textContent = "";
    return;
  }
  el.className = `status ${isError ? "error" : "ok"}`;
  el.textContent = String(message);
  el.classList.remove("hidden");
  if (showToast) {
    createToast(message, isError ? "error" : "success");
  }
}

function setPage(page) {
  const titles = {
    dashboard: ["Dashboard", "Active run snapshot"],
    runs: ["Runs", "Create, inspect, and activate runs"],
    upload: ["Upload", "Upload telemetry CSV into the active run"],
    "run-detail": ["Run Detail", "Deep inspection of run outputs"],
    "result-detail": ["Result Detail", "Focused view for a single result"],
    "demo-sii": ["SII tour", "Narrative walkthrough for demos and onboarding"],
  };
  qsa(".page").forEach((p) => p.classList.add("hidden"));
  const pageEl = qs(`#page-${page}`);
  if (pageEl) pageEl.classList.remove("hidden");
  const [title, subtitle] = titles[page] || ["Neraium", ""];
  const titleEl = qs("#pageTitle");
  const subtitleEl = qs("#pageSubtitle");
  if (titleEl) titleEl.textContent = title;
  if (subtitleEl) subtitleEl.textContent = subtitle;
  qsa(".nav a").forEach((a) => a.classList.remove("active"));
  if (page === "dashboard") qs('[data-nav="dashboard"]')?.classList.add("active");
  if (page === "runs" || page === "run-detail") qs('[data-nav="runs"]')?.classList.add("active");
  if (page === "upload") qs('[data-nav="upload"]')?.classList.add("active");
  if (page === "demo-sii") qs('[data-nav="demo-sii"]')?.classList.add("active");
}

function updateUploadRunInfo() {
  const info = qs("#uploadRunInfo");
  if (!info) return;
  if (state.activeRun?.run_id) {
    info.textContent = `Active run: ${state.activeRun.name} (${state.activeRun.run_id})`;
  } else {
    info.textContent = "No active run selected.";
  }
}

function clearUploadJobPolling() {
  if (state.uploadJob.pollTimer) {
    window.clearTimeout(state.uploadJob.pollTimer);
    state.uploadJob.pollTimer = null;
  }
}

function setUploadProgressUI({
  visible = false,
  mode = "uploading",
  statusText = "",
  uploadedBytes = 0,
  totalBytes = null,
  rowsProcessed = 0,
  rowsSucceeded = 0,
  rowsFailed = 0,
  errorSamples = [],
}) {
  const panel = qs("#uploadProgressPanel");
  const title = qs("#uploadProgressTitle");
  const percent = qs("#uploadProgressPercent");
  const bar = qs("#uploadProgressBar");
  const status = qs("#uploadProgressMessage");
  const rowsMeta = qs("#uploadProgressRows");
  const errors = qs("#uploadProgressErrors");

  if (panel) {
    if (visible) panel.classList.remove("hidden");
    else panel.classList.add("hidden");
  }
  if (status) status.textContent = String(statusText || "");

  const total = Number(totalBytes);
  const received = Number(uploadedBytes || 0);
  const uploadPctRaw =
    Number.isFinite(total) && total > 0
      ? Math.max(0, Math.min(100, (received / total) * 100))
      : 0;
  const uploadCompleteModes = new Set(["queued", "processing", "completed", "partial_success", "failed"]);
  const displayPct = uploadCompleteModes.has(mode) ? 100 : uploadPctRaw;
  if (bar) bar.style.width = `${displayPct}%`;
  if (percent) percent.textContent = `${Math.round(displayPct)}%`;

  if (title) {
    if (mode === "uploading") title.textContent = "Uploading CSV";
    else if (mode === "queued") title.textContent = "Queued for ingest";
    else if (mode === "processing") title.textContent = "Processing rows";
    else if (mode === "completed") title.textContent = "Ingest completed";
    else if (mode === "partial_success") title.textContent = "Ingest partial success";
    else if (mode === "failed") title.textContent = "Ingest failed";
    else title.textContent = "Ingest progress";
  }

  const processed = Number(rowsProcessed || 0);
  const succeeded = Number(rowsSucceeded || 0);
  const failed = Number(rowsFailed || 0);
  if (rowsMeta) {
    const uploadText =
      Number.isFinite(total) && total > 0
        ? `${formatBytes(received)} / ${formatBytes(total)} uploaded`
        : `${formatBytes(received)} uploaded`;
    rowsMeta.textContent = `${uploadText} · ${processed} processed · ${succeeded} succeeded · ${failed} failed`;
  }

  if (errors) {
    if (Array.isArray(errorSamples) && errorSamples.length > 0) {
      errors.innerHTML = errorSamples
        .slice(0, 4)
        .map((e) => `<li>Row ${escapeHtml(e.row)}: ${escapeHtml(e.message)}</li>`)
        .join("");
      errors.classList.remove("hidden");
    } else {
      errors.innerHTML = "";
      errors.classList.add("hidden");
    }
  }
}

async function uploadCsvFileWithProgress(file, runId) {
  const url = apiUrl("/ingest/csv/upload", tenantScopeParams({ run_id: runId }));
  const form = new FormData();
  form.append("file", file, file.name);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.responseType = "json";
    xhr.upload.onprogress = (evt) => {
      const loaded = Number(evt.loaded || 0);
      const total = evt.lengthComputable ? Number(evt.total || 0) : Number(file.size || 0);
      setUploadProgressUI({
        visible: true,
        mode: "uploading",
        statusText: "Uploading CSV...",
        uploadedBytes: loaded,
        totalBytes: total > 0 ? total : null,
        rowsProcessed: 0,
        rowsSucceeded: 0,
        rowsFailed: 0,
      });
    };
    xhr.onerror = () => {
      reject(new Error("Network error during CSV upload."));
    };
    xhr.onload = () => {
      const body = xhr.response || {};
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = body && body.detail ? String(body.detail) : `HTTP ${xhr.status}`;
        reject(new Error(detail));
        return;
      }
      resolve(body);
    };
    xhr.send(form);
  });
}

async function fetchIngestJob(jobId) {
  return fetchJson(apiUrl(`/ingest/jobs/${encodeURIComponent(jobId)}`, tenantScopeParams()));
}

async function waitForIngestJob(jobId) {
  state.uploadJob.id = jobId;
  state.uploadJob.active = true;
  clearUploadJobPolling();
  return new Promise((resolve, reject) => {
    const tick = async () => {
      if (!state.uploadJob.active) {
        reject(new Error("Upload monitoring cancelled."));
        return;
      }
      try {
        const job = await fetchIngestJob(jobId);
        const status = String(job.status || "processing");
        setUploadProgressUI({
          visible: true,
          mode: status,
          statusText: String(job.message || `Ingest status: ${status}`),
          uploadedBytes: Number(job.upload_bytes_received || 0),
          totalBytes: job.upload_bytes_total,
          rowsProcessed: Number(job.rows_processed || 0),
          rowsSucceeded: Number(job.rows_succeeded || 0),
          rowsFailed: Number(job.rows_failed || 0),
          errorSamples: job.error_samples || [],
        });
        if (status === "completed" || status === "partial_success" || status === "failed") {
          clearUploadJobPolling();
          state.uploadJob.active = false;
          resolve(job);
          return;
        }
      } catch (err) {
        clearUploadJobPolling();
        state.uploadJob.active = false;
        reject(err);
        return;
      }
      state.uploadJob.pollTimer = window.setTimeout(tick, 700);
    };
    tick();
  });
}

async function ensureActiveRun() {
  const active = await fetchJson(apiUrl("/runs/active", tenantScopeParams()));
  if (active.run) return active.run;
  const created = await fetchJson(apiUrl("/runs", tenantScopeParams()), {
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
  const nameEl = qs("#activeRunName");
  const idEl = qs("#activeRunId");
  const pillEl = qs("#activeRunStatusPill");
  if (nameEl) nameEl.textContent = run?.name || "No active run";
  if (idEl) idEl.textContent = run?.run_id || "-";
  if (pillEl) {
    if (!run) {
      pillEl.textContent = "—";
      pillEl.setAttribute("data-status", "none");
    } else if (run.is_active) {
      pillEl.textContent = "Live";
      pillEl.setAttribute("data-status", "live");
    } else {
      pillEl.textContent = "Idle";
      pillEl.setAttribute("data-status", "idle");
    }
  }
  if (run?.run_id) {
    window.localStorage.setItem("active_run_id", run.run_id);
  }
  updateUploadRunInfo();
}

function sortRuns(runs, mode) {
  const items = runs.slice();
  if (mode === "created_asc") {
    return items.sort((a, b) => parseTime(a.created_at) - parseTime(b.created_at));
  }
  if (mode === "name_asc") {
    return items.sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")));
  }
  if (mode === "name_desc") {
    return items.sort((a, b) => String(b.name || "").localeCompare(String(a.name || "")));
  }
  return items.sort((a, b) => parseTime(b.created_at) - parseTime(a.created_at));
}

function filteredSortedRuns() {
  const search = state.runsView.search.trim().toLowerCase();
  const status = state.runsView.status;
  const filtered = state.runs.filter((run) => {
    const name = String(run.name || "").toLowerCase();
    const runId = String(run.run_id || "").toLowerCase();
    const matchesSearch = !search || name.includes(search) || runId.includes(search);
    const matchesStatus =
      status === "all" ||
      (status === "active" && !!run.is_active) ||
      (status === "open" && !run.is_active && String(run.status || "").toLowerCase() === "open");
    return matchesSearch && matchesStatus;
  });
  return sortRuns(filtered, state.runsView.sort);
}

async function loadRuns() {
  const runsEnv = await fetchJson(apiUrl("/runs", tenantScopeParams({ limit: 500 })));
  state.runs = runsEnv.runs || [];
  collectKnownSites(state.runs);
  if (runsEnv.active_run) {
    updateActiveRunHeader(runsEnv.active_run);
  } else {
    updateActiveRunHeader(null);
  }
  renderTenantControls();
  renderRunsList();
}

function renderRunsList() {
  const tbody = qs("#runsBody");
  const empty = qs("#runsEmptyHint");
  if (!tbody) return;
  const runs = filteredSortedRuns();
  tbody.innerHTML = "";
  if (empty) {
    if (runs.length === 0) empty.classList.remove("hidden");
    else empty.classList.add("hidden");
  }
  runs.forEach((run) => {
    const statusText = run.is_active ? "active" : String(run.status || "open");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(run.name || "")}</td>
      <td class="mono">${escapeHtml(run.run_id || "")}</td>
      <td>${phaseBadgeHtml(statusText)}</td>
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
        const out = await fetchJson(
          apiUrl(`/runs/${encodeURIComponent(runId)}/activate`, tenantScopeParams()),
          { method: "POST" }
        );
        updateActiveRunHeader(out.run);
        await loadRuns();
        setStatus(`Activated run ${out.run.name}`, false, true);
      } catch (err) {
        setStatus(String(err.message || err), true, true);
      } finally {
        setLoading(false);
      }
    });
  });
}

function renderDashboardMetrics(latest, prev) {
  const metricTrend = qs("#metricTrend");
  const metricRisk = qs("#metricRisk");
  const metricState = qs("#metricState");
  const metricOperator = qs("#metricOperatorMessage");
  const metricRiskBadge = qs("#metricRiskBadge");
  const metricPhaseBadge = qs("#metricPhaseBadge");

  if (metricTrend) metricTrend.textContent = toPretty(trendFromResult(latest));
  if (metricRisk) metricRisk.textContent = toPretty(latest?.risk_level);
  if (metricState) metricState.textContent = toPretty(latest?.state || latest?.interpreted_state);
  const operatorSummary = demoFriendlyOperatorMessage(latest, prev);
  if (metricOperator) metricOperator.textContent = operatorSummary;
  if (metricRiskBadge) metricRiskBadge.innerHTML = riskBadgeHtml(latest?.risk_level);
  if (metricPhaseBadge) metricPhaseBadge.innerHTML = phaseBadgeHtml(phaseFromResult(latest));

  const riskLevel = normalizeRiskLevel(latest?.risk_level);
  if (metricRisk) metricRisk.setAttribute("data-risk", riskLevel);

  const stateValue = String(latest?.state || latest?.interpreted_state || "").toLowerCase();
  if (metricState) {
    let stateTone = "unknown";
    if (stateValue.includes("unstable") || stateValue.includes("alert") || riskLevel === "HIGH") {
      stateTone = "critical";
    } else if (stateValue.includes("watch") || stateValue.includes("drift") || riskLevel === "MEDIUM") {
      stateTone = "watch";
    } else if (stateValue.includes("stable") || stateValue.includes("nominal") || riskLevel === "LOW") {
      stateTone = "stable";
    }
    metricState.setAttribute("data-state", stateTone);
  }

  renderDashboardHero(latest, prev);
  const lu = qs("#dashboardLastUpdated");
  if (lu) {
    const ts = new Date();
    lu.textContent = `Last updated ${ts.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
  }
}

function renderDashboardRecent(results) {
  const tbody = qs("#dashboardRecentBody");
  const empty = qs("#dashboardEmpty");
  if (!tbody) return;
  tbody.innerHTML = "";
  const list = results || [];
  if (empty) {
    if (list.length === 0) empty.classList.remove("hidden");
    else empty.classList.add("hidden");
  }
  list.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${toPretty(r.result_id)}</td>
      <td>${toPretty(r.timestamp || r.persisted_at)}</td>
      <td>${phaseBadgeHtml(phaseFromResult(r))}</td>
      <td>${riskBadgeHtml(r.risk_level)}</td>
      <td>${toPretty(structuralDriftFromResult(r))}</td>
      <td>${toPretty(compositeInstabilityFromResult(r))}</td>
      <td><a href="/app/results/${encodeURIComponent(r.result_id)}?run_id=${encodeURIComponent(state.activeRun?.run_id || "")}&customer_id=${encodeURIComponent(customerIdValue(state.tenant.customerId))}">View</a></td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadDashboard() {
  const runId = state.activeRun?.run_id || "";
  const recentEnv = await fetchJson(apiUrl("/results/recent", tenantScopeParams({ run_id: runId, limit: 200 })));
  const alertsEnv = await fetchJson(
    apiUrl("/alerts", tenantScopeParams({ run_id: runId, limit: 20 }))
  );
  state.dashboardRecent = recentEnv.results || [];
  state.dashboardAlerts = alertsEnv.alerts || [];
  collectKnownSites(state.dashboardRecent);
  renderTenantControls();
  const chron = dashboardChronologicalResults();
  const latest = chron.length ? chron[chron.length - 1] : null;
  const prev = chron.length > 1 ? chron[chron.length - 2] : null;
  renderDashboardMetrics(latest, prev);
  renderDashboardRecent(state.dashboardRecent);
  renderDashboardSparkline(chron);
  bindDashboardSparklineInteractions();
  window.requestAnimationFrame(() => renderDashboardSparkline(dashboardChronologicalResults()));
}

function exportData(format, runId) {
  const url = apiUrl("/results/export", tenantScopeParams({ format, run_id: runId || "", limit: 500 }));
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

function setUploadFile(file) {
  state.uploadFile = file || null;
  const el = qs("#selectedFileName");
  if (!el) return;
  el.textContent = state.uploadFile ? `${state.uploadFile.name} (${state.uploadFile.size} bytes)` : "No file selected";
}

async function uploadCsvToActiveRun() {
  const fileInput = qs("#csvFileInput");
  const file = state.uploadFile || fileInput?.files?.[0];
  if (!file) throw new Error("Choose a CSV file first");
  const runId = state.activeRun?.run_id;
  if (!runId) throw new Error("No active run found");
  const started = await uploadCsvFileWithProgress(file, runId);
  const jobId = String(started.job_id || "");
  if (!jobId) {
    throw new Error("Upload started but did not return a job ID.");
  }
  return waitForIngestJob(jobId);
}

async function createRunFromForm() {
  const name = String(qs("#runNameInput")?.value || "").trim();
  const configRaw = String(qs("#runConfigInput")?.value || "").trim();
  const activate = Boolean(qs("#runActivateInput")?.checked);
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
  const out = await fetchJson(apiUrl("/runs", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config, activate }),
  });
  return out.run;
}

async function seedDemoData() {
  const run = state.activeRun || (await ensureActiveRun());
  updateActiveRunHeader(run);
  const runId = run.run_id;
  const now = Date.now();
  const items = [];
  for (let i = 0; i < 120; i += 1) {
    const t = new Date(now - (120 - i) * 60_000).toISOString();
    const driftFactor = i < 40 ? 0.2 : i < 80 ? 0.6 : 1.0;
    const wave = Math.sin(i / 6);
    items.push({
      timestamp: t,
      site_id: "demo-site",
      asset_id: "demo-asset",
      sensor_values: {
        pressure: 44 + wave * (1 + driftFactor * 0.7) + i * 0.025,
        flow: 28 + Math.cos(i / 7) * (1 + driftFactor * 0.4) + i * 0.015,
        vibration: 6 + Math.sin(i / 3) * (1 + driftFactor * 1.1) + driftFactor * 2.5,
        temperature: 61 + Math.cos(i / 5) * (1 + driftFactor * 0.5) + i * 0.02,
      },
    });
  }
  return fetchJson(apiUrl("/ingest/batch", tenantScopeParams({ run_id: runId })), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      items: items.map((item) => ({
        ...item,
        customer_id: customerIdValue(state.tenant.customerId),
      })),
    }),
  });
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
  const sharedOptions = buildTrendChartOptions();

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
            borderColor: "#79abff",
            backgroundColor: "rgba(106, 156, 250, 0.24)",
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 3,
          },
        ],
      },
      options: sharedOptions,
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
            borderColor: "#ffbf56",
            backgroundColor: "rgba(242, 179, 74, 0.2)",
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 3,
          },
        ],
      },
      options: sharedOptions,
    });
  }
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

function renderRunResultsTable(results) {
  const tbody = qs("#runResultsBody");
  const empty = qs("#runResultsEmpty");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (empty) {
    if (results.length === 0) empty.classList.remove("hidden");
    else empty.classList.add("hidden");
  }
  results.forEach((r, idx) => {
    const prev = idx + 1 < results.length ? results[idx + 1] : null;
    const transition = transitionLabel(prev, r);
    const severity = transitionSeverity(prev, r);
    const stateText = String(r.state || r.interpreted_state || "-");
    const tr = document.createElement("tr");
    tr.className = `result-row result-row-${severity}`;
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

function renderRunDetailFromState() {
  const hasResults = state.runRecent.length > 0;
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
    renderRunSignals(null, null);
    renderRunTransitionStrip(null, null);
    renderPhaseTimeline([]);
    renderOperatorMessages([]);
    renderDemoKeyEvents([]);
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
  const latest = chronological.length ? chronological[chronological.length - 1] : state.runRecent[0];
  const prev = chronological.length > 1 ? chronological[chronological.length - 2] : null;
  renderRunSignals(latest, prev);
  renderRunTransitionStrip(prev, latest);
  setDemoPlaybackUI();
  renderRunDetailCharts(ranged);
  renderPhaseTimeline(ranged);
  renderOperatorMessages(ranged, { emphasize: state.demo.enabled });
  const filtered = filterRunResults(ranged);
  const sorted = sortRunResults(filtered);
  renderRunResultsTable(sorted);
  renderRiskExplanation(latest, {
    panelSelector: "#runRiskExplanationPanel",
    titleSelector: "#runRiskExplanationTitle",
    bodySelector: "#runRiskExplanationText",
    badgeSelector: "#runRiskExplanationBadge",
  });
}

async function loadRunDetail(runId) {
  const runRes = await fetchJson(apiUrl(`/runs/${encodeURIComponent(runId)}`, tenantScopeParams()));
  const run = runRes.run;
  const title = qs("#runDetailTitle");
  const meta = qs("#runDetailMeta");
  if (title) title.textContent = `Run: ${run.name}`;
  if (meta) meta.textContent = `${run.run_id} · status=${run.status} · created=${run.created_at}`;
  const recentEnv = await fetchJson(apiUrl("/results/recent", tenantScopeParams({ run_id: runId, limit: 1000 })));
  state.runRecent = recentEnv.results || [];
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
  renderRunDetailFromState();
  maybeAutoStartDemoPlayback();
  const resultIdForGeometry =
    state.demo.enabled && state.runRecent.length > 0
      ? (state.runRecent
          .slice()
          .reverse()
          .slice(
            0,
            Math.max(1, Number(state.demo.cursor || state.runRecent.length))
          )
          .pop()?.result_id ?? null)
      : null;
  await loadRunGeometry(runId, resultIdForGeometry);

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

function wireUploadInteractions() {
  const fileInput = qs("#csvFileInput");
  const zone = qs("#uploadDropZone");
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      setUploadFile(f);
    });
  }
  if (!zone) return;

  const stop = (evt) => {
    evt.preventDefault();
    evt.stopPropagation();
  };

  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (evt) => {
      stop(evt);
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (evt) => {
      stop(evt);
      zone.classList.remove("dragging");
    });
  });
  zone.addEventListener("drop", (evt) => {
    const dt = evt.dataTransfer;
    const file = dt && dt.files && dt.files[0] ? dt.files[0] : null;
    if (!file) return;
    setUploadFile(file);
    // Keep native file input in sync when possible.
    if (fileInput && dt && dt.files) {
      try {
        fileInput.files = dt.files;
      } catch (_err) {
        // Ignore browser restrictions; upload uses state.uploadFile fallback.
      }
    }
  });
}

async function refreshCurrentPage() {
  const route = getRoute();
  if (route.page !== "run-detail") {
    disposeGeometryRenderer();
  }
  await loadRuns();
  if (route.page === "dashboard") await loadDashboard();
  if (route.page === "runs") renderRunsList();
  if (route.page === "upload") updateUploadRunInfo();
  if (route.page === "run-detail") await loadRunDetail(route.runId);
  if (route.page === "result-detail") await loadResultDetail(route.resultId);
}

const MOBILE_NAV_MQ = window.matchMedia("(max-width: 980px)");

function setMobileNavOpen(open) {
  const sidebar = qs("#appSidebar");
  const backdrop = qs("#navBackdrop");
  const toggle = qs("#mobileNavToggle");
  if (!sidebar || !backdrop || !toggle) return;
  if (open) {
    sidebar.classList.add("nav-drawer-open");
    backdrop.classList.remove("hidden");
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Close menu");
    document.body.classList.add("nav-mobile-open");
  } else {
    sidebar.classList.remove("nav-drawer-open");
    backdrop.classList.add("hidden");
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open menu");
    document.body.classList.remove("nav-mobile-open");
  }
}

function wireMobileNav() {
  const sidebar = qs("#appSidebar");
  const backdrop = qs("#navBackdrop");
  const toggle = qs("#mobileNavToggle");
  if (!sidebar || !backdrop || !toggle || toggle.dataset.wired === "1") return;
  toggle.dataset.wired = "1";

  toggle.addEventListener("click", () => {
    const open = !sidebar.classList.contains("nav-drawer-open");
    setMobileNavOpen(open);
  });

  backdrop.addEventListener("click", () => {
    setMobileNavOpen(false);
  });

  qsa(".nav a").forEach((a) => {
    a.addEventListener("click", () => {
      if (MOBILE_NAV_MQ.matches) setMobileNavOpen(false);
    });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sidebar.classList.contains("nav-drawer-open")) {
      setMobileNavOpen(false);
      toggle.focus();
    }
  });

  const onMq = () => {
    if (!MOBILE_NAV_MQ.matches) setMobileNavOpen(false);
  };
  if (typeof MOBILE_NAV_MQ.addEventListener === "function") {
    MOBILE_NAV_MQ.addEventListener("change", onMq);
  } else {
    MOBILE_NAV_MQ.addListener(onMq);
  }
}

async function wireEvents() {
  wireMobileNav();
  qs("#demoModeToggle")?.addEventListener("change", async (e) => {
    const enabled = Boolean(e.target?.checked);
    try {
      await toggleDemoMode(enabled);
      if (enabled && !state.demo.prepared && state.runs.length === 0) {
        setLoading(true, "Preparing demo runs...");
        const focusRun = await prepareDemoRuns();
        await refreshCurrentPage();
        if (focusRun?.run_id) {
          window.location.href = `/app/runs/${encodeURIComponent(focusRun.run_id)}?customer_id=${encodeURIComponent(customerIdValue(state.tenant.customerId))}`;
          return;
        }
      }
      setStatus(enabled ? "Demo Mode enabled" : "Demo Mode disabled", false, true);
    } catch (err) {
      setStatus(String(err.message || err), true, true);
    } finally {
      setLoading(false);
    }
  });

  qs("#prepareDemoBtn")?.addEventListener("click", async () => {
    try {
      setLoading(true, "Preparing realistic demo runs...");
      const focusRun = await prepareDemoRuns();
      await refreshCurrentPage();
      if (focusRun?.run_id) {
        setStatus(`Demo runs prepared. Opening ${focusRun.name}.`, false, true);
        window.location.href = `/app/runs/${encodeURIComponent(focusRun.run_id)}?customer_id=${encodeURIComponent(customerIdValue(state.tenant.customerId))}`;
        return;
      }
      setStatus("Demo runs prepared.", false, true);
    } catch (err) {
      setStatus(String(err.message || err), true, true);
    } finally {
      setLoading(false);
    }
  });

  qs("#demoPlayPauseBtn")?.addEventListener("click", () => {
    toggleDemoPlayback();
  });

  qs("#demoReplayBtn")?.addEventListener("click", () => {
    replayDemoTimeline();
  });

  qs("#refreshBtn")?.addEventListener("click", async () => {
    try {
      setLoading(true, "Refreshing...");
      await refreshCurrentPage();
      setStatus("Refreshed", false, true);
    } catch (err) {
      setStatus(String(err.message || err), true, true);
    } finally {
      setLoading(false);
    }
  });

  qs("#seedDemoBtn")?.addEventListener("click", async () => {
    try {
      setLoading(true, "Seeding demo data...");
      const out = await seedDemoData();
      await refreshCurrentPage();
      setStatus(`Demo data seeded (${out.count} rows processed)`, false, true);
    } catch (err) {
      setStatus(String(err.message || err), true, true);
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
      const nameInput = qs("#runNameInput");
      const configInput = qs("#runConfigInput");
      const activateInput = qs("#runActivateInput");
      if (nameInput) nameInput.value = "";
      if (configInput) configInput.value = "";
      if (activateInput) activateInput.checked = true;
      await loadRuns();
      setStatus(`Run created: ${run.name}`, false, true);
    } catch (err) {
      setStatus(String(err.message || err), true, true);
    } finally {
      setLoading(false);
    }
  });

  qs("#runsSearchInput")?.addEventListener("input", (e) => {
    state.runsView.search = String(e.target.value || "");
    renderRunsList();
  });
  qs("#runsStatusFilter")?.addEventListener("change", (e) => {
    state.runsView.status = String(e.target.value || "all");
    renderRunsList();
  });
  qs("#runsSortSelect")?.addEventListener("change", (e) => {
    state.runsView.sort = String(e.target.value || "created_desc");
    renderRunsList();
  });
  qs("#customerFilterInput")?.addEventListener("change", async () => {
    await applyTenantFromControls();
  });
  qs("#customerFilterInput")?.addEventListener("blur", async () => {
    await applyTenantFromControls();
  });
  qs("#siteFilterInput")?.addEventListener("change", async () => {
    await applyTenantFromControls();
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

  qs("#csvUploadForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      clearUploadJobPolling();
      state.uploadJob.active = true;
      setUploadProgressUI({
        visible: true,
        mode: "uploading",
        statusText: "Preparing upload...",
        uploadedBytes: 0,
        totalBytes: state.uploadFile?.size || null,
      });
      const out = await uploadCsvToActiveRun();
      const route = getRoute();
      await loadDashboard();
      if (route.page === "run-detail" && route.runId) {
        await loadRunDetail(route.runId);
      }
      const fileInput = qs("#csvFileInput");
      if (fileInput) fileInput.value = "";
      setUploadFile(null);
      const status = String(out.status || "completed");
      const rowsProcessed = Number(out.rows_processed || 0);
      const rowsSucceeded = Number(out.rows_succeeded || 0);
      const rowsFailed = Number(out.rows_failed || 0);
      const success = status === "completed";
      const partial = status === "partial_success";
      const failed = status === "failed";
      if (failed) {
        setStatus(
          out.message || `CSV ingest failed (${rowsFailed} rows failed).`,
          true,
          true
        );
      } else if (partial) {
        setStatus(
          out.message || `CSV ingest partial success (${rowsSucceeded} succeeded, ${rowsFailed} failed).`,
          true,
          true
        );
      } else {
        setStatus(
          out.message || `CSV ingested (${rowsProcessed} rows processed).`,
          false,
          true
        );
      }
      setUploadProgressUI({
        visible: true,
        mode: status,
        statusText: out.message || `Ingest ${status}.`,
        uploadedBytes: Number(out.upload_bytes_received || 0),
        totalBytes: out.upload_bytes_total,
        rowsProcessed,
        rowsSucceeded,
        rowsFailed,
        errorSamples: out.error_samples || [],
      });
    } catch (err) {
      setUploadProgressUI({
        visible: true,
        mode: "failed",
        statusText: String(err.message || err),
        errorSamples: [{ row: "-", message: String(err.message || err) }],
      });
      setStatus(String(err.message || err), true, true);
    } finally {
      state.uploadJob.active = false;
      clearUploadJobPolling();
    }
  });

  qs("#exportJsonBtn")?.addEventListener("click", () => exportData("json", state.activeRun?.run_id || ""));
  qs("#exportCsvBtn")?.addEventListener("click", () => exportData("csv", state.activeRun?.run_id || ""));

  qs("#demoTourEnableBtn")?.addEventListener("click", () => {
    const toggle = qs("#demoModeToggle");
    if (toggle) {
      toggle.checked = true;
      toggle.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (resizeTimer) window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (getRoute().page === "dashboard") {
        renderDashboardSparkline(dashboardChronologicalResults());
      }
    }, 150);
  });

  wireUploadInteractions();
}

async function init() {
  readTenantFromStorage();
  readDemoModeFromStorage();
  const routeScope = routeScopeFromQuery();
  if (routeScope.customer_id) {
    state.tenant.customerId = customerIdValue(routeScope.customer_id);
  }
  if (routeScope.site_id) {
    state.tenant.siteId = siteIdValue(routeScope.site_id);
  }
  renderTenantControls();
  const route = getRoute();
  const routeToPage = {
    dashboard: "dashboard",
    runs: "runs",
    upload: "upload",
    "run-detail": "run-detail",
    "result-detail": "result-detail",
    "demo-sii": "demo-sii",
  };
  setPage(routeToPage[route.page] || "dashboard");
  try {
    setLoading(true, "Initializing...");
    const activeRun = await ensureActiveRun();
    updateActiveRunHeader(activeRun);
    await loadRuns();
    if (route.page === "dashboard") await loadDashboard();
    if (route.page === "runs") renderRunsList();
    if (route.page === "upload") updateUploadRunInfo();
    if (route.page === "demo-sii") renderTenantControls();
    if (route.page === "run-detail") await loadRunDetail(route.runId);
    if (route.page === "result-detail") await loadResultDetail(route.resultId);
    await wireEvents();
    setStatus("");
  } catch (err) {
    setStatus(String(err.message || err), true, true);
  } finally {
    setLoading(false);
  }
}

init();
