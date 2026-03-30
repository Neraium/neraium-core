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
  const link = qs("#dashboardOpenAnalysisLink");
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
  const alertStatus = state.dashboardCurrentAlertStatus
    || (latest && latest.alert_status && typeof latest.alert_status === "object" ? latest.alert_status : null);
  const alertState = String(alertStatus?.state || alertStatus?.alert_state || "CLEAR").toUpperCase();
  const pendingCount = Number(alertStatus?.consecutive_hit_count || 0);
  const threshold = Number(alertStatus?.hit_window_threshold || 3);
  if (alertCountEl) {
    if (alertStatus && (alertStatus.alert_active || alertState === "PENDING_ALERT")) {
      alertCountEl.textContent = "1";
    } else {
      alertCountEl.textContent = String(alerts.length);
    }
  }
  if (alertSummaryEl) {
    if (alertStatus) {
      if (alertState === "PENDING_ALERT") {
        alertSummaryEl.textContent = `Pending alert (${pendingCount}/${threshold} confirmations)`.slice(0, 120);
      } else if (alertStatus.alert_active) {
        if (alertState === "ESCALATED") {
          alertSummaryEl.textContent = "Escalated alert".slice(0, 120);
        } else if (alertStatus.acknowledged) {
          alertSummaryEl.textContent = "Active alert — acknowledged".slice(0, 120);
        } else {
          alertSummaryEl.textContent = "Active alert — unacknowledged".slice(0, 120);
        }
      } else if (alertState === "RESOLVED") {
        alertSummaryEl.textContent = "Resolved after sustained recovery".slice(0, 120);
      } else {
        alertSummaryEl.textContent = "No open alerts for this run.";
      }
    } else {
      const first = alerts[0];
      alertSummaryEl.textContent = first
        ? String(first.message || first.type || "Alert").slice(0, 120)
        : "No open alerts for this run.";
    }
  }
  if (alertTile) {
    const critical = (alertStatus && alertState === "ESCALATED") || alerts.some((a) => String(a.severity || "").toLowerCase() === "critical");
    const high = (alertStatus && (alertStatus.alert_active || alertState === "PENDING_ALERT")) || alerts.some((a) => {
      const s = String(a.severity || "").toLowerCase();
      return s === "high" || s === "critical";
    });
    alertTile.classList.toggle("insight-tile-alert-critical", Boolean(critical));
    alertTile.classList.toggle("insight-tile-alert-watch", !critical && Boolean(high));
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
  const bar = qs("#demoPlaybackBar");
  const narr = qs("#demoPlaybackNarration");
  const badge = qs("#demoPlaybackBadge");
  if (!panel || !progress || !playPauseBtn || !replayBtn) return;
  const route = getRoute();
  const total = state.runRecent.length;
  const show = state.demo.enabled && route.page === "run-detail" && total > 0;
  if (!show) {
    panel.classList.add("hidden");
    progress.textContent = state.demo.enabled ? "Open a run to start replay" : "Replay mode off";
    if (badge) badge.textContent = "";
    if (bar) bar.style.width = "0%";
    if (narr) narr.textContent = "";
    playPauseBtn.textContent = "Play replay";
    replayBtn.disabled = true;
    renderDemoKeyEvents([]);
    return;
  }
  panel.classList.remove("hidden");
  const cursor = Math.max(1, Math.min(total, Number(state.demo.cursor || total)));
  progress.textContent = `Snapshot ${cursor}/${total}`;
  if (badge) badge.textContent = "Replay mode on";
  if (bar) bar.style.width = `${(cursor / Math.max(1, total)) * 100}%`;
  if (narr) narr.textContent = demoPlaybackNarrationText(cursor, total);
  playPauseBtn.textContent = state.demo.isPlaying ? "Pause replay" : "Play replay";
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
  const chronologicalFull = state.runRecent.slice().reverse();
  const chronological = chronologicalFull.slice(0, Math.max(1, Number(state.demo.cursor || chronologicalFull.length)));
  const latest = chronological.length ? chronological[chronological.length - 1] : null;
  const prev = chronological.length > 1 ? chronological[chronological.length - 2] : null;
  renderRunDetailFromState();
  const route = getRoute();
  if (
    route.page === "run-detail"
    && route.runId
    && state.runRecent.length > 0
    && state.ui.runDetailHydratedSections.geometry
  ) {
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
      const wasPlaying = state.demo.isPlaying;
      stopDemoPlayback();
      setDemoPlaybackUI();
      if (wasPlaying) onDemoPlaybackComplete();
      return;
    }
    state.demo.cursor += 1;
    applyDemoSnapshot();
    scheduleDemoTick();
  }, DEMO_PLAYBACK_INTERVAL_MS);
}

function toggleDemoPlayback(forcePlay = null) {
  if (!state.demo.enabled) {
    setStatus("Enable replay mode first.", true, true);
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
  state.demo.playbackCompleteNotified = false;
  state.demo.isPlaying = true;
  applyDemoSnapshot();
  scheduleDemoTick();
}

function replayDemoTimeline() {
  if (!state.demo.enabled || !state.runRecent.length) return;
  state.demo.playbackCompleteNotified = false;
  state.demo.cursor = 1;
  state.demo.isPlaying = true;
  applyDemoSnapshot();
  scheduleDemoTick();
}

async function toggleDemoMode(enabled) {
  state.demo.enabled = !!enabled;
  setConnectionStatus(getOperationalBadgeDisplay(buildFrontendUiState()));
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

async function ingestBatchForRun(runId, items, customerId) {
  const safeItems = Array.isArray(items) && items.length
    ? items
    : [
        {
          timestamp: new Date().toISOString(),
          site_id: "demo-site",
          asset_id: "demo-asset",
          sensor_values: { pressure: 42, flow: 27, vibration: 6.2, temperature: 61.5 },
        },
      ];
  const chunkSize = 20;
  let processed = 0;
  for (let i = 0; i < safeItems.length; i += chunkSize) {
    const chunk = safeItems.slice(i, i + chunkSize);
    try {
      const out = await fetchJson(apiUrl("/ingest/batch", tenantScopeParams({ run_id: runId })), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: chunk.map((item) => ({
            ...item,
            customer_id: customerId,
          })),
        }),
      });
      const maybeProcessed = Number(out?.processed);
      processed += Number.isFinite(maybeProcessed) && maybeProcessed > 0 ? maybeProcessed : chunk.length;
    } catch (err) {
      await ingestFramesForRun(runId, chunk, customerId);
      processed += chunk.length;
      if (typeof console !== "undefined" && console.warn) {
        console.warn("[demo] batch ingest chunk failed, falling back to /ingest/frame", err);
      }
    }
  }
  return { status: "ok", count: processed, processed, run_id: runId };
}

async function ingestFramesForRun(runId, items, customerId, options = {}) {
  const safeItems = Array.isArray(items) && items.length
    ? items
    : [
        {
          timestamp: new Date().toISOString(),
          site_id: "demo-site",
          asset_id: "demo-asset",
          sensor_values: { pressure: 42, flow: 27, vibration: 6.2, temperature: 61.5 },
        },
      ];
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  const onError = typeof options.onError === "function" ? options.onError : null;
  const total = safeItems.length;
  for (let i = 0; i < safeItems.length; i += 1) {
    const item = safeItems[i];
    try {
      await fetchJson(apiUrl("/ingest/frame", tenantScopeParams({ run_id: runId })), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...item,
          customer_id: customerId,
        }),
      });
    } catch (err) {
      const step = i + 1;
      if (onError) onError(err, step, total);
      throw new Error(`Demo ingest failed at frame ${step}/${total}: ${String(err.message || err)}`);
    }
    if (onProgress && (i === total - 1 || (i + 1) % 15 === 0)) {
      onProgress(i + 1, total);
    }
  }
  return { count: total, processed: total, run_id: runId };
}

async function startDemoSeedJob(runId, customerId, payload) {
  return fetchJson(apiUrl("/demo/seed/start", tenantScopeParams({ run_id: runId })), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      customer_id: customerId,
    }),
  });
}

async function startCmapssDemo(customerId, options = {}) {
  return fetchJson(apiUrl("/demo/cmapss/start", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      customer_id: customerId || null,
      max_frames: Number(options.max_frames || 10),
    }),
  });
}

async function getDemoSeedJobStatus(jobId) {
  return fetchJson(apiUrl("/demo/seed/status", tenantScopeParams({ job_id: jobId })), {
    method: "GET",
  });
}

async function waitForDemoSeedJob(jobId, options = {}) {
  const intervalMs = Math.max(300, Number(options.intervalMs || 800));
  const timeoutMs = Math.max(60000, Number(options.timeoutMs || 7 * 60 * 1000));
  const maxSilentPollFailures = Math.max(0, Number(options.maxSilentPollFailures || 3));
  const started = Date.now();
  let lastStatus = null;
  let pollFailures = 0;
  while (Date.now() - started < timeoutMs) {
    let status;
    try {
      status = await getDemoSeedJobStatus(jobId);
      pollFailures = 0;
    } catch (err) {
      pollFailures += 1;
      if (pollFailures > maxSilentPollFailures) {
        throw new Error("Demo failed — retry");
      }
      await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
      continue;
    }
    lastStatus = status;
    const stateLabel = String(status?.status || "").toLowerCase();
    if (options.onProgress) options.onProgress(status);
    if (stateLabel === "complete") return status;
    if (stateLabel === "error") {
      throw new Error(String(status?.error || status?.message || "Demo seed failed on server."));
    }
    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
  throw new Error(`Timed out waiting for demo seed job ${jobId}. Last status: ${JSON.stringify(lastStatus || {})}`);
}

function demoScenarioListForMode(mode) {
  const suffix = new Date().toISOString().slice(11, 16).replace(":", "");
  const all = [
    { name: `Demo Stable ${suffix}`, profile: "stable", siteId: "north-yard", assetId: "compressor-A" },
    { name: `Demo Watch ${suffix}`, profile: "watch", siteId: "north-yard", assetId: "compressor-B" },
    { name: `Demo Escalation ${suffix}`, profile: "critical", siteId: "south-yard", assetId: "compressor-C" },
  ];
  if (!mode || mode === "all") return all;
  const one = all.find((s) => s.profile === mode);
  return one ? [one] : all;
}

async function prepareDemoRuns(options = {}) {
  if (state.demo.preparing) return null;
  const mode = options.mode || "all";
  state.demo.preparing = true;
  renderTenantControls();
  try {
    const scenarios = demoScenarioListForMode(mode);
    const cust = customerIdValue(state.tenant.customerId);
    setDemoProgress({
      visible: true,
      phase: "Preparing historical validation replay",
      current: 0,
      total: scenarios.length,
      text: "Seeding telemetry and building structural state…",
    });
    const created = await Promise.all(
      scenarios.map(async (scenario, idx) => {
        setDemoProgress({
          visible: true,
          phase: "Streaming telemetry",
          current: idx,
          total: scenarios.length,
          text: `Seeding telemetry… (${idx}/${scenarios.length})`,
        });
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
        setLoading(true, "Preparing replay runs…");
        setStatus("Preparing reference replay run...", false);
        const started = await startDemoSeedJob(
          run.run_id,
          cust,
          {
            profile: scenario.profile,
            minutes: 120,
            site_id: scenario.siteId,
            asset_id: scenario.assetId,
          },
        );
        const jobId = String(started?.job_id || "");
        if (!jobId) {
          throw new Error("Demo seed did not return a job ID.");
        }
        const out = await waitForDemoSeedJob(jobId, {
          intervalMs: 900,
          onProgress: (job) => {
            const processed = Number(job?.processed || 0);
            const totalFrames = Number(job?.total_frames || 120);
            const percent = Number(job?.progress || 0);
            setStatus("Seeding telemetry on server...", false);
            setDemoProgress({
              visible: true,
              phase: "Streaming data",
              current: Math.min(totalFrames, processed || Math.round((totalFrames * percent) / 100)),
              total: totalFrames,
              text: `Seeding telemetry on server... (${Math.max(0, Math.min(100, percent))}%)`,
            });
          },
        });
        setDemoProgress({
          visible: true,
          phase: "Rendering model",
          current: idx + 1,
          total: scenarios.length,
          text: "Rendering SII Structural View...",
        });
        setStatus("Loading structural visualization...", false);
        return run;
      }),
    );
    const map = {};
    created.forEach((run) => {
      const prof = run?.config?.scenario;
      if (prof === "stable" || prof === "watch" || prof === "critical") {
        map[prof] = run;
      }
    });
    state.demo.scenarioRunMap = { ...(state.demo.scenarioRunMap || {}), ...map };
    let focusRun = null;
    if (mode === "all") {
      focusRun = created[created.length - 1] || null;
    } else {
      focusRun = created[0] || null;
    }
    if (!focusRun && created.length) {
      focusRun = created[created.length - 1];
    }
    if (focusRun?.run_id) {
      await fetchJson(apiUrl(`/runs/${encodeURIComponent(focusRun.run_id)}/activate`, tenantScopeParams()), {
        method: "POST",
      });
      state.demo.activeRunId = focusRun.run_id;
    }
    state.demo.prepared = true;
    setDemoProgress({ visible: true, phase: "Ready", current: scenarios.length, total: scenarios.length, text: "Reference replay ready." });
    setStatus("Reference replay ready.");
    window.setTimeout(() => setDemoProgress({ visible: false }), 1200);
    return focusRun;
  } finally {
    state.demo.preparing = false;
    if (!state.demo.prepared) setDemoProgress({ visible: false });
    renderTenantControls();
  }
}

function shouldShowDashboardDemoHero() {
  if (state.demo.preparing) return false;
  const n = (state.dashboardRecent || []).length;
  return n === 0;
}

function renderDashboardDemoHero() {
  const el = qs("#dashboardDemoHero");
  if (!el) return;
  el.classList.toggle("hidden", !shouldShowDashboardDemoHero());
}

function onDemoPlaybackComplete() {
  if (state.demo.playbackCompleteNotified) return;
  state.demo.playbackCompleteNotified = true;
  createToast("Ready to test your data? Opening upload with the drop zone highlighted.", "success");
  try {
    const cid = encodeURIComponent(customerIdValue(state.tenant.customerId));
    window.location.href = `/upload?customer_id=${cid}&replay=1&highlight=upload`;
  } catch (_e) {
    // no-op
  }
}

function demoPlaybackNarrationText(cursor, total) {
  if (!total || total < 2) return "";
  const t = Math.max(1, Math.min(total, Number(cursor) || 1));
  const p = (t - 1) / Math.max(1, total - 1);
  if (p < 0.22) return "Baseline window — drift is still largely contained.";
  if (p < 0.45) return "Drift rising — instability compounds across snapshots.";
  if (p < 0.78) return "Risk threshold approaching — watch transitions on the timeline.";
  return "Late-stage escalation — compare operator messages with structural drift.";
}

const RUN_DETAIL_DEMO_HERO_KEY = "neraium_demo_run_detail_hero_dismissed";

function shouldShowRunDetailDemoHero() {
  try {
    if (!state.demo.enabled) return false;
    const p = new URLSearchParams(window.location.search);
    if (p.get("hero") !== "1") return false;
    if (p.get("nohero") === "1") return false;
    if (window.sessionStorage.getItem(RUN_DETAIL_DEMO_HERO_KEY) === "1") return false;
    return true;
  } catch (_e) {
    return false;
  }
}

function dismissRunDetailDemoHero() {
  try {
    window.sessionStorage.setItem(RUN_DETAIL_DEMO_HERO_KEY, "1");
  } catch (_e) {
    // no-op
  }
  const hero = qs("#runDetailDemoHero");
  if (hero) hero.classList.add("hidden");
}

function showRunDetailDemoHero() {
  const hero = qs("#runDetailDemoHero");
  if (!hero) return false;
  hero.classList.remove("hidden");
  return true;
}

function wireRunDetailDemoHero() {
  const play = qs("#runDetailDemoHeroPlayBtn");
  const dismiss = qs("#runDetailDemoHeroDismissBtn");
  if (play && play.dataset.wired !== "1") {
    play.dataset.wired = "1";
    play.addEventListener("click", () => {
      dismissRunDetailDemoHero();
      replayDemoTimeline();
    });
  }
  if (dismiss && dismiss.dataset.wired !== "1") {
    dismiss.dataset.wired = "1";
    dismiss.addEventListener("click", () => dismissRunDetailDemoHero());
  }
}

async function launchGuidedDemo({ mode = "all" } = {}) {
  try {
    setLoading(true, "Loading validation scenario…");
    await toggleDemoMode(true);
    await prepareDemoRuns({ mode });
    await refreshCurrentPage();
    const focusRun = state.activeRun;
    state.demo.playbackCompleteNotified = false;
    if (focusRun?.run_id) {
      const cid = encodeURIComponent(customerIdValue(state.tenant.customerId));
      window.location.href = `/app/runs/${encodeURIComponent(focusRun.run_id)}?customer_id=${cid}&replay=1&autoplay=1`;
      return;
    }
    setStatus("Reference replay runs ready — select a run.", false, true);
  } catch (err) {
    setStatus(String(err.message || err), true, true);
  } finally {
    setLoading(false);
  }
}

function getRoute() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length === 0 || parts[0] === "dashboard") return { page: "dashboard" };
  if (parts[0] === "validation" || parts[0] === "reference" || parts[0] === "historical-validation") return { page: "validation" };
  if (parts[0] === "upload") return { page: "upload" };
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
  /** CSV semantic mapping (preview + user overrides before ingest). */
  uploadCsv: {
    stage: "idle",
    preview: null,
    headers: [],
    issues: [],
    warnings: [],
    requiresConfirmation: false,
    mapping: null,
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
    flowPlaybackMode: "live",
    flowPlaybackSpeed: 1,
    flowHistoryEnabled: true,
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
    perfMode: true,
    curveSegments: 9,
    motionScale: 1,
    useCanvas2d: false,
    canvas2d: null,
    flowCoherence: 0.75,
    /** Set by 2D structural-flow field model (blended coherence); used for panel + summary when present. */
    structuralFlowCoherence01: null,
    /** Full field-derived interpretation (2D); null when only payload-level fallback applies. */
    structuralFlowDerived: null,
    flowDriftN: 0,
    flowInstN: 0,
    flowEdgeScratch: null,
    pendingHoverEvt: null,
    hoverRaf: null,
    resizeRaf: null,
    _2dT: 0,
    keyLight: null,
    groundMesh: null,
    structuralFieldMesh: null,
    sensorDatumShell: null,
    flowUseLine2: false,
    structuralFlowPlaneLayout: false,
    flowInstancedMesh: null,
    flowDummy: null,
    flowParticlesPerEdge: 3,
    geometryRig: null,
    geometryIntroStart: null,
    geometryIntroMs: 0,
    reducedGeometryMotion: false,
    keyLightBasePos: null,
    temporalFlow: {
      points: [],
      localTimeSec: 0,
      replayHoldSec: 0,
      latestResultId: null,
      historyCentroid: [],
      historyDriftTip: [],
      historyContact: [],
      contactPersistence: 0,
      lastTrendCursor: -1,
    },
  },
  demo: {
    enabled: false,
    prepared: false,
    preparing: false,
    seedJobId: "",
    seedRunId: "",
    isPlaying: false,
    timer: null,
    cursor: 0,
    keyEvents: [],
    activeRunId: "",
    /** stable | watch | critical -> run_id after prepare */
    scenarioRunMap: {},
    playbackCompleteNotified: false,
    replay: {
      uiState: "idle",
      runId: "",
      pollTimer: null,
      pollFailures: 0,
      pollBackoffMs: 900,
      startingSinceMs: 0,
      errorMessage: "",
      launchInFlight: false,
      launchPromise: null,
    },
  },
  dashboardSparkline: {
    hoveredIndex: null,
  },
  ui: {
    clockTimer: null,
    connection: "LIVE",
    dashboardPaint: null,
    runDetailObserver: null,
    runDetailHydratedSections: {},
    runDetailDeferredPaint: null,
    runDetailBackgroundHistoryPending: false,
    runDetailBackgroundHistoryLoaded: false,
    loadRunsPromise: null,
    loadDashboardPromise: null,
    recentResultsCache: new Map(),
    recentResultsInflight: new Map(),
  },
  runtimeDegraded: false,
};

const TENANT_STORAGE_KEY = "neraium_customer_id";
const DEMO_MODE_STORAGE_KEY = "neraium_demo_mode";
/** Demo timeline: advance one snapshot per interval (tunable; lower = faster review). */
const DEMO_PLAYBACK_INTERVAL_MS = 1600;
/** How often to poll `/ingest/jobs/{id}` after CSV upload (lower = snappier status UI). */
const INGEST_JOB_POLL_MS = 400;
/** Replay launch/status polling cadence + resilience controls. */
const DEMO_REPLAY_INITIAL_POLL_MS = 900;
const DEMO_REPLAY_MAX_POLL_MS = 8000;
const DEMO_REPLAY_STARTING_TIMEOUT_MS = 45000;
const DEMO_REPLAY_MAX_TRANSIENT_ERRORS = 4;
const DEMO_UI_STATES = Object.freeze({
  idle: "idle",
  starting: "starting",
  running: "running",
  offline: "offline",
  interrupted: "interrupted",
  failed: "failed",
  completed: "completed",
});
/** Default on: lighter WebGL + simpler motion. Set localStorage "neraium_structural_flow_perf" to "0" for richer visuals. */
const GEOMETRY_FLOW_PERF_KEY = "neraium_structural_flow_perf";
/** Origin marker + debug visuals for structural flow. `true` always shows the marker; when `false`, use URL `?geomDebug=1` instead. */
const DEBUG_GEOMETRY = false;

const DASHBOARD_RECENT_LIMIT = 1;
const RUN_DETAIL_INITIAL_LIMIT = 24;
const RUN_DETAIL_BACKGROUND_LIMIT = 1000;
let chartJsLoadPromise = null;

/** Demo/sample ingest: 24 correlated channels for a dense structural flow field. */
const DEMO_STRUCTURAL_SENSOR_KEYS = [
  "pressure",
  "flow",
  "vibration",
  "temperature",
  "motor_current",
  "bearing_temp",
  "load_cell",
  "rpm",
  "humidity",
  "displacement",
  "valve_position",
  "shaft_accel",
  "lubrication_psi",
  "seismic_x",
  "seismic_y",
  "winding_temp",
  "inlet_guide",
  "outlet_guide",
  "torque_est",
  "casing_vibe",
  "oil_quality",
  "stator_temp",
  "field_bus_ok",
  "coolant_flow",
];

function buildDemoSensorValuesRow(i, p, driftLift, vibSpike) {
  const o = {};
  DEMO_STRUCTURAL_SENSOR_KEYS.forEach((key, k) => {
    const phase = k * 0.85;
    const wave = Math.sin(i / (5.2 + k * 0.11) + phase);
    const w2 = Math.cos(i / (7.1 + k * 0.09) + phase * 0.65);
    const base = 18 + k * 6.2;
    o[key] =
      base +
      wave * (1 + driftLift * (0.45 + k * 0.025)) +
      w2 * (0.55 + vibSpike * 0.12) +
      i * (0.011 + k * 0.0008) +
      p * (0.15 + k * 0.02);
  });
  return o;
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

function setConnectionStatus(mode = "LIVE") {
  const badge = qs("#connectionBadge");
  const label = qs("#connectionLabel");
  const normalized = String(mode || "NO DATA INGESTED").toUpperCase();
  state.ui.connection = normalized;
  if (label) label.textContent = normalized;
  if (badge) {
    badge.classList.remove("chip-live", "chip-demo", "chip-offline");
    if (["ALERT ACTIVE", "ANALYSIS INTERRUPTED", "REPLAY INTERRUPTED"].includes(normalized)) {
      badge.classList.add("chip-offline");
    } else if (normalized === "ACTIVE MONITORING") {
      badge.classList.add("chip-live");
    } else {
      badge.classList.add("chip-demo");
    }
  }
}

function setStreamingIndicator(active, text = "Streaming") {
  const badge = qs("#streamingBadge");
  if (!badge) return;
  badge.textContent = text;
  badge.classList.toggle("hidden", !active);
}

function startLiveClock() {
  const el = qs("#liveTimestamp");
  if (!el) return;
  if (state.ui.clockTimer) window.clearInterval(state.ui.clockTimer);
  const tick = () => {
    el.textContent = new Date().toISOString().slice(11, 19);
  };
  tick();
  state.ui.clockTimer = window.setInterval(tick, 1000);
}

function setDemoProgress({ visible = false, phase = "Initializing run", current = 0, total = 0, text = "" } = {}) {
  const panel = qs("#demoProgressPanel");
  const phaseEl = qs("#demoProgressPhase");
  const countEl = qs("#demoProgressCount");
  const fillEl = qs("#demoProgressFill");
  const textEl = qs("#demoProgressText");
  if (panel) panel.classList.toggle("hidden", !visible);
  if (phaseEl) phaseEl.textContent = phase;
  if (countEl) countEl.textContent = `${current}/${total}`;
  const pct = total > 0 ? Math.max(0, Math.min(100, (current / total) * 100)) : 0;
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (textEl) textEl.textContent = text || `${phase}… (${current}/${total})`;
  setStreamingIndicator(visible, phase.includes("Streaming") ? "Telemetry live" : "SII preparing");
}

function normalizeLoadingMessage(message) {
  const msg = String(message || "Loading...");
  const lower = msg.toLowerCase();
  if (lower.includes("seed")) return "Processing NASA CMAPSS dataset...";
  if (lower.includes("demo")) return "Preparing historical validation replay...";
  if (lower.includes("cmapss")) return "Processing NASA CMAPSS dataset...";
  if (lower.includes("structural visualization") || lower.includes("geometry")) return "Rendering SII Structural View...";
  if (lower.includes("refresh")) return "Refreshing Systemic Infrastructure Intelligence state...";
  if (lower.includes("initial")) return "Initializing Systemic Infrastructure Intelligence workspace...";
  return msg;
}

function setLoading(isLoading, message = "Loading...") {
  const overlay = qs("#loadingOverlay");
  const text = qs("#loadingMessage");
  if (!overlay || !text) return;
  if (isLoading) {
    const normalized = normalizeLoadingMessage(message);
    text.textContent = normalized;
    overlay.classList.remove("hidden");
    if (normalized.toLowerCase().includes("processing nasa cmapss") || normalized.toLowerCase().includes("demo")) {
      setStreamingIndicator(true, "Telemetry live");
    }
  } else {
    text.textContent = "Loading...";
    overlay.classList.add("hidden");
    setStreamingIndicator(false);
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
  const route = getRoute();
  const isValidationPage = String(route?.page || "").toLowerCase() === "validation";
  const uiTruth = buildFrontendUiState(null, { analysisInterrupted: isError && !isValidationPage });
  const cleanMessage = isError ? friendlyErrorMessage(message, getErrorDisplayContext(uiTruth, message)) : String(message);
  el.className = `status ${isError ? "error" : "ok"}`;
  el.textContent = cleanMessage;
  el.classList.remove("hidden");
  setConnectionStatus(getOperationalBadgeDisplay(uiTruth));
  if (showToast && !isError) {
    createToast(cleanMessage, isError ? "error" : "success");
  }
}



async function refreshRuntimeModeBanner() {
  const banner = qs("#runtimeModeBanner");
  if (!banner) return;
  try {
    const health = await fetchJson(apiUrl("/health"));
    const degraded = Boolean(health?.core_runtime_fallback) || String(health?.status || "").toLowerCase() === "degraded";
    state.runtimeDegraded = degraded;
    if (!degraded) {
      banner.classList.add("hidden");
      banner.textContent = "";
      return;
    }
    const notes = Array.isArray(health?.core_runtime_notes) ? health.core_runtime_notes.filter(Boolean) : [];
    const noteText = notes.length ? ` ${notes.slice(0, 2).join(" ")}` : "";
    banner.textContent = `Degraded runtime mode: fallback core modules are active. Pilot decisions should be treated as limited-confidence until full core runtime is restored.${noteText}`;
    banner.classList.remove("hidden");
  } catch (_err) {
    state.runtimeDegraded = false;
    banner.classList.add("hidden");
  }
}
function setPage(page) {
  const titles = {
    dashboard: ["Dashboard", "Current state"],
    upload: ["Upload", "Ingest telemetry"],
    runs: ["Runs", "Select a run"],
    validation: ["Validation", "Replay workspace"],
    "run-detail": ["Run Analysis", "Analyze this system"],
    "result-detail": ["Result Detail", "Focused view for a single result"],
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
  if (page === "upload") qs('[data-nav="upload"]')?.classList.add("active");
  if (page === "runs" || page === "run-detail") qs('[data-nav="runs"]')?.classList.add("active");
  if (page === "validation") qs('[data-nav="validation"]')?.classList.add("active");
}

function activateAnalysisWorkspaceTab(_tabName = "executive") {
  // Workspace tabs removed in coherence cleanup; retained as compatibility no-op.
}

function initAnalysisWorkspaceTabs() {
  // Workspace tabs removed in coherence cleanup; retained as compatibility no-op.
}

function renderRunDetailHeaderContext(run, latest) {
  const stateEl = qs("#runStickyState");
  const riskEl = qs("#runStickyRisk");
  const alertEl = qs("#runStickyAlert");
  const recEl = qs("#runStickyRecommendation");
  const updateEl = qs("#runStickyUpdated");
  const uiTruth = buildFrontendUiState(latest);
  const risk = normalizeRiskLevel(latest?.risk_level);
  const currentState = String(latest?.state || latest?.interpreted_state || "UNKNOWN");
  const alertStatus = latest?.alert_status || state.dashboardCurrentAlertStatus || {};
  const alertState = String(alertStatus?.state || alertStatus?.alert_state || "CLEAR").toUpperCase();
  const recommendation = String(latest?.operator_message || "").trim();
  const ts = latest?.timestamp || latest?.persisted_at || latest?.created_at || "";
  if (stateEl) stateEl.textContent = currentState;
  if (riskEl) riskEl.textContent = risk;
  if (alertEl) alertEl.textContent = alertState;
  if (updateEl) updateEl.textContent = getLastUpdateDisplay(uiTruth, ts);
  if (recEl) {
    if (!latest) recEl.textContent = "Upload telemetry.";
    else if (risk === "HIGH") recEl.textContent = "Acknowledge and inspect.";
    else if (risk === "MEDIUM") recEl.textContent = "Maintain watch.";
    else recEl.textContent = recommendation || "Continue monitoring.";
  }
  const overviewMap = [
    ["#runOverviewState", currentState],
    ["#runOverviewRisk", risk],
    ["#runOverviewTrend", String(trendFromResult(latest) || "-")],
    ["#runOverviewAlert", alertState],
    ["#runOverviewRecommendation", recEl?.textContent || "Upload telemetry."],
    ["#runOverviewUpdated", getLastUpdateDisplay(uiTruth, ts)],
  ];
  overviewMap.forEach(([selector, value]) => {
    const el = qs(selector);
    if (el) el.textContent = value;
  });
}

function wireWorkspaceShellEvents() {
  const refreshBtn = qs("#refreshBtn");
  if (refreshBtn && refreshBtn.dataset.wired !== "1") {
    refreshBtn.dataset.wired = "1";
    refreshBtn.addEventListener("click", async () => {
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
  }
  if (window.__neraiumDashboardResizeBound !== true) {
    window.__neraiumDashboardResizeBound = true;
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (resizeTimer) window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        if (getRoute().page === "dashboard") {
          renderDashboardSparkline(dashboardChronologicalResults());
        }
      }, 150);
    });
  }
}

function wireRunsEvents() {
  const runCreateForm = qs("#runCreateForm");
  if (runCreateForm && runCreateForm.dataset.wired !== "1") {
    runCreateForm.dataset.wired = "1";
    runCreateForm.addEventListener("submit", async (e) => {
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
  }
  const searchInput = qs("#runsSearchInput");
  if (searchInput && searchInput.dataset.wired !== "1") {
    searchInput.dataset.wired = "1";
    searchInput.addEventListener("input", (e) => {
      state.runsView.search = String(e.target.value || "");
      renderRunsList();
    });
  }
  const statusFilter = qs("#runsStatusFilter");
  if (statusFilter && statusFilter.dataset.wired !== "1") {
    statusFilter.dataset.wired = "1";
    statusFilter.addEventListener("change", (e) => {
      state.runsView.status = String(e.target.value || "all");
      renderRunsList();
    });
  }
  const sortSelect = qs("#runsSortSelect");
  if (sortSelect && sortSelect.dataset.wired !== "1") {
    sortSelect.dataset.wired = "1";
    sortSelect.addEventListener("change", (e) => {
      state.runsView.sort = String(e.target.value || "created_desc");
      renderRunsList();
    });
  }
}

function updateUploadRunInfo() {
  const info = qs("#uploadRunInfo");
  if (!info) return;
  if (state.activeRun?.run_id) {
    const base = `Active run: ${state.activeRun.name} (${state.activeRun.run_id})`;
    info.textContent = base;
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

function resetUploadPanelIfIdle() {
  if (state.uploadJob.active) return;
  setUploadProgressUI({ visible: false });
}

async function uploadCsvFileWithProgress(file, runId, columnMapping = null) {
  const url = apiUrl("/ingest/csv/upload", tenantScopeParams({ run_id: runId }));
  const form = new FormData();
  form.append("file", file, file.name);
  if (columnMapping && typeof columnMapping === "object") {
    form.append("mapping", JSON.stringify(columnMapping));
  }
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
      state.uploadJob.pollTimer = window.setTimeout(tick, INGEST_JOB_POLL_MS);
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
  const topbarRunId = qs("#topbarActiveRunId");
  const normalizedTopbarRunId = String(run?.run_id || "").replace(/^run_/i, "");
  if (nameEl) nameEl.textContent = run?.name || "No active run";
  if (idEl) idEl.textContent = run?.run_id || "-";
  if (topbarRunId) topbarRunId.textContent = normalizedTopbarRunId || "--";
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
  setConnectionStatus(getOperationalBadgeDisplay(buildFrontendUiState()));
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
  if (state.ui.loadRunsPromise) return state.ui.loadRunsPromise;
  state.ui.loadRunsPromise = (async () => {
  const runsEnv = await fetchJson(apiUrl("/runs", tenantScopeParams({ limit: 500 })));
  state.runs = runsEnv.runs || [];
  collectKnownSites(state.runs);
  if (runsEnv.active_run) {
    updateActiveRunHeader(runsEnv.active_run);
  } else if (state.runs.length > 0) {
    updateActiveRunHeader(state.runs.find((run) => run.is_active) || null);
  } else {
    const created = await ensureActiveRun();
    updateActiveRunHeader(created || null);
  }
  renderTenantControls();
  renderRunsList();
  })();
  try {
    await state.ui.loadRunsPromise;
  } finally {
    state.ui.loadRunsPromise = null;
  }
}

function renderRunsList() {
  const tbody = qs("#runsBody");
  const empty = qs("#runsEmptyHint");
  if (!tbody) return;
  const runs = filteredSortedRuns();
  tbody.innerHTML = "";
  if (empty) {
    if (runs.length === 0) {
      empty.classList.remove("hidden");
      const p = empty.querySelector("p");
      if (p) {
        p.textContent =
          state.runs.length === 0
            ? "No runs yet. Create an active run, then upload telemetry to begin pilot operations."
            : "No runs match your filters.";
      }
    } else empty.classList.add("hidden");
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

function latestTelemetryTimestampMs(result) {
  if (!result) return 0;
  const raw = result.timestamp || result.persisted_at || result.created_at || null;
  if (!raw) return 0;
  const ms = Date.parse(String(raw));
  return Number.isFinite(ms) ? ms : 0;
}

function formatFreshnessLabel(result, uiTruth = null) {
  const truth = uiTruth || buildFrontendUiState(result);
  const label = getLastUpdateDisplay(truth, result?.timestamp || result?.persisted_at || result?.created_at || "");
  if (truth.mode === "validation") return { label, stale: false };
  return { label, stale: label.toLowerCase().includes("stale") || label.toLowerCase().includes("no telemetry") };
}

function normalizedAlertStatusText(latest) {
  const alertStatus = state.dashboardCurrentAlertStatus
    || (latest && latest.alert_status && typeof latest.alert_status === "object" ? latest.alert_status : null);
  const alertState = String(alertStatus?.state || alertStatus?.alert_state || "CLEAR").toUpperCase();
  const pendingCount = Number(alertStatus?.consecutive_hit_count || 0);
  const threshold = Number(alertStatus?.hit_window_threshold || 3);
  if (!alertStatus) return "Alert status clear";
  if (alertState === "PENDING_ALERT") return `Pending alert (${pendingCount}/${threshold} confirmations)`;
  if (alertStatus.alert_active && alertState === "ESCALATED") return "Escalated alert";
  if (alertStatus.alert_active && alertStatus.acknowledged) return "Active alert — acknowledged";
  if (alertStatus.alert_active) return "Active alert — unacknowledged";
  if (alertState === "RESOLVED") return "Resolved after sustained recovery";
  return "Alert status clear";
}

function noTelemetryOperationalMessage(uiTruth = null) {
  return "No telemetry ingested yet. Upload telemetry to begin monitoring.";
}

function renderOperationalSnapshot(latest) {
  const stateEl = qs("#snapshotState");
  const riskEl = qs("#snapshotRisk");
  const trendEl = qs("#snapshotTrend");
  const confEl = qs("#snapshotConfidence");
  const alertEl = qs("#snapshotAlertStatus");
  const freshEl = qs("#snapshotFreshness");
  const recEl = qs("#snapshotRecommendation");

  const uiTruth = buildFrontendUiState(latest);
  const risk = normalizeRiskLevel(latest?.risk_level);
  const trend = String(trendFromResult(latest) || "UNKNOWN").toUpperCase();
  const stateText = String(latest?.state || latest?.interpreted_state || "Unknown");
  const confidenceValue = latest ? (latest.structural_analysis_available ? 92 : 74) : 0;
  const confidenceText = !latest
    ? "Low confidence — more telemetry needed"
    : confidenceValue >= 85
      ? `Confidence: ${confidenceValue}%`
      : "Confidence: moderate";
  const alertText = normalizedAlertStatusText(latest);
  const freshness = formatFreshnessLabel(latest, uiTruth);

  if (stateEl) stateEl.textContent = stateText;
  if (riskEl) riskEl.textContent = risk;
  if (trendEl) trendEl.textContent = trend;
  if (confEl) confEl.textContent = confidenceText;
  if (alertEl) alertEl.textContent = alertText;
  if (freshEl) freshEl.textContent = freshness.label;

  let recommendation = noTelemetryOperationalMessage(uiTruth);
  let nextAction = "Upload telemetry to start active monitoring.";
  if (latest) {
    if (risk === "HIGH") {
      recommendation = "Investigate sustained instability in the active run.";
      nextAction = alertText.startsWith("Active") || alertText.startsWith("Escalated")
        ? "Acknowledge active alert and inspect relationship drift."
        : "Inspect latest drift and recommendation details.";
    } else if (risk === "MEDIUM") {
      recommendation = "Acknowledge active alert and inspect relationship drift.";
      nextAction = "Review active alert state and monitor incoming telemetry.";
    } else {
      recommendation = "Continue monitoring — no intervention recommended.";
      nextAction = freshness.stale ? "Refresh active run or ingest fresh telemetry." : "System stable — continue monitoring.";
    }
  }
  if (recEl) recEl.textContent = latest ? recommendation : `${recommendation} ${nextAction}`;

  const ctaBtn = qs("#primaryPilotActionBtn");
  if (ctaBtn) {
    ctaBtn.textContent = "Upload Telemetry";
    ctaBtn.setAttribute("href", "/upload");
  }
}

function renderDashboardMetrics(latest, prev) {
  const metricTrend = qs("#metricTrend");
  const metricRisk = qs("#metricRisk");
  const metricState = qs("#metricState");
  const metricStateConfidence = qs("#metricStateConfidence");
  const metricOperator = qs("#metricOperatorMessage");
  const healthCaption = qs("#dashboardHealthCaption");
  const metricRiskBadge = qs("#metricRiskBadge");
  const metricPhaseBadge = qs("#metricPhaseBadge");
  const intelAnomaly = qs("#intelAnomalyScore");
  const intelTrend = qs("#intelTrendDirection");
  const intelDeg = qs("#intelDegradation");
  const intelConfidence = qs("#intelConfidence");
  const intelFeed = qs("#intelligenceFeedList");
  const geometryState = qs("#dashboardGeometryState");
  const recommendationPrimary = qs("#recommendationPrimary");
  const recommendationRationale = qs("#recommendationRationale");
  const recommendationOperatorNote = qs("#recommendationOperatorNote");
  const recommendationConfidenceBadge = qs("#recommendationConfidenceBadge");
  const nextActionEl = qs("#dashboardNextAction");

  if (metricTrend) metricTrend.textContent = toPretty(trendFromResult(latest));
  if (metricRisk) metricRisk.textContent = toPretty(latest?.risk_level);
  if (metricState) metricState.textContent = toPretty(latest?.state || latest?.interpreted_state);
  const uiTruth = buildFrontendUiState(latest);
  setConnectionStatus(getOperationalBadgeDisplay(uiTruth));
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
  renderOperationalSnapshot(latest);
  const score = healthScoreFromSignals(latest);
  if (score !== null) {
    animateNumberText(qs("#dashboardHealthScore"), score, { decimals: 0 });
    if (healthCaption) healthCaption.textContent = `${normalizeRiskLevel(latest?.risk_level)} risk`;
  } else if (healthCaption) {
    healthCaption.textContent = "No active telemetry";
  }
  if (intelAnomaly) {
    const drift = structuralDriftFromResult(latest) ?? 0;
    const inst = compositeInstabilityFromResult(latest) ?? 0;
    const anomaly = Math.min(100, Math.round((drift * 52 + inst * 48) * 100));
    animateNumberText(intelAnomaly, anomaly, { decimals: 0, suffix: "%" });
  }
  if (intelTrend) intelTrend.textContent = String(trendFromResult(latest) || "stable").toUpperCase();
  if (intelDeg) {
    const risk = normalizeRiskLevel(latest?.risk_level);
    intelDeg.textContent = risk === "HIGH" ? "Unfamiliar pattern" : risk === "MEDIUM" ? "Mixed pattern" : "Familiar pattern";
  }
  if (intelConfidence) {
    const conf = latest ? (latest.structural_analysis_available ? 92 : 74) : 0;
    animateNumberText(intelConfidence, conf, { decimals: 0, suffix: "%" });
    if (metricStateConfidence) metricStateConfidence.textContent = `${conf}%`;
  } else if (metricStateConfidence) {
    metricStateConfidence.textContent = "--%";
  }
  if (intelFeed) {
    const alerts = (state.dashboardAlerts || []).slice(0, 3);
    const recommendations = latest
      ? [
          `Model: ${String(phaseFromResult(latest) || "-")} / ${normalizeRiskLevel(latest.risk_level)} risk`,
          `Recommendation: ${normalizeRiskLevel(latest.risk_level) === "HIGH" ? "Dispatch immediate inspection." : "Continue monitored operations."}`,
        ]
      : ["Awaiting active-run telemetry upload."];
    const lines = alerts.length
      ? alerts.map((a) => `${a.type || "Event"}: ${a.message || "Signal deviation detected."}`)
      : recommendations;
    intelFeed.innerHTML = lines.map((line) => `<li>${escapeHtml(String(line).slice(0, 140))}</li>`).join("");
  }
  if (geometryState) {
    const available = Boolean(latest?.structural_analysis_available);
    geometryState.textContent = available
      ? "Infrastructure State Geometry is synchronized with current structural relationships."
      : "Infrastructure State Geometry is unavailable for this snapshot. Continue monitoring structural state and relationship drift.";
    geometryState.classList.toggle("geometry-ready", available);
  }
  if (recommendationPrimary || recommendationRationale || recommendationOperatorNote || recommendationConfidenceBadge) {
    const risk = normalizeRiskLevel(latest?.risk_level);
    const trend = String(trendFromResult(latest) || "stable").toLowerCase();
    const operatorSummaryClean = String(operatorSummary || "No recommendation yet.");
    const confidence = latest ? (latest.structural_analysis_available ? 92 : 74) : 0;
    const primaryText =
      !latest
        ? noTelemetryOperationalMessage(uiTruth)
        : risk === "HIGH"
        ? "Dispatch targeted inspection and increase structural watch frequency."
        : risk === "MEDIUM"
          ? "Maintain operations with elevated watch on relationship drift."
          : "Continue monitored operations under current control boundaries.";
    const rationaleText =
      !latest
        ? "After telemetry upload, analysis will populate structural state, relationship drift, pattern memory, and recommendation rationale."
        : risk === "UNKNOWN"
        ? "Structural rationale will appear in the analysis workspace once enough evidence is available."
        : `Summary only: ${String(latest?.state || latest?.interpreted_state || "unknown")} at ${risk} risk with ${trend} trend. Open Analysis Workspace for full rationale and structural context.`;
    const operatorNoteText =
      risk === "HIGH"
        ? "Prioritize assets showing strongest drift and validate containment assumptions before escalation."
        : operatorSummaryClean;
    if (recommendationPrimary) recommendationPrimary.textContent = primaryText;
    if (recommendationRationale) recommendationRationale.textContent = rationaleText;
    if (recommendationOperatorNote) recommendationOperatorNote.textContent = operatorNoteText;
    if (recommendationConfidenceBadge) recommendationConfidenceBadge.textContent = `Confidence ${confidence}%`;
    if (nextActionEl) {
      if (!latest) {
        nextActionEl.textContent = "Upload telemetry to start active monitoring.";
      } else if (risk === "HIGH") {
        nextActionEl.textContent = "Active alert requires acknowledgement and targeted inspection.";
      } else if (risk === "MEDIUM") {
        nextActionEl.textContent = "Maintain elevated watch and verify incoming telemetry quality.";
      } else {
        nextActionEl.textContent = "System stable — monitoring continues.";
      }
    }
  }
  const lu = qs("#dashboardLastUpdated");
  if (lu) {
    const rawTs = latest?.timestamp || latest?.persisted_at || latest?.created_at || "";
    lu.textContent = getLastUpdateDisplay(uiTruth, rawTs);
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
  if (state.ui.loadDashboardPromise) return state.ui.loadDashboardPromise;
  state.ui.loadDashboardPromise = (async () => {
  const runId = state.activeRun?.run_id || "";
  const [recentEnv, alertsEnv] = await Promise.all([
    fetchRecentResults({ run_id: runId, limit: DASHBOARD_RECENT_LIMIT }),
    fetchJson(apiUrl("/alerts", tenantScopeParams({ run_id: runId, limit: 8 }))),
  ]);
  state.dashboardRecent = Array.isArray(recentEnv?.results) ? recentEnv.results : [];
  state.dashboardAlerts = Array.isArray(alertsEnv?.alerts) ? alertsEnv.alerts : [];
  state.dashboardCurrentAlertStatus = alertsEnv.current_status && typeof alertsEnv.current_status === "object"
    ? alertsEnv.current_status
    : null;
  collectKnownSites(state.dashboardRecent);
  renderTenantControls();
  const chron = dashboardChronologicalResults();
  const latest = chron.length ? chron[chron.length - 1] : null;
  const prev = chron.length > 1 ? chron[chron.length - 2] : null;
  const paint = () => {
    renderDashboardMetrics(latest, prev);
    renderDashboardDemoHero();
  };
  if (state.ui.dashboardPaint) window.cancelAnimationFrame(state.ui.dashboardPaint);
  state.ui.dashboardPaint = window.requestAnimationFrame(() => {
    paint();
    state.ui.dashboardPaint = null;
  });
  })();
  try {
    await state.ui.loadDashboardPromise;
  } finally {
    state.ui.loadDashboardPromise = null;
  }
}
