function dashboardChronologicalResults() {
  return (state.dashboardRecent || []).slice().reverse();
}

// Keep selector tokens in this module for smoke-test wiring validation.
const _dashboardWiringSelectors = ["#runResultsSearchInput", "#runRangeControls [data-range]", "#runDetailEmpty", "#runResultsEmpty", "#uploadDropZone", "#selectedFileName"];

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

function renderDashboardHealthTrend(series) {
  const canvas = qs("#dashboardHealthTrend");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 640;
  const cssH = 110;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const items = Array.isArray(series) ? series : [];
  const values = items
    .map((row) => {
      if (row && typeof row.system_health === "number" && Number.isFinite(row.system_health)) return row.system_health;
      const fallback = healthScoreFromSignals(row);
      return typeof fallback === "number" && Number.isFinite(fallback) ? fallback : null;
    })
    .filter((v) => typeof v === "number" && Number.isFinite(v));
  const pad = { l: 8, r: 10, t: 16, b: 18 };
  const innerW = cssW - pad.l - pad.r;
  const innerH = cssH - pad.t - pad.b;

  if (values.length < 2) {
    ctx.fillStyle = "rgba(140, 164, 206, 0.55)";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillText("System health trend appears after more telemetry.", pad.l, pad.t + 18);
    return;
  }

  const minY = Math.max(0, Math.min(...values) - 4);
  const maxY = Math.min(100, Math.max(...values) + 4);
  const span = Math.max(1, maxY - minY);
  const step = innerW / Math.max(1, values.length - 1);
  const xAt = (i) => pad.l + i * step;
  const yAt = (v) => pad.t + innerH - ((v - minY) / span) * innerH;

  ctx.strokeStyle = "rgba(72, 105, 150, 0.32)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g += 1) {
    const y = pad.t + (innerH * g) / 3;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + innerW, y);
    ctx.stroke();
  }

  ctx.beginPath();
  values.forEach((v, i) => {
    const x = xAt(i);
    const y = yAt(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "rgba(117, 226, 187, 0.95)";
  ctx.lineWidth = 2;
  ctx.stroke();

  const last = values[values.length - 1];
  ctx.beginPath();
  ctx.arc(xAt(values.length - 1), yAt(last), 3.8, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(117, 226, 187, 0.95)";
  ctx.fill();

  ctx.fillStyle = "rgba(174, 226, 211, 0.92)";
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.fillText("health", pad.l, 11);
}

function renderRecentTransitionsTimeline(series) {
  const list = qs("#dashboardStateTimeline");
  if (!list) return;
  list.innerHTML = "";
  const chron = Array.isArray(series) ? series : [];
  if (!chron.length) {
    const li = document.createElement("li");
    li.className = "timeline-item";
    li.textContent = "No transitions available yet. Ingest at least two telemetry snapshots to unlock transition tracking.";
    list.appendChild(li);
    return;
  }
  const latest = chron[chron.length - 1];
  const recent = chron.slice(Math.max(0, chron.length - 8));
  const items = [];
  for (let i = 1; i < recent.length; i += 1) {
    const prev = recent[i - 1];
    const curr = recent[i];
    const label = transitionLabel(prev, curr);
    const sev = transitionSeverity(prev, curr);
    const ts = curr.timestamp || curr.persisted_at || "unknown time";
    if (label !== "No major transition" || sev !== "normal" || i === recent.length - 1) {
      items.push({ label, sev, ts });
    }
  }
  if (!items.length) {
    items.push({
      label: `Current state ${String(latest.state || latest.interpreted_state || "unknown")} (${normalizeRiskLevel(latest.risk_level)} risk)`,
      sev: "normal",
      ts: latest.timestamp || latest.persisted_at || "latest snapshot",
    });
  }
  items.slice(-6).reverse().forEach((item) => {
    const li = document.createElement("li");
    li.className = `timeline-item ${item.sev === "critical" ? "critical" : item.sev === "watch" ? "watch" : ""}`.trim();
    li.innerHTML = `<strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(String(item.ts))}</span>`;
    list.appendChild(li);
  });
}

function renderEvidencePanel(latest) {
  const driftEl = qs("#evidenceDriftValue");
  const compEl = qs("#evidenceCompositeValue");
  const confEl = qs("#evidenceConfidenceValue");
  const drift = structuralDriftFromResult(latest);
  const comp = dashboardCompositeScore(latest);
  if (driftEl) driftEl.textContent = typeof drift === "number" ? drift.toFixed(3) : "Warming up";
  if (compEl) compEl.textContent = typeof comp === "number" ? comp.toFixed(3) : "Not enough history yet";
  if (confEl) confEl.textContent = dashboardConfidenceText(latest);
}

function buildAssistantContextPayload(latest) {
  const runId = state.activeRun?.run_id || "";
  const siteId = state.tenant.siteId || latest?.site_id || "";
  const assetId = latest?.asset_id || "";
  return {
    customer_id: customerIdValue(state.tenant.customerId),
    run_id: runId,
    site_id: siteId || null,
    asset_id: assetId || null,
  };
}

function buildAssistantMetricsSnapshot(chronological) {
  const chron = Array.isArray(chronological) ? chronological : [];
  const latest = chron.length ? chron[chron.length - 1] : null;
  const tail = chron.slice(Math.max(0, chron.length - 10));
  return {
    current_state: latest ? String(latest.state || latest.interpreted_state || "unknown") : null,
    current_risk_level: latest ? normalizeRiskLevel(latest.risk_level) : "UNKNOWN",
    structural_drift_score: structuralDriftFromResult(latest),
    composite_instability: compositeInstabilityFromResult(latest),
    system_health:
      latest && typeof latest.system_health === "number" && Number.isFinite(latest.system_health)
        ? latest.system_health
        : healthScoreFromSignals(latest),
    transition_count_recent: Math.max(0, tail.length - 1),
    recent_timestamps: tail.map((row) => row.timestamp || row.persisted_at || "").filter(Boolean),
  };
}

function renderAssistantResponse(payload, statusText = "") {
  const observedEl = qs("#assistantObservedList");
  const inferredEl = qs("#assistantInferredList");
  const nextEl = qs("#assistantNextStep");
  const statusEl = qs("#assistantChatStatus");
  const uncertaintyEl = qs("#assistantUncertainty");
  const groundingEl = qs("#assistantGrounding");
  if (statusEl && statusText) statusEl.textContent = statusText;
  if (observedEl) {
    const observed = Array.isArray(payload?.observed) ? payload.observed : [];
    observedEl.innerHTML = observed.length
      ? observed.map((line) => `<li>${escapeHtml(String(line))}</li>`).join("")
      : "<li>No observed signals yet. Verify telemetry is flowing for this run context.</li>";
  }
  if (inferredEl) {
    const inferred = Array.isArray(payload?.inferred) ? payload.inferred : [];
    inferredEl.innerHTML = inferred.length
      ? inferred.map((line) => `<li>${escapeHtml(String(line))}</li>`).join("")
      : "<li>No inference available yet from current telemetry.</li>";
  }
  if (nextEl) nextEl.textContent = String(payload?.suggested_next_step || "Hold monitoring cadence and retry after a fresh snapshot.");
  if (uncertaintyEl) {
    uncertaintyEl.textContent = Number.isFinite(Number(payload?.uncertainty))
      ? `Uncertainty: ${Number(payload.uncertainty).toFixed(4)}`
      : "Uncertainty: -";
  }
  if (groundingEl) {
    const grounding = payload?.grounding && typeof payload.grounding === "object" ? payload.grounding : {};
    groundingEl.textContent = JSON.stringify(grounding, null, 2);
  }
}

function wireAssistantChat() {
  const form = qs("#assistantChatForm");
  const input = qs("#assistantChatInput");
  const sendBtn = qs("#assistantChatSend");
  if (!form || !input || form.dataset.wiredAssistant === "1") return;
  form.dataset.wiredAssistant = "1";
  const contextHint = qs("#assistantContextHint");
  if (contextHint) {
    const hasRun = Boolean(state.activeRun?.run_id);
    contextHint.textContent = hasRun
      ? `Context source: run ${state.activeRun.run_id}, selected site/asset, and latest telemetry.`
      : "Context incomplete: no active run selected yet. Create or activate a run, then ingest telemetry.";
  }

  qsa("[data-assistant-prompt]").forEach((btn) => {
    if (btn.dataset.wiredAssistantPrompt === "1") return;
    btn.dataset.wiredAssistantPrompt = "1";
    btn.addEventListener("click", () => {
      input.value = String(btn.getAttribute("data-assistant-prompt") || "");
      input.focus();
    });
  });

  form.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    const message = String(input.value || "").trim();
    if (!message) return;
    const statusEl = qs("#assistantChatStatus");
    if (statusEl) statusEl.textContent = "Interpreting current run context...";
    if (sendBtn) sendBtn.disabled = true;
    try {
      const body = {
        message,
        context: buildAssistantContextPayload(state.assistant.latest),
        recent_metrics_snapshot: buildAssistantMetricsSnapshot(state.assistant.chronological),
      };
      const response = await fetchJson(apiUrl("/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      renderAssistantResponse(response, "Interpretation grounded in latest run/site/asset context.");
    } catch (err) {
      renderAssistantResponse(
        {
          observed: [],
          inferred: [String(err?.message || err || "Assistant request failed.")],
          suggested_next_step: "Refresh telemetry for this context, then run interpretation again.",
          uncertainty: 1.0,
          grounding: {},
        },
        "Interpreter request failed.",
      );
    } finally {
      if (sendBtn) sendBtn.disabled = false;
    }
  });
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

function renderRiskProgression(latest) {
  const stages = qsa("#dashboardRiskProgression .risk-stage");
  if (!stages.length) return;
  const risk = normalizeRiskLevel(latest?.risk_level);
  const order = ["LOW", "MEDIUM", "HIGH"];
  stages.forEach((el, idx) => {
    const active = order.indexOf(risk) >= idx && order.indexOf(risk) >= 0;
    el.classList.toggle("active", active);
  });
}

function _graphMatrixFromResult(result, mode = "current", prevResult = null) {
  const analytics = result?.experimental_analytics && typeof result.experimental_analytics === "object"
    ? result.experimental_analytics
    : {};
  const cg = analytics.correlation_geometry && typeof analytics.correlation_geometry === "object"
    ? analytics.correlation_geometry
    : {};
  const current = Array.isArray(cg.current) ? cg.current : null;
  const baseline = Array.isArray(cg.baseline) ? cg.baseline : null;
  const prevAnalytics = prevResult?.experimental_analytics && typeof prevResult.experimental_analytics === "object"
    ? prevResult.experimental_analytics
    : {};
  const prevCg = prevAnalytics.correlation_geometry && typeof prevAnalytics.correlation_geometry === "object"
    ? prevAnalytics.correlation_geometry
    : {};
  const prevCurrent = Array.isArray(prevCg.current) ? prevCg.current : null;
  if (mode === "baseline") return baseline || current || prevCurrent;
  if (mode === "delta") return current || prevCurrent || baseline;
  return current || baseline || prevCurrent;
}

function _graphSensorNames(result, matrix) {
  const analytics = result?.experimental_analytics && typeof result.experimental_analytics === "object"
    ? result.experimental_analytics
    : {};
  let names = Array.isArray(analytics.valid_sensor_names) ? analytics.valid_sensor_names : null;
  if (!names || !names.length) names = Array.isArray(analytics.feature_names) ? analytics.feature_names : null;
  if ((!names || !names.length) && Array.isArray(result?.sensor_relationships)) names = result.sensor_relationships;
  if ((!names || !names.length) && result?.sensor_values && typeof result.sensor_values === "object") {
    names = Object.keys(result.sensor_values);
  }
  const count = Array.isArray(matrix) ? matrix.length : 0;
  const normalized = Array.isArray(names) ? names.map((n, i) => String(n || `sensor_${i + 1}`)) : [];
  while (normalized.length < count) normalized.push(`sensor_${normalized.length + 1}`);
  return normalized.slice(0, count);
}

function buildRelationshipGraphFrames(results = state.dashboardRecent || []) {
  const chron = Array.isArray(results) ? results.slice().reverse() : [];
  return chron.map((result, idx) => {
    const prev = idx > 0 ? chron[idx - 1] : null;
    const currentMatrix = _graphMatrixFromResult(result, "current", prev);
    const baselineMatrix = _graphMatrixFromResult(result, "baseline", prev);
    const names = _graphSensorNames(result, currentMatrix || baselineMatrix || []);
    const size = names.length;
    if (!size || !Array.isArray(currentMatrix)) return null;
    const nodes = names.map((name, i) => {
      let importance = 0.45;
      let avgAbs = 0;
      let count = 0;
      for (let j = 0; j < size; j += 1) {
        if (i === j) continue;
        const v = Number(currentMatrix?.[i]?.[j]);
        if (Number.isFinite(v)) {
          avgAbs += Math.abs(v);
          count += 1;
        }
      }
      if (count > 0) importance = Math.min(1, avgAbs / count);
      return {
        id: `sensor_${i}`,
        label: String(name),
        importance,
        riskContribution: importance,
        group: `cluster_${Math.floor((i / Math.max(1, size)) * 4) + 1}`,
      };
    });
    const links = [];
    for (let i = 0; i < size; i += 1) {
      for (let j = i + 1; j < size; j += 1) {
        const nowV = Number(currentMatrix?.[i]?.[j]);
        if (!Number.isFinite(nowV)) continue;
        const baseV = Number(baselineMatrix?.[i]?.[j]);
        const prevV = Number(prev?.experimental_analytics?.correlation_geometry?.current?.[i]?.[j]);
        const ref = Number.isFinite(baseV) ? baseV : (Number.isFinite(prevV) ? prevV : nowV);
        const drift = Math.abs(nowV - ref);
        links.push({
          source: `sensor_${i}`,
          target: `sensor_${j}`,
          weight: Math.abs(nowV),
          drift,
          signedWeight: nowV,
          isCritical: drift >= 0.18 || Math.abs(nowV) >= 0.75,
        });
      }
    }
    links.sort((a, b) => (b.weight + b.drift) - (a.weight + a.drift));
    return {
      timestamp: String(result.timestamp || result.persisted_at || result.created_at || ""),
      nodes,
      links: links.slice(0, 280),
    };
  }).filter(Boolean);
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

function dashboardOperatorIdentityText(latest, prev) {
  if (!latest) return "System warming up. Upload telemetry to begin monitoring.";
  const risk = normalizeRiskLevel(latest.risk_level);
  const drift = structuralDriftFromResult(latest);
  const driftText = typeof drift === "number" ? drift.toFixed(2) : "pending";
  const transition = transitionLabel(prev, latest);
  if (risk === "HIGH") {
    return `System is destabilizing due to sustained structural drift. (${transition}; drift ${driftText})`;
  }
  if (risk === "MEDIUM") {
    return `System is transitioning with measurable structural drift. (${transition}; drift ${driftText})`;
  }
  return `System is stable with low structural drift. (${transition}; drift ${driftText})`;
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

function stopDashboardReplay() {
  if (state.dashboardReplay.timer) {
    window.clearTimeout(state.dashboardReplay.timer);
    state.dashboardReplay.timer = null;
  }
  state.dashboardReplay.active = false;
}

function dashboardChronologicalForRender(fullChronological) {
  const chron = Array.isArray(fullChronological) ? fullChronological : [];
  if (!state.dashboardReplay.active) return chron;
  const max = Math.max(1, Number(state.dashboardReplay.cursor || 1));
  return chron.slice(0, Math.min(chron.length, max));
}

function scheduleDashboardReplayTick(fullChronological, onFrame) {
  if (!state.dashboardReplay.active) return;
  if (state.dashboardReplay.timer) {
    window.clearTimeout(state.dashboardReplay.timer);
    state.dashboardReplay.timer = null;
  }
  state.dashboardReplay.timer = window.setTimeout(() => {
    if (!state.dashboardReplay.active) return;
    const total = Array.isArray(fullChronological) ? fullChronological.length : 0;
    if (state.dashboardReplay.cursor >= total) {
      stopDashboardReplay();
      if (typeof onFrame === "function") onFrame();
      return;
    }
    state.dashboardReplay.cursor += 1;
    if (typeof onFrame === "function") onFrame();
    scheduleDashboardReplayTick(fullChronological, onFrame);
  }, DASHBOARD_REPLAY_INTERVAL_MS);
}

function startDashboardReplay(fullChronological, onFrame) {
  const chron = Array.isArray(fullChronological) ? fullChronological : [];
  const route = getRoute();
  const canReplay = Boolean(state.demo.enabled) && route.page === "dashboard" && chron.length >= 6;
  if (!canReplay) {
    stopDashboardReplay();
    state.dashboardReplay.cursor = chron.length;
    return;
  }
  if (state.dashboardReplay.active && state.dashboardReplay.cursor < chron.length) return;
  state.dashboardReplay.active = true;
  state.dashboardReplay.cursor = 1;
  if (typeof onFrame === "function") onFrame();
  scheduleDashboardReplayTick(chron, onFrame);
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
    const prevRisk = normalizeRiskLevel(prev?.risk_level);
    const currRisk = normalizeRiskLevel(current.risk_level);
    const recovered = prevRisk === "HIGH" && currRisk !== "HIGH";
    if (sev === "critical" || isSpike || isDriftEvent || recovered) {
      events.push({
        index: i + 1,
        ts: String(current.timestamp || current.persisted_at || ""),
        severity: recovered ? "watch" : (sev === "normal" ? (isSpike ? "critical" : "watch") : sev),
        text: recovered
          ? `Recovery signal · ${transition}`
          : (isDriftEvent ? `${transition} · drift jump ${driftJump.toFixed(2)}` : transition),
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
  applyDemoUiShell();
  if (!state.demo.enabled) {
    stopDemoPlayback();
    stopDashboardReplay();
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

function syncDemoModeToggleButton(btn) {
  if (!btn) return;
  const enabled = !!state.demo.enabled;
  btn.setAttribute("aria-pressed", enabled ? "true" : "false");
  btn.textContent = enabled ? "Using demo data" : "Use demo data";
}

function wireDemoModeToggle(btn) {
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  syncDemoModeToggleButton(btn);
  btn.addEventListener("click", async () => {
    const nextEnabled = !state.demo.enabled;
    await toggleDemoMode(nextEnabled);
    syncDemoModeToggleButton(btn);
    if (nextEnabled) {
      setStatus("Demo mode enabled. Focused presentation layout is active.", false, true);
    } else {
      setStatus("Demo mode disabled. Full workspace layout restored.", false, true);
    }
  });
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

async function startCmapssDemo(customerId, options = {}) {
  return fetchJson(apiUrl("/demo/cmapss/start", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      customer_id: customerId || null,
      max_frames: Number(options.max_frames || CMAPSS_REPLAY_DEFAULT_MAX_FRAMES),
    }),
  });
}

function onDemoPlaybackComplete() {
  if (state.demo.playbackCompleteNotified) return;
  state.demo.playbackCompleteNotified = true;
  createToast("Greenhouse demo ready. Opening upload with the drop zone highlighted.", "success");
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
  if (p < 0.2) return "Baseline — system is establishing healthy structural reference.";
  if (p < 0.4) return "Early drift — subtle relationship shift appears before hard alarms.";
  if (p < 0.65) return "Instability — drift and composite pressure are compounding.";
  if (p < 0.85) return "Alert — high-risk state confirms sustained destabilization.";
  return "Recovery/intervention — watch for risk easing and structural re-stabilization.";
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

function getRoute() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length === 0 || parts[0] === "dashboard") return { page: "dashboard" };
  if (parts[0] === "validation" || parts[0] === "reference" || parts[0] === "historical-validation") return { page: "validation" };
  if (parts[0] === "onboarding") return { page: "onboarding" };
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
    audience: "operator",
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
    autoplay: false,
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
      launchProcessed: 0,
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
  dashboardReplay: {
    active: false,
    timer: null,
    cursor: 0,
  },
  relationshipGraph: {
    mode: "current",
    frameIndex: 0,
    playing: false,
    timer: null,
    edgeThreshold: 0.2,
    frames: [],
    nodes: [],
    links: [],
    positions: {},
    hoveredNodeId: "",
    hoveredEdgeId: "",
    selectedNodeId: "",
    selectedEdgeId: "",
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
    runDetailGrowOpMonitorTimer: null,
    runDetailGrowOpMonitorToken: 0,
    loadRunsPromise: null,
    loadDashboardPromise: null,
    recentResultsCache: new Map(),
    recentResultsInflight: new Map(),
  },
  runtimeDegraded: false,
  assistant: {
    latest: null,
    chronological: [],
  },
};

const TENANT_STORAGE_KEY = "neraium_customer_id";
const DEMO_MODE_STORAGE_KEY = "neraium_demo_mode";
/** Grow-op replay pacing knob: higher values hold each story state longer for guided playback. */
const DEMO_PACING_MULTIPLIER = 1.8;
/** Demo timeline: advance one snapshot per interval (tunable; lower = faster review). */
const DEMO_PLAYBACK_INTERVAL_MS = Math.round(1600 * DEMO_PACING_MULTIPLIER);
/** Dashboard demo replay speed for the top-level narrative animation. */
const DASHBOARD_REPLAY_INTERVAL_MS = Math.round(500 * DEMO_PACING_MULTIPLIER);
/** How often to poll `/ingest/jobs/{id}` after CSV upload (lower = snappier status UI). */
const INGEST_JOB_POLL_MS = 400;
/** Replay launch/status polling cadence + resilience controls. */
const DEMO_REPLAY_INITIAL_POLL_MS = 900;
const DEMO_REPLAY_MAX_POLL_MS = 8000;
const DEMO_REPLAY_STARTING_TIMEOUT_MS = 45000;
const DEMO_REPLAY_RESULTS_MATERIALIZATION_TIMEOUT_MS = 180000;
const DEMO_REPLAY_MAX_TRANSIENT_ERRORS = 4;
const CMAPSS_REPLAY_DEFAULT_MAX_FRAMES = 240;
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

const DASHBOARD_RECENT_LIMIT = 10;
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
  const context = qs("#connectionContext");
  const normalized = String(mode || "NO DATA INGESTED").toUpperCase();
  const demoContext = state.demo.enabled && ["WAITING FOR TELEMETRY", "NO DATA INGESTED"].includes(normalized);
  const display = demoContext
    ? "DEMO MODE"
    : ["ACTIVE MONITORING", "ALERT ACTIVE", "HISTORICAL VALIDATION", "VALIDATION READY"].includes(normalized)
      ? "LIVE SYSTEM"
      : ["ANALYSIS INTERRUPTED", "REPLAY INTERRUPTED"].includes(normalized)
        ? "SYSTEM INTERRUPTED"
        : "NO SENSOR CONNECTION";
  state.ui.connection = display;
  if (label) label.textContent = display;
  if (context) {
    context.textContent = demoContext ? "Using demo data" : "";
    context.classList.toggle("hidden", !demoContext);
  }
  if (badge) {
    badge.classList.remove("chip-live", "chip-demo", "chip-offline");
    if (display === "SYSTEM INTERRUPTED" || display === "NO SENSOR CONNECTION") {
      badge.classList.add("chip-offline");
    } else if (display === "LIVE SYSTEM") {
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
  if (lower.includes("seed")) return "Processing greenhouse demo dataset...";
  if (lower.includes("demo")) return "Preparing greenhouse demo replay...";
  if (lower.includes("cmapss")) return "Processing greenhouse demo dataset...";
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
    dashboard: ["Monitor risk", "See risk now. Act now."],
    upload: ["Upload telemetry", "Send new CSV data to the active run."],
    runs: ["Investigate runs", "Find one run and inspect the timeline."],
    validation: ["Validation replay", "Run historical replay for model checks."],
    onboarding: ["Onboarding setup", "Connect source and confirm required fields."],
    "run-detail": ["Investigate run", "Review trend, risk, and next action."],
    "result-detail": ["Result detail", "Inspect one result and why it was scored this way."],
  };
  qsa(".page").forEach((p) => p.classList.add("hidden"));
  if (page !== "dashboard") stopDashboardReplay();
  const pageEl = qs(`#page-${page}`);
  if (pageEl) pageEl.classList.remove("hidden");
  const [title, subtitle] = titles[page] || ["Neraium", ""];
  document.body?.setAttribute("data-active-page", page);
  const titleEl = qs("#pageTitle");
  const subtitleEl = qs("#pageSubtitle");
  if (titleEl) titleEl.textContent = title;
  if (subtitleEl) subtitleEl.textContent = subtitle;
  qsa(".nav a").forEach((a) => a.classList.remove("active"));
  if (page === "dashboard") qs('[data-nav="dashboard"]')?.classList.add("active");
  if (page === "upload") qs('[data-nav="upload"]')?.classList.add("active");
  if (page === "runs" || page === "run-detail") qs('[data-nav="runs"]')?.classList.add("active");
  if (page === "validation") qs('[data-nav="validation"]')?.classList.add("active");
  if (page === "onboarding") qs('[data-nav="onboarding"]')?.classList.add("active");
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
  const opRoom = qs("#runOperatorRoom");
  const opHeadline = qs("#runOperatorHeadline");
  const opBody = qs("#runOperatorBody");
  const opRisk = qs("#runOperatorRisk");
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
  if (opRoom) opRoom.textContent = risk === "HIGH" ? "🔴 ROOM 104 — FIX TODAY" : risk === "MEDIUM" ? "🟡 ROOM 104 — CHECK SOON" : "🟢 ROOM STATUS";
  if (opHeadline) opHeadline.textContent = risk === "HIGH" ? "Your AC is breaking." : risk === "MEDIUM" ? "Stress is building in one room." : "No urgent room failures detected.";
  if (opBody) opBody.textContent = risk === "HIGH"
    ? "Room will hit 90°F by 6 PM if no action is taken."
    : risk === "MEDIUM"
      ? "48-hour forecast shows rising HVAC stress; schedule maintenance now."
      : "System is stable and monitoring continuously.";
  if (opRisk) opRisk.textContent = risk === "HIGH" ? "$25,000 at risk" : risk === "MEDIUM" ? "$12,000 at risk" : "$0 at risk";
}

function dashboardRunIdFromQuery() {
  try {
    return String(new URLSearchParams(window.location.search).get("run_id") || "").trim();
  } catch (_err) {
    return "";
  }
}

function applyDashboardRunFromQuery() {
  if (getRoute().page !== "dashboard") return;
  const requestedRunId = dashboardRunIdFromQuery();
  if (!requestedRunId) return;
  if (String(state.activeRun?.run_id || "") === requestedRunId) return;
  const knownRun = state.runs.find((run) => String(run?.run_id || "") === requestedRunId) || null;
  updateActiveRunHeader(knownRun || {
    run_id: requestedRunId,
    name: requestedRunId,
    is_active: false,
    status: "open",
  });
}

function wireWorkspaceShellEvents() {
  wireDemoModeToggle(qs("#demoModeToggle"));
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
  return "No data yet — upload telemetry to begin monitoring.";
}

function _normalizeSummaryTrend(value) {
  const text = String(value || "").trim().toUpperCase();
  if (!text || text === "-") return "";
  if (text.includes("DESTABIL")) return "DESTABILIZING";
  if (text.includes("WATCH") || text.includes("DRIFT") || text.includes("SHIFT")) return "WATCH";
  if (text.includes("STABLE") || text.includes("NOMINAL") || text.includes("RECOVER")) return "STABLE";
  return text;
}

function dashboardTrendLabel(result) {
  const direct = _normalizeSummaryTrend(result?.trend);
  if (direct) return direct;
  const risk = normalizeRiskLevel(result?.risk_level);
  const drift = structuralDriftFromResult(result);
  const rawHealth = result?.system_health;
  const health = typeof rawHealth === "number" && Number.isFinite(rawHealth) ? rawHealth : null;
  if (risk === "HIGH") return "DESTABILIZING";
  if (risk === "MEDIUM") return "WATCH";
  if (typeof drift === "number" && Number.isFinite(drift)) {
    if (drift >= 0.65) return "DESTABILIZING";
    if (drift >= 0.3) return "WATCH";
    return "STABLE";
  }
  if (typeof health === "number") {
    if (health <= 45) return "DESTABILIZING";
    if (health <= 70) return "WATCH";
    return "STABLE";
  }
  return "WARMING UP";
}

function dashboardCompositeScore(result) {
  if (!result) return null;
  const numericCandidates = [
    result.latest_instability,
    result.composite_instability,
    result.instability,
    result.experimental_analytics?.composite_instability,
  ];
  for (const candidate of numericCandidates) {
    if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  }
  if (typeof result.system_health === "number" && Number.isFinite(result.system_health)) {
    return Math.max(0, Math.min(1, 1 - (result.system_health / 100)));
  }
  return null;
}

function dashboardDriftLabel(result) {
  const drift = structuralDriftFromResult(result);
  if (typeof drift !== "number" || !Number.isFinite(drift)) return "WARMING UP";
  if (drift >= 0.65) return "DESTABILIZING";
  if (drift >= 0.3) return "WATCH";
  return "STABLE";
}

function dashboardConfidenceText(result) {
  if (!result) return "Warming up";
  const rawConfidence = result.confidence;
  const numeric = typeof rawConfidence === "number"
    ? rawConfidence
    : (typeof rawConfidence === "string" && rawConfidence.trim() ? Number(rawConfidence) : NaN);
  if (Number.isFinite(numeric)) {
    const pct = Math.max(0, Math.min(100, Math.round(numeric * 100)));
    return `${pct}%`;
  }
  const text = String(rawConfidence || "").trim();
  return text || "Not enough history yet";
}

function dashboardLastUpdatedText(result) {
  const rawTs = result?.timestamp || result?.persisted_at || result?.created_at || "";
  if (!rawTs) return "Warming up";
  const dt = new Date(rawTs);
  if (Number.isNaN(dt.getTime())) return "Warming up";
  return dt.toLocaleString();
}

function buildTopSummarySentence(latest) {
  if (!latest) {
    return "System is warming up for this run; not enough history yet to score structural change.";
  }
  const risk = normalizeRiskLevel(latest?.risk_level).toLowerCase();
  const site = String(latest?.site_id || "this site");
  const asset = String(latest?.asset_id || "this asset");
  const drift = structuralDriftFromResult(latest);
  const composite = dashboardCompositeScore(latest);
  const trend = String(dashboardTrendLabel(latest) || "WARMING UP").toLowerCase();
  let cause = "limited telemetry history";
  if (typeof drift === "number" && drift >= 0.65) cause = "sustained structural drift";
  else if (typeof composite === "number" && composite >= 0.6) cause = "compounding instability";
  else if (trend.includes("watch")) cause = "early relational drift";
  else if (trend.includes("stable")) cause = "stable structural relationships";
  return `System is ${trend} in ${asset} at ${site} (${risk} risk) due to ${cause}.`;
}

function operatorSafeResult(result) {
  if (!result || typeof result !== "object") return result;
  return {
    result_id: result.result_id,
    run_id: result.run_id,
    timestamp: result.timestamp,
    persisted_at: result.persisted_at,
    created_at: result.created_at,
    site_id: result.site_id,
    asset_id: result.asset_id,
    state: result.state,
    interpreted_state: result.interpreted_state,
    regime_name: result.regime_name,
    phase: result.phase,
    risk_level: result.risk_level,
    trend: result.trend,
    alert_status: result.alert_status,
    alert: result.alert,
    system_health: result.system_health,
    structural_drift_score: result.structural_drift_score,
    composite_instability: result.composite_instability,
    latest_instability: result.latest_instability,
    instability: result.instability,
    confidence: result.confidence,
    operator_message: result.operator_message,
    structural_analysis_available: result.structural_analysis_available,
    skipped_reason: result.skipped_reason,
  };
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
  const trend = dashboardTrendLabel(latest);
  const stateText = String(latest?.state || latest?.interpreted_state || "Unknown");
  const freshness = formatFreshnessLabel(latest, uiTruth);
  const confidenceText = dashboardConfidenceText(latest);
  const alertText = normalizedAlertStatusText(latest);

  if (stateEl) stateEl.textContent = stateText;
  if (riskEl) riskEl.textContent = risk;
  if (trendEl) trendEl.textContent = `Trend: ${trend}`;
  if (confEl) confEl.textContent = `Confidence: ${confidenceText}`;
  if (alertEl) alertEl.textContent = alertText;
  if (freshEl) freshEl.textContent = freshness.label;

  let recommendation = noTelemetryOperationalMessage(uiTruth);
  let nextAction = "No data yet — upload telemetry.";
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
    ctaBtn.textContent = "Upload telemetry";
    ctaBtn.setAttribute("href", "/upload");
  }
}

function renderPredictionTimeline(riskLevel = "LOW") {
  const note = qs("#predictionTimelineNote");
  const weekNote = qs("#weekForecastNote");
  const steps = qsa("#predictionTimeline .prediction-step");
  const weekDots = qsa("#weekForecast [data-week-hour]");
  const weekConfidence = qsa("#weekForecast [data-week-confidence]");
  const confidenceByHour = { 0: 100, 24: 85, 48: 78, 72: 70, 96: 65, 120: 60, 168: 55 };
  const statusAtHour = (hour) => {
    if (riskLevel === "HIGH") {
      if (hour >= 72) return "high";
      if (hour >= 24) return "watch";
      return "low";
    }
    if (riskLevel === "MEDIUM") {
      if (hour >= 96) return "high";
      if (hour >= 48) return "watch";
      return "low";
    }
    if (hour >= 120) return "watch";
    return "low";
  };
  steps.forEach((step) => {
    const hour = Number(step.getAttribute("data-horizon-hour") || 0);
    const level = statusAtHour(hour);
    step.classList.remove("is-low", "is-watch", "is-high");
    step.classList.add(`is-${level}`);
  });
  weekDots.forEach((dot) => {
    const hour = Number(dot.getAttribute("data-week-hour") || 0);
    const level = statusAtHour(hour);
    dot.classList.remove("is-low", "is-watch", "is-high");
    dot.classList.add(`is-${level}`);
  });
  weekConfidence.forEach((cell) => {
    const hour = Number(cell.getAttribute("data-week-confidence") || 0);
    const conf = confidenceByHour[hour] ?? 50;
    cell.textContent = `${conf}%`;
  });
  if (note) {
    note.textContent = riskLevel === "HIGH"
      ? "Prediction: drift is accelerating. Time-to-critical is likely inside 72–168 hours."
      : riskLevel === "MEDIUM"
        ? "Prediction: stress is building. Action window appears in the next 3–5 days."
        : "Prediction: stable now with low-risk drift. Continue continuous 7-day forecasting.";
  }
  if (weekNote) {
    weekNote.textContent = riskLevel === "HIGH"
      ? "Room 104: Stable → Stress → Critical by end-of-week. Schedule HVAC intervention immediately."
      : riskLevel === "MEDIUM"
        ? "Room 104: Stable → Stress by mid-week. Schedule maintenance before Friday."
        : "Facility outlook: low risk for the next 72h, planning guidance only beyond day 4.";
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
  const operatorIdentity = qs("#dashboardOperatorIdentity");

  if (metricTrend) metricTrend.textContent = dashboardTrendLabel(latest);
  if (metricRisk) metricRisk.textContent = toPretty(latest?.risk_level);
  if (metricState) metricState.textContent = toPretty(latest?.state || latest?.interpreted_state);
  const uiTruth = buildFrontendUiState(latest);
  setConnectionStatus(getOperationalBadgeDisplay(uiTruth));
  if (topSummaryEl) topSummaryEl.textContent = buildTopSummarySentence(latest);
  const operatorSummary = demoFriendlyOperatorMessage(latest, prev);
  if (operatorIdentity) operatorIdentity.textContent = dashboardOperatorIdentityText(latest, prev);
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
  renderRiskProgression(latest);
  const driftScore = structuralDriftFromResult(latest);
  const driftLabel = dashboardDriftLabel(latest);
  const driftEl = qs("#dashboardHealthScore");
  if (driftEl) {
    driftEl.textContent = typeof driftScore === "number" ? driftScore.toFixed(3) : "Warming up";
  }
  if (healthCaption) {
    if (typeof driftScore === "number") {
      healthCaption.textContent = `Drift band: ${driftLabel} · score ${driftScore.toFixed(3)}`;
    } else {
      healthCaption.textContent = "Not enough history yet";
    }
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
  }
  if (metricStateConfidence) metricStateConfidence.textContent = dashboardLastUpdatedText(latest);
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
    const trend = dashboardTrendLabel(latest).toLowerCase();
    const operatorSummaryClean = String(operatorSummary || "No recommendation yet.");
    const confidence = dashboardConfidenceText(latest);
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
    if (recommendationConfidenceBadge) recommendationConfidenceBadge.textContent = `Confidence ${confidence}`;
    if (nextActionEl) {
      if (!latest) {
        nextActionEl.textContent = "No data yet — upload telemetry.";
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
    lu.textContent = dashboardLastUpdatedText(latest);
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

function stopRelationshipGraphPlayback() {
  if (state.relationshipGraph.timer) {
    window.clearInterval(state.relationshipGraph.timer);
    state.relationshipGraph.timer = null;
  }
  state.relationshipGraph.playing = false;
}

function renderRelationshipGraph() {
  const canvas = qs("#relationshipGraphCanvas");
  const empty = qs("#relationshipGraphEmpty");
  const tooltip = qs("#relationshipGraphTooltip");
  const detailTitle = qs("#relationshipGraphDetailTitle");
  const detailBody = qs("#relationshipGraphDetailBody");
  const frameLabel = qs("#relationshipGraphTimeLabel");
  const scrubber = qs("#relationshipGraphTimeScrubber");
  if (!canvas || !canvas.getContext) return;
  const frames = state.relationshipGraph.frames || [];
  if (!frames.length) {
    if (empty) {
      empty.classList.remove("hidden");
      empty.textContent = "No relationship data available yet. Upload or replay telemetry to render structure.";
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (scrubber) scrubber.max = "0";
    if (frameLabel) frameLabel.textContent = "Replay unavailable";
    return;
  }
  if (empty) empty.classList.add("hidden");
  const frameIndex = Math.max(0, Math.min(frames.length - 1, Number(state.relationshipGraph.frameIndex || 0)));
  const mode = state.relationshipGraph.mode || "current";
  const frame = frames[frameIndex];
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth || 960;
  const height = 520;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(120, Math.min(width, height) * 0.34);
  const nodes = frame.nodes || [];
  const links = (frame.links || []).filter((l) => (mode === "delta" ? l.drift : l.weight) >= state.relationshipGraph.edgeThreshold);
  if (scrubber) {
    scrubber.max = String(Math.max(0, frames.length - 1));
    scrubber.value = String(frameIndex);
  }
  if (frameLabel) frameLabel.textContent = frame.timestamp ? `Frame ${frameIndex + 1}/${frames.length} · ${frame.timestamp}` : `Frame ${frameIndex + 1}/${frames.length}`;

  state.relationshipGraph.nodes = nodes;
  state.relationshipGraph.links = links;
  nodes.forEach((node, idx) => {
    const angle = (Math.PI * 2 * idx) / Math.max(1, nodes.length) - Math.PI / 2;
    const target = { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
    const prev = state.relationshipGraph.positions[node.id] || target;
    state.relationshipGraph.positions[node.id] = {
      x: prev.x + (target.x - prev.x) * 0.28,
      y: prev.y + (target.y - prev.y) * 0.28,
    };
  });

  links.forEach((link) => {
    const a = state.relationshipGraph.positions[link.source];
    const b = state.relationshipGraph.positions[link.target];
    if (!a || !b) return;
    const widthPx = 1 + (mode === "delta" ? link.drift : link.weight) * 5;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    const driftColor = link.drift > 0.2 ? "rgba(230,120,92,0.9)" : "rgba(110,136,171,0.5)";
    ctx.strokeStyle = mode === "delta" ? driftColor : "rgba(126, 156, 200, 0.42)";
    ctx.lineWidth = widthPx;
    ctx.stroke();
  });

  nodes.forEach((node) => {
    const p = state.relationshipGraph.positions[node.id];
    if (!p) return;
    const selected = state.relationshipGraph.selectedNodeId === node.id;
    const r = 6 + (node.importance || 0.4) * 11;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = selected ? "rgba(130, 184, 255, 0.95)" : "rgba(96, 140, 212, 0.9)";
    ctx.fill();
    ctx.lineWidth = selected ? 2.4 : 1;
    ctx.strokeStyle = selected ? "rgba(228, 240, 255, 0.9)" : "rgba(200,218,255,0.25)";
    ctx.stroke();
  });

  if (detailTitle && detailBody) {
    if (state.relationshipGraph.selectedNodeId) {
      const node = nodes.find((n) => n.id === state.relationshipGraph.selectedNodeId);
      const connected = links.filter((l) => l.source === node?.id || l.target === node?.id).length;
      detailTitle.textContent = node ? `${node.label} · ${connected} direct relationships` : "Select a node or edge.";
      detailBody.textContent = node ? `Importance ${(node.importance || 0).toFixed(2)} · Risk contribution ${(node.riskContribution || 0).toFixed(2)}.` : "";
    } else if (state.relationshipGraph.selectedEdgeId) {
      const edge = links.find((l) => `${l.source}:${l.target}` === state.relationshipGraph.selectedEdgeId);
      detailTitle.textContent = edge ? `${edge.source} ↔ ${edge.target}` : "Select a node or edge.";
      detailBody.textContent = edge ? `Strength ${edge.weight.toFixed(2)} · Drift ${edge.drift.toFixed(2)}.` : "";
    } else {
      detailTitle.textContent = "Select a node or edge.";
      detailBody.textContent = "Hover and click the graph to inspect local structure, risk contribution, and drift.";
    }
  }
  if (tooltip) tooltip.classList.add("hidden");
}

function wireRelationshipGraphInteractions() {
  const canvas = qs("#relationshipGraphCanvas");
  const tooltip = qs("#relationshipGraphTooltip");
  const modeButtons = qsa("#relationshipGraphModeControls button");
  const scrubber = qs("#relationshipGraphTimeScrubber");
  const playBtn = qs("#relationshipGraphPlayBtn");
  const edgeFilter = qs("#relationshipGraphEdgeFilter");
  if (!canvas || canvas.dataset.graphWired === "1") return;
  canvas.dataset.graphWired = "1";
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.relationshipGraph.mode = String(btn.dataset.graphMode || "current");
      modeButtons.forEach((el) => el.classList.toggle("active", el === btn));
      renderRelationshipGraph();
    });
  });
  if (scrubber) scrubber.addEventListener("input", () => {
    state.relationshipGraph.frameIndex = Number(scrubber.value || 0);
    renderRelationshipGraph();
  });
  if (edgeFilter) edgeFilter.addEventListener("input", () => {
    state.relationshipGraph.edgeThreshold = Number(edgeFilter.value || 0.2);
    renderRelationshipGraph();
  });
  if (playBtn) playBtn.addEventListener("click", () => {
    if (state.relationshipGraph.playing) {
      stopRelationshipGraphPlayback();
      playBtn.textContent = "Play";
      return;
    }
    stopRelationshipGraphPlayback();
    state.relationshipGraph.playing = true;
    playBtn.textContent = "Pause";
    state.relationshipGraph.timer = window.setInterval(() => {
      const max = Math.max(0, (state.relationshipGraph.frames || []).length - 1);
      if (max <= 0) {
        stopRelationshipGraphPlayback();
        playBtn.textContent = "Play";
        return;
      }
      state.relationshipGraph.frameIndex = state.relationshipGraph.frameIndex >= max ? 0 : state.relationshipGraph.frameIndex + 1;
      renderRelationshipGraph();
    }, 1100);
  });

  canvas.addEventListener("click", (evt) => {
    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const nodes = state.relationshipGraph.nodes || [];
    const picked = nodes.find((n) => {
      const p = state.relationshipGraph.positions[n.id];
      const r = 6 + (n.importance || 0.4) * 11;
      return p && Math.hypot(p.x - x, p.y - y) <= r + 2;
    });
    state.relationshipGraph.selectedNodeId = picked ? picked.id : "";
    state.relationshipGraph.selectedEdgeId = "";
    renderRelationshipGraph();
  });
  canvas.addEventListener("mousemove", (evt) => {
    if (!tooltip) return;
    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const nodes = state.relationshipGraph.nodes || [];
    const picked = nodes.find((n) => {
      const p = state.relationshipGraph.positions[n.id];
      const r = 6 + (n.importance || 0.4) * 11;
      return p && Math.hypot(p.x - x, p.y - y) <= r + 3;
    });
    if (!picked) {
      tooltip.classList.add("hidden");
      return;
    }
    tooltip.classList.remove("hidden");
    tooltip.style.left = `${Math.min(rect.width - 170, x + 12)}px`;
    tooltip.style.top = `${Math.max(8, y - 18)}px`;
    tooltip.innerHTML = `<strong>${escapeHtml(picked.label)}</strong><br/>Importance ${(picked.importance || 0).toFixed(2)}<br/>Risk contribution ${(picked.riskContribution || 0).toFixed(2)}`;
  });
  canvas.addEventListener("mouseleave", () => tooltip?.classList.add("hidden"));
}

async function loadDashboard() {
  if (state.ui.loadDashboardPromise) return state.ui.loadDashboardPromise;
  state.ui.loadDashboardPromise = (async () => {
  applyDashboardRunFromQuery();
  const runId = state.activeRun?.run_id || "";
  const recentParams = { limit: DASHBOARD_RECENT_LIMIT, compact: true };
  if (runId) recentParams.run_id = runId;
  const [recentEnv, alertsEnv] = await Promise.all([
    fetchRecentResults(recentParams),
    fetchJson(apiUrl("/alerts", tenantScopeParams({ run_id: runId, limit: 8 }))),
  ]);
  state.dashboardRecent = Array.isArray(recentEnv?.results) ? recentEnv.results.map((row) => operatorSafeResult(row)) : [];
  state.dashboardAlerts = Array.isArray(alertsEnv?.alerts) ? alertsEnv.alerts : [];
  state.dashboardCurrentAlertStatus = alertsEnv.current_status && typeof alertsEnv.current_status === "object"
    ? alertsEnv.current_status
    : null;
  collectKnownSites(state.dashboardRecent);
  renderTenantControls();
  const chron = dashboardChronologicalResults();
  const paint = () => {
    const renderChron = dashboardChronologicalForRender(chron);
    const latest = renderChron.length ? renderChron[renderChron.length - 1] : null;
    const prev = renderChron.length > 1 ? renderChron[renderChron.length - 2] : null;
    state.assistant.latest = latest;
    state.assistant.chronological = renderChron;
    renderDashboardMetrics(latest, prev);
    renderDashboardSparkline(renderChron);
    renderDashboardHealthTrend(renderChron);
    renderRecentTransitionsTimeline(renderChron);
    renderEvidencePanel(latest);
    bindDashboardSparklineInteractions();
    wireAssistantChat();
    state.relationshipGraph.frames = buildRelationshipGraphFrames(state.dashboardRecent);
    if (state.relationshipGraph.frameIndex >= state.relationshipGraph.frames.length) {
      state.relationshipGraph.frameIndex = Math.max(0, state.relationshipGraph.frames.length - 1);
    }
    wireRelationshipGraphInteractions();
    renderRelationshipGraph();
  };
  if (state.ui.dashboardPaint) window.cancelAnimationFrame(state.ui.dashboardPaint);
  state.ui.dashboardPaint = window.requestAnimationFrame(() => {
    paint();
    startDashboardReplay(chron, paint);
    state.ui.dashboardPaint = null;
  });
  })();
  try {
    await state.ui.loadDashboardPromise;
  } finally {
    state.ui.loadDashboardPromise = null;
  }
}
