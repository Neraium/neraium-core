(function () {
  const el = (id) => document.getElementById(id);

  const scopeForm = el("scopeForm");
  const customerIdInput = el("customerId");
  const runIdInput = el("runId");
  const seedDemoBtn = el("seedDemoBtn");
  const loadAssistantBtn = el("loadAssistantBtn");
  const assistantExplainMode = el("assistantExplainMode");
  const reportModeSelect = el("reportMode");
  const generateReportBtn = el("generateReportBtn");
  const copyReportBtn = el("copyReportBtn");
  const statusLine = el("statusLine");

  async function fetchJson(path, params, options) {
    const url = new URL(path, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && String(value).length > 0) {
          url.searchParams.set(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), options);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function addPair(node, label, value) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    node.appendChild(dt);
    node.appendChild(dd);
  }

  function toText(value, fallback = "-") {
    if (value === null || value === undefined || value === "") return fallback;
    if (typeof value === "boolean") return value ? "Yes" : "No";
    return String(value);
  }

  function toConfidence(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "Not available";
    return `${(num * 100).toFixed(1)}%`;
  }

  function titleCase(input) {
    return toText(input, "Unknown").toLowerCase().replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
  }

  function normalizeRiskLevel(rawRisk) {
    const risk = String(rawRisk || "UNKNOWN").toUpperCase();
    if (["LOW", "MEDIUM", "HIGH"].includes(risk)) return risk;
    return "UNKNOWN";
  }

  function riskClass(level) {
    if (level === "LOW") return "risk-low";
    if (level === "MEDIUM") return "risk-medium";
    if (level === "HIGH") return "risk-high";
    return "risk-unknown";
  }

  function describeTrend(trend) {
    const normalized = String(trend || "").toLowerCase();
    if (normalized.includes("rise") || normalized.includes("increas") || normalized.includes("up")) {
      return "↗ Rising risk trend";
    }
    if (normalized.includes("fall") || normalized.includes("improv") || normalized.includes("down")) {
      return "↘ Improving trend";
    }
    if (normalized.includes("stable") || normalized.includes("flat")) {
      return "→ Stable trend";
    }
    return "→ Trend currently unknown";
  }

  function recommendationAvailability(status) {
    if (typeof status === "boolean") return status ? "Available" : "Not available";
    return toText(status, "Not available");
  }

  function appendEmptyItem(listNode, text) {
    const li = document.createElement("li");
    li.textContent = text;
    listNode.appendChild(li);
  }

  function renderPatternSummary(memoryRecall) {
    const container = el("patternSummary");
    clearNode(container);

    const novelty = memoryRecall.novelty || {};
    const nearest = memoryRecall.nearest_match || {};
    const isNovel = novelty.is_novel === true;
    const similarity = nearest.similarity;
    const patternFamily = nearest.family || nearest.pattern_family;

    const headline = document.createElement("p");
    headline.className = "summary-title";
    if (isNovel) {
      headline.textContent = "This pattern appears novel.";
    } else {
      headline.textContent = "This pattern has been seen before.";
    }
    container.appendChild(headline);

    const reason = document.createElement("p");
    reason.className = "summary-item";
    reason.textContent = toText(novelty.reason, "No novelty rationale available yet.");
    container.appendChild(reason);

    const nearestText = document.createElement("p");
    nearestText.className = "summary-item";
    nearestText.textContent = nearest.found
      ? `Nearest prior match: ${toText(nearest.summary, "Summary unavailable")}.`
      : "No prior pattern match found.";
    container.appendChild(nearestText);

    const similarityText = document.createElement("p");
    similarityText.className = "summary-item";
    if (Number.isFinite(Number(similarity))) {
      similarityText.textContent = `Similarity score: ${toConfidence(similarity)}.`;
    } else {
      similarityText.textContent = "Similarity score: Not available.";
    }
    container.appendChild(similarityText);

    const familyText = document.createElement("p");
    familyText.className = "summary-item";
    familyText.textContent = patternFamily
      ? `Pattern family: ${titleCase(patternFamily)}.`
      : "Pattern family: Not identified.";
    container.appendChild(familyText);
  }

  function renderStateView(state) {
    const meta = el("stateMeta");
    const riskView = el("riskView");
    const recView = el("recommendationView");
    const recommendationAction = el("recommendationAction");
    const explanation = el("explanationText");
    const eventsList = el("eventsList");
    const topMatchesList = el("topMatchesList");
    const riskBadge = el("riskBadge");
    const riskTrend = el("riskTrend");

    clearNode(meta);
    clearNode(riskView);
    clearNode(recView);
    clearNode(eventsList);
    clearNode(topMatchesList);

    if (!state) {
      addPair(meta, "Status", "No state available.");
      recommendationAction.textContent = "No recommendation available yet.";
      explanation.textContent = "No explanation available.";
      renderPatternSummary({});
      appendEmptyItem(eventsList, "No events captured yet.");
      appendEmptyItem(topMatchesList, "No prior pattern match found.");
      riskBadge.textContent = "UNKNOWN";
      riskBadge.className = "risk-badge risk-unknown";
      riskTrend.textContent = "Trend: Unknown";
      return;
    }

    const session = state.session || {};
    const risk = state.risk_assessment || {};
    const recommendation = state.operational_recommendation || {};
    const recommendationStatus = recommendation.status || {};
    const memoryRecall = state.memory_recall || {};
    const topMatches = Array.isArray(memoryRecall.top_matches) ? memoryRecall.top_matches : [];

    addPair(meta, "Timestamp", toText(state.timestamp, "Not available"));
    addPair(meta, "Cycle", toText(state.cycle, "Not available"));
    addPair(meta, "Customer", toText(session.customer_id, "Not available"));
    addPair(meta, "Site", toText(session.site_id, "Not available"));
    addPair(meta, "Asset", toText(session.asset_id, "Not available"));
    addPair(meta, "Run", toText(session.run_id, "Not available"));

    const normalizedRisk = normalizeRiskLevel(risk.risk_level);
    riskBadge.textContent = normalizedRisk;
    riskBadge.className = `risk-badge ${riskClass(normalizedRisk)}`;
    riskTrend.textContent = describeTrend(risk.trend);

    addPair(riskView, "Current risk", titleCase(risk.risk_level || "UNKNOWN"));
    addPair(riskView, "Trend", titleCase(risk.trend || "unknown"));
    addPair(riskView, "Latest instability", toText(risk.latest_instability, "Not available"));

    recommendationAction.textContent = toText(
      recommendation.recommended_action,
      "No recommendation available yet.",
    );
    addPair(recView, "Target", toText(recommendation.target || recommendation.target_component, "Not specified"));
    addPair(recView, "Confidence", toConfidence(recommendation.recommendation_confidence));
    addPair(recView, "Availability", recommendationAvailability(recommendationStatus.available));
    addPair(recView, "Operator note", toText(recommendation.operator_note, "Use standard operating judgment before acting."));
    addPair(recView, "Rationale", toText(recommendation.rationale, "No rationale provided yet."));

    explanation.textContent = toText(state.explanation_text, "No explanation available.");

    const events = Array.isArray(state.events) ? state.events : [];
    if (events.length === 0) {
      appendEmptyItem(eventsList, "No events captured for this cycle.");
    } else {
      events.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = titleCase(item);
        eventsList.appendChild(li);
      });
    }

    renderPatternSummary(memoryRecall);

    if (topMatches.length === 0) {
      appendEmptyItem(topMatchesList, "No prior pattern match found.");
    } else {
      topMatches.forEach((match) => {
        const li = document.createElement("li");
        const scope = match.scope ? ` · Scope ${toText(match.scope)}` : "";
        li.textContent = `${toText(match.summary, "Historical match")}${scope} · Similarity ${toConfidence(match.similarity)}`;
        topMatchesList.appendChild(li);
      });
    }
  }

  function makeStatusPill(text) {
    const span = document.createElement("span");
    span.className = "timeline-status";
    span.textContent = text;
    return span;
  }

  function renderTimeline(historyRows) {
    const tbody = el("timelineBody");
    clearNode(tbody);

    if (!Array.isArray(historyRows) || historyRows.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No recent history available.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    historyRows.forEach((entry) => {
      const risk = entry.risk_assessment || {};
      const rec = entry.operational_recommendation || {};
      const recStatus = rec.status || {};
      const mem = entry.memory_recall || {};
      const novelty = mem.novelty || {};
      const events = Array.isArray(entry.events) && entry.events.length > 0
        ? entry.events.slice(0, 3).map((event) => titleCase(event)).join(", ")
        : "No events";
      const patternState = novelty.is_novel ? "Novel pattern" : "Recalled pattern";
      const recommendationState = recommendationAvailability(recStatus.available);

      const tr = document.createElement("tr");
      const values = [
        toText(entry.cycle, "-"),
        toText(entry.timestamp, "-"),
        titleCase(risk.risk_level || "unknown"),
        recommendationState,
        toConfidence(rec.recommendation_confidence),
        events,
        patternState,
      ];

      values.forEach((value, idx) => {
        const td = document.createElement("td");
        if (idx === 2 || idx === 3 || idx === 6) {
          td.appendChild(makeStatusPill(value));
        } else {
          td.textContent = value;
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  async function loadAssistantPanels(scope) {
    const summaryEl = el("assistantSummary");
    const explainEl = el("assistantExplain");
    const handoffEl = el("assistantHandoff");
    const explainMode = assistantExplainMode && assistantExplainMode.value ? assistantExplainMode.value : "why_recommended";

    const body = {
      customer_id: scope.customer_id,
      run_id: scope.run_id,
      mode: explainMode,
      history_limit: 20,
    };

    const [summary, explain, handoff] = await Promise.all([
      fetchJson("/assistant/summary", null, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
      fetchJson("/assistant/explain", null, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
      fetchJson("/assistant/handoff", null, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    ]);

    summaryEl.textContent = toText(summary.text, "No assistant summary available.");
    explainEl.textContent = toText(explain.text, "No assistant explanation available.");
    handoffEl.textContent = toText(handoff.text, "No handoff note available.");
  }

  async function generateReport(scope) {
    const reportText = el("reportText");
    const mode = reportModeSelect && reportModeSelect.value ? reportModeSelect.value : "client_report";
    const payload = {
      customer_id: scope.customer_id,
      run_id: scope.run_id,
      mode,
      history_limit: 20,
    };
    const report = await fetchJson("/assistant/report", null, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    reportText.textContent = toText(report.report_text, "No report available.");
  }

  async function copyReportText() {
    const reportText = el("reportText");
    const text = toText(reportText && reportText.textContent, "");
    if (!text || text === "No report generated.") {
      throw new Error("Generate a report first.");
    }
    await navigator.clipboard.writeText(text);
  }

  async function ingestDemoFrames(scope) {
    const start = Date.now();
    const payloads = [
      { pressure: 49.5, temperature: 80.1 },
      { pressure: 56.0, temperature: 84.2 },
      { pressure: 68.3, temperature: 90.7 },
    ];
    for (let i = 0; i < payloads.length; i += 1) {
      const timestamp = new Date(start + i * 60_000).toISOString();
      await fetchJson(
        "/ingest/frame",
        scope,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestamp,
            customer_id: scope.customer_id,
            site_id: "site-operator-demo",
            asset_id: "asset-operator-demo",
            sensor_values: payloads[i],
          }),
        },
      );
    }
  }

  async function loadWorkflow() {
    const scope = {
      customer_id: customerIdInput.value.trim(),
      run_id: runIdInput.value.trim(),
    };
    statusLine.textContent = "Loading current state and history...";

    try {
      const [stateEnvelope, historyEnvelope] = await Promise.all([
        fetchJson("/state", scope),
        fetchJson("/history", { ...scope, limit: 20 }),
      ]);
      renderStateView(stateEnvelope.state || null);
      renderTimeline(Array.isArray(historyEnvelope.history) ? historyEnvelope.history : []);
      await loadAssistantPanels(scope);
      await generateReport(scope);
      statusLine.textContent = `Loaded ${historyEnvelope.count || 0} history rows, assistant panels, and report.`;
    } catch (error) {
      statusLine.textContent = `Failed to load workflow: ${String(error.message || error)}`;
    }
  }

  scopeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await loadWorkflow();
  });

  seedDemoBtn.addEventListener("click", async () => {
    const scope = {
      customer_id: customerIdInput.value.trim(),
      run_id: runIdInput.value.trim(),
    };
    statusLine.textContent = "Seeding demo frames...";
    try {
      await ingestDemoFrames(scope);
      await loadWorkflow();
      statusLine.textContent = "Demo frames ingested and workflow refreshed.";
    } catch (error) {
      statusLine.textContent = `Failed to seed demo: ${String(error.message || error)}`;
    }
  });

  loadAssistantBtn.addEventListener("click", async () => {
    const scope = {
      customer_id: customerIdInput.value.trim(),
      run_id: runIdInput.value.trim(),
    };
    statusLine.textContent = "Refreshing assistant panels...";
    try {
      await loadAssistantPanels(scope);
      statusLine.textContent = "Assistant panels refreshed.";
    } catch (error) {
      statusLine.textContent = `Failed to load assistant panels: ${String(error.message || error)}`;
    }
  });

  generateReportBtn.addEventListener("click", async () => {
    const scope = {
      customer_id: customerIdInput.value.trim(),
      run_id: runIdInput.value.trim(),
    };
    statusLine.textContent = "Generating report...";
    try {
      await generateReport(scope);
      statusLine.textContent = "Report generated.";
    } catch (error) {
      statusLine.textContent = `Failed to generate report: ${String(error.message || error)}`;
    }
  });

  copyReportBtn.addEventListener("click", async () => {
    statusLine.textContent = "Copying report...";
    try {
      await copyReportText();
      statusLine.textContent = "Report copied to clipboard.";
    } catch (error) {
      statusLine.textContent = `Failed to copy report: ${String(error.message || error)}`;
    }
  });

  loadWorkflow();
})();
