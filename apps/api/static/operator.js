(function () {
  const el = (id) => document.getElementById(id);

  const scopeForm = el("scopeForm");
  const customerIdInput = el("customerId");
  const runIdInput = el("runId");
  const seedDemoBtn = el("seedDemoBtn");
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
    return String(value);
  }

  function toConfidence(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "-";
    return `${(num * 100).toFixed(1)}%`;
  }

  function renderStateView(state) {
    const meta = el("stateMeta");
    const riskView = el("riskView");
    const recView = el("recommendationView");
    const explanation = el("explanationText");
    const eventsList = el("eventsList");
    const memoryView = el("memoryRecallView");
    const topMatchesList = el("topMatchesList");

    clearNode(meta);
    clearNode(riskView);
    clearNode(recView);
    clearNode(eventsList);
    clearNode(memoryView);
    clearNode(topMatchesList);

    if (!state) {
      addPair(meta, "Status", "No state available");
      explanation.textContent = "No explanation loaded.";
      return;
    }

    const session = state.session || {};
    const risk = state.risk_assessment || {};
    const recommendation = state.operational_recommendation || {};
    const recommendationStatus = recommendation.status || {};
    const memoryRecall = state.memory_recall || {};
    const novelty = memoryRecall.novelty || {};
    const nearest = memoryRecall.nearest_match || {};
    const topMatches = Array.isArray(memoryRecall.top_matches) ? memoryRecall.top_matches : [];

    addPair(meta, "Timestamp", toText(state.timestamp));
    addPair(meta, "Cycle", toText(state.cycle));
    addPair(meta, "Asset ID", toText(session.asset_id));
    addPair(meta, "Site ID", toText(session.site_id));
    addPair(meta, "Run ID", toText(session.run_id));

    addPair(riskView, "Risk level", toText(risk.risk_level));
    addPair(riskView, "Trend", toText(risk.trend));
    addPair(riskView, "Latest instability", toText(risk.latest_instability));

    addPair(recView, "Recommended next step", toText(recommendation.recommended_action, "No recommendation"));
    addPair(recView, "Why this is being recommended", toText(recommendation.rationale));
    addPair(recView, "Recommendation confidence", toConfidence(recommendation.recommendation_confidence));
    addPair(recView, "Operator note", toText(recommendation.operator_note));
    addPair(recView, "Recommendation available", toText(recommendationStatus.available));

    explanation.textContent = toText(state.explanation_text, "No explanation available.");

    const events = Array.isArray(state.events) ? state.events : [];
    if (events.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No events";
      eventsList.appendChild(li);
    } else {
      events.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = toText(item);
        eventsList.appendChild(li);
      });
    }

    addPair(memoryView, "Novelty status", novelty.is_novel ? "Novel" : "Recalled pattern");
    addPair(memoryView, "Novelty reason", toText(novelty.reason));
    addPair(memoryView, "Nearest match", nearest.found ? toText(nearest.summary) : "No historical match");
    addPair(memoryView, "Nearest match similarity", toText(nearest.similarity));

    if (topMatches.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No top matches available.";
      topMatchesList.appendChild(li);
    } else {
      topMatches.forEach((match) => {
        const li = document.createElement("li");
        li.textContent = `${toText(match.summary)} (similarity ${toText(match.similarity)}, scope ${toText(match.scope)})`;
        topMatchesList.appendChild(li);
      });
    }
  }

  function renderTimeline(historyRows) {
    const tbody = el("timelineBody");
    clearNode(tbody);
    historyRows.forEach((entry) => {
      const risk = entry.risk_assessment || {};
      const rec = entry.operational_recommendation || {};
      const recStatus = rec.status || {};
      const mem = entry.memory_recall || {};
      const novelty = mem.novelty || {};
      const events = Array.isArray(entry.events) ? entry.events.join(", ") : "-";
      const patternState = novelty.is_novel ? "novel" : "recalled";

      const tr = document.createElement("tr");
      [
        toText(entry.cycle),
        toText(entry.timestamp),
        toText(risk.risk_level),
        toText(recStatus.available),
        toConfidence(rec.recommendation_confidence),
        toText(events),
        patternState,
      ].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
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
      statusLine.textContent = `Loaded ${historyEnvelope.count || 0} history rows.`;
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

  loadWorkflow();
})();
