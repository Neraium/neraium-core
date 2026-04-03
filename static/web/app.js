const scopeSelector = document.getElementById("scopeSelector");
const entitySelector = document.getElementById("entitySelector");
let timelineChart;

const scopeListEndpoint = {
  plant: "/plants",
  zone: "/zones",
  room: "/rooms",
  subsystem: "/subsystems",
  facility: "/facilities/fac-chi-1/state",
  portfolio: "/portfolio/summary",
};

function fmt(value) {
  return value ?? "--";
}

function badge(state) {
  return `<span class="state-badge state-${state}">${state.replaceAll("_", " ")}</span>`;
}

async function fetchJson(path, options = undefined) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path} failed`);
  return response.json();
}

function renderRanking(targetId, rows, labelKey) {
  const root = document.getElementById(targetId);
  root.innerHTML = rows.slice(0, 6).map((row) => `
    <div class="rank-row">
      <div>
        <strong>${fmt(row[labelKey] || row.scope_id)}</strong><br />
        <small>${fmt(row.inefficiency_summary || row.cross_system_driver)}</small>
      </div>
      <div>${badge(row.state)}</div>
    </div>
  `).join("");
}

function drawTimeline(timeline) {
  const labels = timeline.map((row) => new Date(row.timestamp).toLocaleTimeString());
  const drift = timeline.map((row) => row.structural_drift_score);
  const ineff = timeline.map((row) => (row.energy_inefficiency + row.climate_inefficiency + row.process_inefficiency + row.biological_inefficiency) / 4);
  const ctx = document.getElementById("timelineChart");
  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Drift", data: drift, borderColor: "#2e7d4b", backgroundColor: "#2e7d4b22", fill: true, tension: 0.35 },
        { label: "Inefficiency", data: ineff, borderColor: "#c78f3b", backgroundColor: "#c78f3b22", fill: true, tension: 0.35 }
      ]
    },
    options: { responsive: true, animation: false }
  });
}

async function populateEntities(scope) {
  if (scope === "portfolio") {
    entitySelector.innerHTML = `<option value="portfolio-global">Portfolio</option>`;
    return;
  }
  if (scope === "facility") {
    const facilities = (await fetchJson("/portfolio/summary")).facilities || [];
    entitySelector.innerHTML = facilities.map((row) => `<option value="${row.scope_id}">${row.scope_id}</option>`).join("");
    return;
  }
  const rows = await fetchJson(scopeListEndpoint[scope]);
  entitySelector.innerHTML = rows.map((row) => `<option value="${row.scope_id}">${row.scope_id}</option>`).join("");
}

function detailEndpoint(scope, entityId) {
  if (scope === "plant") return [`/plants/${entityId}/state`, `/plants/${entityId}/timeline`];
  if (scope === "zone") return [`/zones/${entityId}/state`, `/zones/${entityId}/timeline`];
  if (scope === "room") return [`/rooms/${entityId}/state`, `/rooms/${entityId}/timeline`];
  if (scope === "subsystem") return [`/subsystems/${entityId}/state`, `/subsystems/${entityId}/timeline`];
  if (scope === "facility") return [`/facilities/${entityId}/state`, `/facilities/${entityId}/timeline`];
  return ["/portfolio/summary", null];
}

async function renderPage() {
  const setup = await fetchJson("/setup/summary");
  document.getElementById("setupSummary").innerHTML = `
    <h2>Setup / Readiness</h2>
    <div class="metric-row"><span>Sources</span><strong>${setup.telemetry_source_count}</strong></div>
    <div class="metric-row"><span>Signals</span><strong>${setup.signal_count_discovered}</strong></div>
    <div class="metric-row"><span>Plants / zones / rooms</span><strong>${setup.plants_discovered} / ${setup.zones_discovered} / ${setup.rooms_discovered}</strong></div>
    <div class="metric-row"><span>Subsystems / facilities / sites</span><strong>${setup.subsystems_discovered} / ${setup.facilities_discovered} / ${setup.portfolio_sites_discovered}</strong></div>
    <div class="metric-row"><span>Model ready</span><strong>${setup.model_ready ? "Ready" : "Learning"}</strong></div>
    <div class="metric-row"><span>Readiness</span><strong>${setup.readiness_percent}%</strong></div>
  `;

  const fleet = await fetchJson("/fleet/summary");
  const scope = scopeSelector.value;
  await populateEntities(scope);
  const selectedId = entitySelector.value;
  const [statePath, timelinePath] = detailEndpoint(scope, selectedId);
  const statePayload = await fetchJson(statePath);
  const state = statePayload.scope_type ? statePayload : (statePayload.portfolio || statePayload);

  document.getElementById("heroSentence").textContent =
    `This entity is drifting relative to its expected structural behavior and its peers, primarily due to ${state.cross_system_driver || "cross-system inefficiency"}, with likely impact in ${state.estimated_time_to_impact_hours ?? "unknown"} hours unless action is taken.`;

  document.getElementById("summaryPanel").innerHTML = `
    <p class="pill">${scope.toUpperCase()} VIEW</p>
    <p>${badge(state.state)}</p>
    <p>${state.operator_message || state.headline}</p>
    <p class="muted">${state.threshold_breach_message || "No individual sensor exceeded thresholds at this point"}</p>
    <p class="muted">Plant mode: ${state.plant_view_mode || "not_applicable"}</p>
  `;

  const decisions = await fetchJson("/decisions/latest");
  const decision = decisions.find((row) => row.scope_type === scope && row.scope_id === selectedId) || decisions[0];
  document.getElementById("decisionPanel").innerHTML = decision ? `
    <p>${badge(decision.state)}</p>
    <p><strong>${decision.dominant_relationship_shift}</strong></p>
    <p>${decision.cross_system_driver}</p>
    <p>${decision.inefficiency_summary}</p>
    <p><strong>Recommended action:</strong> ${decision.recommended_action}</p>
    <p><strong>Estimated time to impact:</strong> ${decision.estimated_time_to_impact}</p>
  ` : "<p>No decision available.</p>";

  const riskRows = scope === "plant" ? fleet.ranked_plants_by_local_risk
    : scope === "zone" ? fleet.ranked_zones_by_risk
    : scope === "room" ? fleet.ranked_rooms_by_risk
    : scope === "subsystem" ? fleet.ranked_subsystems_by_drift_contribution
    : fleet.ranked_facilities_by_instability;
  renderRanking("riskRanking", riskRows, "scope_id");
  renderRanking("inefficiencyRanking", fleet.highest_inefficiency_areas, "scope_id");

  const compare = await fetchJson(`/compare?scope_type=${scope}&scope_id=${selectedId}`);
  document.getElementById("comparisonPanel").innerHTML = `
    <p><strong>Ranking position:</strong> ${compare.summary.ranking_position} of ${compare.summary.peer_count}</p>
    <p><strong>Relative drift:</strong> ${compare.summary.relative_drift}</p>
    <p><strong>Relative inefficiency:</strong> ${compare.summary.relative_inefficiency}</p>
    <p>${(compare.summary.likely_causal_differences || []).join(" | ")}</p>
  `;

  document.getElementById("driverPanel").innerHTML = `
    <p><strong>${state.dominant_relationship_shift}</strong></p>
    <p>${state.cross_system_driver}</p>
    <p>${state.biological_risk}</p>
    <p>${state.inefficiency_summary}</p>
  `;
  document.getElementById("actionsPanel").innerHTML = `
    <p><strong>${state.recommended_action}</strong></p>
    <p>${state.threshold_breach_message}</p>
    <p>Energy ${state.energy_inefficiency} | Climate ${state.climate_inefficiency} | Process ${state.process_inefficiency} | Biological ${state.biological_inefficiency}</p>
  `;
  document.getElementById("detailPanel").innerHTML = `
    <div class="detail-grid">
      <div class="detail-card"><strong>Scope</strong><br />${state.scope_type}</div>
      <div class="detail-card"><strong>Parent</strong><br />${fmt(state.parent_scope_id)}</div>
      <div class="detail-card"><strong>Time in state</strong><br />${state.time_in_state_minutes} minutes</div>
      <div class="detail-card"><strong>Confidence</strong><br />${state.confidence}</div>
      <div class="detail-card"><strong>Peer delta</strong><br />${state.comparison_summary.peer_median_delta}</div>
      <div class="detail-card"><strong>Best performer delta</strong><br />${state.comparison_summary.best_performer_delta}</div>
    </div>
  `;

  if (timelinePath) {
    const timeline = await fetchJson(timelinePath);
    drawTimeline(timeline.timeline);
  } else {
    drawTimeline([{ timestamp: new Date().toISOString(), structural_drift_score: state.structural_drift_score, energy_inefficiency: state.energy_inefficiency, climate_inefficiency: state.climate_inefficiency, process_inefficiency: state.process_inefficiency, biological_inefficiency: state.biological_inefficiency }]);
  }
}

scopeSelector.addEventListener("change", renderPage);
entitySelector.addEventListener("change", renderPage);
renderPage().catch((error) => {
  document.body.insertAdjacentHTML("beforeend", `<pre>${error.message}</pre>`);
});
