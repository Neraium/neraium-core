# Neraium water-infrastructure contractor demo walkthrough

This walkthrough packages the current product into **one deterministic, contractor-facing flow** using a pump degradation scenario.

## 1) What this demo covers

The walkthrough explicitly demonstrates:

- pump/water degradation progression (`pressure`, `flow_rate`, `vibration`, `temperature`)
- early warning in the trend/risk timeline
- advisory recommendation (non-autonomous)
- explanation text
- memory recall signal (when a similar prior pattern is present)
- generated **client report**
- generated **technician summary**
- generated **handoff note**

Scenario source file:

- `examples/demo/water_infrastructure_scenario.json`

---

## 2) Run the demo (deterministic seed)

### Start API

```bash
uvicorn apps.api.main:app --port 8000
```

### Seed the water scenario + generate outputs bundle

```bash
python tools/seed_water_infrastructure_demo.py \
  --base-url http://127.0.0.1:8000 \
  --customer-id contractor-water-demo \
  --run-id run-water-pump-demo-v1
```

By default this writes a reproducible output bundle to:

- `reports/water_demo_outputs.json`

It also prints short excerpts for recommendation, explanation, memory recall, and reports.

---

## 3) Route and click path (operator walkthrough)

Open:

- `http://127.0.0.1:8000/operator`

Set scope:

- `customer_id`: `contractor-water-demo`
- `run_id`: `run-water-pump-demo-v1`

Then walk the contractor through the page top-to-bottom:

1. **Current Structural State**  
   Show timestamp/cycle and identify this as a live pump run.
2. **Risk Assessment**  
   Point out trend moving from stable into drift/instability.
3. **Operational Recommendation**  
   Highlight this is advisory and confidence-scored.
4. **Explanation**  
   Read why recommendation was triggered from observed signal behavior.
5. **Events**  
   Confirm event list aligns with instability onset.
6. **Memory Recall**  
   Show novelty vs recalled pattern; if recalled, call out nearest match summary/similarity.
7. **Timeline table**  
   Scroll through cycles to show phase progression (stable → drift → instability → recommendation).
8. **Assistant panel**  
   Click **Refresh Assistant** if needed and show:
   - current summary
   - explanation mode output
   - handoff text
9. **Generate report**  
   In "Generate Report" panel:
   - choose `client_report` → click **Generate**
   - choose `technician_summary` → click **Generate**
   - choose `handoff_note` → click **Generate**

---

## 4) Expected phase progression from the seeded scenario

The deterministic scenario follows this sequence:

- **stable (minutes 0–1):** pressure/flow near baseline, low vibration and temperature.
- **drift (minutes 2–3):** pressure and flow trend down, vibration/temperature trend up.
- **instability (minutes 4–5):** sharper deterioration, elevated vibration and heat.
- **recommendation (minute 6):** advisory recommendation is expected to be present.
- **memory_recall (minute 7):** repeated instability signature supports nearest-pattern recall where available.

---

## 5) Example outputs to include in the contractor talk-track

Use these as reference snippets (actual wording can vary by engine state/config):

### Recommendation (example)

> Recommended next step: Inspect pump-17A for cavitation and suction-side restriction; schedule near-term maintenance window.

### Explanation (example)

> Recommendation issued because pressure/flow declined while vibration/temperature rose across consecutive cycles, indicating escalating structural instability.

### Memory recall (example)

> Pattern status: Recalled pattern (or Novel if no close match). Nearest match similarity shown in the memory panel.

### Client report (example)

> Client Report: Current risk is elevated with rising instability indicators. Recommended action is advisory inspection and planned maintenance to reduce service disruption risk.

### Technician summary (example)

> Technician Summary: Pressure + flow degradation with elevated vibration/temperature suggest pump wear, possible cavitation, or intake restriction. Prioritize vibration source check, suction path inspection, and thermal verification.

### Handoff note (example)

> Handoff Note: Incoming shift should monitor pump-17A trend each cycle, confirm recommendation execution status, and escalate if instability continues or recommendation confidence increases.

---

## 6) Contractor-facing narrative (recommended)

Use this concise narrative in live sessions:

1. "We start with stable pump behavior and baseline telemetry."
2. "Neraium detects early drift before hard failure conditions."
3. "As instability builds, the system raises risk and provides an advisory next step with rationale."
4. "We can explain exactly what changed in the signals and whether this resembles a prior known pattern."
5. "From the same run, we generate customer-safe reporting, technician-ready detail, and a shift handoff note."

This keeps the demo operationally grounded for contractors while showing full end-to-end value without changing core engine behavior.
