# Neraium Demo - Quick Start

## Run the Demo

```bash
python run_demo.py
```

That's it. This launches a local web app showing the Neraium synthetic demo.

Your browser will open automatically. If not, visit:
```
http://localhost:7860
```

## What You'll See

The demo plays through 120 timesteps of synthetic telemetry showing a complete system lifecycle:

1. **Baseline** (0-20 steps)  
   Stable operation, low drift, high confidence

2. **Drift Watch** (20-35 steps)  
   Early signs of change, system starting to diverge

3. **Transition Active** (35-55 steps)  
   Sustained structural change, drift increasing  
   Gate decision approaches threshold

4. **Reorganization** (55-80 steps)  
   Major structural shift confirmed  
   System undergoes internal reorganization

5. **Recovery** (80-120 steps)  
   System reaches new equilibrium  
   New stable baseline established

## Controls

- **Frame slider** — Jump to any point in the timeline
- **Speed slider** — Adjust playback speed (0.1x to 1.5x)
- **Play** — Start automatic playback from current frame
- **Pause** — Pause playback
- **Restart** — Return to beginning

## What Each Panel Shows

### Header
Current state metrics: confidence score, phase, frame counter

### Verdict Card (Top)
Gate decision (ADMITTED / SUPPRESSED) with supporting confidence and phase labels

### System Geometry (Middle)
Visualization of structural relationships between sensors, showing deformation as drift increases

### Timeline Strip
Historical phase progression at a glance

### Reasoning (Bottom Left)
Analytical explanation of:
- What signal was observed
- How the system assessed it
- What action is appropriate

### Evidence Record (Bottom Center)
Ledger of admitted events and transitions

### Tetrahedral State (Right)
3D visualization of system position in structural state space:
- **STRUCTURAL axis**: structural integrity vs drift
- **RELATIONAL axis**: relationship stability vs instability
- **TRANSITION axis**: transition pressure
- **TEMPORAL axis**: temporal consistency

The orange point shows current position; the trail shows recent history.

## Optional Flags

### Generate a Shareable Link
```bash
python run_demo.py --share
```

This requires cloudflared or ngrok to be installed. A public URL will be printed — others can visit it while the server is running.

### Show Help
```bash
python run_demo.py --help
```

## Key Insights

The demo is designed to show:

1. **Deterministic progression** — Not random. Shows how drift builds systematically.

2. **Gate coherence** — The verdict (admitted/suppressed) changes only when signal persistence and corroboration justify it.

3. **Structural geometry** — The tetrahedral visualization shows the system moving in state space as it changes.

4. **Reasoning transparency** — Each decision is explained in plain language with evidence.

5. **Visual integration** — Charts, geometry, and verdict all stay synchronized.

## Next Steps

- **To use in production**: Read [PRODUCTION_READINESS_MEASURED.md](./PRODUCTION_READINESS_MEASURED.md)
- **To integrate into your app**: See [INTEGRATION_GUIDE.py](./INTEGRATION_GUIDE.py)
- **To understand the engine**: See [neraium_core/engine](./neraium_core/engine)
- **To validate on your data**: See the validation tools in [scripts/](./scripts)

---

**Neraium** detects structural instability before component failures by monitoring how sensor signals relate to each other, not individual sensor values.
