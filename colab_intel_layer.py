# %% [markdown]
# # Neraium intelligence layer — Google Colab
# Run each section in order (or run all).

# %% 1) Clone repo (change URL if your repo is elsewhere) and go to folder
import subprocess
import sys
import os

# Option A: Clone from git (uncomment and set your repo URL)
# !git clone https://github.com/YOUR_USER/neraium-core-1.git
# %cd neraium-core-1

# Option B: If you uploaded the project to Drive or Colab, just cd to that folder
# %cd /content/neraium-core-1   # or your path

# We'll assume we're already in the project root (e.g. after clone or upload)
project_root = os.getcwd()
print("Project root:", project_root)

# %% 2) Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "pytest"])
# If your project has pyproject.toml and you're in project root:
if os.path.isfile(os.path.join(project_root, "pyproject.toml")):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy"])
print("Done installing.")

# %% 3) Run the intelligence layer demo (no pytest — works without tests folder)
import math
import tempfile
import sys
import os

# Add project root so we can import neraium_core
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from neraium_core.alignment import StructuralEngine

print("Neraium intelligence layer — quick demo\n")
with tempfile.TemporaryDirectory() as d:
    engine = StructuralEngine(
        baseline_window=25,
        recent_window=10,
        regime_store_path=os.path.join(d, "r.json"),
    )
    for t in range(45):
        base = math.sin(0.05 * t)
        frame = {
            "timestamp": str(t),
            "site_id": "demo",
            "asset_id": "asset-1",
            "sensor_values": {"s1": base, "s2": 0.95 * base, "s3": 1.0 * base},
        }
        out = engine.process_frame(frame)

    print("Sample output (last frame):")
    print("  interpreted_state:", out.get("interpreted_state"))
    print("  structural_drift_score:", out.get("structural_drift_score"))
    print("  latest_instability:", out.get("latest_instability"))
    print("  confidence_score:", out.get("confidence_score"))
    print("  baseline_mode:", out.get("baseline_mode"))
    print("  causal_attribution.top_drivers:", out.get("causal_attribution", {}).get("top_drivers"))
    print("  dominant_driver:", out.get("dominant_driver"))
    print("  data_quality_summary.gate_passed:", out.get("data_quality_summary", {}).get("gate_passed"))
    print("  active_sensor_count:", out.get("active_sensor_count"))
    print("  regime_memory_state:", out.get("regime_memory_state"))
print("\nDone.")

# %% 4) Optional: run pytest (only if repo has tests/ and neraium_core/)
if os.path.isdir(os.path.join(project_root, "tests")) and os.path.isdir(os.path.join(project_root, "neraium_core")):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_upgrade_scenarios.py", "-v", "--tb=short"],
        cwd=project_root,
    )
    print("Pytest exit code:", result.returncode)
else:
    print("Skipping pytest (no tests/ or neraium_core/ in project root).")
