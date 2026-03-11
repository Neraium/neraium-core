from neraium_core.alignment import AlignmentEngine
from neraium_core.baseline import BASELINE_SYSTEM

engine = AlignmentEngine(BASELINE_SYSTEM)

print("Neraium Alignment Engine initialized.")
print("System:", BASELINE_SYSTEM.system_id)