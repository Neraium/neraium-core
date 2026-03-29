import json
import pathlib
p = pathlib.Path("upgraded_multinode_test_results.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("AUX_TIME configured:", d.get("aux_time_api_configured"))
print("AUX_TIME URL:", d.get("aux_time_time_url"))
print("AUX_TIME used:", d.get("aux_time_used_for_disturbed_time"))
for n in d.get("nodes", []):
    c = d["runs"][n]["coherent_time"]
    t = d["runs"][n]["disturbed_time"]
    print(f"{n}: coherent={c.get('final_interpreted_state')} disturbed={t.get('final_interpreted_state')} | "
          f"coh_mean={c.get('mean_score')} dist_mean={t.get('mean_score')}")
print("Saved:", p.resolve())
