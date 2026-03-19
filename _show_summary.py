import json, pathlib
p = pathlib.Path("upgraded_multinode_test_results.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("GAL2 configured:", d.get("gal2_api_configured"))
print("GAL2 URL:", d.get("gal2_time_url"))
print("GAL2 used:", d.get("gal2_used_for_disturbed_time"))
for n in d.get("nodes", []):
    c = d["runs"][n]["coherent_time"]
    t = d["runs"][n]["disturbed_time"]
    print(f"{n}: coherent={c.get('final_interpreted_state')} disturbed={t.get('final_interpreted_state')} | "
          f"coh_mean={c.get('mean_score')} dist_mean={t.get('mean_score')}")
print("Saved:", p.resolve())
