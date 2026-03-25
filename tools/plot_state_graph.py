from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot state-graph diagnostics over time.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    ent, div, com = [], [], []
    for r in rows:
        g = (r.get("state_graph") or (r.get("experimental_analytics", {}).get("state_graph", {})))
        ent.append(float(g.get("transition_entropy", 0.0)))
        div.append(float(g.get("graph_divergence_score", 0.0)))
        com.append(float(g.get("path_commitment_score", 0.0)))

    plt.figure(figsize=(8, 4))
    plt.plot(ent, label="transition_entropy")
    plt.plot(div, label="graph_divergence_score")
    plt.plot(com, label="path_commitment_score")
    plt.legend()
    plt.title("State transition graph diagnostics")
    plt.xlabel("time index")
    plt.tight_layout()
    plt.savefig(args.output, dpi=140)


if __name__ == "__main__":
    main()
