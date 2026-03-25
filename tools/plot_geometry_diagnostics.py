from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot trajectory geometry diagnostics.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    curvature, consistency, smoothness = [], [], []
    for r in rows:
        g = (r.get("geometry") or (r.get("experimental_analytics", {}).get("geometry", {})))
        curvature.append(float(g.get("curvature", 0.0)))
        consistency.append(float(g.get("directional_consistency", 0.0)))
        smoothness.append(float(g.get("path_smoothness", 0.0)))

    plt.figure(figsize=(8, 4))
    plt.plot(curvature, label="curvature")
    plt.plot(consistency, label="directional_consistency")
    plt.plot(smoothness, label="path_smoothness")
    plt.legend()
    plt.title("Trajectory geometry diagnostics")
    plt.xlabel("time index")
    plt.tight_layout()
    plt.savefig(args.output, dpi=140)


if __name__ == "__main__":
    main()
