from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_vectors(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    vectors = []
    for r in rows:
        exp = (r or {}).get("experimental_analytics", {})
        geom = exp.get("state_space", {})
        dim = int(geom.get("state_vector_dim", 0) or 0)
        if dim:
            vectors.append(np.array([dim, r.get("structural_drift_score", 0.0)], dtype=float))
    return np.vstack(vectors) if vectors else np.zeros((0, 2), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot state-space trajectory (deterministic PCA projection).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    points = _load_vectors(Path(args.input))
    if points.shape[0] == 0:
        raise SystemExit("No state-space vectors found in payload.")

    t = np.arange(points.shape[0])
    plt.figure(figsize=(8, 4))
    sc = plt.scatter(points[:, 0], points[:, 1], c=t, cmap="viridis", s=28)
    plt.plot(points[:, 0], points[:, 1], alpha=0.45)
    plt.colorbar(sc, label="time index")
    plt.title("State-space trajectory")
    plt.xlabel("projection-1")
    plt.ylabel("projection-2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=140)


if __name__ == "__main__":
    main()
