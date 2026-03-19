"""Build intelligence_layer_monolith.py (single copy-paste bundle)."""
from __future__ import annotations

import re
from pathlib import Path


def strip_future(s: str) -> str:
    """Remove every __future__ annotations import (monolith has one at top)."""
    return re.sub(
        r"^from __future__ import annotations\s*\r?\n",
        "",
        s,
        flags=re.MULTILINE,
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parts: list[str] = []
    header = '''"""
Neraium SII intelligence layer — single file for copy/paste.
Dependencies: numpy only. Regime library defaults to ./regime_library.json

For correlation / drift / spectral / composite math only (no StructuralEngine),
see core_math_engine_monolith.py (rebuild with this script).

Usage:
    engine = StructuralEngine(baseline_window=50, recent_window=12)
    out = engine.process_frame({
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"s1": 0.4, "s2": 0.36, "s3": 0.42},
    })
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


# =============================================================================
'''
    parts.append(header)

    blocks: list[tuple[str, Path]] = [
        ("# --- regime_store ---\n", root / "regime_store.py"),
        ("# --- scoring ---\n", root / "neraium_core" / "scoring.py"),
        ("# --- geometry ---\n", root / "neraium_core" / "geometry.py"),
        ("# --- early_warning ---\n", root / "neraium_core" / "early_warning.py"),
        ("# --- forecasting ---\n", root / "neraium_core" / "forecasting.py"),
        ("# --- forecast_models ---\n", root / "neraium_core" / "forecast_models.py"),
        ("# --- entropy ---\n", root / "neraium_core" / "entropy.py"),
        ("# --- spectral ---\n", root / "neraium_core" / "spectral.py"),
        ("# --- graph ---\n", root / "neraium_core" / "graph.py"),
        ("# --- directional ---\n", root / "neraium_core" / "directional.py"),
        ("# --- causal_proxy (casual.py) ---\n", root / "neraium_core" / "casual.py"),
        ("# --- causal_graph ---\n", root / "neraium_core" / "causal_graph.py"),
        ("# --- causal_attribution ---\n", root / "neraium_core" / "causal_attribution.py"),
        ("# --- subsystems ---\n", root / "neraium_core" / "subsystems.py"),
        ("# --- regime ---\n", root / "neraium_core" / "regime.py"),
        ("# --- data_quality ---\n", root / "neraium_core" / "data_quality.py"),
        ("# --- decision_layer ---\n", root / "neraium_core" / "decision_layer.py"),
        ("# --- staged_pipeline ---\n", root / "neraium_core" / "staged_pipeline.py"),
    ]

    for label, p in blocks:
        parts.append(label)
        parts.append(strip_future(p.read_text(encoding="utf-8")))
        parts.append("\n\n")

    align = (root / "neraium_core" / "alignment.py").read_text(encoding="utf-8")
    lines = align.splitlines(keepends=True)
    marker = "# How slowly the rolling baseline adapts"
    start = next((i for i, ln in enumerate(lines) if ln.startswith(marker)), 0)

    parts.append("# --- StructuralEngine (alignment.py) ---\n")
    parts.append("".join(lines[start:]))

    outp = root / "intelligence_layer_monolith.py"
    outp.write_text("".join(parts), encoding="utf-8")
    n = len(outp.read_text(encoding="utf-8").splitlines())
    print(f"Wrote {outp} ({n} lines)")

    _write_core_math_engine(root)


def _write_core_math_engine(root: Path) -> None:
    """Pure / shared numerical SII primitives (no engine, persistence, or API)."""
    header = '''"""
Neraium SII core math engine — single file for copy/paste.
Dependencies: numpy only.

Contains: window normalization, correlation geometry, structural drift norms,
weighted winsorized composite instability, spectral/graph/directional/entropy
metrics, Granger-style causal proxy, subsystem spectral measures, regime
signatures, early-warning and instability forecasting helpers, observational
causal attribution scores.

Does NOT include: StructuralEngine, data-quality gate, decision_layer, or
RegimeStore JSON persistence. Use intelligence_layer_monolith.py for the full
runtime pipeline.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np


# =============================================================================
'''
    chunks: list[str] = [header]
    core_blocks: list[tuple[str, Path]] = [
        ("# --- scoring (composite instability) ---\n", root / "neraium_core" / "scoring.py"),
        ("# --- geometry (windows, correlation, drift) ---\n", root / "neraium_core" / "geometry.py"),
        ("# --- early_warning ---\n", root / "neraium_core" / "early_warning.py"),
        ("# --- forecasting (instability trend / TTI) ---\n", root / "neraium_core" / "forecasting.py"),
        ("# --- forecast_models (AR1) ---\n", root / "neraium_core" / "forecast_models.py"),
        ("# --- entropy ---\n", root / "neraium_core" / "entropy.py"),
        ("# --- spectral ---\n", root / "neraium_core" / "spectral.py"),
        ("# --- graph ---\n", root / "neraium_core" / "graph.py"),
        ("# --- directional ---\n", root / "neraium_core" / "directional.py"),
        ("# --- causal_proxy (Granger-style, casual.py) ---\n", root / "neraium_core" / "casual.py"),
        ("# --- causal_graph ---\n", root / "neraium_core" / "causal_graph.py"),
        ("# --- subsystems ---\n", root / "neraium_core" / "subsystems.py"),
        ("# --- regime (signatures, library distances) ---\n", root / "neraium_core" / "regime.py"),
        ("# --- causal_attribution ---\n", root / "neraium_core" / "causal_attribution.py"),
        ("# --- staged benchmark/runtime stages (bounded z, stages) ---\n", root / "neraium_core" / "staged_pipeline.py"),
    ]
    for label, p in core_blocks:
        chunks.append(label)
        chunks.append(strip_future(p.read_text(encoding="utf-8")))
        chunks.append("\n\n")

    outp = root / "core_math_engine_monolith.py"
    outp.write_text("".join(chunks), encoding="utf-8")
    n = len(outp.read_text(encoding="utf-8").splitlines())
    print(f"Wrote {outp} ({n} lines)")


if __name__ == "__main__":
    main()
