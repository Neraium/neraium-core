"""Generate plain-English explanations from score components."""

from __future__ import annotations


def build_explanation(asset: str, regime: str, components: dict[str, float], breadth: float, vix_change: float) -> str:
    top = sorted(components.items(), key=lambda x: x[1], reverse=True)[:2]
    top_txt = ", ".join(f"{k}={v:.2f}" for k, v in top)
    return (
        f"{asset} flagged as {regime} because structural shifts ({top_txt}) were elevated, "
        f"breadth proxy was {breadth:.2f}, and VIX change was {vix_change:.3f}."
    )
