"""Day 9: scenario path labeling from recent market_regime_label sequences."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import score_action_usefulness

_SHORT = {
    "market_stable": "stable",
    "market_fragile": "fragile",
    "market_risk_off": "risk_off",
    "market_transition": "transition",
    "market_fragmented": "fragmented",
    "market_mixed": "mixed",
}


def _short_regime(s: object) -> str:
    k = str(s)
    return _SHORT.get(k, k.replace("market_", "")[:12])


def _window_labels(series: pd.Series, start: int, i: int) -> tuple[str, ...]:
    """``start`` inclusive, ``i`` inclusive window of short regime labels."""
    return tuple(_short_regime(series.iloc[j]) for j in range(start, i + 1))


def label_scenario_paths(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map recent **2–4** step sequences of ``market_regime_label`` to named scenario paths.

    **Rules**

    - Timeline is sorted by ``timestamp``.
    - At row ``i``, consider windows ``[i-3,i]``, ``[i-2,i]``, ``[i-1,i]`` (shorter windows
      used when ``i`` is small). The **longest** matching pattern wins.
    - ``scenario_path_length`` is the matched window length (2–4), or 1 for ``unclassified``.
    - ``scenario_transition_flag`` is 1 when ``scenario_path_label`` differs from the prior row.

    **Named paths** (illustrative; all matching is on short tokens):

    - ``stable_to_fragile`` — ``stable -> fragile``
    - ``fragile_to_risk_off`` — ``fragile -> risk_off`` or ``fragile -> transition -> risk_off``
    - ``fragmented_recovery`` — ``fragmented -> mixed -> stable`` or ``fragmented -> stable``
    - ``false_calm_break`` — ``stable -> transition`` or ``stable -> fragmented`` after stable
    - ``transition_to_stable`` — ``transition -> stable`` or ``mixed -> stable``
    - ``unstable_chop`` — ``transition -> fragmented -> transition`` or ``mixed -> fragmented -> mixed``
    - ``risk_off_rebound_attempt`` — ``risk_off -> fragile`` or ``risk_off -> transition``
    """
    if "market_regime_label" not in market_df.columns:
        raise KeyError("market_regime_label")

    out = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    lab = out["market_regime_label"].astype(str)
    n = len(out)

    # Longest-first patterns: tuple of short labels -> scenario name
    patterns: list[tuple[tuple[str, ...], str]] = [
        (("fragile", "transition", "risk_off"), "fragile_to_risk_off"),
        (("fragmented", "mixed", "stable"), "fragmented_recovery"),
        (("transition", "fragmented", "transition"), "unstable_chop"),
        (("mixed", "fragmented", "mixed"), "unstable_chop"),
        (("stable", "fragile", "risk_off"), "fragile_to_risk_off"),
        (("risk_off", "fragile"), "risk_off_rebound_attempt"),
        (("risk_off", "transition"), "risk_off_rebound_attempt"),
        (("fragile", "risk_off"), "fragile_to_risk_off"),
        (("fragmented", "stable"), "fragmented_recovery"),
        (("transition", "stable"), "transition_to_stable"),
        (("mixed", "stable"), "transition_to_stable"),
        (("stable", "transition"), "false_calm_break"),
        (("stable", "fragmented"), "false_calm_break"),
        (("stable", "fragile"), "stable_to_fragile"),
    ]

    labels: list[str] = []
    lengths: list[int] = []

    for i in range(n):
        best: str | None = None
        best_len = 1
        for w in (4, 3, 2):
            if i + 1 < w:
                continue
            start = i - w + 1
            tup = _window_labels(lab, start, i)
            for pat, name in patterns:
                if len(pat) != w:
                    continue
                if tup == pat:
                    best = name
                    best_len = w
                    break
            if best is not None:
                break
        if best is None:
            labels.append("unclassified")
            lengths.append(1)
        else:
            labels.append(best)
            lengths.append(best_len)

    out["scenario_path_label"] = labels
    out["scenario_path_length"] = lengths
    sl = pd.Series(labels, dtype=str)
    stf = (sl != sl.shift(1)).astype(int)
    stf.iloc[0] = 0
    out["scenario_transition_flag"] = stf

    return out.loc[:, ~out.columns.duplicated()]


def _map_market_posture_to_action(p: str) -> str:
    return {
        "risk_on": "lean_long",
        "cautious_risk_on": "watch",
        "neutral": "watch",
        "reduce_risk": "reduce_exposure",
        "defensive": "avoid_risk",
        "wait": "wait",
    }.get(str(p), "watch")


def summarize_scenario_paths(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ``scenario_path_label``: count, average path length / run proxy, mean usefulness.

    Requires ``scenario_path_label``, ``fwd_ret_*d``, and ``market_action_posture`` for usefulness.
    """
    if "scenario_path_label" not in market_df.columns:
        raise KeyError("scenario_path_label")

    df = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    df = df.assign(action_posture=df["market_action_posture"].map(_map_market_posture_to_action))
    df = score_action_usefulness(df)

    u1, u5, u10 = "action_useful_1d", "action_useful_5d", "action_useful_10d"
    dur_col = "market_run_length" if "market_run_length" in df.columns else "scenario_path_length"

    rows: list[dict[str, object]] = []
    for scen, g in df.groupby("scenario_path_label", sort=True):
        rows.append(
            {
                "scenario_path_label": scen,
                "count": int(len(g)),
                "avg_duration": float(g[dur_col].astype(float).mean()) if dur_col in g.columns else float("nan"),
                "avg_usefulness_1d": float(g[u1].mean()) if u1 in g.columns else float("nan"),
                "avg_usefulness_5d": float(g[u5].mean()) if u5 in g.columns else float("nan"),
                "avg_usefulness_10d": float(g[u10].mean()) if u10 in g.columns else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario_path_label").reset_index(drop=True)
