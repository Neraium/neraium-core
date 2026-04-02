"""Minimal dashboard stub for local inspection."""

from __future__ import annotations

import plotly.graph_objects as go


def build_signal_count_chart(signal_counts: dict[str, int]) -> go.Figure:
    fig = go.Figure([go.Bar(x=list(signal_counts.keys()), y=list(signal_counts.values()))])
    fig.update_layout(title="Neraium Markets Signals by Asset", xaxis_title="Asset", yaxis_title="Count")
    return fig
