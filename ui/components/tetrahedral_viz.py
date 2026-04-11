from __future__ import annotations

from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - defensive fallback for demo environments
    plt = None
    Figure = Any  # type: ignore[assignment]

_VERTEX_POINTS = {
    "STRUCTURAL": (1.0, 1.0, 1.0),
    "RELATIONAL": (1.0, -1.0, -1.0),
    "TRANSITION": (-1.0, 1.0, -1.0),
    "TEMPORAL": (-1.0, -1.0, 1.0),
}


def _extract_position(tetrahedral_state: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(tetrahedral_state, dict):
        return None
    position = tetrahedral_state.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        try:
            return float(position[0]), float(position[1]), float(position[2])
        except (TypeError, ValueError):
            return None
    if isinstance(position, dict):
        try:
            return float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))
        except (TypeError, ValueError):
            return None
    return None


def build_tetrahedral_plot_and_text(
    latest_record: dict[str, Any] | None,
    history_records: list[dict[str, Any]] | None = None,
    *,
    history_limit: int = 24,
) -> tuple[Any, str]:
    if plt is None:
        return None, (
            "**Interpreted label:** Unavailable\n\n"
            "**Movement summary:** Tetrahedral visualization unavailable (matplotlib not installed).\n\n"
            "**Nearest vertex:** —"
        )

    fig = plt.figure(figsize=(5.8, 4.4), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # Tetrahedron frame
    vertex_names = list(_VERTEX_POINTS.keys())
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    vertices = [_VERTEX_POINTS[name] for name in vertex_names]
    for i, j in edge_pairs:
        x1, y1, z1 = vertices[i]
        x2, y2, z2 = vertices[j]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="#64748B", linewidth=1.2, alpha=0.7)

    for name, (x, y, z) in _VERTEX_POINTS.items():
        ax.scatter([x], [y], [z], color="#93C5FD", s=28, alpha=0.95)
        ax.text(x * 1.08, y * 1.08, z * 1.08, name, color="#E2E8F0", fontsize=8)

    trail_points: list[tuple[float, float, float]] = []
    for row in (history_records or [])[-history_limit:]:
        tetra = row.get("tetrahedral_state") if isinstance(row, dict) else None
        point = _extract_position(tetra)
        if point is not None:
            trail_points.append(point)

    tetra_state = latest_record.get("tetrahedral_state") if isinstance(latest_record, dict) else None
    current_point = _extract_position(tetra_state)

    if trail_points:
        xs = [point[0] for point in trail_points]
        ys = [point[1] for point in trail_points]
        zs = [point[2] for point in trail_points]
        progression = [idx / max(len(trail_points) - 1, 1) for idx in range(len(trail_points))]
        trail_colors = plt.cm.viridis(progression)

        for idx in range(1, len(trail_points)):
            ax.plot(
                [xs[idx - 1], xs[idx]],
                [ys[idx - 1], ys[idx]],
                [zs[idx - 1], zs[idx]],
                color=trail_colors[idx],
                linewidth=1.8,
                alpha=0.8,
            )
        ax.scatter(xs, ys, zs, c=progression, cmap="viridis", s=12, alpha=0.6)

    if current_point is not None:
        ax.scatter([current_point[0]], [current_point[1]], [current_point[2]], color="#F97316", s=72, alpha=1.0)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("X", color="#CBD5E1", fontsize=8)
    ax.set_ylabel("Y", color="#CBD5E1", fontsize=8)
    ax.set_zlabel("Z", color="#CBD5E1", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=7)
    ax.grid(color="#334155", alpha=0.35)
    ax.set_facecolor("#020617")
    fig.patch.set_facecolor("#020617")
    ax.view_init(elev=18, azim=35)
    ax.set_title("Tetrahedral State Trajectory", color="#E2E8F0", fontsize=10, pad=10)
    fig.tight_layout()

    interpreted_label = "Unavailable"
    movement_summary = "No tetrahedral state available in this frame."
    nearest_vertex = "—"
    if isinstance(tetra_state, dict):
        interpreted_label = str(tetra_state.get("interpreted_label") or interpreted_label)
        movement_summary = str(tetra_state.get("movement_summary") or movement_summary)
        nearest_vertex = str(tetra_state.get("nearest_vertex") or nearest_vertex)

    details_md = (
        f"**Interpreted label:** {interpreted_label}\n\n"
        f"**Movement summary:** {movement_summary}\n\n"
        f"**Nearest vertex:** {nearest_vertex}"
    )

    return fig, details_md
