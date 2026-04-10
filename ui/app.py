from __future__ import annotations

from typing import Any

from .config import UIConfig, load_ui_config
from .core_integration import build_system_state
from .layouts import build_demo_view, build_operations_view, build_pilot_view
from .realtime import RealtimeFeed, create_realtime_feed


def _normalize_records(records: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if records is None:
        return None
    if isinstance(records, dict):
        return [records]
    return records


def create_ui_model(records: list[dict[str, Any]] | None = None, *, config: UIConfig | None = None) -> dict[str, object]:
    cfg = config or load_ui_config()
    state = build_system_state(records, config=cfg)
    realtime_runtime = create_realtime_feed(cfg.ws_endpoint)
    realtime = realtime_runtime.status
    return {
        "title": cfg.title,
        "system_state": state.to_dict(),
        "pilot": build_pilot_view(state),
        "operations": build_operations_view(state),
        "demo": build_demo_view(state),
        "realtime": {
            "endpoint": realtime.endpoint,
            "enabled": realtime.enabled,
            "reason": realtime.reason,
        },
    }


def create_app_state(records: list[dict[str, Any]] | dict[str, Any] | None = None) -> dict[str, object]:
    """Integration-guide compatible surface summary plus full model pointers."""
    normalized_records = _normalize_records(records)
    model = create_ui_model(normalized_records)
    realtime_status = model["realtime"]
    realtime = RealtimeFeed(
        endpoint=str(realtime_status["endpoint"]),
        enabled=bool(realtime_status["enabled"]),
        reason=str(realtime_status["reason"]) if realtime_status["reason"] is not None else None,
    )
    regime = model["system_state"]["regime_state"]
    drift = model["system_state"]["drift_intensity"]
    return {
        "summary": f"Navigation surface ready ({regime}, drift={drift:.3f})",
        "realtime": realtime,
        "model": model,
    }


def create_gradio_app(records: list[dict[str, Any]] | None = None):
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(f"Gradio is required to launch UI: {exc}") from exc

    model = create_ui_model(records)
    realtime = model["realtime"]
    realtime_line = f"Realtime {'enabled' if realtime['enabled'] else 'disabled'} @ {realtime['endpoint']}"
    if realtime["reason"]:
        realtime_line = f"{realtime_line} ({realtime['reason']})"

    with gr.Blocks(title=str(model["title"]), css_paths=["ui/themes/neraium_dark.css"]) as app:
        gr.Markdown(f"## {model['title']}")
        gr.Markdown(f"<div class='neraium-surface'>{realtime_line}</div>")
        gr.JSON(model["pilot"], label="Navigation Surface")
    return app


if __name__ == "__main__":
    create_gradio_app().launch(server_name="0.0.0.0", server_port=7860)
