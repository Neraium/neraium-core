from __future__ import annotations

from typing import Any

from .config import UIConfig, load_ui_config
from .core_integration import build_system_state
from .layouts import build_demo_view, build_operations_view, build_pilot_view
from .realtime import create_realtime_feed


def create_ui_model(records: list[dict[str, Any]] | None = None, *, config: UIConfig | None = None) -> dict[str, object]:
    cfg = config or load_ui_config()
    state = build_system_state(records, config=cfg)
    return {
        "title": cfg.title,
        "system_state": state.to_dict(),
        "pilot": build_pilot_view(state),
        "operations": build_operations_view(state),
        "demo": build_demo_view(state),
        "realtime": create_realtime_feed(cfg.ws_endpoint).status,
    }


def create_gradio_app(records: list[dict[str, Any]] | None = None):
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(f"Gradio is required to launch UI: {exc}") from exc

    model = create_ui_model(records)
    with gr.Blocks(title=str(model["title"]), css_paths=["ui/themes/neraium_dark.css"]) as app:
        gr.Markdown(f"## {model['title']}")
        gr.JSON(model["pilot"], label="System Navigation Surface")
        gr.JSON(model["realtime"], label="Realtime")
    return app


if __name__ == "__main__":
    create_gradio_app().launch(server_name="0.0.0.0", server_port=7860)
