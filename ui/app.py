def create_app_state(records):
    """
    Minimal integration-safe builder used by INTEGRATION_GUIDE.py.
    Does not depend on Gradio.
    """
    if not records:
        return {
            "summary": {},
            "realtime": {"enabled": False},
        }

    latest = records[-1]

    return {
        "summary": {
            "timestamp": latest.get("timestamp"),
            "system_health": latest.get("system_health"),
            "confidence": latest.get("confidence_score"),
            "drift": latest.get("structural_drift_score"),
            "stability": latest.get("relational_stability_score"),
            "regime": latest.get("regime_name"),
        },
        "realtime": {
            "enabled": False,
        },
    }

def create_gradio_app():
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio is not installed")

    def dummy():
        return "Neraium UI running"

    with gr.Blocks() as app:
        gr.Markdown("# Neraium UI")
        out = gr.Textbox(label="Status")
        gr.Button("Test").click(fn=dummy, outputs=out)

    return app

def create_ui_model(data):
    return {
        "summary": data[-1] if data else {},
        "realtime": {"enabled": False}
    }

def create_ui_model(data):
    return {
        "summary": data[-1] if data else {},
        "realtime": {"enabled": False}
    }
