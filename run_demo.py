#!/usr/bin/env python3
"""
Run the Neraium synthetic demo with Gradio UI.

The canonical way to launch the Neraium demo:

    python run_demo.py

This starts a local Gradio app that shows:
- Live tetrahedral state visualization
- System drift and health metrics
- Verdict (admitted/suppressed) gate decisions
- Reasoning and evidence panels
- Playback controls (Play, Pause, Restart, Speed)

The demo plays through 120 timesteps of synthetic data showing a complete
system progression: Baseline → Drift → Transition → Reorganization → Recovery.

Optional flags:
    --help          Show this help message
    --share         Generate a public sharing URL (requires cloudflared or ngrok)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _ensure_repo_on_path() -> None:
    """Add repo to sys.path so imports work."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dependencies() -> None:
    """Install dependencies from pyproject.toml if needed."""
    import subprocess

    need_install = False
    try:
        import gradio  # noqa: F401
    except ImportError:
        need_install = True

    if not need_install:
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Neraium synthetic demo with Gradio UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public sharing URL (requires cloudflared or ngrok)",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    _ensure_repo_on_path()
    ensure_dependencies()

    # Import after ensuring dependencies
    from ui.app import create_gradio_app

    print("=" * 70)
    print("Neraium Demo - Structural Instability Detection")
    print("=" * 70)
    print()
    print("Starting synthetic demo with Gradio UI...")
    print()
    print("The demo shows a complete system progression:")
    print("  • Baseline        (0-20 steps):  Stable operation")
    print("  • Drift Watch     (20-35 steps): Early signs of change")
    print("  • Transition      (35-55 steps): System changing")
    print("  • Reorganization  (55-80 steps): Major structural shift")
    print("  • Recovery        (80-120 steps): System reaching new equilibrium")
    print()
    print("Use the controls to:")
    print("  • Play   - Start automatic playback")
    print("  • Pause  - Pause playback")
    print("  • Restart - Go back to beginning")
    print("  • Speed  - Adjust playback speed")
    print("  • Frame  - Jump to any point in the timeline")
    print()

    app = create_gradio_app()
    app.launch(
        inbrowser=True,
        share=args.share,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
