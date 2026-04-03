"""Explicit ASGI entrypoint for Neraium Markets deployments."""

from __future__ import annotations

from neraium_core.markets.app.api import create_app

# Explicit deployment app for App Runner and other ASGI hosts.
app = create_app()
