"""
AImSafe — Factory Floor Safety Co-Pilot
Entry point: launches the NiceGUI dashboard.

Usage:
    python main.py

    Set OPENROUTER_API_KEY env var to enable AI coaching:
    export OPENROUTER_API_KEY=sk-or-...   (Linux/Mac)
    set OPENROUTER_API_KEY=sk-or-...      (Windows)

Dashboard will be at http://localhost:8080
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from nicegui import ui, app
import config
from ui.dashboard import build_dashboard, build_worker_page


@ui.page("/")
def index():
    build_dashboard()


@ui.page("/worker/{worker_id}")
def worker_detail(worker_id: int):
    """Full-page worker detail view — navigate here from dashboard."""
    build_worker_page(worker_id)


if __name__ == "__main__":
    print("=" * 60)
    print("  AImSafe — Factory Floor Safety Co-Pilot")
    print("=" * 60)
    print(f"  Dashboard:  http://localhost:{config.DASHBOARD_PORT}")
    print(f"  LLM:        {'Enabled (OpenRouter)' if config.OPENROUTER_API_KEY else 'Disabled (set OPENROUTER_API_KEY)'}")
    print(f"  MQTT:       {'Enabled' if config.MQTT_ENABLED else 'Disabled'}")
    print("=" * 60)

    ui.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        title="AImSafe",
        dark=config.DARK_THEME,
        reload=False,
        storage_secret="aimsafe-secret",
    )
