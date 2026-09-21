"""Shared server function called by the three root launchers."""

from pathlib import Path

import uvicorn

from robotino_fleet.api.app import create_app
from robotino_fleet.bootstrap import build_fleet_runtime
from robotino_fleet.fleet import RuntimeMode


def serve(mode: RuntimeMode, *, model_path: Path | None = None) -> None:
    """Serve mode on localhost port 8000 using optional model_path.

    Loads the fleet runtime, creates the dashboard API, and blocks in Uvicorn
    until shutdown. No value is returned.

    Args:
        mode: Execution mode: simulation, shadow, or live.
        model_path: Path to the trained policy artifact.
    """

    fleet = build_fleet_runtime(mode, model_path=model_path)
    uvicorn.run(create_app(fleet), host="127.0.0.1", port=8000)
