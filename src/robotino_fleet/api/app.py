"""Small HTTP/WebSocket boundary consumed by the fleet dashboard."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from robotino_fleet.config import CONTROL_PERIOD_S, DISTANCE_SENSOR_CLEAR_M, ROBOT_DIAMETER_M
from robotino_fleet.fleet import FleetRuntime


class GoalRequest(BaseModel):
    locationId: str


async def _fleet_loop(fleet: FleetRuntime, stop: asyncio.Event) -> None:
    """Advance fleet at the configured control period until stop is set.

    This background task is created by the app lifespan and returns only when
    the server shuts down or cancels it.

    Args:
        fleet: Advance fleet at the configured control period until stop is set.
        stop: Event signalling that the background loop should finish.
    """

    while not stop.is_set():
        await asyncio.sleep(CONTROL_PERIOD_S)
        fleet.tick(CONTROL_PERIOD_S)


def create_app(fleet: FleetRuntime, *, start_background: bool = True) -> FastAPI:
    """Expose fleet to the dashboard and optionally run its control loop.

    start_background=False lets tests drive the runtime explicitly. The
    returned app starts physical polling during its lifespan when configured.

    Args:
        fleet: Expose fleet to the dashboard and optionally run its control loop.
            start_background=False lets tests drive the runtime explicitly.
        start_background: Whether to launch polling and control with the API lifespan.

    Returns:
        FastAPI: Configured dashboard API application.
    """

    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Start polling/control on app startup and stop them on shutdown.

        The FastAPI argument is unused; the yielded lifespan keeps services
        running. On exit, polling is stopped and the fleet command sink closes.

        Args:
            _: Unused framework-provided argument.
        """

        polling = fleet.polling_service
        if polling:
            await polling.start()
        task = (
            asyncio.create_task(_fleet_loop(fleet, stop))
            if start_background
            else None
        )
        try:
            yield
        finally:
            stop.set()
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if polling:
                await polling.stop()
            fleet.shutdown()

    app = FastAPI(title="Robotino Fleet Manager", version="0.2.0", lifespan=lifespan)
    app.state.fleet = fleet
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Return backend mode, health, sequence, and physical-output status.

        Returns:
            dict[str, object]: Backend mode, health, sequence, and physical-output status.
        """

        return {
            "status": "ok",
            "mode": fleet.mode,
            "physicalCommandsEnabled": fleet.physical_commands_enabled,
            "sequence": fleet.sequence,
        }

    @app.get("/api/map")
    def map_data() -> dict[str, object]:
        """Return map geometry, labels, and the effective Robotino diameter.

        Returns:
            dict[str, object]: Map geometry, labels, and the effective Robotino diameter.
        """

        result = fleet.map.to_dict()
        result["robotDiameterM"] = ROBOT_DIAMETER_M
        result["distanceSensorMaxM"] = DISTANCE_SENSOR_CLEAR_M
        return result

    @app.get("/assets/iot_factory_map.png", response_class=FileResponse)
    def map_image() -> Path:
        """Serve the generated factory image or return HTTP 404 if absent.

        Returns:
            Path: Factory map image served to the dashboard.

        Raises:
            HTTPException: If the requested dashboard action is unavailable or invalid.
        """

        path = fleet.settings.project_root / "data" / "generated" / "iot_factory_map.png"
        if not path.exists():
            raise HTTPException(404, "Factory map image is missing")
        return path

    @app.get("/api/world")
    def world() -> dict[str, object]:
        """Return the latest world snapshot for initial dashboard loading.

        Returns:
            dict[str, object]: The latest world snapshot for initial dashboard loading.
        """

        return fleet.world_dict()

    @app.post("/api/robotinos/{robotino_ip}/goal")
    def goal(robotino_ip: str, request: GoalRequest) -> dict[str, object]:
        """Assign request.locationId to robotino_ip and return state.

        Args:
            robotino_ip: Robotino IP identifying the dashboard request target.
            request: Validated dashboard request body.

        Returns:
            dict[str, object]: World snapshot after assigning the new goal.

        Raises:
            HTTPException: If the requested dashboard action is unavailable or invalid.
        """

        try:
            fleet.assign_goal(robotino_ip, request.locationId)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        return fleet.world_dict()

    @app.post("/api/robotinos/{robotino_ip}/stop")
    def stop_robot(robotino_ip: str) -> dict[str, object]:
        """Cancel robotino_ip's order and return updated world state.

        Args:
            robotino_ip: Robotino IP identifying the dashboard request target.

        Returns:
            dict[str, object]: World snapshot after cancelling the order.

        Raises:
            HTTPException: If the requested dashboard action is unavailable or invalid.
        """

        try:
            fleet.stop_robot(robotino_ip)
        except KeyError as error:
            raise HTTPException(404, f"Unknown Robotino {robotino_ip}") from error
        return fleet.world_dict()

    @app.post("/api/simulation/reset")
    def reset_simulation() -> dict[str, object]:
        """Return reset simulation state; physical modes respond with 409.

        Returns:
            dict[str, object]: Reset simulation state; physical modes respond with 409.

        Raises:
            HTTPException: If the requested dashboard action is unavailable or invalid.
        """

        try:
            fleet.reset()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return fleet.world_dict()

    @app.post("/api/simulation/pause")
    def pause_simulation() -> dict[str, object]:
        """Pause simulated control ticks and return the resulting state.

        Returns:
            dict[str, object]: World snapshot with simulation paused.
        """

        fleet.set_paused(True)
        return fleet.world_dict()

    @app.post("/api/simulation/resume")
    def resume_simulation() -> dict[str, object]:
        """Resume simulated control ticks and return the resulting state.

        Returns:
            dict[str, object]: World snapshot with simulation running.
        """

        fleet.set_paused(False)
        return fleet.world_dict()

    @app.websocket("/api/world/stream")
    async def stream(websocket: WebSocket) -> None:
        """Accept websocket and push world snapshots until disconnect.

        Args:
            websocket: Connected dashboard WebSocket to receive world snapshots.
        """

        await websocket.accept()
        try:
            while True:
                await websocket.send_json(fleet.world_dict())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass

    return app
