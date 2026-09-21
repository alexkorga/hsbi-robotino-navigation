"""Build one free-space runtime for the purpose selected by its launcher."""

from pathlib import Path

from robotino_fleet.adapters.robotino.executors import (
    RobotinoCommandSink,
    ShadowCommandSink,
)
from robotino_fleet.adapters.robotino.http_api import RobotinoHttpClient
from robotino_fleet.config import ProjectSettings, load_settings
from robotino_fleet.execution.polling import RobotinoPollingService
from robotino_fleet.fleet import FleetRuntime, RuntimeMode
from robotino_fleet.maps import PlanarTransform, load_map_bundle
from robotino_fleet.navigation.controller import DirectGoalController
from robotino_fleet.navigation.sb3_forward_turn_controller import (
    Sb3ForwardTurnNavigationController,
)
from robotino_fleet.navigation.sb3_controller import Sb3NavigationController
from robotino_fleet.navigation.local_target_policy_controller import (
    LocalTargetPolicyController,
)
from robotino_fleet.learning.profile import load_policy_profile, profile_path
from robotino_fleet.simulation.world import SimulationWorld


def build_fleet_runtime(
    mode: RuntimeMode,
    *,
    model_path: Path | None = None,
    settings: ProjectSettings | None = None,
) -> FleetRuntime:
    """Return a fleet runtime for mode and optional model_path.

    settings overrides the project defaults. A model profile selects its
    controller; with no model the direct-goal controller is used. Simulation
    receives a local world; shadow/live receive HTTP polling and separate
    nontransmitting/transmitting command sinks.

    Args:
        mode: Execution mode: simulation, shadow, or live.
        model_path: Path to the trained policy artifact.
        settings: Configuration settings for this component.

    Returns:
        FleetRuntime: A fleet runtime for mode and optional model_path.
    """

    settings = settings or load_settings()
    map_bundle = load_map_bundle(settings.map_bundle)
    if model_path is None:
        controller = DirectGoalController(settings.motion)
    elif profile_path(model_path).exists():
        profile = load_policy_profile(model_path)
        if profile.action_mode == "forward-turn":
            controller = Sb3ForwardTurnNavigationController(
                model_path, motion=settings.motion
            )
        else:
            controller = LocalTargetPolicyController(
                model_path, motion=settings.motion
            )
    else:
        controller = Sb3NavigationController(model_path, motion=settings.motion)
    if mode == "simulation":
        world = SimulationWorld(
            map_bundle,
            settings.motion,
            robot_starts={
                robotino.ip: robotino.simulation_start
                for robotino in settings.robotinos
            },
        )
        return FleetRuntime(
            settings,
            map_bundle,
            controller,
            mode=mode,
            world=world,
        )

    clients = {
        robotino.ip: RobotinoHttpClient(
            robotino.ip,
            preferred_port=robotino.api_port,
            fallback_port=robotino.fallback_port,
        )
        for robotino in settings.robotinos
    }
    sink = ShadowCommandSink() if mode == "shadow" else RobotinoCommandSink(clients)
    runtime = FleetRuntime(
        settings,
        map_bundle,
        controller,
        mode=mode,
        command_sink=sink,
        localization_available=settings.localization is not None,
    )
    localization = settings.localization
    transform = (
        PlanarTransform(
            localization.x_m, localization.y_m, localization.theta_rad
        )
        if localization
        else PlanarTransform()
    )
    runtime.polling_service = RobotinoPollingService(
        clients,
        transform=transform,
        lidar_extrinsics={
            robotino.ip: robotino.lidar for robotino in settings.robotinos
        },
        on_pose=runtime.on_pose,
        on_odometry=runtime.on_odometry,
        on_lidar=runtime.on_lidar,
        on_distance=runtime.on_distance,
        on_bumper=runtime.on_bumper,
        on_battery=runtime.on_battery,
        on_error=runtime.on_error,
    )
    return runtime
