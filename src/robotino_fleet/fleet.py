"""Compact free-space fleet runtime shared by all three launchers."""

import time
from dataclasses import replace
from typing import Literal

from robotino_fleet.adapters.robotino.executors import (
    RobotinoCommandSink,
    ShadowCommandSink,
)
from robotino_fleet.config import (
    CONTROL_PERIOD_S,
    GOAL_HEADING_TOLERANCE_RAD,
    GOAL_TOLERANCE_M,
    ProjectSettings,
    ROBOT_DIAMETER_M,
    ROBOT_RADIUS_M,
)
from robotino_fleet.domain.models import (
    FleetEvent,
    LidarScanObservation,
    OdometryObservation,
    Point,
    PoseObservation,
    RobotState,
    RobotStatus,
    VelocityCommand,
)
from robotino_fleet.maps.models import MapBundle
from robotino_fleet.navigation.controller import NavigationController
from robotino_fleet.navigation.goals import DestinationCatalog, NavigationGoal
from robotino_fleet.navigation.motion import MotionLimiter
from robotino_fleet.navigation.safety import (
    NavigationSafetySupervisor,
    POSE_STALE_S,
)
from robotino_fleet.navigation.virtual_lidar import apply_virtual_lidar_constraints
from robotino_fleet.simulation.physical_scene import PhysicalScene
from robotino_fleet.simulation.world import COLORS, SimulationWorld


RuntimeMode = Literal["simulation", "shadow", "live"]


class FleetRuntime:
    """Hold fleet state and run one controller in simulation, shadow, or live mode."""

    def __init__(
        self,
        settings: ProjectSettings,
        map_bundle: MapBundle,
        controller: NavigationController,
        *,
        mode: RuntimeMode,
        world: SimulationWorld | None = None,
        command_sink: ShadowCommandSink | RobotinoCommandSink | None = None,
        localization_available: bool = True,
    ) -> None:
        """Bind settings, map_bundle, and controller for mode.

        world is supplied only for simulation; physical modes receive a
        command_sink and later telemetry through on_* callbacks.
        localization_available controls whether physical poses may be
        marked valid. The constructor returns a ready fleet state.

        Args:
            settings: Configuration settings for this component.
            map_bundle: Factory map metadata, goal labels, and obstacle geometry.
            controller: Navigation controller that proposes velocity commands.
            mode: Execution mode: simulation, shadow, or live.
            world: Simulation world, when running without physical Robotinos.
            command_sink: Physical command sender, present only in live mode.
            localization_available: Whether a validated map transform permits physical poses.
        """

        self.settings = settings
        self.map = map_bundle
        self.controller = controller
        self.mode = mode
        self.world = world
        self.command_sink = command_sink
        self.localization_available = localization_available
        self.navigation_scene = (
            world.scene
            if world is not None
            else PhysicalScene(
                map_bundle.environment_geometry,
                robot_radius_m=ROBOT_RADIUS_M,
            )
        )
        self.safety = NavigationSafetySupervisor()
        self.limiter = MotionLimiter(settings.motion)
        self.destinations = DestinationCatalog.from_map(
            map_bundle,
            position_tolerance_m=GOAL_TOLERANCE_M,
            heading_tolerance_rad=GOAL_HEADING_TOLERANCE_RAD,
        )
        self.clock_s = 0.0
        self.sequence = 0
        self.paused = False
        self.polling_service = None
        self.events: list[FleetEvent] = []
        self.goals: dict[str, NavigationGoal] = {}
        self._immediate_stops: set[str] = set()
        self.errors: dict[str, dict[str, str]] = {}
        if world is not None:
            self.robots = world.robots
        else:
            try:
                minimum_x, minimum_y, _, _ = self.navigation_scene.movement_bounds
            except ValueError:
                minimum_x = minimum_y = -1000.0
            self.robots = {
                definition.ip: RobotState(
                    ip=definition.ip,
                    color=COLORS[index % len(COLORS)],
                    # Live and shadow Robotinos do not exist at their configured
                    # simulation start. Stage them outside the movement area
                    # until localization supplies their first valid pose.
                    x=minimum_x - (index + 2) * ROBOT_DIAMETER_M,
                    y=minimum_y - 2.0 * ROBOT_DIAMETER_M,
                    heading_rad=0.0,
                    status=RobotStatus.OFFLINE,
                    online=False,
                    pose_valid=False,
                    pose_source="unavailable",
                )
                for index, definition in enumerate(settings.robotinos)
            }
        self._started = time.monotonic()
        self._event(None, "SYSTEM", f"{mode} runtime ready")

    @property
    def physical_commands_enabled(self) -> bool:
        """Return whether this runtime has a live physical command sink.

        Returns:
            bool: Whether this runtime has a live physical command sink.
        """

        return bool(
            self.command_sink
            and self.command_sink.physical_output_enabled
        )

    def _event(self, robot_id: str | None, kind: str, message: str) -> None:
        """Append a timestamped fleet event and retain only recent history.

        robot_id may be None for system events; kind and
        message become the dashboard event type and text. Nothing returns.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            kind: Event category displayed in the dashboard.
            message: Human-readable event or error text.
        """

        self.events.append(
            FleetEvent(len(self.events) + 1, self.clock_s, robot_id, kind, message)
        )
        del self.events[:-100]

    def assign_goal(self, robot_id: str, location_id: str) -> NavigationGoal:
        """Assign location_id to robot_id and reset policy state.

        Return the resolved map goal. Unknown Robotinos or labels raise
        KeyError; a new navigation-metrics session starts immediately.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            location_id: Factory-map label ID to use as the navigation goal.

        Returns:
            NavigationGoal: Resolved goal assigned to the selected Robotino.
        """

        robot = self.robots[robot_id]
        goal = self.destinations.get(location_id)
        if self.world is not None:
            tracking_source = "simulation"
            tracking_point = robot.point
            tracking_timestamp_s = self.clock_s
        else:
            tracking_source = "odometry"
            tracking_point = (
                Point(robot.odometry.x, robot.odometry.y)
                if robot.odometry is not None
                else None
            )
            tracking_timestamp_s = (
                robot.odometry.timestamp_s if robot.odometry is not None else None
            )
        robot.navigation_metrics.start(
            robot.point if robot.pose_valid else None,
            goal.point,
            tracking_source=tracking_source,
            tracking_point=tracking_point,
            timestamp_s=tracking_timestamp_s,
            assignment_started_at_s=self.clock_s,
        )
        self.goals[robot_id] = goal
        robot.goal_location_id = location_id
        robot.status = RobotStatus.NAVIGATING
        robot.stop_reason = None
        self.controller.reset(robot_id)
        self.limiter.reset(robot_id)
        if self.world:
            self.world.set_goal(robot_id, goal.point)
        self._event(robot_id, "GOAL", f"Driving to label {location_id}")
        return goal

    def stop_robot(self, robot_id: str, reason: str = "operator stop") -> None:
        """Cancel robot_id's order and request an immediate stop.

        reason is retained in robot state and the event feed. Policy,
        motion limiter, and navigation metrics are reset; nothing is returned.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            reason: Human-readable reason recorded with the stop event.
        """

        robot = self.robots[robot_id]
        self.goals.pop(robot_id, None)
        robot.goal_location_id = None
        robot.status = RobotStatus.STOPPED
        robot.stop_reason = reason
        robot.proposed_command = VelocityCommand()
        robot.applied_command = VelocityCommand()
        robot.navigation_metrics.cancel()
        self.controller.reset(robot_id)
        self.limiter.stop(robot_id)
        if self.world:
            self.world.set_goal(robot_id, None)
            self.world.stop(robot_id)
        elif self.command_sink:
            try:
                robot.command_queued = self.command_sink.apply(
                    robot_id, VelocityCommand()
                )
            except Exception as error:
                robot.stop_reason = f"{reason}; stop transmission failed: {error}"
        self._event(robot_id, "STOP", reason)

    def _commands(self, delta_s: float) -> dict[str, VelocityCommand]:
        """Infer active orders for this delta_s control period.

        Gate proposed commands through the safety supervisor, mark immediate
        stops or arrivals, and return a body-command mapping for every robot.
        Unassigned physical robots receive a zero placeholder but are not sent
        periodic commands by tick.

        Args:
            delta_s: Elapsed control or simulation time in seconds.

        Returns:
            dict[str, VelocityCommand]: Commands proposed for Robotinos with active orders.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        commands: dict[str, VelocityCommand] = {}
        self._immediate_stops.clear()
        physical = self.mode in {"shadow", "live"}
        active_robots = {
            robot_id: robot
            for robot_id, robot in self.robots.items()
            if robot_id in self.goals
        }
        active_goals = {
            robot_id: self.goals[robot_id] for robot_id in active_robots
        }
        outputs = {}
        batch_error: str | None = None
        if active_robots:
            try:
                outputs = self.controller.compute_commands(
                    active_robots,
                    active_goals,
                    delta_s=delta_s,
                    now_s=self.clock_s,
                )
            except Exception as error:
                batch_error = str(error)
        for robot_id, robot in self.robots.items():
            goal = self.goals.get(robot_id)
            if goal is None:
                commands[robot_id] = VelocityCommand()
                continue
            try:
                if batch_error is not None:
                    raise RuntimeError(batch_error)
                output = outputs[robot_id]
                robot.proposed_command = output.command
                robot.policy_id = output.policy_id
                robot.policy_inference_ms = output.inference_ms
                policy_error = output.reason if output.state.value == "FAULT" else None
                if output.arrived:
                    self.goals.pop(robot_id, None)
                    robot.navigation_metrics.complete(timestamp_s=self.clock_s)
                    ready_to_dock = output.state.value == "READY_TO_DOCK"
                    robot.status = (
                        RobotStatus.READY_TO_DOCK
                        if ready_to_dock
                        else RobotStatus.ARRIVED
                    )
                    robot.applied_command = VelocityCommand()
                    commands[robot_id] = VelocityCommand()
                    self._immediate_stops.add(robot_id)
                    self._event(
                        robot_id,
                        "READY_TO_DOCK" if ready_to_dock else "ARRIVED",
                        f"Reached handover for label {goal.location_id}"
                        if ready_to_dock
                        else f"Reached label {goal.location_id}",
                    )
                    continue
            except Exception as error:
                robot.proposed_command = VelocityCommand()
                policy_error = str(error)
            decision = self.safety.evaluate(
                robot,
                goal,
                robot.proposed_command,
                now_s=self.clock_s,
                physical=physical,
                policy_error=policy_error,
            )
            if not decision.allowed:
                self._immediate_stops.add(robot_id)
                robot.status = RobotStatus.WAITING
                robot.stop_reason = decision.reason
            else:
                robot.status = RobotStatus.NAVIGATING
                robot.stop_reason = None
            commands[robot_id] = decision.command
        return commands

    def tick(self, delta_s: float = CONTROL_PERIOD_S) -> None:
        """Advance one delta_s control period in the selected mode.

        Simulation advances world; physical modes pass limited commands to
        the sink. Robotinos without an order receive no physical command.

        Args:
            delta_s: Elapsed control or simulation time in seconds.
        """

        if self.paused:
            return
        self.clock_s = self.world.time_s if self.world else time.monotonic() - self._started
        active_before_step = set(self.goals)
        proposed = self._commands(delta_s)
        if self.world:
            for robot_id in self._immediate_stops:
                self.world.stop(robot_id)
            collisions = self.world.step(proposed, delta_s)
            self.clock_s = self.world.time_s
            maximum_speed_mps = max(
                self.settings.motion.forward_max_mps,
                self.settings.motion.sideways_max_mps,
            )
            for robot_id in active_before_step:
                robot = self.robots[robot_id]
                robot.navigation_metrics.observe(
                    robot.point,
                    tracking_source="simulation",
                    timestamp_s=self.clock_s,
                    maximum_speed_mps=maximum_speed_mps,
                )
            for robot_id, collided in collisions.items():
                if collided:
                    self.goals.pop(robot_id, None)
                    robot = self.robots[robot_id]
                    robot.navigation_metrics.cancel()
                    robot.status = RobotStatus.FAULT
                    robot.stop_reason = "simulation collision"
                    self._event(robot_id, "COLLISION", "Simulation collision")
        else:
            for robot_id, desired in proposed.items():
                robot = self.robots[robot_id]
                if (
                    robot_id not in self.goals
                    and robot_id not in self._immediate_stops
                ):
                    robot.applied_command = VelocityCommand()
                    continue
                command = (
                    self.limiter.stop(robot_id)
                    if robot_id in self._immediate_stops
                    else self.limiter.apply(robot_id, desired, delta_s)
                )
                robot.applied_command = command
                if self.command_sink:
                    try:
                        robot.command_queued = self.command_sink.apply(
                            robot_id, command
                        )
                    except Exception as error:
                        robot.command_queued = False
                        robot.stop_reason = f"command failed: {error}"
                        robot.status = RobotStatus.FAULT
        self.sequence += 1

    def reset(self) -> None:
        """Reset simulated robots, goals, and metrics; reject physical modes.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        if not self.world:
            raise RuntimeError("Only the simulation can be reset")
        self.world.reset()
        self.robots = self.world.robots
        self.goals.clear()
        self.events.clear()
        for robot in self.robots.values():
            robot.navigation_metrics.reset()
        self.clock_s = 0.0
        self.sequence += 1
        self._event(None, "RESET", "Simulation reset")

    def set_paused(self, paused: bool) -> None:
        """Set simulation pause state from paused; return nothing.

        Args:
            paused: Whether simulation ticks should be paused.
        """

        self.paused = paused

    def on_pose(self, observation: PoseObservation) -> None:
        """Apply factory-map-frame observation from physical pose polling.

        Clear its last pose error, update map pose/validity/online state, and
        mark a newly localized Robotino idle. No value is returned.

        Args:
            observation: Current policy observation or measured robot state.
        """

        robot = self.robots[observation.robot_id]
        self._clear_error(observation.robot_id, "pose")
        robot.x = observation.x
        robot.y = observation.y
        robot.heading_rad = observation.heading
        robot.pose_received_at_s = max(0.0, observation.timestamp_s - self._started)
        robot.pose_valid = observation.valid and self.localization_available
        robot.pose_source = observation.source
        robot.online = True
        if robot.status is RobotStatus.OFFLINE and robot.pose_valid:
            robot.status = RobotStatus.IDLE

    def on_odometry(self, robot_id: str, value: OdometryObservation) -> None:
        """Store robot_id's local odometry value and measured velocity.

        For an active order, update driven distance and travel metrics. The
        odometry frame remains distinct from factory-map coordinates.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            value: Local-frame odometry and measured velocity sample.
        """

        self._clear_error(robot_id, "odometry")
        robot = self.robots[robot_id]
        odometry = replace(
            value, timestamp_s=max(0.0, value.timestamp_s - self._started)
        )
        robot.odometry = odometry
        robot.measured_velocity = VelocityCommand(value.vx, value.vy, value.omega)
        robot.online = True
        if robot_id in self.goals:
            robot.navigation_metrics.observe(
                Point(odometry.x, odometry.y),
                tracking_source="odometry",
                timestamp_s=odometry.timestamp_s,
                maximum_speed_mps=max(
                    self.settings.motion.forward_max_mps,
                    self.settings.motion.sideways_max_mps,
                ),
            )

    def on_lidar(self, robot_id: str, value: LidarScanObservation) -> None:
        """Store robot_id's physical scan value in separate views.

        measured_lidar remains untouched for emergency stops; lidar
        receives virtual map/peer constraints for policy input and display.
        Nothing is returned.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            value: Physical LiDAR scan received from the Robotino.
        """

        self._clear_error(robot_id, "lidar")
        robot = self.robots[robot_id]
        localized_scan = replace(
            value, timestamp_s=max(0.0, value.timestamp_s - self._started)
        )
        # Keep the physical measurement separate from the map/peer constraints
        # added for policy inference and dashboard visualization. Safety stops
        # must never be triggered by a synthetic return from an inaccurate pose.
        robot.measured_lidar = localized_scan
        robot.lidar = apply_virtual_lidar_constraints(
            robot,
            localized_scan,
            self.navigation_scene,
            robots=self._fresh_localized_peers(
                robot_id,
                max(
                    localized_scan.timestamp_s,
                    time.monotonic() - self._started,
                ),
            ),
        )

    def _fresh_localized_peers(
        self,
        robot_id: str,
        now_s: float,
    ) -> tuple[RobotState, ...]:
        """Return localized peers safe to synthesize for robot_id.

        now_s bounds pose age; stale or invalid peers are excluded so a
        ghost marker cannot become a virtual policy obstacle.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            tuple[RobotState, ...]: Localized peers safe to synthesize for robot_id.
        """

        return tuple(
            peer
            for peer_id, peer in self.robots.items()
            if peer_id != robot_id
            and peer.pose_valid
            and 0.0 <= now_s - peer.pose_received_at_s <= POSE_STALE_S
        )

    def on_distance(
        self, robot_id: str, values: tuple[float, ...], timestamp_s: float
    ) -> None:
        """Store robot_id's nine values received at timestamp_s.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            values: Measured or computed values supplied to this operation.
            timestamp_s: Measurement or event time in seconds.
        """

        self._clear_error(robot_id, "distance")
        robot = self.robots[robot_id]
        robot.sensors = list(values)
        robot.sensors_updated_at_s = max(0.0, timestamp_s - self._started)

    def on_bumper(self, robot_id: str, value: bool) -> None:
        """Store measured bumper value for robot_id and clear its error.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            value: Measured bumper contact state.
        """

        self._clear_error(robot_id, "bumper")
        self.robots[robot_id].bumper_pressed = value

    def on_battery(self, robot_id: str, value: bool) -> None:
        """Store battery-low value for robot_id and clear its error.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            value: Measured battery-low state.
        """

        self._clear_error(robot_id, "battery")
        self.robots[robot_id].battery_low = value

    def _clear_error(self, robot_id: str, endpoint: str) -> None:
        """Remove endpoint's last polling error for robot_id if present.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            endpoint: Robotino HTTP telemetry endpoint or endpoint name.
        """

        errors = self.errors.get(robot_id)
        if errors is None:
            return
        errors.pop(endpoint, None)
        if not errors:
            self.errors.pop(robot_id, None)

    def on_error(self, robot_id: str, endpoint: str, message: str) -> None:
        """Expose message as robot_id's latest endpoint error.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            endpoint: Robotino HTTP telemetry endpoint or endpoint name.
            message: Human-readable event or error text.
        """

        self.errors.setdefault(robot_id, {})[endpoint] = message

    def world_dict(self) -> dict[str, object]:
        """Return the mode-independent snapshot consumed by the dashboard.

        Returns:
            dict[str, object]: The mode-independent snapshot consumed by the dashboard.
        """

        return {
            "schemaVersion": 3,
            "sequence": self.sequence,
            "clockS": self.clock_s,
            "mode": self.mode,
            "paused": self.paused,
            "physicalCommandsEnabled": self.physical_commands_enabled,
            "robots": [
                robot.to_dict(now_s=self.clock_s)
                for robot in self.robots.values()
            ],
            "goals": {
                robot_id: goal.to_dict() for robot_id, goal in self.goals.items()
            },
            "temporaryObstacles": [
                {
                    "id": obstacle.id,
                    "x": obstacle.x,
                    "y": obstacle.y,
                    "radiusM": ROBOT_RADIUS_M,
                }
                for obstacle in (self.world.temporary_obstacles if self.world else [])
            ],
            "errors": self.errors,
            "events": [event.to_dict() for event in reversed(self.events)],
        }

    def shutdown(self) -> None:
        """Stop active orders and close the physical command sink, if any."""

        for robot_id in list(self.goals):
            self.stop_robot(robot_id, "backend shutdown")
        close = getattr(self.command_sink, "close", None)
        if close is not None:
            close()
