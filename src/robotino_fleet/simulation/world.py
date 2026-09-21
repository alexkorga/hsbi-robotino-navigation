"""Canonical 2-D Robotino world used by the dashboard and RL environments."""

import math
import random
from collections import deque
from dataclasses import replace
from typing import Mapping

from robotino_fleet.config import MotionSettings, ROBOT_RADIUS_M
from robotino_fleet.domain.models import (
    Point,
    RobotState,
    RobotStatus,
    VelocityCommand,
)
from robotino_fleet.maps.models import EnvironmentGeometry, MapBundle
from robotino_fleet.maps.transforms import normalize_angle
from robotino_fleet.navigation.motion import MotionLimiter
from robotino_fleet.simulation.physical_scene import PhysicalScene
from robotino_fleet.simulation.sensors import (
    DistanceSensorSimulationSettings,
    LidarSimulationSettings,
    SensorSimulationSettings,
    SensorSimulator,
)


COLORS = ("#31d7c5", "#ffb454", "#8ea8ff", "#f06da8", "#a8e063", "#d29cff")


class SimulationWorld:
    """Continuous kinematics, collisions, motion calibration, and sensors."""

    def __init__(
        self,
        map_bundle: MapBundle,
        motion: MotionSettings,
        *,
        robot_starts: Mapping[str, str],
        seed: int = 42,
        use_factory_obstacles: bool = True,
        random_obstacles: int = 0,
        sensor_noise: bool = False,
        dynamics_randomization: bool = False,
    ) -> None:
        """Build a world from map_bundle and calibrated motion.

        robot_starts maps Robotino IPs to label IDs and seed controls
        deterministic episodes. use_factory_obstacles selects fixed
        geometry, random_obstacles adds temporary footprints, and
        sensor_noise/dynamics_randomization add training variation.

        Args:
            map_bundle: Factory map metadata, goal labels, and obstacle geometry.
            motion: Calibrated Robotino speed and acceleration limits.
            robot_starts: Initial map labels assigned to each simulated Robotino.
            seed: Optional seed for reproducible random sampling.
            use_factory_obstacles: Whether to include configured factory fixtures in collision
                checks.
            random_obstacles: Whether episodes should add variable obstacle fixtures.
            sensor_noise: Whether simulated sensor noise is enabled.
            dynamics_randomization: Whether physical response varies across episodes.
        """

        self.map = map_bundle
        self.motion = motion
        self.robot_starts = dict(robot_starts)
        self.seed = seed
        self.use_factory_obstacles = use_factory_obstacles
        self.random_obstacle_count = random_obstacles
        self.sensor_noise = sensor_noise
        self.dynamics_randomization = dynamics_randomization
        geometry = map_bundle.environment_geometry
        if geometry is not None and not use_factory_obstacles:
            geometry = EnvironmentGeometry(
                source=f"{geometry.source}:open",
                movement_area=geometry.movement_area,
                obstacles=(),
            )
        self.scene = PhysicalScene(geometry, robot_radius_m=ROBOT_RADIUS_M)
        self.limiter = MotionLimiter(motion)
        self.robots: dict[str, RobotState] = {}
        self.goals: dict[str, Point] = {}
        self.temporary_obstacles: list[RobotState] = []
        self.inactive_robot_ids: set[str] = set()
        self.last_collision_kinds: dict[str, tuple[str, ...]] = {}
        self.time_s = 0.0
        self._random = random.Random(seed)
        self._motion_scale = 1.0
        self._queues: dict[str, deque[tuple[float, VelocityCommand]]] = {}
        self._delayed: dict[str, VelocityCommand] = {}
        self.sensor_simulator = SensorSimulator(self._sensor_settings(seed), self.scene)
        self.reset(seed=seed)

    def _sensor_settings(self, seed: int) -> SensorSimulationSettings:
        """Return sensor timing/noise settings for the episode seed.

        Args:
            seed: Optional seed for reproducible random sampling.

        Returns:
            SensorSimulationSettings: Sensor timing/noise settings for the episode seed.
        """

        lidar = LidarSimulationSettings()
        distance = DistanceSensorSimulationSettings()
        if self.sensor_noise:
            lidar = replace(lidar, range_noise_std_m=0.015, dropout_probability=0.01)
            distance = replace(
                distance, range_noise_std_m=0.008, dropout_probability=0.01
            )
        return SensorSimulationSettings(seed, lidar, distance)

    def reset(self, *, seed: int | None = None) -> dict[str, RobotState]:
        """Reset robots, obstacles, sensors, and clock with optional seed.

        Return the newly placed Robotino states keyed by IP.

        Args:
            seed: Optional seed for reproducible random sampling.

        Returns:
            dict[str, RobotState]: Robotino states at the start of the simulated episode.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if seed is not None:
            self.seed = int(seed)
        self._random.seed(self.seed)
        self.time_s = 0.0
        self.goals.clear()
        self.robots.clear()
        self.temporary_obstacles.clear()
        self.inactive_robot_ids.clear()
        self.last_collision_kinds = {
            robot_id: () for robot_id in self.robot_starts
        }
        self.limiter.reset()
        self._queues.clear()
        self._delayed.clear()
        self._motion_scale = (
            self._random.uniform(0.85, 1.10)
            if self.dynamics_randomization
            else 1.0
        )
        for index, (robot_id, location_id) in enumerate(self.robot_starts.items()):
            try:
                location = self.map.locations[location_id]
            except KeyError as error:
                raise ValueError(
                    f"Unknown simulation_start {location_id!r} for {robot_id}"
                ) from error
            robot = RobotState(
                ip=robot_id,
                color=COLORS[index % len(COLORS)],
                x=location.point.x,
                y=location.point.y,
                heading_rad=location.heading_rad,
                pose_received_at_s=0.0,
                pose_source="simulation",
            )
            if not self.scene.is_valid_center(robot.point, robots=self.robots.values()):
                raise ValueError(
                    f"simulation_start {location_id} is not collision-free for {robot_id}"
                )
            self.robots[robot_id] = robot
            self._queues[robot_id] = deque()
            self._delayed[robot_id] = VelocityCommand()
        self._place_random_obstacles()
        self.sensor_simulator = SensorSimulator(
            self._sensor_settings(self.seed), self.scene
        )
        self.sensor_simulator.advance(
            0.0, 0.0, self.robots, obstacles=self.temporary_obstacles
        )
        return self.robots

    def _place_random_obstacles(self) -> None:
        """Place configured temporary obstacles without overlapping robots.

        Called by reset after Robotinos have been placed. Raises
        RuntimeError if a requested obstacle cannot fit in the movement
        area after repeated samples; nothing is returned.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        if self.random_obstacle_count <= 0:
            return
        min_x, min_y, max_x, max_y = self.scene.movement_bounds
        occupied = list(self.robots.values())
        for index in range(self.random_obstacle_count):
            for _ in range(500):
                point = Point(
                    self._random.uniform(min_x, max_x),
                    self._random.uniform(min_y, max_y),
                )
                if self.scene.is_valid_center(point, robots=occupied):
                    obstacle = RobotState(
                        ip=f"obstacle-{index + 1}",
                        color="#ff6673",
                        x=point.x,
                        y=point.y,
                        heading_rad=0.0,
                        pose_source="simulation-obstacle",
                    )
                    self.temporary_obstacles.append(obstacle)
                    occupied.append(obstacle)
                    break
            else:
                raise RuntimeError("Could not place random obstacle")

    def set_goal(self, robot_id: str, goal: Point | None) -> None:
        """Set or clear robot_id's simulated goal for display/state.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            goal: Assigned navigation target.

        Raises:
            KeyError: If a requested Robotino, goal, or key does not exist.
        """

        if robot_id not in self.robots:
            raise KeyError(robot_id)
        if goal is None:
            self.goals.pop(robot_id, None)
        else:
            self.goals[robot_id] = goal

    def stop(self, robot_id: str) -> None:
        """Clear robot_id's queued and delayed motion immediately.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.

        Raises:
            KeyError: If a requested Robotino, goal, or key does not exist.
        """

        if robot_id not in self.robots:
            raise KeyError(robot_id)
        self.limiter.stop(robot_id)
        self._queues[robot_id].clear()
        self._delayed[robot_id] = VelocityCommand()
        self.robots[robot_id].applied_command = VelocityCommand()

    def deactivate(self, robot_id: str) -> None:
        """Remove failed robot_id from physics and sensing until reset.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.

        Raises:
            KeyError: If a requested Robotino, goal, or key does not exist.
        """

        if robot_id not in self.robots:
            raise KeyError(robot_id)
        self.stop(robot_id)
        self.inactive_robot_ids.add(robot_id)
        robot = self.robots[robot_id]
        robot.pose_valid = False
        robot.status = RobotStatus.FAULT
        robot.bumper_pressed = True
        robot.measured_velocity = VelocityCommand()
        robot.proposed_command = VelocityCommand()
        robot.applied_command = VelocityCommand()
        robot.sensors = [DistanceSensorSimulationSettings().clear_value_m] * 9
        robot.measured_lidar = None
        robot.lidar = None

    def _queue_commands(
        self, desired: Mapping[str, VelocityCommand], delta_s: float
    ) -> None:
        """Limit and queue desired commands for this delta_s step.

        Per-robot command delay is applied before motion integration. Inactive
        robots are skipped; the resulting delayed commands update world state.

        Args:
            desired: Limit and queue desired commands for this delta_s step.
            delta_s: Elapsed control or simulation time in seconds.
        """

        for robot_id in self.robots:
            if robot_id in self.inactive_robot_ids:
                continue
            limited = self.limiter.apply(
                robot_id, desired.get(robot_id, VelocityCommand()), delta_s
            )
            self._queues[robot_id].append(
                (self.time_s + self.motion.command_delay_s, limited)
            )
            while (
                self._queues[robot_id]
                and self._queues[robot_id][0][0] <= self.time_s + 1e-9
            ):
                _, self._delayed[robot_id] = self._queues[robot_id].popleft()

    def step(
        self, desired: Mapping[str, VelocityCommand], delta_s: float
    ) -> dict[str, bool]:
        """Advance calibrated kinematics and sensors by delta_s seconds.

        desired supplies body-frame commands by IP. The world substeps
        collision checks and returns a collision flag for every Robotino.

        Args:
            desired: Advance calibrated kinematics and sensors by delta_s seconds. desired
                supplies body-frame commands by IP.
            delta_s: Elapsed control or simulation time in seconds.

        Returns:
            dict[str, bool]: Per-Robotino collision flags after the simulation step.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if delta_s <= 0 or not math.isfinite(delta_s):
            raise ValueError("delta_s must be positive and finite")
        self._queue_commands(desired, delta_s)
        collisions = {robot_id: False for robot_id in self.robots}
        collision_kinds = {robot_id: set() for robot_id in self.robots}
        maximum = max(
            (
                math.hypot(command.vx, command.vy) * delta_s
                for robot_id, command in self._delayed.items()
                if robot_id not in self.inactive_robot_ids
            ),
            default=0.0,
        )
        substeps = max(1, math.ceil(maximum / (ROBOT_RADIUS_M * 0.25)))
        sub_delta = delta_s / substeps
        for _ in range(substeps):
            candidates: dict[str, tuple[Point, float]] = {}
            for robot_id, robot in self.robots.items():
                if robot_id in self.inactive_robot_ids:
                    continue
                command = self._delayed[robot_id]
                cosine = math.cos(robot.heading_rad)
                sine = math.sin(robot.heading_rad)
                point = Point(
                    robot.x
                    + (cosine * command.vx - sine * command.vy)
                    * sub_delta
                    * self._motion_scale,
                    robot.y
                    + (sine * command.vx + cosine * command.vy)
                    * sub_delta
                    * self._motion_scale,
                )
                heading = normalize_angle(
                    robot.heading_rad
                    + command.omega * sub_delta * self._motion_scale
                )
                candidates[robot_id] = (point, heading)
                if self.scene.collision_at(
                    point,
                    robots=self.temporary_obstacles,
                    exclude_robot_id=robot_id,
                ):
                    collisions[robot_id] = True
                    collision_kinds[robot_id].add("environment")
            ids = [
                robot_id
                for robot_id in self.robots
                if robot_id not in self.inactive_robot_ids
            ]
            for index, first in enumerate(ids):
                for second in ids[index + 1 :]:
                    if candidates[first][0].distance_to(candidates[second][0]) < 2 * ROBOT_RADIUS_M:
                        collisions[first] = True
                        collisions[second] = True
                        collision_kinds[first].add("robotino")
                        collision_kinds[second].add("robotino")
            for robot_id, robot in self.robots.items():
                if robot_id in self.inactive_robot_ids:
                    continue
                if collisions[robot_id]:
                    continue
                point, heading = candidates[robot_id]
                robot.x, robot.y, robot.heading_rad = point.x, point.y, heading
            if any(collisions.values()):
                break
        self.time_s += delta_s
        for robot_id, robot in self.robots.items():
            if robot_id in self.inactive_robot_ids:
                robot.measured_velocity = VelocityCommand()
                robot.applied_command = VelocityCommand()
                robot.pose_received_at_s = self.time_s
                continue
            applied = self._delayed[robot_id] if not collisions[robot_id] else VelocityCommand()
            if collisions[robot_id]:
                self.limiter.stop(robot_id)
                self._delayed[robot_id] = VelocityCommand()
                robot.bumper_pressed = True
            noise = self.motion.odometry_noise_mps
            robot.measured_velocity = VelocityCommand(
                applied.vx + (self._random.gauss(0.0, noise) if noise else 0.0),
                applied.vy + (self._random.gauss(0.0, noise) if noise else 0.0),
                applied.omega,
            )
            robot.applied_command = applied
            robot.pose_received_at_s = self.time_s
            robot.trajectory.append(robot.point)
            del robot.trajectory[:-300]
        self.sensor_simulator.advance(
            self.time_s,
            self.time_s,
            self.robots,
            obstacles=self.temporary_obstacles,
        )
        for robot_id in self.inactive_robot_ids:
            robot = self.robots[robot_id]
            robot.sensors = [DistanceSensorSimulationSettings().clear_value_m] * 9
            robot.measured_lidar = None
            robot.lidar = None
        self.last_collision_kinds = {
            robot_id: tuple(sorted(kinds))
            for robot_id, kinds in collision_kinds.items()
        }
        return collisions
