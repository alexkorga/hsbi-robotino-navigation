"""RL episode semantics on top of the canonical SimulationWorld."""

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from robotino_fleet.config import (
    DISTANCE_SENSOR_CLEAR_M,
    GOAL_HEADING_TOLERANCE_RAD,
    GOAL_TOLERANCE_M,
    ProjectSettings,
    ROBOT_RADIUS_M,
)
from robotino_fleet.domain.models import Point, RobotState, VelocityCommand
from robotino_fleet.learning.actions import ActionScaler
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.observation import (
    BinnedLocalTargetObservationBuilder,
    ModelObservationBuilder,
    PoseCorrectedLidar360Memory,
    TemporalObservationHistory,
)
from robotino_fleet.learning.rewards import navigation_reward
from robotino_fleet.maps.models import EnvironmentGeometry, MapBundle
from robotino_fleet.maps.transforms import normalize_angle
from robotino_fleet.navigation.goals import DestinationCatalog, NavigationGoal
from robotino_fleet.navigation.forward_turn import ForwardTurnMotionController
from robotino_fleet.navigation.local_target import LocalTargetMotionController
from robotino_fleet.simulation.sensors import (
    DistanceSensorSimulationSettings,
    LidarSimulationSettings,
)
from robotino_fleet.simulation.world import SimulationWorld


@dataclass(frozen=True)
class CoreStep:
    observations: dict[str, dict[str, np.ndarray]]
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    infos: dict[str, dict[str, Any]]


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return whether routes a–b and c–d cross or touch.

    Args:
        a: Start of the first line segment.
        b: End of the first line segment.
        c: Start of the second line segment.
        d: End of the second line segment.

    Returns:
        bool: Whether routes a–b and c–d cross or touch.
    """

    def orientation(first: Point, second: Point, third: Point) -> float:
        """Return the signed turn of the first–second–third points.

        Args:
            first: First point in the ordered orientation test.
            second: Vertex point in the ordered orientation test.
            third: Last point in the ordered orientation test.

        Returns:
            float: The signed turn of the first–second–third points.
        """

        return (second.x - first.x) * (third.y - first.y) - (
            second.y - first.y
        ) * (third.x - first.x)

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return first * second <= 0.0 and third * fourth <= 0.0


class FactoryLearningCore:
    """Goals and rewards around exactly the world used by dashboard simulation."""

    def __init__(
        self,
        fleet_settings: ProjectSettings,
        map_bundle: MapBundle,
        settings: LearningEnvironmentSettings,
    ) -> None:
        """Bind the canonical world, scenario settings, and observation contract.

        fleet_settings supplies Robotino identities/motion, map_bundle
        the factory geometry, and settings robot count, sampling, rewards,
        and direct-velocity versus profile-defined local-target actions.

        Args:
            fleet_settings: Robotino fleet and physical-motion settings.
            map_bundle: Factory map metadata, goal labels, and obstacle geometry.
            settings: Configuration settings for this component.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if not 1 <= settings.robot_count <= len(fleet_settings.robotinos):
            raise ValueError("robot_count must fit the configured Robotino fleet")
        if (
            settings.maximum_start_goal_distance_m is not None
            and settings.maximum_start_goal_distance_m
            < settings.minimum_start_goal_distance_m
        ):
            raise ValueError(
                "maximum_start_goal_distance_m cannot be smaller than the minimum"
            )
        if not 0.0 <= settings.multi_robot_conflict_probability <= 1.0:
            raise ValueError("multi_robot_conflict_probability must be between zero and one")
        if settings.scenario_mode not in {
            "waypoints",
            "orientation-runway",
            "island-corners",
            "island-corner-mix",
        }:
            raise ValueError(f"Unsupported scenario mode: {settings.scenario_mode}")
        if settings.orientation_runway_min_distance_m <= 0:
            raise ValueError("orientation runway minimum distance must be positive")
        if settings.orientation_runway_lane_spacing_m <= 2.0 * ROBOT_RADIUS_M:
            raise ValueError("orientation runway lanes must clear two Robotino radii")
        if (
            settings.scenario_mode == "orientation-runway"
            and settings.maximum_start_goal_distance_m is None
        ):
            raise ValueError("orientation runway requires a maximum distance")
        if settings.multi_robot_scenario and settings.robot_count < 2:
            raise ValueError("A multi-robot scenario requires at least two Robotinos")
        self.fleet_settings = fleet_settings
        self.map = map_bundle
        self.settings = settings
        definitions = fleet_settings.robotinos[: settings.robot_count]
        self.agent_ids = tuple(definition.ip for definition in definitions)
        world_map = (
            self._orientation_runway_map(map_bundle)
            if settings.scenario_mode == "orientation-runway"
            else map_bundle
        )
        self.world = SimulationWorld(
            world_map,
            fleet_settings.motion,
            robot_starts={
                definition.ip: definition.simulation_start for definition in definitions
            },
            seed=settings.random_seed,
            use_factory_obstacles=settings.use_factory_obstacles,
            random_obstacles=settings.random_obstacles,
            sensor_noise=settings.sensor_noise,
            dynamics_randomization=settings.dynamics_randomization,
        )
        self.scene = self.world.scene
        self.destination_catalog = DestinationCatalog.from_map(
            map_bundle,
            position_tolerance_m=GOAL_TOLERANCE_M,
            heading_tolerance_rad=GOAL_HEADING_TOLERANCE_RAD,
        )
        self.action_scaler = ActionScaler(
            fleet_settings.motion.forward_max_mps,
            fleet_settings.motion.sideways_max_mps,
            fleet_settings.motion.rotation_max_rps,
        )
        lidar = LidarSimulationSettings()
        distance = DistanceSensorSimulationSettings()
        self.policy_profile = settings.policy_profile
        self.local_target_controller = None
        self.forward_turn_controller = None
        if self.policy_profile is not None:
            if self.policy_profile.action_mode == "forward-turn":
                self.forward_turn_controller = ForwardTurnMotionController(
                    fleet_settings.motion, self.policy_profile
                )
            else:
                self.local_target_controller = LocalTargetMotionController(
                    fleet_settings.motion, self.policy_profile
                )
        if self.policy_profile is not None:
            if not math.isclose(
                settings.time_step_s,
                self.policy_profile.policy_period_s,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "time_step_s must match the local-target policy period"
                )
            expected_contract = (
                (settings.max_goal_distance_m, self.policy_profile.observation_goal_distance_m),
                (distance.clear_value_m, self.policy_profile.proximity_clear_distance_m),
            )
            # Physical motion limits and model velocity-normalization scales are
            # independent. A validated policy may intentionally run at a higher
            # actuator limit than the scale used while it was trained.
            if any(
                not math.isclose(actual, expected, abs_tol=1e-9)
                for actual, expected in expected_contract
            ) or distance.count != self.policy_profile.proximity_sensor_count:
                raise ValueError(
                    "Environment normalization does not match the local-target profile"
                )
            self.action_dimensions = self.policy_profile.action_dimensions
            self.observation_builder = BinnedLocalTargetObservationBuilder(
                profile=self.policy_profile,
            )
        else:
            self.action_dimensions = 3
            self.observation_builder = ModelObservationBuilder(
                lidar_beam_count=lidar.beam_count,
                distance_sensor_count=distance.count,
                lidar_range_max_m=lidar.range_max_m,
                distance_clear_value_m=distance.clear_value_m,
                max_goal_distance_m=settings.max_goal_distance_m,
                max_forward_mps=fleet_settings.motion.forward_max_mps,
                max_sideways_mps=fleet_settings.motion.sideways_max_mps,
                max_rotation_rps=fleet_settings.motion.rotation_max_rps,
            )
        self.observation_history = TemporalObservationHistory(
            self.policy_profile.temporal_frames
            if self.policy_profile is not None
            else 1
        )
        self.lidar_memory_360 = (
            PoseCorrectedLidar360Memory(self.policy_profile)
            if self.policy_profile is not None
            and self.policy_profile.uses_pose_corrected_lidar_360
            else None
        )
        self.robots = self.world.robots
        self.goals: dict[str, NavigationGoal] = {}
        self.start_location_ids: dict[str, str] = {}
        self.velocities: dict[str, VelocityCommand] = {}
        self.previous_actions: dict[str, np.ndarray] = {}
        self.previous_distances: dict[str, float] = {}
        self.best_distances: dict[str, float] = {}
        self.best_heading_errors: dict[str, float] = {}
        self.path_lengths: dict[str, float] = {}
        self.minimum_clearances: dict[str, float] = {}
        self.action_variations: dict[str, float] = {}
        self.command_variations: dict[str, float] = {}
        self.angular_effort: dict[str, float] = {}
        self.previous_commands: dict[str, VelocityCommand] = {}
        self.completed_agents: set[str] = set()
        self.failed_agents: set[str] = set()
        self.failure_collision_types: dict[str, set[str]] = {}
        self.step_count = 0
        self.time_s = 0.0
        self.seed = settings.random_seed
        self.episode_seed = settings.random_seed
        self._random = np.random.default_rng(self.seed)

    @staticmethod
    def _orientation_runway_map(map_bundle: MapBundle) -> MapBundle:
        """Return map_bundle with a large empty orientation-runway arena.

        Args:
            map_bundle: Factory map metadata, goal labels, and obstacle geometry.

        Returns:
            MapBundle: Map_bundle with a large empty orientation-runway arena.
        """

        half_extent = 40.0
        movement_area = (
            Point(-half_extent, -half_extent),
            Point(half_extent, -half_extent),
            Point(half_extent, half_extent),
            Point(-half_extent, half_extent),
        )
        return replace(
            map_bundle,
            environment_geometry=EnvironmentGeometry(
                source="orientation-runway",
                movement_area=movement_area,
                obstacles=(),
            ),
        )

    @property
    def temporary_obstacles(self) -> list[RobotState]:
        """Return the world's currently sampled temporary obstacles.

        Returns:
            list[RobotState]: The world's currently sampled temporary obstacles.
        """

        return self.world.temporary_obstacles

    @property
    def observation_space(self):
        """Return the Gymnasium observation space matching the active profile.

        Returns:
            object: The Gymnasium observation space matching the active profile.
        """

        return self.observation_builder.space()

    @property
    def action_space(self):
        """Return a normalized action box with the active policy dimensions.

        Returns:
            object: A normalized action box with the active policy dimensions.
        """

        from gymnasium import spaces

        return spaces.Box(
            -1.0, 1.0, shape=(self.action_dimensions,), dtype=np.float32
        )

    def _valid_goals(self) -> list[NavigationGoal]:
        """Return map destinations whose centers fit the Robotino footprint.

        Returns:
            list[NavigationGoal]: Map destinations whose centers fit the Robotino footprint.
        """

        return [
            goal
            for goal in self.destination_catalog.values()
            if self.scene.is_valid_center(goal.point, radius_m=ROBOT_RADIUS_M)
        ]

    def _configured_pair(
        self, agent_id: str, options: Mapping[str, Any] | None
    ) -> tuple[NavigationGoal, NavigationGoal] | None:
        """Resolve an explicit start/goal pair for agent_id from options.

        Return None when the reset options do not configure that agent.
        Location IDs are looked up in the destination catalog.

        Args:
            agent_id: Identifier of the simulated agent whose pair is resolved.
            options: Optional reset overrides, including fixed starts and goals.

        Returns:
            tuple[NavigationGoal, NavigationGoal] | None: Requested start/goal pair, or None if
                no pair was configured.
        """

        if not options or not isinstance(options.get("pairs"), Mapping):
            return None
        pair = options["pairs"].get(agent_id)
        if not isinstance(pair, Mapping):
            return None
        return (
            self.destination_catalog.get(str(pair["start_location"])),
            self.destination_catalog.get(str(pair["goal_location"])),
        )

    def _sample_pair(
        self, candidates: Sequence[NavigationGoal], occupied: list[RobotState]
    ) -> tuple[NavigationGoal, NavigationGoal]:
        """Sample a reachable start/goal pair from candidates.

        occupied supplies already placed Robotinos. Return two distinct
        destinations satisfying distance limits and footprint clearance, or
        raise RuntimeError if no such pair can be found.

        Args:
            candidates: Reachable factory positions eligible for start and goal sampling.
            occupied: Starting positions already assigned to other Robotinos.

        Returns:
            tuple[NavigationGoal, NavigationGoal]: Reachable, distinct start and goal positions.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        for start_index in self._random.permutation(len(candidates)):
            start = candidates[int(start_index)]
            if not self.scene.is_valid_center(
                start.point,
                robots=(*occupied, *self.world.temporary_obstacles),
            ):
                continue
            for goal_index in self._random.permutation(len(candidates)):
                goal = candidates[int(goal_index)]
                distance = start.point.distance_to(goal.point)
                if (
                    start.location_id != goal.location_id
                    and distance >= self.settings.minimum_start_goal_distance_m
                    and (
                        self.settings.maximum_start_goal_distance_m is None
                        or distance <= self.settings.maximum_start_goal_distance_m
                    )
                ):
                    return start, goal
        raise RuntimeError("Could not sample a collision-free start and goal")

    def _sample_multi_robot_pairs(
        self, candidates: Sequence[NavigationGoal]
    ) -> dict[str, tuple[NavigationGoal, NavigationGoal]]:
        """Return unique start/goal pairs from candidates for all agents.

        Depending on configured conflict probability, prefer intersecting
        direct routes to expose shared-policy collision handling. Raise
        RuntimeError if disjoint starts and goals cannot be assigned.

        Args:
            candidates: Reachable factory positions eligible for start and goal sampling.

        Returns:
            dict[str, tuple[NavigationGoal, NavigationGoal]]: Unique start/goal pairs from
                candidates for all agents.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        prefer_conflict = (
            self._random.random() < self.settings.multi_robot_conflict_probability
        )
        for require_conflict in ((True, False) if prefer_conflict else (False,)):
            for _ in range(500):
                starts: list[NavigationGoal] = []
                for raw_index in self._random.permutation(len(candidates)):
                    candidate = candidates[int(raw_index)]
                    if not self.scene.is_valid_center(
                        candidate.point,
                        robots=self.world.temporary_obstacles,
                    ):
                        continue
                    if any(
                        candidate.point.distance_to(start.point)
                        < 2.0 * ROBOT_RADIUS_M
                        for start in starts
                    ):
                        continue
                    starts.append(candidate)
                    if len(starts) == len(self.agent_ids):
                        break
                if len(starts) != len(self.agent_ids):
                    continue

                excluded_ids = {start.location_id for start in starts}
                selected_goal_ids: set[str] = set()
                goals: list[NavigationGoal] = []
                for start in starts:
                    choices = []
                    for raw_index in self._random.permutation(len(candidates)):
                        goal = candidates[int(raw_index)]
                        distance = start.point.distance_to(goal.point)
                        if (
                            goal.location_id not in excluded_ids
                            and goal.location_id not in selected_goal_ids
                            and distance >= self.settings.minimum_start_goal_distance_m
                            and (
                                self.settings.maximum_start_goal_distance_m is None
                                or distance
                                <= self.settings.maximum_start_goal_distance_m
                            )
                        ):
                            choices.append(goal)
                    if not choices:
                        break
                    goal = choices[0]
                    goals.append(goal)
                    selected_goal_ids.add(goal.location_id)
                if len(goals) != len(starts):
                    continue
                routes = list(zip(starts, goals, strict=True))
                has_conflict = any(
                    _segments_intersect(
                        first_start.point,
                        first_goal.point,
                        second_start.point,
                        second_goal.point,
                    )
                    for index, (first_start, first_goal) in enumerate(routes)
                    for second_start, second_goal in routes[index + 1 :]
                )
                if require_conflict and not has_conflict:
                    continue
                return {
                    agent_id: pair
                    for agent_id, pair in zip(self.agent_ids, routes, strict=True)
                }
        raise RuntimeError("Could not sample unique multi-Robotino waypoint pairs")

    def _sample_island_corner_pairs(
        self,
    ) -> dict[str, tuple[NavigationGoal, NavigationGoal]]:
        """Return up to three Robotino routes between stations around the island.

        Returns:
            dict[str, tuple[NavigationGoal, NavigationGoal]]: Up to three Robotino routes
                between stations around the island.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if len(self.agent_ids) > 3:
            raise ValueError("Island-corner scenario supports at most three Robotinos")
        station_ids = ["7", "10", "9", "8"]
        offset = int(self._random.integers(0, len(station_ids)))
        direction = int(self._random.choice((1, 2, 3)))
        ordered = station_ids[offset:] + station_ids[:offset]
        routes = [
            (station_id, station_ids[(station_ids.index(station_id) + direction) % 4])
            for station_id in ordered[: len(self.agent_ids)]
        ]
        return {
            agent_id: (
                self.destination_catalog.get(start_id),
                self.destination_catalog.get(goal_id),
            )
            for agent_id, (start_id, goal_id) in zip(
                self.agent_ids, routes, strict=True
            )
        }

    def _sample_orientation_runway_pairs(
        self,
    ) -> dict[str, tuple[NavigationGoal, NavigationGoal]]:
        """Return fixed, mutually isolated routes in the synthetic empty arena.

        Returns:
            dict[str, tuple[NavigationGoal, NavigationGoal]]: Fixed, mutually isolated routes in
                the synthetic empty arena.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        maximum = self.settings.maximum_start_goal_distance_m
        if maximum is None:
            raise RuntimeError("Orientation runway requires a maximum distance")
        minimum = self.settings.orientation_runway_min_distance_m
        if maximum < minimum:
            raise RuntimeError("Orientation runway distance range is empty")
        spacing = self.settings.orientation_runway_lane_spacing_m
        half = float(maximum) / 2.0
        pairs: list[tuple[NavigationGoal, NavigationGoal]] = []
        for index in range(len(self.agent_ids)):
            lane_y = (index - (len(self.agent_ids) - 1) / 2.0) * spacing
            start_point = Point(-half, lane_y)
            goal_point = Point(half, lane_y)
            if not all(
                self.scene.is_valid_center(point)
                for point in (start_point, goal_point)
            ):
                raise RuntimeError("Orientation runway does not fit its arena")
            pairs.append(
                (
                    NavigationGoal(
                        f"runway-start-{index + 1}",
                        start_point,
                        None,
                        GOAL_TOLERANCE_M,
                        GOAL_HEADING_TOLERANCE_RAD,
                    ),
                    NavigationGoal(
                        f"runway-goal-{index + 1}",
                        goal_point,
                        None,
                        GOAL_TOLERANCE_M,
                        GOAL_HEADING_TOLERANCE_RAD,
                    ),
                )
            )
        return {
            agent_id: pair
            for agent_id, pair in zip(self.agent_ids, pairs, strict=True)
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        """Start a sampled or explicitly configured factory episode.

        seed resets the random stream when supplied; options may provide
        start/goal pairs. Returns observations and episode metadata by agent.
        Subsequent seed=None resets continue the existing random stream.

        Args:
            seed: Optional seed for reproducible random sampling.
            options: Optional reset overrides, including fixed starts and goals.

        Returns:
            tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]: Initial
                observations for every simulated agent.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if seed is not None:
            self.seed = int(seed)
            self._random = np.random.default_rng(self.seed)
        # Gymnasium supplies a seed once and then calls reset(seed=None).  Keep
        # advancing that seeded stream; rebuilding it here would repeat one
        # start/goal/heading forever in every vector worker.
        self.episode_seed = int(
            self._random.integers(0, np.iinfo(np.int32).max)
        )
        self.world.reset(seed=self.episode_seed)
        self.robots = self.world.robots
        self.goals.clear()
        self.start_location_ids.clear()
        self.velocities.clear()
        self.previous_actions.clear()
        self.previous_distances.clear()
        self.best_distances.clear()
        self.best_heading_errors.clear()
        self.path_lengths.clear()
        self.minimum_clearances.clear()
        self.action_variations.clear()
        self.command_variations.clear()
        self.angular_effort.clear()
        self.previous_commands.clear()
        self.completed_agents.clear()
        self.failed_agents.clear()
        self.failure_collision_types.clear()
        self.observation_history.reset()
        if self.lidar_memory_360 is not None:
            self.lidar_memory_360.reset()
        self.step_count = 0
        self.time_s = 0.0
        candidates = self._valid_goals()
        if options and options.get("pairs"):
            sampled_pairs = {}
        elif self.settings.scenario_mode == "orientation-runway":
            sampled_pairs = self._sample_orientation_runway_pairs()
        elif self.settings.scenario_mode == "island-corners":
            sampled_pairs = self._sample_island_corner_pairs()
        elif (
            self.settings.scenario_mode == "island-corner-mix"
            and self._random.random() < 0.5
        ):
            sampled_pairs = self._sample_island_corner_pairs()
        elif self.settings.multi_robot_scenario:
            sampled_pairs = self._sample_multi_robot_pairs(candidates)
        else:
            sampled_pairs = {}
        occupied: list[RobotState] = []
        for agent_id in self.agent_ids:
            pair = self._configured_pair(agent_id, options)
            start, goal = pair or sampled_pairs.get(agent_id) or self._sample_pair(
                candidates, occupied
            )
            if not self.scene.is_valid_center(
                start.point,
                robots=(*occupied, *self.world.temporary_obstacles),
            ):
                raise ValueError(
                    f"Configured start {start.location_id} is not collision-free"
                )
            robot = self.robots[agent_id]
            robot.x = start.point.x
            robot.y = start.point.y
            robot.heading_rad = float(self._random.uniform(-math.pi, math.pi))
            robot.trajectory.clear()
            occupied.append(robot)
            selected_goal = replace(
                goal,
                heading_rad=(
                    goal.heading_rad
                    if self.settings.require_final_heading
                    and self.policy_profile is None
                    else None
                ),
            )
            self.goals[agent_id] = selected_goal
            self.start_location_ids[agent_id] = start.location_id
            self.world.set_goal(agent_id, selected_goal.point)
            self.velocities[agent_id] = VelocityCommand()
            self.previous_actions[agent_id] = np.zeros(
                self.action_dimensions, dtype=np.float32
            )
            distance = robot.point.distance_to(selected_goal.point)
            self.previous_distances[agent_id] = distance
            self.best_distances[agent_id] = distance
            initial_bearing = math.atan2(
                selected_goal.point.y - robot.y,
                selected_goal.point.x - robot.x,
            )
            self.best_heading_errors[agent_id] = abs(
                normalize_angle(initial_bearing - robot.heading_rad)
            )
            self.path_lengths[agent_id] = 0.0
            self.minimum_clearances[agent_id] = math.inf
            self.action_variations[agent_id] = 0.0
            self.command_variations[agent_id] = 0.0
            self.angular_effort[agent_id] = 0.0
            self.previous_commands[agent_id] = VelocityCommand()
        self.world.sensor_simulator.reset()
        self.world.sensor_simulator.advance(
            0.0,
            0.0,
            self.robots,
            obstacles=self.world.temporary_obstacles,
        )
        return self.observations(), {
            agent_id: {
                "seed": self.seed,
                "episodeSeed": self.episode_seed,
                "start": self.robots[agent_id].point.to_dict(),
                "goal": self.goals[agent_id].to_dict(),
                "startLocationId": self.start_location_ids[agent_id],
                "goalLocationId": self.goals[agent_id].location_id,
                "features": {
                    "factoryObstacles": self.settings.use_factory_obstacles,
                    "randomObstacles": self.settings.random_obstacles,
                    "sensorNoise": self.settings.sensor_noise,
                    "dynamicsRandomization": self.settings.dynamics_randomization,
                    "scenarioMode": self.settings.scenario_mode,
                },
            }
            for agent_id in self.agent_ids
        }

    def observations(
        self, *, advance_history: bool = True
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return each agent's normalized model observation.

        Set advance_history=False to inspect state without adding another
        temporal frame to the policy input.

        Args:
            advance_history: Whether this observation should add a new temporal frame.

        Returns:
            dict[str, dict[str, np.ndarray]]: Each agent's normalized model observation.
        """

        observations = {}
        for agent_id, robot in self.robots.items():
            lidar_360 = (
                self.lidar_memory_360.observe(
                    robot,
                    now_s=self.time_s,
                    robots=tuple(self.robots.values()),
                )
                if self.lidar_memory_360 is not None
                else None
            )
            observation_arguments = {
                "velocity": (
                    self.velocities[agent_id].vx,
                    self.velocities[agent_id].vy,
                    self.velocities[agent_id].omega,
                ),
                "previous_action": self.previous_actions[agent_id],
            }
            if lidar_360 is not None:
                observation_arguments["lidar_360"] = lidar_360
            current = self.observation_builder.build(
                robot,
                self.goals[agent_id],
                **observation_arguments,
            )
            observations[agent_id] = self.observation_history.augment(
                agent_id, current, advance=advance_history
            )
        return observations

    @staticmethod
    def _minimum_clearance(robot: RobotState) -> float:
        """Return the smallest valid LiDAR/proximity clearance for robot.

        Args:
            robot: Robotino state used by this operation.

        Returns:
            float: The smallest valid LiDAR/proximity clearance for robot.
        """

        values = [
            value
            for value in robot.sensors
            if value < DISTANCE_SENSOR_CLEAR_M - 1e-8
        ]
        if robot.lidar:
            minimum = max(robot.lidar.range_min, 1e-6)
            values.extend(
                max(0.0, value - ROBOT_RADIUS_M)
                for value in robot.lidar.ranges
                if minimum <= value <= robot.lidar.range_max
            )
        return min(values) if values else DISTANCE_SENSOR_CLEAR_M

    def _record_motion_metrics(
        self, previous_points: Mapping[str, Point], period_s: float
    ) -> None:
        """Accumulate path and control metrics after one world-control step.

        previous_points holds positions before physics integration;
        period_s scales angular effort. Updates per-agent metric state in
        place and returns nothing.

        Args:
            previous_points: Accumulate path and control metrics after one world-control step.
                previous_points holds positions before physics integration; period_s scales
                angular effort.
            period_s: Accumulate path and control metrics after one world-control step.
                previous_points holds positions before physics integration; period_s scales
                angular effort.
        """

        for agent_id, robot in self.robots.items():
            self.path_lengths[agent_id] += previous_points[agent_id].distance_to(
                robot.point
            )
            previous_command = self.previous_commands[agent_id]
            applied = robot.applied_command
            self.command_variations[agent_id] += math.sqrt(
                (applied.vx - previous_command.vx) ** 2
                + (applied.vy - previous_command.vy) ** 2
                + (applied.omega - previous_command.omega) ** 2
            )
            self.angular_effort[agent_id] += abs(applied.omega) * period_s
            self.previous_commands[agent_id] = applied

    def step(self, actions: Mapping[str, Sequence[float]]) -> CoreStep:
        """Advance the world once for every configured agent action.

        actions must contain finite profile-sized vectors for all agents.
        Returns next observations, rewards, terminal flags, and diagnostics;
        completed agents receive zero actions until the shared episode ends.

        Args:
            actions: Normalized actions keyed by agent or ordered by rollout slot.

        Returns:
            CoreStep: Next observations, rewards, termination flags, and diagnostics.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        missing = set(self.agent_ids) - set(actions)
        if missing:
            raise ValueError(f"Missing actions for {', '.join(sorted(missing))}")
        normalized: dict[str, np.ndarray] = {}
        for agent_id in self.agent_ids:
            action = np.asarray(actions[agent_id], dtype=np.float32)
            if action.shape != (self.action_dimensions,) or not np.all(
                np.isfinite(action)
            ):
                raise ValueError(
                    f"{agent_id} action must be a finite "
                    f"shape-({self.action_dimensions},) vector"
                )
            action = np.clip(action, -1.0, 1.0)
            if (
                agent_id in self.completed_agents
                or agent_id in self.failed_agents
            ):
                action = np.zeros(self.action_dimensions, dtype=np.float32)
            normalized[agent_id] = action
            self.action_variations[agent_id] += float(
                np.linalg.norm(action - self.previous_actions[agent_id])
            )

        collisions = {agent_id: False for agent_id in self.agent_ids}
        if self.policy_profile is not None:
            plans = (
                {
                    agent_id: self.local_target_controller.plan(
                        self.robots[agent_id], normalized[agent_id]
                    )
                    for agent_id in self.agent_ids
                    if agent_id not in self.failed_agents
                }
                if self.local_target_controller is not None
                else {}
            )
            control_steps = self.policy_profile.control_steps_per_action
            control_period = self.policy_profile.control_period_s
            for _ in range(control_steps):
                previous_points = {
                    key: robot.point for key, robot in self.robots.items()
                }
                if self.local_target_controller is not None:
                    control_outputs = {
                        agent_id: self.local_target_controller.command(
                            self.robots[agent_id],
                            self.goals[agent_id],
                            plans[agent_id],
                        )
                        for agent_id in self.agent_ids
                        if agent_id not in self.failed_agents
                    }
                else:
                    assert self.forward_turn_controller is not None
                    control_outputs = {
                        agent_id: self.forward_turn_controller.command(
                            self.robots[agent_id],
                            self.goals[agent_id],
                            normalized[agent_id],
                        )
                        for agent_id in self.agent_ids
                        if agent_id not in self.failed_agents
                    }
                for agent_id, output in control_outputs.items():
                    if output.arrived:
                        self.world.stop(agent_id)
                desired = {
                    agent_id: output.command
                    for agent_id, output in control_outputs.items()
                }
                current_collisions = self.world.step(desired, control_period)
                for agent_id in self.robots:
                    collisions[agent_id] = (
                        collisions[agent_id] or current_collisions[agent_id]
                    )
                self._record_motion_metrics(previous_points, control_period)
                if any(current_collisions.values()):
                    break
        else:
            desired = {
                agent_id: self.action_scaler.scale(normalized[agent_id])
                for agent_id in self.agent_ids
            }
            previous_points = {
                key: robot.point for key, robot in self.robots.items()
            }
            collisions = self.world.step(desired, self.settings.time_step_s)
            self._record_motion_metrics(previous_points, self.settings.time_step_s)
        self.step_count += 1
        self.time_s = self.world.time_s
        for agent_id, robot in self.robots.items():
            self.velocities[agent_id] = robot.measured_velocity

        removal_mode = (
            self.settings.multi_robot_scenario
            and self.settings.remove_collided_agents
        )
        newly_failed = {
            agent_id
            for agent_id, collided in collisions.items()
            if removal_mode
            and collided
            and agent_id not in self.completed_agents
            and agent_id not in self.failed_agents
        }
        for agent_id in newly_failed:
            self.failure_collision_types[agent_id] = set(
                self.world.last_collision_kinds.get(agent_id, ())
            )
            self.world.deactivate(agent_id)
        self.failed_agents.update(newly_failed)

        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}
        successes: dict[str, bool] = {}
        newly_completed: set[str] = set()
        for agent_id, robot in self.robots.items():
            goal = self.goals[agent_id]
            distance = robot.point.distance_to(goal.point)
            heading_error = (
                normalize_angle(goal.heading_rad - robot.heading_rad)
                if goal.heading_rad is not None
                else 0.0
            )
            success = agent_id not in self.failed_agents and distance <= goal.position_tolerance_m and (
                goal.heading_rad is None
                or abs(heading_error) <= goal.heading_tolerance_rad
            )
            successes[agent_id] = success
            if success and agent_id not in self.completed_agents:
                newly_completed.add(agent_id)
        fleet_success = all(successes.values())
        if self.settings.multi_robot_scenario:
            for agent_id in newly_completed:
                self.world.stop(agent_id)
            self.completed_agents.update(newly_completed)
        for agent_id, robot in self.robots.items():
            goal = self.goals[agent_id]
            distance = robot.point.distance_to(goal.point)
            heading_error = (
                normalize_angle(goal.heading_rad - robot.heading_rad)
                if goal.heading_rad is not None
                else 0.0
            )
            navigation_heading_error = normalize_angle(
                math.atan2(goal.point.y - robot.y, goal.point.x - robot.x)
                - robot.heading_rad
            )
            success = successes[agent_id]
            clearance = self._minimum_clearance(robot)
            self.minimum_clearances[agent_id] = min(
                self.minimum_clearances[agent_id], clearance
            )
            resolved_before_step = (
                (
                    agent_id in self.completed_agents
                    and agent_id not in newly_completed
                )
                or (
                    agent_id in self.failed_agents
                    and agent_id not in newly_failed
                )
            )
            reward = navigation_reward(
                self.settings.reward,
                previous_distance_m=(
                    self.best_distances[agent_id]
                    if self.settings.reward.progress_mode == "best-distance"
                    else self.previous_distances[agent_id]
                ),
                distance_m=distance,
                minimum_clearance_m=clearance,
                collision=(
                    agent_id in newly_failed
                    if removal_mode
                    else collisions[agent_id]
                ),
                success=(
                    agent_id in newly_completed
                    if removal_mode
                    else (
                        fleet_success
                        if self.settings.multi_robot_scenario
                        else success
                    )
                ),
                action=normalized[agent_id],
                previous_action=self.previous_actions[agent_id],
                heading_error_rad=heading_error,
                lateral_speed_mps=robot.measured_velocity.vy,
                forward_speed_mps=robot.measured_velocity.vx,
                angular_speed_rps=robot.measured_velocity.omega,
                previous_navigation_heading_error_rad=self.best_heading_errors[
                    agent_id
                ],
                navigation_heading_error_rad=navigation_heading_error,
            )
            rewards[agent_id] = 0.0 if resolved_before_step else reward.total
            collision_recorded = (
                agent_id in self.failed_agents
                if removal_mode
                else collisions[agent_id]
            )
            collision_types = (
                self.failure_collision_types.get(agent_id, set())
                if removal_mode
                else set(self.world.last_collision_kinds.get(agent_id, ()))
            )
            infos[agent_id] = {
                "rewardComponents": reward.components,
                "success": success,
                "collision": collision_recorded,
                "collisionThisStep": agent_id in newly_failed,
                "collisionTypes": sorted(collision_types),
                "failed": agent_id in self.failed_agents,
                "active": (
                    agent_id not in self.completed_agents
                    and agent_id not in self.failed_agents
                ),
                "fleetSuccess": fleet_success,
                "timeout": False,
                "distanceToGoalM": distance,
                "headingErrorRad": heading_error,
                "navigationHeadingErrorRad": navigation_heading_error,
                "pathLengthM": self.path_lengths[agent_id],
                "minimumClearanceM": self.minimum_clearances[agent_id],
                "policyActionVariation": self.action_variations[agent_id],
                "meanPolicyActionVariationPerStep": (
                    self.action_variations[agent_id] / self.step_count
                ),
                "appliedCommandVariation": self.command_variations[agent_id],
                "meanAppliedCommandVariationPerSecond": (
                    self.command_variations[agent_id] / self.time_s
                    if self.time_s > 0
                    else 0.0
                ),
                "meanAbsOmegaRps": (
                    self.angular_effort[agent_id] / self.time_s
                    if self.time_s > 0
                    else 0.0
                ),
                "elapsedS": self.time_s,
                "step": self.step_count,
                "episodeSeed": self.episode_seed,
                "startLocationId": self.start_location_ids[agent_id],
                "goalLocationId": goal.location_id,
            }
            self.previous_distances[agent_id] = distance
            self.best_distances[agent_id] = min(
                self.best_distances[agent_id], distance
            )
            self.best_heading_errors[agent_id] = min(
                self.best_heading_errors[agent_id],
                abs(navigation_heading_error),
            )
            self.previous_actions[agent_id] = normalized[agent_id]
        all_agents_resolved = len(
            self.completed_agents | self.failed_agents
        ) == len(self.agent_ids)
        terminated = (
            all_agents_resolved
            if removal_mode
            else any(collisions.values()) or fleet_success
        )
        truncated = not terminated and self.step_count >= self.settings.max_steps
        if truncated:
            for agent_id, info in infos.items():
                timed_out = (
                    agent_id not in self.completed_agents
                    and agent_id not in self.failed_agents
                )
                info["timeout"] = timed_out
                if timed_out:
                    penalty = self.settings.reward.timeout
                    rewards[agent_id] += penalty
                    info["rewardComponents"]["timeout"] = penalty
        return CoreStep(self.observations(), rewards, terminated, truncated, infos)
