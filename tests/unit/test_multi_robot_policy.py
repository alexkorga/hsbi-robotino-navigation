"""Test shared-policy inference and multi-Robotino navigation contracts."""

import math
import unittest
from dataclasses import replace

import numpy as np

from robotino_fleet.domain.models import Point, RobotState, VelocityCommand
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.observation import (
    BinnedLocalTargetObservationBuilder,
    TemporalObservationHistory,
)
from robotino_fleet.learning.profile import BinnedLocalTargetProfile
from robotino_fleet.learning.shared_policy_vec_env import SharedPolicyVecEnv
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle
from robotino_fleet.navigation.forward_turn import ForwardTurnMotionController
from robotino_fleet.navigation.local_target import LocalTargetMotionController
from robotino_fleet.navigation.sb3_forward_turn_controller import (
    Sb3ForwardTurnNavigationController,
)
from robotino_fleet.navigation.local_target_policy_controller import (
    LocalTargetPolicyController,
)
from tests.fixtures import TEST_MOTION, fixed_project_settings


class FakeBatchModel:
    def __init__(self) -> None:
        """Prepare a batch-size log for local-target inference tests."""

        self.batch_sizes: list[int] = []

    def predict(self, observation, *, deterministic):
        """Return one fixed target action per observation batch row.

        Args:
            observation: Current policy observation or measured robot state.
            deterministic: Whether to choose the policy's deterministic action.

        Returns:
            object: One fixed target action per observation batch row.
        """

        del deterministic
        batch_size = len(observation["goal_body"])
        self.batch_sizes.append(batch_size)
        return np.tile(np.array([[0.2, 0.0]], dtype=np.float32), (batch_size, 1)), None


class FakeForwardTurnBatchModel:
    def __init__(self) -> None:
        """Prepare a batch-size log for forward-turn inference tests."""

        self.batch_sizes: list[int] = []

    def predict(self, observation, *, deterministic):
        """Return fixed forward-turn actions in observed batch order.

        Args:
            observation: Current policy observation or measured robot state.
            deterministic: Whether to choose the policy's deterministic action.

        Returns:
            object: Fixed forward-turn actions in observed batch order.
        """

        del deterministic
        batch_size = len(observation["goal_direction"])
        self.batch_sizes.append(batch_size)
        return np.tile(np.array([[1.0, 0.2]], dtype=np.float32), (batch_size, 1)), None


class FakeLearnedYawBatchModel:
    def __init__(self) -> None:
        """Prepare a batch-size log for learned-yaw inference tests."""

        self.batch_sizes: list[int] = []

    def predict(self, observation, *, deterministic):
        """Return fixed three-value local-target/yaw actions for a batch.

        Args:
            observation: Current policy observation or measured robot state.
            deterministic: Whether to choose the policy's deterministic action.

        Returns:
            object: Fixed three-value local-target/yaw actions for a batch.
        """

        del deterministic
        batch_size = len(observation["goal_body"])
        self.batch_sizes.append(batch_size)
        return np.tile(
            np.array([[0.2, 0.0, -0.5]], dtype=np.float32),
            (batch_size, 1),
        ), None


class MultiRobotPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use the baseline binned profile in shared-policy scenarios."""

        self.profile = BinnedLocalTargetProfile()

    def settings(self) -> LearningEnvironmentSettings:
        """Return three-interacting-Robotino settings for short test episodes.

        Returns:
            LearningEnvironmentSettings: Three-interacting-Robotino settings for short test
                episodes.
        """

        return LearningEnvironmentSettings(
            robot_count=3,
            time_step_s=self.profile.policy_period_s,
            max_steps=5,
            policy_profile=self.profile,
            multi_robot_scenario=True,
            multi_robot_conflict_probability=0.5,
        )

    def test_joint_reset_uses_unique_disjoint_waypoints(self) -> None:
        """Place each robot at a unique start and assign nonoverlapping goals."""

        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=self.settings(),
            world_count=2,
            seed=123,
        )
        try:
            observation = environment.reset()
            self.assertEqual(environment.num_envs, 6)
            self.assertEqual(observation["lidar_closeness"].shape, (6, 48))
            for world in environment.worlds:
                core = world.core
                starts = set(core.start_location_ids.values())
                goals = {goal.location_id for goal in core.goals.values()}
                self.assertEqual(len(starts), 3)
                self.assertEqual(len(goals), 3)
                self.assertTrue(starts.isdisjoint(goals))
        finally:
            environment.close()

    def test_vec_env_returns_one_transition_per_robot(self) -> None:
        """Flatten a shared world into one SB3 transition per Robotino."""

        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=self.settings(),
            world_count=1,
            seed=456,
        )
        try:
            environment.reset()
            observation, rewards, dones, infos = environment.step(
                np.zeros((3, 2), dtype=np.float32)
            )
            self.assertEqual(observation["goal_body"].shape, (3, 2))
            self.assertEqual(rewards.shape, (3,))
            self.assertEqual(dones.shape, (3,))
            self.assertEqual(len(infos), 3)
            self.assertTrue(all("fleetSuccess" in info for info in infos))
        finally:
            environment.close()

    def test_six_robotino_learned_yaw_world_has_six_three_value_actions(self) -> None:
        """Match six-agent yaw actions and temporal observations to the profile."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        settings = LearningEnvironmentSettings(
            robot_count=6,
            time_step_s=profile.policy_period_s,
            max_steps=2,
            policy_profile=profile,
            multi_robot_scenario=True,
            multi_robot_conflict_probability=0.5,
        )
        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=settings,
            world_count=1,
            seed=457,
        )
        try:
            observation = environment.reset()
            core = environment.worlds[0].core
            self.assertEqual(environment.num_envs, 6)
            self.assertEqual(environment.action_space.shape, (3,))
            self.assertEqual(observation["action_history"].shape, (6, 4, 3))
            self.assertEqual(len(set(core.start_location_ids.values())), 6)
            self.assertEqual(len({goal.location_id for goal in core.goals.values()}), 6)
            _, rewards, dones, infos = environment.step(
                np.zeros((6, 3), dtype=np.float32)
            )
            self.assertEqual(rewards.shape, (6,))
            self.assertEqual(dones.shape, (6,))
            self.assertEqual(len(infos), 6)
        finally:
            environment.close()

    def test_orientation_runway_uses_fixed_isolated_lanes(self) -> None:
        """Generate independent runway lanes rather than conflicting routes."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        settings = LearningEnvironmentSettings(
            robot_count=6,
            time_step_s=profile.policy_period_s,
            max_steps=2,
            maximum_start_goal_distance_m=4.0,
            use_factory_obstacles=False,
            policy_profile=profile,
            multi_robot_scenario=True,
            scenario_mode="orientation-runway",
        )
        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=settings,
            world_count=1,
            seed=458,
        )
        try:
            observation = environment.reset()
            core = environment.worlds[0].core
            self.assertEqual(
                core.scene.movement_bounds,
                (-40.0, -40.0, 40.0, 40.0),
            )
            first_headings = {}
            first_routes = {}
            for index, agent_id in enumerate(environment.agent_ids):
                robot = core.robots[agent_id]
                goal = core.goals[agent_id]
                expected_y = (index - 2.5) * 10.0
                self.assertEqual(robot.point, Point(-2.0, expected_y))
                self.assertEqual(goal.point, Point(2.0, expected_y))
                self.assertAlmostEqual(robot.point.distance_to(goal.point), 4.0)
                self.assertTrue(
                    core.start_location_ids[agent_id].startswith("runway-start-")
                )
                self.assertTrue(goal.location_id.startswith("runway-goal-"))
                self.assertGreaterEqual(robot.heading_rad, -math.pi)
                self.assertLessEqual(robot.heading_rad, math.pi)
                self.assertAlmostEqual(
                    float(observation["lidar_closeness"][index].max()),
                    0.0,
                )
                self.assertAlmostEqual(
                    float(observation["proximity_closeness"][index].max()),
                    0.0,
                )
                first_headings[agent_id] = robot.heading_rad
                first_routes[agent_id] = (robot.point, goal.point)

            environment.reset()
            self.assertTrue(
                any(
                    not math.isclose(
                        core.robots[agent_id].heading_rad,
                        first_headings[agent_id],
                    )
                    for agent_id in environment.agent_ids
                )
            )
            self.assertEqual(
                {
                    agent_id: (
                        core.robots[agent_id].point,
                        core.goals[agent_id].point,
                    )
                    for agent_id in environment.agent_ids
                },
                first_routes,
            )
        finally:
            environment.close()

    def test_island_corner_scenario_uses_unique_island_station_routes(self) -> None:
        """Sample distinct island-station starts/goals for corner training."""

        settings = replace(
            self.settings(),
            scenario_mode="island-corners",
            use_factory_obstacles=True,
        )
        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=settings,
            world_count=1,
            seed=459,
        )
        try:
            environment.reset()
            core = environment.worlds[0].core
            starts = set(core.start_location_ids.values())
            goals = {goal.location_id for goal in core.goals.values()}
            island_stations = {"7", "8", "9", "10"}
            self.assertEqual(len(starts), 3)
            self.assertEqual(len(goals), 3)
            self.assertTrue(starts <= island_stations)
            self.assertTrue(goals <= island_stations)
            self.assertNotEqual(starts, goals)
            self.assertTrue(
                all(
                    core.start_location_ids[agent_id]
                    != core.goals[agent_id].location_id
                    for agent_id in environment.agent_ids
                )
            )
        finally:
            environment.close()

    def test_parallel_world_workers_return_all_robot_transitions(self) -> None:
        """Collect every Robotino transition from spawned world processes."""

        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=self.settings(),
            world_count=2,
            seed=654,
            parallel_worlds=True,
        )
        try:
            observation = environment.reset()
            self.assertEqual(observation["goal_body"].shape, (6, 2))
            next_observation, rewards, dones, infos = environment.step(
                np.zeros((6, 2), dtype=np.float32)
            )
            self.assertEqual(next_observation["lidar_closeness"].shape, (6, 48))
            self.assertEqual(rewards.shape, (6,))
            self.assertEqual(dones.shape, (6,))
            self.assertEqual(len(infos), 6)
        finally:
            environment.close()

    def test_robotino_collisions_are_identified_separately(self) -> None:
        """Distinguish peer collisions from environment collisions in info."""

        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=self.settings(),
            world_count=1,
            seed=789,
        )
        try:
            environment.reset()
            world = environment.worlds[0].core.world
            first_id, second_id, _ = environment.agent_ids
            first = world.robots[first_id]
            second = world.robots[second_id]
            second.x = first.x + 0.1
            second.y = first.y
            collisions = world.step(
                {
                    robot_id: VelocityCommand()
                    for robot_id in environment.agent_ids
                },
                0.05,
            )
            self.assertTrue(collisions[first_id])
            self.assertTrue(collisions[second_id])
            self.assertIn("robotino", world.last_collision_kinds[first_id])
            self.assertIn("robotino", world.last_collision_kinds[second_id])
        finally:
            environment.close()

    def test_collided_agents_disappear_without_resetting_survivors(self) -> None:
        """Deactivate a collided agent while other robots continue the episode."""

        settings = replace(
            self.settings(),
            max_steps=2,
            remove_collided_agents=True,
        )
        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=settings,
            world_count=1,
            seed=790,
        )
        try:
            environment.reset()
            core = environment.worlds[0].core
            world = core.world
            first_id, second_id, survivor_id = environment.agent_ids
            first = world.robots[first_id]
            second = world.robots[second_id]
            second.x = first.x + 0.1
            second.y = first.y

            observation, rewards, dones, infos = environment.step(
                np.zeros((3, 2), dtype=np.float32)
            )

            self.assertFalse(dones.any())
            self.assertEqual(core.failed_agents, {first_id, second_id})
            self.assertEqual(world.inactive_robot_ids, {first_id, second_id})
            self.assertFalse(first.pose_valid)
            self.assertFalse(second.pose_valid)
            self.assertIsNone(first.lidar)
            self.assertIsNone(second.lidar)
            self.assertTrue(infos[0]["collision"])
            self.assertTrue(infos[1]["collision"])
            self.assertFalse(infos[2]["collision"])
            self.assertTrue(infos[2]["active"])
            self.assertEqual(observation["goal_body"].shape, (3, 2))

            _, next_rewards, final_dones, final_infos = environment.step(
                np.zeros((3, 2), dtype=np.float32)
            )

            self.assertTrue(final_dones.all())
            self.assertEqual(float(next_rewards[0]), 0.0)
            self.assertEqual(float(next_rewards[1]), 0.0)
            self.assertTrue(final_infos[0]["collision"])
            self.assertTrue(final_infos[1]["collision"])
            self.assertFalse(final_infos[0]["timeout"])
            self.assertFalse(final_infos[1]["timeout"])
            self.assertTrue(final_infos[2]["timeout"])
            self.assertTrue(world.robots[first_id].pose_valid)
            self.assertTrue(world.robots[second_id].pose_valid)
            self.assertFalse(core.failed_agents)
            self.assertIn(survivor_id, world.robots)
        finally:
            environment.close()

    def test_individual_success_is_rewarded_without_resetting_world(self) -> None:
        """Reward a finished agent without ending the other robots' rollout."""

        environment = SharedPolicyVecEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=self.settings(),
            world_count=1,
            seed=791,
        )
        try:
            environment.reset()
            core = environment.worlds[0].core
            first_id = environment.agent_ids[0]
            robot = core.robots[first_id]
            goal = core.goals[first_id]
            robot.x = goal.point.x
            robot.y = goal.point.y

            _, rewards, dones, infos = environment.step(
                np.zeros((3, 2), dtype=np.float32)
            )

            self.assertFalse(dones.any())
            self.assertIn(first_id, core.completed_agents)
            self.assertTrue(infos[0]["success"])
            self.assertFalse(infos[0]["active"])
            self.assertEqual(
                infos[0]["rewardComponents"]["success"],
                core.settings.reward.success,
            )
            self.assertGreater(float(rewards[0]), 0.0)
        finally:
            environment.close()

    def test_runtime_batches_only_robotinos_whose_inference_is_due(self) -> None:
        """Infer only due local-target policies and hold other agents' plans."""

        controller = LocalTargetPolicyController.__new__(
            LocalTargetPolicyController
        )
        controller.profile = self.profile
        controller.observation_builder = BinnedLocalTargetObservationBuilder(
            profile=self.profile
        )
        controller.observation_history = TemporalObservationHistory(
            self.profile.temporal_frames
        )
        controller.lidar_memory_360 = None
        controller.motion_controller = LocalTargetMotionController(
            TEST_MOTION, self.profile
        )
        controller.model = FakeBatchModel()
        controller.deterministic = True
        controller.policy_id = "test-shared-policy"
        controller._plans = {}
        controller._previous_actions = {}
        controller._last_inference_s = {}
        controller._last_inference_ms = {}
        controller._handover_robot_ids = set()

        robots = {
            robot_id: RobotState(robot_id, "#fff", 0.0, float(index), 0.0)
            for index, robot_id in enumerate(("r1", "r2"))
        }
        goals = {
            robot_id: NavigationGoal(
                f"g-{robot_id}", Point(3.0, robot.y), None, 0.1, 0.15
            )
            for robot_id, robot in robots.items()
        }
        first = controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.0)
        self.assertEqual(set(first), {"r1", "r2"})
        self.assertEqual(controller.model.batch_sizes, [2])

        controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.10)
        self.assertEqual(controller.model.batch_sizes, [2])

        robots["r3"] = RobotState("r3", "#fff", 0.0, 2.0, 0.0)
        goals["r3"] = NavigationGoal("g-r3", Point(3.0, 2.0), None, 0.1, 0.15)
        third = controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.10)
        self.assertEqual(set(third), {"r1", "r2", "r3"})
        self.assertEqual(controller.model.batch_sizes, [2, 1])
        self.assertEqual(set(controller._previous_actions), {"r1", "r2", "r3"})

    def test_forward_turn_runtime_dynamically_batches_staggered_robotinos(self) -> None:
        """Batch staggered forward-turn inference only when each period expires."""

        profile = BinnedLocalTargetProfile(
            name="binned-forward-turn-temporal-v1",
            schema_version=3,
            action_mode="forward-turn",
            temporal_frames=6,
        )
        controller = Sb3ForwardTurnNavigationController.__new__(
            Sb3ForwardTurnNavigationController
        )
        controller.profile = profile
        controller.observation_builder = BinnedLocalTargetObservationBuilder(profile)
        controller.observation_history = TemporalObservationHistory(6)
        controller.motion_controller = ForwardTurnMotionController(
            TEST_MOTION, profile
        )
        controller.model = FakeForwardTurnBatchModel()
        controller.deterministic = True
        controller.policy_id = "test-forward-turn-policy"
        controller._actions = {}
        controller._last_inference_s = {}
        controller._last_inference_ms = {}
        controller._handover_robot_ids = set()

        robots = {
            robot_id: RobotState(robot_id, "#fff", 0.0, float(index), 0.0)
            for index, robot_id in enumerate(("r1", "r2"))
        }
        goals = {
            robot_id: NavigationGoal(
                f"g-{robot_id}", Point(3.0, robot.y), None, 0.1, 0.15
            )
            for robot_id, robot in robots.items()
        }
        first = controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.0)
        self.assertEqual(controller.model.batch_sizes, [2])
        self.assertTrue(all(output.command.vx > 0.0 for output in first.values()))
        self.assertTrue(all(output.command.vy == 0.0 for output in first.values()))

        controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.10)
        self.assertEqual(controller.model.batch_sizes, [2])

        robots["r3"] = RobotState("r3", "#fff", 0.0, 2.0, 0.0)
        goals["r3"] = NavigationGoal("g-r3", Point(3.0, 2.0), None, 0.1, 0.15)
        controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.10)
        self.assertEqual(controller.model.batch_sizes, [2, 1])
        self.assertEqual(set(controller._actions), {"r1", "r2", "r3"})

    def test_learned_yaw_runtime_keeps_updating_inside_approach_region(self) -> None:
        """Continue learned yaw inference during final positional approach."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        controller = LocalTargetPolicyController.__new__(
            LocalTargetPolicyController
        )
        controller.profile = profile
        controller.observation_builder = BinnedLocalTargetObservationBuilder(profile)
        controller.observation_history = TemporalObservationHistory(4)
        controller.lidar_memory_360 = None
        controller.motion_controller = LocalTargetMotionController(
            TEST_MOTION, profile
        )
        controller.model = FakeLearnedYawBatchModel()
        controller.deterministic = True
        controller.policy_id = "test-learned-yaw-policy"
        controller._plans = {}
        controller._previous_actions = {}
        controller._last_inference_s = {}
        controller._last_inference_ms = {}
        controller._handover_robot_ids = set()

        robot = RobotState("r1", "#fff", 0.0, 0.0, 0.0)
        goal = NavigationGoal("g-r1", Point(0.3, 0.0), None, 0.1, 0.15)
        output = controller.compute_command(
            robot,
            goal,
            delta_s=0.05,
            now_s=0.0,
        )

        self.assertEqual(controller.model.batch_sizes, [1])
        self.assertLess(output.command.omega, 0.0)

    def test_model_is_not_called_after_position_handover(self) -> None:
        """Use deterministic alignment, not policy inference, after handover."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        controller = LocalTargetPolicyController.__new__(
            LocalTargetPolicyController
        )
        controller.profile = profile
        controller.observation_builder = BinnedLocalTargetObservationBuilder(profile)
        controller.observation_history = TemporalObservationHistory(4)
        controller.lidar_memory_360 = None
        controller.motion_controller = LocalTargetMotionController(
            TEST_MOTION, profile
        )
        controller.model = FakeLearnedYawBatchModel()
        controller.deterministic = True
        controller.policy_id = "test-learned-yaw-policy"
        controller._plans = {}
        controller._previous_actions = {}
        controller._last_inference_s = {}
        controller._last_inference_ms = {}
        controller._handover_robot_ids = set()

        robot = RobotState("r1", "#fff", 0.0, 0.0, 0.0)
        goal = NavigationGoal(
            "g-r1",
            Point(0.08, 0.0),
            math.pi / 2,
            0.1,
            0.1,
            docking_handoff="dock:test",
        )
        output = controller.compute_command(robot, goal, delta_s=0.05, now_s=0.0)

        self.assertEqual(controller.model.batch_sizes, [])
        self.assertEqual(output.state, NavigationLifecycle.ALIGNING)
        self.assertEqual(output.command.vx, 0.0)
        self.assertEqual(output.command.vy, 0.0)

        robot.x = 0.25
        output = controller.compute_command(robot, goal, delta_s=0.05, now_s=0.1)
        self.assertEqual(controller.model.batch_sizes, [])
        self.assertEqual(output.state, NavigationLifecycle.ALIGNING)
        self.assertEqual(output.command.vx, 0.0)
        self.assertEqual(output.command.vy, 0.0)


if __name__ == "__main__":
    unittest.main()
