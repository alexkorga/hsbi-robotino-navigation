"""Validate local-target profiles, observations, handover, and control."""

import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from robotino_fleet.domain.models import LidarScanObservation, Point, RobotState
from robotino_fleet.learning.config import LearningEnvironmentSettings, RewardSettings
from robotino_fleet.learning.gym_env import SingleRobotinoGymEnv
from robotino_fleet.learning.extractors import (
    BinnedLocalTargetExtractor,
    ExpandedBinnedLocalTargetExtractor,
    ForwardTurnTemporalExtractor,
    PoseCorrected360TemporalExtractor,
    TemporalExpandedBinnedLocalTargetExtractor,
    WidePoseCorrected360TemporalExtractor,
    WideTemporalBinnedLocalTargetExtractor,
)
from robotino_fleet.learning.observation import (
    BinnedLocalTargetObservationBuilder,
    PoseCorrectedLidar360Memory,
    TemporalObservationHistory,
)
from robotino_fleet.learning.profile import (
    BinnedLocalTargetProfile,
    load_policy_profile,
    write_policy_profile,
)
from robotino_fleet.learning.rewards import navigation_reward
from robotino_fleet.navigation.goals import NavigationGoal
from robotino_fleet.navigation.goals import NavigationLifecycle
from robotino_fleet.navigation.forward_turn import ForwardTurnMotionController
from robotino_fleet.navigation.local_target import LocalTargetMotionController
from robotino_fleet.navigation.motion import MotionLimiter
from tests.fixtures import TEST_MOTION, fixed_project_settings


class LocalTargetPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use a small LiDAR-bin profile for policy-contract tests."""

        self.profile = BinnedLocalTargetProfile(lidar_bin_count=8)

    def test_binning_has_fixed_shape_and_preserves_missing_returns(self) -> None:
        """Keep absent scan rays distinct from valid close readings in bins."""

        sample_count = 16
        span = self.profile.lidar_angle_max_rad - self.profile.lidar_angle_min_rad
        ranges = [0.0] * sample_count
        ranges[0] = 1.0
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        robot.lidar = LidarScanObservation(
            sequence=1,
            source_timestamp_s=0.0,
            timestamp_s=0.0,
            angle_min=self.profile.lidar_angle_min_rad,
            angle_max=self.profile.lidar_angle_max_rad,
            angle_increment=span / (sample_count - 1),
            range_min=0.0,
            range_max=20.0,
            ranges=tuple(ranges),
        )
        builder = BinnedLocalTargetObservationBuilder(
            profile=self.profile,
        )
        goal = NavigationGoal("g", Point(1.0, 0.0), None, 0.1, 0.15)
        observation = builder.build(robot, goal)
        self.assertEqual(observation["lidar_closeness"].shape, (8,))
        self.assertEqual(observation["lidar_validity"].shape, (8,))
        self.assertGreater(observation["lidar_closeness"][0], 0.79)
        self.assertGreater(observation["lidar_validity"][0], 0.0)
        self.assertLess(observation["lidar_validity"][0], 1.0)
        self.assertEqual(float(observation["lidar_validity"][4]), 0.0)

    def test_expanded_extractor_starts_as_exact_baseline(self) -> None:
        """Initialize residual extractor capacity without changing baseline output."""

        space = BinnedLocalTargetObservationBuilder(profile=self.profile).space()
        baseline = BinnedLocalTargetExtractor(space)
        expanded = ExpandedBinnedLocalTargetExtractor(space)
        transfer = expanded.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(transfer.unexpected_keys)
        self.assertTrue(transfer.missing_keys)
        sample = {
            key: torch.as_tensor(value.sample()).unsqueeze(0)
            for key, value in space.spaces.items()
        }
        torch.testing.assert_close(expanded(sample), baseline(sample))

    def test_temporal_history_repeats_first_frame_then_rolls(self) -> None:
        """Fill initial history with one frame, then advance the frame window."""

        temporal_profile = BinnedLocalTargetProfile(
            name="binned-local-target-temporal-v1",
            schema_version=2,
            temporal_frames=4,
            lidar_bin_count=8,
        )
        builder = BinnedLocalTargetObservationBuilder(temporal_profile)
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        goal = NavigationGoal("g", Point(1.0, 0.0), None, 0.1, 0.15)
        history = TemporalObservationHistory(4)
        first = history.augment("r", builder.build(robot, goal))
        self.assertEqual(first["lidar_closeness_history"].shape, (4, 8))
        np.testing.assert_array_equal(
            first["velocity_history"], np.zeros((4, 3), dtype=np.float32)
        )
        second = history.augment(
            "r", builder.build(robot, goal, velocity=(0.5, 0.0, 0.0))
        )
        np.testing.assert_array_equal(
            second["velocity_history"][:, 0],
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_temporal_extractor_starts_as_exact_expanded_model(self) -> None:
        """Start temporal residual training from identical expanded-CNN features."""

        temporal_profile = BinnedLocalTargetProfile(
            name="binned-local-target-temporal-v1",
            schema_version=2,
            temporal_frames=4,
            lidar_bin_count=8,
        )
        temporal_space = BinnedLocalTargetObservationBuilder(
            temporal_profile
        ).space()
        current_space = BinnedLocalTargetObservationBuilder(self.profile).space()
        expanded = ExpandedBinnedLocalTargetExtractor(current_space)
        temporal = TemporalExpandedBinnedLocalTargetExtractor(temporal_space)
        transfer = temporal.load_state_dict(expanded.state_dict(), strict=False)
        self.assertFalse(transfer.unexpected_keys)
        self.assertTrue(transfer.missing_keys)
        sample = {
            key: torch.as_tensor(value.sample()).unsqueeze(0)
            for key, value in temporal_space.spaces.items()
        }
        current_sample = {key: sample[key] for key in current_space.spaces}
        torch.testing.assert_close(temporal(sample), expanded(current_sample))
        self.assertEqual(
            WideTemporalBinnedLocalTargetExtractor(temporal_space)(sample).shape,
            (1, 256),
        )

    def test_learned_yaw_contract_expands_action_and_temporal_history(self) -> None:
        """Require a third action and matching history shape for learned yaw."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
            lidar_bin_count=8,
        )
        builder = BinnedLocalTargetObservationBuilder(profile)
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        goal = NavigationGoal("g", Point(1.0, 0.0), None, 0.1, 0.15)
        observation = builder.build(robot, goal)
        temporal = TemporalObservationHistory(4).augment("r", observation)

        self.assertEqual(profile.action_dimensions, 3)
        self.assertEqual(builder.space()["previous_action"].shape, (3,))
        self.assertEqual(temporal["action_history"].shape, (4, 3))
        sample = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in temporal.items()
        }
        self.assertEqual(
            WideTemporalBinnedLocalTargetExtractor(builder.space())(sample).shape,
            (1, 256),
        )

    def test_learned_yaw_action_controls_rotation_instead_of_goal_bearing(self) -> None:
        """Use policy rotation output rather than geometric bearing control."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        motion = TEST_MOTION
        controller = LocalTargetMotionController(motion, profile)
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        # The goal lies to the left, but the learned action explicitly turns right.
        goal = NavigationGoal("g", Point(0.0, 2.0), None, 0.1, 0.15)
        output = controller.command(robot, goal, controller.plan(robot, (1.0, 0.0, -0.5)))

        self.assertAlmostEqual(output.command.omega, -0.5 * motion.rotation_max_rps)

    def test_forward_turn_observation_has_six_frames_and_unit_bearing(self) -> None:
        """Represent forward-turn history and goal direction as trained."""

        profile = BinnedLocalTargetProfile(
            name="binned-forward-turn-temporal-v1",
            schema_version=3,
            action_mode="forward-turn",
            temporal_frames=6,
            lidar_bin_count=8,
        )
        builder = BinnedLocalTargetObservationBuilder(profile)
        robot = RobotState("r", "#fff", 1.0, 1.0, math.pi / 2)
        goal = NavigationGoal("g", Point(3.0, 1.0), None, 0.1, 0.15)
        observation = builder.build(robot, goal)
        self.assertNotIn("goal_body", observation)
        np.testing.assert_allclose(
            observation["goal_direction"],
            np.asarray([0.0, -1.0], dtype=np.float32),
            atol=1e-6,
        )
        temporal = TemporalObservationHistory(6).augment("r", observation)
        self.assertEqual(temporal["lidar_closeness_history"].shape, (6, 8))
        self.assertEqual(temporal["proximity_closeness_history"].shape, (6, 9))
        self.assertEqual(temporal["velocity_history"].shape, (6, 3))
        self.assertEqual(temporal["action_history"].shape, (6, 2))

    def test_forward_turn_extractor_produces_policy_features(self) -> None:
        """Encode six-frame forward-turn observations to the expected width."""

        profile = BinnedLocalTargetProfile(
            name="binned-forward-turn-temporal-v1",
            schema_version=3,
            action_mode="forward-turn",
            temporal_frames=6,
            lidar_bin_count=8,
        )
        space = BinnedLocalTargetObservationBuilder(profile).space()
        extractor = ForwardTurnTemporalExtractor(space)
        sample = {
            key: torch.as_tensor(value.sample()).unsqueeze(0)
            for key, value in space.spaces.items()
        }
        self.assertEqual(extractor(sample).shape, (1, 192))

    def test_forward_turn_controller_never_commands_sideways_or_reverse(self) -> None:
        """Constrain forward-turn policy actions to forward drive and yaw."""

        profile = BinnedLocalTargetProfile(
            name="binned-forward-turn-temporal-v1",
            schema_version=3,
            action_mode="forward-turn",
            temporal_frames=6,
        )
        motion = TEST_MOTION
        controller = ForwardTurnMotionController(motion, profile)
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        far_goal = NavigationGoal("g", Point(2.0, 0.0), None, 0.1, 0.15)

        rotate = controller.command(robot, far_goal, (-1.0, 1.0))
        self.assertEqual(rotate.command.vx, 0.0)
        self.assertEqual(rotate.command.vy, 0.0)
        self.assertAlmostEqual(rotate.command.omega, motion.rotation_max_rps)

        forward = controller.command(robot, far_goal, (1.0, 0.5))
        self.assertAlmostEqual(forward.command.vx, motion.forward_max_mps)
        self.assertEqual(forward.command.vy, 0.0)
        self.assertAlmostEqual(forward.command.omega, motion.rotation_max_rps * 0.5)

        near_goal = NavigationGoal("near", Point(0.100001, 0.0), None, 0.1, 0.15)
        near = controller.command(robot, near_goal, (1.0, 0.0))
        self.assertAlmostEqual(near.command.vx, profile.handover_min_speed_mps, places=5)

        arrived_goal = NavigationGoal("done", Point(0.1, 0.0), None, 0.1, 0.15)
        arrived = controller.command(robot, arrived_goal, (1.0, 1.0))
        self.assertTrue(arrived.arrived)
        self.assertFalse(arrived.command.moving)

    def test_pose_corrected_lidar_memory_moves_hits_into_current_frame(self) -> None:
        """Reproject stored world hits when the Robotino pose changes."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-360-temporal-v1",
            schema_version=4,
            temporal_frames=6,
            lidar_bin_count=8,
            lidar_angle_min_rad=-math.pi,
            lidar_angle_max_rad=math.pi,
        )
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        robot.lidar = LidarScanObservation(
            sequence=1,
            source_timestamp_s=0.0,
            timestamp_s=0.0,
            angle_min=0.0,
            angle_max=0.0,
            angle_increment=1.0,
            range_min=0.01,
            range_max=20.0,
            ranges=(1.0,),
        )
        memory = PoseCorrectedLidar360Memory(profile)
        front = memory.observe(robot, now_s=0.0)
        self.assertGreater(front[0][4], 0.0)
        self.assertEqual(front[2][4], 1.0)

        robot.heading_rad = math.pi
        rear = memory.observe(robot, now_s=1.0)
        rear_index = int(np.argmax(rear[0]))
        self.assertIn(rear_index, (0, 7))
        self.assertAlmostEqual(float(rear[2][rear_index]), 0.75)

    def test_360_temporal_extractor_accepts_six_frame_contract(self) -> None:
        """Accept six-frame pose-corrected 360° LiDAR observation fields."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-360-temporal-v1",
            schema_version=4,
            temporal_frames=6,
            lidar_bin_count=16,
            lidar_angle_min_rad=-math.pi,
            lidar_angle_max_rad=math.pi,
        )
        builder = BinnedLocalTargetObservationBuilder(profile)
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        goal = NavigationGoal("g", Point(1.0, 0.0), None, 0.1, 0.15)
        memory = PoseCorrectedLidar360Memory(profile)
        current = builder.build(
            robot,
            goal,
            lidar_360=memory.observe(robot, now_s=0.0),
        )
        observation = TemporalObservationHistory(6).augment("r", current)
        self.assertEqual(observation["lidar_freshness_history"].shape, (6, 16))
        extractor = PoseCorrected360TemporalExtractor(builder.space())
        sample = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in observation.items()
        }
        self.assertEqual(extractor(sample).shape, (1, 192))
        self.assertEqual(
            WidePoseCorrected360TemporalExtractor(builder.space())(sample).shape,
            (1, 256),
        )

    def test_local_target_is_frozen_in_map_frame(self) -> None:
        """Hold a learned lookahead at one world point while the robot turns."""

        robot = RobotState("r", "#fff", 1.0, 2.0, 0.0)
        controller = LocalTargetMotionController(TEST_MOTION, self.profile)
        plan = controller.plan(robot, (1.0, 0.0))
        self.assertAlmostEqual(plan.point.x, 1.75)
        self.assertAlmostEqual(plan.point.y, 2.0)
        robot.heading_rad = math.pi / 2
        goal = NavigationGoal("g", Point(4.0, 2.0), None, 0.1, 0.15)
        controller.command(robot, goal, plan)
        self.assertAlmostEqual(plan.point.x, 1.75)
        self.assertAlmostEqual(plan.point.y, 2.0)

    def test_handover_speed_falls_with_remaining_waypoint_distance(self) -> None:
        """Taper local-target translation as waypoint handover approaches."""

        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        controller = LocalTargetMotionController(TEST_MOTION, self.profile)
        plan = controller.plan(robot, (1.0, 0.0))

        goal = NavigationGoal("g", Point(0.30, 0.0), None, 0.10, 0.15)
        at_thirty_cm = controller.command(robot, goal, plan)
        self.assertAlmostEqual(
            abs(at_thirty_cm.command.vx), 0.10 + 0.40 * 0.20 / 0.90
        )

        goal = NavigationGoal("g", Point(0.20, 0.0), None, 0.10, 0.15)
        at_twenty_cm = controller.command(robot, goal, plan)
        self.assertAlmostEqual(
            abs(at_twenty_cm.command.vx), 0.10 + 0.40 * 0.10 / 0.90
        )

        goal = NavigationGoal("g", Point(0.15, 0.0), None, 0.10, 0.15)
        at_fifteen_cm = controller.command(robot, goal, plan)
        self.assertAlmostEqual(
            abs(at_fifteen_cm.command.vx), 0.10 + 0.40 * 0.05 / 0.90
        )

    def test_handover_minimum_speed_crosses_completion_boundary(self) -> None:
        """Keep a usable crawl until positional handover completes."""

        motion = TEST_MOTION
        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        controller = LocalTargetMotionController(motion, self.profile)
        limiter = MotionLimiter(motion)
        goal = NavigationGoal("g", Point(0.30, 0.0), None, 0.10, 0.15)
        output = None
        for _ in range(200):
            output = controller.command(robot, goal, controller.plan(robot, (1.0, 0.0)))
            applied = limiter.apply(robot.id, output.command, 0.05)
            robot.x += applied.vx * 0.05
            if output.arrived:
                break
        assert output is not None
        self.assertTrue(output.arrived)
        self.assertLessEqual(robot.point.distance_to(goal.point), 0.10)

    def test_handover_speed_region_has_no_entry_discontinuity(self) -> None:
        """Enter the slowdown radius without a sudden commanded-speed jump."""

        robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)
        controller = LocalTargetMotionController(TEST_MOTION, self.profile)
        plan = controller.plan(robot, (1.0, 0.0))
        outside = controller.command(
            robot,
            NavigationGoal("g", Point(1.01, 0.0), None, 0.10, 0.15),
            plan,
        )
        boundary = controller.command(
            robot,
            NavigationGoal("g", Point(1.0, 0.0), None, 0.10, 0.15),
            plan,
        )
        self.assertAlmostEqual(outside.command.vx, 0.5)
        self.assertAlmostEqual(boundary.command.vx, 0.5)

    def test_handover_aligns_after_position_control_finishes(self) -> None:
        """Rotate to destination heading only after positional tolerance is met."""

        robot = RobotState("r", "#fff", 0.0, 0.0, math.pi)
        motion = TEST_MOTION
        controller = LocalTargetMotionController(motion, self.profile)
        plan = controller.plan(robot, (1.0, 0.0))
        goal = NavigationGoal(
            "dock",
            Point(0.08, 0.0),
            0.0,
            0.10,
            0.10,
            docking_handoff="dock:test",
        )
        output = controller.command(robot, goal, plan)
        self.assertFalse(output.arrived)
        self.assertEqual(output.state, NavigationLifecycle.ALIGNING)
        self.assertEqual(output.command.vx, 0.0)
        self.assertEqual(output.command.vy, 0.0)
        self.assertAlmostEqual(abs(output.command.omega), motion.rotation_max_rps)

        robot.heading_rad = -0.14
        output = controller.command(robot, goal, plan)
        self.assertEqual(output.state, NavigationLifecycle.ALIGNING)
        self.assertAlmostEqual(output.command.omega, 0.10)

        robot.heading_rad = -0.099
        output = controller.command(robot, goal, plan)
        self.assertTrue(output.arrived)
        self.assertEqual(output.state, NavigationLifecycle.READY_TO_DOCK)
        self.assertFalse(output.command.moving)

    def test_environment_uses_two_value_action_and_control_substeps(self) -> None:
        """Apply one local-target action across its faster control substeps."""

        environment = SingleRobotinoGymEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=LearningEnvironmentSettings(
                time_step_s=self.profile.policy_period_s,
                max_steps=2,
                require_final_heading=True,
                policy_profile=self.profile,
            )
        )
        try:
            observation, _ = environment.reset(seed=7)
            self.assertEqual(environment.action_space.shape, (2,))
            self.assertIn("lidar_closeness", observation)
            _, _, _, _, info = environment.step(np.zeros(2, dtype=np.float32))
            self.assertAlmostEqual(info["elapsedS"], self.profile.policy_period_s)
            self.assertIn("appliedCommandVariation", info)
            self.assertIn("meanAppliedCommandVariationPerSecond", info)
        finally:
            environment.close()

    def test_motion_limits_can_exceed_observation_normalization(self) -> None:
        """Allow calibrated actuator speed above the saved input normalization."""

        motion = replace(
            TEST_MOTION,
            forward_max_mps=0.6,
            sideways_max_mps=0.6,
        )
        environment = SingleRobotinoGymEnv(
            fleet_settings=fixed_project_settings(motion=motion),
            learning_settings=LearningEnvironmentSettings(
                time_step_s=self.profile.policy_period_s,
                max_steps=1,
                policy_profile=self.profile,
            ),
        )
        try:
            self.assertEqual(
                environment.parallel_env.core.world.motion.forward_max_mps,
                0.6,
            )
            self.assertEqual(self.profile.observation_forward_mps, 0.5)
        finally:
            environment.close()

    def test_collision_dominates_long_route_progress(self) -> None:
        """Make collision penalty outweigh progress on a long path."""

        result = navigation_reward(
            RewardSettings(collision=-120.0),
            previous_distance_m=10.0,
            distance_m=0.0,
            minimum_clearance_m=0.0,
            collision=True,
            success=False,
            action=(0.0, 0.0),
            previous_action=(0.0, 0.0),
            heading_error_rad=0.0,
        )
        self.assertLess(result.total, 0.0)

    def test_best_distance_progress_does_not_penalize_detours(self) -> None:
        """Avoid punishing a temporary detour under best-distance reward mode."""

        settings = RewardSettings(
            progress=4.0,
            progress_mode="best-distance",
            time=0.0,
            smoothness=0.0,
            control_effort=0.0,
            proximity=0.0,
            approach_heading=0.0,
        )
        detour = navigation_reward(
            settings,
            previous_distance_m=2.0,
            distance_m=2.5,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            heading_error_rad=0.0,
        )
        improvement = navigation_reward(
            settings,
            previous_distance_m=2.0,
            distance_m=1.5,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            heading_error_rad=0.0,
        )
        self.assertEqual(detour.components["progress"], 0.0)
        self.assertEqual(improvement.components["progress"], 2.0)

    def test_runway_reward_encourages_alignment_and_penalizes_lateral_motion(self) -> None:
        """Reward forward alignment and discourage sideways runway travel."""

        result = navigation_reward(
            RewardSettings(
                progress=0.0,
                proximity=0.0,
                time=0.0,
                smoothness=0.0,
                control_effort=0.0,
                lateral_motion=-0.5,
                heading_progress=2.0,
                approach_heading=0.0,
            ),
            previous_distance_m=3.0,
            distance_m=3.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            heading_error_rad=0.0,
            lateral_speed_mps=0.2,
            previous_navigation_heading_error_rad=1.0,
            navigation_heading_error_rad=0.5,
        )

        self.assertAlmostEqual(result.components["lateral_motion"], -0.1)
        self.assertAlmostEqual(result.components["heading_progress"], 1.0)
        self.assertAlmostEqual(result.total, 0.9)

    def test_runway_reward_continuously_exposes_heading_alignment(self) -> None:
        """Provide graded heading feedback before reaching a runway goal."""

        settings = RewardSettings(
            progress=0.0,
            proximity=0.0,
            time=0.0,
            smoothness=0.0,
            control_effort=0.0,
            heading_alignment=0.05,
            approach_heading=0.0,
        )
        aligned = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=4.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            heading_error_rad=0.0,
            navigation_heading_error_rad=0.0,
        )
        reversed_heading = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=4.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            heading_error_rad=0.0,
            navigation_heading_error_rad=math.pi,
        )

        self.assertAlmostEqual(aligned.total, 0.05)
        self.assertAlmostEqual(reversed_heading.total, -0.05)

    def test_runway_progress_requires_alignment_and_penalizes_reverse_motion(self) -> None:
        """Reward aligned runway progress while discouraging reverse motion."""

        settings = RewardSettings(
            progress=4.0,
            progress_mode="best-distance",
            heading_weighted_progress=True,
            proximity=0.0,
            time=0.0,
            smoothness=0.0,
            control_effort=0.0,
            reverse_motion=-0.3,
            approach_heading=0.0,
        )
        aligned = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=3.5,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(1.0, 0.0, 0.0),
            previous_action=(1.0, 0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=0.5,
            navigation_heading_error_rad=0.0,
        )
        reversed_heading = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=3.5,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(-1.0, 0.0, 0.0),
            previous_action=(-1.0, 0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=-0.5,
            navigation_heading_error_rad=math.pi,
        )

        self.assertAlmostEqual(aligned.components["progress"], 2.0)
        self.assertAlmostEqual(aligned.components["reverse_motion"], 0.0)
        self.assertAlmostEqual(reversed_heading.components["progress"], 0.0)
        self.assertAlmostEqual(
            reversed_heading.components["reverse_motion"],
            -0.15,
        )
        self.assertAlmostEqual(reversed_heading.total, -0.15)

    def test_runway_stages_rotation_before_forward_motion(self) -> None:
        """Favor initial heading correction before long forward translation."""

        settings = RewardSettings(
            progress=0.0,
            proximity=0.0,
            time=0.0,
            smoothness=0.0,
            control_effort=0.0,
            misaligned_translation=-0.4,
            aligned_forward_motion=0.5,
            approach_heading=0.0,
        )
        misaligned = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=4.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(1.0, 0.0, 0.0),
            previous_action=(1.0, 0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=0.5,
            navigation_heading_error_rad=math.pi / 2.0,
        )
        aligned = navigation_reward(
            settings,
            previous_distance_m=4.0,
            distance_m=4.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(1.0, 0.0, 0.0),
            previous_action=(1.0, 0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=0.5,
            navigation_heading_error_rad=0.0,
        )

        self.assertAlmostEqual(
            misaligned.components["misaligned_translation"],
            -0.2,
        )
        self.assertAlmostEqual(
            misaligned.components["aligned_forward_motion"],
            0.0,
        )
        self.assertAlmostEqual(
            aligned.components["misaligned_translation"],
            0.0,
        )
        self.assertAlmostEqual(
            aligned.components["aligned_forward_motion"],
            0.25,
        )

    def test_turning_translation_penalty_uses_actual_speed_and_turn_rate(self) -> None:
        """Compute turning-while-moving penalty from measured robot motion."""

        settings = RewardSettings(
            progress=0.0,
            proximity=0.0,
            time=0.0,
            smoothness=0.0,
            control_effort=0.0,
            turning_translation=-0.01,
            approach_heading=0.0,
        )

        fast_turn = navigation_reward(
            settings,
            previous_distance_m=2.0,
            distance_m=2.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0),
            previous_action=(0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=0.3,
            angular_speed_rps=0.45,
        )
        slow_or_gentle = navigation_reward(
            settings,
            previous_distance_m=2.0,
            distance_m=2.0,
            minimum_clearance_m=1.0,
            collision=False,
            success=False,
            action=(0.0, 0.0),
            previous_action=(0.0, 0.0),
            heading_error_rad=0.0,
            forward_speed_mps=0.1,
            angular_speed_rps=0.75,
        )

        self.assertAlmostEqual(
            fast_turn.components["turning_translation"],
            -0.03,
        )
        self.assertEqual(slow_or_gentle.components["turning_translation"], 0.0)

    def test_learned_yaw_environment_applies_timeout_penalty(self) -> None:
        """Apply the configured timeout penalty in learned-yaw episodes."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-yaw-temporal-v1",
            schema_version=5,
            action_mode="local-target-yaw",
            temporal_frames=4,
        )
        environment = SingleRobotinoGymEnv(
            fleet_settings=fixed_project_settings(),
            learning_settings=LearningEnvironmentSettings(
                time_step_s=profile.policy_period_s,
                max_steps=1,
                policy_profile=profile,
                reward=RewardSettings(timeout=-60.0),
            )
        )
        try:
            environment.reset(seed=71)
            _, reward, terminated, truncated, info = environment.step(
                np.zeros(3, dtype=np.float32)
            )
            self.assertEqual(environment.action_space.shape, (3,))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["rewardComponents"]["timeout"], -60.0)
            self.assertLess(reward, -59.0)
        finally:
            environment.close()

    def test_policy_profile_round_trip(self) -> None:
        """Reload a written model sidecar without changing its contract."""

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "policy.zip"
            write_policy_profile(model_path, self.profile)
            self.assertEqual(load_policy_profile(model_path), self.profile)

    def test_unseeded_resets_advance_a_reproducible_episode_sequence(self) -> None:
        """Advance episode seeds on repeated resets, reproducibly from a seed."""

        def signatures() -> list[tuple[object, ...]]:
            """Return ordered start/goal/heading signatures from one run.

            Returns:
                list[tuple[object, ...]]: Ordered start/goal/heading signatures from one run.
            """

            environment = SingleRobotinoGymEnv(
                fleet_settings=fixed_project_settings(),
                learning_settings=LearningEnvironmentSettings()
            )
            try:
                result: list[tuple[object, ...]] = []
                for seed in (123, None, None, None):
                    _, info = environment.reset(seed=seed)
                    robot = environment.parallel_env.core.robots[
                        environment.controlled_agent
                    ]
                    result.append(
                        (
                            info["episodeSeed"],
                            info["start"]["x"],
                            info["start"]["y"],
                            info["goal"]["locationId"],
                            robot.heading_rad,
                        )
                    )
                return result
            finally:
                environment.close()

        first = signatures()
        second = signatures()
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_vector_worker_seeds_start_distinct_episode_sequences(self) -> None:
        """Give separately seeded workers distinct initial episode states."""

        signatures: set[tuple[object, ...]] = set()
        for seed in range(500, 504):
            environment = SingleRobotinoGymEnv(
                fleet_settings=fixed_project_settings(),
                learning_settings=LearningEnvironmentSettings()
            )
            try:
                _, info = environment.reset(seed=seed)
                robot = environment.parallel_env.core.robots[
                    environment.controlled_agent
                ]
                signatures.add(
                    (
                        info["episodeSeed"],
                        info["start"]["x"],
                        info["start"]["y"],
                        info["goal"]["locationId"],
                        robot.heading_rad,
                    )
                )
            finally:
                environment.close()
        self.assertEqual(len(signatures), 4)


if __name__ == "__main__":
    unittest.main()
