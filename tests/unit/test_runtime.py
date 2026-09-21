"""Exercise the fleet runtime, its settings, and simulation integration."""

from dataclasses import replace
import unittest

import numpy as np

from robotino_fleet.bootstrap import build_fleet_runtime
from robotino_fleet.config import load_settings
from robotino_fleet.domain.models import NavigationMetrics, Point, VelocityCommand
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.actions import ActionScaler
from robotino_fleet.learning.gym_env import SingleRobotinoGymEnv
from robotino_fleet.navigation.motion import MotionLimiter


class RuntimeTests(unittest.TestCase):
    def test_path_efficiency_uses_straight_distance_over_driven_distance(self) -> None:
        """Compute efficiency and travel time from one completed detouring trip."""

        metric = NavigationMetrics()
        metric.start(
            Point(0.0, 0.0),
            Point(3.0, 4.0),
            tracking_source="odometry",
            tracking_point=Point(0.0, 0.0),
            timestamp_s=0.0,
            assignment_started_at_s=10.0,
        )
        metric.observe(
            Point(3.0, 0.0),
            tracking_source="odometry",
            timestamp_s=2.0,
            maximum_speed_mps=2.0,
        )
        metric.observe(
            Point(3.0, 4.0),
            tracking_source="odometry",
            timestamp_s=4.0,
            maximum_speed_mps=2.0,
        )
        metric.complete(timestamp_s=17.25)

        self.assertAlmostEqual(metric.straight_path_length_m or 0.0, 5.0)
        self.assertAlmostEqual(metric.driven_path_length_m, 7.0)
        self.assertAlmostEqual(metric.path_efficiency or 0.0, 5.0 / 7.0)
        self.assertAlmostEqual(metric.travel_time_s or 0.0, 7.25)

    def test_new_assignment_resets_path_efficiency(self) -> None:
        """Clear old trip metrics when a new destination is assigned."""

        metric = NavigationMetrics(
            straight_path_length_m=2.0,
            driven_path_length_m=3.0,
            path_efficiency=2.0 / 3.0,
        )
        metric.start(
            Point(1.0, 1.0),
            Point(2.0, 1.0),
            tracking_source="simulation",
            tracking_point=Point(1.0, 1.0),
            timestamp_s=5.0,
            assignment_started_at_s=5.0,
        )

        self.assertEqual(metric.straight_path_length_m, 1.0)
        self.assertEqual(metric.driven_path_length_m, 0.0)
        self.assertIsNone(metric.path_efficiency)
        self.assertIsNone(metric.travel_time_s)

    def test_motion_is_limited_once_with_separate_acceleration(self) -> None:
        """Apply configured linear and angular acceleration exactly once."""

        settings = load_settings().motion
        limiter = MotionLimiter(settings)
        command = limiter.apply("r", VelocityCommand(10, 0, 10), 0.1)
        self.assertAlmostEqual(command.vx, settings.forward_acceleration_mps2 * 0.1)
        self.assertAlmostEqual(command.omega, settings.rotation_acceleration_rps2 * 0.1)

    def test_sideways_motion_uses_its_own_constants(self) -> None:
        """Use independent lateral limits in command limiting and action scaling."""

        settings = replace(
            load_settings().motion,
            sideways_max_mps=0.2,
            sideways_acceleration_mps2=0.1,
            sideways_deadband_mps=0.0,
        )
        limiter = MotionLimiter(settings)
        command = limiter.apply("r", VelocityCommand(0, 10, 0), 0.1)
        self.assertAlmostEqual(command.vy, 0.01)

        scaler = ActionScaler(0.5, 0.2, 1.0)
        scaled = scaler.scale((1.0, 1.0, 0.5))
        self.assertAlmostEqual(scaled.vx, 0.5 / np.sqrt(2))
        self.assertAlmostEqual(scaled.vy, 0.2 / np.sqrt(2))
        self.assertAlmostEqual(scaled.omega, 0.5)

    def test_dashboard_simulation_and_learning_use_same_world_type(self) -> None:
        """Share one simulation implementation between dashboard and Gymnasium."""

        runtime = build_fleet_runtime("simulation")
        env = SingleRobotinoGymEnv(
            learning_settings=LearningEnvironmentSettings(max_steps=2)
        )
        self.assertEqual(type(runtime.world), type(env.parallel_env.core.world))
        observation, _ = env.reset(seed=5)
        self.assertIn("lidar", observation)
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertIn("distanceToGoalM", info)
        env.close()

    def test_goal_is_direct_and_stop_clears_it(self) -> None:
        """Expose assignment metrics and remove an order on operator stop."""

        runtime = build_fleet_runtime("simulation")
        robot_id = next(iter(runtime.robots))
        runtime.assign_goal(robot_id, "13")
        self.assertEqual(runtime.robots[robot_id].goal_location_id, "13")
        robot_payload = next(
            robot
            for robot in runtime.world_dict()["robots"]
            if robot["id"] == robot_id
        )
        self.assertIsNotNone(robot_payload["straightPathLengthM"])
        self.assertEqual(robot_payload["drivenPathLengthM"], 0.0)
        self.assertIsNone(robot_payload["pathEfficiency"])
        self.assertIsNone(robot_payload["travelTimeS"])
        runtime.stop_robot(robot_id)
        self.assertIsNone(runtime.robots[robot_id].goal_location_id)

    def test_physical_runtime_stages_unlocalized_robotinos_outside_factory(self) -> None:
        """Do not display an unlocalized physical Robotino at a fake map start."""

        runtime = build_fleet_runtime("shadow")
        try:
            self.assertTrue(runtime.robots)
            for robot in runtime.robots.values():
                self.assertFalse(robot.pose_valid)
                self.assertFalse(runtime.navigation_scene.is_valid_center(robot.point))
        finally:
            runtime.shutdown()

    def test_live_lidar_peer_filter_requires_fresh_valid_pose(self) -> None:
        """Synthesize only fresh, localized peers into the policy LiDAR view."""

        runtime = build_fleet_runtime("shadow")
        try:
            subject_id, fresh_id, stale_id = tuple(runtime.robots)[:3]
            fresh = runtime.robots[fresh_id]
            fresh.pose_valid = True
            fresh.pose_received_at_s = 9.0
            stale = runtime.robots[stale_id]
            stale.pose_valid = True
            stale.pose_received_at_s = 1.0

            peers = runtime._fresh_localized_peers(subject_id, now_s=10.0)

            self.assertEqual(tuple(peer.id for peer in peers), (fresh_id,))
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
