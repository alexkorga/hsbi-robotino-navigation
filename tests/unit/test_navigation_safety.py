"""Exercise physical-sensor safety decisions and navigation stop conditions."""

import unittest

from robotino_fleet.domain.models import (
    LidarScanObservation,
    Point,
    RobotState,
    VelocityCommand,
)
from robotino_fleet.navigation.goals import NavigationGoal
from robotino_fleet.navigation.safety import NavigationSafetySupervisor


def scan(distance_m: float) -> LidarScanObservation:
    """Return a one-beam test scan with a hit at distance_m.

    Args:
        distance_m: Distance in meters.

    Returns:
        LidarScanObservation: A one-beam test scan with a hit at distance_m.
    """

    return LidarScanObservation(
        sequence=1,
        source_timestamp_s=0.0,
        timestamp_s=0.0,
        angle_min=0.0,
        angle_max=0.0,
        angle_increment=1.0,
        range_min=0.05,
        range_max=20.0,
        ranges=(distance_m,),
    )


class NavigationSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        """Prepare a clear route, proposed command, and safety supervisor."""

        self.goal = NavigationGoal("1", Point(1.0, 0.0), None, 0.1, 0.1)
        self.command = VelocityCommand(vx=0.2)
        self.supervisor = NavigationSafetySupervisor()

    @staticmethod
    def robot() -> RobotState:
        """Return a localized test Robotino with fresh proximity telemetry.

        Returns:
            RobotState: A localized test Robotino with fresh proximity telemetry.
        """

        return RobotState(
            "172.21.20.90",
            "#fff",
            0.0,
            0.0,
            0.0,
            sensors_updated_at_s=0.0,
        )

    def test_physical_stop_ignores_virtual_lidar_constraint(self) -> None:
        """Allow live motion when only a synthetic map ray reports danger."""

        robot = self.robot()
        robot.measured_lidar = scan(1.0)
        robot.lidar = scan(0.24)

        decision = self.supervisor.evaluate(
            robot,
            self.goal,
            self.command,
            now_s=0.0,
            physical=True,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.command, self.command)

    def test_physical_stop_still_uses_real_lidar_measurement(self) -> None:
        """Stop live motion for a critical actual LiDAR reading."""

        robot = self.robot()
        robot.measured_lidar = scan(0.24)
        robot.lidar = scan(1.0)

        decision = self.supervisor.evaluate(
            robot,
            self.goal,
            self.command,
            now_s=0.0,
            physical=True,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "critical LiDAR clearance")

    def test_simulation_uses_its_simulated_lidar(self) -> None:
        """Stop simulated motion from the simulator's scan data."""

        robot = self.robot()
        robot.lidar = scan(0.24)

        decision = self.supervisor.evaluate(
            robot,
            self.goal,
            self.command,
            now_s=0.0,
            physical=False,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "critical LiDAR clearance")


if __name__ == "__main__":
    unittest.main()
