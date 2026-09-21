"""Verify simulated LiDAR and proximity readings against scene geometry."""

import unittest
from dataclasses import replace

from robotino_fleet.config import ROBOT_DIAMETER_M, ROBOT_RADIUS_M
from robotino_fleet.domain.models import Point, RobotState
from robotino_fleet.maps.models import EnvironmentGeometry, EnvironmentObstacle
from robotino_fleet.simulation.physical_scene import PhysicalScene
from robotino_fleet.simulation.sensors import (
    DistanceSensorSimulationSettings,
    LidarSimulationSettings,
    SensorSimulationSettings,
    SensorSimulator,
)


def robot(identifier: str, x: float = 0.0) -> RobotState:
    """Return a test Robotino identifier centered at (x, 0).

    Args:
        identifier: Robotino identifier used in the test state.
        x: Initial X position of the test Robotino, in meters.

    Returns:
        RobotState: A test Robotino identifier centered at (x, 0).
    """

    return RobotState(identifier, "#fff", x, 0.0, 0.0)


class SensorSimulationTests(unittest.TestCase):
    def test_transport_overhang_expands_collision_and_robotino_lidar_radius(self) -> None:
        """Use the configured overhang footprint in peer LiDAR intersections."""

        self.assertAlmostEqual(ROBOT_RADIUS_M, 0.25)
        self.assertAlmostEqual(ROBOT_DIAMETER_M, 0.50)
        lidar = replace(
            LidarSimulationSettings(),
            beam_count=1,
            angle_min_rad=0.0,
            angle_max_rad=0.0,
            angle_increment_rad=0.01,
            range_noise_std_m=0.0,
            range_quantization_m=0.0,
        )
        subject = robot("r1")
        other = robot("r2", x=1.0)
        simulator = SensorSimulator(
            SensorSimulationSettings(lidar=lidar),
            PhysicalScene(None, robot_radius_m=ROBOT_RADIUS_M),
        )

        simulator.advance(0.0, 0.0, {subject.id: subject, other.id: other})

        assert subject.lidar
        self.assertAlmostEqual(subject.lidar.ranges[0], 1.0 - ROBOT_RADIUS_M)

    def test_lidar_and_distance_sensor_match_physical_geometry(self) -> None:
        """Ray-cast a known wall from the LiDAR and perimeter sensor origins."""

        geometry = EnvironmentGeometry(
            "test",
            (
                Point(-2, -2), Point(2, -2), Point(2, 2), Point(-2, 2),
            ),
            (
                EnvironmentObstacle(
                    "wall", "wall", "WALL",
                    (Point(0.375, -0.1), Point(0.385, -0.1), Point(0.385, 0.1), Point(0.375, 0.1)),
                ),
            ),
        )
        lidar = replace(
            LidarSimulationSettings(),
            beam_count=1,
            angle_min_rad=0.0,
            angle_max_rad=0.0,
            angle_increment_rad=0.01,
            range_noise_std_m=0.0,
            range_quantization_m=0.0,
        )
        distance = replace(
            DistanceSensorSimulationSettings(),
            rays_per_sensor=1,
            range_noise_std_m=0.0,
            range_quantization_m=0.0,
        )
        subject = robot("r1")
        simulator = SensorSimulator(
            SensorSimulationSettings(7, lidar, distance),
            PhysicalScene(geometry, robot_radius_m=0.225),
        )
        simulator.advance(0.0, 0.0, {subject.id: subject})
        assert subject.lidar
        self.assertAlmostEqual(subject.lidar.ranges[0], 0.375)
        self.assertAlmostEqual(subject.sensors[0], 0.150)

    def test_default_sensor_shapes_match_robotino_api(self) -> None:
        """Emit the hardware-compatible 441 scan beams and nine distances."""

        subject = robot("r1")
        simulator = SensorSimulator(
            SensorSimulationSettings(),
            PhysicalScene(None, robot_radius_m=0.225),
        )
        simulator.advance(0.0, 0.0, {subject.id: subject})
        assert subject.lidar
        self.assertEqual(len(subject.lidar.ranges), 441)
        self.assertEqual(len(subject.sensors), 9)


if __name__ == "__main__":
    unittest.main()
