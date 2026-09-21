"""Verify virtual LiDAR augmentation and physical-scan separation."""

import unittest

from robotino_fleet.config import ROBOT_RADIUS_M
from robotino_fleet.domain.models import LidarScanObservation, Point, RobotState
from robotino_fleet.maps.models import EnvironmentGeometry, EnvironmentObstacle
from robotino_fleet.navigation.virtual_lidar import apply_virtual_lidar_constraints
from robotino_fleet.simulation.physical_scene import PhysicalScene


class VirtualLidarTests(unittest.TestCase):
    def setUp(self) -> None:
        """Build a test scene with a boundary and enlarged machine keep-out."""

        geometry = EnvironmentGeometry(
            source="test",
            movement_area=(
                Point(-2.0, -2.0),
                Point(2.0, -2.0),
                Point(2.0, 2.0),
                Point(-2.0, 2.0),
            ),
            obstacles=(
                EnvironmentObstacle(
                    "overhead-machine",
                    "Overhead machine",
                    "MACHINE",
                    (
                        Point(1.0, -0.5),
                        Point(1.5, -0.5),
                        Point(1.5, 0.5),
                        Point(1.0, 0.5),
                    ),
                    navigation_margin_m=0.3,
                ),
            ),
        )
        self.scene = PhysicalScene(geometry, robot_radius_m=0.225)
        self.robot = RobotState("r", "#fff", 0.0, 0.0, 0.0)

    @staticmethod
    def scan(distance: float) -> LidarScanObservation:
        """Return a single-beam physical scan with distance meters.

        Args:
            distance: Distance of the synthetic LiDAR hit, in meters.

        Returns:
            LidarScanObservation: A single-beam physical scan with distance meters.
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
            ranges=(distance,),
            intensities=(1.0,),
        )

    def test_no_return_is_cut_at_enlarged_keep_out(self) -> None:
        """Replace an absent physical return with the nearer virtual margin."""

        result = apply_virtual_lidar_constraints(
            self.robot, self.scan(0.0), self.scene
        )
        self.assertAlmostEqual(result.ranges[0], 0.7)
        self.assertEqual(result.intensities[0], 0.0)

    def test_nearer_real_return_is_never_hidden(self) -> None:
        """Keep a valid physical hit when it precedes the virtual obstacle."""

        result = apply_virtual_lidar_constraints(
            self.robot, self.scan(0.4), self.scene
        )
        self.assertAlmostEqual(result.ranges[0], 0.4)
        self.assertEqual(result.intensities[0], 1.0)

    def test_localized_peer_is_inflated_to_collision_radius(self) -> None:
        """Represent a localized peer's full collision footprint in virtual rays."""

        scene = PhysicalScene(None, robot_radius_m=ROBOT_RADIUS_M)
        peer = RobotState("peer", "#fff", 1.0, 0.0, 0.0)

        result = apply_virtual_lidar_constraints(
            self.robot,
            self.scan(0.0),
            scene,
            robots=(peer,),
        )

        self.assertAlmostEqual(result.ranges[0], 1.0 - ROBOT_RADIUS_M)
        self.assertEqual(result.intensities[0], 0.0)

    def test_peer_without_valid_pose_does_not_create_ghost_return(self) -> None:
        """Exclude unlocalized peers from synthesized LiDAR returns."""

        scene = PhysicalScene(None, robot_radius_m=ROBOT_RADIUS_M)
        peer = RobotState(
            "peer",
            "#fff",
            1.0,
            0.0,
            0.0,
            pose_valid=False,
        )

        result = apply_virtual_lidar_constraints(
            self.robot,
            self.scan(0.0),
            scene,
            robots=(peer,),
        )

        self.assertEqual(result.ranges[0], 0.0)

    def test_access_boundary_becomes_virtual_wall(self) -> None:
        """Make the movement-area edge visible to navigation ray casting."""

        result = self.scene.virtual_ray_cast(Point(0.0, 1.0), 1.57079632679, 20.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.distance_m, 1.0)
        self.assertEqual(result.category, "VIRTUAL_BOUNDARY")

    def test_navigation_margin_is_also_collision_geometry(self) -> None:
        """Enforce virtual keep-out margins in simulated collision detection."""

        collision = self.scene.collision_at(Point(0.8, 0.0), radius_m=0.05)
        self.assertIsNotNone(collision)
        assert collision is not None
        self.assertEqual(collision.category, "VIRTUAL_KEEP_OUT")

    def test_navigation_only_fixture_is_virtual_but_collision_enforced(self) -> None:
        """Hide navigation-only geometry from physical rays while blocking motion."""

        geometry = EnvironmentGeometry(
            source="test",
            movement_area=(
                Point(-2.0, -2.0),
                Point(2.0, -2.0),
                Point(2.0, 2.0),
                Point(-2.0, 2.0),
            ),
            obstacles=(
                EnvironmentObstacle(
                    "upper-corner",
                    "Upper corner",
                    "MACHINE",
                    (
                        Point(1.0, -0.5),
                        Point(1.5, -0.5),
                        Point(1.5, 0.5),
                        Point(1.0, 0.5),
                    ),
                    navigation_only=True,
                ),
            ),
        )
        scene = PhysicalScene(geometry, robot_radius_m=0.225)
        self.assertIsNone(scene.ray_cast(Point(0.0, 0.0), 0.0, 20.0))
        virtual = scene.virtual_ray_cast(Point(0.0, 0.0), 0.0, 20.0)
        self.assertIsNotNone(virtual)
        assert virtual is not None
        self.assertAlmostEqual(virtual.distance_m, 1.0)
        self.assertIsNotNone(scene.collision_at(Point(0.9, 0.0)))


if __name__ == "__main__":
    unittest.main()
