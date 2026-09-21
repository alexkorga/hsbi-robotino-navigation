"""Fuse map-derived keep-out boundaries into a physical 2-D LiDAR scan."""

import math
from collections.abc import Iterable
from dataclasses import replace

from robotino_fleet.domain.models import LidarScanObservation, Point, RobotState
from robotino_fleet.simulation.physical_scene import PhysicalScene


def apply_virtual_lidar_constraints(
    robot: RobotState,
    scan: LidarScanObservation,
    scene: PhysicalScene,
    *,
    robots: Iterable[RobotState] = (),
) -> LidarScanObservation:
    """Overlay map boundaries and peer footprints on a physical scan.

    robot supplies the localized map pose, scan the physical ranges
    and LiDAR extrinsics, scene the keep-out geometry, and robots the
    other localized footprints. Return a copy with beams shortened only when
    a virtual hit is nearer than a valid real reading. An invalid robot pose
    returns the original scan. The augmented result is for policy perception;
    physical emergency stops use the unmodified scan.

    Args:
        robot: Robotino state used by this operation.
        scan: An invalid robot pose returns the original scan.
        scene: Physical factory geometry used for collision and ray casting.
        robots: Robotino states in the shared world.

    Returns:
        LidarScanObservation: Display/policy scan with virtual geometry overlaid.
    """

    if not robot.pose_valid:
        return scan
    cosine = math.cos(robot.heading_rad)
    sine = math.sin(robot.heading_rad)
    origin = Point(
        robot.x + cosine * scan.extrinsic_x_m - sine * scan.extrinsic_y_m,
        robot.y + sine * scan.extrinsic_x_m + cosine * scan.extrinsic_y_m,
    )
    minimum = max(scan.range_min, 1e-6)
    ranges = list(scan.ranges)
    intensities = list(scan.intensities)
    for index, raw_distance in enumerate(ranges):
        beam_angle = scan.angle_min + index * scan.angle_increment
        direction = robot.heading_rad + scan.extrinsic_yaw_rad + beam_angle
        virtual = scene.virtual_ray_cast(
            origin,
            direction,
            scan.range_max,
        )
        peer = scene.robot_ray_cast(
            origin,
            direction,
            scan.range_max,
            robots=robots,
            exclude_robot_id=robot.id,
        )
        constraint = virtual
        if peer is not None and (
            constraint is None or peer.distance_m < constraint.distance_m
        ):
            constraint = peer
        if constraint is None:
            continue
        real_is_valid = (
            math.isfinite(raw_distance)
            and minimum <= raw_distance <= scan.range_max
        )
        if real_is_valid and raw_distance <= constraint.distance_m:
            continue
        ranges[index] = min(scan.range_max, max(minimum, constraint.distance_m))
        if index < len(intensities):
            intensities[index] = 0.0
    return replace(scan, ranges=tuple(ranges), intensities=tuple(intensities))
