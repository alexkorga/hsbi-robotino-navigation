"""Deterministic simulation of the Robotino LiDAR and nine distance sensors."""

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from robotino_fleet.config import LidarExtrinsic
from robotino_fleet.domain.models import LidarScanObservation, Point, RobotState
from robotino_fleet.simulation.physical_scene import PhysicalScene


_SCHEDULE_EPSILON = 1e-9


@dataclass(frozen=True)
class LidarSimulationSettings:
    frequency_hz: float = 5.0
    beam_count: int = 441
    angle_min_rad: float = -1.9198628664016724
    angle_max_rad: float = 1.9285876750946045
    angle_increment_rad: float = 0.00872664526104927
    range_min_m: float = 0.0
    ray_start_m: float = 0.04
    range_max_m: float = 20.0
    no_return_value_m: float = 0.0
    range_noise_std_m: float = 0.0
    angle_noise_std_rad: float = 0.0
    range_quantization_m: float = 0.01
    dropout_probability: float = 0.0
    latency_s: float = 0.0
    intensity_hit_probability: float = 0.05
    extrinsic: LidarExtrinsic = LidarExtrinsic()


@dataclass(frozen=True)
class DistanceSensorSimulationSettings:
    frequency_hz: float = 5.0
    count: int = 9
    angular_spacing_deg: float = 40.0
    clockwise: bool = True
    sensor_0_angle_deg: float = 0.0
    mounting_radius_m: float = 0.225
    detection_min_m: float = 0.04
    detection_max_m: float = 0.30
    clear_value_m: float = 0.410
    beam_half_angle_deg: float = 3.0
    rays_per_sensor: int = 3
    range_noise_std_m: float = 0.0
    range_quantization_m: float = 0.001
    dropout_probability: float = 0.0
    latency_s: float = 0.0


@dataclass(frozen=True)
class SensorSimulationSettings:
    random_seed: int = 42
    lidar: LidarSimulationSettings = LidarSimulationSettings()
    distance_sensors: DistanceSensorSimulationSettings = (
        DistanceSensorSimulationSettings()
    )


def _quantize(value: float, step: float) -> float:
    """Return value rounded to step units, or unchanged if disabled.

    Args:
        value: Sensor reading to round to the configured resolution.
        step: Sensor quantization interval; zero disables rounding.

    Returns:
        float: Value rounded to step units, or unchanged if disabled.
    """

    if step <= 0:
        return value
    return round(value / step) * step


@dataclass(frozen=True)
class _PendingLidar:
    robot_id: str
    deliver_at_s: float
    measured_at_s: float
    sequence: int
    ranges: tuple[float, ...]
    intensities: tuple[float, ...]


@dataclass(frozen=True)
class _PendingDistanceSensors:
    robot_id: str
    deliver_at_s: float
    measured_at_s: float
    values: tuple[float, ...]


class SensorSimulator:
    """Publishes simulated sensors with independent rates and optional latency.

    The simulation clock controls measurement schedules, making repeated runs
    deterministic.  The supplied monotonic receive time is retained for the
    same age semantics used by live HTTP observations.
    """

    def __init__(
        self,
        settings: SensorSimulationSettings,
        scene: PhysicalScene,
    ) -> None:
        """Use scene for ray casting and settings for timing/noise.

        Args:
            settings: Configuration settings for this component.
            scene: Physical factory geometry used for collision and ray casting.
        """

        self.settings = settings
        self.scene = scene
        self._random = random.Random(settings.random_seed)
        self._next_lidar_s: dict[str, float] = {}
        self._next_distance_s: dict[str, float] = {}
        self._lidar_sequences: dict[str, int] = {}
        self._pending_lidar: list[_PendingLidar] = []
        self._pending_distance: list[_PendingDistanceSensors] = []

    def reset(self) -> None:
        """Reset seeded noise, sensor schedules, sequences, and pending samples."""

        self._random.seed(self.settings.random_seed)
        self._next_lidar_s.clear()
        self._next_distance_s.clear()
        self._lidar_sequences.clear()
        self._pending_lidar.clear()
        self._pending_distance.clear()

    @staticmethod
    def _body_point(robot: RobotState, x_m: float, y_m: float) -> Point:
        """Return body-frame offset (x_m, y_m) from robot in map space.

        Args:
            robot: Robotino state used by this operation.
            x_m: X coordinate or offset in meters.
            y_m: Y coordinate or offset in meters.

        Returns:
            Point: Body-frame offset (x_m, y_m) from robot in map space.
        """

        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        return Point(
            robot.x + cosine * x_m - sine * y_m,
            robot.y + sine * x_m + cosine * y_m,
        )

    def _lidar_measurement(
        self,
        robot: RobotState,
        robots: tuple[RobotState, ...],
        measured_at_s: float,
    ) -> _PendingLidar:
        """Sample one LiDAR scan for robot at measured_at_s.

        robots supplies peer footprints for ray casting. Return a pending
        sample with configured beam geometry, noise, dropout, and delivery time;
        the scan sequence advances even when delivery is delayed.

        Args:
            robot: Robotino state used by this operation.
            robots: Robotino states in the shared world.
            measured_at_s: Time when the sensor measurement was taken, in seconds.

        Returns:
            _PendingLidar: Simulated LiDAR sample before latency delivery.
        """

        settings = self.settings.lidar
        origin = self._body_point(
            robot,
            settings.extrinsic.x_m,
            settings.extrinsic.y_m,
        )
        ranges: list[float] = []
        intensities: list[float] = []
        for index in range(settings.beam_count):
            angle_noise = (
                self._random.gauss(0.0, settings.angle_noise_std_rad)
                if settings.angle_noise_std_rad > 0
                else 0.0
            )
            beam_angle = settings.angle_min_rad + index * settings.angle_increment_rad
            world_angle = (
                robot.heading_rad
                + settings.extrinsic.yaw_rad
                + beam_angle
                + angle_noise
            )
            hit = self.scene.navigation_ray_cast(
                origin,
                world_angle,
                settings.range_max_m,
                robots=robots,
                exclude_robot_id=robot.id,
            )
            dropped = self._random.random() < settings.dropout_probability
            if (
                hit is None
                or dropped
                or hit.distance_m < settings.ray_start_m
                or hit.distance_m < settings.range_min_m
            ):
                ranges.append(settings.no_return_value_m)
                intensities.append(0.0)
                continue
            value = hit.distance_m
            if settings.range_noise_std_m > 0:
                value += self._random.gauss(0.0, settings.range_noise_std_m)
            value = _quantize(value, settings.range_quantization_m)
            if value < max(settings.range_min_m, settings.ray_start_m):
                value = max(settings.range_min_m, settings.ray_start_m)
            if value > settings.range_max_m:
                ranges.append(settings.no_return_value_m)
                intensities.append(0.0)
                continue
            ranges.append(value)
            intensities.append(
                1.0
                if self._random.random() < settings.intensity_hit_probability
                else 0.0
            )
        sequence = self._lidar_sequences.get(robot.id, 0) + 1
        self._lidar_sequences[robot.id] = sequence
        return _PendingLidar(
            robot_id=robot.id,
            deliver_at_s=measured_at_s + settings.latency_s,
            measured_at_s=measured_at_s,
            sequence=sequence,
            ranges=tuple(ranges),
            intensities=tuple(intensities),
        )

    @staticmethod
    def _fan_offsets(settings: DistanceSensorSimulationSettings) -> tuple[float, ...]:
        """Return ray offsets covering the fan in distance-sensor settings.

        Args:
            settings: Configuration settings for this component.

        Returns:
            tuple[float, ...]: Ray offsets covering the fan in distance-sensor settings.
        """

        if settings.rays_per_sensor == 1 or settings.beam_half_angle_deg <= 0:
            return (0.0,)
        width = 2.0 * settings.beam_half_angle_deg
        return tuple(
            math.radians(
                -settings.beam_half_angle_deg
                + width * index / (settings.rays_per_sensor - 1)
            )
            for index in range(settings.rays_per_sensor)
        )

    def _distance_measurement(
        self,
        robot: RobotState,
        robots: tuple[RobotState, ...],
        measured_at_s: float,
    ) -> _PendingDistanceSensors:
        """Sample nine body-mounted distance sensors for robot.

        robots supplies peer footprints; measured_at_s timestamps the
        sample. Return pending readings, using the clear value when no object
        lies in a sensor's detection range.

        Args:
            robot: Robotino state used by this operation.
            robots: Robotino states in the shared world.
            measured_at_s: Time when the sensor measurement was taken, in seconds.

        Returns:
            _PendingDistanceSensors: Simulated proximity-sensor sample before latency delivery.
        """

        settings = self.settings.distance_sensors
        direction_sign = -1.0 if settings.clockwise else 1.0
        fan_offsets = self._fan_offsets(settings)
        values: list[float] = []
        for index in range(settings.count):
            body_angle = math.radians(
                settings.sensor_0_angle_deg
                + direction_sign * index * settings.angular_spacing_deg
            )
            world_center_angle = robot.heading_rad + body_angle
            origin = Point(
                robot.x + math.cos(world_center_angle) * settings.mounting_radius_m,
                robot.y + math.sin(world_center_angle) * settings.mounting_radius_m,
            )
            nearest: float | None = None
            for offset in fan_offsets:
                hit = self.scene.ray_cast(
                    origin,
                    world_center_angle + offset,
                    settings.detection_max_m,
                    robots=robots,
                    exclude_robot_id=robot.id,
                )
                if hit is not None and (nearest is None or hit.distance_m < nearest):
                    nearest = hit.distance_m
            if (
                nearest is None
                or nearest > settings.detection_max_m
                or self._random.random() < settings.dropout_probability
            ):
                values.append(settings.clear_value_m)
                continue
            value = max(settings.detection_min_m, nearest)
            if settings.range_noise_std_m > 0:
                value += self._random.gauss(0.0, settings.range_noise_std_m)
            value = _quantize(value, settings.range_quantization_m)
            values.append(
                min(
                    settings.detection_max_m,
                    max(settings.detection_min_m, value),
                )
            )
        return _PendingDistanceSensors(
            robot_id=robot.id,
            deliver_at_s=measured_at_s + settings.latency_s,
            measured_at_s=measured_at_s,
            values=tuple(values),
        )

    def _publish_due(
        self,
        sim_time_s: float,
        received_at_s: float,
        robots: dict[str, RobotState],
    ) -> bool:
        """Deliver latency-queued samples due by sim_time_s.

        received_at_s becomes each observation's age clock and robots
        receives the published data. Return whether any robot changed.

        Args:
            sim_time_s: Current simulation time in seconds.
            received_at_s: Time when telemetry was received, in seconds.
            robots: Robotino states in the shared world.

        Returns:
            bool: Whether any queued sensor sample was delivered.
        """

        changed = False
        remaining_lidar: list[_PendingLidar] = []
        for pending in self._pending_lidar:
            if pending.deliver_at_s > sim_time_s + _SCHEDULE_EPSILON:
                remaining_lidar.append(pending)
                continue
            robot = robots.get(pending.robot_id)
            if robot is None:
                continue
            settings = self.settings.lidar
            scan = LidarScanObservation(
                sequence=pending.sequence,
                source_timestamp_s=pending.measured_at_s,
                timestamp_s=received_at_s,
                angle_min=settings.angle_min_rad,
                angle_max=settings.angle_max_rad,
                angle_increment=settings.angle_increment_rad,
                range_min=settings.range_min_m,
                range_max=settings.range_max_m,
                ranges=pending.ranges,
                intensities=pending.intensities,
                extrinsic_x_m=settings.extrinsic.x_m,
                extrinsic_y_m=settings.extrinsic.y_m,
                extrinsic_yaw_rad=settings.extrinsic.yaw_rad,
            )
            robot.measured_lidar = scan
            robot.lidar = scan
            changed = True
        self._pending_lidar = remaining_lidar

        remaining_distance: list[_PendingDistanceSensors] = []
        for pending in self._pending_distance:
            if pending.deliver_at_s > sim_time_s + _SCHEDULE_EPSILON:
                remaining_distance.append(pending)
                continue
            robot = robots.get(pending.robot_id)
            if robot is None:
                continue
            robot.sensors = list(pending.values)
            robot.sensors_updated_at_s = received_at_s
            changed = True
        self._pending_distance = remaining_distance
        return changed

    def advance(
        self,
        sim_time_s: float,
        received_at_s: float,
        robots: dict[str, RobotState],
        *,
        obstacles: Iterable[RobotState] = (),
    ) -> bool:
        """Advance LiDAR and proximity schedules through sim_time_s.

        received_at_s is the observation receipt clock; robots are
        both sensing agents and output holders; obstacles adds temporary
        ray-cast footprints. Generate all due samples, publish completed
        latency queues, and return whether telemetry changed.

        Args:
            sim_time_s: Current simulation time in seconds.
            received_at_s: Time when telemetry was received, in seconds.
            robots: Robotino states in the shared world.
            obstacles: Advance LiDAR and proximity schedules through sim_time_s. received_at_s
                is the observation receipt clock; robots are both sensing agents and output
                holders; obstacles adds temporary ray-cast footprints.

        Returns:
            bool: Whether advancing the sensor schedules published new data.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if sim_time_s < 0:
            raise ValueError("sim_time_s cannot be negative")
        robot_ids = set(robots)
        for schedule in (
            self._next_lidar_s,
            self._next_distance_s,
            self._lidar_sequences,
        ):
            for robot_id in tuple(schedule):
                if robot_id not in robot_ids:
                    del schedule[robot_id]

        robot_snapshot = tuple(robots[robot_id] for robot_id in sorted(robots))
        ray_snapshot = (*robot_snapshot, *tuple(obstacles))
        lidar_period = 1.0 / self.settings.lidar.frequency_hz
        distance_period = 1.0 / self.settings.distance_sensors.frequency_hz
        for robot in robot_snapshot:
            next_lidar = self._next_lidar_s.setdefault(robot.id, 0.0)
            while next_lidar <= sim_time_s + _SCHEDULE_EPSILON:
                self._pending_lidar.append(
                    self._lidar_measurement(robot, ray_snapshot, next_lidar)
                )
                next_lidar += lidar_period
            self._next_lidar_s[robot.id] = next_lidar

            next_distance = self._next_distance_s.setdefault(robot.id, 0.0)
            while next_distance <= sim_time_s + _SCHEDULE_EPSILON:
                self._pending_distance.append(
                    self._distance_measurement(robot, ray_snapshot, next_distance)
                )
                next_distance += distance_period
            self._next_distance_s[robot.id] = next_distance
        return self._publish_due(sim_time_s, received_at_s, robots)
