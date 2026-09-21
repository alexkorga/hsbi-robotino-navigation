"""Small data objects shared by runtime, simulation, learning, and the API."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RobotStatus(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    WAITING = "WAITING"
    ARRIVED = "ARRIVED"
    READY_TO_DOCK = "READY_TO_DOCK"
    STOPPED = "STOPPED"
    OFFLINE = "OFFLINE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Return Euclidean distance in meters from this point to other.

        Args:
            other: Other map-frame point used for the distance calculation.

        Returns:
            float: Euclidean distance in meters from this point to other.
        """

        return math.hypot(other.x - self.x, other.y - self.y)

    def to_dict(self) -> dict[str, float]:
        """Return map-frame x and y for API serialization.

        Returns:
            dict[str, float]: Map-frame x and y for API serialization.
        """

        return {"x": self.x, "y": self.y}


@dataclass
class NavigationMetrics:
    """Distance and duration measurements for one navigation assignment."""

    straight_path_length_m: float | None = None
    driven_path_length_m: float = 0.0
    path_efficiency: float | None = None
    travel_time_s: float | None = None
    started_at_s: float | None = None
    tracking_source: str | None = None
    last_point: Point | None = None
    last_timestamp_s: float | None = None

    def start(
        self,
        start: Point | None,
        goal: Point,
        *,
        tracking_source: str,
        tracking_point: Point | None,
        timestamp_s: float | None,
        assignment_started_at_s: float,
    ) -> None:
        """Initialize metrics for the new start-to-goal assignment.

        tracking_source selects pose or odometry displacement; its initial
        tracking_point and timestamp_s seed the path-length accumulator.
        assignment_started_at_s starts the travel-time clock. No metric is
        published until complete is called.

        Args:
            start: Starting position, state, or timestamp.
            goal: Assigned navigation target.
            tracking_source: Coordinate source used to accumulate traveled distance.
            tracking_point: Initial position in the selected tracking frame.
            timestamp_s: Measurement or event time in seconds.
            assignment_started_at_s: Time the current assignment began, in seconds.
        """

        self.straight_path_length_m = (
            start.distance_to(goal) if start is not None else None
        )
        self.driven_path_length_m = 0.0
        self.path_efficiency = None
        self.travel_time_s = None
        self.started_at_s = assignment_started_at_s
        self.tracking_source = tracking_source
        self.last_point = tracking_point
        self.last_timestamp_s = timestamp_s

    def observe(
        self,
        point: Point,
        *,
        tracking_source: str,
        timestamp_s: float,
        maximum_speed_mps: float,
    ) -> None:
        """Add distance from a new point in the selected tracking frame.

        Samples with another tracking_source or nonincreasing timestamp_s
        are ignored. maximum_speed_mps bounds plausible displacement so a
        localization jump does not inflate the dashboard's driven distance.

        Args:
            point: Position in the coordinate frame described above.
            tracking_source: Coordinate source used to accumulate traveled distance.
            timestamp_s: Measurement or event time in seconds.
            maximum_speed_mps: Maximum plausible speed used to reject pose jumps.
        """

        if tracking_source != self.tracking_source:
            return
        previous_point = self.last_point
        previous_timestamp = self.last_timestamp_s
        if previous_point is None or previous_timestamp is None:
            self.last_point = point
            self.last_timestamp_s = timestamp_s
            return
        elapsed_s = timestamp_s - previous_timestamp
        if elapsed_s <= 0.0:
            return
        displacement_m = previous_point.distance_to(point)
        maximum_plausible_m = maximum_speed_mps * elapsed_s * 1.5 + 0.05
        if displacement_m <= maximum_plausible_m:
            self.driven_path_length_m += displacement_m
        self.last_point = point
        self.last_timestamp_s = timestamp_s

    def complete(self, *, timestamp_s: float) -> None:
        """Finalize travel time and straight/driven ratio at timestamp_s.

        The result remains None when the start or driven distance is unknown;
        a zero-length assignment completed without travel has efficiency 1.

        Args:
            timestamp_s: Measurement or event time in seconds.
        """

        self.tracking_source = None
        if self.started_at_s is not None:
            self.travel_time_s = max(0.0, timestamp_s - self.started_at_s)
        if self.straight_path_length_m is None:
            self.path_efficiency = None
        elif self.driven_path_length_m > 1e-9:
            self.path_efficiency = (
                self.straight_path_length_m / self.driven_path_length_m
            )
        elif self.straight_path_length_m <= 1e-9:
            self.path_efficiency = 1.0

    def cancel(self) -> None:
        """Discard unfinished time and efficiency while keeping raw distances."""

        self.tracking_source = None
        self.path_efficiency = None
        self.travel_time_s = None

    def reset(self) -> None:
        """Clear distances, timestamps, source, and published assignment metrics."""

        self.straight_path_length_m = None
        self.driven_path_length_m = 0.0
        self.path_efficiency = None
        self.travel_time_s = None
        self.started_at_s = None
        self.tracking_source = None
        self.last_point = None
        self.last_timestamp_s = None

    def to_dict(self) -> dict[str, float | None]:
        """Return the four dashboard metric fields, including unavailable values.

        Returns:
            dict[str, float | None]: The four dashboard metric fields, including unavailable
                values.
        """

        return {
            "straightPathLengthM": self.straight_path_length_m,
            "drivenPathLengthM": self.driven_path_length_m,
            "pathEfficiency": self.path_efficiency,
            "travelTimeS": self.travel_time_s,
        }


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0

    @property
    def moving(self) -> bool:
        """Return whether any commanded body-frame axis is nonzero.

        Returns:
            bool: Whether any commanded body-frame axis is nonzero.
        """

        return any(abs(value) > 1e-9 for value in (self.vx, self.vy, self.omega))

    def to_dict(self) -> dict[str, float]:
        """Return body-frame translation and angular speed in SI units.

        Returns:
            dict[str, float]: Body-frame translation and angular speed in SI units.
        """

        return {"vx": self.vx, "vy": self.vy, "omega": self.omega}


@dataclass(frozen=True)
class PoseObservation:
    """Robot center and heading in the calibrated factory-map frame."""

    robot_id: str
    x: float
    y: float
    heading: float
    timestamp_s: float
    sequence: int | None = None
    valid: bool = True
    source: str = "pose"
    error: str | None = None

    @property
    def point(self) -> Point:
        """Return the observed x/y as a point without changing its validity.

        Returns:
            Point: The observed x/y as a point without changing its validity.
        """

        return Point(self.x, self.y)


@dataclass(frozen=True)
class OdometryObservation:
    """Raw Robotino odometry; its local origin is not the factory-map origin."""

    x: float
    y: float
    heading: float
    vx: float
    vy: float
    omega: float
    sequence: int
    timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        """Return local-frame odometry and measured velocity for telemetry.

        Returns:
            dict[str, Any]: Local-frame odometry and measured velocity for telemetry.
        """

        return {
            "x": self.x,
            "y": self.y,
            "headingRad": self.heading,
            "vx": self.vx,
            "vy": self.vy,
            "omega": self.omega,
            "sequence": self.sequence,
            "timestampS": self.timestamp_s,
        }


@dataclass(frozen=True)
class LidarScanObservation:
    sequence: int
    source_timestamp_s: float
    timestamp_s: float
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: tuple[float, ...]
    intensities: tuple[float, ...] = ()
    extrinsic_x_m: float = 0.0
    extrinsic_y_m: float = 0.0
    extrinsic_yaw_rad: float = 0.0

    def valid_indices(self) -> list[int]:
        """Return indices whose ranges fall inside the scan's valid meter span.

        Returns:
            list[int]: Indices whose ranges fall inside the scan's valid meter span.
        """

        minimum = max(self.range_min, 1e-6)
        return [
            index
            for index, value in enumerate(self.ranges)
            if minimum <= value <= self.range_max
        ]

    def to_dict(self, *, now_s: float, max_points: int = 240) -> dict[str, Any]:
        """Return scan metadata and a bounded set of Robotino-frame hit points.

        now_s determines display age. max_points limits transfer size by
        evenly striding valid beams; LiDAR extrinsics move hits from the sensor
        frame into the Robotino body frame for frontend rendering.

        Args:
            now_s: Current clock time in seconds, used for age and expiry checks.
            max_points: Maximum LiDAR points included in the dashboard response.

        Returns:
            dict[str, Any]: Scan metadata and a bounded set of Robotino-frame hit points.
        """

        valid = self.valid_indices()
        stride = max(1, math.ceil(len(valid) / max_points))
        cosine = math.cos(self.extrinsic_yaw_rad)
        sine = math.sin(self.extrinsic_yaw_rad)
        points = []
        for index in valid[::stride]:
            angle = self.angle_min + index * self.angle_increment
            distance = self.ranges[index]
            lx = math.cos(angle) * distance
            ly = math.sin(angle) * distance
            points.append(
                {
                    "x": self.extrinsic_x_m + cosine * lx - sine * ly,
                    "y": self.extrinsic_y_m + sine * lx + cosine * ly,
                    "rangeM": distance,
                    "intensity": (
                        self.intensities[index]
                        if index < len(self.intensities)
                        else None
                    ),
                }
            )
        return {
            "sequence": self.sequence,
            "timestampS": self.timestamp_s,
            "ageS": max(0.0, now_s - self.timestamp_s),
            "rangeMinM": self.range_min,
            "rangeMaxM": self.range_max,
            "sampleCount": len(self.ranges),
            "validCount": len(valid),
            "points": points,
        }


@dataclass
class RobotState:
    ip: str
    color: str
    x: float
    y: float
    heading_rad: float
    status: RobotStatus = RobotStatus.IDLE
    goal_location_id: str | None = None
    sensors: list[float] = field(default_factory=lambda: [0.410] * 9)
    sensors_updated_at_s: float | None = None
    measured_lidar: LidarScanObservation | None = None
    lidar: LidarScanObservation | None = None
    bumper_pressed: bool = False
    battery_low: bool = False
    online: bool = True
    pose_valid: bool = True
    pose_received_at_s: float = 0.0
    pose_source: str = "simulation"
    odometry: OdometryObservation | None = None
    measured_velocity: VelocityCommand = field(default_factory=VelocityCommand)
    proposed_command: VelocityCommand = field(default_factory=VelocityCommand)
    applied_command: VelocityCommand = field(default_factory=VelocityCommand)
    command_queued: bool = False
    stop_reason: str | None = None
    policy_id: str | None = None
    policy_inference_ms: float | None = None
    trajectory: list[Point] = field(default_factory=list)
    navigation_metrics: NavigationMetrics = field(default_factory=NavigationMetrics)

    @property
    def id(self) -> str:
        """Return the Robotino IP used as its unique fleet identifier.

        Returns:
            str: The Robotino IP used as its unique fleet identifier.
        """

        return self.ip

    @property
    def name(self) -> str:
        """Return the same IP used as the dashboard's Robotino display name.

        Returns:
            str: The same IP used as the dashboard's Robotino display name.
        """

        return self.ip

    @property
    def point(self) -> Point:
        """Return the current center in factory-map coordinates.

        Returns:
            Point: The current center in factory-map coordinates.
        """

        return Point(self.x, self.y)

    @property
    def speed_mps(self) -> float:
        """Return planar speed from measured body-frame velocities, in m/s.

        Returns:
            float: Planar speed from measured body-frame velocities, in m/s.
        """

        return math.hypot(self.measured_velocity.vx, self.measured_velocity.vy)

    def to_dict(self, *, now_s: float) -> dict[str, Any]:
        """Return the dashboard snapshot for this Robotino at now_s.

        The result includes map pose, commands, measured and display sensors,
        ages derived from now_s, and completed navigation metrics. Sensor
        absence is represented by null rather than a fabricated measurement.

        Args:
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, Any]: The dashboard snapshot for this Robotino at now_s.
        """

        sensor_age = (
            max(0.0, now_s - self.sensors_updated_at_s)
            if self.sensors_updated_at_s is not None
            else None
        )
        return {
            "id": self.ip,
            "ip": self.ip,
            "name": self.ip,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "headingRad": self.heading_rad,
            "status": self.status.value,
            "online": self.online,
            "poseValid": self.pose_valid,
            "poseSource": self.pose_source,
            "poseAgeS": max(0.0, now_s - self.pose_received_at_s),
            "goalLocationId": self.goal_location_id,
            "speedMps": self.speed_mps,
            "sensors": [
                {"index": index, "rangeM": value}
                for index, value in enumerate(self.sensors)
            ],
            "distanceSensorAgeS": sensor_age,
            "lidar": self.lidar.to_dict(now_s=now_s) if self.lidar else None,
            "bumperPressed": self.bumper_pressed,
            "batteryLow": self.battery_low,
            "odometry": self.odometry.to_dict() if self.odometry else None,
            "measuredVelocity": self.measured_velocity.to_dict(),
            "proposedCommand": self.proposed_command.to_dict(),
            "appliedCommand": self.applied_command.to_dict(),
            "commandQueued": self.command_queued,
            "stopReason": self.stop_reason,
            "policyId": self.policy_id,
            "policyInferenceMs": self.policy_inference_ms,
            "trajectory": [point.to_dict() for point in self.trajectory],
            **self.navigation_metrics.to_dict(),
        }


@dataclass(frozen=True)
class FleetEvent:
    id: int
    time_s: float
    robot_id: str | None
    type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return this event with fleet-clock time formatted for the dashboard.

        Returns:
            dict[str, Any]: This event with fleet-clock time formatted for the dashboard.
        """

        minutes = int(self.time_s // 60)
        seconds = int(self.time_s % 60)
        return {
            "id": self.id,
            "time": f"{minutes:02d}:{seconds:02d}",
            "timeS": self.time_s,
            "robotId": self.robot_id,
            "type": self.type,
            "message": self.message,
        }
