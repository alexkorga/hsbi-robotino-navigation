"""One readable polling worker per physical Robotino."""

import asyncio
import time
from collections.abc import Callable

from robotino_fleet.adapters.robotino.http_api import (
    LaserScan,
    Odometry,
    Pose,
    RobotinoHttpClient,
)
from robotino_fleet.config import LidarExtrinsic
from robotino_fleet.domain.models import (
    LidarScanObservation,
    OdometryObservation,
    Point,
    PoseObservation,
)
from robotino_fleet.maps.transforms import PlanarTransform


POLL_TICK_S = 0.05
POSE_PERIOD_S = 0.10
TELEMETRY_PERIOD_S = 0.20
LIDAR_PERIOD_S = 0.20
BATTERY_PERIOD_S = 5.0
POSE_FAST_RETRY_LIMIT = 5
POSE_RETRY_S = 0.50
POSE_DEGRADED_RETRY_S = 15.0


class RobotinoPollingService:
    """Poll each IP independently and publish parsed telemetry to FleetRuntime."""

    def __init__(
        self,
        clients: dict[str, RobotinoHttpClient],
        *,
        transform: PlanarTransform,
        lidar_extrinsics: dict[str, LidarExtrinsic],
        on_pose: Callable[[PoseObservation], None],
        on_odometry: Callable[[str, OdometryObservation], None],
        on_lidar: Callable[[str, LidarScanObservation], None],
        on_distance: Callable[[str, tuple[float, ...], float], None],
        on_bumper: Callable[[str, bool], None],
        on_battery: Callable[[str, bool], None],
        on_error: Callable[[str, str, str], None],
    ) -> None:
        """Bind physical polling inputs and FleetRuntime callbacks.

        Every Robotino is polled independently, so one unreachable endpoint
        does not stop the others. Polling begins only when start is called.

        Args:
            clients: HTTP clients keyed by Robotino IP.
            transform: Validated indoor-tracking-to-factory-map coordinate transform.
            lidar_extrinsics: Per-Robotino LiDAR mounting offsets for scan projection.
            on_pose: Callback receiving transformed physical pose observations.
            on_odometry: Callback receiving local odometry measurements.
            on_lidar: Callback receiving physical LiDAR scans.
            on_distance: Callback receiving proximity-sensor measurements.
            on_bumper: Callback receiving physical bumper state.
            on_battery: Callback receiving battery-low state.
            on_error: Callback receiving endpoint polling errors.
        """

        self.clients = clients
        self.transform = transform
        self.lidar_extrinsics = lidar_extrinsics
        self.on_pose = on_pose
        self.on_odometry = on_odometry
        self.on_lidar = on_lidar
        self.on_distance = on_distance
        self.on_bumper = on_bumper
        self.on_battery = on_battery
        self.on_error = on_error
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self.pose_failures = {robot_id: 0 for robot_id in clients}

    async def start(self) -> None:
        """Start one asynchronous polling task per configured Robotino."""

        if self._tasks:
            return
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._robot_loop(robot_id, client), name=f"poll-{robot_id}")
            for robot_id, client in self.clients.items()
        ]

    async def stop(self) -> None:
        """Cancel all polling tasks and wait for their shutdown."""

        self._stop.set()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call(self, robot_id: str, endpoint: str, function) -> object | None:
        """Run blocking function in a thread for robot_id/endpoint.

        Return its parsed value, or publish an endpoint error and return
        None. Cancellation propagates to stop the polling loop promptly.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            endpoint: Robotino HTTP telemetry endpoint or endpoint name.
            function: Blocking HTTP operation executed outside the event loop.

        Returns:
            object | None: Parsed endpoint value, or None after a polling error.
        """

        try:
            return await asyncio.to_thread(function)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.on_error(robot_id, endpoint, str(error))
            return None

    async def _robot_loop(
        self, robot_id: str, client: RobotinoHttpClient
    ) -> None:
        """Poll client for robot_id until service shutdown.

        Publish transformed poses and parsed telemetry through callbacks.
        Failed pose reads first retry quickly, then back off after the fast
        retry limit; other endpoints keep independent schedules.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            client: HTTP client for the selected physical Robotino.
        """

        next_due = {
            "pose": 0.0,
            "odometry": 0.0,
            "distance": 0.0,
            "bumper": 0.0,
            "lidar": 0.0,
            "battery": 0.0,
        }
        while not self._stop.is_set():
            now = time.monotonic()
            due = [name for name, instant in next_due.items() if now >= instant]
            jobs = {
                "pose": client.get_pose,
                "odometry": client.get_odometry,
                "distance": client.get_distance_sensors,
                "bumper": client.get_bumper,
                "lidar": client.get_laser_scan,
                "battery": client.get_battery_low,
            }
            results = await asyncio.gather(
                *(self._call(robot_id, name, jobs[name]) for name in due)
            )
            received = time.monotonic()
            for name, value in zip(due, results, strict=True):
                if name == "pose":
                    if value is None:
                        self.pose_failures[robot_id] += 1
                        next_due[name] = received + (
                            POSE_RETRY_S
                            if self.pose_failures[robot_id] <= POSE_FAST_RETRY_LIMIT
                            else POSE_DEGRADED_RETRY_S
                        )
                    else:
                        self.pose_failures[robot_id] = 0
                        next_due[name] = received + POSE_PERIOD_S
                        pose = value
                        assert isinstance(pose, Pose)
                        point, heading = self.transform.apply(
                            Point(pose.x, pose.y), pose.heading
                        )
                        self.on_pose(
                            PoseObservation(
                                robot_id,
                                point.x,
                                point.y,
                                heading,
                                received,
                                pose.sequence,
                                source="http-pose",
                            )
                        )
                elif name == "odometry":
                    next_due[name] = received + TELEMETRY_PERIOD_S
                    if isinstance(value, Odometry):
                        self.on_odometry(
                            robot_id,
                            OdometryObservation(
                                value.x,
                                value.y,
                                value.rotation,
                                value.vx,
                                value.vy,
                                value.omega,
                                value.sequence,
                                received,
                            ),
                        )
                elif name == "distance":
                    next_due[name] = received + TELEMETRY_PERIOD_S
                    if isinstance(value, tuple):
                        self.on_distance(robot_id, value, received)
                elif name == "bumper":
                    next_due[name] = received + TELEMETRY_PERIOD_S
                    if isinstance(value, bool):
                        self.on_bumper(robot_id, value)
                elif name == "lidar":
                    next_due[name] = received + LIDAR_PERIOD_S
                    if isinstance(value, LaserScan):
                        extrinsic = self.lidar_extrinsics.get(robot_id, LidarExtrinsic())
                        self.on_lidar(
                            robot_id,
                            LidarScanObservation(
                                value.sequence,
                                value.stamp,
                                received,
                                value.angle_min,
                                value.angle_max,
                                value.angle_increment,
                                value.range_min,
                                value.range_max,
                                value.ranges,
                                value.intensities,
                                extrinsic.x_m,
                                extrinsic.y_m,
                                extrinsic.yaw_rad,
                            ),
                        )
                elif name == "battery":
                    next_due[name] = received + BATTERY_PERIOD_S
                    if isinstance(value, bool):
                        self.on_battery(robot_id, value)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_TICK_S)
            except TimeoutError:
                pass
