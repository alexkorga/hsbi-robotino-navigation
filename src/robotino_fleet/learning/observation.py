"""Fixed model observation contract shared by simulation and live inference."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from robotino_fleet.domain.models import RobotState
from robotino_fleet.learning.profile import BinnedLocalTargetProfile
from robotino_fleet.maps.transforms import normalize_angle
from robotino_fleet.navigation.goals import NavigationGoal


@dataclass(frozen=True)
class ModelObservationBuilder:
    """Normalized observation contract for a direct-velocity policy."""

    lidar_beam_count: int
    distance_sensor_count: int
    lidar_range_max_m: float
    distance_clear_value_m: float
    max_goal_distance_m: float
    max_forward_mps: float
    max_sideways_mps: float
    max_rotation_rps: float

    def space(self):
        """Return the Gymnasium Dict space expected by the direct policy.

        Returns:
            object: The Gymnasium Dict space expected by the direct policy.
        """

        from gymnasium import spaces

        return spaces.Dict(
            {
                "lidar": spaces.Box(0.0, 1.0, (self.lidar_beam_count,), np.float32),
                "lidar_valid": spaces.Box(
                    0.0, 1.0, (self.lidar_beam_count,), np.float32
                ),
                "proximity": spaces.Box(
                    0.0, 1.0, (self.distance_sensor_count,), np.float32
                ),
                "goal_body": spaces.Box(-1.0, 1.0, (2,), np.float32),
                "goal_distance": spaces.Box(0.0, 1.0, (1,), np.float32),
                "goal_heading": spaces.Box(-1.0, 1.0, (2,), np.float32),
                "velocity": spaces.Box(-1.0, 1.0, (3,), np.float32),
                "previous_action": spaces.Box(-1.0, 1.0, (3,), np.float32),
            }
        )

    @staticmethod
    def _fixed(values: Sequence[float], count: int, fill: float) -> np.ndarray:
        """Return values truncated or padded to count with fill.

        Args:
            values: Measured or computed values supplied to this operation.
            count: Required number of elements.
            fill: Value used to pad missing elements.

        Returns:
            np.ndarray: Values truncated or padded to count with fill.
        """

        result = np.full((count,), fill, dtype=np.float32)
        copied = min(count, len(values))
        if copied:
            result[:copied] = np.asarray(values[:copied], dtype=np.float32)
        return result

    def build(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        previous_action: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        """Return normalized direct-velocity features for robot/goal.

        velocity is measured body motion and previous_action is the
        last normalized three-value policy output. Missing scan readings have
        separate validity flags; returned arrays match space. Invalid
        vector lengths or nonfinite features raise ValueError.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            velocity: Measured or requested Robotino body-frame velocity.
            previous_action: Previous normalized policy action for temporal context.

        Returns:
            dict[str, np.ndarray]: Normalized direct-velocity features for robot/goal.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        lidar = np.ones((self.lidar_beam_count,), dtype=np.float32)
        lidar_valid = np.zeros((self.lidar_beam_count,), dtype=np.float32)
        if robot.lidar is not None:
            raw = self._fixed(robot.lidar.ranges, self.lidar_beam_count, 0.0)
            minimum = max(robot.lidar.range_min, 1e-6)
            valid = (raw >= minimum) & (raw <= robot.lidar.range_max)
            lidar_valid = valid.astype(np.float32)
            lidar[valid] = np.clip(raw[valid] / self.lidar_range_max_m, 0.0, 1.0)

        proximity_raw = self._fixed(
            robot.sensors, self.distance_sensor_count, self.distance_clear_value_m
        )
        proximity = np.clip(
            proximity_raw / self.distance_clear_value_m, 0.0, 1.0
        ).astype(np.float32)

        dx = goal.point.x - robot.x
        dy = goal.point.y - robot.y
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        body_x = cosine * dx + sine * dy
        body_y = -sine * dx + cosine * dy
        distance = math.hypot(dx, dy)
        if goal.heading_rad is None:
            heading_error = 0.0
        else:
            heading_error = normalize_angle(goal.heading_rad - robot.heading_rad)
        velocity_values = list(velocity)
        if len(velocity_values) != 3:
            raise ValueError("velocity must contain vx, vy, omega")
        previous_values = list(previous_action)
        if len(previous_values) != 3:
            raise ValueError("previous_action must contain three values")
        observation = {
            "lidar": lidar,
            "lidar_valid": lidar_valid,
            "proximity": proximity,
            "goal_body": np.asarray(
                [body_x / self.max_goal_distance_m, body_y / self.max_goal_distance_m],
                dtype=np.float32,
            ),
            "goal_distance": np.asarray(
                [distance / self.max_goal_distance_m], dtype=np.float32
            ),
            "goal_heading": np.asarray(
                [math.sin(heading_error), math.cos(heading_error)], dtype=np.float32
            ),
            "velocity": np.asarray(
                [
                    velocity_values[0] / self.max_forward_mps,
                    velocity_values[1] / self.max_sideways_mps,
                    velocity_values[2] / self.max_rotation_rps,
                ],
                dtype=np.float32,
            ),
            "previous_action": np.asarray(previous_values, dtype=np.float32),
        }
        for value in observation.values():
            np.clip(value, -1.0, 1.0, out=value)
            if not np.all(np.isfinite(value)):
                raise ValueError("Model observation contains non-finite values")
        return observation


@dataclass(frozen=True)
class BinnedLocalTargetObservationBuilder:
    """Compact fixed-angle LiDAR contract for the local-target policy.

    The complete scan remains on ``RobotState`` for the safety supervisor.  The
    learned policy receives robust per-sector closeness plus the fraction of
    valid measurements, so a missing return is never confused with an obstacle.
    """

    profile: BinnedLocalTargetProfile

    def space(self):
        """Return the profile-specific Gymnasium observation space.

        Returns:
            object: The profile-specific Gymnasium observation space.
        """

        from gymnasium import spaces

        bins = self.profile.lidar_bin_count
        values = {
                "lidar_closeness": spaces.Box(0.0, 1.0, (bins,), np.float32),
                "lidar_validity": spaces.Box(0.0, 1.0, (bins,), np.float32),
                "proximity_closeness": spaces.Box(
                    0.0,
                    1.0,
                    (self.profile.proximity_sensor_count,),
                    np.float32,
                ),
                "goal_distance": spaces.Box(0.0, 1.0, (1,), np.float32),
                "velocity": spaces.Box(-1.0, 1.0, (3,), np.float32),
                "previous_action": spaces.Box(
                    -1.0,
                    1.0,
                    (self.profile.action_dimensions,),
                    np.float32,
                ),
        }
        if self.profile.uses_pose_corrected_lidar_360:
            values["lidar_freshness"] = spaces.Box(
                0.0, 1.0, (bins,), np.float32
            )
        if self.profile.action_mode == "forward-turn":
            values["goal_direction"] = spaces.Box(
                -1.0, 1.0, (2,), np.float32
            )
        else:
            values["goal_body"] = spaces.Box(-1.0, 1.0, (2,), np.float32)
        frames = self.profile.temporal_frames
        if frames > 1:
            values.update(
                {
                    "lidar_closeness_history": spaces.Box(
                        0.0, 1.0, (frames, bins), np.float32
                    ),
                    "lidar_validity_history": spaces.Box(
                        0.0, 1.0, (frames, bins), np.float32
                    ),
                    "proximity_closeness_history": spaces.Box(
                        0.0,
                        1.0,
                        (frames, self.profile.proximity_sensor_count),
                        np.float32,
                    ),
                    "velocity_history": spaces.Box(
                        -1.0, 1.0, (frames, 3), np.float32
                    ),
                    "action_history": spaces.Box(
                        -1.0,
                        1.0,
                        (frames, self.profile.action_dimensions),
                        np.float32,
                    ),
                }
            )
            if self.profile.uses_pose_corrected_lidar_360:
                values["lidar_freshness_history"] = spaces.Box(
                    0.0, 1.0, (frames, bins), np.float32
                )
        return spaces.Dict(values)

    def _binned_lidar(self, robot: RobotState) -> tuple[np.ndarray, np.ndarray]:
        """Return per-angle closeness and valid-return fractions for robot.

        Invalid/missing scan values do not masquerade as near obstacles.
        Distance quantiles and bin angles come from the saved policy profile.

        Args:
            robot: Robotino state used by this operation.

        Returns:
            tuple[np.ndarray, np.ndarray]: Per-angle closeness and valid-return fractions for
                robot.
        """

        count = self.profile.lidar_bin_count
        closeness = np.zeros((count,), dtype=np.float32)
        validity = np.zeros((count,), dtype=np.float32)
        scan = robot.lidar
        if scan is None:
            return closeness, validity

        values: list[list[float]] = [[] for _ in range(count)]
        totals = [0 for _ in range(count)]
        configured_span = (
            self.profile.lidar_angle_max_rad - self.profile.lidar_angle_min_rad
        )
        minimum = max(scan.range_min, 1e-6)
        for index, raw_distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            relative = (angle - self.profile.lidar_angle_min_rad) / configured_span
            if relative < -1e-7 or relative > 1.0 + 1e-7:
                continue
            bin_index = min(count - 1, max(0, int(relative * count)))
            totals[bin_index] += 1
            distance = float(raw_distance)
            if not math.isfinite(distance) or not minimum <= distance <= scan.range_max:
                continue
            values[bin_index].append(
                min(distance, self.profile.lidar_clip_distance_m)
            )

        for index, bin_values in enumerate(values):
            if totals[index]:
                validity[index] = len(bin_values) / totals[index]
            if not bin_values:
                continue
            ordered = sorted(bin_values)
            percentile_index = int(
                self.profile.lidar_percentile * (len(ordered) - 1)
            )
            distance = ordered[percentile_index]
            closeness[index] = 1.0 - distance / self.profile.lidar_clip_distance_m
        return closeness, validity

    @staticmethod
    def _fixed(values: Sequence[float], count: int, fill: float) -> np.ndarray:
        """Return count float32 values, padding absent readings with fill.

        Args:
            values: Measured or computed values supplied to this operation.
            count: Required number of elements.
            fill: Value used to pad missing elements.

        Returns:
            np.ndarray: Count float32 values, padding absent readings with fill.
        """

        result = np.full((count,), fill, dtype=np.float32)
        copied = min(count, len(values))
        if copied:
            result[:copied] = np.asarray(values[:copied], dtype=np.float32)
        return result

    def build(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        previous_action: Sequence[float] | None = None,
        lidar_360: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return binned LiDAR and local-goal features for robot/goal.

        velocity is measured body motion, previous_action the last
        normalized policy output, and lidar_360 optional reprojected scan
        memory. Returned float32 keys/shapes match the saved profile; invalid
        lengths or incompatible 360 input raise ValueError.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            velocity: Measured or requested Robotino body-frame velocity.
            previous_action: Previous normalized policy action for temporal context.
            lidar_360: Pose-corrected 360-degree LiDAR features when available.

        Returns:
            dict[str, np.ndarray]: Binned LiDAR and local-goal features for robot/goal.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if lidar_360 is None:
            lidar_closeness, lidar_validity = self._binned_lidar(robot)
            lidar_freshness = None
        else:
            lidar_closeness, lidar_validity, lidar_freshness = lidar_360
            expected = (self.profile.lidar_bin_count,)
            if any(value.shape != expected for value in lidar_360):
                raise ValueError(f"360 LiDAR values must have shape {expected}")
        proximity = self._fixed(
            robot.sensors,
            self.profile.proximity_sensor_count,
            self.profile.proximity_clear_distance_m,
        )
        proximity_closeness = 1.0 - np.clip(
            proximity / self.profile.proximity_clear_distance_m, 0.0, 1.0
        )

        dx = goal.point.x - robot.x
        dy = goal.point.y - robot.y
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        body_x = cosine * dx + sine * dy
        body_y = -sine * dx + cosine * dy
        distance = math.hypot(dx, dy)

        velocity_values = list(velocity)
        if len(velocity_values) != 3:
            raise ValueError("velocity must contain vx, vy, omega")
        previous_values = list(
            previous_action
            if previous_action is not None
            else (0.0,) * self.profile.action_dimensions
        )
        if len(previous_values) != self.profile.action_dimensions:
            raise ValueError(
                "previous_action must contain "
                f"{self.profile.action_dimensions} policy values"
            )

        observation = {
            "lidar_closeness": lidar_closeness,
            "lidar_validity": lidar_validity,
            "proximity_closeness": proximity_closeness.astype(np.float32),
            "goal_distance": np.asarray(
                [distance / self.profile.observation_goal_distance_m],
                dtype=np.float32,
            ),
            "velocity": np.asarray(
                [
                    velocity_values[0] / self.profile.observation_forward_mps,
                    velocity_values[1] / self.profile.observation_sideways_mps,
                    velocity_values[2] / self.profile.observation_rotation_rps,
                ],
                dtype=np.float32,
            ),
            "previous_action": np.asarray(previous_values, dtype=np.float32),
        }
        if self.profile.uses_pose_corrected_lidar_360:
            if lidar_freshness is None:
                raise ValueError("The 360 policy requires pose-corrected LiDAR memory")
            observation["lidar_freshness"] = np.asarray(
                lidar_freshness, dtype=np.float32
            )
        if self.profile.action_mode == "forward-turn":
            inverse_distance = 1.0 / max(distance, 1e-9)
            observation["goal_direction"] = np.asarray(
                [body_x * inverse_distance, body_y * inverse_distance],
                dtype=np.float32,
            )
        else:
            observation["goal_body"] = np.asarray(
                [
                    body_x / self.profile.observation_goal_distance_m,
                    body_y / self.profile.observation_goal_distance_m,
                ],
                dtype=np.float32,
            )
        for value in observation.values():
            np.clip(value, -1.0, 1.0, out=value)
            if not np.all(np.isfinite(value)):
                raise ValueError("Binned model observation contains non-finite values")
        return observation


class TemporalObservationHistory:
    """Add fixed-rate sensor and ego-motion history to current observations."""

    HISTORY_KEYS = (
        "lidar_closeness",
        "lidar_validity",
        "proximity_closeness",
        "velocity",
        "previous_action",
    )

    def __init__(self, frames: int) -> None:
        """Keep frames temporal observations per robot ID.

        A single frame returns current observations unchanged; nonpositive
        frame counts raise ValueError.

        Args:
            frames: Keep frames temporal observations per robot ID.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if frames < 1:
            raise ValueError("Temporal history must contain at least one frame")
        self.frames = frames
        self._history: dict[str, deque[dict[str, np.ndarray]]] = {}

    def reset(self, robot_id: str | None = None) -> None:
        """Clear robot_id history, or all histories if it is None.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        if robot_id is None:
            self._history.clear()
        else:
            self._history.pop(robot_id, None)

    def augment(
        self,
        robot_id: str,
        observation: dict[str, np.ndarray],
        *,
        advance: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return observation augmented with robot_id's frame history.

        advance controls whether the current frame is recorded. An empty
        history is filled by repeating the first frame; read-only inspection
        can request advance=False to avoid shifting model state.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            observation: Current policy observation or measured robot state.
            advance: Whether to record the current observation in temporal memory.

        Returns:
            dict[str, np.ndarray]: Observation augmented with robot_id's frame history.
        """

        if self.frames == 1:
            return observation
        history_keys = self.HISTORY_KEYS + (
            ("lidar_freshness",) if "lidar_freshness" in observation else ()
        )
        snapshot = {
            key: np.asarray(observation[key], dtype=np.float32).copy()
            for key in history_keys
        }
        history = self._history.setdefault(robot_id, deque(maxlen=self.frames))
        if advance or not history:
            history.append(snapshot)
        while len(history) < self.frames:
            history.appendleft(
                {key: value.copy() for key, value in snapshot.items()}
            )
        result = dict(observation)
        result.update(
            {
                "lidar_closeness_history": np.stack(
                    [frame["lidar_closeness"] for frame in history]
                ),
                "lidar_validity_history": np.stack(
                    [frame["lidar_validity"] for frame in history]
                ),
                "proximity_closeness_history": np.stack(
                    [frame["proximity_closeness"] for frame in history]
                ),
                "velocity_history": np.stack(
                    [frame["velocity"] for frame in history]
                ),
                "action_history": np.stack(
                    [frame["previous_action"] for frame in history]
                ),
            }
        )
        if "lidar_freshness" in observation:
            result["lidar_freshness_history"] = np.stack(
                [frame["lidar_freshness"] for frame in history]
            )
        return result


@dataclass(frozen=True)
class _LidarMemoryFrame:
    observed_at_s: float
    world_points: np.ndarray


class PoseCorrectedLidar360Memory:
    """Reproject recent LiDAR hits into a Robotino's current body frame."""

    def __init__(self, profile: BinnedLocalTargetProfile) -> None:
        """Set up scan history using a 360°-capable profile.

        A profile without the pose-corrected 360° contract is rejected.

        Args:
            profile: Saved policy contract defining observations and actions.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if not profile.uses_pose_corrected_lidar_360:
            raise ValueError("360 LiDAR memory requires a 360 policy profile")
        self.profile = profile
        self._frames: dict[str, deque[_LidarMemoryFrame]] = {}
        self._sequences: dict[str, int] = {}

    def reset(self, robot_id: str | None = None) -> None:
        """Discard robot_id's scan memory, or every robot if None.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        if robot_id is None:
            self._frames.clear()
            self._sequences.clear()
        else:
            self._frames.pop(robot_id, None)
            self._sequences.pop(robot_id, None)

    def _world_points(
        self,
        robot: RobotState,
        other_robots: Sequence[RobotState],
    ) -> np.ndarray:
        """Project valid robot scan returns into map coordinates.

        other_robots suppresses hits likely belonging to peers so stale
        peer positions do not persist in memory. Return an (N, 2) float32
        array of world points, possibly empty.

        Args:
            robot: Robotino state used by this operation.
            other_robots: Peer Robotino states to exclude from static scan memory.

        Returns:
            np.ndarray: Valid LiDAR hit positions in factory-map coordinates.
        """

        scan = robot.lidar
        if scan is None:
            return np.empty((0, 2), dtype=np.float32)
        points = []
        robot_cosine = math.cos(robot.heading_rad)
        robot_sine = math.sin(robot.heading_rad)
        sensor_cosine = math.cos(scan.extrinsic_yaw_rad)
        sensor_sine = math.sin(scan.extrinsic_yaw_rad)
        minimum = max(scan.range_min, 1e-6)
        for index, raw_distance in enumerate(scan.ranges):
            distance = float(raw_distance)
            if (
                not math.isfinite(distance)
                or not minimum <= distance <= scan.range_max
                or distance > self.profile.lidar_clip_distance_m
            ):
                continue
            angle = scan.angle_min + index * scan.angle_increment
            laser_x = math.cos(angle) * distance
            laser_y = math.sin(angle) * distance
            body_x = (
                scan.extrinsic_x_m
                + sensor_cosine * laser_x
                - sensor_sine * laser_y
            )
            body_y = (
                scan.extrinsic_y_m
                + sensor_sine * laser_x
                + sensor_cosine * laser_y
            )
            world_x = robot.x + robot_cosine * body_x - robot_sine * body_y
            world_y = robot.y + robot_sine * body_x + robot_cosine * body_y
            if any(
                math.hypot(world_x - other.x, world_y - other.y) <= 0.35
                for other in other_robots
                if other.id != robot.id
            ):
                continue
            points.append((world_x, world_y))
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)

    def observe(
        self,
        robot: RobotState,
        *,
        now_s: float,
        robots: Sequence[RobotState] = (),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reproject fresh scan hits to current body-frame bins.

        Uses now_s to expire old frames and robots to exclude peer
        footprints. Returns closeness, validity, and freshness arrays.

        Args:
            robot: Robotino state used by this operation.
            now_s: Current clock time in seconds, used for age and expiry checks.
            robots: Robotino states in the shared world.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Current scan bins, validity, and age
                features.
        """

        frames = self._frames.setdefault(robot.id, deque())
        cutoff = now_s - self.profile.lidar_memory_duration_s
        while frames and frames[0].observed_at_s < cutoff:
            frames.popleft()
        scan = robot.lidar
        if scan is not None and self._sequences.get(robot.id) != scan.sequence:
            frames.append(
                _LidarMemoryFrame(now_s, self._world_points(robot, robots))
            )
            self._sequences[robot.id] = scan.sequence

        count = self.profile.lidar_bin_count
        distances: list[list[tuple[float, float]]] = [[] for _ in range(count)]
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        duration = self.profile.lidar_memory_duration_s
        for frame in frames:
            freshness = max(0.0, 1.0 - (now_s - frame.observed_at_s) / duration)
            for world_x, world_y in frame.world_points:
                dx = float(world_x) - robot.x
                dy = float(world_y) - robot.y
                body_x = cosine * dx + sine * dy
                body_y = -sine * dx + cosine * dy
                distance = math.hypot(body_x, body_y)
                if distance > self.profile.lidar_clip_distance_m:
                    continue
                angle = math.atan2(body_y, body_x)
                relative = (angle + math.pi) / (2.0 * math.pi)
                bin_index = min(count - 1, max(0, int(relative * count)))
                distances[bin_index].append((distance, freshness))

        closeness = np.zeros(count, dtype=np.float32)
        validity = np.zeros(count, dtype=np.float32)
        freshness_values = np.zeros(count, dtype=np.float32)
        for index, values in enumerate(distances):
            if not values:
                continue
            values.sort(key=lambda item: item[0])
            selected = values[
                int(self.profile.lidar_percentile * (len(values) - 1))
            ]
            closeness[index] = 1.0 - selected[0] / self.profile.lidar_clip_distance_m
            validity[index] = 1.0
            freshness_values[index] = max(value[1] for value in values)
        return closeness, validity, freshness_values
