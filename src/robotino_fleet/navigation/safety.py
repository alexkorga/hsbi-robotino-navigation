"""Deterministic stop conditions; motion limiting happens exactly once later."""

import math
from dataclasses import dataclass

from robotino_fleet.config import DISTANCE_SENSOR_CLEAR_M, ROBOT_RADIUS_M
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.navigation.goals import NavigationGoal


# Physical HTTP telemetry can briefly exceed one polling period when several
# Robotinos are active. Keep using the last observation through short API
# hiccups instead of alternating between motion and an immediate stop.
POSE_STALE_S = 2.0
SENSOR_STALE_S = 2.0
CRITICAL_CLEARANCE_M = 0.06


@dataclass(frozen=True)
class SafetyDecision:
    command: VelocityCommand
    allowed: bool
    reason: str | None


class NavigationSafetySupervisor:
    """Stop unsafe commands using actual telemetry, not virtual LiDAR returns.

    Map containment is not a hard stop: a temporarily displaced map pose must
    not prevent a Robotino from recovering while real sensor stops stay active.
    """

    def evaluate(
        self,
        robot: RobotState,
        goal: NavigationGoal | None,
        proposed: VelocityCommand,
        *,
        now_s: float,
        physical: bool,
        policy_error: str | None = None,
    ) -> SafetyDecision:
        """Decide whether a proposed command may reach the drive adapter.

        robot supplies pose and sensor observations; goal must be
        assigned; proposed is the navigation output to gate. now_s
        sets telemetry-age checks, while physical selects untouched LiDAR
        for live emergency stops instead of virtual policy/visualization rays.
        policy_error forces a stop when inference failed. Return a
        SafetyDecision containing the original command if allowed, or a
        zero command and human-readable reason otherwise.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            proposed: Velocity command proposed by the navigation controller.
            now_s: Current clock time in seconds, used for age and expiry checks.
            physical: Whether safety must rely on current physical sensor data.
            policy_error: Policy inference failure to expose as a safety stop.

        Returns:
            SafetyDecision: Command and safety state permitted by the physical-data checks.
        """

        reason: str | None = None
        # On hardware, only the untouched scan received from /data/scan0 is a
        # valid emergency-stop input. robot.lidar may additionally contain
        # virtual map boundaries and peer footprints for inference/display.
        lidar = robot.measured_lidar if physical else robot.lidar
        if goal is None:
            reason = "no navigation goal"
        elif policy_error:
            reason = policy_error
        elif not all(math.isfinite(value) for value in (proposed.vx, proposed.vy, proposed.omega)):
            reason = "invalid policy output"
        elif not robot.pose_valid:
            reason = "map pose unavailable"
        elif physical and now_s - robot.pose_received_at_s > POSE_STALE_S:
            reason = "map pose stale"
        elif robot.bumper_pressed:
            reason = "bumper pressed"
        elif lidar is None:
            reason = "LiDAR missing"
        elif now_s - lidar.timestamp_s > SENSOR_STALE_S:
            reason = "LiDAR stale"
        elif robot.sensors_updated_at_s is None:
            reason = "distance sensors missing"
        elif now_s - robot.sensors_updated_at_s > SENSOR_STALE_S:
            reason = "distance sensors stale"
        if reason is None and lidar:
            valid = [
                value
                for value in lidar.ranges
                if max(lidar.range_min, 1e-6) <= value <= lidar.range_max
            ]
            if valid and min(valid) - ROBOT_RADIUS_M <= CRITICAL_CLEARANCE_M:
                reason = "critical LiDAR clearance"
        detected = [
            value for value in robot.sensors if value < DISTANCE_SENSOR_CLEAR_M - 1e-8
        ]
        if reason is None and detected and min(detected) <= CRITICAL_CLEARANCE_M:
            reason = "critical proximity distance"
        if reason:
            return SafetyDecision(VelocityCommand(), False, reason)
        return SafetyDecision(proposed, True, None)
