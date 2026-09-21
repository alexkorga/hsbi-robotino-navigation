"""Deterministic tracking for a learned two-dimensional local target."""

import math
from dataclasses import dataclass
from typing import Sequence

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import Point, RobotState, VelocityCommand
from robotino_fleet.learning.profile import BinnedLocalTargetProfile
from robotino_fleet.maps.transforms import normalize_angle
from robotino_fleet.navigation.controller import (
    NavigationControlOutput,
    handover_alignment_output,
)
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle


@dataclass(frozen=True)
class LocalTargetPlan:
    """A frozen map-frame lookahead point and its normalized policy action."""

    point: Point
    action: tuple[float, ...]
    rotation: float | None = None


class LocalTargetMotionController:
    """Convert a slowly changing local target into smooth body-frame commands.

    The local target is frozen in the map frame for one policy period.  This is
    important while the chassis rotates: a body-relative point recomputed on
    every control tick would itself move and reintroduce oscillation.
    """

    def __init__(
        self,
        motion: MotionSettings,
        profile: BinnedLocalTargetProfile,
    ) -> None:
        """Store Robotino motion limits and the learned-target profile.

        motion bounds physical commands; profile defines the policy
        lookahead, translation gains, action dimensions, and handover speeds.

        Args:
            motion: Calibrated Robotino speed and acceleration limits.
            profile: Saved policy contract defining observations and actions.
        """
        self.motion = motion
        self.profile = profile

    def plan(self, robot: RobotState, action: Sequence[float]) -> LocalTargetPlan:
        """Convert a normalized body-frame action to a fixed map-frame target.

        robot supplies the current map pose and heading; action holds
        normalized x/y offsets and, when configured, a rotation value. Return
        a LocalTargetPlan whose map point stays fixed until next inference.
        Wrong-sized or nonfinite actions raise ValueError.

        Args:
            robot: Robotino state used by this operation.
            action: Normalized policy action for the controlled Robotino.

        Returns:
            LocalTargetPlan: Fixed map-frame local target for the policy action.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        expected = self.profile.action_dimensions
        if len(action) != expected:
            meaning = "[x, y, omega]" if expected == 3 else "[x, y]"
            raise ValueError(f"Local-target action must contain {meaning}")
        values = tuple(float(value) for value in action)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Local-target action must be finite")
        x, y = values[:2]
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        magnitude = math.hypot(x, y)
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude
        body_x = x * self.profile.local_target_lookahead_m
        body_y = y * self.profile.local_target_lookahead_m
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        return LocalTargetPlan(
            Point(
                robot.x + cosine * body_x - sine * body_y,
                robot.y + sine * body_x + cosine * body_y,
            ),
            (x, y, *values[2:]),
            rotation=(
                max(-1.0, min(1.0, values[2]))
                if self.profile.uses_learned_rotation
                else None
            ),
        )

    def _translation(
        self,
        robot: RobotState,
        target: Point,
        *,
        speed_limit_mps: float | None = None,
    ) -> VelocityCommand:
        """Return bounded body-frame translation toward target.

        robot supplies pose and heading for map-to-body conversion.
        speed_limit_mps may further cap both axes during handover; a
        nonpositive cap returns a stationary command. The combined command
        stays within the forward/sideways speed ellipse.

        Args:
            robot: Robotino state used by this operation.
            target: Fixed local target to track between policy calls.
            speed_limit_mps: Maximum translation speed for the current approach.

        Returns:
            VelocityCommand: Bounded body-frame translation toward target.
        """
        map_dx = target.x - robot.x
        map_dy = target.y - robot.y
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        vx = self.profile.translation_kp * (cosine * map_dx + sine * map_dy)
        vy = self.profile.translation_kp * (-sine * map_dx + cosine * map_dy)
        forward_limit = self.motion.forward_max_mps
        sideways_limit = self.motion.sideways_max_mps
        if speed_limit_mps is not None:
            if speed_limit_mps <= 0:
                return VelocityCommand()
            forward_limit = min(forward_limit, speed_limit_mps)
            sideways_limit = min(sideways_limit, speed_limit_mps)
        normalized = math.hypot(vx / forward_limit, vy / sideways_limit)
        if normalized > 1.0:
            vx /= normalized
            vy /= normalized
        return VelocityCommand(vx, vy, 0.0)

    def _handover_speed_limit(
        self,
        distance_m: float,
        position_tolerance_m: float,
    ) -> float:
        """Return the shared translation cap near positional handover.

        distance_m is remaining goal distance and
        position_tolerance_m is the arrival boundary. The cap decreases
        linearly toward crawl speed while respecting both axis limits.

        Args:
            distance_m: Distance in meters.
            position_tolerance_m: Maximum accepted position error in meters.

        Returns:
            float: The shared translation cap near positional handover.
        """

        remaining = max(0.0, distance_m - position_tolerance_m)
        robot_maximum = min(
            self.motion.forward_max_mps,
            self.motion.sideways_max_mps,
        )
        slowdown_span = max(
            1e-9,
            self.profile.handover_slowdown_radius_m - position_tolerance_m,
        )
        fraction = min(1.0, remaining / slowdown_span)
        minimum = min(self.profile.handover_min_speed_mps, robot_maximum)
        return min(
            robot_maximum,
            minimum + (robot_maximum - minimum) * fraction,
        )

    def command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        plan: LocalTargetPlan,
        *,
        handover_active: bool = False,
        policy_id: str | None = None,
        inference_ms: float | None = None,
    ) -> NavigationControlOutput:
        """Track plan between policy calls and handle goal approach.

        robot supplies current pose, goal the final label and
        tolerance, and plan the frozen learned lookahead. Near the label,
        speed tapers and tracking switches to the exact goal. At tolerance or
        when handover_active, deterministic heading alignment takes over.
        policy_id and inference_ms pass through to the returned
        NavigationControlOutput for monitoring.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            plan: Held map-frame target tracked between policy inferences.
            handover_active: Whether position tolerance has triggered final goal handover.
            policy_id: Identifier of the policy producing the navigation output.
            inference_ms: Measured model inference time in milliseconds.

        Returns:
            NavigationControlOutput: Body-frame command tracking the held target.
        """

        distance = robot.point.distance_to(goal.point)
        position_ready = distance <= goal.position_tolerance_m
        if handover_active or position_ready:
            return handover_alignment_output(
                robot,
                goal,
                rotation_max_rps=self.motion.rotation_max_rps,
                inference_ms=inference_ms,
                policy_id=policy_id,
            )

        approaching = distance <= self.profile.approach_radius_m
        handover_slowdown = distance <= self.profile.handover_slowdown_radius_m
        translation = self._translation(
            robot,
            goal.point if approaching else plan.point,
            speed_limit_mps=(
                self._handover_speed_limit(
                    distance,
                    goal.position_tolerance_m,
                )
                if handover_slowdown
                else None
            ),
        )
        if plan.rotation is None:
            goal_bearing = math.atan2(goal.point.y - robot.y, goal.point.x - robot.x)
            heading_error = normalize_angle(goal_bearing - robot.heading_rad)
            omega = max(
                -self.motion.rotation_max_rps,
                min(
                    self.motion.rotation_max_rps,
                    self.profile.heading_kp * heading_error,
                ),
            )
        else:
            omega = plan.rotation * self.motion.rotation_max_rps
        return NavigationControlOutput(
            VelocityCommand(translation.vx, translation.vy, omega),
            NavigationLifecycle.APPROACHING
            if approaching
            else NavigationLifecycle.NAVIGATING,
            inference_ms=inference_ms,
            policy_id=policy_id,
        )
