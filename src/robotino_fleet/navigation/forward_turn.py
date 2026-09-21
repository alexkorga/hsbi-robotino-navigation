"""Deterministic command mapping for a learned forward-and-turn policy."""

import math
from typing import Sequence

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.learning.profile import BinnedLocalTargetProfile
from robotino_fleet.navigation.controller import (
    NavigationControlOutput,
    handover_alignment_output,
)
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle


class ForwardTurnMotionController:
    """Map continuous throttle and turn actions to non-holonomic motion."""

    def __init__(
        self,
        motion: MotionSettings,
        profile: BinnedLocalTargetProfile,
    ) -> None:
        """Bind physical speed limits and a forward-turn policy profile.

        motion supplies Robotino command limits; profile supplies the
        policy action contract and handover behavior. A different action mode
        is rejected before any commands can be produced.

        Args:
            motion: Calibrated Robotino speed and acceleration limits.
            profile: Saved policy contract defining observations and actions.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """
        if profile.action_mode != "forward-turn":
            raise ValueError("Forward-turn controller requires a forward-turn profile")
        self.motion = motion
        self.profile = profile

    def _speed_limit(
        self,
        distance_m: float,
        position_tolerance_m: float,
    ) -> float:
        """Return the forward-speed cap at distance_m from the goal.

        position_tolerance_m marks the handover boundary. Inside the
        slowdown radius, the cap tapers toward the profile's crawl speed;
        outside it, the physical forward maximum is returned.

        Args:
            distance_m: Distance in meters.
            position_tolerance_m: Maximum accepted position error in meters.

        Returns:
            float: The forward-speed cap at distance_m from the goal.
        """
        maximum = self.motion.forward_max_mps
        if distance_m >= self.profile.handover_slowdown_radius_m:
            return maximum
        remaining = max(0.0, distance_m - position_tolerance_m)
        span = max(
            1e-9,
            self.profile.handover_slowdown_radius_m - position_tolerance_m,
        )
        fraction = min(1.0, remaining / span)
        minimum = min(self.profile.handover_min_speed_mps, maximum)
        return minimum + (maximum - minimum) * fraction

    def command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        action: Sequence[float],
        *,
        handover_active: bool = False,
        policy_id: str | None = None,
        inference_ms: float | None = None,
    ) -> NavigationControlOutput:
        """Turn one policy action into the next body-frame drive command.

        robot and goal determine distance and lifecycle; action is
        normalized [drive, turn]. Negative drive does not request reverse
        motion. handover_active keeps deterministic alignment in control
        after positional arrival. policy_id and inference_ms are
        carried into the returned output for fleet telemetry. Invalid action
        shapes or nonfinite values raise ValueError.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            action: Normalized policy action for the controlled Robotino.
            handover_active: Whether position tolerance has triggered final goal handover.
            policy_id: Identifier of the policy producing the navigation output.
            inference_ms: Measured model inference time in milliseconds.

        Returns:
            NavigationControlOutput: Body-frame drive command scaled from the policy action.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if len(action) != 2:
            raise ValueError("Forward-turn action must contain [drive, turn]")
        drive, turn = (float(value) for value in action)
        if not math.isfinite(drive) or not math.isfinite(turn):
            raise ValueError("Forward-turn action must be finite")
        drive = max(-1.0, min(1.0, drive))
        turn = max(-1.0, min(1.0, turn))
        distance = robot.point.distance_to(goal.point)
        if handover_active or distance <= goal.position_tolerance_m:
            return handover_alignment_output(
                robot,
                goal,
                rotation_max_rps=self.motion.rotation_max_rps,
                inference_ms=inference_ms,
                policy_id=policy_id,
            )

        vx = 0.0
        if drive > 0.0:
            limit = self._speed_limit(distance, goal.position_tolerance_m)
            minimum = min(self.profile.handover_min_speed_mps, limit)
            vx = minimum + (limit - minimum) * drive
        return NavigationControlOutput(
            VelocityCommand(
                vx=vx,
                vy=0.0,
                omega=turn * self.motion.rotation_max_rps,
            ),
            NavigationLifecycle.APPROACHING
            if distance <= self.profile.approach_radius_m
            else NavigationLifecycle.NAVIGATING,
            inference_ms=inference_ms,
            policy_id=policy_id,
        )
