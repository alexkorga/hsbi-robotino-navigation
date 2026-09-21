"""Policy boundary plus a deliberately simple non-avoiding baseline."""

import math
from dataclasses import dataclass
from typing import Mapping, Protocol

from robotino_fleet.config import (
    HANDOVER_MIN_ROTATION_RPS,
    HANDOVER_NEAR_HEADING_ERROR_RAD,
    MotionSettings,
)
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.maps.transforms import normalize_angle
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle


@dataclass(frozen=True)
class NavigationControlOutput:
    command: VelocityCommand
    state: NavigationLifecycle
    arrived: bool = False
    reason: str | None = None
    inference_ms: float | None = None
    policy_id: str | None = None


class NavigationController(Protocol):
    """Policy interface shared by simulation, shadow, and live runtimes.

    Implementations may batch inference, but must return one body-frame command
    and lifecycle state for every active Robotino order.
    """

    policy_id: str

    def reset(self, robot_id: str) -> None:
        """Discard policy memory and held actions for robot_id after an order.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        ...

    def compute_command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        delta_s: float,
        now_s: float,
    ) -> NavigationControlOutput:
        """Return one command for robot pursuing goal.

        delta_s is the elapsed control step; now_s is the fleet clock
        used for policy timing and sensor freshness. The output carries the
        proposed velocity, lifecycle state, and optional inference diagnostics.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            NavigationControlOutput: One command for robot pursuing goal.
        """

        ...

    def compute_commands(
        self,
        robots: Mapping[str, RobotState],
        goals: Mapping[str, NavigationGoal],
        *,
        delta_s: float,
        now_s: float,
    ) -> dict[str, NavigationControlOutput]:
        """Return outputs keyed by Robotino ID for matching robots/goals.

        delta_s and now_s have the same meaning as in compute_command;
        batching allows one inference call for robots due at the same time.

        Args:
            robots: Robotino states in the shared world.
            goals: Navigation targets for the active agents.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: Outputs keyed by Robotino ID for matching
                robots/goals.
        """

        ...


def handover_alignment_output(
    robot: RobotState,
    goal: NavigationGoal,
    *,
    rotation_max_rps: float,
    policy_id: str | None = None,
    inference_ms: float | None = None,
) -> NavigationControlOutput:
    """Finish a reached goal with a deterministic heading command.

    robot supplies current heading; rotation_max_rps caps output and
    the shared handover minimum slows the final approach. policy_id and
    inference_ms are passed through for telemetry. Returns ALIGNING while
    outside heading tolerance, otherwise ARRIVED or READY_TO_DOCK with zero
    velocity. A goal without heading completes immediately.

    Args:
        robot: Robotino state used by this operation.
        goal: Assigned navigation target.
        rotation_max_rps: Maximum final-handover turn rate in radians per second.
        policy_id: Identifier of the policy producing the navigation output.
        inference_ms: Measured model inference time in milliseconds.

    Returns:
        NavigationControlOutput: Alignment command, or None when handover is not active.
    """

    if goal.heading_rad is None:
        return NavigationControlOutput(
            VelocityCommand(),
            NavigationLifecycle.READY_TO_DOCK
            if goal.docking_handoff
            else NavigationLifecycle.ARRIVED,
            True,
            inference_ms=inference_ms,
            policy_id=policy_id,
        )
    error = normalize_angle(goal.heading_rad - robot.heading_rad)
    magnitude = abs(error)
    if magnitude <= goal.heading_tolerance_rad:
        return NavigationControlOutput(
            VelocityCommand(),
            NavigationLifecycle.READY_TO_DOCK
            if goal.docking_handoff
            else NavigationLifecycle.ARRIVED,
            True,
            inference_ms=inference_ms,
            policy_id=policy_id,
        )
    minimum = min(HANDOVER_MIN_ROTATION_RPS, rotation_max_rps)
    span = max(1e-9, math.pi - HANDOVER_NEAR_HEADING_ERROR_RAD)
    fraction = min(
        1.0,
        max(0.0, magnitude - HANDOVER_NEAR_HEADING_ERROR_RAD) / span,
    )
    speed = minimum + (rotation_max_rps - minimum) * fraction
    return NavigationControlOutput(
        VelocityCommand(omega=math.copysign(speed, error)),
        NavigationLifecycle.ALIGNING,
        inference_ms=inference_ms,
        policy_id=policy_id,
    )


class DirectGoalController:
    """Free-space baseline for smoke tests; it intentionally does not avoid."""

    policy_id = "direct-goal-baseline"

    def __init__(self, motion: MotionSettings) -> None:
        """Use motion speed limits for a non-avoiding smoke-test policy.

        Args:
            motion: Calibrated Robotino speed and acceleration limits.
        """

        self.motion = motion
        self._handover_robot_ids: set[str] = set()

    def reset(self, robot_id: str) -> None:
        """Forget whether robot_id already entered arrival handover.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        self._handover_robot_ids.discard(robot_id)

    def compute_command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        delta_s: float,
        now_s: float,
    ) -> NavigationControlOutput:
        """Drive robot directly toward goal without obstacle avoidance.

        delta_s and now_s are accepted for the shared controller
        interface but unused here. Returns a map-to-body transformed velocity
        or the deterministic arrival/heading-handover output.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            NavigationControlOutput: Body-frame command for direct goal tracking.
        """

        del delta_s, now_s
        dx = goal.point.x - robot.x
        dy = goal.point.y - robot.y
        distance = math.hypot(dx, dy)
        if robot.id in self._handover_robot_ids or distance <= goal.position_tolerance_m:
            self._handover_robot_ids.add(robot.id)
            return handover_alignment_output(
                robot,
                goal,
                rotation_max_rps=self.motion.rotation_max_rps,
                policy_id=self.policy_id,
            )
        cosine = math.cos(robot.heading_rad)
        sine = math.sin(robot.heading_rad)
        speed = min(self.motion.forward_max_mps, math.sqrt(0.6 * distance))
        map_vx = speed * dx / distance
        map_vy = speed * dy / distance
        return NavigationControlOutput(
            VelocityCommand(
                cosine * map_vx + sine * map_vy,
                -sine * map_vx + cosine * map_vy,
                0.0,
            ),
            (
                NavigationLifecycle.APPROACHING
                if distance <= 0.5
                else NavigationLifecycle.NAVIGATING
            ),
            policy_id=self.policy_id,
        )

    def compute_commands(
        self,
        robots: Mapping[str, RobotState],
        goals: Mapping[str, NavigationGoal],
        *,
        delta_s: float,
        now_s: float,
    ) -> dict[str, NavigationControlOutput]:
        """Return direct goal commands for IDs in robots/goals.

        delta_s and now_s are forwarded unchanged; the result maps each
        Robotino ID to its individual command and lifecycle state.

        Args:
            robots: Robotino states in the shared world.
            goals: Navigation targets for the active agents.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: Direct goal commands for IDs in robots/goals.
        """

        return {
            robot_id: self.compute_command(
                robot,
                goals[robot_id],
                delta_s=delta_s,
                now_s=now_s,
            )
            for robot_id, robot in robots.items()
        }
