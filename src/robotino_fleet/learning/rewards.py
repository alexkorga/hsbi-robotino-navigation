"""Small, inspectable reward function for local navigation."""

import math
from dataclasses import dataclass
from typing import Sequence

from robotino_fleet.learning.config import RewardSettings


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


def navigation_reward(
    settings: RewardSettings,
    *,
    previous_distance_m: float,
    distance_m: float,
    minimum_clearance_m: float,
    collision: bool,
    success: bool,
    action: Sequence[float],
    previous_action: Sequence[float],
    heading_error_rad: float,
    lateral_speed_mps: float = 0.0,
    forward_speed_mps: float = 0.0,
    angular_speed_rps: float = 0.0,
    previous_navigation_heading_error_rad: float | None = None,
    navigation_heading_error_rad: float | None = None,
) -> RewardResult:
    """Return weighted navigation reward and named diagnostic components.

    Combine goal progress, clearance, terminal outcomes, control effort,
    action change, and heading progress using the configured weights.
    Distances are in meters, headings in radians, and speeds in SI units.

    Args:
        settings: Reward weights, thresholds, and progress mode.
        previous_distance_m: Goal distance before the current control step, in meters.
        distance_m: Current distance from the Robotino to its goal, in meters.
        minimum_clearance_m: Smallest valid obstacle clearance measured this step.
        collision: Whether the step ended in a collision.
        success: Whether the navigation goal was reached.
        action: Normalized policy action for the controlled Robotino.
        previous_action: Previous normalized policy action for temporal context.
        heading_error_rad: Current heading error toward the goal, in radians.
        lateral_speed_mps: Measured sideways speed in meters per second.
        forward_speed_mps: Measured forward speed in meters per second.
        angular_speed_rps: Measured turn rate in radians per second.
        previous_navigation_heading_error_rad: Navigation heading error before the step.
        navigation_heading_error_rad: Navigation heading error after the step.

    Returns:
        RewardResult: Weighted navigation reward and named diagnostic components.

    Raises:
        ValueError: If the configured progress mode is unsupported.
    """

    if settings.progress_mode not in {"step", "best-distance"}:
        raise ValueError(f"Unsupported progress mode: {settings.progress_mode}")
    action_norm = math.sqrt(sum(float(value) ** 2 for value in action))
    change_norm = math.sqrt(
        sum(
            (float(current) - float(previous)) ** 2
            for current, previous in zip(action, previous_action, strict=True)
        )
    )
    proximity_ratio = (
        max(0.0, 1.0 - minimum_clearance_m / settings.proximity_threshold_m)
        if settings.proximity_threshold_m > 0
        else 0.0
    )
    approach_ratio = (
        max(0.0, 1.0 - distance_m / settings.approach_radius_m)
        if settings.approach_radius_m > 0
        else 0.0
    )
    distance_gain = previous_distance_m - distance_m
    if settings.progress_mode == "best-distance":
        distance_gain = max(0.0, distance_gain)
    if settings.heading_weighted_progress:
        distance_gain *= (
            max(0.0, math.cos(navigation_heading_error_rad))
            if navigation_heading_error_rad is not None
            else 0.0
        )
    forward_alignment = (
        max(0.0, math.cos(navigation_heading_error_rad))
        if navigation_heading_error_rad is not None
        else 0.0
    )
    translation_speed = math.hypot(forward_speed_mps, lateral_speed_mps)
    turning_translation = 0.0
    if (
        translation_speed > settings.turning_translation_min_speed_mps
        and abs(angular_speed_rps) > settings.turning_translation_start_rps
    ):
        turn_span = max(
            1e-9,
            settings.turning_translation_full_rps
            - settings.turning_translation_start_rps,
        )
        turn_factor = min(
            1.0,
            (abs(angular_speed_rps) - settings.turning_translation_start_rps)
            / turn_span,
        )
        speed_factor = (
            translation_speed / settings.turning_translation_min_speed_mps
        )
        turning_translation = (
            settings.turning_translation * speed_factor * turn_factor
        )
    heading_gain = 0.0
    if (
        previous_navigation_heading_error_rad is not None
        and navigation_heading_error_rad is not None
    ):
        heading_gain = max(
            0.0,
            abs(previous_navigation_heading_error_rad)
            - abs(navigation_heading_error_rad),
        )
    components = {
        "progress": settings.progress * distance_gain,
        "success": settings.success if success else 0.0,
        "collision": settings.collision if collision else 0.0,
        "proximity": settings.proximity * proximity_ratio,
        "time": settings.time,
        "smoothness": settings.smoothness * change_norm,
        "control_effort": settings.control_effort * action_norm,
        "lateral_motion": settings.lateral_motion * abs(lateral_speed_mps),
        "reverse_motion": settings.reverse_motion * max(0.0, -forward_speed_mps),
        "misaligned_translation": settings.misaligned_translation
        * translation_speed
        * (1.0 - forward_alignment),
        "aligned_forward_motion": settings.aligned_forward_motion
        * max(0.0, forward_speed_mps)
        * forward_alignment,
        "heading_progress": settings.heading_progress * heading_gain,
        "heading_alignment": (
            settings.heading_alignment * math.cos(navigation_heading_error_rad)
            if navigation_heading_error_rad is not None
            else 0.0
        ),
        "turning_translation": turning_translation,
        "approach_heading": settings.approach_heading
        * approach_ratio
        * math.cos(heading_error_rad),
        "timeout": 0.0,
    }
    return RewardResult(sum(components.values()), components)
