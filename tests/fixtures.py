"""Stable test settings that do not depend on laboratory calibration values."""

from dataclasses import replace

from robotino_fleet.config import MotionSettings, ProjectSettings, load_settings


TEST_MOTION = MotionSettings(
    forward_max_mps=0.5,
    sideways_max_mps=0.5,
    rotation_max_rps=0.75,
    forward_acceleration_mps2=0.2682,
    sideways_acceleration_mps2=0.2249,
    forward_deceleration_mps2=0.35,
    sideways_deceleration_mps2=0.35,
    rotation_acceleration_rps2=0.9191,
    rotation_deceleration_rps2=2.6729,
    forward_deadband_mps=0.01,
    sideways_deadband_mps=0.01,
    rotation_deadband_rps=0.04,
    command_delay_s=0.2164,
    odometry_noise_mps=0.0001,
)


def fixed_project_settings(
    *, motion: MotionSettings = TEST_MOTION
) -> ProjectSettings:
    """Return project settings with explicit motion values for unit tests.

    Args:
        motion: Calibrated Robotino speed and acceleration limits.

    Returns:
        ProjectSettings: Project settings with explicit motion values for unit tests.
    """

    return replace(load_settings(), motion=motion)
