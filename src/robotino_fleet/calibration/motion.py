"""Derive the simulator's motion constants from a guided Robotino run."""

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml

from robotino_fleet.config import MotionSettings


@dataclass(frozen=True)
class MotionSample:
    """One commanded and measured body-velocity sample in a named test phase."""

    time_s: float
    phase: str
    command_vx: float
    command_vy: float
    command_omega: float
    measured_vx: float
    measured_vy: float
    measured_omega: float
    sequence: int


def write_samples(path: Path, samples: list[MotionSample]) -> None:
    """Write samples as raw calibration CSV at path for inspection.

    Args:
        path: Destination file for the generated data or model.
        samples: Recorded motion-calibration samples.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(MotionSample.__dataclass_fields__))
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def _measurement(sample: MotionSample, axis: str) -> float:
    """Return absolute measured speed on axis in sample.

    Args:
        sample: One recorded calibration sample with command and measured velocity.
        axis: Translation or rotation axis being measured.

    Returns:
        float: Absolute measured speed on axis in sample.
    """

    if axis == "forward":
        return abs(sample.measured_vx)
    if axis == "sideways":
        return abs(sample.measured_vy)
    return abs(sample.measured_omega)


def _command(sample: MotionSample, axis: str) -> float:
    """Return absolute commanded speed on axis in sample.

    Args:
        sample: One recorded calibration sample with command and measured velocity.
        axis: Translation or rotation axis being measured.

    Returns:
        float: Absolute commanded speed on axis in sample.
    """

    if axis == "forward":
        return abs(sample.command_vx)
    if axis == "sideways":
        return abs(sample.command_vy)
    return abs(sample.command_omega)


def _phase_matches(phase: str, axis: str) -> bool:
    """Return whether phase belongs to acceleration tests for axis.

    Args:
        phase: Named motion-calibration phase.
        axis: Translation or rotation axis being measured.

    Returns:
        bool: Whether phase belongs to acceleration tests for axis.
    """

    prefixes = {
        "forward": ("translation-forward-", "translation-return-"),
        "sideways": ("translation-left-", "translation-right-return-"),
        "rotation": ("rotation-left-", "rotation-return-"),
    }
    return phase.startswith(prefixes[axis])


def _rates(
    samples: list[MotionSample],
    *,
    axis: str,
    accelerating: bool,
) -> list[float]:
    """Estimate rates from samples for axis in matching phases.

    accelerating selects speed-up or braking transitions. Return positive
    rates in axis units per second; no matching measurements yields an empty
    list for the caller's fallback logic.

    Args:
        samples: Recorded motion-calibration samples.
        axis: Translation or rotation axis being measured.
        accelerating: Select acceleration phases when true, braking phases otherwise.

    Returns:
        list[float]: Positive acceleration or braking estimates from matching samples.
    """

    values: list[float] = []
    for first, second in zip(samples, samples[1:], strict=False):
        dt = second.time_s - first.time_s
        if (
            dt <= 0
            or first.phase != second.phase
            or not _phase_matches(second.phase, axis)
        ):
            continue
        previous = _measurement(first, axis)
        current = _measurement(second, axis)
        commanded = _command(second, axis)
        rate = (current - previous) / dt
        if accelerating and commanded > 0 and rate > 0.01:
            values.append(rate)
        elif not accelerating and commanded == 0 and rate < -0.01:
            values.append(-rate)
    return values


def _robust_rate(values: list[float], fallback: float) -> float:
    """Return the upper-half median of values, or fallback if empty.

    Args:
        values: Measured or computed values supplied to this operation.
        fallback: Rate to use when no valid sample-derived estimate exists.

    Returns:
        float: The upper-half median of values, or fallback if empty.
    """

    if not values:
        return fallback
    ordered = sorted(values)
    # The upper-middle measurement captures useful acceleration without
    # fitting single noisy spikes.
    return statistics.median(ordered[len(ordered) // 2 :])


def _deadband(samples: list[MotionSample], *, axis: str, noise: float) -> float:
    """Estimate the axis command deadband from samples and noise.

    Return the smallest commanded speed whose measured motion exceeds the
    noise threshold, or a conservative axis-specific fallback.

    Args:
        samples: Recorded motion-calibration samples.
        axis: Translation or rotation axis being measured.
        noise: Measured motion-noise threshold used to detect real movement.

    Returns:
        float: Smallest command reliably producing measured motion.
    """

    detected: list[float] = []
    threshold = max(0.003, noise * 3.0)
    by_phase: dict[str, list[MotionSample]] = {}
    for sample in samples:
        if sample.phase.startswith(f"{axis}-deadband-"):
            by_phase.setdefault(sample.phase, []).append(sample)
    for values in by_phase.values():
        measured = statistics.median(_measurement(item, axis) for item in values)
        command = statistics.median(_command(item, axis) for item in values)
        if measured > threshold:
            detected.append(command)
    return min(detected) if detected else (0.05 if axis == "rotation" else 0.02)


def _response_delays(samples: list[MotionSample]) -> list[float]:
    """Return command-to-motion delays for rest transitions in samples.

    Args:
        samples: Recorded motion-calibration samples.

    Returns:
        list[float]: Command-to-motion delays for rest transitions in samples.
    """

    result: list[float] = []
    for index, sample in enumerate(samples):
        command = math.hypot(sample.command_vx, sample.command_vy) + abs(sample.command_omega)
        previous = (
            math.hypot(samples[index - 1].command_vx, samples[index - 1].command_vy)
            + abs(samples[index - 1].command_omega)
            if index
            else 0.0
        )
        if command <= 0 or previous > 0:
            continue
        for later in samples[index:]:
            measured = math.hypot(later.measured_vx, later.measured_vy) + abs(later.measured_omega)
            if measured > 0.01:
                result.append(max(0.0, later.time_s - sample.time_s))
                break
    return result


def derive_motion_settings(samples: list[MotionSample]) -> MotionSettings:
    """Return speed, rate, deadband, delay, and noise settings from samples.

    Uses robust measured estimates where possible and conservative fallbacks
    when a particular maneuver yielded insufficient data.

    Args:
        samples: Recorded motion-calibration samples.

    Returns:
        MotionSettings: Speed, rate, deadband, delay, and noise settings from samples.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if not samples:
        raise ValueError("No calibration samples were recorded")
    stationary = [sample for sample in samples if sample.phase == "stationary"]
    noise_values = [
        math.hypot(sample.measured_vx, sample.measured_vy) for sample in stationary
    ]
    noise = statistics.pstdev(noise_values) if len(noise_values) > 1 else 0.002
    commanded_forward = [abs(sample.command_vx) for sample in samples]
    commanded_sideways = [abs(sample.command_vy) for sample in samples]
    commanded_rotation = [abs(sample.command_omega) for sample in samples]
    delays = _response_delays(samples)
    return MotionSettings(
        forward_max_mps=max(commanded_forward),
        sideways_max_mps=max(commanded_sideways),
        rotation_max_rps=max(commanded_rotation),
        forward_acceleration_mps2=_robust_rate(
            _rates(samples, axis="forward", accelerating=True), 0.30
        ),
        sideways_acceleration_mps2=_robust_rate(
            _rates(samples, axis="sideways", accelerating=True), 0.30
        ),
        forward_deceleration_mps2=_robust_rate(
            _rates(samples, axis="forward", accelerating=False), 0.35
        ),
        sideways_deceleration_mps2=_robust_rate(
            _rates(samples, axis="sideways", accelerating=False), 0.35
        ),
        rotation_acceleration_rps2=_robust_rate(
            _rates(samples, axis="rotation", accelerating=True), 0.70
        ),
        rotation_deceleration_rps2=_robust_rate(
            _rates(samples, axis="rotation", accelerating=False), 0.80
        ),
        forward_deadband_mps=_deadband(
            samples, axis="forward", noise=noise
        ),
        sideways_deadband_mps=_deadband(
            samples, axis="sideways", noise=noise
        ),
        rotation_deadband_rps=_deadband(samples, axis="rotation", noise=noise),
        command_delay_s=statistics.median(delays) if delays else 0.03,
        odometry_noise_mps=max(0.0001, noise),
    )


def write_motion_settings(path: Path, settings: MotionSettings) -> None:
    """Write derived settings to the project's motion YAML at path.

    Args:
        path: Destination file for the generated data or model.
        settings: Configuration settings for this component.
    """

    path.write_text(
        yaml.safe_dump(settings.__dict__, sort_keys=False),
        encoding="utf-8",
    )
