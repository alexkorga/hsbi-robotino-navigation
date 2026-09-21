"""Explicit planar transforms between indoor tracking and the map frame."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robotino_fleet.domain.models import Point


def normalize_angle(angle_rad: float) -> float:
    """Return angle_rad wrapped to the equivalent angle in [-π, π].

    Args:
        angle_rad: Input angle in radians before wrapping to [-pi, pi].

    Returns:
        float: Angle_rad wrapped to the equivalent angle in [-π, π].
    """

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class PlanarTransform:
    """Rigid rotation and translation from one 2-D frame to another."""

    x_m: float = 0.0
    y_m: float = 0.0
    theta_rad: float = 0.0

    def apply(self, point: Point, heading_rad: float = 0.0) -> tuple[Point, float]:
        """Map source-frame point and heading_rad to destination frame.

        Returns the translated/rotated point and normalized heading. The same
        transform is used after calibration to align indoor tracking with the
        fixed factory map.

        Args:
            point: Position in the coordinate frame described above.
            heading_rad: Heading in the source coordinate frame, in radians.

        Returns:
            tuple[Point, float]: Point and heading transformed into the destination frame.
        """

        cosine = math.cos(self.theta_rad)
        sine = math.sin(self.theta_rad)
        transformed = Point(
            self.x_m + cosine * point.x - sine * point.y,
            self.y_m + sine * point.x + cosine * point.y,
        )
        return transformed, normalize_angle(heading_rad + self.theta_rad)

    def inverse(self) -> "PlanarTransform":
        """Return a transform that reverses this rotation and translation.

        Returns:
            'PlanarTransform': A transform that reverses this rotation and translation.
        """

        cosine = math.cos(self.theta_rad)
        sine = math.sin(self.theta_rad)
        return PlanarTransform(
            x_m=-(cosine * self.x_m + sine * self.y_m),
            y_m=sine * self.x_m - cosine * self.y_m,
            theta_rad=-self.theta_rad,
        )


@dataclass(frozen=True)
class CalibrationPair:
    indoor: Point
    map: Point


@dataclass(frozen=True)
class CalibrationFit:
    transform: PlanarTransform
    residual_rmse_m: float
    maximum_error_m: float
    errors_m: tuple[float, ...]


def fit_rigid_transform(pairs: list[CalibrationPair]) -> CalibrationFit:
    """Fit indoor-tracking coordinates to factory-map coordinates.

    pairs contains corresponding indoor and map points from calibration;
    at least two spatially separated pairs are required. Returns a rigid
    transform plus per-point, maximum, and RMS residual errors in meters so
    the operator can judge whether the alignment is credible.

    Args:
        pairs: Fit indoor-tracking coordinates to factory-map coordinates. pairs contains
            corresponding indoor and map points from calibration; at least two spatially
            separated pairs are required.

    Returns:
        CalibrationFit: Rigid transformation from indoor tracking to map coordinates.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if len(pairs) < 2:
        raise ValueError("At least two calibration pairs are required")
    indoor_center = Point(
        sum(pair.indoor.x for pair in pairs) / len(pairs),
        sum(pair.indoor.y for pair in pairs) / len(pairs),
    )
    map_center = Point(
        sum(pair.map.x for pair in pairs) / len(pairs),
        sum(pair.map.y for pair in pairs) / len(pairs),
    )
    dot = 0.0
    cross = 0.0
    spread = 0.0
    for pair in pairs:
        ax = pair.indoor.x - indoor_center.x
        ay = pair.indoor.y - indoor_center.y
        bx = pair.map.x - map_center.x
        by = pair.map.y - map_center.y
        dot += ax * bx + ay * by
        cross += ax * by - ay * bx
        spread += ax * ax + ay * ay
    if spread <= 1e-12:
        raise ValueError("Calibration points must be spatially separated")
    theta = math.atan2(cross, dot)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    transform = PlanarTransform(
        x_m=map_center.x - (cosine * indoor_center.x - sine * indoor_center.y),
        y_m=map_center.y - (sine * indoor_center.x + cosine * indoor_center.y),
        theta_rad=theta,
    )
    errors = tuple(
        transform.apply(pair.indoor)[0].distance_to(pair.map) for pair in pairs
    )
    return CalibrationFit(
        transform=transform,
        residual_rmse_m=math.sqrt(sum(error * error for error in errors) / len(errors)),
        maximum_error_m=max(errors),
        errors_m=errors,
    )

def load_map_from_indoor_tracking(path: Path) -> PlanarTransform:
    """Return the indoor-tracking-to-map transform stored in YAML at path.

    Missing transform components default to zero; a malformed transform object
    raises ValueError. Physical pose polling applies this result before
    navigation uses the coordinates.

    Args:
        path: Path to the YAML configuration or map file.

    Returns:
        PlanarTransform: The indoor-tracking-to-map transform stored in YAML at path.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    value: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = value.get("map_from_indoor_tracking", {})
    if not isinstance(raw, dict):
        raise ValueError("map_from_indoor_tracking must be an object")
    return PlanarTransform(
        x_m=float(raw.get("x_m", 0.0)),
        y_m=float(raw.get("y_m", 0.0)),
        theta_rad=float(raw.get("theta_rad", 0.0)),
    )
