"""Small project configuration for the shared Robotino fleet."""

import ipaddress
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_BUNDLE = PROJECT_ROOT / "data" / "maps" / "iot_factory_baseline"
# The manual specifies a 45 cm base. The transport structure overhangs the
# chassis, so collision, clearance, and simulated sensor geometry use an
# additional 2.5 cm on every side, for a 50 cm effective diameter.
ROBOT_BASE_DIAMETER_M = 0.45
ROBOT_COLLISION_MARGIN_M = 0.025
ROBOT_RADIUS_M = ROBOT_BASE_DIAMETER_M / 2.0 + ROBOT_COLLISION_MARGIN_M
ROBOT_DIAMETER_M = ROBOT_RADIUS_M * 2.0
DISTANCE_SENSOR_CLEAR_M = 0.410
CONTROL_PERIOD_S = 0.05
GOAL_TOLERANCE_M = 0.10
GOAL_HEADING_TOLERANCE_RAD = 0.10
HANDOVER_NEAR_HEADING_ERROR_RAD = 0.15
HANDOVER_MIN_ROTATION_RPS = 0.10


@dataclass(frozen=True)
class LidarExtrinsic:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class RobotinoConfig:
    ip: str
    api_port: int = 8154
    fallback_port: int = 80
    simulation_start: str = "1"
    lidar: LidarExtrinsic = LidarExtrinsic()

    @property
    def id(self) -> str:
        """Return the IP as the Robotino identity used by the project.

        Returns:
            str: The IP as the Robotino identity used by the project.
        """

        return self.ip

    @property
    def preferred_url(self) -> str:
        """Return this Robotino's advanced API origin, including its port.

        Returns:
            str: This Robotino's advanced API origin, including its port.
        """

        return _base_url(self.ip, self.api_port)

    @property
    def fallback_url(self) -> str:
        """Return this Robotino's base API origin for core endpoints.

        Returns:
            str: This Robotino's base API origin for core endpoints.
        """

        return _base_url(self.ip, self.fallback_port)


@dataclass(frozen=True)
class MotionSettings:
    forward_max_mps: float
    sideways_max_mps: float
    rotation_max_rps: float
    forward_acceleration_mps2: float
    sideways_acceleration_mps2: float
    forward_deceleration_mps2: float
    sideways_deceleration_mps2: float
    rotation_acceleration_rps2: float
    rotation_deceleration_rps2: float
    forward_deadband_mps: float
    sideways_deadband_mps: float
    rotation_deadband_rps: float
    command_delay_s: float
    odometry_noise_mps: float


@dataclass(frozen=True)
class LocalizationSettings:
    x_m: float
    y_m: float
    theta_rad: float


@dataclass(frozen=True)
class ProjectSettings:
    project_root: Path
    map_bundle: Path
    robotinos: tuple[RobotinoConfig, ...]
    motion: MotionSettings
    localization: LocalizationSettings | None


def _base_url(ip: str, port: int) -> str:
    """Return the HTTP origin for ip and port, omitting the default 80.

    Args:
        ip: Robotino IP address.
        port: TCP port of the Robotino HTTP API.

    Returns:
        str: The HTTP origin for ip and port, omitting the default 80.
    """

    return f"http://{ip}" if port == 80 else f"http://{ip}:{port}"


def _yaml(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Read a YAML mapping from path.

    An absent optional file returns an empty mapping; absent required files,
    or a non-mapping YAML root, raise instead of silently changing settings.

    Args:
        path: Path to the YAML configuration or map file.
        required: Whether a missing file is an error.

    Returns:
        dict[str, Any]: Parsed mapping; empty when an optional file is absent.

    Raises:
        FileNotFoundError: If a required configuration file or policy artifact is absent.
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def _number(raw: dict[str, Any], key: str, *, positive: bool = False) -> float:
    """Return finite numeric raw[key] as a float.

    positive additionally rejects zero and negative values. Invalid or
    missing fields raise ValueError with the field name for diagnostics.

    Args:
        raw: Configuration mapping containing the field to validate.
        key: Name of the field to read or validate.
        positive: Whether the numeric value must also be greater than zero.

    Returns:
        float: Finite numeric raw[key] as a float.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{key} must be {qualifier}")
    return result


def _port(value: object, field: str) -> int:
    """Validate value as a TCP port and return it as an integer.

    field names the configuration entry in the error for invalid values.

    Args:
        value: Candidate integer TCP port.
        field: Configuration field name used in validation errors.

    Returns:
        int: Validated TCP port number.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{field} must be an integer port")
    return value


def _load_robotinos(path: Path) -> tuple[RobotinoConfig, ...]:
    """Return unique, validated Robotino definitions from the YAML at path.

    Each IP becomes the runtime identity; API ports, simulation starts, and
    LiDAR offsets travel with that identity into every execution mode.

    Args:
        path: Path to the YAML configuration or map file.

    Returns:
        tuple[RobotinoConfig, ...]: Unique, validated Robotino definitions from the YAML at
            path.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    data = _yaml(path)
    raw_items = data.get("robotinos")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"robotinos must be a non-empty list in {path}")
    result: list[RobotinoConfig] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"robotinos[{index}] must be an object")
        ip = str(raw.get("ip", "")).strip()
        try:
            ipaddress.ip_address(ip)
        except ValueError as error:
            raise ValueError(f"robotinos[{index}].ip is invalid: {ip!r}") from error
        if ip in seen:
            raise ValueError(f"Duplicate Robotino IP {ip}")
        seen.add(ip)
        lidar_raw = raw.get("lidar", {})
        if not isinstance(lidar_raw, dict):
            raise ValueError(f"robotinos[{index}].lidar must be an object")
        lidar = LidarExtrinsic(
            x_m=float(lidar_raw.get("x_m", 0.0)),
            y_m=float(lidar_raw.get("y_m", 0.0)),
            yaw_rad=float(lidar_raw.get("yaw_rad", 0.0)),
        )
        if not all(math.isfinite(value) for value in (lidar.x_m, lidar.y_m, lidar.yaw_rad)):
            raise ValueError(f"robotinos[{index}].lidar values must be finite")
        start = str(raw.get("simulation_start", "")).strip()
        if not start:
            raise ValueError(f"robotinos[{index}].simulation_start is required")
        result.append(
            RobotinoConfig(
                ip=ip,
                api_port=_port(raw.get("api_port", 8154), f"{ip}.api_port"),
                fallback_port=_port(
                    raw.get("fallback_port", 80), f"{ip}.fallback_port"
                ),
                simulation_start=start,
                lidar=lidar,
            )
        )
    return tuple(result)


def _load_motion(path: Path) -> MotionSettings:
    """Return calibrated speed, acceleration, and delay settings from path.

    Speed and rate limits must be positive; deadbands, delay, and noise may be
    zero but cannot be negative. Malformed fields raise ValueError.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        MotionSettings: Calibrated speed, acceleration, and delay settings from path.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    raw = _yaml(path)
    settings = MotionSettings(
        forward_max_mps=_number(raw, "forward_max_mps", positive=True),
        sideways_max_mps=_number(raw, "sideways_max_mps", positive=True),
        rotation_max_rps=_number(raw, "rotation_max_rps", positive=True),
        forward_acceleration_mps2=_number(
            raw, "forward_acceleration_mps2", positive=True
        ),
        sideways_acceleration_mps2=_number(
            raw, "sideways_acceleration_mps2", positive=True
        ),
        forward_deceleration_mps2=_number(
            raw, "forward_deceleration_mps2", positive=True
        ),
        sideways_deceleration_mps2=_number(
            raw, "sideways_deceleration_mps2", positive=True
        ),
        rotation_acceleration_rps2=_number(
            raw, "rotation_acceleration_rps2", positive=True
        ),
        rotation_deceleration_rps2=_number(
            raw, "rotation_deceleration_rps2", positive=True
        ),
        forward_deadband_mps=_number(raw, "forward_deadband_mps"),
        sideways_deadband_mps=_number(raw, "sideways_deadband_mps"),
        rotation_deadband_rps=_number(raw, "rotation_deadband_rps"),
        command_delay_s=_number(raw, "command_delay_s"),
        odometry_noise_mps=_number(raw, "odometry_noise_mps"),
    )
    if min(
        settings.forward_deadband_mps,
        settings.sideways_deadband_mps,
        settings.rotation_deadband_rps,
        settings.command_delay_s,
        settings.odometry_noise_mps,
    ) < 0:
        raise ValueError("Deadband, delay, and noise values cannot be negative")
    return settings


def _load_localization(path: Path) -> LocalizationSettings | None:
    """Read the indoor-tracking-to-map transform from path if available.

    An absent file or transform returns None, which keeps physical poses
    invalid for autonomous control; nonfinite transform values raise.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        LocalizationSettings | None: Parsed the indoor-tracking-to-map transform from path
            if available.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    raw = _yaml(path, required=False)
    transform = raw.get("map_from_indoor_tracking")
    if transform is None:
        return None
    if not isinstance(transform, dict):
        raise ValueError("map_from_indoor_tracking must be an object")
    values = LocalizationSettings(
        x_m=float(transform.get("x_m", 0.0)),
        y_m=float(transform.get("y_m", 0.0)),
        theta_rad=float(transform.get("theta_rad", 0.0)),
    )
    if not all(math.isfinite(value) for value in (values.x_m, values.y_m, values.theta_rad)):
        raise ValueError("Localization transform must contain finite values")
    return values


def load_settings(project_root: Path | None = None) -> ProjectSettings:
    """Assemble the settings shared by simulation, shadow, and live runtimes.

    project_root overrides the source-tree root for tests or another
    checkout. Returns map paths plus Robotino, motion, and optional localization
    settings; invalid required files raise during loading.

    Args:
        project_root: Project root to load, or the default source-tree root.

    Returns:
        ProjectSettings: Project settings assembled from the shared configuration files.
    """

    root = (project_root or PROJECT_ROOT).resolve()
    return ProjectSettings(
        project_root=root,
        map_bundle=root / "data" / "maps" / "iot_factory_baseline",
        robotinos=_load_robotinos(root / "config" / "robotinos.yaml"),
        motion=_load_motion(root / "config" / "motion.yaml"),
        localization=_load_localization(root / "config" / "localization.yaml"),
    )
