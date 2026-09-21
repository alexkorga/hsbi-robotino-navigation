"""Versioned contracts shared by local-target training and runtime inference."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BinnedLocalTargetProfile:
    """Everything outside PPO weights that changes policy input or output meaning."""

    name: str = "binned-local-target-v1"
    schema_version: int = 1
    action_mode: str = "local-target"
    temporal_frames: int = 1
    lidar_memory_duration_s: float = 4.0
    lidar_bin_count: int = 48
    lidar_clip_distance_m: float = 5.0
    lidar_percentile: float = 0.10
    lidar_angle_min_rad: float = -1.9198628664016724
    lidar_angle_max_rad: float = 1.9285876750946045
    proximity_sensor_count: int = 9
    proximity_clear_distance_m: float = 0.410
    observation_goal_distance_m: float = 20.0
    observation_forward_mps: float = 0.5
    observation_sideways_mps: float = 0.5
    observation_rotation_rps: float = 0.75
    policy_period_s: float = 0.20
    control_period_s: float = 0.05
    local_target_lookahead_m: float = 0.75
    translation_kp: float = 1.6
    approach_radius_m: float = 0.50
    approach_max_speed_mps: float = 0.18
    handover_slowdown_radius_m: float = 1.0
    handover_min_speed_mps: float = 0.10
    heading_kp: float = 1.5
    recurrent_hidden_size: int = 0

    def __post_init__(self) -> None:
        """Reject inconsistent policy schema, sensor, action, or timing fields.

        This validation runs after dataclass construction so a saved profile
        cannot silently change observation/action meaning at inference time.
        Nothing is returned; invalid contracts raise ValueError.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        supported_contracts = {
            ("binned-local-target-v1", 1),
            ("binned-local-target-temporal-v1", 2),
            ("binned-forward-turn-temporal-v1", 3),
            ("binned-local-target-360-temporal-v1", 4),
            ("binned-local-target-yaw-temporal-v1", 5),
            ("binned-local-target-gru-v1", 7),
        }
        if (self.name, self.schema_version) not in supported_contracts:
            raise ValueError("Unsupported local-target policy profile")
        if not isinstance(self.temporal_frames, int) or self.temporal_frames < 1:
            raise ValueError("temporal_frames must be a positive integer")
        if self.schema_version == 1 and self.temporal_frames != 1:
            raise ValueError("The v1 policy contract does not support temporal frames")
        if self.schema_version == 2 and self.temporal_frames < 2:
            raise ValueError("The temporal policy contract requires multiple frames")
        expected_action_mode = {
            3: "forward-turn",
            5: "local-target-yaw",
        }.get(self.schema_version, "local-target")
        if self.action_mode != expected_action_mode:
            raise ValueError(
                f"Profile schema {self.schema_version} requires "
                f"action_mode={expected_action_mode!r}"
            )
        if self.schema_version == 3 and self.temporal_frames != 6:
            raise ValueError("The forward-turn temporal contract requires six frames")
        if self.schema_version == 4:
            if self.action_mode != "local-target" or self.temporal_frames != 6:
                raise ValueError(
                    "The 360 temporal contract requires local-target actions and six frames"
                )
            if not math.isclose(
                self.lidar_angle_max_rad - self.lidar_angle_min_rad,
                2.0 * math.pi,
                abs_tol=1e-6,
            ):
                raise ValueError("The 360 temporal contract requires a full-circle scan")
        if self.schema_version == 5 and self.temporal_frames != 4:
            raise ValueError(
                "The learned-yaw temporal contract requires four frames"
            )
        if self.schema_version == 7:
            if self.temporal_frames != 1:
                raise ValueError("The GRU contract consumes only the current frame")
            if self.recurrent_hidden_size < 1:
                raise ValueError("The GRU contract requires recurrent memory")
        elif self.recurrent_hidden_size != 0:
            raise ValueError("Only the GRU contract may declare recurrent memory")
        if self.lidar_bin_count < 8:
            raise ValueError("lidar_bin_count must be at least 8")
        if self.proximity_sensor_count < 1:
            raise ValueError("proximity_sensor_count must be positive")
        if not 0.0 <= self.lidar_percentile <= 1.0:
            raise ValueError("lidar_percentile must be between zero and one")
        positive = (
            self.lidar_clip_distance_m,
            self.lidar_memory_duration_s,
            self.proximity_clear_distance_m,
            self.observation_goal_distance_m,
            self.observation_forward_mps,
            self.observation_sideways_mps,
            self.observation_rotation_rps,
            self.policy_period_s,
            self.control_period_s,
            self.local_target_lookahead_m,
            self.translation_kp,
            self.approach_radius_m,
            self.approach_max_speed_mps,
            self.handover_slowdown_radius_m,
            self.handover_min_speed_mps,
            self.heading_kp,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Local-target profile distances, periods, and gains must be positive")
        if not self.lidar_angle_max_rad > self.lidar_angle_min_rad:
            raise ValueError("LiDAR maximum angle must exceed its minimum angle")
        if self.handover_min_speed_mps > min(
            self.observation_forward_mps,
            self.observation_sideways_mps,
        ):
            raise ValueError("Handover minimum speed cannot exceed translation limits")
        ratio = self.policy_period_s / self.control_period_s
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("policy_period_s must be an integer multiple of control_period_s")

    @property
    def control_steps_per_action(self) -> int:
        """Return low-level control ticks per policy inference interval.

        Returns:
            int: Low-level control ticks per policy inference interval.
        """

        return round(self.policy_period_s / self.control_period_s)

    @property
    def uses_pose_corrected_lidar_360(self) -> bool:
        """Return whether this schema requires pose-corrected 360° LiDAR memory.

        Returns:
            bool: Whether this schema requires pose-corrected 360° LiDAR memory.
        """

        return self.schema_version == 4

    @property
    def uses_learned_rotation(self) -> bool:
        """Return whether a third policy action controls rotation.

        Returns:
            bool: Whether a third policy action controls rotation.
        """

        return self.action_mode == "local-target-yaw"

    @property
    def uses_recurrent_memory(self) -> bool:
        """Return whether recurrent hidden state is part of policy inference.

        Returns:
            bool: Whether recurrent hidden state is part of policy inference.
        """

        return self.schema_version == 7

    @property
    def action_dimensions(self) -> int:
        """Return two target offsets or three values when yaw is learned.

        Returns:
            int: Two target offsets or three values when yaw is learned.
        """

        return 3 if self.uses_learned_rotation else 2

    def to_dict(self) -> dict[str, Any]:
        """Return all profile fields for the model's JSON sidecar.

        Returns:
            dict[str, Any]: All profile fields for the model's JSON sidecar.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "BinnedLocalTargetProfile":
        """Construct and validate a policy contract from decoded value.

        Return the immutable profile, or raise ValueError if JSON shape,
        field names, or schema constraints do not match this runtime.

        Args:
            value: Decoded JSON mapping of saved policy-profile fields.

        Returns:
            'BinnedLocalTargetProfile': Validated immutable policy profile.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if not isinstance(value, dict):
            raise ValueError("Policy profile must be a JSON object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid policy profile fields: {error}") from error


def profile_path(model_path: Path) -> Path:
    """Return the JSON sidecar path next to model_path.

    Both SB3 archives and custom GRU artifacts use this profile to preserve
    the observation/action contract across training and live inference.

    Args:
        model_path: Path to the trained policy artifact.

    Returns:
        Path: The JSON sidecar path next to model_path.
    """

    return model_path.with_suffix(".profile.json")


def write_policy_profile(model_path: Path, profile: BinnedLocalTargetProfile) -> Path:
    """Save profile beside model_path and return its sidecar path.

    Args:
        model_path: Path to the trained policy artifact.
        profile: Saved policy contract defining observations and actions.

    Returns:
        Path: Path to the written policy-profile sidecar.
    """

    destination = profile_path(model_path)
    destination.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_policy_profile(model_path: Path) -> BinnedLocalTargetProfile:
    """Return the validated profile beside model_path.

    Missing sidecars raise FileNotFoundError; invalid profile fields raise
    ValueError. Controllers use this before accepting a model.

    Args:
        model_path: Path to the trained policy artifact.

    Returns:
        BinnedLocalTargetProfile: The validated profile beside model_path.

    Raises:
        FileNotFoundError: If a required configuration file or policy artifact is absent.
    """

    source = profile_path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Policy profile is missing: {source}")
    return BinnedLocalTargetProfile.from_dict(
        json.loads(source.read_text(encoding="utf-8"))
    )
