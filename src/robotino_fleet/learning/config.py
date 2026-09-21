"""Environment choices written directly in each training experiment."""

from dataclasses import dataclass, field

from robotino_fleet.learning.profile import BinnedLocalTargetProfile


@dataclass(frozen=True)
class RewardSettings:
    progress: float = 4.0
    progress_mode: str = "step"
    heading_weighted_progress: bool = False
    success: float = 25.0
    collision: float = -30.0
    timeout: float = 0.0
    proximity: float = -0.6
    proximity_threshold_m: float = 0.20
    time: float = -0.01
    smoothness: float = -0.025
    control_effort: float = -0.005
    lateral_motion: float = 0.0
    reverse_motion: float = 0.0
    misaligned_translation: float = 0.0
    aligned_forward_motion: float = 0.0
    heading_progress: float = 0.0
    heading_alignment: float = 0.0
    turning_translation: float = 0.0
    turning_translation_min_speed_mps: float = 0.10
    turning_translation_start_rps: float = 0.20
    turning_translation_full_rps: float = 0.45
    approach_heading: float = 0.15
    approach_radius_m: float = 0.50


@dataclass(frozen=True)
class LearningEnvironmentSettings:
    robot_count: int = 1
    time_step_s: float = 0.10
    max_steps: int = 1200
    max_goal_distance_m: float = 20.0
    minimum_start_goal_distance_m: float = 0.75
    maximum_start_goal_distance_m: float | None = None
    require_final_heading: bool = False
    use_factory_obstacles: bool = True
    random_obstacles: int = 0
    sensor_noise: bool = False
    dynamics_randomization: bool = False
    other_robot_policy: str = "stationary"
    multi_robot_scenario: bool = False
    multi_robot_conflict_probability: float = 0.0
    remove_collided_agents: bool = True
    scenario_mode: str = "waypoints"
    orientation_runway_min_distance_m: float = 2.0
    orientation_runway_lane_spacing_m: float = 10.0
    random_seed: int = 42
    reward: RewardSettings = field(default_factory=RewardSettings)
    policy_profile: BinnedLocalTargetProfile | None = None
