"""Train the four-frame wide temporal CNN from scratch."""

import argparse
from dataclasses import dataclass
import faulthandler
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import ConstantSchedule, update_learning_rate
from stable_baselines3.common.vec_env import VecMonitor
from torch import nn
import torch

from common import (
    TrainingProgressCallback,
    project_path,
    resolve_torch_device,
    trainable_parameter_count,
)
from robotino_fleet.config import load_settings
from robotino_fleet.learning.config import LearningEnvironmentSettings, RewardSettings
from robotino_fleet.learning.extractors import WideTemporalBinnedLocalTargetExtractor
from robotino_fleet.learning.profile import (
    BinnedLocalTargetProfile,
    load_policy_profile,
    write_policy_profile,
)
from robotino_fleet.learning.shared_policy_vec_env import SharedPolicyVecEnv


EXPERIMENT_NAME = "shared_multi_robot_cnn_temporal_wide"
MODEL_PATH = Path("models/shared_multi_robot_cnn_temporal_wide.zip")

PROFILE = BinnedLocalTargetProfile(
    name="binned-local-target-temporal-v1",
    schema_version=2,
    action_mode="local-target",
    temporal_frames=4,
    recurrent_hidden_size=0,
    lidar_memory_duration_s=4.0,
    lidar_bin_count=48,
    lidar_clip_distance_m=5.0,
    lidar_percentile=0.10,
    lidar_angle_min_rad=-1.9198628664016724,
    lidar_angle_max_rad=1.9285876750946045,
    proximity_sensor_count=9,
    proximity_clear_distance_m=0.410,
    observation_goal_distance_m=20.0,
    observation_forward_mps=0.5,
    observation_sideways_mps=0.5,
    observation_rotation_rps=0.75,
    policy_period_s=0.20,
    control_period_s=0.05,
    local_target_lookahead_m=0.75,
    translation_kp=1.6,
    approach_radius_m=0.50,
    approach_max_speed_mps=0.18,
    handover_slowdown_radius_m=1.0,
    handover_min_speed_mps=0.10,
    heading_kp=1.5,
)

REWARDS = RewardSettings(
    progress=4.0,
    progress_mode="step",
    heading_weighted_progress=False,
    success=60.0,
    collision=-120.0,
    timeout=0.0,
    proximity=-1.2,
    proximity_threshold_m=0.30,
    time=-0.01,
    smoothness=-0.08,
    control_effort=-0.01,
    lateral_motion=0.0,
    reverse_motion=0.0,
    misaligned_translation=0.0,
    aligned_forward_motion=0.0,
    heading_progress=0.0,
    heading_alignment=0.0,
    turning_translation=0.0,
    turning_translation_min_speed_mps=0.10,
    turning_translation_start_rps=0.20,
    turning_translation_full_rps=0.45,
    approach_heading=0.10,
    approach_radius_m=0.50,
)


@dataclass(frozen=True)
class Phase:
    name: str
    timesteps: int
    factory_obstacles: bool
    maximum_distance_m: float | None
    conflict_probability: float
    scenario_mode: str
    learning_rate: float


PHASES = (
    Phase("open_shared", 250_000, False, 3.0, 0.25, "waypoints", 3e-4),
    Phase("factory_shared", 750_000, True, 4.0, 0.40, "waypoints", 3e-4),
    Phase("factory_conflicts", 1_000_000, True, None, 0.75, "waypoints", 3e-4),
)

POLICY_NET_ARCH = {"pi": [192, 128, 64], "vf": [256, 192, 96]}
ROLLOUT_STEPS = 1024
BATCH_SIZE = 1024
EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.20
ENTROPY_COEFFICIENT = 0.005
TARGET_KL = 0.03
LOG_STD_INIT = -1.0
CHECKPOINT_INTERVAL = 100_000


def environment_settings(seed: int, phase: Phase) -> LearningEnvironmentSettings:
    """Return waypoint-scenario settings for phase and random seed.

    Args:
        seed: Optional seed for reproducible random sampling.
        phase: Named motion-calibration phase.

    Returns:
        LearningEnvironmentSettings: Waypoint-scenario settings for phase and random seed.
    """

    return LearningEnvironmentSettings(
        robot_count=3,
        time_step_s=0.20,
        max_steps=600,
        max_goal_distance_m=20.0,
        minimum_start_goal_distance_m=0.75,
        maximum_start_goal_distance_m=phase.maximum_distance_m,
        require_final_heading=False,
        use_factory_obstacles=phase.factory_obstacles,
        random_obstacles=0,
        sensor_noise=False,
        dynamics_randomization=False,
        other_robot_policy="stationary",
        multi_robot_scenario=True,
        multi_robot_conflict_probability=phase.conflict_probability,
        remove_collided_agents=True,
        scenario_mode=phase.scenario_mode,
        orientation_runway_min_distance_m=2.0,
        orientation_runway_lane_spacing_m=10.0,
        random_seed=seed,
        reward=REWARDS,
        policy_profile=PROFILE,
    )


def argument_parser() -> argparse.ArgumentParser:
    """Return CLI options for training the wide temporal CNN.

    Returns:
        argparse.ArgumentParser: CLI options for training the wide temporal CNN.
    """

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="auto")
    result.add_argument("--worlds", type=int, default=2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--steps-scale", type=float, default=1.0)
    result.add_argument("--startup-batch-size", type=int, default=4)
    result.add_argument("--source", type=Path, default=None)
    result.add_argument("--output", type=Path, default=MODEL_PATH)
    result.add_argument("--run-name", default=None)
    result.add_argument("--no-parallel-worlds", action="store_true")
    return result


def make_environment(
    phase: Phase,
    *,
    worlds: int,
    seed: int,
    parallel_worlds: bool,
    startup_batch_size: int,
) -> VecMonitor:
    """Create a monitored vector environment for phase.

    worlds controls independent factories, seed their initial random
    stream, parallel_worlds worker isolation, and startup_batch_size
    startup concurrency. Return SB3's monitored rollout environment.

    Args:
        phase: Named motion-calibration phase.
        worlds: Independent simulation worlds.
        seed: Optional seed for reproducible random sampling.
        parallel_worlds: Whether to run simulation worlds in separate processes.
        startup_batch_size: Number of worker worlds to start together.

    Returns:
        VecMonitor: Created a monitored vector environment for phase.
    """

    return VecMonitor(
        SharedPolicyVecEnv(
            fleet_settings=load_settings(),
            learning_settings=environment_settings(seed, phase),
            world_count=worlds,
            seed=seed,
            parallel_worlds=parallel_worlds,
            show_startup_progress=True,
            startup_batch_size=startup_batch_size,
        )
    )


def main() -> None:
    """Train the wide temporal CNN through waypoint curriculum phases.

    Parses CLI options, optionally imports source weights, writes per-phase
    checkpoints and the final model/profile, and closes each environment.

    Raises:
        FileNotFoundError: If a required configuration file or policy artifact is absent.
        RuntimeError: If the operation cannot complete in the current runtime state.
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    faulthandler.enable(all_threads=True)
    arguments = argument_parser().parse_args()
    if arguments.worlds < 1:
        raise ValueError("worlds must be positive")
    if arguments.steps_scale <= 0:
        raise ValueError("steps-scale must be positive")
    if arguments.startup_batch_size < 1:
        raise ValueError("startup-batch-size must be positive")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = resolve_torch_device(arguments.device)
    run_name = arguments.run_name or EXPERIMENT_NAME
    destination = project_path(arguments.output).with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = project_path(arguments.source) if arguments.source is not None else None
    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Source model is missing: {source}")
        if load_policy_profile(source) != PROFILE:
            raise ValueError("Source model uses a different policy profile")

    model: PPO | None = None
    total_requested = 0
    for phase_number, phase in enumerate(PHASES, start=1):
        steps = max(1, round(phase.timesteps * arguments.steps_scale))
        total_requested += steps
        agents = arguments.worlds * 3
        print(
            f"\nPreparing curriculum phase {phase_number}/{len(PHASES)}: "
            f"{phase.name} ({arguments.worlds} worlds, {agents} agents)",
            flush=True,
        )
        environment = make_environment(
            phase,
            worlds=arguments.worlds,
            seed=arguments.seed + phase_number * 10_000,
            parallel_worlds=not arguments.no_parallel_worlds,
            startup_batch_size=arguments.startup_batch_size,
        )
        try:
            if model is None:
                model = PPO(
                    policy="MultiInputPolicy",
                    env=environment,
                    policy_kwargs={
                        "features_extractor_class": WideTemporalBinnedLocalTargetExtractor,
                        "share_features_extractor": True,
                        "activation_fn": nn.SiLU,
                        "net_arch": POLICY_NET_ARCH,
                        "log_std_init": LOG_STD_INIT,
                        "ortho_init": False,
                    },
                    learning_rate=phase.learning_rate,
                    n_steps=ROLLOUT_STEPS,
                    batch_size=BATCH_SIZE,
                    n_epochs=EPOCHS,
                    gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA,
                    clip_range=CLIP_RANGE,
                    ent_coef=ENTROPY_COEFFICIENT,
                    target_kl=TARGET_KL,
                    seed=arguments.seed,
                    tensorboard_log=str(
                        project_path(Path("runtime/tensorboard") / run_name)
                    ),
                    device=device,
                    verbose=0,
                )
                print(
                    f"{EXPERIMENT_NAME}: "
                    f"{trainable_parameter_count(model):,} trainable parameters"
                )
                if source is None:
                    print(f"Initialized {EXPERIMENT_NAME} from scratch")
                else:
                    source_model = PPO.load(source, device="cpu")
                    model.policy.load_state_dict(source_model.policy.state_dict())
                    del source_model
                    print(f"Initialized shared policy from {source}")
            else:
                model.set_env(environment)
                model.learning_rate = phase.learning_rate
                model.lr_schedule = ConstantSchedule(phase.learning_rate)
                update_learning_rate(model.policy.optimizer, phase.learning_rate)

            checkpoint_directory = project_path(
                Path("runtime/checkpoints") / run_name / phase.name
            )
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            callbacks = CallbackList(
                [
                    CheckpointCallback(
                        save_freq=max(1, CHECKPOINT_INTERVAL // agents),
                        save_path=str(checkpoint_directory),
                        name_prefix=f"{run_name}_{phase.name}",
                    ),
                    TrainingProgressCallback(
                        experiment_name=f"{run_name}:{phase.name}",
                        total_timesteps=steps,
                    ),
                ]
            )
            print(
                f"Curriculum phase {phase_number}/{len(PHASES)}: {phase.name} "
                f"({steps:,} requested agent steps, {arguments.worlds} worlds, "
                f"{agents} agents, learning rate {phase.learning_rate:g})",
                flush=True,
            )
            model.learn(
                total_timesteps=steps,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=phase_number == 1,
            )
            phase_path = destination.with_name(
                f"{destination.stem}_phase{phase_number}.zip"
            )
            model.save(phase_path)
            write_policy_profile(phase_path, PROFILE)
            print(f"Saved phase model to {phase_path}")
        finally:
            environment.close()

    if model is None:
        raise RuntimeError("No curriculum phases were configured")
    model.save(destination)
    profile_path = write_policy_profile(destination, PROFILE)
    print(f"\nTraining complete: {total_requested:,} requested agent steps")
    print(f"Model saved to {destination}")
    print(f"Policy profile saved to {profile_path}")


if __name__ == "__main__":
    main()
