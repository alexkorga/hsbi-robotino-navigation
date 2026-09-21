"""Fine-tune the pretrained shared GRU on corner and mixed factory routes."""

import argparse
from dataclasses import dataclass
import faulthandler
from pathlib import Path

import torch
from stable_baselines3.common.vec_env import VecMonitor

from common import project_path, resolve_torch_device
from robotino_fleet.config import load_settings
from robotino_fleet.learning.config import LearningEnvironmentSettings, RewardSettings
from robotino_fleet.learning.gru_policy import GruPolicyArchitecture, GruPolicyModel
from robotino_fleet.learning.gru_ppo import GruPPOTrainer
from robotino_fleet.learning.profile import (
    BinnedLocalTargetProfile,
    load_policy_profile,
    write_policy_profile,
)
from robotino_fleet.learning.shared_policy_vec_env import SharedPolicyVecEnv


EXPERIMENT_NAME = "shared_multi_robot_gru_corners"
MODEL_PATH = Path("models/shared_multi_robot_gru_corners.zip")
SOURCE_PATH = Path("models/shared_multi_robot_gru.zip")

PROFILE = BinnedLocalTargetProfile(
    name="binned-local-target-gru-v1",
    schema_version=7,
    temporal_frames=1,
    recurrent_hidden_size=288,
    action_mode="local-target",
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
    progress_mode="best-distance",
    heading_weighted_progress=False,
    success=60.0,
    collision=-120.0,
    timeout=-60.0,
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
    entropy_coefficient: float


PHASES = (
    Phase("island_corner_focus", 750_000, True, None, 0.0, "island-corners", 1e-4, 0.005),
    Phase("factory_corner_mix", 1_250_000, True, None, 0.50, "island-corner-mix", 5e-5, 0.005),
)

ROLLOUT_STEPS = 256
SEQUENCE_LENGTH = 32
SEQUENCE_BATCH_SIZE = 32
EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.20
VALUE_COEFFICIENT = 0.5
TARGET_KL = 0.03
MAX_GRADIENT_NORM = 0.5
ARCHITECTURE = GruPolicyArchitecture(
    scan_features=192,
    state_hidden=128,
    state_features=96,
    fused_features=256,
    hidden_size=288,
)


def argument_parser() -> argparse.ArgumentParser:
    """Return CLI options for pretrained-GRU corner fine-tuning.

    Returns:
        argparse.ArgumentParser: CLI options for pretrained-GRU corner fine-tuning.
    """

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="auto")
    result.add_argument("--worlds", type=int, default=2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--steps-scale", type=float, default=1.0)
    result.add_argument("--startup-batch-size", type=int, default=4)
    result.add_argument("--source", type=Path, default=SOURCE_PATH)
    result.add_argument("--output", type=Path, default=MODEL_PATH)
    result.add_argument("--run-name", default=None)
    result.add_argument("--no-parallel-worlds", action="store_true")
    return result


def environment_settings(seed: int, phase: Phase) -> LearningEnvironmentSettings:
    """Return corner-scenario settings for phase under seed.

    Args:
        seed: Optional seed for reproducible random sampling.
        phase: Named motion-calibration phase.

    Returns:
        LearningEnvironmentSettings: Corner-scenario settings for phase under seed.
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


def make_environment(
    phase: Phase,
    *,
    worlds: int,
    seed: int,
    parallel_worlds: bool,
    startup_batch_size: int,
) -> VecMonitor:
    """Build monitored corner worlds for phase and worker settings.

    worlds and seed determine rollout worlds, parallel_worlds
    selects subprocesses, and startup_batch_size bounds launch bursts.
    Return the VecMonitor used by the shared GRU trainer.

    Args:
        phase: Named motion-calibration phase.
        worlds: Independent simulation worlds.
        seed: Optional seed for reproducible random sampling.
        parallel_worlds: Whether to run simulation worlds in separate processes.
        startup_batch_size: Number of worker worlds to start together.

    Returns:
        VecMonitor: Constructed monitored corner worlds for phase and worker settings.
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
    """Load the pretrained GRU and fine-tune it on island-corner routes.

    Parses CLI overrides, validates source architecture/profile, writes phase
    checkpoints and the final model/profile, then closes training resources.

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
    source = project_path(arguments.source).with_suffix(".zip")
    if not source.exists():
        raise FileNotFoundError(
            f"Pretrained GRU is missing: {source}. Run "
            "train_shared_multi_robot_gru.py first."
        )
    if load_policy_profile(source) != PROFILE:
        raise ValueError("Pretrained GRU uses a different policy profile")

    run_name = arguments.run_name or EXPERIMENT_NAME
    destination = project_path(arguments.output).with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    trainer: GruPPOTrainer | None = None
    total_requested = 0

    try:
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
                if trainer is None:
                    loaded = GruPolicyModel.load(source, device=device)
                    if loaded.policy.architecture != ARCHITECTURE:
                        raise ValueError("Pretrained GRU uses a different architecture")
                    parameters = sum(
                        value.numel()
                        for value in loaded.policy.parameters()
                        if value.requires_grad
                    )
                    print(f"{EXPERIMENT_NAME}: {parameters:,} trainable parameters")
                    print(f"Initialized shared GRU policy from {source}")
                    trainer = GruPPOTrainer(
                        loaded.policy,
                        environment,
                        device=device,
                        learning_rate=phase.learning_rate,
                        rollout_steps=ROLLOUT_STEPS,
                        sequence_length=SEQUENCE_LENGTH,
                        sequence_batch_size=SEQUENCE_BATCH_SIZE,
                        epochs=EPOCHS,
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
                        clip_range=CLIP_RANGE,
                        entropy_coefficient=phase.entropy_coefficient,
                        value_coefficient=VALUE_COEFFICIENT,
                        target_kl=TARGET_KL,
                        max_gradient_norm=MAX_GRADIENT_NORM,
                        seed=arguments.seed,
                        tensorboard_directory=project_path(
                            Path("runtime/tensorboard") / run_name
                        ),
                    )
                else:
                    trainer.set_environment(environment)
                    trainer.set_learning_rate(phase.learning_rate)
                    trainer.set_entropy_coefficient(phase.entropy_coefficient)

                checkpoint_directory = project_path(
                    Path("runtime/checkpoints") / run_name / phase.name
                )
                print(
                    f"Curriculum phase {phase_number}/{len(PHASES)}: {phase.name} "
                    f"({steps:,} requested agent steps, {arguments.worlds} worlds, "
                    f"{agents} agents, learning rate {phase.learning_rate:g}, "
                    f"entropy {phase.entropy_coefficient:g})",
                    flush=True,
                )
                trainer.learn(
                    steps,
                    description=f"{run_name}:{phase.name}",
                    checkpoint_directory=checkpoint_directory,
                    checkpoint_prefix=f"{run_name}_{phase.name}",
                    checkpoint_profile=PROFILE,
                )
                phase_path = destination.with_name(
                    f"{destination.stem}_phase{phase_number}.zip"
                )
                trainer.save(phase_path)
                write_policy_profile(phase_path, PROFILE)
                print(f"Saved phase model to {phase_path}")
            finally:
                environment.close()

        if trainer is None:
            raise RuntimeError("No GRU fine-tuning phases were configured")
        trainer.save(destination)
        profile_path = write_policy_profile(destination, PROFILE)
        print(f"\nFine-tuning complete: {total_requested:,} requested agent steps")
        print(f"Model saved to {destination}")
        print(f"Policy profile saved to {profile_path}")
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
