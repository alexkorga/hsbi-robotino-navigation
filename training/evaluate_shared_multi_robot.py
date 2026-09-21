"""Evaluate a shared policy with interacting Robotinos in parallel worlds."""

import argparse
import faulthandler
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import project_path, resolve_torch_device
from robotino_fleet.config import load_settings
from robotino_fleet.learning.config import LearningEnvironmentSettings, RewardSettings
from robotino_fleet.learning.evaluation import load_policy_model
from robotino_fleet.learning.profile import load_policy_profile
from robotino_fleet.learning.shared_policy_vec_env import SharedPolicyVecEnv


def main() -> None:
    """Evaluate one shared policy in interacting multi-Robotino worlds.

    Parses episode/world/device settings, runs deterministic inference, then
    writes fleet success, collision, timeout, and reward rates to JSON.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/deliverable/shared_multi_robot_cnn_temporal_wide_corners.zip"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument(
        "--robots-per-world",
        type=int,
        default=3,
        help="Interacting Robotinos in each fleet episode (maximum: configured fleet)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=8,
        help="Independent fleet evaluation worlds",
    )
    parser.add_argument(
        "--startup-batch-size",
        type=int,
        default=4,
        help="World processes initialized concurrently per startup group",
    )
    parser.add_argument(
        "--no-parallel-worlds",
        action="store_true",
        help="Run evaluation worlds serially in the main process",
    )
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        raise ValueError("episodes must be positive")
    if arguments.worlds < 1:
        raise ValueError("worlds must be positive")
    if arguments.robots_per_world < 1:
        raise ValueError("robots-per-world must be positive")
    if arguments.startup_batch_size < 1:
        raise ValueError("startup-batch-size must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = resolve_torch_device(arguments.device, purpose="Evaluation")
    model_path = project_path(arguments.model)
    profile = load_policy_profile(model_path)
    # Initialize CUDA before Windows spawns the simulation workers. Loading a
    # CUDA policy after many PyTorch-importing children exist can terminate the
    # parent in native code without raising a Python exception.
    print("Loading evaluation model...", flush=True)
    model = load_policy_model(model_path, device=device)
    settings = LearningEnvironmentSettings(
        robot_count=arguments.robots_per_world,
        time_step_s=profile.policy_period_s,
        max_steps=600,
        max_goal_distance_m=20.0,
        minimum_start_goal_distance_m=0.75,
        require_final_heading=False,
        use_factory_obstacles=True,
        random_obstacles=0,
        sensor_noise=False,
        dynamics_randomization=False,
        multi_robot_scenario=True,
        multi_robot_conflict_probability=0.50,
        random_seed=arguments.seed,
        reward=RewardSettings(
            progress=4.0,
            success=60.0,
            collision=-120.0,
            timeout=-60.0,
            proximity=-1.2,
            proximity_threshold_m=0.30,
            time=-0.01,
            smoothness=-0.08,
            control_effort=-0.01,
            approach_heading=0.10,
            approach_radius_m=0.50,
        ),
        policy_profile=profile,
    )
    world_count = min(arguments.worlds, arguments.episodes)
    environment = SharedPolicyVecEnv(
        fleet_settings=load_settings(),
        learning_settings=settings,
        world_count=world_count,
        seed=arguments.seed,
        parallel_worlds=not arguments.no_parallel_worlds,
        show_startup_progress=True,
        startup_batch_size=arguments.startup_batch_size,
    )
    agents_per_world = environment.agents_per_world
    totals = {
        "fleetSuccesses": 0,
        "collisions": 0,
        "environmentCollisions": 0,
        "robotinoCollisions": 0,
        "timeouts": 0,
        "individualSuccesses": 0,
    }
    fleet_rewards: list[float] = []
    active_rewards = np.zeros(world_count, dtype=np.float64)
    base_quota, extra = divmod(arguments.episodes, world_count)
    world_quotas = np.asarray(
        [base_quota + int(index < extra) for index in range(world_count)],
        dtype=np.int32,
    )
    world_completed = np.zeros(world_count, dtype=np.int32)
    completed = 0
    observations = environment.reset()
    recurrent_state = None
    episode_starts = np.ones(environment.num_envs, dtype=bool)
    progress = tqdm(
        total=arguments.episodes,
        desc="evaluate shared multi-Robotino",
        unit="episode",
    )
    try:
        while completed < arguments.episodes:
            raw, recurrent_state = model.predict(
                observations,
                state=recurrent_state,
                episode_start=episode_starts,
                deterministic=True,
            )
            observations, rewards, dones, infos = environment.step(
                np.asarray(raw, dtype=np.float32)
            )
            episode_starts = np.asarray(dones, dtype=bool) | np.asarray(
                [not bool(info.get("active", True)) for info in infos],
                dtype=bool,
            )
            for world_index in range(world_count):
                base = world_index * agents_per_world
                stop = base + agents_per_world
                active_rewards[world_index] += float(np.sum(rewards[base:stop]))
                if not bool(dones[base]):
                    continue
                final_infos = infos[base:stop]
                if world_completed[world_index] < world_quotas[world_index]:
                    fleet_rewards.append(float(active_rewards[world_index]))
                    collisions = [
                        info for info in final_infos if info["collision"]
                    ]
                    collision_types = {
                        value
                        for info in collisions
                        for value in info.get("collisionTypes", [])
                    }
                    totals["fleetSuccesses"] += int(
                        any(info["fleetSuccess"] for info in final_infos)
                    )
                    totals["collisions"] += int(bool(collisions))
                    totals["environmentCollisions"] += int(
                        "environment" in collision_types
                    )
                    totals["robotinoCollisions"] += int(
                        "robotino" in collision_types
                    )
                    totals["timeouts"] += int(
                        any(info["timeout"] for info in final_infos)
                    )
                    totals["individualSuccesses"] += sum(
                        int(info["success"]) for info in final_infos
                    )
                    completed += 1
                    world_completed[world_index] += 1
                    progress.update(1)
                active_rewards[world_index] = 0.0
            if completed:
                progress.set_postfix(
                    success=f'{100 * totals["fleetSuccesses"] / completed:.0f}%',
                    collision=f'{100 * totals["collisions"] / completed:.0f}%',
                    refresh=False,
                )
    finally:
        progress.close()
        environment.close()
    result = {
        "model": str(model_path),
        "episodes": arguments.episodes,
        "robotsPerEpisode": agents_per_world,
        "parallelWorlds": world_count,
        "fleetSuccessRate": totals["fleetSuccesses"] / arguments.episodes,
        "collisionRate": totals["collisions"] / arguments.episodes,
        "environmentCollisionRate": totals["environmentCollisions"]
        / arguments.episodes,
        "robotinoCollisionRate": totals["robotinoCollisions"] / arguments.episodes,
        "timeoutRate": totals["timeouts"] / arguments.episodes,
        "individualSuccessRate": totals["individualSuccesses"]
        / (arguments.episodes * agents_per_world),
        "meanFleetReward": sum(fleet_rewards) / len(fleet_rewards),
    }
    destination = model_path.with_suffix(".multi_evaluation.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
