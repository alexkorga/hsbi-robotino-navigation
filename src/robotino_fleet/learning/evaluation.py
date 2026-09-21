"""Load SB3 or GRU policies and evaluate single-Robotino episodes."""

from pathlib import Path
from statistics import mean
import sys
from typing import Any

import numpy as np
from stable_baselines3.common.utils import check_for_correct_spaces
from tqdm import tqdm

from robotino_fleet.config import ProjectSettings, load_settings
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.gym_env import SingleRobotinoGymEnv
from robotino_fleet.learning.profile import load_policy_profile, profile_path


def load_sb3_model(model_path: Path, *, env=None, device: str = "auto"):
    """Return the PPO policy at model_path on device.

    If env is given, attach it and validate observation/action spaces
    before evaluation or runtime inference.

    Args:
        model_path: Path to the trained policy artifact.
        env: Environment used to validate the policy interface.
        device: PyTorch device for training or inference.

    Returns:
        object: The PPO policy at model_path on device.
    """

    training_directory = Path(__file__).resolve().parents[3] / "training"
    if training_directory.is_dir() and str(training_directory) not in sys.path:
        sys.path.insert(0, str(training_directory))
    from stable_baselines3 import PPO

    model = PPO.load(model_path, env=env, device=device)
    if env is not None:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
    return model


def load_policy_model(model_path: Path, *, env=None, device: str = "auto"):
    """Return the SB3 or GRU policy at model_path on device.

    A recurrent profile sidecar selects the custom GRU loader. env is
    passed to the SB3 loader for space checking when applicable.

    Args:
        model_path: Path to the trained policy artifact.
        env: Environment used to validate the policy interface.
        device: PyTorch device for training or inference.

    Returns:
        object: The SB3 or GRU policy at model_path on device.
    """

    if profile_path(model_path).exists():
        profile = load_policy_profile(model_path)
        if profile.uses_recurrent_memory:
            from robotino_fleet.learning.gru_policy import GruPolicyModel

            return GruPolicyModel.load(model_path, device=device)
    return load_sb3_model(model_path, env=env, device=device)


def evaluate_model(
    model_path: Path,
    *,
    fleet_settings: ProjectSettings | None = None,
    learning_settings: LearningEnvironmentSettings | None = None,
    episodes: int = 10,
    seed: int = 10_000,
    deterministic: bool = True,
    device: str = "auto",
    show_progress: bool = False,
    progress_description: str | None = None,
) -> dict[str, Any]:
    """Evaluate model_path over independently seeded factory episodes.

    Each episode uses a distinct seed starting at seed. The returned metrics
    include aggregate outcomes and individual episode records. The
    environment is closed even if evaluation fails.

    Args:
        model_path: Path to the trained policy artifact.
        fleet_settings: Robotino fleet and physical-motion settings.
        learning_settings: Scenario, observation, and reward settings for learning.
        episodes: Number of independently seeded evaluation episodes.
        seed: Optional seed for reproducible random sampling.
        deterministic: Whether to choose the policy's deterministic action.
        device: PyTorch device for training or inference.
        show_progress: Whether to display a live evaluation progress bar.
        progress_description: Label displayed next to the evaluation progress bar.

    Returns:
        dict[str, Any]: Aggregate evaluation metrics for the selected policy.

    Raises:
        ValueError: If episodes is less than one.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = SingleRobotinoGymEnv(
        fleet_settings=fleet_settings or load_settings(),
        learning_settings=learning_settings or LearningEnvironmentSettings(),
    )
    model = load_policy_model(model_path, env=env, device=device)
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=episodes,
        desc=progress_description or f"evaluate {model_path.stem}",
        unit="episode",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    successes = collisions = 0
    reward_total = 0.0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            recurrent_state = None
            episode_start = np.ones(1, dtype=bool)
            episode_reward = 0.0
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, recurrent_state = model.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
                episode_start[:] = False
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
            success = bool(info.get("success"))
            collision = bool(info.get("collision"))
            successes += int(success)
            collisions += int(collision)
            reward_total += episode_reward
            rows.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "reward": episode_reward,
                    "success": success,
                    "collision": collision,
                    "timeout": bool(info.get("timeout")),
                    "elapsedS": float(info.get("elapsedS", 0.0)),
                    "pathLengthM": float(info.get("pathLengthM", 0.0)),
                    "minimumClearanceM": float(info.get("minimumClearanceM", 0.0)),
                    "policyActionVariation": float(
                        info.get("policyActionVariation", 0.0)
                    ),
                    "meanPolicyActionVariationPerStep": float(
                        info.get("meanPolicyActionVariationPerStep", 0.0)
                    ),
                    "appliedCommandVariation": float(
                        info.get("appliedCommandVariation", 0.0)
                    ),
                    "meanAppliedCommandVariationPerSecond": float(
                        info.get("meanAppliedCommandVariationPerSecond", 0.0)
                    ),
                    "meanAbsOmegaRps": float(info.get("meanAbsOmegaRps", 0.0)),
                    "finalPositionErrorM": float(info.get("distanceToGoalM", 0.0)),
                    "finalHeadingErrorRad": abs(
                        float(info.get("headingErrorRad", 0.0))
                    ),
                }
            )
            completed = episode + 1
            progress.set_postfix(
                success=f"{successes / completed:.0%}",
                collision=f"{collisions / completed:.0%}",
                reward=f"{reward_total / completed:.2f}",
                refresh=False,
            )
            progress.update(1)
    finally:
        progress.close()
        env.close()
    return {
        "model": str(model_path),
        "algorithm": "PPO",
        "episodes": episodes,
        "seed": seed,
        "successRate": mean(float(row["success"]) for row in rows),
        "collisionRate": mean(float(row["collision"]) for row in rows),
        "timeoutRate": mean(float(row["timeout"]) for row in rows),
        "meanReward": mean(float(row["reward"]) for row in rows),
        "meanElapsedS": mean(float(row["elapsedS"]) for row in rows),
        "meanPathLengthM": mean(float(row["pathLengthM"]) for row in rows),
        "minimumClearanceM": min(float(row["minimumClearanceM"]) for row in rows),
        "meanPolicyActionVariation": mean(
            float(row["policyActionVariation"]) for row in rows
        ),
        "meanPolicyActionVariationPerStep": mean(
            float(row["meanPolicyActionVariationPerStep"]) for row in rows
        ),
        "meanAppliedCommandVariation": mean(
            float(row["appliedCommandVariation"]) for row in rows
        ),
        "meanAppliedCommandVariationPerSecond": mean(
            float(row["meanAppliedCommandVariationPerSecond"])
            for row in rows
        ),
        "meanAbsOmegaRps": mean(float(row["meanAbsOmegaRps"]) for row in rows),
        "meanFinalPositionErrorM": mean(
            float(row["finalPositionErrorM"]) for row in rows
        ),
        "meanFinalHeadingErrorRad": mean(
            float(row["finalHeadingErrorRad"]) for row in rows
        ),
        "results": rows,
    }
