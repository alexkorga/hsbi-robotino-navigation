"""Evaluate the retained wide temporal policy candidates."""

import argparse
import json
from pathlib import Path

from common import PROJECT_ROOT, project_path, resolve_torch_device
from robotino_fleet.learning.config import LearningEnvironmentSettings, RewardSettings
from robotino_fleet.learning.evaluation import evaluate_model
from robotino_fleet.learning.profile import load_policy_profile


DEFAULT_MODEL_PATHS = (
    "models/deliverable/shared_multi_robot_cnn_temporal_wide.zip",
    "models/deliverable/shared_multi_robot_cnn_temporal_wide_corners.zip",
)

EVALUATION_REWARDS = RewardSettings(
    progress=4.0,
    success=60.0,
    collision=-120.0,
    proximity=-1.2,
    proximity_threshold_m=0.30,
    time=-0.01,
    smoothness=-0.08,
    control_effort=-0.01,
    approach_heading=0.10,
    approach_radius_m=0.50,
)


def parser() -> argparse.ArgumentParser:
    """Return CLI options for model paths, episode count, seed, and device.

    Returns:
        argparse.ArgumentParser: CLI options for model paths, episode count, seed, and
            device.
    """

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("models", nargs="*", default=DEFAULT_MODEL_PATHS)
    result.add_argument("--episodes", type=int, default=500)
    result.add_argument("--seed", type=int, default=20_000)
    result.add_argument("--device", default="cpu")
    return result


def main() -> None:
    """Evaluate selected models individually and save JSON summaries.

    Each model's profile sets its observation contract. Prints outcome and
    control-quality rates after writing a sibling .evaluation.json file.

    Raises:
        FileNotFoundError: If a required configuration file or policy artifact is absent.
    """

    arguments = parser().parse_args()
    device = resolve_torch_device(arguments.device, purpose="Evaluation")
    reports: list[dict[str, object]] = []
    for value in arguments.models:
        model_path = project_path(value)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        profile = load_policy_profile(model_path)
        environment = LearningEnvironmentSettings(
            robot_count=1,
            time_step_s=profile.policy_period_s,
            max_steps=600,
            max_goal_distance_m=20.0,
            minimum_start_goal_distance_m=0.75,
            maximum_start_goal_distance_m=None,
            require_final_heading=False,
            use_factory_obstacles=True,
            random_obstacles=0,
            sensor_noise=False,
            dynamics_randomization=False,
            random_seed=arguments.seed,
            reward=EVALUATION_REWARDS,
            policy_profile=profile,
        )
        report = evaluate_model(
            model_path,
            learning_settings=environment,
            episodes=arguments.episodes,
            seed=arguments.seed,
            deterministic=True,
            device=device,
            show_progress=True,
            progress_description=f"evaluate {model_path.stem}",
        )
        output = model_path.with_suffix(".evaluation.json")
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["evaluationFile"] = str(output.relative_to(PROJECT_ROOT))
        reports.append(report)

    print(
        f"{'Model':<38} {'Success':>8} {'Collision':>10} {'Timeout':>8} "
        f"{'Clearance':>10} {'Cmd variation/s':>16}"
    )
    print("-" * 94)
    for report in reports:
        print(
            f"{Path(str(report['model'])).name:<38} "
            f"{float(report['successRate']):>7.1%} "
            f"{float(report['collisionRate']):>9.1%} "
            f"{float(report['timeoutRate']):>7.1%} "
            f"{float(report['minimumClearanceM']):>9.3f}m "
            f"{float(report['meanAppliedCommandVariationPerSecond']):>16.3f}"
        )
        print(f"  saved {report['evaluationFile']}")


if __name__ == "__main__":
    main()
