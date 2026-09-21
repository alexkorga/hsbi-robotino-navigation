"""Shared environment, device, progress, and path helpers for experiments."""

from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from robotino_fleet.config import load_settings
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.gym_env import SingleRobotinoGymEnv
from robotino_fleet.learning.training_progress import TrainingProgressCallback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Return path resolved relative to the project root if needed.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        Path: Path resolved relative to the project root if needed.
    """

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def make_training_environment(
    settings: LearningEnvironmentSettings,
    *,
    parallel_environments: int,
    seed: int,
    use_subprocesses: bool = False,
):
    """Create SB3 vector environments for single-controlled-Robotino experiments.

    settings configures each world, parallel_environments its count,
    and seed its random stream. use_subprocesses selects spawned
    workers. Return an SB3 vector environment; experiment scripts still own
    policy, optimizer, and training loop.

    Args:
        settings: Configuration settings for this component.
        parallel_environments: Number of simultaneous training environments.
        seed: Optional seed for reproducible random sampling.
        use_subprocesses: Whether to run parallel environments in separate processes.

    Returns:
        object: Created SB3 vector environments for single-controlled-Robotino experiments.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if parallel_environments < 1:
        raise ValueError("parallel_environments must be positive")
    fleet_settings = load_settings()

    def factory() -> SingleRobotinoGymEnv:
        """Return one Gymnasium wrapper using the shared fleet and settings.

        Returns:
            SingleRobotinoGymEnv: One Gymnasium wrapper using the shared fleet and settings.
        """

        return SingleRobotinoGymEnv(
            fleet_settings=fleet_settings,
            learning_settings=settings,
        )

    return make_vec_env(
        factory,
        n_envs=parallel_environments,
        seed=seed,
        vec_env_cls=SubprocVecEnv if use_subprocesses else None,
        vec_env_kwargs={"start_method": "spawn"} if use_subprocesses else None,
    )


def resolve_torch_device(requested: str, *, purpose: str = "Training") -> str:
    """Return a resolved PyTorch device from requested.

    purpose labels the console report. Reject unavailable CUDA devices
    rather than silently using another GPU; auto picks CUDA or CPU.

    Args:
        requested: Requested PyTorch device name, or automatic selection.
        purpose: Label used when reporting device selection.

    Returns:
        str: A resolved PyTorch device from requested.

    Raises:
        RuntimeError: If the operation cannot complete in the current runtime state.
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    value = requested.strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA support")
        index = device.index if device.index is not None else 0
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise RuntimeError(
                f"Requested cuda:{index}, but PyTorch sees {count} CUDA device(s)"
            )
        torch.cuda.set_device(index)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"{purpose} device: cuda:{index} ({torch.cuda.get_device_name(index)})")
        return f"cuda:{index}"
    if device.type != "cpu":
        raise ValueError(f"Unsupported training device {requested!r}")
    print(f"{purpose} device: CPU")
    return "cpu"


def trainable_parameter_count(model) -> int:
    """Return the count of trainable policy parameters in model.

    Args:
        model: Policy network whose trainable parameters are counted.

    Returns:
        int: The count of trainable policy parameters in model.
    """

    return sum(
        parameter.numel()
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    )
