"""Shared live progress reporting for SB3 and recurrent policy training."""

from collections import deque
from typing import Iterable, Mapping

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class EpisodeTrainingProgress:
    """Track individual-agent episode results on one live tqdm bar."""

    def __init__(self, *, description: str, total_timesteps: int) -> None:
        """Track total_timesteps and recent episode outcomes under description.

        Args:
            description: Short label displayed with the training progress bar.
            total_timesteps: Total agent steps planned for the run.
        """

        self.description = description
        self.total_timesteps = total_timesteps
        self.progress: tqdm | None = None
        self.started_at_timesteps = 0
        self.last_timesteps = 0
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_successes: deque[float] = deque(maxlen=100)
        self.episode_collisions: deque[float] = deque(maxlen=100)
        self.episode_timeouts: deque[float] = deque(maxlen=100)

    def start(self, current_timesteps: int) -> None:
        """Start a bar from current_timesteps for the requested new steps.

        Args:
            current_timesteps: Agent steps completed so far.
        """

        self.started_at_timesteps = current_timesteps
        self.last_timesteps = current_timesteps
        self.progress = tqdm(
            total=self.total_timesteps,
            desc=self.description,
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    def update(
        self,
        current_timesteps: int,
        infos: Iterable[Mapping],
    ) -> None:
        """Update from current_timesteps and completed-episode infos.

        The bar shows moving reward, success, collision, and timeout averages
        over up to 100 recent individual-agent episodes. Nothing is returned.

        Args:
            current_timesteps: Agent steps completed so far.
            infos: Per-agent episode information and metrics.
        """

        if self.progress is None:
            return
        completed = min(
            self.total_timesteps,
            max(0, current_timesteps - self.started_at_timesteps),
        )
        previous = min(
            self.total_timesteps,
            max(0, self.last_timesteps - self.started_at_timesteps),
        )
        self.progress.update(max(0, completed - previous))
        self.last_timesteps = current_timesteps

        for info in infos:
            episode = info.get("episode")
            if episode is None or "r" not in episode:
                continue
            self.episode_rewards.append(float(episode["r"]))
            if "success" in info:
                self.episode_successes.append(float(bool(info["success"])))
            if "collision" in info:
                self.episode_collisions.append(float(bool(info["collision"])))
            if "timeout" in info:
                self.episode_timeouts.append(float(bool(info["timeout"])))

        if self.episode_rewards:
            postfix = {
                "reward_100": (
                    f"{sum(self.episode_rewards) / len(self.episode_rewards):.2f}"
                )
            }
            if self.episode_successes:
                postfix["success_100"] = (
                    f"{sum(self.episode_successes) / len(self.episode_successes):.0%}"
                )
            if self.episode_collisions:
                postfix["collision_100"] = (
                    f"{sum(self.episode_collisions) / len(self.episode_collisions):.0%}"
                )
            if self.episode_timeouts:
                postfix["timeout_100"] = (
                    f"{sum(self.episode_timeouts) / len(self.episode_timeouts):.0%}"
                )
            self.progress.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        """Finish the progress bar and release its console output."""

        if self.progress is not None:
            self.progress.close()
            self.progress = None


class TrainingProgressCallback(BaseCallback):
    """SB3 adapter for the progress reporter also used by the GRU trainer."""

    def __init__(self, *, experiment_name: str, total_timesteps: int) -> None:
        """Track total_timesteps for an SB3 experiment_name run.

        Args:
            experiment_name: Track total_timesteps for an SB3 experiment_name run.
            total_timesteps: Total agent steps planned for the run.
        """

        super().__init__(verbose=0)
        self.reporter = EpisodeTrainingProgress(
            description=experiment_name,
            total_timesteps=total_timesteps,
        )

    def _on_training_start(self) -> None:
        """Open the bar at SB3's current global timestep count."""

        self.reporter.start(self.num_timesteps)

    def _on_step(self) -> bool:
        """Record SB3 step and episode infos, returning True to continue.

        Returns:
            bool: True to allow SB3 training to continue.
        """

        self.reporter.update(
            self.num_timesteps,
            self.locals.get("infos", ()),
        )
        return True

    def _on_training_end(self) -> None:
        """Close the progress bar after SB3 training ends."""

        self.reporter.close()
