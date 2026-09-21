"""Sequence-aware PPO training for the shared GRU navigation policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from robotino_fleet.learning.gru_policy import GruActorCritic, GruPolicyModel
from robotino_fleet.learning.profile import BinnedLocalTargetProfile, write_policy_profile
from robotino_fleet.learning.training_progress import EpisodeTrainingProgress


@dataclass
class GruRollout:
    """Time-major observations, memory, actions, and GAE targets for one rollout."""

    observations: dict[str, np.ndarray]
    hidden: np.ndarray
    episode_starts: np.ndarray
    actions: np.ndarray
    log_probabilities: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    valid: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class GruPPOTrainer:
    """PPO with ordered truncated-BPTT minibatches and per-agent memory."""

    def __init__(
        self,
        policy: GruActorCritic,
        environment,
        *,
        device: str,
        learning_rate: float,
        rollout_steps: int = 128,
        sequence_length: int = 32,
        sequence_batch_size: int = 32,
        epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.20,
        entropy_coefficient: float = 0.005,
        value_coefficient: float = 0.5,
        target_kl: float = 0.03,
        max_gradient_norm: float = 0.5,
        seed: int = 42,
        tensorboard_directory: Path | None = None,
    ) -> None:
        """Configure sequence-aware PPO and its optimizer.

        Collect fixed-length rollouts from flattened environment slots and
        update the policy with ordered truncated-BPTT sequences. A rollout
        must split evenly into sequences.

        Args:
            policy: Trainable or loaded navigation policy.
            environment: Training environment supplying agent observations and rewards.
            device: PyTorch device for training or inference.
            learning_rate: Optimizer learning rate for policy updates.
            rollout_steps: Time steps collected from each environment before one PPO update.
            sequence_length: Unroll length for truncated backpropagation through time.
            sequence_batch_size: Number of ordered sequences per optimizer minibatch.
            epochs: Optimizer passes over each collected rollout.
            gamma: Discount factor applied to future rewards.
            gae_lambda: Generalized advantage estimation smoothing factor.
            clip_range: Maximum PPO probability-ratio deviation from the old policy.
            entropy_coefficient: Weight of the entropy bonus in the PPO loss.
            value_coefficient: Weight of critic value error in the PPO loss.
            target_kl: Approximate KL threshold for stopping an update early.
            max_gradient_norm: Norm used to clip optimizer gradients.
            seed: Optional seed for reproducible random sampling.
            tensorboard_directory: Optional directory for TensorBoard event logs.

        Raises:
            ValueError: If rollout_steps is not divisible by sequence_length.
        """

        if rollout_steps % sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.environment = environment
        self.rollout_steps = rollout_steps
        self.sequence_length = sequence_length
        self.sequence_batch_size = sequence_batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.target_kl = target_kl
        self.max_gradient_norm = max_gradient_norm
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.writer = (
            SummaryWriter(str(tensorboard_directory))
            if tensorboard_directory is not None
            else None
        )
        self.random = random.Random(seed)
        self.total_agent_steps = 0
        self.update_count = 0
        self.observations: dict[str, np.ndarray] | None = None
        self.hidden: torch.Tensor | None = None
        self.episode_starts: np.ndarray | None = None
        self.active: np.ndarray | None = None

    def set_environment(self, environment) -> None:
        """Replace the curriculum environment and reset cached rollout state.

        Args:
            environment: Training environment supplying agent observations and rewards.
        """

        self.environment = environment
        self.observations = None
        self.hidden = None
        self.episode_starts = None
        self.active = None

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set every optimizer group's rate to learning_rate.

        Args:
            learning_rate: Optimizer learning rate for policy updates.
        """

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def set_entropy_coefficient(self, entropy_coefficient: float) -> None:
        """Set entropy_coefficient for later PPO policy updates.

        Args:
            entropy_coefficient: Weight of the entropy bonus in the PPO loss.
        """

        self.entropy_coefficient = entropy_coefficient

    def _reset_rollout_state(self) -> None:
        """Reset vector worlds and initialize per-agent GRU rollout memory."""

        self.observations = self.environment.reset()
        count = self.environment.num_envs
        self.hidden = self.policy.initial_state(count, device=self.device)
        self.episode_starts = np.ones(count, dtype=bool)
        self.active = np.ones(count, dtype=bool)

    def _tensors(
        self, observations: Mapping[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Return observations as float32 tensors on the trainer device.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            dict[str, torch.Tensor]: Observations as float32 tensors on the trainer device.
        """

        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in observations.items()
            if key in self.policy.observation_shapes
        }

    def collect(
        self,
        progress_callback: Callable[[list[dict]], None] | None = None,
    ) -> GruRollout:
        """Collect one rollout with per-agent GRU memory and GAE targets.

        progress_callback receives episode infos while sampling. Returns
        time-major data, including a mask for still-active agent transitions.

        Args:
            progress_callback: Collect one rollout with per-agent GRU memory and GAE targets.
                progress_callback receives episode infos while sampling.

        Returns:
            GruRollout: Transitions and advantage targets for one PPO rollout.
        """

        if self.observations is None:
            self._reset_rollout_state()
        assert self.observations is not None
        assert self.hidden is not None
        assert self.episode_starts is not None
        assert self.active is not None
        observation_rows = {key: [] for key in self.policy.observation_shapes}
        hidden_rows = []
        start_rows = []
        action_rows = []
        log_probability_rows = []
        value_rows = []
        reward_rows = []
        terminal_rows = []
        valid_rows = []

        self.policy.eval()
        for _ in range(self.rollout_steps):
            for key in observation_rows:
                observation_rows[key].append(self.observations[key].copy())
            hidden_rows.append(self.hidden[0].detach().cpu().numpy())
            start_rows.append(self.episode_starts.copy())
            valid_rows.append(self.active.copy())
            with torch.no_grad():
                actions, values, log_probabilities, next_hidden = self.policy.act(
                    self._tensors(self.observations),
                    self.hidden,
                    torch.as_tensor(
                        self.episode_starts,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    deterministic=False,
                )
            raw_actions = actions.cpu().numpy().astype(np.float32)
            next_observations, rewards, dones, infos = self.environment.step(
                np.clip(raw_actions, -1.0, 1.0)
            )
            next_active = np.asarray(
                [bool(info.get("active", True)) for info in infos], dtype=bool
            )
            terminals = np.asarray(dones, dtype=bool) | (self.active & ~next_active)
            action_rows.append(raw_actions)
            value_rows.append(values.cpu().numpy().astype(np.float32))
            log_probability_rows.append(
                log_probabilities.cpu().numpy().astype(np.float32)
            )
            reward_rows.append(np.asarray(rewards, dtype=np.float32))
            terminal_rows.append(terminals)

            reset_memory = np.asarray(dones, dtype=bool) | ~next_active
            next_hidden[:, reset_memory, :] = 0.0
            self.hidden = next_hidden
            self.episode_starts = reset_memory
            self.active = np.where(dones, True, next_active)
            self.observations = next_observations
            self.total_agent_steps += self.environment.num_envs
            if progress_callback is not None:
                progress_callback(infos)

        with torch.no_grad():
            last_values = self.policy.value(
                self._tensors(self.observations),
                self.hidden,
                torch.as_tensor(
                    self.episode_starts,
                    dtype=torch.bool,
                    device=self.device,
                ),
            ).cpu().numpy().astype(np.float32)
        last_values[~self.active] = 0.0

        rewards_array = np.stack(reward_rows)
        values_array = np.stack(value_rows)
        terminals_array = np.stack(terminal_rows)
        valid_array = np.stack(valid_rows)
        advantages = np.zeros_like(rewards_array)
        next_advantage = np.zeros(self.environment.num_envs, dtype=np.float32)
        next_value = last_values
        for step in range(self.rollout_steps - 1, -1, -1):
            continuation = 1.0 - terminals_array[step].astype(np.float32)
            delta = (
                rewards_array[step]
                + self.gamma * next_value * continuation
                - values_array[step]
            )
            next_advantage = (
                delta
                + self.gamma * self.gae_lambda * continuation * next_advantage
            )
            advantages[step] = np.where(valid_array[step], next_advantage, 0.0)
            next_value = values_array[step]
            next_advantage = np.where(valid_array[step], next_advantage, 0.0)
        returns = advantages + values_array
        return GruRollout(
            observations={key: np.stack(rows) for key, rows in observation_rows.items()},
            hidden=np.stack(hidden_rows),
            episode_starts=np.stack(start_rows),
            actions=np.stack(action_rows),
            log_probabilities=np.stack(log_probability_rows),
            values=values_array,
            rewards=rewards_array,
            terminals=terminals_array,
            valid=valid_array,
            advantages=advantages,
            returns=returns,
        )

    def update(self, rollout: GruRollout) -> dict[str, float]:
        """Optimize ordered sequences from rollout with clipped PPO losses.

        Invalid transitions are masked; optimization can stop early at
        target_kl. Returns mean policy/value/entropy/KL diagnostics.

        Args:
            rollout: Collected transitions and advantage targets for a PPO update.

        Returns:
            dict[str, float]: Training loss and optimizer diagnostics for the update.
        """

        valid_advantages = rollout.advantages[rollout.valid]
        advantage_mean = float(valid_advantages.mean())
        advantage_std = float(valid_advantages.std()) + 1e-8
        normalized_advantages = (
            rollout.advantages - advantage_mean
        ) / advantage_std
        sequences = [
            (start, environment)
            for environment in range(self.environment.num_envs)
            for start in range(0, self.rollout_steps, self.sequence_length)
            if rollout.valid[start : start + self.sequence_length, environment].any()
        ]
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }
        self.policy.train()
        stop_early = False
        for _ in range(self.epochs):
            self.random.shuffle(sequences)
            for offset in range(0, len(sequences), self.sequence_batch_size):
                batch = sequences[offset : offset + self.sequence_batch_size]
                observations = {
                    key: torch.as_tensor(
                        np.stack(
                            [
                                values[start : start + self.sequence_length, env]
                                for start, env in batch
                            ],
                            axis=1,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for key, values in rollout.observations.items()
                }
                initial_hidden = torch.as_tensor(
                    np.stack([rollout.hidden[start, env] for start, env in batch]),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                starts = torch.as_tensor(
                    np.stack(
                        [
                            rollout.episode_starts[
                                start : start + self.sequence_length, env
                            ]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.bool,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    np.stack(
                        [
                            rollout.actions[start : start + self.sequence_length, env]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                old_log_probabilities = torch.as_tensor(
                    np.stack(
                        [
                            rollout.log_probabilities[
                                start : start + self.sequence_length, env
                            ]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    np.stack(
                        [
                            normalized_advantages[
                                start : start + self.sequence_length, env
                            ]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    np.stack(
                        [
                            rollout.returns[start : start + self.sequence_length, env]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                valid = torch.as_tensor(
                    np.stack(
                        [
                            rollout.valid[start : start + self.sequence_length, env]
                            for start, env in batch
                        ],
                        axis=1,
                    ),
                    dtype=torch.bool,
                    device=self.device,
                )
                values, log_probabilities, entropy = self.policy.evaluate_sequence(
                    observations,
                    initial_hidden,
                    starts,
                    actions,
                )
                ratios = torch.exp(log_probabilities - old_log_probabilities)
                unclipped = advantages * ratios
                clipped = advantages * torch.clamp(
                    ratios,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                )
                policy_loss = -torch.min(unclipped, clipped)[valid].mean()
                value_loss = torch.square(returns - values)[valid].mean()
                entropy_mean = entropy[valid].mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_gradient_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = (
                        log_probabilities[valid] - old_log_probabilities[valid]
                    )
                    approximate_kl = (
                        (torch.exp(log_ratio) - 1.0) - log_ratio
                    ).mean()
                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_mean.detach().cpu()))
                metrics["approx_kl"].append(float(approximate_kl.cpu()))
                if approximate_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        self.update_count += 1
        result = {
            key: sum(values) / max(1, len(values)) for key, values in metrics.items()
        }
        if self.writer is not None:
            for key, value in result.items():
                self.writer.add_scalar(f"train/{key}", value, self.total_agent_steps)
            self.writer.add_scalar(
                "train/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                self.total_agent_steps,
            )
            self.writer.add_scalar(
                "train/entropy_coefficient",
                self.entropy_coefficient,
                self.total_agent_steps,
            )
        return result

    def save(self, path: Path) -> None:
        """Save path as an inference artifact without optimizer state.

        Args:
            path: Destination file for the generated data or model.
        """

        GruPolicyModel(self.policy, device=self.device).save(path)
        self.policy.train()

    def learn(
        self,
        requested_agent_steps: int,
        *,
        description: str,
        checkpoint_directory: Path | None = None,
        checkpoint_prefix: str = "gru_policy",
        checkpoint_interval: int = 100_000,
        checkpoint_profile: BinnedLocalTargetProfile | None = None,
    ) -> None:
        """Run rollouts until at least requested_agent_steps are collected.

        description labels console progress. checkpoint_directory
        enables periodic saves named with checkpoint_prefix after each
        checkpoint_interval agent steps. If checkpoint_profile is set,
        sidecars accompany checkpoints so they can be evaluated or deployed.
        The trainer updates in place and returns nothing.

        Args:
            requested_agent_steps: Minimum number of agent transitions to collect.
            description: Short label displayed with the training progress bar.
            checkpoint_directory: Directory for periodic inference checkpoints, if enabled.
            checkpoint_prefix: Filename prefix for periodic checkpoints.
            checkpoint_interval: Agent-step interval between checkpoints.
            checkpoint_profile: If checkpoint_profile is set, sidecars accompany checkpoints so
                they can be evaluated or deployed.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        started = self.total_agent_steps
        next_checkpoint = started + checkpoint_interval
        progress = EpisodeTrainingProgress(
            description=description,
            total_timesteps=requested_agent_steps,
        )
        progress.start(started)
        try:
            def refresh_progress(infos: list[dict]) -> None:
                """Update the live progress bar from completed-agent infos.

                Args:
                    infos: Per-agent episode information and metrics.
                """

                progress.update(self.total_agent_steps, infos)

            while self.total_agent_steps - started < requested_agent_steps:
                before = self.total_agent_steps
                rollout = self.collect(progress_callback=refresh_progress)
                self.update(rollout)
                gained = self.total_agent_steps - before
                if checkpoint_directory is not None and self.total_agent_steps >= next_checkpoint:
                    checkpoint_directory.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = (
                        checkpoint_directory
                        / f"{checkpoint_prefix}_{self.total_agent_steps}_steps.zip"
                    )
                    self.save(checkpoint_path)
                    if checkpoint_profile is not None:
                        write_policy_profile(checkpoint_path, checkpoint_profile)
                    next_checkpoint += checkpoint_interval
                if gained <= 0:
                    raise RuntimeError("GRU rollout did not advance training")
        finally:
            progress.close()

    def close(self) -> None:
        """Flush and close TensorBoard logging when it was enabled."""

        if self.writer is not None:
            self.writer.close()
