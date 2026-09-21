"""GRU actor-critic policy and portable inference artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class GruPolicyArchitecture:
    """Editable feature widths and recurrent-state size for the GRU model."""

    scan_features: int = 192
    state_hidden: int = 128
    state_features: int = 96
    fused_features: int = 256
    hidden_size: int = 288


class GruActorCritic(nn.Module):
    """Encode one observation at a time and retain per-agent GRU memory."""

    SCAN_KEYS = ("lidar_closeness", "lidar_validity")
    STATE_KEYS = (
        "proximity_closeness",
        "goal_body",
        "goal_distance",
        "velocity",
        "previous_action",
    )

    def __init__(
        self,
        observation_shapes: Mapping[str, tuple[int, ...]],
        action_dimensions: int = 2,
        architecture: GruPolicyArchitecture = GruPolicyArchitecture(),
    ) -> None:
        """Build separate scan/state encoders, one GRU, and actor/critic heads.

        observation_shapes must include all named sensor and goal features;
        action_dimensions sets the local-target output width, and
        architecture configures encoder, fusion, and memory widths.

        Args:
            observation_shapes: Input shapes for the named scan and state observation fields.
            action_dimensions: Number of normalized actions emitted per Robotino.
            architecture: GRU encoder and actor/critic layer sizes.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        super().__init__()
        required = {*self.SCAN_KEYS, *self.STATE_KEYS}
        missing = required - set(observation_shapes)
        if missing:
            raise ValueError(
                "GRU policy observation keys are missing: "
                + ", ".join(sorted(missing))
            )
        bins = int(observation_shapes["lidar_closeness"][0])
        if observation_shapes["lidar_validity"] != (bins,):
            raise ValueError("GRU LiDAR closeness and validity shapes must match")
        state_size = sum(
            int(np.prod(observation_shapes[key])) for key in self.STATE_KEYS
        )
        self.observation_shapes = dict(observation_shapes)
        self.action_dimensions = action_dimensions
        self.architecture = architecture

        self.scan_network = nn.Sequential(
            nn.Conv1d(2, 48, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, 96, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(96 * 4, architecture.scan_features),
            nn.SiLU(),
            nn.LayerNorm(architecture.scan_features),
        )
        self.state_network = nn.Sequential(
            nn.Linear(state_size, architecture.state_hidden),
            nn.SiLU(),
            nn.Linear(architecture.state_hidden, architecture.state_features),
            nn.SiLU(),
            nn.LayerNorm(architecture.state_features),
        )
        self.step_fusion = nn.Sequential(
            nn.Linear(
                architecture.scan_features + architecture.state_features,
                architecture.fused_features,
            ),
            nn.SiLU(),
            nn.LayerNorm(architecture.fused_features),
        )
        self.gru = nn.GRU(
            input_size=architecture.fused_features,
            hidden_size=architecture.hidden_size,
            num_layers=1,
        )
        self.memory_refinement = nn.Sequential(
            nn.LayerNorm(architecture.hidden_size),
            nn.Linear(architecture.hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, architecture.hidden_size),
        )
        self.actor = nn.Sequential(
            nn.Linear(architecture.hidden_size, 192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dimensions),
        )
        self.critic = nn.Sequential(
            nn.Linear(architecture.hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dimensions,), -1.0))

    @property
    def hidden_size(self) -> int:
        """Return the recurrent width defined by this policy's architecture.

        Returns:
            int: The recurrent width defined by this policy's architecture.
        """

        return self.architecture.hidden_size

    def initial_state(self, batch_size: int, *, device=None) -> torch.Tensor:
        """Return zero memory shaped [1, batch_size, hidden_size].

        batch_size is the number of parallel agents; device overrides
        the policy parameter device when supplied.

        Args:
            batch_size: Number of parallel agents represented in the tensor batch.
            device: PyTorch device for training or inference.

        Returns:
            torch.Tensor: Zero memory shaped [1, batch_size, hidden_size].
        """

        reference = next(self.parameters())
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size,
            dtype=reference.dtype,
            device=device or reference.device,
        )

    def _encode(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into fused per-step scan/state features.

        Preserve leading batch/time dimensions and return tensors whose final
        axis has the configured fused-feature width for the GRU.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            torch.Tensor: Fused scan and state features, preserving batch/time axes.
        """

        leading_shape = observations["lidar_closeness"].shape[:-1]
        scan = torch.stack(
            [observations[key] for key in self.SCAN_KEYS], dim=-2
        ).reshape(-1, 2, self.observation_shapes["lidar_closeness"][0])
        state = torch.cat(
            [
                observations[key].flatten(
                    start_dim=-len(self.observation_shapes[key])
                )
                for key in self.STATE_KEYS
            ],
            dim=-1,
        ).reshape(
            -1,
            sum(
                int(np.prod(self.observation_shapes[key]))
                for key in self.STATE_KEYS
            ),
        )
        encoded = self.step_fusion(
            torch.cat([self.scan_network(scan), self.state_network(state)], dim=-1)
        )
        return encoded.reshape(*leading_shape, -1)

    def recurrent(
        self,
        observations: Mapping[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence while resetting memory at episode boundaries.

        observations may contain one batch step or a time-major sequence;
        hidden is incoming GRU state and episode_starts clears slots
        at boundaries. Return refined memory for each step and final state.

        Args:
            observations: Policy observations for one or more agents.
            hidden: Previous recurrent hidden state for the agent batch.
            episode_starts: Flags identifying agents whose recurrent memory must reset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded sequence features and updated recurrent
                memory.
        """

        encoded = self._encode(observations)
        if encoded.ndim == 2:
            encoded = encoded.unsqueeze(0)
            episode_starts = episode_starts.unsqueeze(0)
        outputs = []
        current = hidden
        for step in range(encoded.shape[0]):
            reset = episode_starts[step].to(encoded.dtype).view(1, -1, 1)
            current = current * (1.0 - reset)
            output, current = self.gru(encoded[step : step + 1], current)
            outputs.append(output[0])
        memory = torch.stack(outputs)
        memory = memory + self.memory_refinement(memory)
        return memory, current

    def distribution(self, memory: torch.Tensor) -> Normal:
        """Return the learned Gaussian action distribution for encoded memory.

        Args:
            memory: Encoded recurrent policy state.

        Returns:
            Normal: The learned Gaussian action distribution for encoded memory.
        """

        mean = self.actor(memory)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def act(
        self,
        observations: Mapping[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Act on observations using hidden recurrent state.

        episode_starts resets selected agents and deterministic chooses
        mean rather than sampled Gaussian actions. Return action, state value,
        log-probability, and next recurrent state in that order.

        Args:
            observations: Policy observations for one or more agents.
            hidden: Previous recurrent hidden state for the agent batch.
            episode_starts: Flags identifying agents whose recurrent memory must reset.
            deterministic: Whether to choose the policy's deterministic action.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Sampled actions, log
                probabilities, state values, and next memory.
        """

        memory, next_hidden = self.recurrent(observations, hidden, episode_starts)
        memory = memory[0]
        distribution = self.distribution(memory)
        action = distribution.mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        value = self.critic(memory).squeeze(-1)
        return action, value, log_probability, next_hidden

    def evaluate_sequence(
        self,
        observations: Mapping[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score actions for PPO under a recurrent observation sequence.

        observations and hidden provide initial sequence state;
        episode_starts resets per-agent memory at episode boundaries.
        Return values, action log-probabilities, and entropy tensors.

        Args:
            observations: Policy observations for one or more agents.
            hidden: Previous recurrent hidden state for the agent batch.
            episode_starts: Flags identifying agents whose recurrent memory must reset.
            actions: Normalized actions keyed by agent or ordered by rollout slot.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Action log probabilities,
                entropies, and state values.
        """

        memory, _ = self.recurrent(observations, hidden, episode_starts)
        distribution = self.distribution(memory)
        log_probability = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(memory).squeeze(-1)
        return values, log_probability, entropy

    def value(
        self,
        observations: Mapping[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Return first-step state values from observations/hidden.

        episode_starts marks agents whose recurrent memory must reset.

        Args:
            observations: Policy observations for one or more agents.
            hidden: Previous recurrent hidden state for the agent batch.
            episode_starts: Flags identifying agents whose recurrent memory must reset.

        Returns:
            torch.Tensor: First-step state values from observations/hidden.
        """

        memory, _ = self.recurrent(observations, hidden, episode_starts)
        return self.critic(memory[0]).squeeze(-1)


class GruPolicyModel:
    """Small predict/save/load facade used by evaluation and live inference."""

    FORMAT_VERSION = 1

    def __init__(self, policy: GruActorCritic, *, device: str | torch.device) -> None:
        """Place policy on the requested device for evaluation/inference.

        Args:
            policy: Trainable or loaded navigation policy.
            device: PyTorch device for training or inference.
        """

        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.policy.eval()
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            (policy.action_dimensions,),
            dtype=np.float32,
        )

    def _tensors(
        self, observations: Mapping[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Convert NumPy observations to device tensors with batch axes.

        Return only keys declared in the saved policy observation contract.
        A single unbatched observation is expanded to a one-robot batch.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            dict[str, torch.Tensor]: Device tensors keyed by the saved policy's observation
                names.
        """

        result = {}
        for key in self.policy.observation_shapes:
            value = np.asarray(observations[key], dtype=np.float32)
            if value.ndim == len(self.policy.observation_shapes[key]):
                value = value[None, ...]
            result[key] = torch.as_tensor(value, device=self.device)
        return result

    def predict(
        self,
        observations: Mapping[str, np.ndarray],
        state: np.ndarray | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer clipped actions and next memory for one or many Robotinos.

        observations supplies one or many Robotino feature mappings;
        state carries their GRU memory; episode_start resets selected
        slots. deterministic selects mean actions rather than samples.
        Return clipped actions plus memory for the next call.

        Args:
            observations: Policy observations for one or more agents.
            state: Previous policy or simulator state.
            episode_start: Whether this agent starts a new episode and resets memory.
            deterministic: Whether to choose the policy's deterministic action.

        Returns:
            tuple[np.ndarray, np.ndarray]: Clipped actions and next recurrent memory for the
                input batch.
        """

        first_key = next(iter(self.policy.observation_shapes))
        vectorized = np.asarray(observations[first_key]).ndim > len(
            self.policy.observation_shapes[first_key]
        )
        tensors = self._tensors(observations)
        batch = next(iter(tensors.values())).shape[0]
        hidden = (
            self.policy.initial_state(batch, device=self.device)
            if state is None
            else torch.as_tensor(state, dtype=torch.float32, device=self.device)
        )
        starts = torch.as_tensor(
            np.ones(batch, dtype=bool) if episode_start is None else episode_start,
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            actions, _, _, next_hidden = self.policy.act(
                tensors,
                hidden,
                starts,
                deterministic=deterministic,
            )
        clipped = actions.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        if not vectorized:
            clipped = clipped[0]
        return clipped, next_hidden.cpu().numpy().astype(np.float32)

    def save(self, path: Path) -> None:
        """Write path as a ZIP-named PyTorch artifact with contract/weights.

        Args:
            path: Destination file for the generated data or model.
        """

        destination = Path(path).with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "robotino-gru-ppo",
                "format_version": self.FORMAT_VERSION,
                "observation_shapes": self.policy.observation_shapes,
                "action_dimensions": self.policy.action_dimensions,
                "architecture": asdict(self.policy.architecture),
                "state_dict": self.policy.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "GruPolicyModel":
        """Return a GRU model loaded from path onto device.

        Reject artifacts without the expected Robotino GRU format marker.

        Args:
            path: Path to the input file or model artifact.
            device: PyTorch device for training or inference.

        Returns:
            'GruPolicyModel': A GRU model loaded from path onto device.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        source = Path(path)
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        if checkpoint.get("format") != "robotino-gru-ppo":
            raise ValueError(f"Not a Robotino GRU policy: {source}")
        architecture = GruPolicyArchitecture(**checkpoint["architecture"])
        policy = GruActorCritic(
            {
                key: tuple(value)
                for key, value in checkpoint["observation_shapes"].items()
            },
            action_dimensions=int(checkpoint["action_dimensions"]),
            architecture=architecture,
        )
        policy.load_state_dict(checkpoint["state_dict"])
        return cls(policy, device=device)
