"""Importable PyTorch feature extractors used by saved SB3 policies."""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BinnedLocalTargetExtractor(BaseFeaturesExtractor):
    """Preserve angular LiDAR structure while keeping the network small."""

    STATE_KEYS = (
        "proximity_closeness",
        "goal_body",
        "goal_distance",
        "velocity",
        "previous_action",
    )
    SCAN_KEYS = ("lidar_closeness", "lidar_validity")

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Build 192 features from observation_space's scan/state fields.

        Args:
            observation_space: Declared observation space used to size the network inputs.
        """

        state_size = sum(
            int(observation_space[key].shape[0]) for key in self.STATE_KEYS
        )
        super().__init__(observation_space, features_dim=192)
        self.scan_network = nn.Sequential(
            nn.Conv1d(len(self.SCAN_KEYS), 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64 * 4, 128),
            nn.SiLU(),
        )
        self.state_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return 192 policy features from current observations tensors.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            torch.Tensor: 192 policy features from current observations tensors.
        """

        scan = torch.stack(
            [observations[key] for key in self.SCAN_KEYS], dim=1
        )
        state = torch.cat(
            [observations[key] for key in self.STATE_KEYS], dim=1
        )
        return torch.cat(
            [self.scan_network(scan), self.state_network(state)], dim=1
        )


class ExpandedBinnedLocalTargetExtractor(BinnedLocalTargetExtractor):
    """A drop-in, residual capacity increase for the established CNN policy.

    The original extractor and its 192-value output contract stay unchanged,
    allowing all baseline weights to be transferred. The zero-initialized final
    layers make the two residual branches identities at the start of training.
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Add zero-initialized residual branches to the baseline extractor.

        observation_space retains the baseline input contract. The output
        remains 192 features so baseline weights can initialize this model.

        Args:
            observation_space: Declared observation space used to size the network inputs.
        """

        super().__init__(observation_space)
        self.scan_refinement = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.state_refinement = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )
        nn.init.zeros_(self.scan_refinement[-1].weight)
        nn.init.zeros_(self.scan_refinement[-1].bias)
        nn.init.zeros_(self.state_refinement[-1].weight)
        nn.init.zeros_(self.state_refinement[-1].bias)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return baseline observations features plus residual refinements.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            torch.Tensor: Baseline observations features plus residual refinements.
        """

        features = super().forward(observations)
        scan, state = torch.split(features, (128, 64), dim=1)
        return torch.cat(
            [
                scan + self.scan_refinement(scan),
                state + self.state_refinement(state),
            ],
            dim=1,
        )


class TemporalExpandedBinnedLocalTargetExtractor(
    ExpandedBinnedLocalTargetExtractor
):
    """Four-frame sensor fusion added as an identity-initialized correction.

    The current-frame expanded CNN remains intact. A separate temporal branch
    learns motion in the LiDAR and distance-sensor returns together with ego
    velocity and action history, then adds a residual correction to the same
    192-value feature contract.
    """

    TEMPORAL_SCAN_KEYS = (
        "lidar_closeness_history",
        "lidar_validity_history",
    )

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Add a temporal correction for observation_space histories.

        At least four frames and the specified history keys are required.
        Zero initialization preserves the expanded current-frame behavior at
        the start of training; the final feature width remains 192.

        Args:
            observation_space: Declared observation space used to size the network inputs.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        required = {
            "lidar_closeness_history",
            "lidar_validity_history",
            "proximity_closeness_history",
            "velocity_history",
            "action_history",
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(
                "Temporal extractor observation keys are missing: "
                + ", ".join(sorted(missing))
            )
        frames = int(observation_space["velocity_history"].shape[0])
        motion_values_per_frame = int(
            observation_space["velocity_history"].shape[1]
            + observation_space["action_history"].shape[1]
        )
        if frames < 4:
            raise ValueError("Temporal extractor requires at least four frames")
        super().__init__(observation_space)
        self.temporal_scan = nn.Sequential(
            nn.Conv2d(
                len(self.TEMPORAL_SCAN_KEYS),
                32,
                kernel_size=(2, 5),
                padding=(0, 2),
            ),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 6)),
            nn.Flatten(),
            nn.Linear(96 * 6, 192),
            nn.SiLU(),
        )
        self.temporal_proximity = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 3)),
            nn.Flatten(),
            nn.Linear(32 * 3, 64),
            nn.SiLU(),
        )
        self.temporal_motion = nn.Sequential(
            nn.Linear(frames * motion_values_per_frame, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(192 + 64 + 64, 256),
            nn.SiLU(),
            nn.Linear(256, 192),
        )
        nn.init.zeros_(self.temporal_fusion[-1].weight)
        nn.init.zeros_(self.temporal_fusion[-1].bias)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return current observations features plus temporal correction.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            torch.Tensor: Current observations features plus temporal correction.
        """

        current = super().forward(observations)
        scan_history = torch.stack(
            [observations[key] for key in self.TEMPORAL_SCAN_KEYS], dim=1
        )
        proximity_history = observations[
            "proximity_closeness_history"
        ].unsqueeze(1)
        motion_history = torch.cat(
            [observations["velocity_history"], observations["action_history"]],
            dim=2,
        ).flatten(start_dim=1)
        correction = self.temporal_fusion(
            torch.cat(
                [
                    self.temporal_scan(scan_history),
                    self.temporal_proximity(proximity_history),
                    self.temporal_motion(motion_history),
                ],
                dim=1,
            )
        )
        return current + correction


class ForwardTurnTemporalExtractor(TemporalExpandedBinnedLocalTargetExtractor):
    """Six-frame temporal encoder with an explicit unit target bearing."""

    STATE_KEYS = (
        "proximity_closeness",
        "goal_direction",
        "goal_distance",
        "velocity",
        "previous_action",
    )

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Require six frames in observation_space for forward/turn input.

        Args:
            observation_space: Declared observation space used to size the network inputs.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        frames = int(observation_space["velocity_history"].shape[0])
        motion_values_per_frame = int(
            observation_space["velocity_history"].shape[1]
            + observation_space["action_history"].shape[1]
        )
        if frames != 6:
            raise ValueError("Forward-turn temporal extractor requires six frames")
        super().__init__(observation_space)


class PoseCorrected360TemporalExtractor(
    TemporalExpandedBinnedLocalTargetExtractor
):
    """Six-frame temporal CNN over pose-corrected 360-degree LiDAR memory."""

    SCAN_KEYS = (
        "lidar_closeness",
        "lidar_validity",
        "lidar_freshness",
    )
    TEMPORAL_SCAN_KEYS = (
        "lidar_closeness_history",
        "lidar_validity_history",
        "lidar_freshness_history",
    )

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Require six frames and freshness keys in observation_space.

        Args:
            observation_space: Declared observation space used to size the network inputs.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        frames = int(observation_space["velocity_history"].shape[0])
        if frames != 6:
            raise ValueError("360 temporal extractor requires six frames")
        required = {"lidar_freshness", "lidar_freshness_history"}
        if not required.issubset(observation_space.spaces):
            raise ValueError("360 temporal extractor requires LiDAR freshness")
        super().__init__(observation_space)


class WideTemporalBinnedLocalTargetExtractor(BaseFeaturesExtractor):
    """Wider temporal CNN with the established depth and feature structure."""

    STATE_KEYS = (
        "proximity_closeness",
        "goal_body",
        "goal_distance",
        "velocity",
        "previous_action",
    )
    SCAN_KEYS = ("lidar_closeness", "lidar_validity")
    TEMPORAL_SCAN_KEYS = (
        "lidar_closeness_history",
        "lidar_validity_history",
    )

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Build a 256-feature wide CNN from current and temporal inputs.

        observation_space must declare the current scan/state fields and
        at least four history frames. Missing keys or too few frames raise
        ValueError before the model is used.

        Args:
            observation_space: Declared observation space used to size the network inputs.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        required = {
            *self.SCAN_KEYS,
            *self.TEMPORAL_SCAN_KEYS,
            *self.STATE_KEYS,
            "proximity_closeness_history",
            "velocity_history",
            "action_history",
        }
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError(
                "Wide temporal extractor observation keys are missing: "
                + ", ".join(sorted(missing))
            )
        frames = int(observation_space["velocity_history"].shape[0])
        motion_values_per_frame = int(
            observation_space["velocity_history"].shape[1]
            + observation_space["action_history"].shape[1]
        )
        if frames < 4:
            raise ValueError("Wide temporal extractor requires at least four frames")
        state_size = sum(
            int(observation_space[key].shape[0]) for key in self.STATE_KEYS
        )
        super().__init__(observation_space, features_dim=256)
        self.scan_network = nn.Sequential(
            nn.Conv1d(len(self.SCAN_KEYS), 40, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(40, 80, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(80 * 4, 160),
            nn.SiLU(),
        )
        self.state_network = nn.Sequential(
            nn.Linear(state_size, 96),
            nn.SiLU(),
        )
        self.scan_refinement = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, 320),
            nn.SiLU(),
            nn.Linear(320, 160),
        )
        self.state_refinement = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
        )
        self.temporal_scan = nn.Sequential(
            nn.Conv2d(
                len(self.TEMPORAL_SCAN_KEYS),
                40,
                kernel_size=(2, 5),
                padding=(0, 2),
            ),
            nn.SiLU(),
            nn.Conv2d(40, 80, kernel_size=(2, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(80, 120, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 6)),
            nn.Flatten(),
            nn.Linear(120 * 6, 256),
            nn.SiLU(),
        )
        self.temporal_proximity = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=(2, 3), padding=(0, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 3)),
            nn.Flatten(),
            nn.Linear(48 * 3, 96),
            nn.SiLU(),
        )
        self.temporal_motion = nn.Sequential(
            nn.Linear(frames * motion_values_per_frame, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(256 + 96 + 96, 320),
            nn.SiLU(),
            nn.Linear(320, 256),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return 256 fused scan/state/temporal features from observations.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            torch.Tensor: 256 fused scan/state/temporal features from observations.
        """

        scan = self.scan_network(
            torch.stack(
                [observations[key] for key in self.SCAN_KEYS], dim=1
            )
        )
        state = self.state_network(
            torch.cat([observations[key] for key in self.STATE_KEYS], dim=1)
        )
        current = torch.cat(
            [
                scan + self.scan_refinement(scan),
                state + self.state_refinement(state),
            ],
            dim=1,
        )
        temporal = self.temporal_fusion(
            torch.cat(
                [
                    self.temporal_scan(
                        torch.stack(
                            [
                                observations[key]
                                for key in self.TEMPORAL_SCAN_KEYS
                            ],
                            dim=1,
                        )
                    ),
                    self.temporal_proximity(
                        observations["proximity_closeness_history"].unsqueeze(1)
                    ),
                    self.temporal_motion(
                        torch.cat(
                            [
                                observations["velocity_history"],
                                observations["action_history"],
                            ],
                            dim=2,
                        ).flatten(start_dim=1)
                    ),
                ],
                dim=1,
            )
        )
        return current + temporal

class WidePoseCorrected360TemporalExtractor(
    WideTemporalBinnedLocalTargetExtractor
):
    """Wider six-frame CNN for the pose-corrected 360-degree contract."""

    SCAN_KEYS = (
        "lidar_closeness",
        "lidar_validity",
        "lidar_freshness",
    )
    TEMPORAL_SCAN_KEYS = (
        "lidar_closeness_history",
        "lidar_validity_history",
        "lidar_freshness_history",
    )

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Require six frames in observation_space for wide 360° encoding.

        Args:
            observation_space: Declared observation space used to size the network inputs.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        frames = int(observation_space["velocity_history"].shape[0])
        if frames != 6:
            raise ValueError("Wide 360 temporal extractor requires six frames")
        super().__init__(observation_space)
