"""PettingZoo Parallel API for continuous factory navigation."""

from typing import Any, Mapping

import numpy as np
from pettingzoo import ParallelEnv

from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.core import FactoryLearningCore
from robotino_fleet.maps.loader import load_map_bundle
from robotino_fleet.config import ProjectSettings, load_settings


class RobotinoFactoryParallelEnv(ParallelEnv):
    """Present every simulated Robotino through PettingZoo's Parallel API."""

    metadata = {
        "name": "robotino_factory_navigation_v1",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        fleet_settings: ProjectSettings | None = None,
        learning_settings: LearningEnvironmentSettings | None = None,
    ) -> None:
        """Create one factory world for all PettingZoo agents.

        fleet_settings and learning_settings override project defaults.
        Each configured Robotino becomes one parallel agent with matching
        observation and action spaces.

        Args:
            fleet_settings: Robotino fleet and physical-motion settings.
            learning_settings: Scenario, observation, and reward settings for learning.
        """

        self.render_mode = None
        fleet = fleet_settings or load_settings()
        learning = learning_settings or LearningEnvironmentSettings()
        map_bundle = load_map_bundle(fleet.map_bundle)
        self.core = FactoryLearningCore(fleet, map_bundle, learning)
        self.possible_agents = list(self.core.agent_ids)
        self.agents: list[str] = []
        self.observation_spaces = {
            agent: self.core.observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.core.action_space for agent in self.possible_agents
        }

    def observation_space(self, agent: str):
        """Return the policy observation space registered for agent.

        Args:
            agent: PettingZoo agent identifier.

        Returns:
            object: The policy observation space registered for agent.
        """

        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Return the normalized action space registered for agent.

        Args:
            agent: PettingZoo agent identifier.

        Returns:
            object: The normalized action space registered for agent.
        """

        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        """Reset using optional seed/options and return agent data.

        The result is a pair of observation and info mappings keyed by agent
        ID, as required by the PettingZoo Parallel API.

        Args:
            seed: Optional seed for reproducible random sampling.
            options: Optional reset overrides, including fixed starts and goals.

        Returns:
            object: Initial observations and info mappings for all agents.
        """

        self.agents = self.possible_agents[:]
        return self.core.reset(seed=seed, options=options)

    def step(self, actions: Mapping[str, np.ndarray]):
        """Apply actions for all active agents and return Parallel API data.

        All agents end together when the shared episode terminates or truncates.
        Calling this after episode end raises until reset is called.

        Args:
            actions: Normalized actions keyed by agent or ordered by rollout slot.

        Returns:
            object: Parallel API observations, rewards, terminations, truncations, and infos.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        if not self.agents:
            raise RuntimeError("step() called after episode end; call reset()")
        result = self.core.step(actions)
        terminations = {agent: result.terminated for agent in self.agents}
        truncations = {agent: result.truncated for agent in self.agents}
        observations = {agent: result.observations[agent] for agent in self.agents}
        rewards = {agent: result.rewards[agent] for agent in self.agents}
        infos = {agent: result.infos[agent] for agent in self.agents}
        if result.terminated or result.truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return all agents' flattened observations as global state.

        Returns:
            np.ndarray: All agents' flattened observations as global state.
        """

        observations = self.core.observations(advance_history=False)
        values: list[np.ndarray] = []
        for agent in self.possible_agents:
            for key in sorted(observations[agent]):
                values.append(observations[agent][key].reshape(-1))
        return np.concatenate(values).astype(np.float32)

    def close(self) -> None:
        """Mark the environment closed to active PettingZoo agents."""

        self.agents = []


def parallel_env(**kwargs: Any) -> RobotinoFactoryParallelEnv:
    """Return a parallel environment configured by kwargs settings.

    Args:
        **kwargs: Keyword settings forwarded to the parallel environment.

    Returns:
        RobotinoFactoryParallelEnv: A parallel environment configured by kwargs settings.
    """

    return RobotinoFactoryParallelEnv(**kwargs)
    """Return a parallel environment configured by kwargs settings.

    Args:
        **kwargs: Return a parallel environment configured by kwargs settings.

    Returns:
        RobotinoFactoryParallelEnv: A parallel environment configured by kwargs settings.
    """
