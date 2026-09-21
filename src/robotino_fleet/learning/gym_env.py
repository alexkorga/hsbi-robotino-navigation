"""Single-controlled-Robotino Gymnasium adapter for Stable-Baselines3."""

import math
from typing import Any

import gymnasium as gym
import numpy as np

from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.pettingzoo_env import RobotinoFactoryParallelEnv


class SingleRobotinoGymEnv(gym.Env):
    """Expose one controlled agent from the shared PettingZoo factory world."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        fleet_settings=None,
        learning_settings: LearningEnvironmentSettings | None = None,
        controlled_agent: str | None = None,
    ) -> None:
        """Select one controlled agent from a shared PettingZoo world.

        fleet_settings and learning_settings configure the factory;
        controlled_agent chooses its IP, defaulting to the first. Other
        agents follow configured scripted or stationary behavior.

        Args:
            fleet_settings: Robotino fleet and physical-motion settings.
            learning_settings: Scenario, observation, and reward settings for learning.
            controlled_agent: Select one controlled agent from a shared PettingZoo world.
                fleet_settings and learning_settings configure the factory; controlled_agent
                chooses its IP, defaulting to the first.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        self.parallel_env = RobotinoFactoryParallelEnv(
            fleet_settings=fleet_settings,
            learning_settings=learning_settings,
        )
        controlled_agent = controlled_agent or self.parallel_env.possible_agents[0]
        if controlled_agent not in self.parallel_env.possible_agents:
            raise ValueError(f"Unknown controlled agent {controlled_agent}")
        self.controlled_agent = controlled_agent
        self.observation_space = self.parallel_env.observation_space(controlled_agent)
        self.action_space = self.parallel_env.action_space(controlled_agent)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset with optional seed/options and return agent observation/info.

        Args:
            seed: Optional seed for reproducible random sampling.
            options: Optional reset overrides, including fixed starts and goals.

        Returns:
            object: Initial observation and info for the controlled agent.
        """

        super().reset(seed=seed)
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        return observations[self.controlled_agent], infos[self.controlled_agent]

    def step(self, action: np.ndarray):
        """Advance all agents using action only for the controlled Robotino.

        Returns the Gymnasium observation, reward, termination, truncation, and
        info tuple for that agent; peers use the configured scripted behavior.

        Args:
            action: Normalized policy action for the controlled Robotino.

        Returns:
            object: Gymnasium observation, reward, termination, truncation, and info.
        """

        actions: dict[str, np.ndarray] = {}
        for agent in self.parallel_env.agents:
            if agent == self.controlled_agent:
                actions[agent] = np.asarray(action, dtype=np.float32)
                continue
            if self.parallel_env.core.settings.other_robot_policy == "scripted-goal":
                robot = self.parallel_env.core.robots[agent]
                goal = self.parallel_env.core.goals[agent]
                dx = goal.point.x - robot.x
                dy = goal.point.y - robot.y
                distance = max(math.hypot(dx, dy), 1e-9)
                cosine = math.cos(robot.heading_rad)
                sine = math.sin(robot.heading_rad)
                actions[agent] = np.asarray(
                    [
                        (cosine * dx + sine * dy) / distance,
                        (-sine * dx + cosine * dy) / distance,
                        0.0,
                    ],
                    dtype=np.float32,
                )
            else:
                actions[agent] = np.zeros((3,), dtype=np.float32)
        observations, rewards, terminations, truncations, infos = (
            self.parallel_env.step(actions)
        )
        agent = self.controlled_agent
        return (
            observations[agent],
            rewards[agent],
            terminations[agent],
            truncations[agent],
            infos[agent],
        )

    def close(self) -> None:
        """Release the underlying multi-agent environment."""

        self.parallel_env.close()
