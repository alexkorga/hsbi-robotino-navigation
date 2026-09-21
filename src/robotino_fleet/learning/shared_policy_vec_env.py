"""SB3 vector environment for one policy shared by interacting Robotinos."""

import multiprocessing as mp
import os
import traceback
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

from robotino_fleet.config import ProjectSettings
from robotino_fleet.learning.config import LearningEnvironmentSettings
from robotino_fleet.learning.pettingzoo_env import RobotinoFactoryParallelEnv


def _world_worker(remote, parent_remote, fleet_settings, learning_settings) -> None:
    """Own one world in a spawned process and report failures to its parent.

    remote is the worker pipe, parent_remote is closed in the child,
    and fleet_settings/learning_settings configure its PettingZoo
    world. Commands produce tagged results until close or pipe shutdown.

    Args:
        remote: Own one world in a spawned process and report failures to its parent. remote
            is the worker pipe, parent_remote is closed in the child, and
            fleet_settings/learning_settings configure its PettingZoo world.
        parent_remote: Own one world in a spawned process and report failures to its parent.
            remote is the worker pipe, parent_remote is closed in the child, and
            fleet_settings/learning_settings configure its PettingZoo world.
        fleet_settings: Robotino fleet and physical-motion settings.
        learning_settings: Scenario, observation, and reward settings for learning.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    parent_remote.close()
    world = None
    clean_shutdown = False
    try:
        world = RobotinoFactoryParallelEnv(
            fleet_settings=fleet_settings,
            learning_settings=learning_settings,
        )
        agent_ids = tuple(world.possible_agents)
        remote.send(
            (
                "ready",
                (
                    agent_ids,
                    world.observation_space(agent_ids[0]),
                    world.action_space(agent_ids[0]),
                ),
            )
        )
        while True:
            command, data = remote.recv()
            if command == "reset":
                seed, options = data
                remote.send(("ok", world.reset(seed=seed, options=options)))
            elif command == "step":
                observations, rewards, terminated, truncated, infos = world.step(data)
                done = all(terminated.values()) or all(truncated.values())
                terminal_observations = observations if done else None
                reset_infos = None
                if done:
                    observations, reset_infos = world.reset()
                remote.send(
                    (
                        "ok",
                        (
                            observations,
                            rewards,
                            terminated,
                            truncated,
                            infos,
                            terminal_observations,
                            reset_infos,
                        ),
                    )
                )
            elif command == "get_attr":
                remote.send(("ok", getattr(world, data)))
            elif command == "set_attr":
                name, value = data
                setattr(world, name, value)
                remote.send(("ok", None))
            elif command == "env_method":
                name, args, kwargs = data
                remote.send(("ok", getattr(world, name)(*args, **kwargs)))
            elif command == "close":
                remote.send(("ok", None))
                clean_shutdown = True
                break
            else:
                raise ValueError(f"Unknown world-worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if world is not None:
            world.close()
        remote.close()
        if clean_shutdown:
            # This worker owns no model or persistent state. Avoid unloading
            # PyTorch and other native runtimes during interpreter teardown:
            # on Windows that can leave a thread-pool wait registered against
            # a handle which a DLL finalizer has already closed (0xc000070a).
            os._exit(0)


class SharedPolicyVecEnv(VecEnv):
    """Flatten several multi-agent worlds into homogeneous PPO rollout slots.

    A slot is one Robotino. Agents that share a world are stepped together and
    therefore see and collide with each other. PPO still receives one transition
    per agent and optimizes a single set of policy parameters from the combined
    rollout batch.
    """

    def __init__(
        self,
        *,
        fleet_settings: ProjectSettings,
        learning_settings: LearningEnvironmentSettings,
        world_count: int,
        seed: int,
        parallel_worlds: bool = False,
        show_startup_progress: bool = False,
        startup_batch_size: int = 1,
    ) -> None:
        """Create world_count worlds and flatten their agents into SB3 slots.

        fleet_settings and learning_settings configure each world;
        seed initializes episodes. parallel_worlds uses spawned
        workers; show_startup_progress displays their preparation and
        startup_batch_size limits concurrent imports on Windows.

        Args:
            fleet_settings: Robotino fleet and physical-motion settings.
            learning_settings: Scenario, observation, and reward settings for learning.
            world_count: Number of independent simulated worlds to create.
            seed: Optional seed for reproducible random sampling.
            parallel_worlds: Whether to run simulation worlds in separate processes.
            show_startup_progress: Whether to display progress while starting worker worlds.
            startup_batch_size: Number of worker worlds to start together.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if world_count < 1:
            raise ValueError("world_count must be positive")
        if learning_settings.robot_count < 2:
            raise ValueError("shared-policy training requires at least two Robotinos")
        if startup_batch_size < 1:
            raise ValueError("startup_batch_size must be positive")
        self._parallel_worlds = parallel_worlds and world_count > 1
        self._processes: list[mp.Process] = []
        self._remotes: list[Any] = []
        if self._parallel_worlds:
            context = mp.get_context("spawn")
            specifications = []
            startup_progress = tqdm(
                total=world_count,
                desc="prepare simulation worlds",
                unit="world",
                disable=not show_startup_progress,
            )
            try:
                for batch_start in range(0, world_count, startup_batch_size):
                    batch_indices = range(
                        batch_start,
                        min(world_count, batch_start + startup_batch_size),
                    )
                    for world_index in batch_indices:
                        parent, child = context.Pipe()
                        self._remotes.append(parent)
                        process = context.Process(
                            target=_world_worker,
                            args=(child, parent, fleet_settings, learning_settings),
                            daemon=False,
                        )
                        try:
                            process.start()
                        finally:
                            child.close()
                        self._processes.append(process)
                    # Bound simultaneous Python/PyTorch imports on Windows,
                    # then wait until this complete group is ready before
                    # launching the next group.
                    for world_index in batch_indices:
                        specifications.append(
                            self._receive(world_index, expected="ready")
                        )
                        startup_progress.update(1)
            except BaseException:
                self.close()
                raise
            finally:
                startup_progress.close()
            first_agents, observation_space, action_space = specifications[0]
            if any(specification[0] != first_agents for specification in specifications[1:]):
                self.close()
                raise RuntimeError("Simulation workers expose different Robotino ids")
            self.agent_ids = tuple(first_agents)
            self.worlds: list[RobotinoFactoryParallelEnv] = []
        else:
            self.worlds = [
                RobotinoFactoryParallelEnv(
                    fleet_settings=fleet_settings,
                    learning_settings=learning_settings,
                )
                for _ in range(world_count)
            ]
            self.agent_ids = tuple(self.worlds[0].possible_agents)
            observation_space = self.worlds[0].observation_space(self.agent_ids[0])
            action_space = self.worlds[0].action_space(self.agent_ids[0])
        self.world_count = world_count
        self.agents_per_world = len(self.agent_ids)
        self._actions: np.ndarray | None = None
        super().__init__(
            world_count * self.agents_per_world,
            observation_space,
            action_space,
        )
        self.seed(seed)

    def _receive(self, world_index: int, *, expected: str = "ok") -> Any:
        """Read and validate one response from worker world_index.

        expected is the required status tag (normally ok). Return the
        worker payload, or raise RuntimeError with the remote failure or
        unexpected-exit details.

        Args:
            world_index: Index of the worker world expected to reply.
            expected: Required worker-response status tag.

        Returns:
            Any: Parsed and validate one response from worker world_index.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        remote = self._remotes[world_index]
        try:
            status, payload = remote.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            process = self._processes[world_index]
            raise RuntimeError(
                f"Simulation worker {world_index + 1} exited unexpectedly "
                f"(exit code {process.exitcode})"
            ) from error
        if status == "error":
            raise RuntimeError(
                f"Simulation worker {world_index + 1} failed:\n{payload}"
            )
        if status != expected:
            raise RuntimeError(
                f"Simulation worker {world_index + 1} returned {status!r}, "
                f"expected {expected!r}"
            )
        return payload

    def _stack(self, observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Return nonempty observations stacked in rollout-slot order.

        Args:
            observations: Policy observations for one or more agents.

        Returns:
            dict[str, np.ndarray]: Nonempty observations stacked in rollout-slot order.
        """

        return {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }

    def reset(self) -> dict[str, np.ndarray]:
        """Reset all worlds and return observations stacked by rollout slot.

        Returns:
            dict[str, np.ndarray]: Initial observations stacked in rollout-slot order.
        """

        flattened: list[dict[str, np.ndarray]] = []
        if self._parallel_worlds:
            for world_index, remote in enumerate(self._remotes):
                base = world_index * self.agents_per_world
                remote.send(("reset", (self._seeds[base], self._options[base])))
            results = [self._receive(index) for index in range(self.world_count)]
            for world_index, (observations, infos) in enumerate(results):
                base = world_index * self.agents_per_world
                for index, agent_id in enumerate(self.agent_ids):
                    flattened.append(observations[agent_id])
                    self.reset_infos[base + index] = infos[agent_id]
            self._reset_seeds()
            self._reset_options()
            return self._stack(flattened)
        for world_index, world in enumerate(self.worlds):
            base = world_index * self.agents_per_world
            seed = self._seeds[base]
            options = self._options[base]
            observations, infos = world.reset(seed=seed, options=options)
            for agent_id in self.agent_ids:
                flattened.append(observations[agent_id])
                self.reset_infos[base + self.agent_ids.index(agent_id)] = infos[
                    agent_id
                ]
        self._reset_seeds()
        self._reset_options()
        return self._stack(flattened)

    def step_async(self, actions: np.ndarray) -> None:
        """Validate slot-ordered actions and dispatch them to world workers.

        Args:
            actions: Normalized actions keyed by agent or ordered by rollout slot.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        values = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, *self.action_space.shape)
        if values.shape != expected:
            raise ValueError(f"Expected action batch {expected}, got {values.shape}")
        self._actions = values
        if self._parallel_worlds:
            for world_index, remote in enumerate(self._remotes):
                base = world_index * self.agents_per_world
                remote.send(
                    (
                        "step",
                        {
                            agent_id: values[base + index]
                            for index, agent_id in enumerate(self.agent_ids)
                        },
                    )
                )

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Collect a shared-world step and auto-reset completed worlds.

        Returns SB3's observation, reward, done, and info arrays in flattened
        Robotino-slot order; terminal observations remain in each final info.

        Returns:
            tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]: Result
                of collect a shared-world step and auto-reset completed worlds.

        Raises:
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        if self._actions is None:
            raise RuntimeError("step_wait() called before step_async()")
        flattened: list[dict[str, np.ndarray]] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos_out: list[dict[str, Any]] = []
        if self._parallel_worlds:
            world_results = [self._receive(index) for index in range(self.world_count)]
        else:
            world_results = []
        for world_index in range(self.world_count):
            base = world_index * self.agents_per_world
            if self._parallel_worlds:
                (
                    observations,
                    world_rewards,
                    terminated,
                    truncated,
                    infos,
                    terminal_observations,
                    reset_infos,
                ) = world_results[world_index]
                done = terminal_observations is not None
            else:
                world = self.worlds[world_index]
                action_map = {
                    agent_id: self._actions[base + index]
                    for index, agent_id in enumerate(self.agent_ids)
                }
                observations, world_rewards, terminated, truncated, infos = world.step(
                    action_map
                )
                done = all(terminated.values()) or all(truncated.values())
                terminal_observations = observations
                reset_infos = None
                if done:
                    observations, reset_infos = world.reset()
            for index, agent_id in enumerate(self.agent_ids):
                info = dict(infos[agent_id])
                info["TimeLimit.truncated"] = bool(truncated[agent_id]) and not bool(
                    terminated[agent_id]
                )
                if done:
                    info["terminal_observation"] = terminal_observations[agent_id]
                    self.reset_infos[base + index] = reset_infos[agent_id]
                flattened.append(observations[agent_id])
                rewards.append(float(world_rewards[agent_id]))
                dones.append(done)
                infos_out.append(info)
        self._actions = None
        return (
            self._stack(flattened),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos_out,
        )

    def close(self) -> None:
        """Close local worlds or stop all spawned simulation workers."""

        if self._parallel_worlds:
            for remote in self._remotes:
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for index, process in enumerate(self._processes):
                if process.is_alive():
                    try:
                        self._receive(index)
                    except RuntimeError:
                        pass
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            for remote in self._remotes:
                remote.close()
            self._remotes.clear()
            self._processes.clear()
            return
        for world in self.worlds:
            world.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        """Return attr_name values for the flattened slot indices.

        With indices=None, query every slot. Parallel worlds forward the
        request through worker pipes; the result is a list in slot order.

        Args:
            attr_name: Attribute name to read or assign on selected worlds.
            indices: Selected agent or rollout-slot indices.

        Returns:
            list[Any]: Attr_name values for the flattened slot indices.
        """

        if self._parallel_worlds:
            slots = self._get_indices(indices)
            for index in slots:
                self._remotes[index // self.agents_per_world].send(
                    ("get_attr", attr_name)
                )
            return [
                self._receive(index // self.agents_per_world) for index in slots
            ]
        return [
            getattr(self.worlds[index // self.agents_per_world], attr_name)
            for index in self._get_indices(indices)
        ]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        """Set attr_name to value on worlds owning indices.

        indices=None selects all slots. The method waits for worker
        acknowledgement in parallel mode and returns nothing.

        Args:
            attr_name: Attribute name to read or assign on selected worlds.
            value: Attribute value to assign in the selected worlds.
            indices: Selected agent or rollout-slot indices.
        """

        if self._parallel_worlds:
            slots = self._get_indices(indices)
            for index in slots:
                self._remotes[index // self.agents_per_world].send(
                    ("set_attr", (attr_name, value))
                )
            for index in slots:
                self._receive(index // self.agents_per_world)
            return
        for index in self._get_indices(indices):
            setattr(self.worlds[index // self.agents_per_world], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices=None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call method_name on worlds selected by indices.

        method_args and method_kwargs are forwarded unchanged. Return
        results in flattened slot order; indices=None selects all slots.

        Args:
            method_name: Environment method to invoke on selected worlds.
            indices: Selected agent or rollout-slot indices.
            *method_args: Positional arguments forwarded to the environment method.
            **method_kwargs: Keyword arguments forwarded to the environment method.

        Returns:
            list[Any]: Method results for the selected rollout slots.
        """

        if self._parallel_worlds:
            slots = self._get_indices(indices)
            for index in slots:
                self._remotes[index // self.agents_per_world].send(
                    ("env_method", (method_name, method_args, method_kwargs))
                )
            return [
                self._receive(index // self.agents_per_world) for index in slots
            ]
        return [
            getattr(self.worlds[index // self.agents_per_world], method_name)(
                *method_args, **method_kwargs
            )
            for index in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        """Return false for each selected slot in indices.

        wrapper_class is accepted for the VecEnv interface, but no
        per-slot SB3 wrapper is used by these Robotino worlds.

        Args:
            wrapper_class: Wrapper class queried by the VecEnv interface.
            indices: Selected agent or rollout-slot indices.

        Returns:
            list[bool]: False for each selected slot in indices.
        """

        del wrapper_class
        return [False for _ in self._get_indices(indices)]
