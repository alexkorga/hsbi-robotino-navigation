"""Run SB3 or recurrent local-target policies for the Robotino fleet."""

import time
from pathlib import Path
from typing import Mapping

import numpy as np

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.learning.evaluation import load_policy_model
from robotino_fleet.learning.observation import (
    BinnedLocalTargetObservationBuilder,
    PoseCorrectedLidar360Memory,
    TemporalObservationHistory,
)
from robotino_fleet.learning.profile import load_policy_profile
from robotino_fleet.navigation.controller import NavigationControlOutput
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle
from robotino_fleet.navigation.local_target import (
    LocalTargetMotionController,
    LocalTargetPlan,
)
from robotino_fleet.navigation.policy_batch import (
    normalize_action_batch,
    stack_observations,
)


class LocalTargetPolicyController:
    """Infer at sensor rate and track the frozen local target at control rate."""

    def __init__(
        self,
        model_path: Path,
        *,
        motion: MotionSettings,
        deterministic: bool = True,
    ) -> None:
        """Load a local-target policy and verify its runtime contract.

        model_path identifies the model and companion profile; motion
        supplies calibrated low-level limits; deterministic selects
        reproducible inference. Mismatched observation/action spaces fail at
        construction rather than at the first fleet command.

        Args:
            model_path: Path to the trained policy artifact.
            motion: Calibrated Robotino speed and acceleration limits.
            deterministic: Whether to choose the policy's deterministic action.

        Raises:
            FileNotFoundError: If a required configuration file or policy artifact is absent.
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        self.model_path = model_path.resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.profile = load_policy_profile(self.model_path)
        if self.profile.action_mode not in {"local-target", "local-target-yaw"}:
            raise ValueError("Model profile is not a local-target policy")
        self.observation_builder = BinnedLocalTargetObservationBuilder(
            profile=self.profile,
        )
        self.observation_history = TemporalObservationHistory(
            self.profile.temporal_frames
        )
        self.lidar_memory_360 = (
            PoseCorrectedLidar360Memory(self.profile)
            if self.profile.uses_pose_corrected_lidar_360
            else None
        )
        self.motion_controller = LocalTargetMotionController(motion, self.profile)
        self.model = load_policy_model(self.model_path, device="cpu")
        if self.profile.uses_recurrent_memory:
            expected_shapes = {
                key: tuple(space.shape)
                for key, space in self.observation_builder.space().spaces.items()
            }
            if self.model.policy.observation_shapes != expected_shapes:
                raise ValueError(
                    "GRU model observation space does not match its profile"
                )
        elif self.model.observation_space != self.observation_builder.space():
            raise ValueError(
                "Model observation space does not match its local-target profile"
            )
        expected_shape = (self.profile.action_dimensions,)
        if tuple(self.model.action_space.shape or ()) != expected_shape:
            raise ValueError(
                "Local-target model action space must contain "
                + ("[x, y, omega]" if self.profile.uses_learned_rotation else "[x, y]")
            )
        self.deterministic = deterministic
        algorithm = "GRU-PPO" if self.profile.uses_recurrent_memory else "PPO"
        self.policy_id = f"{algorithm}:{self.model_path.stem}:{self.profile.name}"
        self._plans: dict[str, LocalTargetPlan] = {}
        self._previous_actions: dict[str, np.ndarray] = {}
        self._last_inference_s: dict[str, float] = {}
        self._last_inference_ms: dict[str, float] = {}
        self._handover_robot_ids: set[str] = set()
        self._recurrent_states: dict[str, np.ndarray] = {}

    def reset(self, robot_id: str) -> None:
        """Forget robot_id's observations, held plan, and GRU memory.

        The fleet calls this on a new goal so temporal policy state does not
        leak from the previous trip. Nothing is returned.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        self.observation_history.reset(robot_id)
        if self.lidar_memory_360 is not None:
            self.lidar_memory_360.reset(robot_id)
        self._plans.pop(robot_id, None)
        self._previous_actions.pop(robot_id, None)
        self._last_inference_s.pop(robot_id, None)
        self._last_inference_ms.pop(robot_id, None)
        self._handover_robot_ids.discard(robot_id)
        self._recurrent_states.pop(robot_id, None)

    def _infer_batch(
        self,
        due: list[tuple[str, RobotState, NavigationGoal]],
        now_s: float,
        robots: Mapping[str, RobotState],
    ) -> dict[str, NavigationControlOutput]:
        """Update held plans for the Robotinos due for policy inference.

        due contains robot ID/state/goal triples in batch order;
        now_s timestamps the new plans; robots supplies peers for
        optional 360-degree LiDAR memory. Return only per-robot fault outputs;
        successful actions are stored for subsequent control ticks.

        Args:
            due: Robotinos whose policy inference interval has elapsed.
            now_s: Current clock time in seconds, used for age and expiry checks.
            robots: Robotino states in the shared world.

        Returns:
            dict[str, NavigationControlOutput]: Updated control outputs for Robotinos due for
                inference.
        """

        observations = []
        for robot_id, robot, goal in due:
            previous = self._previous_actions.get(
                robot_id,
                np.zeros(self.profile.action_dimensions, dtype=np.float32),
            )
            velocity = robot.measured_velocity
            lidar_360 = (
                self.lidar_memory_360.observe(
                    robot,
                    now_s=now_s,
                    robots=tuple(robots.values()),
                )
                if self.lidar_memory_360 is not None
                else None
            )
            current = self.observation_builder.build(
                robot,
                goal,
                velocity=(velocity.vx, velocity.vy, velocity.omega),
                previous_action=previous,
                lidar_360=lidar_360,
            )
            observations.append(
                self.observation_history.augment(robot_id, current)
            )
        started = time.perf_counter()
        stacked = stack_observations(observations)
        if self.profile.uses_recurrent_memory:
            hidden_size = self.profile.recurrent_hidden_size
            recurrent_state = np.stack(
                [
                    self._recurrent_states.get(
                        robot_id,
                        np.zeros(hidden_size, dtype=np.float32),
                    )
                    for robot_id, _, _ in due
                ],
                axis=0,
            )[None, ...]
            episode_starts = np.asarray(
                [robot_id not in self._recurrent_states for robot_id, _, _ in due],
                dtype=bool,
            )
            raw, next_recurrent_state = self.model.predict(
                stacked,
                state=recurrent_state,
                episode_start=episode_starts,
                deterministic=self.deterministic,
            )
            for index, (robot_id, _, _) in enumerate(due):
                self._recurrent_states[robot_id] = next_recurrent_state[0, index]
        else:
            raw, _ = self.model.predict(
                stacked,
                deterministic=self.deterministic,
            )
        elapsed = time.perf_counter() - started
        action_dimensions = self.profile.action_dimensions
        try:
            actions = normalize_action_batch(
                raw, robot_count=len(due), action_dimensions=action_dimensions
            )
        except ValueError as error:
            return {
                robot_id: NavigationControlOutput(
                    VelocityCommand(),
                    NavigationLifecycle.FAULT,
                    reason=str(error),
                    inference_ms=elapsed * 1000,
                    policy_id=self.policy_id,
                )
                for robot_id, _, _ in due
            }
        faults: dict[str, NavigationControlOutput] = {}
        inference_ms = elapsed * 1000
        for index, (robot_id, robot, _) in enumerate(due):
            action = actions[index]
            if not np.all(np.isfinite(action)):
                faults[robot_id] = NavigationControlOutput(
                    VelocityCommand(),
                    NavigationLifecycle.FAULT,
                    reason="policy produced an invalid local target",
                    inference_ms=inference_ms,
                    policy_id=self.policy_id,
                )
                continue
            action = np.clip(action, -1.0, 1.0)
            self._previous_actions[robot_id] = action
            self._plans[robot_id] = self.motion_controller.plan(robot, action)
            self._last_inference_s[robot_id] = now_s
            self._last_inference_ms[robot_id] = inference_ms
        return faults

    def compute_command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        delta_s: float,
        now_s: float,
    ) -> NavigationControlOutput:
        """Return one robot's command through the shared batched path.

        robot and goal form a one-element fleet; delta_s and
        now_s retain the common controller timing interface. The returned
        output carries command, lifecycle, and inference telemetry.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            NavigationControlOutput: One robot's command through the shared batched path.
        """

        return self.compute_commands(
            {robot.id: robot},
            {robot.id: goal},
            delta_s=delta_s,
            now_s=now_s,
        )[robot.id]

    def compute_commands(
        self,
        robots: Mapping[str, RobotState],
        goals: Mapping[str, NavigationGoal],
        *,
        delta_s: float,
        now_s: float,
    ) -> dict[str, NavigationControlOutput]:
        """Return one command for every matched robots/goals ID.

        now_s schedules batched inference; delta_s is accepted for the
        common controller interface but is not used. Held local plans run
        between inferences, and positional handover remains active until goal
        alignment. Mismatched robot/goal IDs raise ValueError.

        Args:
            robots: Robotino states in the shared world.
            goals: Navigation targets for the active agents.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: One command for every matched robots/goals ID.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        del delta_s
        if set(robots) != set(goals):
            raise ValueError("robots and goals must contain the same robot IDs")
        due: list[tuple[str, RobotState, NavigationGoal]] = []
        for robot_id, robot in robots.items():
            distance = robot.point.distance_to(goals[robot_id].point)
            if (
                robot_id in self._handover_robot_ids
                or distance <= goals[robot_id].position_tolerance_m
            ):
                self._handover_robot_ids.add(robot_id)
                continue
            last = self._last_inference_s.get(robot_id)
            inference_due = (
                last is None
                or now_s < last
                or now_s - last >= self.profile.policy_period_s - 1e-9
            )
            if (
                inference_due
                and (
                    self.profile.uses_learned_rotation
                    or distance > self.profile.approach_radius_m
                )
            ):
                due.append((robot_id, robot, goals[robot_id]))
        faults = self._infer_batch(due, now_s, robots) if due else {}
        outputs: dict[str, NavigationControlOutput] = {}
        for robot_id, robot in robots.items():
            if robot_id in faults:
                outputs[robot_id] = faults[robot_id]
                continue
            plan = self._plans.get(robot_id)
            if plan is None:
                plan = self.motion_controller.plan(
                    robot,
                    (0.0,) * self.profile.action_dimensions,
                )
                self._plans[robot_id] = plan
            outputs[robot_id] = self.motion_controller.command(
                robot,
                goals[robot_id],
                plan,
                handover_active=robot_id in self._handover_robot_ids,
                policy_id=self.policy_id,
                inference_ms=self._last_inference_ms.get(robot_id),
            )
        return outputs
