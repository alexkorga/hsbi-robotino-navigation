"""SB3 inference for the six-frame forward-and-turn policy."""

import time
from pathlib import Path
from typing import Mapping

import numpy as np

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.learning.evaluation import load_sb3_model
from robotino_fleet.learning.observation import (
    BinnedLocalTargetObservationBuilder,
    TemporalObservationHistory,
)
from robotino_fleet.learning.profile import load_policy_profile
from robotino_fleet.navigation.controller import NavigationControlOutput
from robotino_fleet.navigation.forward_turn import ForwardTurnMotionController
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle
from robotino_fleet.navigation.policy_batch import (
    normalize_action_batch,
    stack_observations,
)


class Sb3ForwardTurnNavigationController:
    """Dynamically batch six-frame inference and hold commands at 20 Hz."""

    def __init__(
        self,
        model_path: Path,
        *,
        motion: MotionSettings,
        deterministic: bool = True,
    ) -> None:
        """Load model_path and its matching forward-turn profile.

        motion bounds output velocities; deterministic controls SB3
        prediction sampling. Reject incompatible observation or action spaces
        before the controller enters the fleet runtime.

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
        if self.profile.action_mode != "forward-turn":
            raise ValueError("Model profile is not a forward-turn policy")
        self.observation_builder = BinnedLocalTargetObservationBuilder(self.profile)
        self.observation_history = TemporalObservationHistory(
            self.profile.temporal_frames
        )
        self.motion_controller = ForwardTurnMotionController(motion, self.profile)
        self.model = load_sb3_model(self.model_path, device="cpu")
        if self.model.observation_space != self.observation_builder.space():
            raise ValueError(
                "Model observation space does not match its forward-turn profile"
            )
        if tuple(self.model.action_space.shape or ()) != (2,):
            raise ValueError("Forward-turn action space must contain [drive, turn]")
        self.deterministic = deterministic
        self.policy_id = f"PPO:{self.model_path.stem}:{self.profile.name}"
        self._actions: dict[str, np.ndarray] = {}
        self._last_inference_s: dict[str, float] = {}
        self._last_inference_ms: dict[str, float] = {}
        self._handover_robot_ids: set[str] = set()

    def reset(self, robot_id: str) -> None:
        """Clear robot_id's temporal frames and held action for a new trip.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        self.observation_history.reset(robot_id)
        self._actions.pop(robot_id, None)
        self._last_inference_s.pop(robot_id, None)
        self._last_inference_ms.pop(robot_id, None)
        self._handover_robot_ids.discard(robot_id)

    def _infer_batch(
        self,
        due: list[tuple[str, RobotState, NavigationGoal]],
        now_s: float,
    ) -> dict[str, NavigationControlOutput]:
        """Infer new forward/turn actions for due robots in one batch.

        due holds ID/state/goal triples in policy batch order and
        now_s records their inference time. Valid actions update held
        controller state. Return per-robot fault outputs for invalid model
        shape or nonfinite actions; successful robots are omitted.

        Args:
            due: Robotinos whose policy inference interval has elapsed.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: New forward/turn outputs for Robotinos due for
                inference.
        """

        observations = []
        for robot_id, robot, goal in due:
            previous = self._actions.get(robot_id, np.zeros(2, dtype=np.float32))
            velocity = robot.measured_velocity
            current = self.observation_builder.build(
                robot,
                goal,
                velocity=(velocity.vx, velocity.vy, velocity.omega),
                previous_action=previous,
            )
            observations.append(self.observation_history.augment(robot_id, current))
        started = time.perf_counter()
        raw, _ = self.model.predict(
            stack_observations(observations),
            deterministic=self.deterministic,
        )
        elapsed = time.perf_counter() - started
        try:
            actions = normalize_action_batch(
                raw, robot_count=len(due), action_dimensions=2
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
        for index, (robot_id, _, _) in enumerate(due):
            action = actions[index]
            if not np.all(np.isfinite(action)):
                faults[robot_id] = NavigationControlOutput(
                    VelocityCommand(),
                    NavigationLifecycle.FAULT,
                    reason="policy produced an invalid forward-turn action",
                    inference_ms=inference_ms,
                    policy_id=self.policy_id,
                )
                continue
            self._actions[robot_id] = np.clip(action, -1.0, 1.0)
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
        """Return robot's output toward goal via the batch interface.

        delta_s and now_s preserve the fleet control clock contract.
        The returned output includes the command, lifecycle, and policy timing.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            NavigationControlOutput: Robot's output toward goal via the batch interface.
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
        """Compute matched robots/goals commands at now_s.

        Due robots are inferred together; prior actions are held between
        policy periods. delta_s is part of the controller interface but
        unused here. Return one output per robot or raise ValueError for
        mismatched ID sets. Positional arrival starts deterministic handover.

        Args:
            robots: Robotino states in the shared world.
            goals: Navigation targets for the active agents.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: Control outputs keyed by Robotino IP.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        del delta_s
        if set(robots) != set(goals):
            raise ValueError("robots and goals must contain the same robot IDs")
        due = []
        for robot_id, robot in robots.items():
            distance = robot.point.distance_to(goals[robot_id].point)
            if (
                robot_id in self._handover_robot_ids
                or distance <= goals[robot_id].position_tolerance_m
            ):
                self._handover_robot_ids.add(robot_id)
                continue
            last = self._last_inference_s.get(robot_id)
            if (
                (
                    last is None
                    or now_s < last
                    or now_s - last >= self.profile.policy_period_s - 1e-9
                )
            ):
                due.append((robot_id, robot, goals[robot_id]))
        faults = self._infer_batch(due, now_s) if due else {}
        outputs = {}
        for robot_id, robot in robots.items():
            if robot_id in faults:
                outputs[robot_id] = faults[robot_id]
                continue
            outputs[robot_id] = self.motion_controller.command(
                robot,
                goals[robot_id],
                self._actions.get(robot_id, np.zeros(2, dtype=np.float32)),
                handover_active=robot_id in self._handover_robot_ids,
                policy_id=self.policy_id,
                inference_ms=self._last_inference_ms.get(robot_id),
            )
        return outputs
