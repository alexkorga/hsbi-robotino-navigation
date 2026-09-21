"""Stable-Baselines3 PPO inference without hidden rate limiting."""

import time
from pathlib import Path
from typing import Mapping

import numpy as np

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import RobotState, VelocityCommand
from robotino_fleet.learning.actions import ActionScaler
from robotino_fleet.learning.evaluation import load_sb3_model
from robotino_fleet.learning.observation import ModelObservationBuilder
from robotino_fleet.navigation.controller import (
    NavigationControlOutput,
    handover_alignment_output,
)
from robotino_fleet.navigation.goals import NavigationGoal, NavigationLifecycle
from robotino_fleet.simulation.sensors import (
    DistanceSensorSimulationSettings,
    LidarSimulationSettings,
)


def runtime_observation_builder(
    motion: MotionSettings, max_goal_distance_m: float = 20.0
) -> ModelObservationBuilder:
    """Build the direct-velocity policy's runtime observation contract.

    motion supplies velocity normalization limits and
    max_goal_distance_m bounds goal-distance features. Return a builder
    using the current simulated LiDAR/proximity sensor dimensions.

    Args:
        motion: Calibrated Robotino speed and acceleration limits.
        max_goal_distance_m: Distance used to normalize goal features, in meters.

    Returns:
        ModelObservationBuilder: Constructed the direct-velocity policy's runtime
            observation contract.
    """

    lidar = LidarSimulationSettings()
    distance = DistanceSensorSimulationSettings()
    return ModelObservationBuilder(
        lidar_beam_count=lidar.beam_count,
        distance_sensor_count=distance.count,
        lidar_range_max_m=lidar.range_max_m,
        distance_clear_value_m=distance.clear_value_m,
        max_goal_distance_m=max_goal_distance_m,
        max_forward_mps=motion.forward_max_mps,
        max_sideways_mps=motion.sideways_max_mps,
        max_rotation_rps=motion.rotation_max_rps,
    )


class Sb3NavigationController:
    """Run a direct-velocity SB3 policy contract."""

    def __init__(
        self,
        model_path: Path,
        *,
        motion: MotionSettings,
        deterministic: bool = True,
    ) -> None:
        """Load a direct-velocity model with compatible runtime spaces.

        model_path selects the SB3 archive, motion supplies action and
        observation scaling, and deterministic controls prediction
        sampling. Reject missing models and incompatible spaces on startup.

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
        self.deterministic = deterministic
        self.observation_builder = runtime_observation_builder(motion)
        self.action_scaler = ActionScaler(
            motion.forward_max_mps,
            motion.sideways_max_mps,
            motion.rotation_max_rps,
        )
        self.model = load_sb3_model(self.model_path, device="cpu")
        if self.model.observation_space != self.observation_builder.space():
            raise ValueError("Model observation space does not match the Robotino runtime")
        if tuple(self.model.action_space.shape or ()) != (3,):
            raise ValueError("Model action space must be a three-value body velocity")
        self.policy_id = f"PPO:{self.model_path.stem}"
        self._previous_actions: dict[str, np.ndarray] = {}
        self._handover_robot_ids: set[str] = set()

    def reset(self, robot_id: str) -> None:
        """Clear robot_id's prior action and handover flag on a new goal.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        self._previous_actions.pop(robot_id, None)
        self._handover_robot_ids.discard(robot_id)

    def compute_command(
        self,
        robot: RobotState,
        goal: NavigationGoal,
        *,
        delta_s: float,
        now_s: float,
    ) -> NavigationControlOutput:
        """Infer robot's body-frame velocity toward goal.

        delta_s and now_s are accepted for the controller interface;
        this model infers on every call. At position tolerance, return
        deterministic heading handover. Otherwise return a scaled command or
        a fault output if the policy action is invalid.

        Args:
            robot: Robotino state used by this operation.
            goal: Assigned navigation target.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            NavigationControlOutput: Body-frame command for direct goal tracking.
        """

        del delta_s, now_s
        distance = robot.point.distance_to(goal.point)
        if robot.id in self._handover_robot_ids or distance <= goal.position_tolerance_m:
            self._handover_robot_ids.add(robot.id)
            return handover_alignment_output(
                robot,
                goal,
                rotation_max_rps=self.action_scaler.max_rotation_rps,
                policy_id=self.policy_id,
            )
        previous = self._previous_actions.get(robot.id, np.zeros(3, dtype=np.float32))
        velocity = robot.measured_velocity
        observation = self.observation_builder.build(
            robot,
            goal,
            velocity=(velocity.vx, velocity.vy, velocity.omega),
            previous_action=previous,
        )
        started = time.perf_counter()
        raw, _ = self.model.predict(observation, deterministic=self.deterministic)
        elapsed = time.perf_counter() - started
        action = np.asarray(raw, dtype=np.float32).reshape(-1)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            return NavigationControlOutput(
                VelocityCommand(),
                NavigationLifecycle.FAULT,
                reason="policy produced an invalid action",
                inference_ms=elapsed * 1000,
                policy_id=self.policy_id,
            )
        action = np.clip(action, -1.0, 1.0)
        self._previous_actions[robot.id] = action
        command = self.action_scaler.scale(action)
        state = (
            NavigationLifecycle.APPROACHING
            if distance <= 0.5
            else NavigationLifecycle.NAVIGATING
        )
        return NavigationControlOutput(
            command,
            state,
            inference_ms=elapsed * 1000,
            policy_id=self.policy_id,
        )

    def compute_commands(
        self,
        robots: Mapping[str, RobotState],
        goals: Mapping[str, NavigationGoal],
        *,
        delta_s: float,
        now_s: float,
    ) -> dict[str, NavigationControlOutput]:
        """Return commands for robots using ID-matched goals.

        delta_s and now_s are passed to each single-robot call. This
        direct-velocity controller does not batch model inference; the result
        maps each robot ID to its control output.

        Args:
            robots: Robotino states in the shared world.
            goals: Navigation targets for the active agents.
            delta_s: Elapsed control or simulation time in seconds.
            now_s: Current clock time in seconds, used for age and expiry checks.

        Returns:
            dict[str, NavigationControlOutput]: Commands for robots using ID-matched goals.
        """

        return {
            robot_id: self.compute_command(
                robot,
                goals[robot_id],
                delta_s=delta_s,
                now_s=now_s,
            )
            for robot_id, robot in robots.items()
        }
