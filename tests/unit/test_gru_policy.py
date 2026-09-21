"""Check recurrent policy loading, inference state, and model contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from robotino_fleet.domain.models import Point, RobotState
from robotino_fleet.learning.gru_policy import GruActorCritic, GruPolicyModel
from robotino_fleet.learning.profile import (
    BinnedLocalTargetProfile,
    write_policy_profile,
)
from robotino_fleet.navigation.goals import NavigationGoal
from robotino_fleet.navigation.local_target_policy_controller import (
    LocalTargetPolicyController,
)
from tests.fixtures import TEST_MOTION


class GruPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        """Prepare model-compatible shapes and a two-agent observation batch."""

        self.shapes = {
            "lidar_closeness": (48,),
            "lidar_validity": (48,),
            "proximity_closeness": (9,),
            "goal_body": (2,),
            "goal_distance": (1,),
            "velocity": (3,),
            "previous_action": (2,),
        }
        self.observation = {
            key: np.full((2, *shape), 0.1, dtype=np.float32)
            for key, shape in self.shapes.items()
        }

    def test_architecture_is_near_one_million_parameters(self) -> None:
        """Keep the configured GRU within the intended model-size range."""

        policy = GruActorCritic(self.shapes)
        parameters = sum(value.numel() for value in policy.parameters())
        self.assertGreater(parameters, 950_000)
        self.assertLess(parameters, 1_150_000)

    def test_predict_keeps_and_independently_resets_agent_memory(self) -> None:
        """Carry independent hidden states and reset them at episode starts."""

        torch.manual_seed(4)
        model = GruPolicyModel(GruActorCritic(self.shapes), device="cpu")
        first_actions, first_state = model.predict(
            self.observation,
            episode_start=np.ones(2, dtype=bool),
        )
        _, second_state = model.predict(
            self.observation,
            state=first_state,
            episode_start=np.zeros(2, dtype=bool),
        )
        reset_actions, reset_state = model.predict(
            self.observation,
            state=second_state,
            episode_start=np.ones(2, dtype=bool),
        )

        self.assertEqual(first_actions.shape, (2, 2))
        self.assertEqual(first_state.shape, (1, 2, 288))
        self.assertFalse(np.allclose(first_state, second_state))
        np.testing.assert_allclose(reset_actions, first_actions, atol=1e-6)
        np.testing.assert_allclose(reset_state, first_state, atol=1e-6)

    def test_artifact_round_trip_preserves_deterministic_output(self) -> None:
        """Preserve inference output when saving and reloading a GRU model."""

        model = GruPolicyModel(GruActorCritic(self.shapes), device="cpu")
        expected, _ = model.predict(
            self.observation,
            episode_start=np.ones(2, dtype=bool),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.zip"
            model.save(path)
            loaded = GruPolicyModel.load(path, device="cpu")
            actual, _ = loaded.predict(
                self.observation,
                episode_start=np.ones(2, dtype=bool),
            )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_profile_marks_current_frame_recurrent_contract(self) -> None:
        """Declare single-frame input plus recurrent memory in the GRU profile."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-gru-v1",
            schema_version=7,
            temporal_frames=1,
            recurrent_hidden_size=288,
        )
        self.assertTrue(profile.uses_recurrent_memory)
        self.assertEqual(profile.action_dimensions, 2)

    def test_runtime_dynamically_batches_and_resets_robot_memory(self) -> None:
        """Batch due robot inference and clear hidden state on goal reset."""

        profile = BinnedLocalTargetProfile(
            name="binned-local-target-gru-v1",
            schema_version=7,
            temporal_frames=1,
            recurrent_hidden_size=288,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.zip"
            GruPolicyModel(GruActorCritic(self.shapes), device="cpu").save(path)
            write_policy_profile(path, profile)
            controller = LocalTargetPolicyController(
                path,
                motion=TEST_MOTION,
            )
            robots = {
                robot_id: RobotState(robot_id, "#fff", 0.0, float(index), 0.0)
                for index, robot_id in enumerate(("r1", "r2"))
            }
            goals = {
                robot_id: NavigationGoal(
                    f"g-{robot_id}", Point(3.0, robot.y), None, 0.1, 0.15
                )
                for robot_id, robot in robots.items()
            }

            controller.compute_commands(robots, goals, delta_s=0.05, now_s=0.0)
            self.assertEqual(set(controller._recurrent_states), {"r1", "r2"})
            controller.reset("r1")
            self.assertEqual(set(controller._recurrent_states), {"r2"})


if __name__ == "__main__":
    unittest.main()
