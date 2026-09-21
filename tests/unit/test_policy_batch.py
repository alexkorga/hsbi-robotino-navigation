"""Check shared observation and action shape handling for fleet policies."""

import unittest

import numpy as np

from robotino_fleet.navigation.policy_batch import (
    normalize_action_batch,
    stack_observations,
)


class PolicyBatchTests(unittest.TestCase):
    def test_stack_observations_preserves_robot_order(self) -> None:
        """Batch observations without changing the order of Robotino rows."""

        observations = [
            {"goal": np.asarray([1.0, 2.0], dtype=np.float32)},
            {"goal": np.asarray([3.0, 4.0], dtype=np.float32)},
        ]
        stacked = stack_observations(observations)
        np.testing.assert_array_equal(
            stacked["goal"], np.asarray([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_single_action_vector_becomes_one_row(self) -> None:
        """Accept an unbatched action vector for a one-robot policy call."""

        actions = normalize_action_batch(
            [0.25, -0.5], robot_count=1, action_dimensions=2
        )
        self.assertEqual(actions.shape, (1, 2))

    def test_wrong_action_shape_is_rejected(self) -> None:
        """Reject model output that cannot provide one action per robot."""

        with self.assertRaisesRegex(ValueError, "expected \\(2, 2\\)"):
            normalize_action_batch(
                [0.25, -0.5], robot_count=2, action_dimensions=2
            )


if __name__ == "__main__":
    unittest.main()
