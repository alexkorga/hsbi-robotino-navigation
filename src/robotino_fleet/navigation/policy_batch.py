"""Shared shape handling for batched navigation-policy inference."""

from collections.abc import Mapping, Sequence

import numpy as np


def stack_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build a model batch from observations in robot order.

    Each mapping must contain the same observation keys and compatible array
    shapes. Return arrays with a leading batch dimension for the shared policy
    call. The caller must provide at least one observation.

    Args:
        observations: Policy observations for one or more agents.

    Returns:
        dict[str, np.ndarray]: Observation arrays stacked in Robotino order.
    """

    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def normalize_action_batch(
    raw: object, *, robot_count: int, action_dimensions: int
) -> np.ndarray:
    """Normalize raw model output to one action row per Robotino.

    robot_count and action_dimensions define the required (N, D)
    shape. A single-robot (D,) output is accepted. Return float32
    actions or raise ValueError when the model contract differs.

    Args:
        raw: Configuration mapping containing the field to validate.
        robot_count: Normalize raw model output to one action row per Robotino. robot_count
            and action_dimensions define the required (N, D) shape.
        action_dimensions: Number of normalized actions emitted per Robotino.

    Returns:
        np.ndarray: Finite action rows, one per Robotino.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    actions = np.asarray(raw, dtype=np.float32)
    if actions.shape == (action_dimensions,) and robot_count == 1:
        actions = actions.reshape(1, action_dimensions)
    expected = (robot_count, action_dimensions)
    if actions.shape != expected:
        raise ValueError(
            f"policy produced action batch shape {actions.shape}, expected {expected}"
        )
    return actions
