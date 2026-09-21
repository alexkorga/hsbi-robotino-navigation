"""Normalized action conversion shared by training and execution."""

import math
from dataclasses import dataclass
from typing import Sequence

from robotino_fleet.domain.models import VelocityCommand


@dataclass(frozen=True)
class ActionScaler:
    max_forward_mps: float
    max_sideways_mps: float
    max_rotation_rps: float

    def scale(self, action: Sequence[float]) -> VelocityCommand:
        """Return body velocity scaled from normalized action values.

        action holds [vx, vy, omega] in [-1, 1]. Diagonal translation
        is normalized before applying separate forward/sideways speed limits;
        invalid length or nonfinite values raise ValueError.

        Args:
            action: Normalized policy action for the controlled Robotino.

        Returns:
            VelocityCommand: Body velocity scaled from normalized action values.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if len(action) != 3:
            raise ValueError("Robotino action must contain [vx, vy, omega]")
        values = [float(value) for value in action]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Robotino action must be finite")
        x, y, omega = (max(-1.0, min(1.0, value)) for value in values)
        magnitude = math.hypot(x, y)
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude
        return VelocityCommand(
            x * self.max_forward_mps,
            y * self.max_sideways_mps,
            omega * self.max_rotation_rps,
        )
