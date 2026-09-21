"""Verify keyboard input produces live manual-control commands."""

import unittest

from robotino_fleet.adapters.robotino.keyboard import (
    VK_SPACE,
    requested_velocity,
)


class FakeKeyboard:
    def __init__(self, *keys: str | int) -> None:
        """Treat keys as the held keyboard state for a test tick.

        Args:
            *keys: Treat keys as the held keyboard state for a test tick.
        """

        self.keys = set(keys)

    def is_down(self, key: str | int) -> bool:
        """Return whether key belongs to the fake held-key set.

        Args:
            key: Name of the field to read or validate.

        Returns:
            bool: Whether key belongs to the fake held-key set.
        """

        return key in self.keys


class KeyboardTests(unittest.TestCase):
    def test_held_keys_create_live_holonomic_velocity(self) -> None:
        """Combine diagonal translation and rotation while honoring speed cap."""

        command = requested_velocity(FakeKeyboard("W", "A", "Q"), 0.2, 0.4)
        self.assertAlmostEqual(command.vx, 0.2 / 2**0.5)
        self.assertAlmostEqual(command.vy, 0.2 / 2**0.5)
        self.assertEqual(command.omega, 0.4)

    def test_release_and_space_stop_immediately(self) -> None:
        """Verify released keys and Space both request zero physical velocity."""

        released = requested_velocity(FakeKeyboard(), 0.2, 0.4)
        stopped = requested_velocity(FakeKeyboard("W", VK_SPACE), 0.2, 0.4)
        self.assertEqual(released.as_payload(), [0.0, 0.0, 0.0])
        self.assertEqual(stopped.as_payload(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
