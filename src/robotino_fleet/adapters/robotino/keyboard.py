"""Live Robotino keyboard state on Windows with a terminal fallback."""

import math
import os
import select
import sys

from robotino_fleet.adapters.robotino.http_api import Velocity


VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_RETURN = 0x0D


class Keyboard:
    """Read held keys on Windows or individual terminal keys elsewhere."""

    def __init__(self) -> None:
        """Select held-key polling on Windows or terminal input elsewhere."""

        self.has_live_key_state = os.name == "nt"
        if self.has_live_key_state:
            import ctypes

            self._get_key_state = ctypes.windll.user32.GetAsyncKeyState

    def __enter__(self):
        """Enable nonblocking terminal key reads and return this keyboard.

        Returns:
            object: Keyboard reader with nonblocking terminal input enabled.
        """

        if not self.has_live_key_state:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_):
        """Restore terminal settings on context exit; Windows needs no reset.

        Args:
            *_: Unused framework-provided argument.
        """

        if not self.has_live_key_state:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def is_down(self, key: str | int) -> bool:
        """Return the current held-key state on Windows, otherwise false.

        Args:
            key: Name of the field to read or validate.

        Returns:
            bool: The current held-key state on Windows, otherwise false.
        """

        if not self.has_live_key_state:
            return False
        virtual_key = ord(key.upper()) if isinstance(key, str) else key
        return bool(self._get_key_state(virtual_key) & 0x8000)

    def read(self) -> str | None:
        """Return one available non-Windows key, or None, without blocking.

        Returns:
            str | None: One available non-Windows key, or None, without blocking.
        """

        if self.has_live_key_state:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1).lower() if ready else None


def requested_velocity(
    keyboard: Keyboard,
    translation_speed_mps: float,
    rotation_speed_rps: float,
) -> Velocity:
    """Return a holonomic velocity from keys held on keyboard.

    translation_speed_mps caps planar speed, including diagonals;
    rotation_speed_rps scales Q/E yaw. Space overrides all keys with stop.

    Args:
        keyboard: Live keyboard reader tracking currently held keys.
        translation_speed_mps: Manual translation speed in meters per second.
        rotation_speed_rps: Manual turn speed in radians per second.

    Returns:
        Velocity: A holonomic velocity from keys held on keyboard.
    """

    if keyboard.is_down(VK_SPACE):
        return Velocity(0.0, 0.0, 0.0)

    vx = translation_speed_mps * (
        keyboard.is_down("W") - keyboard.is_down("S")
    )
    vy = translation_speed_mps * (
        keyboard.is_down("A") - keyboard.is_down("D")
    )
    omega = rotation_speed_rps * (
        keyboard.is_down("Q") - keyboard.is_down("E")
    )

    magnitude = math.hypot(vx, vy)
    if magnitude > translation_speed_mps:
        scale = translation_speed_mps / magnitude
        vx *= scale
        vy *= scale

    return Velocity(vx, vy, omega)
