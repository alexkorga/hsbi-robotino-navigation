"""The one velocity limiting implementation used by simulation and hardware."""

import math

from robotino_fleet.config import MotionSettings
from robotino_fleet.domain.models import VelocityCommand


class MotionLimiter:
    """Apply shared calibrated speed, acceleration, and deadband limits."""

    def __init__(self, settings: MotionSettings) -> None:
        """Use calibrated settings to retain per-Robotino command history.

        Args:
            settings: Configuration settings for this component.
        """

        self.settings = settings
        self._state: dict[str, VelocityCommand] = {}

    def reset(self, robot_id: str | None = None) -> None:
        """Forget the previous command for robot_id, or for every robot.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
        """

        if robot_id is None:
            self._state.clear()
        else:
            self._state.pop(robot_id, None)

    def stop(self, robot_id: str) -> VelocityCommand:
        """Store and return an immediate zero command for robot_id.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.

        Returns:
            VelocityCommand: Immediate zero-velocity command stored for the Robotino.
        """

        command = VelocityCommand()
        self._state[robot_id] = command
        return command

    def apply(
        self, robot_id: str, desired: VelocityCommand, delta_s: float
    ) -> VelocityCommand:
        """Limit desired for robot_id over delta_s seconds.

        Returns a body-frame command after speed, acceleration/deceleration,
        rotational-rate, and deadband limits. Previous pre-deadband command
        state is kept by Robotino ID; nonfinite input returns an immediate zero
        command instead of passing invalid numbers to hardware.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            desired: Limit desired for robot_id over delta_s seconds.
            delta_s: Elapsed control or simulation time in seconds.

        Returns:
            VelocityCommand: Acceleration- and speed-limited velocity command.
        """

        values = (desired.vx, desired.vy, desired.omega)
        if not all(math.isfinite(value) for value in values):
            return self.stop(robot_id)
        dt = max(0.0, float(delta_s))
        translation = math.hypot(
            desired.vx / self.settings.forward_max_mps,
            desired.vy / self.settings.sideways_max_mps,
        )
        if translation > 1.0:
            scale = 1.0 / translation
            desired = VelocityCommand(
                desired.vx * scale,
                desired.vy * scale,
                desired.omega,
            )
        desired = VelocityCommand(
            desired.vx,
            desired.vy,
            max(
                -self.settings.rotation_max_rps,
                min(self.settings.rotation_max_rps, desired.omega),
            ),
        )
        previous = self._state.get(robot_id, VelocityCommand())
        dvx = desired.vx - previous.vx
        dvy = desired.vy - previous.vy

        forward_accelerating = (
            desired.vx * previous.vx >= 0.0
            and abs(desired.vx) >= abs(previous.vx)
        )
        sideways_accelerating = (
            desired.vy * previous.vy >= 0.0
            and abs(desired.vy) >= abs(previous.vy)
        )
        forward_rate = (
            self.settings.forward_acceleration_mps2
            if forward_accelerating
            else self.settings.forward_deceleration_mps2
        )
        sideways_rate = (
            self.settings.sideways_acceleration_mps2
            if sideways_accelerating
            else self.settings.sideways_deceleration_mps2
        )
        forward_change = forward_rate * dt
        sideways_change = sideways_rate * dt
        if dt == 0.0:
            dvx = 0.0
            dvy = 0.0
        else:
            normalized_change = math.hypot(
                dvx / forward_change,
                dvy / sideways_change,
            )
            if normalized_change > 1.0:
                scale = 1.0 / normalized_change
                dvx *= scale
                dvy *= scale
        angular_rate = (
            self.settings.rotation_acceleration_rps2
            if abs(desired.omega) >= abs(previous.omega)
            and desired.omega * previous.omega >= 0.0
            else self.settings.rotation_deceleration_rps2
        )
        angular_change = max(
            -angular_rate * dt,
            min(angular_rate * dt, desired.omega - previous.omega),
        )
        state = VelocityCommand(
            previous.vx + dvx,
            previous.vy + dvy,
            previous.omega + angular_change,
        )
        self._state[robot_id] = state
        result = state
        if abs(result.vx) < self.settings.forward_deadband_mps:
            result = VelocityCommand(0.0, result.vy, result.omega)
        if abs(result.vy) < self.settings.sideways_deadband_mps:
            result = VelocityCommand(result.vx, 0.0, result.omega)
        if abs(result.omega) < self.settings.rotation_deadband_rps:
            result = VelocityCommand(result.vx, result.vy, 0.0)
        return result
