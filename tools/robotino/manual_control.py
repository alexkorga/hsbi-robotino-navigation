"""Cross-platform WASDQE control and compact live telemetry."""

import sys
import time

from robotino_fleet.adapters.robotino.http_api import RobotinoHttpClient, Velocity
from robotino_fleet.adapters.robotino.keyboard import (
    Keyboard,
    VK_ESCAPE,
    requested_velocity,
)


ROBOTINO_IP = "172.21.22.90"
TRANSLATION_SPEED_MPS = 0.4
ROTATION_SPEED_RPS = 0.40
COMMAND_HOLD_S = 0.20


def command_for(key: str | None) -> Velocity:
    """Map one terminal key to a short manual body-velocity command.

    Unknown or absent keys return zero velocity; held-key control uses the
    shared keyboard adapter instead.

    Args:
        key: Name of the field to read or validate.

    Returns:
        Velocity: Manual body-velocity command for the pressed key.
    """

    return {
        "w": Velocity(TRANSLATION_SPEED_MPS, 0.0, 0.0),
        "s": Velocity(-TRANSLATION_SPEED_MPS, 0.0, 0.0),
        "a": Velocity(0.0, TRANSLATION_SPEED_MPS, 0.0),
        "d": Velocity(0.0, -TRANSLATION_SPEED_MPS, 0.0),
        "q": Velocity(0.0, 0.0, ROTATION_SPEED_RPS),
        "e": Velocity(0.0, 0.0, -ROTATION_SPEED_RPS),
        " ": Velocity(0.0, 0.0, 0.0),
    }.get(key, Velocity(0.0, 0.0, 0.0))


def sensor_lines(values: tuple[float, ...]) -> list[str]:
    """Format nine values as a clockwise Robotino sensor diagram.

    Return fixed-width CLI lines with S0 at the front/top and S1–S8 clockwise.

    Args:
        values: Measured or computed values supplied to this operation.

    Returns:
        list[str]: Clockwise text diagram of the nine distance sensors.
    """

    text = [f"{value:.3f}" for value in values]
    return [
        f"                         S0 {text[0]}",
        f"              S8 {text[8]}       S1 {text[1]}",
        f"    S7 {text[7]}                         S2 {text[2]}",
        "                    +----------+",
        "                    | Robotino |",
        "                    +----------+",
        f"    S6 {text[6]}                         S3 {text[3]}",
        f"              S5 {text[5]}       S4 {text[4]}",
    ]


def stop(client: RobotinoHttpClient) -> None:
    """Try three zero-velocity sends through client before exiting.

    Args:
        client: HTTP client for the selected physical Robotino.
    """

    for _ in range(3):
        try:
            client.set_velocity(Velocity(0.0, 0.0, 0.0))
        except Exception:
            pass
        time.sleep(0.05)


def main() -> None:
    """Drive the configured Robotino from live keyboard input and show telemetry."""

    client = RobotinoHttpClient(ROBOTINO_IP)
    command = Velocity(0.0, 0.0, 0.0)
    print(f"Robotino manual control + telemetry: {ROBOTINO_IP}")
    print("Tap/hold W/S forward/back | A/D left/right | Q/E rotate")
    print("Movement stops after release | Space stop | Esc exit")
    command_expires_at = 0.0
    try:
        with Keyboard() as keyboard:
            while True:
                now = time.monotonic()
                if keyboard.has_live_key_state:
                    if keyboard.is_down(VK_ESCAPE):
                        break
                    command = requested_velocity(
                        keyboard,
                        TRANSLATION_SPEED_MPS,
                        ROTATION_SPEED_RPS,
                    )
                else:
                    key = keyboard.read()
                    if key == "\x1b":
                        break
                    if key is not None:
                        command = command_for(key)
                        command_expires_at = (
                            now + COMMAND_HOLD_S
                            if command != Velocity(0.0, 0.0, 0.0)
                            else 0.0
                        )
                    elif now >= command_expires_at:
                        command = Velocity(0.0, 0.0, 0.0)
                bumper = client.get_bumper()
                if bumper:
                    command = Velocity(0.0, 0.0, 0.0)
                    command_expires_at = 0.0
                applied = command
                client.set_velocity(applied)
                odometry = client.get_odometry()
                sensors = client.get_distance_sensors()
                battery_low = client.get_battery_low()
                lines = [
                    f"Requested: vx={command.vx:+.3f}  vy={command.vy:+.3f}  omega={command.omega:+.3f}",
                    f"Applied:   vx={applied.vx:+.3f}  vy={applied.vy:+.3f}  omega={applied.omega:+.3f}",
                    "",
                    f"Odometry: x={odometry.x:+.3f}  y={odometry.y:+.3f}  heading={odometry.rotation:+.3f}",
                    f"Measured: vx={odometry.vx:+.3f}  vy={odometry.vy:+.3f}  omega={odometry.omega:+.3f}",
                    "",
                    f"Bumper: {'ACTIVE' if bumper else 'clear'}",
                    "",
                    "Distance sensors (S0 front, then clockwise):",
                    *sensor_lines(sensors),
                    "",
                    f"Battery low: {str(battery_low).lower()}",
                ]
                sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines))
                sys.stdout.flush()
                time.sleep(0.10)
    finally:
        stop(client)


if __name__ == "__main__":
    main()
