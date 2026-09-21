"""Guided physical calibration with positioning before every measurement."""

import sys
import time
from datetime import datetime

from robotino_fleet.adapters.robotino.http_api import RobotinoHttpClient, Velocity
from robotino_fleet.adapters.robotino.keyboard import (
    Keyboard,
    VK_ESCAPE,
    VK_RETURN,
    requested_velocity,
)
from robotino_fleet.calibration.motion import (
    MotionSample,
    derive_motion_settings,
    write_motion_settings,
    write_samples,
)
from robotino_fleet.config import PROJECT_ROOT


ROBOTINO_IP = "172.21.20.90"
TRANSLATION_SPEEDS = (0.10, 0.20, 0.30, 0.40, 0.50)
ROTATION_SPEEDS = (0.25, 0.50, 0.75, 1.00)
REPETITIONS = 3


SAMPLE_PERIOD_S = 0.05
TRANSLATION_MOVE_TIME_S = 5.0
ROTATION_MOVE_TIME_S = 2.0
STOP_TIME_S = 1.5
POSITION_TRANSLATION_SPEED_MPS = 0.12
POSITION_ROTATION_SPEED_RPS = 0.35
POSITION_COMMAND_HOLD_S = 0.20
POSITION_SETTLE_S = 0.75


def positioning_command(key: str | None) -> Velocity:
    """Return slow manual velocity or stop command for positioning key.

    Args:
        key: Name of the field to read or validate.

    Returns:
        Velocity: Slow manual velocity or stop command for positioning key.
    """

    return {
        "w": Velocity(POSITION_TRANSLATION_SPEED_MPS, 0.0, 0.0),
        "s": Velocity(-POSITION_TRANSLATION_SPEED_MPS, 0.0, 0.0),
        "a": Velocity(0.0, POSITION_TRANSLATION_SPEED_MPS, 0.0),
        "d": Velocity(0.0, -POSITION_TRANSLATION_SPEED_MPS, 0.0),
        "q": Velocity(0.0, 0.0, POSITION_ROTATION_SPEED_RPS),
        "e": Velocity(0.0, 0.0, -POSITION_ROTATION_SPEED_RPS),
        " ": Velocity(0.0, 0.0, 0.0),
    }.get(key, Velocity(0.0, 0.0, 0.0))


def command_text(command: Velocity) -> str:
    """Return a compact operator-facing summary of command.

    Args:
        command: Requested Robotino velocity command.

    Returns:
        str: A compact operator-facing summary of command.
    """

    if command == Velocity(0.0, 0.0, 0.0):
        return "stopped"
    return (
        f"vx={command.vx:+.2f}  vy={command.vy:+.2f}  "
        f"omega={command.omega:+.2f}"
    )


def stop(client: RobotinoHttpClient) -> None:
    """Try three zero-velocity sends to client after a measurement.

    Args:
        client: HTTP client for the selected physical Robotino.
    """

    for _ in range(3):
        try:
            client.set_velocity(Velocity(0.0, 0.0, 0.0))
        except Exception:
            pass
        time.sleep(0.05)


def position_for_measurement(
    client: RobotinoHttpClient,
    title: str,
    required_space: str,
) -> None:
    """Let the operator position client before measurement title.

    Displays required_space, blocks until Enter, and always requests a
    stop on exit. Escape aborts the calibration session; no value is returned.

    Args:
        client: HTTP client for the selected physical Robotino.
        title: Let the operator position client before measurement title.
        required_space: Displays required_space, blocks until Enter, and always requests a
            stop on exit.

    Raises:
        KeyboardInterrupt: If the operator cancels the calibration maneuver.
    """

    stop(client)
    print("\n" + "=" * 72)
    print(title)
    print(f"Required space: {required_space}")
    print("Tap/hold W/S forward/back | A/D left/right | Q/E rotate")
    print("Movement stops automatically after the last key event | Space stop")
    print("Enter: stop and start this measurement | Esc: abort calibration")
    command = Velocity(0.0, 0.0, 0.0)
    command_expires_at = 0.0
    previous_applied = command
    try:
        with Keyboard() as keyboard:
            while True:
                now = time.monotonic()
                if keyboard.has_live_key_state:
                    if keyboard.is_down(VK_ESCAPE):
                        raise KeyboardInterrupt
                    if keyboard.is_down(VK_RETURN):
                        break
                    command = requested_velocity(
                        keyboard,
                        POSITION_TRANSLATION_SPEED_MPS,
                        POSITION_ROTATION_SPEED_RPS,
                    )
                else:
                    key = keyboard.read()
                    if key in {"\r", "\n"}:
                        break
                    if key == "\x1b":
                        raise KeyboardInterrupt
                    if key is not None:
                        command = positioning_command(key)
                        command_expires_at = (
                            now + POSITION_COMMAND_HOLD_S
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
                if applied != previous_applied or bumper:
                    status = (
                        "bumper active - stopped" if bumper else command_text(applied)
                    )
                    sys.stdout.write(f"\rPositioning: {status:<52}")
                    sys.stdout.flush()
                previous_applied = applied
                time.sleep(0.05)
    finally:
        stop(client)
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
    time.sleep(POSITION_SETTLE_S)
    print("Measurement starting.")


def record(
    client: RobotinoHttpClient,
    samples: list[MotionSample],
    phase: str,
    command: Velocity,
    duration_s: float,
    started_s: float,
) -> None:
    """Send command through client for duration_s.

    Append odometry to samples under phase with time relative to
    started_s. A bumper press or stalled odometry raises an error; nothing
    is returned.

    Args:
        client: HTTP client for the selected physical Robotino.
        samples: Recorded motion-calibration samples.
        phase: Named motion-calibration phase.
        command: Requested Robotino velocity command.
        duration_s: Send command through client for duration_s.
        started_s: Start time of the measurement or recording in seconds.

    Raises:
        RuntimeError: If the operation cannot complete in the current runtime state.
    """

    deadline = time.monotonic() + duration_s
    last_sequence: int | None = None
    unchanged_since = time.monotonic()
    while time.monotonic() < deadline:
        client.set_velocity(command)
        odometry = client.get_odometry()
        if client.get_bumper():
            raise RuntimeError("Bumper pressed")
        if odometry.sequence != last_sequence:
            last_sequence = odometry.sequence
            unchanged_since = time.monotonic()
        elif time.monotonic() - unchanged_since > 0.5:
            raise RuntimeError("Odometry stopped updating")
        samples.append(
            MotionSample(
                time.monotonic() - started_s,
                phase,
                command.vx,
                command.vy,
                command.omega,
                odometry.vx,
                odometry.vy,
                odometry.omega,
                odometry.sequence,
            )
        )
        time.sleep(SAMPLE_PERIOD_S)


def maneuver(
    client: RobotinoHttpClient,
    samples: list[MotionSample],
    name: str,
    command: Velocity,
    move_time_s: float,
    started_s: float,
) -> None:
    """Record one powered name phase and its braking phase.

    client executes command for move_time_s; both phases append to
    samples using session origin started_s. Nothing is returned.

    Args:
        client: HTTP client for the selected physical Robotino.
        samples: Recorded motion-calibration samples.
        name: Name of the motion-calibration maneuver.
        command: Requested Robotino velocity command.
        move_time_s: Duration of the powered movement phase, in seconds.
        started_s: Start time of the measurement or recording in seconds.
    """

    print(f"  {name}")
    record(client, samples, name, command, move_time_s, started_s)
    record(client, samples, f"{name}-brake", Velocity(0.0, 0.0, 0.0), STOP_TIME_S, started_s)


def main() -> None:
    """Guide physical maneuvers and write samples plus shared motion constants."""

    print("Robotino motion calibration")
    print(f"Robotino: {ROBOTINO_IP}")
    print("The tool states the required directional clearance before every run.")
    print("Use the positioning controls at each prompt, then press Enter.")

    client = RobotinoHttpClient(ROBOTINO_IP)
    # Fail before movement if the essential motion endpoint is unavailable.
    client.get_odometry()
    samples: list[MotionSample] = []
    started = time.monotonic()
    try:
        print("1/4 Measuring stationary odometry noise")
        position_for_measurement(
            client,
            "Stationary odometry noise",
            "No travel; leave about 0.35 m clear around the Robotino.",
        )
        record(
            client,
            samples,
            "stationary",
            Velocity(0.0, 0.0, 0.0),
            3.0,
            started,
        )

        print("2/4 Measuring forward, sideways, and rotation deadbands")
        position_for_measurement(
            client,
            "Forward translation deadband",
            "At least 0.25 m directly in front of the Robotino.",
        )
        for value in (0.01, 0.02, 0.03, 0.04, 0.05):
            record(
                client,
                samples,
                f"forward-deadband-{value}",
                Velocity(value, 0.0, 0.0),
                0.7,
                started,
            )
            stop(client)
        position_for_measurement(
            client,
            "Sideways translation deadband",
            "At least 0.25 m to the Robotino's left.",
        )
        for value in (0.01, 0.02, 0.03, 0.04, 0.05):
            record(
                client,
                samples,
                f"sideways-deadband-{value}",
                Velocity(0.0, value, 0.0),
                0.7,
                started,
            )
            stop(client)
        position_for_measurement(
            client,
            "Rotation deadband",
            "A 0.35 m clear radius around the Robotino; rotation is in place.",
        )
        for value in (0.02, 0.04, 0.06, 0.08, 0.10):
            record(
                client,
                samples,
                f"rotation-deadband-{value}",
                Velocity(0.0, 0.0, value),
                0.7,
                started,
            )
            stop(client)

        print("3/4 Measuring translation acceleration and braking")
        for repetition in range(REPETITIONS):
            for speed in TRANSLATION_SPEEDS:
                travel_clearance = speed * TRANSLATION_MOVE_TIME_S + 0.50
                position_for_measurement(
                    client,
                    (
                        f"Forward translation {speed:.2f} m/s - "
                        f"run {repetition + 1}/{REPETITIONS}"
                    ),
                    (
                        f"{travel_clearance:.2f} m in front and 0.30 m behind the "
                        "Robotino. It returns along the same path."
                    ),
                )
                maneuver(
                    client,
                    samples,
                    f"translation-forward-{speed}-{repetition}",
                    Velocity(speed, 0.0, 0.0),
                    TRANSLATION_MOVE_TIME_S,
                    started,
                )
                maneuver(
                    client,
                    samples,
                    f"translation-return-{speed}-{repetition}",
                    Velocity(-speed, 0.0, 0.0),
                    TRANSLATION_MOVE_TIME_S,
                    started,
                )

                position_for_measurement(
                    client,
                    (
                        f"Sideways translation {speed:.2f} m/s - "
                        f"run {repetition + 1}/{REPETITIONS}"
                    ),
                    (
                        f"{travel_clearance:.2f} m to the Robotino's left and "
                        "0.30 m to its right. Rotate the Robotino during positioning "
                        "so its left side points along the clear corridor."
                    ),
                )
                maneuver(
                    client,
                    samples,
                    f"translation-left-{speed}-{repetition}",
                    Velocity(0.0, speed, 0.0),
                    TRANSLATION_MOVE_TIME_S,
                    started,
                )
                maneuver(
                    client,
                    samples,
                    f"translation-right-return-{speed}-{repetition}",
                    Velocity(0.0, -speed, 0.0),
                    TRANSLATION_MOVE_TIME_S,
                    started,
                )

        print("4/4 Measuring rotation acceleration and braking")
        for repetition in range(REPETITIONS):
            for speed in ROTATION_SPEEDS:
                position_for_measurement(
                    client,
                    f"Rotation {speed:.2f} rad/s - run {repetition + 1}/{REPETITIONS}",
                    (
                        "A 0.35 m clear radius around the Robotino. It rotates left, "
                        "brakes, then rotates back."
                    ),
                )
                maneuver(
                    client,
                    samples,
                    f"rotation-left-{speed}-{repetition}",
                    Velocity(0.0, 0.0, speed),
                    ROTATION_MOVE_TIME_S,
                    started,
                )
                maneuver(
                    client,
                    samples,
                    f"rotation-return-{speed}-{repetition}",
                    Velocity(0.0, 0.0, -speed),
                    ROTATION_MOVE_TIME_S,
                    started,
                )
    except KeyboardInterrupt:
        print("\nCalibration aborted. No measurements or constants were written.")
        return
    finally:
        stop(client)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = PROJECT_ROOT / "runtime" / "calibration" / f"motion-{timestamp}.csv"
    motion_path = PROJECT_ROOT / "config" / "motion.yaml"
    write_samples(raw_path, samples)
    settings = derive_motion_settings(samples)
    write_motion_settings(motion_path, settings)
    print(f"Raw measurements: {raw_path}")
    print(f"Simulation constants updated: {motion_path}")


if __name__ == "__main__":
    main()
