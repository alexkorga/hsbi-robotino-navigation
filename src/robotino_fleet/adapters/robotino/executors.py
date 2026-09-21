"""Command sinks. Commands arrive here already safety-checked and limited."""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Condition, Thread

from robotino_fleet.adapters.robotino.http_api import RobotinoHttpClient, Velocity
from robotino_fleet.domain.models import VelocityCommand


@dataclass(frozen=True)
class CommandRecord:
    """One queued or attempted command and its eventual worker outcome."""

    robot_id: str
    command: VelocityCommand
    timestamp_s: float
    transmitted: bool
    error: str | None = None


class ShadowCommandSink:
    """Record proposals without opening a physical transmit path."""

    physical_output_enabled = False
    destination_label = "RECORDING ONLY"

    def __init__(self) -> None:
        """Create an in-memory log with physical transmission disabled."""

        self.records: list[CommandRecord] = []

    def apply(self, robot_id: str, command: VelocityCommand) -> bool:
        """Record command for robot_id and return false: no HTTP send.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            command: Requested Robotino velocity command.

        Returns:
            bool: Whether the command was accepted for physical output.
        """

        self.records.append(CommandRecord(robot_id, command, time.monotonic(), False))
        del self.records[:-1000]
        return False

    def stop_all(self, robot_ids: Iterable[str]) -> None:
        """Record zero commands for every ID in robot_ids; return nothing.

        Args:
            robot_ids: Robotino IPs included in the operation.
        """

        for robot_id in robot_ids:
            self.apply(robot_id, VelocityCommand())


class RobotinoCommandSink:
    """Queue the newest command per Robotino for independent HTTP workers."""

    physical_output_enabled = True
    destination_label = "ROBOTINO OMNIDRIVE"

    def __init__(self, clients: dict[str, RobotinoHttpClient]) -> None:
        """Start one independent command worker for each clients entry.

        The latest queued command replaces older pending commands per robot,
        keeping a slow HTTP endpoint from blocking the fleet control loop.

        Args:
            clients: HTTP clients keyed by Robotino IP.
        """

        self.clients = clients
        self.records: list[CommandRecord] = []
        self._condition = Condition()
        self._pending: dict[str, VelocityCommand] = {}
        self._closed = False
        self._workers = {
            robot_id: Thread(
                target=self._run_worker,
                args=(robot_id, client),
                name=f"command-{robot_id}",
                daemon=True,
            )
            for robot_id, client in clients.items()
        }
        for worker in self._workers.values():
            worker.start()

    def _record(self, record: CommandRecord) -> None:
        """Append record to the bounded command-outcome history.

        Args:
            record: Append record to the bounded command-outcome history.
        """

        with self._condition:
            self.records.append(record)
            del self.records[:-1000]

    def _run_worker(
        self,
        robot_id: str,
        client: RobotinoHttpClient,
    ) -> None:
        """Transmit the newest queued command for robot_id via client.

        Runs until sink closure. Every attempt records either successful HTTP
        completion or an error; it does not return a command result directly.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            client: HTTP client for the selected physical Robotino.
        """

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or robot_id in self._pending
                )
                if robot_id not in self._pending:
                    return
                command = self._pending.pop(robot_id)
            try:
                client.set_velocity(Velocity(command.vx, command.vy, command.omega))
            except Exception as error:
                self._record(
                    CommandRecord(
                        robot_id,
                        VelocityCommand(),
                        time.monotonic(),
                        False,
                        str(error),
                    )
                )
            else:
                self._record(
                    CommandRecord(robot_id, command, time.monotonic(), True)
                )

    def apply(self, robot_id: str, command: VelocityCommand) -> bool:
        """Queue command for robot_id and return true once accepted.

        This does not confirm HTTP delivery; records contains the later
        worker outcome. Unknown IDs and closed sinks raise immediately.

        Args:
            robot_id: Robotino IP identifying the affected fleet member.
            command: Requested Robotino velocity command.

        Returns:
            bool: Whether the command was accepted for physical output.

        Raises:
            KeyError: If a requested Robotino, goal, or key does not exist.
            RuntimeError: If the operation cannot complete in the current runtime state.
        """

        if robot_id not in self.clients:
            raise KeyError(robot_id)
        with self._condition:
            if self._closed:
                raise RuntimeError("Robotino command sink is closed")
            self._pending[robot_id] = command
            self._condition.notify_all()
        return True

    def stop_all(self, robot_ids: Iterable[str]) -> None:
        """Queue zero commands for robot_ids through their workers.

        Args:
            robot_ids: Robotino IPs included in the operation.
        """

        for robot_id in robot_ids:
            self.apply(robot_id, VelocityCommand())

    def close(self) -> None:
        """Tell command workers to exit and wait briefly for each one."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for worker in self._workers.values():
            worker.join(timeout=2.0)
