"""Verify real and shadow command sinks handle Robotino commands correctly."""

import threading
import time
import unittest

from robotino_fleet.adapters.robotino.executors import RobotinoCommandSink
from robotino_fleet.domain.models import VelocityCommand


class BlockingClient:
    def __init__(self) -> None:
        """Prepare events and a list for observing delayed HTTP sends."""

        self.started = threading.Event()
        self.release = threading.Event()
        self.velocities = []

    def set_velocity(self, velocity) -> None:
        """Block until released, then retain the transmitted velocity.

        Args:
            velocity: Measured or requested Robotino body-frame velocity.
        """

        self.started.set()
        self.release.wait(timeout=1.0)
        self.velocities.append(velocity)


class RobotinoCommandSinkTests(unittest.TestCase):
    def test_apply_does_not_wait_for_http_and_keeps_latest_command(self) -> None:
        """Queue control ticks promptly while a slow client sends only the latest."""

        client = BlockingClient()
        sink = RobotinoCommandSink({"r1": client})

        started = time.perf_counter()
        sink.apply("r1", VelocityCommand(0.1, 0.0, 0.0))
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertTrue(client.started.wait(timeout=0.5))

        sink.apply("r1", VelocityCommand(0.2, 0.0, 0.0))
        sink.apply("r1", VelocityCommand(0.5, 0.0, 0.0))
        client.release.set()
        sink.close()

        self.assertEqual(len(client.velocities), 2)
        self.assertEqual(client.velocities[-1].vx, 0.5)


if __name__ == "__main__":
    unittest.main()
