"""Check Robotino HTTP endpoint parsing and fallback behavior."""

import json
import unittest
from unittest.mock import patch
import urllib.error

from robotino_fleet.adapters.robotino.http_api import (
    LaserScan, Odometry, Pose, RobotinoHttpClient, RobotinoProtocolError, Velocity,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        """Store raw response body for a mocked HTTP request.

        Args:
            body: Request body sent to the Robotino API.
        """

        self.body = body
    def __enter__(self):
        """Return this fake response for with-statement compatibility.

        Returns:
            object: This fake response for with-statement compatibility.
        """
        return self

    def __exit__(self, *_):
        """Complete the fake response context without suppressing errors.

        Args:
            *_: Unused framework-provided argument.

        Returns:
            object: False so the fake response context propagates exceptions.
        """
        return None

    def read(self) -> bytes:
        """Return the raw mocked HTTP response body.

        Returns:
            bytes: The raw mocked HTTP response body.
        """
        return self.body


class ProtocolTests(unittest.TestCase):
    def test_observed_pose_and_odometry_schemas(self) -> None:
        """Parse the known advanced pose and seven-value odometry schemas."""

        self.assertEqual(Pose.from_payload({"x": 1, "y": 2, "phi": 0.3, "seq": 7}).sequence, 7)
        self.assertEqual(Odometry.from_payload([0, 0, 0, 0, 0, 0, 8]).sequence, 8)

    def test_scan_arrays_must_match(self) -> None:
        """Reject scans whose intensity count differs from range count."""

        with self.assertRaises(RobotinoProtocolError):
            LaserScan.from_payload({
                "seq": 1, "angle_min": 0, "angle_max": 1, "angle_increment": .1,
                "range_min": 0, "range_max": 20, "ranges": [1, 2], "intensities": [0],
            })

    @patch("robotino_fleet.adapters.robotino.http_api.urllib.request.urlopen")
    def test_core_endpoints_use_port_80_first(self, urlopen) -> None:
        """Send omnidrive commands to port 80 before the advanced API.

        Args:
            urlopen: Mock HTTP transport used to capture outgoing requests.
        """

        urlopen.return_value = FakeResponse(b"")
        client = RobotinoHttpClient("172.21.20.90", preferred_port=8154)
        client.set_velocity(Velocity(0.1, -0.2, 0.3))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://172.21.20.90/data/omnidrive")
        self.assertEqual(json.loads(request.data), [0.1, -0.2, 0.3])

    @patch("robotino_fleet.adapters.robotino.http_api.urllib.request.urlopen")
    def test_pose_uses_advanced_port_first(self, urlopen) -> None:
        """Request indoor pose from the advanced API port first.

        Args:
            urlopen: Mock HTTP transport used to capture outgoing requests.
        """

        urlopen.return_value = FakeResponse(b'{"x":1,"y":2,"phi":0,"seq":3}')
        client = RobotinoHttpClient("172.21.20.90", preferred_port=8154)
        client.get_pose()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://172.21.20.90:8154/data/pose")

    @patch("robotino_fleet.adapters.robotino.http_api.urllib.request.urlopen")
    def test_core_endpoint_falls_back_to_advanced_port(self, urlopen) -> None:
        """Retry odometry on the advanced port when port 80 is unreachable.

        Args:
            urlopen: Mock HTTP transport used to capture outgoing requests.
        """

        urlopen.side_effect = [
            urllib.error.URLError("base unavailable"),
            FakeResponse(b"[0,0,0,0,0,0,9]"),
        ]
        client = RobotinoHttpClient("172.21.20.90", preferred_port=8154)
        self.assertEqual(client.get_odometry().sequence, 9)
        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://172.21.20.90/data/odometry",
                "http://172.21.20.90:8154/data/odometry",
            ],
        )

    @patch("robotino_fleet.adapters.robotino.http_api.urllib.request.urlopen")
    def test_invalid_core_response_falls_back_to_advanced_port(self, urlopen) -> None:
        """Retry sensor reads if the base endpoint returns invalid JSON.

        Args:
            urlopen: Mock HTTP transport used to capture outgoing requests.
        """

        urlopen.side_effect = [
            FakeResponse(b"not json"),
            FakeResponse(b"[0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41]"),
        ]
        client = RobotinoHttpClient("172.21.20.90", preferred_port=8154)
        self.assertEqual(len(client.get_distance_sensors()), 9)
        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(
            urls,
            [
                "http://172.21.20.90/data/distancesensorarray",
                "http://172.21.20.90:8154/data/distancesensorarray",
            ],
        )


if __name__ == "__main__":
    unittest.main()
