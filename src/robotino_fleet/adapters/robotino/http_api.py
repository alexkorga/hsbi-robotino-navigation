"""Direct client for the Robotino HTTP endpoints used by this project."""

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


class RobotinoProtocolError(RuntimeError):
    pass


ADVANCED_FIRST_ENDPOINTS = frozenset({"/data/pose"})


def _finite(value: object, field: str) -> float:
    """Return finite numeric value or raise for invalid JSON field.

    Args:
        value: Candidate JSON number to check for finiteness.
        field: Configuration field name used in validation errors.

    Returns:
        float: Finite numeric value or raise for invalid JSON field.

    Raises:
        RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
            data.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobotinoProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RobotinoProtocolError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class Velocity:
    vx: float
    vy: float
    omega: float

    def as_payload(self) -> list[float]:
        """Return [vx, vy, omega] for the omnidrive HTTP request body.

        Returns:
            list[float]: [vx, vy, omega] for the omnidrive HTTP request body.
        """

        return [self.vx, self.vy, self.omega]


@dataclass(frozen=True)
class Pose:
    """Untransformed coordinates returned by the advanced /data/pose API."""

    x: float
    y: float
    heading: float
    sequence: int | None

    @classmethod
    def from_payload(cls, payload: object) -> "Pose":
        """Return a raw pose parsed from payload x, y, phi, and seq.

        Args:
            payload: Decoded response or API payload to parse.

        Returns:
            'Pose': A raw pose parsed from payload x, y, phi, and seq.

        Raises:
            RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                data.
        """

        if not isinstance(payload, dict):
            raise RobotinoProtocolError("pose must be an object")
        sequence = payload.get("seq")
        if sequence is not None:
            sequence_value = _finite(sequence, "pose seq")
            if int(sequence_value) != sequence_value:
                raise RobotinoProtocolError("pose seq must be an integer")
            sequence = int(sequence_value)
        return cls(
            _finite(payload.get("x"), "pose x"),
            _finite(payload.get("y"), "pose y"),
            _finite(payload.get("phi"), "pose phi"),
            sequence,
        )


@dataclass(frozen=True)
class Odometry:
    """Locally referenced /data/odometry values and body-frame velocity."""

    x: float
    y: float
    rotation: float
    vx: float
    vy: float
    omega: float
    sequence: int

    @classmethod
    def from_payload(cls, payload: object) -> "Odometry":
        """Return an odometry object from seven-value payload.

        Args:
            payload: Decoded response or API payload to parse.

        Returns:
            'Odometry': An odometry object from seven-value payload.

        Raises:
            RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                data.
        """

        if not isinstance(payload, list) or len(payload) != 7:
            raise RobotinoProtocolError(
                "odometry must be [x, y, rot, vx, vy, omega, seq]"
            )
        values = [_finite(value, f"odometry[{index}]") for index, value in enumerate(payload)]
        if int(values[6]) != values[6]:
            raise RobotinoProtocolError("odometry sequence must be an integer")
        return cls(*values[:6], int(values[6]))


@dataclass(frozen=True)
class LaserScan:
    sequence: int
    stamp: float
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: tuple[float, ...]
    intensities: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "LaserScan":
        """Return a scan from payload after validating arrays and metadata.

        Args:
            payload: Decoded response or API payload to parse.

        Returns:
            'LaserScan': A scan from payload after validating arrays and metadata.

        Raises:
            RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                data.
        """

        if not isinstance(payload, dict):
            raise RobotinoProtocolError("scan0 must be an object")
        ranges = payload.get("ranges")
        intensities = payload.get("intensities", [])
        if not isinstance(ranges, list) or not ranges:
            raise RobotinoProtocolError("scan0 ranges must be a non-empty array")
        if not isinstance(intensities, list) or (intensities and len(intensities) != len(ranges)):
            raise RobotinoProtocolError("scan0 intensities must match ranges")
        sequence = _finite(payload.get("seq"), "scan0 seq")
        if int(sequence) != sequence:
            raise RobotinoProtocolError("scan0 seq must be an integer")
        return cls(
            int(sequence),
            _finite(payload.get("stamp", 0), "scan0 stamp"),
            _finite(payload.get("angle_min"), "scan0 angle_min"),
            _finite(payload.get("angle_max"), "scan0 angle_max"),
            _finite(payload.get("angle_increment"), "scan0 angle_increment"),
            _finite(payload.get("range_min"), "scan0 range_min"),
            _finite(payload.get("range_max"), "scan0 range_max"),
            tuple(_finite(value, f"scan0 range {index}") for index, value in enumerate(ranges)),
            tuple(_finite(value, f"scan0 intensity {index}") for index, value in enumerate(intensities)),
        )


class RobotinoHttpClient:
    """Port-80 core API with advanced-first routing for named endpoints."""

    def __init__(
        self,
        ip: str,
        *,
        preferred_port: int = 8154,
        fallback_port: int = 80,
        timeout_s: float = 0.25,
    ) -> None:
        """Configure base/advanced URLs and a per-request timeout in seconds.

        ip identifies a Robotino or full URL; timeout_s bounds each
        attempt. Core calls try fallback_port first; pose tries
        preferred_port. A full http:// URL is accepted for tools/tests.

        Args:
            ip: Robotino IP address.
            preferred_port: Core calls try fallback_port first; pose tries preferred_port.
            fallback_port: Core calls try fallback_port first; pose tries preferred_port.
            timeout_s: Maximum wait before treating the request as failed, in seconds.
        """

        if ip.startswith("http://"):
            # Convenient for the standalone manual tool and local protocol tests.
            self.urls = (ip.rstrip("/"),)
        else:
            advanced = (
                f"http://{ip}"
                if preferred_port == 80
                else f"http://{ip}:{preferred_port}"
            )
            base = (
                f"http://{ip}"
                if fallback_port == 80
                else f"http://{ip}:{fallback_port}"
            )
            self.urls = tuple(dict.fromkeys((base, advanced)))
        self.timeout_s = timeout_s

    def get_pose(self) -> Pose:
        """Return raw indoor-tracking pose, preferring the advanced API.

        Returns:
            Pose: Raw indoor-tracking pose, preferring the advanced API.
        """

        return self._get("/data/pose", Pose.from_payload, advanced_first=True)

    def get_odometry(self) -> Odometry:
        """Return Robotino-local position and measured body velocity.

        Returns:
            Odometry: Robotino-local position and measured body velocity.
        """

        return self._get("/data/odometry", Odometry.from_payload)

    def get_laser_scan(self) -> LaserScan:
        """Return the unaugmented physical scan0 LiDAR measurement.

        Returns:
            LaserScan: The unaugmented physical scan0 LiDAR measurement.
        """

        return self._get("/data/scan0", LaserScan.from_payload)

    def get_distance_sensors(self) -> tuple[float, ...]:
        """Return all nine proximity distances in hardware sensor order.

        Returns:
            tuple[float, ...]: All nine proximity distances in hardware sensor order.
        """

        def parse(payload: object) -> tuple[float, ...]:
            """Validate payload and return nine ordered finite readings.

            Args:
                payload: Decoded response or API payload to parse.

            Returns:
                tuple[float, ...]: Validated payload.

            Raises:
                RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                    data.
            """

            if not isinstance(payload, list) or len(payload) != 9:
                raise RobotinoProtocolError(
                    "distance sensor array must contain nine values"
                )
            return tuple(
                _finite(value, f"distance sensor {index}")
                for index, value in enumerate(payload)
            )

        return self._get("/data/distancesensorarray", parse)

    def get_bumper(self) -> bool:
        """Return whether the physical bumper reports contact.

        Returns:
            bool: Whether the physical bumper reports contact.
        """

        def parse(payload: object) -> bool:
            """Return the Boolean bumper value from payload.

            Args:
                payload: Decoded response or API payload to parse.

            Returns:
                bool: The Boolean bumper value from payload.

            Raises:
                RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                    data.
            """

            if not isinstance(payload, dict) or not isinstance(
                payload.get("value"), bool
            ):
                raise RobotinoProtocolError("bumper must contain Boolean value")
            return payload["value"]

        return self._get("/data/bumper", parse)

    def get_battery_low(self) -> bool:
        """Return only the power-management battery-low flag.

        Returns:
            bool: Only the power-management battery-low flag.
        """

        def parse(payload: object) -> bool:
            """Return the Boolean battery-low field from payload.

            Args:
                payload: Decoded response or API payload to parse.

            Returns:
                bool: The Boolean battery-low field from payload.

            Raises:
                RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                    data.
            """

            if not isinstance(payload, dict) or not isinstance(
                payload.get("batteryLow"), bool
            ):
                raise RobotinoProtocolError(
                    "power management must contain Boolean batteryLow"
                )
            return payload["batteryLow"]

        return self._get("/data/powermanagement", parse)

    def set_velocity(self, velocity: Velocity) -> None:
        """Send body-frame velocity to omnidrive; raise if both ports fail.

        Args:
            velocity: Measured or requested Robotino body-frame velocity.
        """

        self._request("/data/omnidrive", method="PUT", body=velocity.as_payload())

    def _ordered_urls(self, path: str, advanced_first: bool | None) -> tuple[str, ...]:
        """Return fallback URLs for path in the requested port order.

        advanced_first overrides the endpoint default when provided.
        Core data normally tries port 80 first; pose tries the advanced port.

        Args:
            path: Path to the input file or model artifact.
            advanced_first: Whether to try the advanced API port before the base API.

        Returns:
            tuple[str, ...]: Fallback URLs for path in the requested port order.
        """

        use_advanced_first = (
            path in ADVANCED_FIRST_ENDPOINTS
            if advanced_first is None
            else advanced_first
        )
        return tuple(reversed(self.urls)) if use_advanced_first else self.urls

    @staticmethod
    def _request_url(
        url: str,
        path: str,
        *,
        method: str,
        encoded: bytes | None,
        timeout_s: float,
    ) -> bytes:
        """Send one HTTP request and return its raw response body.

        url and path form the target, method selects the verb,
        encoded is an optional JSON body, and timeout_s bounds the
        call. Network/HTTP errors propagate to the port-fallback caller.

        Args:
            url: Send one HTTP request and return its raw response body. url and path form the
                target, method selects the verb, encoded is an optional JSON body, and timeout_s
                bounds the call.
            path: Path to the input file or model artifact.
            method: HTTP method or callable to execute.
            encoded: Send one HTTP request and return its raw response body. url and path form
                the target, method selects the verb, encoded is an optional JSON body, and
                timeout_s bounds the call.
            timeout_s: Maximum wait before treating the request as failed, in seconds.

        Returns:
            bytes: Raw HTTP response body from the requested URL.
        """

        request = urllib.request.Request(
            f"{url}{path}",
            data=encoded,
            headers={
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if encoded else {}),
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()

    def _get(
        self,
        path: str,
        parser: Callable[[object], Any],
        *,
        advanced_first: bool | None = None,
    ) -> Any:
        """Read path and return the first JSON value accepted by parser.

        advanced_first may override endpoint port order. Raises a
        protocol error with both failures when no port returns valid data.

        Args:
            path: Path to the input file or model artifact.
            parser: Callable validating and decoding the JSON response.
            advanced_first: Whether to try the advanced API port before the base API.

        Returns:
            Any: Parsed path and return the first JSON value accepted by parser.

        Raises:
            RobotinoProtocolError: If the Robotino endpoint returns malformed or unsupported
                data.
        """

        errors: list[str] = []
        for url in self._ordered_urls(path, advanced_first):
            try:
                body = self._request_url(
                    url,
                    path,
                    method="GET",
                    encoded=None,
                    timeout_s=self.timeout_s,
                )
                payload = json.loads(body.decode("utf-8"))
                return parser(payload)
            except (
                OSError,
                TimeoutError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RobotinoProtocolError,
            ) as error:
                errors.append(f"{url}: {error}")
        raise RobotinoProtocolError(
            f"{path} returned no valid response on base or advanced API: "
            + "; ".join(errors)
        )

    def _request(
        self,
        path: str,
        *,
        method: str,
        body: object | None = None,
        advanced_first: bool | None = None,
    ) -> bytes:
        """Send method to path with optional JSON body.

        advanced_first may override port order. Return response bytes or
        raise ConnectionError after both ports fail.

        Args:
            path: Path to the input file or model artifact.
            method: HTTP method or callable to execute.
            body: Request body sent to the Robotino API.
            advanced_first: Whether to try the advanced API port before the base API.

        Returns:
            bytes: Raw response body from the first working API origin.

        Raises:
            ConnectionError: If neither configured Robotino API origin can be reached.
        """

        encoded = None if body is None else json.dumps(body).encode("utf-8")
        errors: list[str] = []
        for url in self._ordered_urls(path, advanced_first):
            try:
                return self._request_url(
                    url,
                    path,
                    method=method,
                    encoded=encoded,
                    timeout_s=self.timeout_s,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                errors.append(f"{url}: {error}")
        raise ConnectionError(f"{path} failed on base and advanced API: {'; '.join(errors)}")
