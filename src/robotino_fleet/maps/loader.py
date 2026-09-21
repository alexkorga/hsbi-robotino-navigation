"""Load the factory image metadata, destinations, and physical geometry."""

import math
import re
from pathlib import Path
from typing import Any

import yaml

from robotino_fleet.domain.models import Point
from robotino_fleet.maps.models import (
    Bounds,
    EnvironmentGeometry,
    EnvironmentObstacle,
    MapBundle,
    MapLocation,
    MapMetadata,
    Station,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Return a mapping from YAML path, including ROS-style `%YAML:` files.

    Empty content becomes an empty mapping; non-mapping content raises because
    downstream map loaders expect named fields.

    Args:
        path: Path to the YAML configuration or map file.

    Returns:
        dict[str, Any]: A mapping from YAML path, including ROS-style `%YAML:` files.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:"):
        text = "\n".join(text.splitlines()[1:])
    value = yaml.safe_load(text) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return value


def _pgm_size(path: Path) -> tuple[int, int]:
    """Return image width and height in pixels from PGM/PPM path.

    Pixel data is not loaded; unsupported or truncated headers raise.

    Args:
        path: Path to the factory map image.

    Returns:
        tuple[int, int]: Image width and height in pixels from PGM/PPM path.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    with path.open("rb") as file:
        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = file.readline()
            if not line:
                break
            tokens.extend(line.split(b"#", 1)[0].split())
    if len(tokens) < 4 or tokens[0] not in {b"P2", b"P3", b"P5", b"P6"}:
        raise ValueError(f"Unsupported PGM/PPM header in {path}")
    return int(tokens[1]), int(tokens[2])


def _records(text: str, kind: str) -> list[str]:
    """Split factory Lisp text into records marked `(IS-A kind)`.

    Returns the matching record spans for the location or station parser; this
    deliberately preserves the source format rather than evaluating Lisp.

    Args:
        text: Split factory Lisp text into records marked `(IS-A kind)`.
        kind: Event category displayed in the dashboard.

    Returns:
        list[str]: Matching Lisp record spans from the factory data.
    """

    marker = f"((IS-A {kind})"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    return [
        text[start : starts[index + 1] if index + 1 < len(starts) else len(text)]
        for index, start in enumerate(starts)
    ]


def _locations(path: Path) -> dict[str, MapLocation]:
    """Return label goals keyed by ID from the factory positions file.

    Position fields in path are millimeters and are converted to meters;
    optional orientation degrees become radians, defaulting to zero.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        dict[str, MapLocation]: Label goals keyed by ID from the factory positions file.
    """

    result: dict[str, MapLocation] = {}
    for record in _records(path.read_text(encoding="utf-8"), "LOCATION"):
        name = re.search(r"\(NAME\s+([^\s)]+)\)", record)
        pose = re.search(r"\(APPROACH-EXACT-POSE\s+\(([-+0-9.]+)\s+([-+0-9.]+)", record)
        heading = re.search(r"\(ORIENTATION-EXACT\s+\(ANGLE-ABSOLUTE\s+([-+0-9.]+)\)\)", record)
        if name is None or pose is None:
            continue
        identifier = name.group(1)
        result[identifier] = MapLocation(
            identifier,
            Point(float(pose.group(1)) / 1000.0, float(pose.group(2)) / 1000.0),
            math.radians(float(heading.group(1))) if heading else 0.0,
        )
    return result


def _stations(path: Path) -> dict[str, Station]:
    """Return machine stations keyed by ID from station file path.

    The station approach-location ID links a machine to a navigation label;
    records without an ID or pose are skipped.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        dict[str, Station]: Machine stations keyed by ID from station file path.
    """

    result: dict[str, Station] = {}
    for record in _records(path.read_text(encoding="utf-8"), "STATION"):
        identifier = re.search(r"\(ID\s+([^\s)]+)\)", record)
        kind = re.search(r"\(TYPE\s+([^\s)]+)\)", record)
        approach = re.search(r"\(APPROACH-LOCATION\s+([^\s)]+)\)", record)
        pose = re.search(r"\(POSE\s+\(([-+0-9.]+)\s+([-+0-9.]+).*?\)\)", record, re.DOTALL)
        if identifier is None or pose is None:
            continue
        value = identifier.group(1)
        result[value] = Station(
            value,
            kind.group(1) if kind else "UNKNOWN",
            approach.group(1) if approach else None,
            Point(float(pose.group(1)), float(pose.group(2))),
            0.0,
        )
    return result


def _polygon(value: object, field: str) -> tuple[Point, ...]:
    """Validate value as at least three finite `[x, y]` map-frame points.

    field identifies malformed geometry in errors; the returned tuple is
    used for movement bounds and obstacles in simulation and visualization.

    Args:
        value: Sequence of map-frame coordinate pairs defining the polygon.
        field: Configuration field name used in validation errors.

    Returns:
        tuple[Point, ...]: Validated map-frame polygon vertices.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{field} must contain at least three points")
    points: list[Point] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{field}[{index}] must be [x, y]")
        point = Point(float(item[0]), float(item[1]))
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ValueError(f"{field}[{index}] must be finite")
        points.append(point)
    return tuple(points)


def _geometry(path: Path) -> EnvironmentGeometry | None:
    """Load manual movement and obstacle polygons from path if present.

    Returns None for an absent optional layer. Present geometry must name
    valid WALL/MACHINE obstacles; navigation-only polygons remain distinct from
    physical collision geometry.

    Args:
        path: Path to the input file or model artifact.

    Returns:
        EnvironmentGeometry | None: Loaded manual movement and obstacle polygons from path
            if present.

    Raises:
        ValueError: If configuration, shapes, or supplied values violate this operation's
            contract.
    """

    if not path.exists():
        return None
    data = _read_yaml(path)
    movement = data.get("movement_area")
    if not isinstance(movement, dict):
        raise ValueError("movement_area must be an object")
    raw_obstacles = data.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        raise ValueError("obstacles must be a list")
    obstacles: list[EnvironmentObstacle] = []
    for index, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            raise ValueError(f"obstacles[{index}] must be an object")
        identifier = str(raw.get("id", "")).strip()
        category = str(raw.get("category", "")).strip().upper()
        if not identifier or category not in {"MACHINE", "WALL"}:
            raise ValueError(f"Invalid obstacle at index {index}")
        navigation_margin_m = float(raw.get("navigation_margin_m", 0.0))
        if not math.isfinite(navigation_margin_m) or navigation_margin_m < 0:
            raise ValueError(
                f"{identifier}.navigation_margin_m must be finite and non-negative"
            )
        navigation_only = raw.get("navigation_only", False)
        if not isinstance(navigation_only, bool):
            raise ValueError(f"{identifier}.navigation_only must be a boolean")
        obstacles.append(
            EnvironmentObstacle(
                identifier,
                str(raw.get("name", identifier)),
                category,
                _polygon(raw.get("points"), f"{identifier}.points"),
                navigation_margin_m,
                navigation_only,
            )
        )
    return EnvironmentGeometry(
        source=str(data.get("source", "manual")),
        movement_area=_polygon(movement.get("points"), "movement_area.points"),
        obstacles=tuple(obstacles),
    )


def load_map_bundle(directory: Path) -> MapBundle:
    """Load the map image metadata, labels, stations, and manual geometry.

    directory must contain the YAML/PGM map and Lisp label/station source
    files; manual geometry is optional. Returns one meter-based bundle shared
    by simulation, runtime destinations, and the dashboard.

    Args:
        directory: Directory containing the factory map image and label files.

    Returns:
        MapBundle: Loaded the map image metadata, labels, stations, and manual geometry.
    """

    directory = directory.resolve()
    metadata = _read_yaml(directory / "navigation-map.yaml")
    width, height = _pgm_size(directory / "navigation-map.pgm")
    resolution = float(metadata["resolution"])
    origin = [float(value) for value in metadata["origin"]]
    return MapBundle(
        metadata=MapMetadata(
            id="iot_factory",
            name="IoT Factory",
            image_url="/assets/iot_factory_map.png",
            width_px=width,
            height_px=height,
            resolution_m_per_px=resolution,
            origin_x=origin[0],
            origin_y=origin[1],
            origin_theta=origin[2],
            bounds=Bounds(
                origin[0],
                origin[1],
                origin[0] + width * resolution,
                origin[1] + height * resolution,
            ),
        ),
        locations=_locations(directory / "positions.lisp"),
        stations=_stations(directory / "stations.lisp"),
        environment_geometry=_geometry(directory / "environment-geometry.yaml"),
    )
