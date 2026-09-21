"""Factory geometry and destination models; no navigation graph."""

from dataclasses import dataclass
from typing import Any

from robotino_fleet.domain.models import Point


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def to_dict(self) -> dict[str, float]:
        """Return the map-frame minimum and maximum coordinates for the API.

        Returns:
            dict[str, float]: The map-frame minimum and maximum coordinates for the API.
        """

        return {
            "minX": self.min_x,
            "minY": self.min_y,
            "maxX": self.max_x,
            "maxY": self.max_y,
        }


@dataclass(frozen=True)
class MapMetadata:
    id: str
    name: str
    image_url: str
    width_px: int
    height_px: int
    resolution_m_per_px: float
    origin_x: float
    origin_y: float
    origin_theta: float
    bounds: Bounds

    def to_dict(self) -> dict[str, Any]:
        """Return image dimensions, resolution, origin, and map-frame bounds.

        Returns:
            dict[str, Any]: Image dimensions, resolution, origin, and map-frame bounds.
        """

        return {
            "id": self.id,
            "name": self.name,
            "imageUrl": self.image_url,
            "widthPx": self.width_px,
            "heightPx": self.height_px,
            "resolutionMPerPx": self.resolution_m_per_px,
            "origin": {
                "x": self.origin_x,
                "y": self.origin_y,
                "theta": self.origin_theta,
            },
            "bounds": self.bounds.to_dict(),
        }


@dataclass(frozen=True)
class MapLocation:
    id: str
    point: Point
    heading_rad: float

    def to_dict(self) -> dict[str, Any]:
        """Return this selectable label's map position and target heading.

        Returns:
            dict[str, Any]: This selectable label's map position and target heading.
        """

        return {"id": self.id, **self.point.to_dict(), "headingRad": self.heading_rad}


@dataclass(frozen=True)
class Station:
    id: str
    type: str
    approach_location: str | None
    point: Point
    heading_rad: float

    def to_dict(self) -> dict[str, Any]:
        """Return this machine's approach label, center, and orientation.

        Returns:
            dict[str, Any]: This machine's approach label, center, and orientation.
        """

        return {
            "id": self.id,
            "type": self.type,
            "approachLocation": self.approach_location,
            **self.point.to_dict(),
            "headingRad": self.heading_rad,
        }


@dataclass(frozen=True)
class EnvironmentObstacle:
    id: str
    name: str
    category: str
    points: tuple[Point, ...]
    navigation_margin_m: float = 0.0
    navigation_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return polygon vertices and collision/visualization metadata.

        Returns:
            dict[str, Any]: Polygon vertices and collision/visualization metadata.
        """

        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "points": [point.to_dict() for point in self.points],
            "navigationMarginM": self.navigation_margin_m,
            "navigationOnly": self.navigation_only,
        }


@dataclass(frozen=True)
class EnvironmentGeometry:
    source: str
    movement_area: tuple[Point, ...]
    obstacles: tuple[EnvironmentObstacle, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return movement-area and obstacle polygons for the dashboard.

        Returns:
            dict[str, Any]: Movement-area and obstacle polygons for the dashboard.
        """

        return {
            "source": self.source,
            "movementArea": [point.to_dict() for point in self.movement_area],
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }


@dataclass
class MapBundle:
    metadata: MapMetadata
    locations: dict[str, MapLocation]
    stations: dict[str, Station]
    environment_geometry: EnvironmentGeometry | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the complete map API payload with sorted labels and stations.

        A missing geometry layer is represented by empty polygons so the
        dashboard can use one shape even when a fixture omits physical bounds.

        Returns:
            dict[str, Any]: The complete map API payload with sorted labels and stations.
        """

        def sort_key(value: MapLocation | Station) -> tuple[int, str]:
            """Return a numeric-first ordering key for location/station value.

            Args:
                value: Location or station label to order.

            Returns:
                tuple[int, str]: A numeric-first ordering key for location/station value.
            """

            return (int(value.id), value.id) if value.id.isdigit() else (10**9, value.id)

        return {
            "schemaVersion": 3,
            "map": self.metadata.to_dict(),
            "locations": [
                item.to_dict() for item in sorted(self.locations.values(), key=sort_key)
            ],
            "stations": [
                item.to_dict() for item in sorted(self.stations.values(), key=sort_key)
            ],
            "environmentGeometry": (
                self.environment_geometry.to_dict()
                if self.environment_geometry
                else {"source": "none", "movementArea": [], "obstacles": []}
            ),
        }
