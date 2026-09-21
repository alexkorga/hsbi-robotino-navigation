"""Machine-label goals, including optional docking approach headings."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from robotino_fleet.domain.models import Point
from robotino_fleet.maps.models import MapBundle


class NavigationLifecycle(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    APPROACHING = "APPROACHING"
    ALIGNING = "ALIGNING"
    ARRIVED = "ARRIVED"
    READY_TO_DOCK = "READY_TO_DOCK"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class NavigationGoal:
    """A map label, arrival tolerances, and optional docking handover marker."""

    location_id: str
    point: Point
    heading_rad: float | None
    position_tolerance_m: float
    heading_tolerance_rad: float
    docking_handoff: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return target position, optional heading, tolerances, and dock marker.

        Returns:
            dict[str, Any]: Target position, optional heading, tolerances, and dock marker.
        """

        return {
            "locationId": self.location_id,
            "x": self.point.x,
            "y": self.point.y,
            "headingRad": self.heading_rad,
            "positionToleranceM": self.position_tolerance_m,
            "headingToleranceRad": self.heading_tolerance_rad,
            "dockingHandoff": self.docking_handoff,
        }


class DestinationCatalog:
    """Resolve selectable map labels to precise navigation goals."""

    def __init__(self, goals: dict[str, NavigationGoal]) -> None:
        """Index nonempty goals by label ID; reject an empty factory catalog.

        Args:
            goals: Navigation targets for the active agents.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if not goals:
            raise ValueError("Factory has no destination labels")
        self._goals = goals

    @classmethod
    def from_map(
        cls,
        map_bundle: MapBundle,
        *,
        position_tolerance_m: float,
        heading_tolerance_rad: float,
    ) -> "DestinationCatalog":
        """Build goals from map_bundle labels and station approaches.

        position_tolerance_m and heading_tolerance_rad become the
        handover bounds for every goal. Station approach labels receive a dock
        marker; the returned catalog does not execute docking itself.

        Args:
            map_bundle: Factory map metadata, goal labels, and obstacle geometry.
            position_tolerance_m: Maximum accepted position error in meters.
            heading_tolerance_rad: Maximum permitted goal-heading error, in radians.

        Returns:
            'DestinationCatalog': Constructed goals from map_bundle labels and station
                approaches.
        """

        docking = {
            station.approach_location: f"dock:{station.id}"
            for station in map_bundle.stations.values()
            if station.approach_location
        }
        return cls(
            {
                location.id: NavigationGoal(
                    location.id,
                    location.point,
                    location.heading_rad,
                    position_tolerance_m,
                    heading_tolerance_rad,
                    docking.get(location.id),
                )
                for location in map_bundle.locations.values()
            }
        )

    def get(self, location_id: str) -> NavigationGoal:
        """Return the goal named by location_id or raise KeyError.

        Args:
            location_id: Factory-map label ID to use as the navigation goal.

        Returns:
            NavigationGoal: The goal named by location_id or raise KeyError.

        Raises:
            KeyError: If a requested Robotino, goal, or key does not exist.
        """

        try:
            return self._goals[location_id]
        except KeyError as error:
            raise KeyError(f"Unknown destination {location_id}") from error

    def values(self) -> tuple[NavigationGoal, ...]:
        """Return every selectable goal in catalog insertion order.

        Returns:
            tuple[NavigationGoal, ...]: Every selectable goal in catalog insertion order.
        """

        return tuple(self._goals.values())
