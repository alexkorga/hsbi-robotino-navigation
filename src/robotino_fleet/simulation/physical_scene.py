"""Fast 2-D collision geometry used by the dry-run sensor simulator.

``ray_cast`` contains only physical obstacles. ``navigation_ray_cast`` also
contains the configured movement boundary and enlarged upper-level keep-outs,
which lets training reproduce the virtual scan used by the live controller.
"""

import math
from dataclasses import dataclass
from typing import Iterable

from robotino_fleet.domain.models import Point, RobotState
from robotino_fleet.maps.models import EnvironmentGeometry


_EPSILON = 1e-9


@dataclass(frozen=True)
class RayHit:
    distance_m: float
    object_id: str
    category: str


@dataclass(frozen=True)
class CollisionHit:
    object_id: str
    category: str


@dataclass(frozen=True)
class _PolygonFixture:
    object_id: str
    category: str
    points: tuple[Point, ...]
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    axis_aligned_rectangle: bool


def _is_axis_aligned_rectangle(points: tuple[Point, ...]) -> bool:
    """Return whether four points form an axis-aligned rectangle.

    Args:
        points: Positions in the coordinate frame described above.

    Returns:
        bool: Whether four points form an axis-aligned rectangle.
    """

    if len(points) != 4:
        return False
    xs = {round(point.x, 10) for point in points}
    ys = {round(point.y, 10) for point in points}
    if len(xs) != 2 or len(ys) != 2:
        return False
    return {(round(point.x, 10), round(point.y, 10)) for point in points} == {
        (x, y) for x in xs for y in ys
    }


def _ray_aabb_distance(
    origin: Point,
    direction_x: float,
    direction_y: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    maximum: float,
) -> float | None:
    """Return the first ray hit against an axis-aligned bounding box.

    origin and unit components direction_x/direction_y define the
    ray; min_x, min_y, max_x, and max_y bound the box in meters;
    maximum caps the ray. Return the entry
    distance, the exit distance if already inside, or None on a miss.

    Args:
        origin: Origin of the map-frame or body-frame geometry.
        direction_x: X component of the unit ray direction.
        direction_y: Y component of the unit ray direction.
        min_x: Left bound of the axis-aligned rectangle, in meters.
        min_y: Lower bound of the axis-aligned rectangle, in meters.
        max_x: Right bound of the axis-aligned rectangle, in meters.
        max_y: Upper bound of the axis-aligned rectangle, in meters.
        maximum: Maximum ray distance to consider, in meters.

    Returns:
        float | None: The first ray hit against an axis-aligned bounding box.
    """

    entry = 0.0
    exit_ = maximum
    for coordinate, direction, lower, upper in (
        (origin.x, direction_x, min_x, max_x),
        (origin.y, direction_y, min_y, max_y),
    ):
        if abs(direction) <= _EPSILON:
            if coordinate < lower or coordinate > upper:
                return None
            continue
        first = (lower - coordinate) / direction
        second = (upper - coordinate) / direction
        if first > second:
            first, second = second, first
        entry = max(entry, first)
        exit_ = min(exit_, second)
        if entry > exit_ + _EPSILON:
            return None
    if exit_ < -_EPSILON:
        return None
    if entry > _EPSILON:
        return entry
    # When a ray starts inside a fixture, the first physical surface ahead is
    # the exit face.  This also makes malformed overlapping fixtures safe.
    return exit_ if exit_ > _EPSILON else 0.0


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """Return the scalar cross product of vectors (ax, ay) and (bx, by).

    Args:
        ax: X component of the first vector.
        ay: Y component of the first vector.
        bx: X component of the second vector.
        by: Y component of the second vector.

    Returns:
        float: The scalar cross product of vectors (ax, ay) and (bx, by).
    """

    return ax * by - ay * bx


def _ray_segment_distance(
    origin: Point,
    direction_x: float,
    direction_y: float,
    start: Point,
    end: Point,
    maximum: float,
) -> float | None:
    """Return a ray/segment intersection distance within maximum meters.

    origin, direction_x, and direction_y describe the ray; start and
    end delimit the segment. Parallel, behind-ray, or out-of-range
    intersections return None.

    Args:
        origin: Origin of the map-frame or body-frame geometry.
        direction_x: X component of the unit ray direction.
        direction_y: Y component of the unit ray direction.
        start: Starting position, state, or timestamp.
        end: Ending position or timestamp.
        maximum: Maximum ray distance to consider, in meters.

    Returns:
        float | None: A ray/segment intersection distance within maximum meters.
    """

    segment_x = end.x - start.x
    segment_y = end.y - start.y
    denominator = _cross(direction_x, direction_y, segment_x, segment_y)
    if abs(denominator) <= _EPSILON:
        return None
    offset_x = start.x - origin.x
    offset_y = start.y - origin.y
    distance = _cross(offset_x, offset_y, segment_x, segment_y) / denominator
    fraction = _cross(offset_x, offset_y, direction_x, direction_y) / denominator
    if distance < -_EPSILON or distance > maximum + _EPSILON:
        return None
    if fraction < -_EPSILON or fraction > 1.0 + _EPSILON:
        return None
    return max(0.0, distance)


def _ray_circle_distance(
    origin: Point,
    direction_x: float,
    direction_y: float,
    center: Point,
    radius: float,
    maximum: float,
) -> float | None:
    """Return the first ray hit on a circular Robotino footprint.

    origin, direction_x, and direction_y describe the ray; center and
    radius describe the circle. Return a distance no greater than
    maximum, using the exit hit when inside, or None on a miss.

    Args:
        origin: Origin of the map-frame or body-frame geometry.
        direction_x: X component of the unit ray direction.
        direction_y: Y component of the unit ray direction.
        center: Center of the tested footprint or circular obstacle.
        radius: Radius of the circular footprint, in meters.
        maximum: Maximum ray distance to consider, in meters.

    Returns:
        float | None: The first ray hit on a circular Robotino footprint.
    """

    offset_x = origin.x - center.x
    offset_y = origin.y - center.y
    projection = offset_x * direction_x + offset_y * direction_y
    constant = offset_x * offset_x + offset_y * offset_y - radius * radius
    discriminant = projection * projection - constant
    if discriminant < 0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    near = -projection - root
    far = -projection + root
    distance = near if near >= -_EPSILON else far
    if distance < -_EPSILON or distance > maximum + _EPSILON:
        return None
    return max(0.0, distance)


def _offset_convex_polygon(
    points: tuple[Point, ...], margin_m: float
) -> tuple[Point, ...]:
    """Return points expanded outward by margin_m meters.

    Each convex edge is offset and neighboring lines are intersected to form
    a conservative navigation keep-out. A nonpositive margin leaves the
    original points unchanged; zero-length edges raise ValueError.

    Args:
        points: Positions in the coordinate frame described above.
        margin_m: Outward polygon offset in meters.

    Returns:
        tuple[Point, ...]: Points expanded outward by margin_m meters.
    """

    if margin_m <= 0:
        return points
    signed_area = sum(
        point.x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * point.y
        for index, point in enumerate(points)
    )
    orientation = 1.0 if signed_area > 0 else -1.0

    def offset_line(start: Point, end: Point) -> tuple[Point, float, float]:
        """Return offset origin/direction for the edge from start to end.

        Args:
            start: Starting position, state, or timestamp.
            end: Ending position or timestamp.

        Returns:
            tuple[Point, float, float]: Offset origin/direction for the edge from start to end.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= _EPSILON:
            raise ValueError("Obstacle polygon contains a zero-length edge")
        normal_x = orientation * dy / length
        normal_y = orientation * -dx / length
        return (
            Point(start.x + normal_x * margin_m, start.y + normal_y * margin_m),
            dx,
            dy,
        )

    result: list[Point] = []
    for index, point in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        previous_origin, previous_dx, previous_dy = offset_line(previous, point)
        current_origin, current_dx, current_dy = offset_line(point, following)
        denominator = _cross(previous_dx, previous_dy, current_dx, current_dy)
        if abs(denominator) <= _EPSILON:
            result.append(current_origin)
            continue
        between_x = current_origin.x - previous_origin.x
        between_y = current_origin.y - previous_origin.y
        ratio = _cross(between_x, between_y, current_dx, current_dy) / denominator
        result.append(
            Point(
                previous_origin.x + ratio * previous_dx,
                previous_origin.y + ratio * previous_dy,
            )
        )
    return tuple(result)


def _fixture(
    object_id: str, category: str, points: tuple[Point, ...]
) -> _PolygonFixture:
    """Precompute bounds and rectangle status for one polygon fixture.

    object_id and category label future hits; points are its map
    vertices. Return the compact fixture used by collision and ray queries.

    Args:
        object_id: Precompute bounds and rectangle status for one polygon fixture. object_id
            and category label future hits; points are its map vertices.
        category: Precompute bounds and rectangle status for one polygon fixture. object_id
            and category label future hits; points are its map vertices.
        points: Positions in the coordinate frame described above.

    Returns:
        _PolygonFixture: Cached bounds and shape information for the polygon.
    """

    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return _PolygonFixture(
        object_id=object_id,
        category=category,
        points=points,
        min_x=min(xs),
        min_y=min(ys),
        max_x=max(xs),
        max_y=max(ys),
        axis_aligned_rectangle=_is_axis_aligned_rectangle(points),
    )


class PhysicalScene:
    """Precompiled static polygons plus dynamic circular Robotino footprints."""

    def __init__(
        self,
        geometry: EnvironmentGeometry | None,
        *,
        robot_radius_m: float,
    ) -> None:
        """Compile physical and virtual fixtures from geometry.

        robot_radius_m is the circular footprint used for peer collision
        and ray tests. With no geometry, the scene contains no fixed fixtures
        or movement boundary. A nonpositive radius raises ValueError.

        Args:
            geometry: With no geometry, the scene contains no fixed fixtures or movement
                boundary.
            robot_radius_m: Robotino collision radius in meters.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if robot_radius_m <= 0:
            raise ValueError("robot_radius_m must be positive")
        self.robot_radius_m = robot_radius_m
        self._movement_area = geometry.movement_area if geometry is not None else ()
        self._grid_cell_m = 1.0
        fixtures: list[_PolygonFixture] = []
        navigation_fixtures: list[_PolygonFixture] = []
        if geometry is not None:
            for obstacle in geometry.obstacles:
                if len(obstacle.points) < 3:
                    continue
                if not obstacle.navigation_only:
                    fixtures.append(
                        _fixture(obstacle.id, obstacle.category, obstacle.points)
                    )
                if obstacle.navigation_only or obstacle.navigation_margin_m > 0:
                    navigation_points = (
                        _offset_convex_polygon(
                            obstacle.points, obstacle.navigation_margin_m
                        )
                        if obstacle.navigation_margin_m > 0
                        else obstacle.points
                    )
                    navigation_fixtures.append(
                        _fixture(
                            (
                                obstacle.id
                                if obstacle.navigation_only
                                else f"{obstacle.id}:navigation-margin"
                            ),
                            "VIRTUAL_KEEP_OUT",
                            navigation_points,
                        )
                    )
        self._fixtures = tuple(fixtures)
        self._navigation_fixtures = tuple(navigation_fixtures)
        grid: dict[tuple[int, int], list[int]] = {}
        for index, fixture in enumerate(self._fixtures):
            min_cell_x = math.floor(fixture.min_x / self._grid_cell_m)
            max_cell_x = math.floor(fixture.max_x / self._grid_cell_m)
            min_cell_y = math.floor(fixture.min_y / self._grid_cell_m)
            max_cell_y = math.floor(fixture.max_y / self._grid_cell_m)
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    grid.setdefault((cell_x, cell_y), []).append(index)
        self._fixture_grid = {
            cell: tuple(indices) for cell, indices in grid.items()
        }

    @property
    def movement_bounds(self) -> tuple[float, float, float, float]:
        """Return movement polygon bounds as (min_x, min_y, max_x, max_y).

        Raises ValueError when no movement area was configured.

        Returns:
            tuple[float, float, float, float]: Movement polygon bounds as (min_x, min_y, max_x,
                max_y).

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if not self._movement_area:
            raise ValueError("Physical scene has no configured movement area")
        return (
            min(point.x for point in self._movement_area),
            min(point.y for point in self._movement_area),
            max(point.x for point in self._movement_area),
            max(point.y for point in self._movement_area),
        )

    @property
    def static_fixture_count(self) -> int:
        """Return the count of physical fixed-obstacle polygons.

        Returns:
            int: The count of physical fixed-obstacle polygons.
        """

        return len(self._fixtures)

    @staticmethod
    def _point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
        """Return whether point is inside polygon by ray crossing.

        Args:
            point: Position in the coordinate frame described above.
            polygon: Obstacle or movement-area polygon in map coordinates.

        Returns:
            bool: Whether point is inside polygon by ray crossing.
        """

        inside = False
        previous = polygon[-1]
        for current in polygon:
            if (current.y > point.y) != (previous.y > point.y):
                crossing_x = (
                    (previous.x - current.x)
                    * (point.y - current.y)
                    / (previous.y - current.y)
                    + current.x
                )
                if point.x < crossing_x:
                    inside = not inside
            previous = current
        return inside

    @staticmethod
    def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
        """Return meters from point to the finite start–end edge.

        Args:
            point: Position in the coordinate frame described above.
            start: Starting position, state, or timestamp.
            end: Ending position or timestamp.

        Returns:
            float: Meters from point to the finite start–end edge.
        """

        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq <= _EPSILON:
            return point.distance_to(start)
        ratio = max(
            0.0,
            min(
                1.0,
                ((point.x - start.x) * dx + (point.y - start.y) * dy)
                / length_sq,
            ),
        )
        projected = Point(start.x + ratio * dx, start.y + ratio * dy)
        return point.distance_to(projected)

    @classmethod
    def _distance_to_boundary(cls, point: Point, polygon: tuple[Point, ...]) -> float:
        """Return the shortest distance from point to a polygon edge.

        Args:
            point: Position in the coordinate frame described above.
            polygon: Obstacle or movement-area polygon in map coordinates.

        Returns:
            float: The shortest distance from point to a polygon edge.
        """

        return min(
            cls._distance_to_segment(point, start, polygon[(index + 1) % len(polygon)])
            for index, start in enumerate(polygon)
        )

    def collision_at(
        self,
        center: Point,
        *,
        radius_m: float | None = None,
        robots: Iterable[RobotState] = (),
        exclude_robot_id: str | None = None,
    ) -> CollisionHit | None:
        """Find the first collision at the proposed center in map meters.

        radius_m overrides the default footprint, robots adds dynamic
        peers, and exclude_robot_id omits the current Robotino. Return an
        identified boundary, fixture, or peer hit, otherwise None. Both
        physical and navigation-only keep-outs constrain simulated movement.

        Args:
            center: Center of the tested footprint or circular obstacle.
            radius_m: Collision radius in meters.
            robots: Robotino states in the shared world.
            exclude_robot_id: Robotino IP to omit from peer or obstacle calculations.

        Returns:
            CollisionHit | None: First colliding fixture, or None when the center is clear.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        radius = self.robot_radius_m if radius_m is None else radius_m
        if radius <= 0:
            raise ValueError("radius_m must be positive")
        if self._movement_area:
            if not self._point_in_polygon(center, self._movement_area):
                return CollisionHit("movement-area", "boundary")
            if self._distance_to_boundary(center, self._movement_area) < radius - _EPSILON:
                return CollisionHit("movement-area", "boundary")
        for fixture in self._fixtures:
            if (
                center.x + radius < fixture.min_x
                or center.x - radius > fixture.max_x
                or center.y + radius < fixture.min_y
                or center.y - radius > fixture.max_y
            ):
                continue
            if self._point_in_polygon(center, fixture.points):
                return CollisionHit(fixture.object_id, fixture.category)
            if self._distance_to_boundary(center, fixture.points) < radius - _EPSILON:
                return CollisionHit(fixture.object_id, fixture.category)
        for fixture in self._navigation_fixtures:
            if (
                center.x + radius < fixture.min_x
                or center.x - radius > fixture.max_x
                or center.y + radius < fixture.min_y
                or center.y - radius > fixture.max_y
            ):
                continue
            if self._point_in_polygon(center, fixture.points):
                return CollisionHit(fixture.object_id, fixture.category)
            if self._distance_to_boundary(center, fixture.points) < radius - _EPSILON:
                return CollisionHit(fixture.object_id, fixture.category)
        for robot in robots:
            if robot.id == exclude_robot_id or not robot.pose_valid:
                continue
            if center.distance_to(robot.point) < radius + self.robot_radius_m - _EPSILON:
                return CollisionHit(robot.id, "robotino")
        return None

    def is_valid_center(
        self,
        center: Point,
        *,
        radius_m: float | None = None,
        robots: Iterable[RobotState] = (),
        exclude_robot_id: str | None = None,
    ) -> bool:
        """Return whether the footprint at center is collision-free.

        radius_m, robots, and exclude_robot_id are passed to
        collision_at; its absence of a hit becomes True.

        Args:
            center: Center of the tested footprint or circular obstacle.
            radius_m: Collision radius in meters.
            robots: Robotino states in the shared world.
            exclude_robot_id: Robotino IP to omit from peer or obstacle calculations.

        Returns:
            bool: Whether the footprint at center is collision-free.
        """

        return self.collision_at(
            center,
            radius_m=radius_m,
            robots=robots,
            exclude_robot_id=exclude_robot_id,
        ) is None

    def ray_cast(
        self,
        origin: Point,
        direction_rad: float,
        maximum_m: float,
        *,
        robots: Iterable[RobotState] = (),
        exclude_robot_id: str | None = None,
    ) -> RayHit | None:
        """Return the nearest physical obstacle hit along a map-frame ray.

        origin and direction_rad define the ray; maximum_m caps
        travel and must be positive. robots adds peer footprints and
        exclude_robot_id avoids self-hits. Return None for no hit.
        Virtual boundaries are excluded so simulated physical scans match
        what actual hardware can see.

        Args:
            origin: Origin of the map-frame or body-frame geometry.
            direction_rad: Direction angle in radians.
            maximum_m: Maximum distance in meters.
            robots: Robotino states in the shared world.
            exclude_robot_id: Robotino IP to omit from peer or obstacle calculations.

        Returns:
            RayHit | None: The nearest physical obstacle hit along a map-frame ray.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if maximum_m <= 0:
            raise ValueError("maximum_m must be positive")
        direction_x = math.cos(direction_rad)
        direction_y = math.sin(direction_rad)
        result = self.robot_ray_cast(
            origin,
            direction_rad,
            maximum_m,
            robots=robots,
            exclude_robot_id=exclude_robot_id,
        )
        closest = result.distance_m if result is not None else maximum_m + _EPSILON

        cell_size = self._grid_cell_m
        cell_x = math.floor(origin.x / cell_size)
        cell_y = math.floor(origin.y / cell_size)
        if direction_x > _EPSILON:
            step_x = 1
            next_x = (cell_x + 1) * cell_size
            next_crossing_x = (next_x - origin.x) / direction_x
            crossing_step_x = cell_size / direction_x
        elif direction_x < -_EPSILON:
            step_x = -1
            next_x = cell_x * cell_size
            next_crossing_x = (next_x - origin.x) / direction_x
            crossing_step_x = -cell_size / direction_x
        else:
            step_x = 0
            next_crossing_x = math.inf
            crossing_step_x = math.inf
        if direction_y > _EPSILON:
            step_y = 1
            next_y = (cell_y + 1) * cell_size
            next_crossing_y = (next_y - origin.y) / direction_y
            crossing_step_y = cell_size / direction_y
        elif direction_y < -_EPSILON:
            step_y = -1
            next_y = cell_y * cell_size
            next_crossing_y = (next_y - origin.y) / direction_y
            crossing_step_y = -cell_size / direction_y
        else:
            step_y = 0
            next_crossing_y = math.inf
            crossing_step_y = math.inf

        checked: set[int] = set()
        travelled = 0.0
        while travelled <= min(maximum_m, closest) + _EPSILON:
            for fixture_index in self._fixture_grid.get((cell_x, cell_y), ()):
                if fixture_index in checked:
                    continue
                checked.add(fixture_index)
                fixture = self._fixtures[fixture_index]
                broad_hit = _ray_aabb_distance(
                    origin,
                    direction_x,
                    direction_y,
                    fixture.min_x,
                    fixture.min_y,
                    fixture.max_x,
                    fixture.max_y,
                    min(maximum_m, closest),
                )
                if broad_hit is None:
                    continue
                if fixture.axis_aligned_rectangle:
                    distance = broad_hit
                else:
                    distance = None
                    for index, start in enumerate(fixture.points):
                        end = fixture.points[(index + 1) % len(fixture.points)]
                        candidate = _ray_segment_distance(
                            origin,
                            direction_x,
                            direction_y,
                            start,
                            end,
                            min(maximum_m, closest),
                        )
                        if candidate is not None and (
                            distance is None or candidate < distance
                        ):
                            distance = candidate
                if distance is not None and distance < closest:
                    closest = distance
                    result = RayHit(distance, fixture.object_id, fixture.category)

            next_crossing = min(next_crossing_x, next_crossing_y)
            if next_crossing > min(maximum_m, closest) + _EPSILON:
                break
            advance_x = next_crossing_x <= next_crossing_y + _EPSILON
            advance_y = next_crossing_y <= next_crossing_x + _EPSILON
            if advance_x:
                cell_x += step_x
                next_crossing_x += crossing_step_x
            if advance_y:
                cell_y += step_y
                next_crossing_y += crossing_step_y
            travelled = next_crossing
        return result

    def robot_ray_cast(
        self,
        origin: Point,
        direction_rad: float,
        maximum_m: float,
        *,
        robots: Iterable[RobotState] = (),
        exclude_robot_id: str | None = None,
    ) -> RayHit | None:
        """Ray-cast only localized peers in robots.

        origin and direction_rad define the ray, maximum_m limits
        its length, and exclude_robot_id suppresses self-hits. Return the
        nearest RayHit or None.

        Args:
            origin: Origin of the map-frame or body-frame geometry.
            direction_rad: Direction angle in radians.
            maximum_m: Maximum distance in meters.
            robots: Robotino states in the shared world.
            exclude_robot_id: Robotino IP to omit from peer or obstacle calculations.

        Returns:
            RayHit | None: Nearest ray hit against localized peers, if any.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if maximum_m <= 0:
            raise ValueError("maximum_m must be positive")
        direction_x = math.cos(direction_rad)
        direction_y = math.sin(direction_rad)
        closest = maximum_m + _EPSILON
        result: RayHit | None = None
        for robot in robots:
            if robot.id == exclude_robot_id or not robot.pose_valid:
                continue
            distance = _ray_circle_distance(
                origin,
                direction_x,
                direction_y,
                robot.point,
                self.robot_radius_m,
                min(maximum_m, closest),
            )
            if distance is not None and distance < closest:
                closest = distance
                result = RayHit(distance, robot.id, "robotino")
        return result

    def virtual_ray_cast(
        self,
        origin: Point,
        direction_rad: float,
        maximum_m: float,
    ) -> RayHit | None:
        """Find the nearest configured virtual boundary/keep-out hit.

        origin and direction_rad define a map-frame ray and
        maximum_m limits its reach. Return a RayHit or None. These
        boundaries are overlaid for navigation, not emitted by physical scans.

        Args:
            origin: Origin of the map-frame or body-frame geometry.
            direction_rad: Direction angle in radians.
            maximum_m: Maximum distance in meters.

        Returns:
            RayHit | None: Nearest configured virtual ray hit, if any.

        Raises:
            ValueError: If configuration, shapes, or supplied values violate this operation's
                contract.
        """

        if maximum_m <= 0:
            raise ValueError("maximum_m must be positive")
        direction_x = math.cos(direction_rad)
        direction_y = math.sin(direction_rad)
        closest = maximum_m + _EPSILON
        result: RayHit | None = None

        if self._movement_area and self._point_in_polygon(origin, self._movement_area):
            for index, start in enumerate(self._movement_area):
                end = self._movement_area[(index + 1) % len(self._movement_area)]
                distance = _ray_segment_distance(
                    origin,
                    direction_x,
                    direction_y,
                    start,
                    end,
                    min(maximum_m, closest),
                )
                if distance is not None and distance < closest:
                    closest = distance
                    result = RayHit(distance, "movement-area", "VIRTUAL_BOUNDARY")

        for fixture in self._navigation_fixtures:
            for index, start in enumerate(fixture.points):
                end = fixture.points[(index + 1) % len(fixture.points)]
                distance = _ray_segment_distance(
                    origin,
                    direction_x,
                    direction_y,
                    start,
                    end,
                    min(maximum_m, closest),
                )
                if distance is not None and distance < closest:
                    closest = distance
                    result = RayHit(distance, fixture.object_id, fixture.category)
        return result

    def navigation_ray_cast(
        self,
        origin: Point,
        direction_rad: float,
        maximum_m: float,
        *,
        robots: Iterable[RobotState] = (),
        exclude_robot_id: str | None = None,
    ) -> RayHit | None:
        """Return the nearer physical or virtual obstacle along a ray.

        origin, direction_rad, and maximum_m describe the ray;
        robots and exclude_robot_id control peer hits. The result is
        used by navigation-facing simulated scans and is None if clear.

        Args:
            origin: Origin of the map-frame or body-frame geometry.
            direction_rad: Direction angle in radians.
            maximum_m: Maximum distance in meters.
            robots: Robotino states in the shared world.
            exclude_robot_id: Robotino IP to omit from peer or obstacle calculations.

        Returns:
            RayHit | None: The nearer physical or virtual obstacle along a ray.
        """

        physical = self.ray_cast(
            origin,
            direction_rad,
            maximum_m,
            robots=robots,
            exclude_robot_id=exclude_robot_id,
        )
        virtual = self.virtual_ray_cast(origin, direction_rad, maximum_m)
        if physical is None:
            return virtual
        if virtual is None or physical.distance_m <= virtual.distance_m:
            return physical
        return virtual
