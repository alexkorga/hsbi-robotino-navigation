"""Factory geometry, destinations, and coordinate transforms."""

from .loader import load_map_bundle
from .models import (
    EnvironmentGeometry,
    EnvironmentObstacle,
    MapBundle,
    MapLocation,
    MapMetadata,
    Station,
)
from .transforms import (
    CalibrationFit,
    CalibrationPair,
    PlanarTransform,
    fit_rigid_transform,
    normalize_angle,
)
