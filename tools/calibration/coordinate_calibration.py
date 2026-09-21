"""Fit /data/pose to the factory map from three manually placed samples."""

import time

import yaml

from robotino_fleet.adapters.robotino.http_api import (
    Pose,
    RobotinoHttpClient,
    RobotinoProtocolError,
)
from robotino_fleet.config import PROJECT_ROOT, load_settings
from robotino_fleet.domain.models import Point
from robotino_fleet.maps import CalibrationPair, fit_rigid_transform, load_map_bundle


ROBOTINO_IP = "172.21.23.90"
REFERENCE_LABELS = ("13", "20", "1")
MAXIMUM_FIT_ERROR_M = 0.15
POSE_TIMEOUT_S = 2.0
POSE_RETRY_DELAY_S = 1.0


def get_pose_with_retry(client: RobotinoHttpClient) -> Pose:
    """Return a valid pose from client, retrying until interrupted.

    Each protocol failure is printed with its next retry delay so an operator
    can diagnose the advanced API without restarting calibration.

    Args:
        client: HTTP client for the selected physical Robotino.

    Returns:
        Pose: A valid pose from client, retrying until interrupted.
    """

    attempt = 1
    while True:
        try:
            return client.get_pose()
        except RobotinoProtocolError as error:
            print(f"  Pose request {attempt} failed: {error}")
            print(f"  Retrying in {POSE_RETRY_DELAY_S:.1f} s (Ctrl+C to abort)...")
            attempt += 1
            time.sleep(POSE_RETRY_DELAY_S)


def main() -> None:
    """Fit three label observations and save a validated map transform.

    Raises:
        RuntimeError: If the operation cannot complete in the current runtime state.
    """

    settings = load_settings()
    factory_map = load_map_bundle(settings.map_bundle)
    client = RobotinoHttpClient(ROBOTINO_IP, timeout_s=POSE_TIMEOUT_S)
    pairs: list[CalibrationPair] = []
    print("Coordinate calibration")
    print("Place the Robotino center at each requested map label.")
    for label in REFERENCE_LABELS:
        location = factory_map.locations[label]
        input(f"Place it at label {label} and press Enter...")
        pose = get_pose_with_retry(client)
        pairs.append(
            CalibrationPair(
                indoor=Point(pose.x, pose.y),
                map=location.point,
            )
        )
        print(f"  pose ({pose.x:.3f}, {pose.y:.3f}) -> map ({location.point.x:.3f}, {location.point.y:.3f})")
    fit = fit_rigid_transform(pairs)
    print(f"RMSE: {fit.residual_rmse_m:.3f} m; maximum error: {fit.maximum_error_m:.3f} m")
    if fit.maximum_error_m > MAXIMUM_FIT_ERROR_M:
        raise RuntimeError(
            "Fit is too inaccurate; repeat with more precise Robotino placement"
        )
    output = PROJECT_ROOT / "config" / "localization.yaml"
    output.write_text(
        yaml.safe_dump(
            {
                "map_from_indoor_tracking": {
                    "x_m": fit.transform.x_m,
                    "y_m": fit.transform.y_m,
                    "theta_rad": fit.transform.theta_rad,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
