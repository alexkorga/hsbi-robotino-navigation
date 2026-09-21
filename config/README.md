# Configuration

- `robotinos.yaml` is the shared six-Robotino fleet. Edit an IP or LiDAR
  extrinsic directly when the lab value changes.
- `motion.yaml` is written by the guided motion calibration tool and consumed by
  simulation, learning, and runtime motion limiting.
- `localization.yaml` is written by coordinate calibration and contains only
  the planar pose-to-map transform.

An intentionally empty `localization.yaml` is valid: it keeps physical poses
untrusted and prevents autonomous physical motion until calibration succeeds.

Fixed Robotino dimensions, observed sensor geometry, polling periods, and
runtime stop thresholds are named constants in code.
