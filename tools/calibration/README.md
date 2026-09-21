# Calibration

`coordinate_calibration.py` asks for three precise placements at known labels,
retries `/data/pose` indefinitely when the advanced API is unavailable, fits
the rigid planar transform, rejects an inaccurate fit, and writes
`config/localization.yaml`:

    uv run python tools/calibration/coordinate_calibration.py

`robotino_motion_calibration.py` measures stationary noise, forward, sideways,
and rotation deadbands, response delay, acceleration, and braking. It
states the required directional clearance before every run and then opens a
positioning phase. Hold W/S/A/D to translate and Q/E to rotate. On Windows the
physical key state is read continuously, so releasing every movement key stops
immediately. Other terminals use short, automatically expiring keypresses.
Space stops immediately. Enter stops and starts the announced measurement, and
Esc aborts the calibration.
Forward measurements return along the same cleared path, and no experiment
requires three metres of open space in every direction. The tool stores the raw
CSV in runtime/calibration/ and writes every consumed value to
`config/motion.yaml`:

- Translation is measured from 0.10 through 0.50 m/s for both the forward and
  sideways body axes. Their maximum speed, acceleration, braking, and deadband
  are stored separately. The longest run requests 2.50 m in its measurement
  direction and 0.30 m behind it. Rotate the Robotino during positioning to use
  the same clear corridor for the sideways run.
- Rotation is measured from 0.25 through 1.00 rad/s and only needs a 0.35 m
  clear radius.

    uv run python tools/calibration/robotino_motion_calibration.py

Both tools have their editable values at the top. Keep the emergency stop
available and do not run the motion tool near people, walls, machines, or
another Robotino.
