# Manual Robotino control

Edit the IP and conservative speeds at the top of `manual_control.py`, then run:

    uv run python tools/robotino/manual_control.py

The tool works on Windows, macOS, and Linux. W/S move forward/back, A/D move
sideways, Q/E rotate, Space stops, and Escape exits. Bumper state always
overrides movement, and shutdown sends repeated zero commands. On Windows it
reads held-key state continuously, so releasing the key stops movement. Other
terminals use short commands that automatically expire after each keypress.

## Camera recording

Edit `ROBOTINO_IP` at the top of `camera_recorder.py`, then run:

    uv run python tools/robotino/camera_recorder.py

Press Enter once to start recording and again to stop. The recorder requests
the current `/cam0` JPEG at up to the editable `TARGET_FPS` rate (30 FPS by default)
and writes an MJPEG AVI file under `recordings/camera/`.
