# Run Robotino Navigation

[Overview](README.md) · [Setup](SETUP.md) · Run · [API](API.md) · [Training](TRAINING.md) · [Architecture](ARCHITECTURE.md) · [Future work](FUTURE_WORK.md)

Complete [Setup](SETUP.md), then start one backend mode from the repository
root:

```text
uv run python run_simulation.py
uv run python run_shadow.py
uv run python run_live.py
```

Run one of these commands at a time. Each launcher starts the Python
navigation runtime and its HTTP/WebSocket API at `http://127.0.0.1:8000`.
Open `http://127.0.0.1:8000/docs` for the interactive API documentation or
`http://127.0.0.1:8000/api/world` for the current world snapshot. The
[API guide](API.md) describes goal assignment, robot state, and streaming.

| Mode | Robotino data | Navigation output |
| --- | --- | --- |
| Simulation | Virtual Robotinos and sensors | Applied in the simulator |
| Shadow | Physical Robotino HTTP services | Recorded for observation |
| Live | Physical Robotino HTTP services | Queued for transmission |

The launchers select `models/deliverable/shared_multi_robot_gru_corners.zip`
and its matching profile. To run another compatible model, change `MODEL_PATH`
in the selected launcher; see [Training](TRAINING.md#use-a-trained-model).

## Dashboard (optional)

With a backend running, start the dashboard in another terminal:

```text
cd frontend
pnpm dev
```

Open `http://127.0.0.1:5173`. Select a Robotino and a destination label, then
choose **Drive to label**. **Stop** cancels that Robotino's goal. The map shows
robot positions, goals, trajectories, sensors, and factory geometry. The
dashboard's Pause and Reset controls are available in simulation mode.

## Physical operation

Shadow and live modes use the IoT Factory network. Set the Robotino IPs and
ports in `config/robotinos.yaml`, and calibrate the indoor-tracking-to-map
transform in `config/localization.yaml`. Use shadow mode to inspect physical
poses, LiDAR alignment, and policy output before starting live mode.

Operate live mode in the laboratory under direct supervision, with the
Robotino's physical emergency stop available. The Swagger UI at `/docs` can
submit real goal and stop requests when a live backend is running. A goal
reaches its destination after position alignment and, where specified by the
label, heading alignment. Station approach goals enter `READY_TO_DOCK` for the
factory's separate docking workflow.

Core telemetry and motion requests try port `80` before `8154`; `/data/pose`
tries `8154` first. A queued command reported by the backend means its
per-Robotino worker accepted the request, not that the Robotino completed it.
Confirm physical behavior at the robot and through the event/error state.

## Configuration and tools

The shared configuration lives in `config/robotinos.yaml` (IPs, ports,
simulation starts, LiDAR offsets), `config/motion.yaml` (calibrated motion),
and `config/localization.yaml` (indoor tracking to map). See
[Configuration](config/README.md) for details.

Run laboratory tools from the repository root:

```text
uv run python tools/robotino/manual_control.py
uv run python tools/calibration/coordinate_calibration.py
uv run python tools/calibration/robotino_motion_calibration.py
uv run python tools/robotino/camera_recorder.py
```

Coordinate calibration writes `config/localization.yaml`. Motion calibration
moves a physical Robotino through guided maneuvers and writes
`config/motion.yaml`. Read the [calibration guide](tools/calibration/README.md)
before using those tools.
