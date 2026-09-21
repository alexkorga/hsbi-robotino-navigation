# Robotino Navigation

Overview · [Setup](SETUP.md) · [Run](RUN.md) · [API](API.md) · [Training](TRAINING.md) · [Architecture](ARCHITECTURE.md) · [Future work](FUTURE_WORK.md)

Robotino Navigation is a learned navigation system for Festo Robotinos in the
IoT Factory at HSBI (Hochschule Bielefeld). Given a Robotino and a destination
label on the factory map, it uses a trained policy to navigate through the
factory floor and align with the destination. A Python runtime combines
Robotino telemetry, navigation inference, motion limits, and safety checks.
It exposes robot state and goal controls through a local HTTP/WebSocket API.

The project includes a calibrated 2-D simulation of the factory, Robotino
motion, LiDAR, and distance sensors. The same navigation runtime operates with
virtual Robotinos in simulation or with physical Robotinos in the IoT Factory.
Models can be trained and evaluated entirely offline. A React/Three.js
dashboard provides an optional map-based view and control interface for the
API.

## Quick start

Install the Python dependencies and start the standalone simulation from the
repository root:

```text
uv sync --locked
uv run python run_simulation.py
```

Open [the interactive API documentation](http://127.0.0.1:8000/docs) to inspect
the running system and assign a destination label. To start the optional
dashboard, open another terminal:

```text
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

The dashboard runs at `http://127.0.0.1:5173`. See [Setup](SETUP.md) for
prerequisites and [Run](RUN.md) for all operating modes.

## Operating modes

| Mode | Data source | Navigation output |
| --- | --- | --- |
| Simulation | Virtual Robotinos and simulated sensors | Applied to the 2-D simulation |
| Shadow | Physical Robotinos in the IoT Factory | Recorded for observation |
| Live | Physical Robotinos in the IoT Factory | Sent to the Robotinos |

Simulation runs locally without factory access. Shadow and live use the
IoT Factory network and Robotino HTTP services. All three modes expose the
same project API; the dashboard can connect to any of them.

## Repository guide

- `src/robotino_fleet/` contains the runtime, navigation policies, simulation,
  Robotino adapters, learning environments, and API.
- `training/` contains the model-training and evaluation experiments.
- `models/deliverable/` contains the retained models and evaluation reports.
- `data/` contains factory map data and environment geometry.
- `config/` contains Robotino, motion, and localization settings.
- `frontend/` contains the optional dashboard.
