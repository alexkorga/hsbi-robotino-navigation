# Navigation API

[Overview](README.md) · [Setup](SETUP.md) · [Run](RUN.md) · API · [Training](TRAINING.md) · [Architecture](ARCHITECTURE.md) · [Future work](FUTURE_WORK.md)

The Python backend exposes the same HTTP and WebSocket interface in
simulation, shadow, and live modes. The default launchers bind it to
`http://127.0.0.1:8000`. Any local client can use this interface; the web
dashboard is one such client.

FastAPI generates an interactive [Swagger UI](http://127.0.0.1:8000/docs),
[ReDoc](http://127.0.0.1:8000/redoc), and the
[OpenAPI schema](http://127.0.0.1:8000/openapi.json) while the backend is
running. The WebSocket stream is described below because it is separate from
the HTTP operations in OpenAPI.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Backend status, mode, sequence, and physical-command state |
| `GET` | `/api/map` | Factory map metadata, destination labels, and geometry |
| `GET` | `/assets/iot_factory_map.png` | Factory map image used by the dashboard |
| `GET` | `/api/world` | Current robot and navigation-state snapshot |
| `POST` | `/api/robotinos/{robotino_ip}/goal` | Assign a destination label to a Robotino |
| `POST` | `/api/robotinos/{robotino_ip}/stop` | Cancel that Robotino's active goal and request a stop |
| `POST` | `/api/simulation/reset` | Reset the simulated world |
| `POST` | `/api/simulation/pause` | Pause control ticks for simulation use |
| `POST` | `/api/simulation/resume` | Resume control ticks for simulation use |

Use the simulation controls with `run_simulation.py`.

Use a Robotino IP listed in `/api/world` or `config/robotinos.yaml` as
`{robotino_ip}`. For a goal request, send JSON with the destination's map
label ID:

```http
POST /api/robotinos/{robotino_ip}/goal
Content-Type: application/json

{"locationId": "13"}
```

Goal and stop requests return the updated world snapshot. Assigning a goal
to an active Robotino replaces its current goal. Destination labels and their
coordinates are available from `GET /api/map`.

The world snapshot includes `schemaVersion`, `sequence`, `clockS`, `mode`,
`robots`, `goals`, `temporaryObstacles`, `errors`, and `events`. Robot entries
include position, orientation, navigation status, sensor data, commands, and
navigation metrics. Clients can use `GET /api/world` for initial state and
the stream below for ongoing updates.

Unknown Robotino IDs or destination labels return HTTP `404`; invalid goal
request bodies return `422`. Resetting a physical runtime returns `409`.

## World-state stream

Connect to `ws://127.0.0.1:8000/api/world/stream` to receive JSON world
snapshots approximately every 100 ms. Each message has the same structure as
`GET /api/world`. The stream sends state from the backend to the client;
goal and stop actions use the HTTP endpoints above.

## Physical command behavior

In shadow mode, navigation commands are recorded for inspection. In live
mode, commands pass through the safety supervisor and motion limiter before
entering a per-Robotino HTTP worker queue. An API response confirms that the
backend accepted the request; it is not a physical-motion acknowledgement.
When using Swagger UI with a live backend, its **Try it out** actions can
submit real navigation goals.

This page describes the project's API. The separate
[Robotino HTTP API inventory](docs/reference/robotino_http_api_inventory.md)
documents the device endpoints consumed by the physical adapters.
