# Navigation Dashboard

The optional React/Three.js dashboard is a map-based client of the Python
[navigation API](../API.md). It visualizes the factory image, valid movement
area, machines, walls, destination labels, Robotino footprints, goals,
trajectories, LiDAR, distance sensors, and temporary obstacles.

Use **Drive to label** to assign a navigation goal and **Stop** to cancel one.
Navigation, safety, motion limiting, simulation, and physical Robotino access
remain in Python.

Follow [Run](../RUN.md) to start one backend. Then, from the repository root,
run:

    cd frontend
    pnpm dev

Open `http://127.0.0.1:5173`. By default, the dashboard connects to
`http://127.0.0.1:8000`. To use another backend address, copy `.env.example`
to `.env.local` and set `VITE_FLEET_API_URL` there. It uses the same API in
simulation, shadow, and live modes.
Pause and Reset remain visible but are disabled in physical modes.

Verify frontend changes with:

    pnpm test
    pnpm run build

First-time dependency installation is covered in [Setup](../SETUP.md).
