# Shared simulation

`SimulationWorld` is the canonical deterministic 2-D world. The backend
simulation and the Gymnasium/PettingZoo learning adapters both use it for
calibrated motion, command delay, footprint collision, physical geometry,
Robotino collision, LiDAR, and nine distance sensors.

Start the simulation backend from the repository root with:

    uv run python run_simulation.py

The launcher uses the retained GRU policy by default. Set `MODEL_PATH` in
`run_simulation.py` to inspect another compatible trained model through the
[API](../../../API.md) or optional dashboard. See [Run](../../../RUN.md) and
[Training](../../../TRAINING.md).
