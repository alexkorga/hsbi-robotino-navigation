# Data

`maps/iot_factory_baseline/` is the current baseline Robotino map bundle. It
contains the occupancy map, metadata, destination labels, station definitions,
and manually validated physical geometry. The runtime
uses these inputs for navigation, simulation, and visualization.

The runtime reads the map bundle without modifying it; changes to factory
geometry should be deliberate edits to the source files. `generated/` contains
the PNG map image served by the backend. Put transient overlays, logs,
calibration samples, and simulation results in `runtime/`.
