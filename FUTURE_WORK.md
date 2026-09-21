# Future Work

[Overview](README.md) · [Setup](SETUP.md) · [Run](RUN.md) · [API](API.md) · [Training](TRAINING.md) · [Architecture](ARCHITECTURE.md) · Future work

The navigation runtime, shared simulation, and API provide a foundation for
further research and deployment work in the IoT Factory. Potential next steps
include:

- **Task allocation:** assign incoming transport jobs to Robotinos using
  estimated travel time, robot availability, and station demand.
- **Fleet-level coordination:** compare learned and optimization-based methods
  for scheduling concurrent journeys through shared factory space.
- **Dynamic-obstacle training:** extend simulation scenarios with moving
  people and movable objects, and evaluate response to changing traffic.
- **Localization recovery:** validate initial map poses automatically and
  handle pose reacquisition during an active journey.
- **Broader evaluation:** benchmark navigation and coordination across more
  maps, robot counts, obstacle patterns, and hardware conditions.
- **Operational observability:** retain run-level telemetry and evaluation
  metrics for longitudinal comparisons between model versions.
