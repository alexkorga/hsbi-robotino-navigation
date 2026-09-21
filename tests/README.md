# Tests

Automated tests cover the physical geometry and sensors, HTTP schemas, the
shared simulation world, motion limiting, the learning adapter, and the small
dashboard API.

Run the complete active suite from the repository root:

    uv run python -m unittest discover -s tests -v

Frontend tests are run separately from `frontend/` with `pnpm test`.
