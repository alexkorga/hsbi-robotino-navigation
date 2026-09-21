"""Render the canonical PGM occupancy map as a backend-served PNG asset."""

import argparse
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data" / "maps" / "iot_factory_baseline" / "navigation-map.pgm"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "iot_factory_map.png"


def generate(source: Path, output: Path) -> None:
    """Convert source PGM to backend-served PNG at output.

    Args:
        source: Convert source PGM to backend-served PNG at output.
        output: Convert source PGM to backend-served PNG at output.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.save(output, format="PNG", optimize=True)
    print(f"Generated backend map image: {output}")


def main() -> None:
    """Accept source/output paths and generate the dashboard map image."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.source.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
