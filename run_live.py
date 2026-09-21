"""Run the trained policy on physical Robotinos."""

from pathlib import Path

from robotino_fleet.main import serve


MODEL_PATH = Path(
    "models/deliverable/shared_multi_robot_gru_corners.zip"
)


if __name__ == "__main__":
    serve("live", model_path=MODEL_PATH)
