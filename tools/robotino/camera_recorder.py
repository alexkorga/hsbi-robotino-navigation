"""Record a Robotino /cam0 stream as a video controlled by Enter."""

import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np


# Change this to the IP address of the Robotino whose camera should be recorded.
ROBOTINO_IP = "172.21.22.90"

TARGET_FPS = 30.0
REQUEST_TIMEOUT_S = 1.0
CAMERA_PATH = "/cam0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "recordings" / "camera"


def fetch_frame() -> np.ndarray:
    """Return a fresh decoded JPEG frame from the configured Robotino.

    Returns:
        np.ndarray: A fresh decoded JPEG frame from the configured Robotino.

    Raises:
        RuntimeError: If the operation cannot complete in the current runtime state.
    """

    # The changing query parameter prevents browsers, proxies, or the Robotino
    # API from returning a cached camera frame.
    url = f"http://{ROBOTINO_IP}{CAMERA_PATH}?stamp={time.time_ns()}"
    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        jpeg = response.read()

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("The camera response was not a valid JPEG image")
    return frame


def wait_for_stop(stop_event: threading.Event) -> None:
    """Set stop_event when the operator presses Enter or stdin closes.

    Args:
        stop_event: Event set when the operator presses Enter or stdin closes.
    """

    try:
        input()
    except EOFError:
        pass
    stop_event.set()


def output_path() -> Path:
    """Return a timestamped AVI path under the ignored recordings directory.

    Returns:
        Path: A timestamped AVI path under the ignored recordings directory.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_ip = ROBOTINO_IP.replace(".", "-")
    return OUTPUT_DIRECTORY / f"robotino_{safe_ip}_{timestamp}.avi"


def main() -> None:
    """Capture frames until Enter and write an MJPEG video at target FPS.

    Raises:
        RuntimeError: If the operation cannot complete in the current runtime state.
    """

    print("Robotino camera recorder")
    print(f"Camera: http://{ROBOTINO_IP}{CAMERA_PATH}")
    print(f"Target frame rate: {TARGET_FPS:.0f} FPS")
    input("Press Enter to start recording...")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    video_path = output_path()
    stop_event = threading.Event()
    stop_thread = threading.Thread(target=wait_for_stop, args=(stop_event,), daemon=True)

    writer: cv2.VideoWriter | None = None
    frame_size: tuple[int, int] | None = None
    frame_count = 0
    failed_requests = 0
    started_s = time.monotonic()
    next_frame_s = started_s

    print("Recording. Press Enter to stop.")
    stop_thread.start()

    try:
        while not stop_event.is_set():
            now_s = time.monotonic()
            if now_s < next_frame_s:
                stop_event.wait(next_frame_s - now_s)
                continue

            # Schedule requests from the current time if the camera/network is
            # slower than the target FPS instead of trying to catch up in a burst.
            next_frame_s = max(next_frame_s + 1.0 / TARGET_FPS, now_s)

            try:
                frame = fetch_frame()
            except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
                failed_requests += 1
                print(f"\rWaiting for camera: {error}                         ", end="", flush=True)
                continue

            if writer is None:
                height, width = frame.shape[:2]
                frame_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    TARGET_FPS,
                    frame_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video file: {video_path}")
            elif frame_size is not None and (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)

            writer.write(frame)
            frame_count += 1
            elapsed_s = max(time.monotonic() - started_s, 0.001)
            measured_fps = frame_count / elapsed_s
            print(
                f"\rFrames: {frame_count} | capture: {measured_fps:4.1f} FPS"
                f" | failed requests: {failed_requests}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()

    elapsed_s = time.monotonic() - started_s
    print()
    if frame_count == 0:
        print("No video was written because no camera frame was received.")
        return

    print(f"Saved {frame_count} frames ({elapsed_s:.1f} s) to:")
    print(video_path)


if __name__ == "__main__":
    main()
