"""Extract every Nth valid frame from a video."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - reported when a command is executed
    cv2 = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=4, help="Save one frame every N decoded frames.")
    args = parser.parse_args()
    if cv2 is None:
        raise SystemExit("OpenCV is required. Install it with: python -m pip install opencv-python")
    if args.step < 1:
        raise ValueError("--step must be at least 1.")
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    frame_index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.step == 0:
            output_path = args.output / f"{saved:06d}.png"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write frame: {output_path}")
            saved += 1
        frame_index += 1
    capture.release()
    print(f"Decoded {frame_index} frames; saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
