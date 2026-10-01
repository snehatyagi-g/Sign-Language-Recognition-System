from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp

FEATURE_COUNT = 63
BASE_DIR = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FEATURE_COLUMNS = [f"f{index}" for index in range(FEATURE_COUNT)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize(points: list[object]) -> list[float] | None:
    if len(points) != 21:
        return None
    wrist = points[0]
    translated = [
        (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z)
        for point in points
    ]
    scale = max((x * x + y * y + z * z) ** 0.5 for x, y, z in translated)
    if not scale or not scale < float("inf"):
        return None
    features = [coordinate / scale for point in translated for coordinate in point]
    return features if len(features) == FEATURE_COUNT else None


def get_model_path() -> Path:
    path = Path.home() / ".cache" / "sign-language-recognition" / "hand_landmarker.task"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import urllib.request

        print("Downloading the MediaPipe hand-landmarker task model…", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect real hand-landmark samples from a local webcam with OpenCV."
    )
    parser.add_argument("--label", required=True, help="Configured label for this recording session.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "landmarks.csv",
        help="Dataset CSV path (default: ml_service/data/landmarks.csv).",
    )
    args = parser.parse_args()

    labels = json.loads(
        (BASE_DIR / "config" / "gesture_labels.json").read_text(encoding="utf-8")
    )["labels"]
    if args.label not in labels:
        parser.error(f"'{args.label}' is not in config/gesture_labels.json")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(get_model_path())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open webcam index {args.camera}.", flush=True)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", *FEATURE_COLUMNS]
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    print(
        f"Collecting {args.label}. Press C to capture the visible hand, Q to quit.",
        flush=True,
    )
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker, args.output.open(
            "a", newline="", encoding="utf-8"
        ) as dataset:
            writer = csv.DictWriter(dataset, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from the webcam.", flush=True)
                    return 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(image, timestamp_ms)
                points = result.hand_landmarks[0] if result.hand_landmarks else None
                features = normalize(points) if points else None

                if points:
                    height, width = frame.shape[:2]
                    for first, second in HAND_CONNECTIONS:
                        start = points[first]
                        end = points[second]
                        cv2.line(
                            frame,
                            (int(start.x * width), int(start.y * height)),
                            (int(end.x * width), int(end.y * height)),
                            (74, 165, 135),
                            2,
                        )
                    for point in points:
                        cv2.circle(
                            frame,
                            (int(point.x * width), int(point.y * height)),
                            4,
                            (90, 199, 167),
                            -1,
                        )
                instruction = f"{args.label} · hand tracked" if features else f"{args.label} · show one hand"
                cv2.putText(
                    frame,
                    instruction,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (245, 245, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Sign gesture sample collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and features:
                    writer.writerow(
                        {
                            "label": args.label,
                            **{
                                f"f{index}": f"{value:.8f}"
                                for index, value in enumerate(features)
                            },
                        }
                    )
                    dataset.flush()
                    print(f"Saved one {args.label} sample to {args.output}", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())