from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .webcam_collector import FEATURE_COLUMNS, get_model_path, normalize

SOURCE_TO_TARGET = {
    "Fist": "CLOSED_FIST",
    "OpenPalm": "OPEN_PALM",
    "PeaceSign": "PEACE",
    "ThumbsUp": "THUMBS_UP",
}
BASE_DIR = Path(__file__).resolve().parent


def extract(input_dir: Path, output: Path) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise ValueError(f"Input image directory does not exist: {input_dir}")
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing CSV: {output}")

    configured_labels = set(
        json.loads(
            (BASE_DIR / "config" / "gesture_labels.json").read_text(encoding="utf-8")
        )["labels"]
    )
    missing_labels = sorted(set(SOURCE_TO_TARGET.values()) - configured_labels)
    if missing_labels:
        raise ValueError(
            "Dataset mapping references labels not in gesture_labels.json: "
            + ", ".join(missing_labels)
        )

    counts = {
        label: {"sourceImages": 0, "extracted": 0, "skipped": 0}
        for label in SOURCE_TO_TARGET.values()
    }
    rows: list[tuple[str, list[float]]] = []
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(get_model_path())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for image_path in sorted(input_dir.rglob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            source_class = image_path.stem.rsplit("_", 1)[0]
            target_label = SOURCE_TO_TARGET.get(source_class)
            if target_label is None:
                continue
            counts[target_label]["sourceImages"] += 1

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                counts[target_label]["skipped"] += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            features = normalize(landmarks) if landmarks else None
            if (
                features is None
                or len(features) != len(FEATURE_COLUMNS)
                or not all(math.isfinite(value) for value in features)
            ):
                counts[target_label]["skipped"] += 1
                continue
            rows.append((target_label, features))
            counts[target_label]["extracted"] += 1

    if not rows:
        raise ValueError("MediaPipe did not detect a valid hand in any mapped image.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as dataset:
            writer = csv.writer(dataset)
            writer.writerow(["label", *FEATURE_COLUMNS])
            writer.writerows((label, *features) for label, features in rows)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    return {"output": str(output), "totalSamples": len(rows), "classes": counts}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract normalized MediaPipe landmarks from the supported image dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the source dataset images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New CSV path to create; existing files are never overwritten.",
    )
    args = parser.parse_args()
    try:
        summary = extract(args.input_dir, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())