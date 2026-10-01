from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trainer import train_model

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the TensorFlow/Keras classifier from collected hand landmarks."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=Path, default=BASE_DIR / "data" / "landmarks.csv")
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "models")
    args = parser.parse_args()

    if not 5 <= args.epochs <= 100:
        parser.error("--epochs must be between 5 and 100")
    if not 8 <= args.batch_size <= 128:
        parser.error("--batch-size must be between 8 and 128")

    labels_file = BASE_DIR / "config" / "gesture_labels.json"
    try:
        configured_labels = json.loads(labels_file.read_text(encoding="utf-8"))["labels"]
        result = train_model(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            configured_labels=configured_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            progress_callback=lambda progress, message: print(
                f"[{progress:3d}%] {message}", flush=True
            ),
        )
        print(
            f"Held-out test accuracy: {result['accuracy']:.1%} "
            f"({result['testSamples']} test samples).",
            flush=True,
        )
        print(f"Saved Keras model to {args.model_dir / 'sign_gesture.keras'}", flush=True)
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())