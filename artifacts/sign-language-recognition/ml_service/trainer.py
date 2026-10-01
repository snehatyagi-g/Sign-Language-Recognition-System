from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

FEATURE_COUNT = 63
MINIMUM_SAMPLES_PER_CLASS = 5
logger = logging.getLogger("uvicorn.error")


def train_model(
    *,
    dataset_path: Path,
    model_dir: Path,
    configured_labels: list[str],
    epochs: int,
    batch_size: int,
    progress_callback: Callable[[int, str], None],
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if not dataset_path.is_file():
        raise ValueError("The landmark dataset file was not found.")

    frame = pd.read_csv(dataset_path)
    feature_columns = [f"f{index}" for index in range(FEATURE_COUNT)]
    missing = [column for column in ["label", *feature_columns] if column not in frame]
    if missing:
        raise ValueError("The dataset is missing required label or landmark columns.")
    frame = frame[frame["label"].isin(configured_labels)].dropna(
        subset=["label", *feature_columns]
    )
    if frame.empty:
        raise ValueError("The dataset contains no valid configured gesture samples.")

    counts = frame["label"].value_counts()
    eligible_labels = [
        label
        for label in configured_labels
        if counts.get(label, 0) >= MINIMUM_SAMPLES_PER_CLASS
    ]
    if len(eligible_labels) < 2:
        raise ValueError(
            f"At least {MINIMUM_SAMPLES_PER_CLASS} examples in each of two classes are required."
        )
    frame = frame[frame["label"].isin(eligible_labels)]

    features = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Dataset features must all be finite numbers.")
    encoder = LabelEncoder()
    encoder.fit(eligible_labels)
    targets = encoder.transform(frame["label"].to_numpy())

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )
    progress_callback(12, f"Training on {len(x_train)} samples across {len(eligible_labels)} classes.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,), name="normalized_hand_landmarks"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(len(eligible_labels), activation="softmax"),
        ],
        name="sign_gesture_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            percentage = 12 + int(((epoch + 1) / epochs) * 78)
            accuracy = (logs or {}).get("accuracy")
            detail = f"Epoch {epoch + 1} of {epochs}"
            if accuracy is not None:
                detail += f" · training accuracy {accuracy:.1%}"
            progress_callback(percentage, detail)

    callbacks = [
        TrainingProgress(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)

    model_dir.mkdir(parents=True, exist_ok=True)
    final_model = model_dir / "sign_gesture.keras"
    temporary_model = model_dir / "sign_gesture.pending.keras"
    temporary_labels = model_dir / "model_labels.pending.json"
    labels_file = model_dir / "model_labels.json"
    model.save(temporary_model)
    temporary_labels.write_text(
        json.dumps({"labels": encoder.classes_.tolist()}, indent=2),
        encoding="utf-8",
    )
    temporary_labels.replace(labels_file)
    temporary_model.replace(final_model)

    progress_callback(96, "Evaluating the held-out test set and saving the model.")
    logger.info(
        "Saved Keras gesture classifier with %s labels and %.3f held-out accuracy",
        len(eligible_labels),
        accuracy,
    )
    return {
        "model": model,
        "labels": encoder.classes_.tolist(),
        "accuracy": float(accuracy),
        "testSamples": int(len(y_test)),
    }