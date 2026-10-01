from __future__ import annotations

import csv
import json
import logging
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .trainer import FEATURE_COUNT, MINIMUM_SAMPLES_PER_CLASS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "sign_gesture.keras"
MODEL_LABELS_FILE = MODEL_DIR / "model_labels.json"
CONFIG_LABELS_FILE = BASE_DIR / "config" / "gesture_labels.json"
API_PREFIX = "/api/sign-language"

logger = logging.getLogger("uvicorn.error")
app = FastAPI(
    title="Sign Language Recognition API",
    version="1.0.0",
    description="MediaPipe landmark ingestion, Keras inference, and model training.",
)


class GesturePredictionInput(BaseModel):
    features: list[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)


class GestureSample(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    features: list[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)


class GestureDatasetInput(BaseModel):
    samples: list[GestureSample] = Field(min_length=1, max_length=10000)


class GestureTrainingInput(BaseModel):
    epochs: int = Field(ge=5, le=100)
    batchSize: int = Field(ge=8, le=128)


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "status": "idle",
            "progress": 0,
            "accuracy": None,
            "testSamples": None,
            "message": "No training run has started.",
            "startedAt": None,
            "completedAt": None,
            "jobId": None,
        }

    def update(self, **changes: Any) -> None:
        with self.lock:
            self.value.update(changes)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {key: value for key, value in self.value.items() if key != "jobId"}


training_state = TrainingState()
model_lock = threading.Lock()
cached_model: Any = None
cached_labels: list[str] = []
cached_model_mtime_ns: int | None = None


def configured_labels() -> list[str]:
    try:
        raw = json.loads(CONFIG_LABELS_FILE.read_text(encoding="utf-8"))
        labels = raw.get("labels")
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gesture labels could not be loaded. Check config/gesture_labels.json.",
        ) from exc
    if not isinstance(labels, list) or not labels or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gesture labels must be a non-empty list of strings.",
        )
    normalized = [label.strip() for label in labels]
    if len(set(normalized)) != len(normalized):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gesture labels must be unique.",
        )
    return normalized


def _validate_features(features: list[float]) -> None:
    if len(features) != FEATURE_COUNT or not all(math.isfinite(value) for value in features):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Each hand must contain exactly {FEATURE_COUNT} finite features.",
        )


def dataset_summary() -> tuple[int, dict[str, int]]:
    labels = configured_labels()
    counts = {label: 0 for label in labels}
    total = 0
    if not DATA_FILE.exists():
        return total, counts
    try:
        with DATA_FILE.open(newline="", encoding="utf-8") as dataset:
            reader = csv.DictReader(dataset)
            for row in reader:
                label = row.get("label", "")
                if label in counts:
                    counts[label] += 1
                    total += 1
    except (OSError, csv.Error) as exc:
        logger.exception("Could not read the gesture dataset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The gesture dataset could not be read.",
        ) from exc
    return total, counts


def _load_model() -> tuple[Any, list[str]]:
    global cached_model, cached_labels, cached_model_mtime_ns
    if not MODEL_FILE.is_file() or not MODEL_LABELS_FILE.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained Keras model is available. Collect labeled samples and train a model first.",
        )
    try:
        mtime_ns = MODEL_FILE.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trained model file could not be accessed.",
        ) from exc

    with model_lock:
        if cached_model is not None and cached_model_mtime_ns == mtime_ns:
            return cached_model, cached_labels
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(MODEL_FILE)
            labels_payload = json.loads(MODEL_LABELS_FILE.read_text(encoding="utf-8"))
            labels = labels_payload["labels"]
            if model.input_shape[-1] != FEATURE_COUNT:
                raise ValueError("Model input size does not match the landmark feature count.")
            if model.output_shape[-1] != len(labels):
                raise ValueError("Model output size does not match its label metadata.")
        except Exception as exc:
            logger.exception("Could not load the trained Keras model")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The saved Keras model is invalid or incompatible. Train it again.",
            ) from exc
        cached_model = model
        cached_labels = labels
        cached_model_mtime_ns = mtime_ns
        return model, labels


@app.get(f"{API_PREFIX}/labels")
def get_gesture_labels() -> dict[str, list[str]]:
    return {"labels": configured_labels()}


@app.get(f"{API_PREFIX}/model/status")
def get_model_status() -> dict[str, Any]:
    total, counts = dataset_summary()
    labels = configured_labels()
    present_classes = sum(1 for count in counts.values() if count > 0)
    model_ready = False
    if MODEL_FILE.is_file() and MODEL_LABELS_FILE.is_file():
        try:
            _load_model()
            model_ready = True
        except HTTPException:
            logger.warning("A saved model was found but could not be loaded")

    if model_ready:
        message = "A trained Keras model is ready for real-time predictions."
    elif MODEL_FILE.exists():
        message = "A saved model could not be loaded. Train a new model to replace it."
    elif total:
        message = "Samples are saved. Train a model to enable predictions."
    else:
        message = "No trained model or collected samples yet. Start by collecting examples for each gesture."
    return {
        "modelReady": model_ready,
        "modelPath": "models/sign_gesture.keras" if model_ready else None,
        "labels": labels,
        "datasetSamples": total,
        "classesWithSamples": present_classes,
        "minimumSamplesPerClass": MINIMUM_SAMPLES_PER_CLASS,
        "message": message,
    }


@app.get(f"{API_PREFIX}/training/status")
def get_training_status() -> dict[str, Any]:
    return training_state.snapshot()


@app.post(f"{API_PREFIX}/predict")
def predict_gesture(payload: GesturePredictionInput) -> dict[str, Any]:
    _validate_features(payload.features)
    model, labels = _load_model()
    try:
        import numpy as np

        values = np.asarray([payload.features], dtype=np.float32)
        probabilities = model.predict(values, verbose=0)[0]
        selected_index = int(np.argmax(probabilities))
        return {
            "label": labels[selected_index],
            "confidence": float(probabilities[selected_index]),
            "probabilities": {
                label: float(probability)
                for label, probability in zip(labels, probabilities, strict=True)
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gesture prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Keras model could not classify this hand landmark vector.",
        ) from exc


@app.post(f"{API_PREFIX}/dataset/import")
def import_gesture_dataset(payload: GestureDatasetInput) -> dict[str, Any]:
    labels = configured_labels()
    allowed = set(labels)
    for sample in payload.samples:
        if sample.label not in allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"'{sample.label}' is not a configured gesture label.",
            )
        _validate_features(sample.features)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", *(f"f{index}" for index in range(FEATURE_COUNT))]
    needs_header = not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0
    with DATA_FILE.open("a", newline="", encoding="utf-8") as dataset:
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for sample in payload.samples:
            writer.writerow(
                {
                    "label": sample.label,
                    **{
                        f"f{index}": f"{value:.8f}"
                        for index, value in enumerate(sample.features)
                    },
                }
            )

    total, counts = dataset_summary()
    return {
        "importedSamples": len(payload.samples),
        "totalSamples": total,
        "classCounts": [{"label": label, "count": counts[label]} for label in labels],
    }


def _set_progress(progress: int, message: str) -> None:
    training_state.update(progress=progress, message=message)


def _train_in_background(job_id: str, epochs: int, batch_size: int) -> None:
    global cached_model, cached_labels, cached_model_mtime_ns
    try:
        from .trainer import train_model

        result = train_model(
            dataset_path=DATA_FILE,
            model_dir=MODEL_DIR,
            configured_labels=configured_labels(),
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=_set_progress,
        )
        with model_lock:
            cached_model = result.pop("model")
            cached_labels = result["labels"]
            cached_model_mtime_ns = MODEL_FILE.stat().st_mtime_ns
        training_state.update(
            status="succeeded",
            progress=100,
            accuracy=result["accuracy"],
            testSamples=result["testSamples"],
            message=(
                f"Training complete. Held-out accuracy: {result['accuracy']:.1%} "
                f"across {result['testSamples']} samples."
            ),
            completedAt=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.exception("Gesture model training failed")
        training_state.update(
            status="failed",
            progress=0,
            accuracy=None,
            testSamples=None,
            message=f"Training failed: {exc}",
            completedAt=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        training_state.update(jobId=job_id)


@app.post(
    f"{API_PREFIX}/training/start",
    status_code=status.HTTP_202_ACCEPTED,
)
def start_gesture_training(payload: GestureTrainingInput) -> dict[str, str]:
    state = training_state.snapshot()
    if state["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A training job is already running.",
        )
    total, counts = dataset_summary()
    eligible = {
        label: count
        for label, count in counts.items()
        if count >= MINIMUM_SAMPLES_PER_CLASS
    }
    if len(eligible) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Collect at least "
                f"{MINIMUM_SAMPLES_PER_CLASS} samples for each of two or more labels "
                "before training."
            ),
        )
    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No gesture samples have been imported yet.",
        )

    job_id = str(uuid.uuid4())
    training_state.update(
        status="running",
        progress=5,
        accuracy=None,
        testSamples=None,
        message="Preparing the labeled landmark dataset.",
        startedAt=datetime.now(timezone.utc).isoformat(),
        completedAt=None,
        jobId=job_id,
    )
    worker = threading.Thread(
        target=_train_in_background,
        args=(job_id, payload.epochs, payload.batchSize),
        name=f"gesture-training-{job_id[:8]}",
        daemon=True,
    )
    worker.start()
    return {"jobId": job_id, "status": "running"}