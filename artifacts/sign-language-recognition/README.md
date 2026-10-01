# Sign Language Recognition System

A webcam-based static hand-gesture recognition workbench. MediaPipe Tasks Vision detects 21 hand landmarks in the browser; a FastAPI service sends the resulting normalized 63-value feature vector to a TensorFlow/Keras classifier. The app also collects labeled samples and trains the classifier from those samples.

This is a gesture-classification project, not a complete sign-language translator. The initial labels describe static hand shapes and do not encode sign-language grammar, sentence meaning, or motion over time.

## What is included

- **Live recognition:** browser camera capture, MediaPipe landmark overlay, confidence-thresholded API predictions, an Unknown state for low-confidence results, recognized class-label text, recognition history with a clear button, and optional speech synthesis.
- **Data collection:** collect normalized landmark samples under configurable labels, review per-label totals, export CSV, and import samples into the Python service.
- **Model training and evaluation:** start training from the saved dataset and see job progress, held-out test accuracy, and the number of test samples.
- **Python ML service:** FastAPI prediction/data/training endpoints, TensorFlow/Keras classifier, and dataset validation.

The web app uses MediaPipe Tasks Vision because the webcam belongs to the user's browser, not the remote Replit server. Camera frames stay in the browser; only extracted landmark features are sent to FastAPI. Recognition history is kept in that browser's local storage and can be cleared from the live workbench. For local desktop capture, the optional Python collector uses OpenCV and MediaPipe.

## Requirements

- Python 3.11+
- Node.js 20+ and pnpm
- A modern browser with camera access
- Internet access in the browser for the first MediaPipe WebAssembly and hand-landmarker model download

## Install

From the project root:

```bash
pnpm install
uv sync
```

The Python dependencies are recorded in the workspace `pyproject.toml` and `uv.lock`; the web dependency list is in `artifacts/sign-language-recognition/package.json`.

## Run in Replit

Use the Replit Run control to start the managed web, shared API, and Python ML API services. To start the two app services manually from the project root:

```bash
PORT=22917 BASE_PATH=/ pnpm --filter @workspace/sign-language-recognition run dev
uv run python -m uvicorn ml_service.main:app --app-dir artifacts/sign-language-recognition --host 0.0.0.0 --port 8000
```

The web app is served at the project preview. The API is mounted at `/api/sign-language`. If running outside Replit, make sure the Python service is reachable at that same path or update the generated API server configuration.

## Use the webcam recognizer

1. Open the app over HTTPS (the Replit preview qualifies) or on `localhost`; browsers block camera access on most insecure origins.
2. Allow camera access when prompted.
3. Position one hand so the full palm and fingertips are visible. The overlay should show 21 tracked landmarks.
4. If the model status says no model is trained, collect labeled samples first; the app intentionally will not show made-up predictions.
5. Once a trained model is available, keep a steady hand pose in view. The adjustable confidence threshold shows weak results as **Unknown**; speech speaks only accepted predictions.
6. Confident results appear as class labels in recognized text and the browser-local history. The history clear button removes both for this browser.

The browser loads the MediaPipe WASM files and the hand-landmarker model from jsDelivr and Google-hosted assets the first time. The camera can work only in the browser that grants permission; the remote Python service cannot access your local webcam.

## Collect a dataset

1. Open the data collection view and choose one configured gesture label.
2. Start the camera and hold the selected static pose clearly in view.
3. Record a varied set of examples. Change distance, rotation, lighting, background, and hand position between samples; collect both hands if that is part of the intended use.
4. Repeat for at least two labels. Training requires at least **5 samples per class**; for a useful student project, aim for 100–300 varied samples per class.
5. Review the class counts. Export a CSV backup and import the session into the backend dataset before training.

Each sample contains the wrist-relative, scale-normalized x/y/z coordinates of 21 landmarks. The dataset is stored at:

```text
artifacts/sign-language-recognition/ml_service/data/landmarks.csv
```

The web collector exports the same `label,f0,...,f62` format. Do not mix raw pixel coordinates or a different landmark ordering into that file.

## Configure labels

Edit `ml_service/config/gesture_labels.json` and restart the Python service. Labels are class names, not translations. The defaults are static poses such as `OPEN_PALM`, `ONE_FINGER`, `PEACE`, and `THUMBS_UP`.

Dynamic signs such as waving “hello” or “thank you” cannot be reliably represented by one static hand shape. Adding those requires collecting ordered landmark sequences and training a temporal model (for example an LSTM, GRU, or Transformer), not just adding a label.

## Train the TensorFlow/Keras model

Training can be started in the app's model/data view. The equivalent Python command for a collected CSV is:

```bash
uv run --directory artifacts/sign-language-recognition python -m ml_service.trainer_cli
```

The trainer:

1. Validates the 63 landmark feature columns and configured class labels.
2. Requires at least five samples for each of two or more classes.
3. Uses a stratified 80/20 train/test split.
4. Fits a dense Keras classifier (128 and 64 ReLU units with dropout and a softmax output).
5. Applies early stopping using a validation subset and reports accuracy on the held-out test set.
6. Saves the model and its class-order metadata under `ml_service/models/`.

After training, the UI reports the held-out test accuracy and test sample count. This is an evaluation on the collected dataset, not a guarantee of real-world accuracy. The UI does not show evaluation numbers until an actual training run completes.

The `.keras` model is local and is intentionally git-ignored. Keep a backup if you need to transfer a trained model to another environment.

### Optional OpenCV desktop collector

The browser collector is recommended. To capture directly from a webcam with Python/OpenCV instead, run one recording session per label:

```bash
uv run --directory artifacts/sign-language-recognition python -m ml_service.webcam_collector --label OPEN_PALM
```

Press **C** to save a sample and **Q** to stop. Choose each label from `ml_service/config/gesture_labels.json`. This writes the same normalized `label,f0,...,f62` dataset used by training. A webcam cannot be tested from the remote Replit environment; run this command on a computer with a webcam and a graphical display.

## API

- `GET /api/sign-language/labels` — configured classes
- `GET /api/sign-language/model/status` — model readiness and dataset counts
- `GET /api/sign-language/training/status` — current training job
- `POST /api/sign-language/predict` — classify one normalized feature vector
- `POST /api/sign-language/dataset/import` — append labeled samples to the CSV dataset
- `POST /api/sign-language/training/start` — start a real background training run

FastAPI's interactive API docs are available at `/docs` on the Python service's local port when running it directly.

## Limitations and future work

- Only predefined static gestures are supported; the current model is not a sign-language sentence translator.
- Accuracy depends on lighting, camera quality, hand visibility and orientation, and dataset diversity. Reported test accuracy is specific to the locally collected held-out samples and does not establish real-world accuracy.
- One hand is classified at a time. Two-hand classes and motion-based signs need a different feature and model pipeline.
- Browser camera access and the initial MediaPipe model download depend on device permissions and network access.
- Future work: dynamic sequence classification with LSTM/GRU/Transformer models, two-hand tracking, more classes, better lighting robustness, language-specific sentence interpretation, mobile support, and personalization.