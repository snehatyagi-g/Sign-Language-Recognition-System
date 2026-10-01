# Sign Language Recognition System

An academic hand-gesture workbench with browser-based MediaPipe tracking, a Python FastAPI service, and a TensorFlow/Keras classifier trained on collected landmark samples.

## Run & Operate

- `pnpm --filter @workspace/sign-language-recognition run dev` — run the web app through its managed Replit workflow (the workflow provides `PORT` and `BASE_PATH`)
- `python -m uvicorn ml_service.main:app --app-dir artifacts/sign-language-recognition --host 0.0.0.0 --port 8000` — run the Python ML API (or use its managed workflow)
- `uv sync` — install Python dependencies
- `pnpm install` — install workspace dependencies
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- Optional: `uv run --directory artifacts/sign-language-recognition python -m ml_service.trainer_cli` — train the classifier from the saved CSV

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React, Vite, browser MediaPipe Tasks Vision
- ML API: FastAPI, TensorFlow/Keras, pandas, scikit-learn, Python 3.11
- Optional local webcam collector: OpenCV + Python MediaPipe
- API contracts: OpenAPI + generated Orval client
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/sign-language-recognition/src/` — webcam UI, data collection, and generated API client consumers
- `artifacts/sign-language-recognition/ml_service/` — FastAPI routes, dataset, Keras trainer, and configured gesture labels
- `lib/api-spec/openapi.yaml` — source of truth for typed frontend API calls
- `artifacts/sign-language-recognition/README.md` — install, data collection, training, and limitation guide

## Architecture decisions

- Webcam video and MediaPipe landmark extraction stay in the browser; only normalized landmark vectors are sent to FastAPI.
- Predictions are disabled unless the Python service loads a real trained Keras model; no bundled or synthetic prediction model is presented.
- Training data and trained model files are local runtime artifacts and are excluded from version control.
- The initial labels represent static hand poses, not sign-language words or sentence translation.

## Product

Users can track a live hand, inspect landmark features, collect and import labeled samples, train a Keras classifier, and see predictions only when the model is ready. Browser speech output is optional.

## User preferences

Keep the project functional and truthful: do not replace webcam, model, training, or prediction behavior with hardcoded outputs.

## Gotchas

- The app preview needs both the managed web service and the `ml-api` service.
- Browser webcam access requires HTTPS or localhost; webcam use cannot be verified from the remote server.
- If you change endpoint shapes, update `lib/api-spec/openapi.yaml` and rerun codegen before frontend changes.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
