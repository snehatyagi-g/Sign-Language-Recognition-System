# Imported landmark data provenance

## Source

- Repository: [HandGesture-Recognizer](https://github.com/shimaazizi/HandGesture-Recognizer)
- License: MIT; see [`SOURCE_LICENSE.txt`](./SOURCE_LICENSE.txt).
- Source dataset: 200 images, 50 each for Fist, OpenPalm, PeaceSign, and ThumbsUp.
- The original images are 128 × 128 and were collected in a limited set of backgrounds and by a limited number of contributors. This is a small starter dataset, not evidence of broad real-world accuracy.

## This import

MediaPipe extracted 133 valid rows from the 200 source images. The other 67 images did not produce a valid 21-landmark result and were skipped.

| Configured label | Valid rows | Skipped source images |
| --- | ---: | ---: |
| `CLOSED_FIST` | 34 | 16 |
| `OPEN_PALM` | 46 | 4 |
| `PEACE` | 22 | 28 |
| `THUMBS_UP` | 31 | 19 |

These rows are available in [`public_image_dataset_landmarks.csv`](./public_image_dataset_landmarks.csv) and were appended to the app's aggregate dataset through its import API. That API reported 1,910 existing rows and 2,043 total rows after this 133-row import. The earlier rows have separate, unverified provenance; the counts in this document describe only the MIT-source batch.

## Mapping and extraction

| Source class | Configured label |
| --- | --- |
| Fist | `CLOSED_FIST` |
| OpenPalm | `OPEN_PALM` |
| PeaceSign | `PEACE` |
| ThumbsUp | `THUMBS_UP` |

Only these direct mappings are used. Images are not duplicated into visually similar labels. The other configured labels remain available and unchanged, but have no imported examples from this source.

`extract_image_dataset.py` runs MediaPipe Tasks Vision in image mode, keeps one detected hand, and writes the same feature representation used by the browser collector: subtract the wrist landmark from each of the 21 x/y/z landmarks, divide by the maximum wrist-relative 3D distance, then flatten in MediaPipe landmark order into 63 values. Images without a valid 21-landmark result are skipped.

To reproduce the extraction after downloading/unpacking the source repository:

```bash
uv run --directory artifacts/sign-language-recognition \
  python -m ml_service.extract_image_dataset \
  --input-dir /path/to/HandGesture-Recognizer/dataset \
  --output /path/to/new-landmarks.csv
```

The extractor refuses to overwrite an existing output file. To import another CSV into the running app, use **Dataset & training → Choose CSV → Upload staged samples**. The generated CSV has a `label,f0,...,f62` header.

The imported `landmarks.csv` is local training data and is intentionally git-ignored. Keep a separate backup if it must be preserved across workspace resets or moved to another environment.