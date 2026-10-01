---
name: Dataset import state
description: Preserve live samples and distinguish a newly imported batch from aggregate counts and model freshness.
---

Treat the dataset as live user data. Keep provenance-specific batch counts separate from aggregate CSV counts, and check whether the latest training run finished before or after a new import. A successful model status does not prove that the model includes recently appended samples.

**Why:** A live project status changed between a preflight check and an append while the app continued receiving user activity. The import preserved existing rows, but reporting only the earlier count would have misrepresented dataset coverage and model freshness.

**How to apply:** Use the service’s import endpoint, record its imported and total counts, verify the saved aggregate, and compare the training completion time with the import time before saying the model was retrained on the new batch.