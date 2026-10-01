---
name: FastAPI API smoke tests
description: A test-client dependency quirk in this workspace and a reliable alternative for API checks.
---

In this workspace, importing `fastapi.testclient.TestClient` raises a runtime error unless the separate `httpx2` package is installed. For smoke tests that only need to verify routes, request validation, or no-model behavior, exercise the running FastAPI workflow through the shared proxy instead of adding a test-only package.

**Why:** A local test-client import failed even though the API service and its route dependencies were working; proxy requests verified the same service behavior without changing runtime dependencies.

**How to apply:** Use the configured service route at `localhost:80` for ad hoc API smoke tests. Add `httpx2` only if the project adopts automated in-process FastAPI tests.