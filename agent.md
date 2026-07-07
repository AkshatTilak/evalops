# EvalOps Developer Agent Guidelines

This document details coding standards and requirements specific to the **EvalOps** submodule within the monorepo architecture. For general platform standards, refer to the [Root Monorepo Guidelines](../../agent.md).

---

## 1. Submodule Boundary & Interfaces

EvalOps serves as the continuous evaluation gate and dashboard. It integrates via:
1. **API Router (`api.py`)**: Mounts FastAPI endpoints (e.g. `/api/evalops/dashboard`, `/api/evalops/reports/*`) to serve evaluation results to developers.
2. **CLI Runner**: Script entrypoints (e.g. `bench_gguf.py`, `bench_mmlu.py`) run as standalone jobs during CI/CD execution.

---

## 2. Test Execution Context

Because EvalOps is a submodule of the monorepo:
- **No Local Virtualenv**: Do not create or run tests inside a separate virtualenv under `projects/evalops/`. All tests run inside the parent poetry environment from the monorepo root.
- **Qualified Imports**: Always use fully qualified imports to reference submodules and shared packages:
  ```python
  from common.config import settings
  from projects.guardroute.src.main import app
  ```
- **Service Integration**: Tests should spin up (or connect to) the running `gateway` and `inference` servers using `httpx.AsyncClient` rather than mocking internal FastAPI state.

---

## 3. Evaluation Workflows

- **RAGAS Evaluations**: Runs context recall and faithfulness evaluations against SyntraFlow's collection chunks.
- **Classifier Verification**: Asserts precision/recall and cold-start loading latencies of the local classifier model via the `/infer/classify` endpoint.
- **LiteLLM Fallback Assertions**: Inject mock RateLimit (HTTP 429) errors on Google API endpoints to assert that LiteLLM transparently routes requests to OpenRouter fallbacks without transaction failure.
