# EvalOps: Continuous AI Evaluation & Observability Harness

This harness acts as the continuous quality-assurance gate for production. It integrates RAGAS, DeepEval, and a specialized MMLU/Routing benchmarking engine to validate both model answers and routing proxy performance.

---

## Getting Started

### Prerequisites

- Python `>=3.10` and `<3.14` (Tested with Python `3.13.11`)
- [Poetry](https://python-poetry.org/) (Tested with Poetry `2.0.1`)

### Installation

1. Clone or navigate to the repository directory.
2. Install dependencies (including development and evaluation libraries):
   ```bash
   poetry install
   ```

### Environment Configuration

1. Create a `.env` file from the example template:
   ```bash
   copy .env.example .env
   ```
2. Open `.env` and fill in the necessary API keys and database URLs (e.g., Qdrant, OpenRouter, and LangSmith).

---

## Directory Structure

```text
├── .github/                 # CI/CD Workflows (GitHub Actions)
├── config/                  # Configuration files (YAML, system settings)
├── deploy/                  # Deployment configurations (K8s, Docker, Spark)
├── docs/                    # Architectural documents & diagrams
│   └── features/            # Feature-specific markdown explorations
├── scripts/                 # Utility and automation scripts (seeders, launchers)
├── src/                     # Core Source Code
│   ├── api/                 # API controllers and gateways (FastAPI)
│   ├── core/                # Core configurations, environments, constants
│   ├── database/            # Relational database schemas & client wrappers
│   ├── vectors/             # Vector database setups, collections, index configs
│   ├── agents/              # LangGraph, LangChain, or smolagents definition
│   ├── utils/               # Helper utilities (logging, formatting)
│   └── main.py              # Application Entrypoint
├── tests/                   # Test Suite
│   ├── unit/                # Pure logic and isolated component tests
│   ├── integration/         # DB, API, and multi-agent workflow tests
│   └── evaluation/          # DeepEval & RAGAS test suites
├── .env.example             # Template for local environment parameters
├── .gitignore               # Tailored Python gitignore
├── README.md                # Project startup & overview
├── pyproject.toml           # Poetry package dependencies & metadata
└── poetry.lock              # Poetry locked dependencies manifest
```

---

## Running the Project

### Starting the FastAPI Gateway Server
To run the server locally in development mode:
```bash
poetry run uvicorn src.main:app --reload
```
You can access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and verify health at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### Running Tests
To run unit and integration tests:
```bash
poetry run pytest
```
*Note: Evaluation tests requiring external LLM API keys are skipped by default. You can run them once the keys are configured in your local `.env`.*
