# Streamlit App

The Streamlit UI lets users enter a patient description and view potentially-eligible clinical trials along with inclusion/exclusion bullets.

## Running Locally

### Inside Docker (recommended)
```
make up  # starts qdrant + streamlit containers
```
Visit http://localhost:8501. The UI uses the backend services available at `QDRANT_URL` from the environment.

### Outside Docker
```
make app  # PYTHONPATH=. uv run streamlit run app/main.py
```
Requires `uv sync` and a running Qdrant instance (e.g. `make up` without the app container, or manual `docker compose up qdrant`).

## Environment
- Reads `.env` via `python-dotenv`.
- Needs `OPENAI_API_KEY` and `QDRANT_URL` if LLM parsing/judging is enabled.

## Code Layout
- `main.py` – single Streamlit page.
  - Loads settings via `clinical_rag.config`.
  - Runs parser → retrieval → optional judge pipeline per patient description.
  - Displays trial cards with inclusion/exclusion bullets and links.

## Development Tips
- Use `make parse-text` / `make retrieve-text` / `make judge-text` to debug pipeline components directly.
- When developing the UI, run `make app` while Qdrant runs via Docker (`make up qdrant`).
- Health endpoint: Streamlit exposes `_stcore/health` (used in Docker healthcheck).
