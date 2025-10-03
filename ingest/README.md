# Ingestion Pipeline Overview

The ingestion stack pulls trial data from the ClinicalTrials.gov v2 API, normalizes the payload, and writes one vector per trial into Qdrant. Everything lives in `ingest/main.py`, which exposes a CLI (`uv run python -m ingest.main`) and is wired to make targets (`make pipeline`, `make pipeline-demo`, etc.).

## Data Flow
1. **Source fetch** (`iter_study_fields` in `ctgov_client.py`)
   - Calls the v2 `/studies` endpoint with `LastUpdatePostDate` filters derived from `DATA_START_DATE` / `DATA_END_DATE` in the `.env`.
   - Limits to actionable statuses (`RECRUITING`, `ENROLLING_BY_INVITATION`, `NOT_YET_RECRUITING`, `ACTIVE_NOT_RECRUITING`).
   - Requests the `protocolSection` and derived modules necessary for conditions, interventions, MeSH terms, and geo locations.
2. **Normalization** (`map_study_v2`)
   - Converts protocol fields into a compact record (`trial_title`, `conditions`, `interventions`, ages, gender, location list, etc.).
   - Splits eligibility markdown into inclusion and exclusion bullet arrays.
3. **Embedding & upsert** (`ingest.main`)
   - Builds a deterministic text from title/conditions/interventions.
   - Uses FastEmbed (MiniLM / BGE-small) to embed the text.
   - Uploads vectors plus payload to the configured Qdrant collection, ensuring indexes exist for gender and location filters.

## Snapshotting
The CLI accepts `--snapshot` / `--prepare-only` flags:
- `--snapshot` writes `data/trials.jsonl` plus a timestamped folder under `data/snapshots/` with a `manifest.json` capturing date window, model, and counts.
- `--prepare-only --snapshot` downloads and normalizes without touching Qdrant.
- `--upsert-from-snapshot` or `--upsert-from <path>` re-embeds and uploads from a stored snapshot, letting you restore/share datasets without re-hitting the API.

Snapshots can be archived (`.tar.gz`) for sharing; `make pipeline-snapshot` automates generation.

## Key CLI Flags
- `--demo` – sample ~50 trials (useful for local smoke tests).
- `--resume-from`, `--page-size`, `--batch-size`, `--parallel` – tune ingestion resilience and performance.
- `--upsert-from` / `--upsert-from-snapshot` – replay existing data.

## Make Targets
- `make pipeline` – full ingest (download → embed → upsert).
- `make pipeline-demo` – demo-sized ingest.
- `make pipeline-snapshot` / `make ingest-snapshot` – snapshot without upsert.
- `make upsert-snapshot` / `make upsert-from SNAPSHOT=...` – replay a stored dataset.

## Environment
`ingest.main` calls `load_settings()` which reads `.env` for:
- `CTGOV_API_BASE`
- `DATA_START_DATE`, `DATA_END_DATE`
- `QDRANT_URL`, `COLLECTION_NAME`

OpenAI keys are not required for ingestion.

The script prints summary counts (trials prepared, Qdrant point total) and optionally writes snapshots, enabling reproducible evaluation runs.
