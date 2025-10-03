# TrialGPT

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![TrialGPT Cover](images/cover.png)

## Project Overview

TrialGPT is a retrieval-augmented generation (RAG) system that helps patients quickly identify clinical trials from [ClinicalTrials.gov](https://clinicaltrials.gov/) that might fit their medical situation. It turns free-text descriptions into structured search queries, retrieves likely matches from a Qdrant vector database of trials, and displays trial metadata and inclusion/exclusion information along with LLM-based eligibility checking.

Key features:

- **Focused ingest pipeline** to pull trial data from ClinicalTrials.gov v2 API.
- **Streamlit dashboard** for interactive querying and exploration.
- **Sentiment and keyword retrieval** that matches queries to relevant trials.
- **Eligibility determination** by an LLM for parsing trial eligibility.

## Dataset

We load data from the `ClinicalTrials.gov v2 /studies API`. During ingest we:

- Filter to trials whose `overall_status` is one of `RECRUITING`, `ENROLLING_BY_INVITATION`, `NOT_YET_RECRUITING`, or `ACTIVE_NOT_RECRUITING`.
- Normalize the raw protocol payload into a compact schema (title, conditions, interventions, age/sex limits, locations, etc.) of relevant datapoints for retrieval.
- Split the eligibility markdown into inclusion and exclusion bullet lists.
- Embed **one vector per trial** (MiniLM / BGE-small) and upsert into a vector database for efficient search and filtering.

Snapshots of normalized data can be generated for reproducibility and stored under `data/snapshots/`.

## Project Structure

```
├── app/                    # Streamlit UI
├── clinical_rag/           # Core configuration, parser, retrieval, judge, prompts
├── ingest/                 # ClinicalTrials.gov client and ingest CLI
├── eval/                   # Gold generation, judge pipeline, evaluation notebooks
├── data/                   # Generated artifacts (trials.jsonl, snapshots, etc.)
├── images/                 # Static assets (cover image, etc.)
├── Dockerfile              # uv-based image for the app container
├── docker-compose.yml      # Qdrant + Streamlit services
├── Makefile                # Convenience commands (ingest, eval, up/down, smoke tests)
├── pyproject.toml          # Dependency management with uv
└── README.md               # You are here
```

## Getting Started

### 1. Prerequisites

- macOS or Linux with **Docker** and **Docker Compose**
- **Python 3.12+** and the **uv** dependency manager ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/))
- `make` utility (optional but recommended)

### 2. Environment File

Fill in your credentials for `OPEN_API_KEY` in the `.env.example` file then copy.
Modify `DATA_START_DATE` / `DATA_END_DATE` if a broader trial window is desired.

```bash
cp .env.example .env
```

The app, ingest, and evaluation entry points auto-load `.env` via `python-dotenv`.

### 3. Install Python Dependencies

```bash
make setup  # uv sync
```

### 4. Start the Local Stack

Build the Streamlit container and bring up Qdrant (local vector database).

```bash
make up  
# After startup the URLs are printed:
# - Streamlit UI: http://localhost:8501
# - Qdrant dashboard: http://localhost:6333/dashboard
```

Useful companions:

- `make logs` – follow combined service logs
- `make down` – stop containers (retains volumes)
- `make down-clean` – stop and drop volumes (clears Qdrant data)

### 5. Load the Qdrant Database with Clinical Trials

```bash
make pipeline-demo  # download a small set of trials and upsert them
```

Additional options are as follows:

> **Options for Adding Data to Qdrant (Choose one)**
>
> - `make pipeline-demo` downloads only ~50 trials and upserts them—ideal for smoke testing.
> - `make pipeline` runs the full ingest (all pages in the configured date window) and pushes directly to Qdrant.
> - `make load-archive-qd` downloads and restores a full snapshot (dated **September 19, 2025**) that powers the evaluation notebooks.
> - `make down-clean` stop stack and remove named volumes (clears Qdrant data).

### 6. Visit the App

Open http://localhost:8501 in your browser. With the shared snapshot (September 19, 2025) in place, try a query such as `“65 y/o male with non-small cell lung cancer”` to see eligible trials.

### (Optional) CLI Smoke Tests

Run targeted CLI flows without opening the UI (Qdrant must be running):

```bash
make parse-text TEXT="65 year old woman with type 2 diabetes"
make retrieve-text TEXT="..."
make judge-text TEXT="..."
```

Those commands exercise the parser, retrieval, and judge modules respectively.

## Evaluation

The evaluation suite lives in `eval/` and can be executed end-to-end with:

```bash
make eval
```

`eval.main` orchestrates four steps:

1. `eval.generate_gold` builds a gold retrieval dataset using the latest snapshot and a generative LLM.
2. `retrieval_eval.ipynb` computes retrieval metrics such as Hit@K and MRR.
3. `eval.generate_judge` re-runs retrieval and classifies eligibility with the LLM judge.
4. `judge_eval.ipynb` analyzes judge outcomes using LLM-As-A-Judge, highlighting false negatives/positives.

For deeper inspection, open the notebooks in `eval/` directly.

## License

Released under the [MIT License](LICENSE).
