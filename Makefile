.PHONY: help setup pipeline pipeline-demo pipeline-snapshot ingest-snapshot upsert-snapshot upsert-from eval app up down down-clean logs qstatus parse-text retrieve-text judge-text load-archive-qd

PROJECT_RAW := $(if $(COMPOSE_PROJECT_NAME),$(COMPOSE_PROJECT_NAME),$(notdir $(CURDIR)))
PROJECT_LOWER := $(shell echo $(PROJECT_RAW) | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
QDRANT_VOLUME := $(PROJECT_LOWER)_qdrant_storage

help:
	@echo "Targets:"
	@echo "  setup             Install deps via uv (lock respected)"
	@echo "  pipeline          Full pipeline: download, chunk, embed, upsert"
	@echo "  pipeline-demo     Demo pipeline: full pipeline using small subset"
	@echo "  pipeline-snapshot Full pipeline + create timestamped snapshot"
	@echo "  ingest-snapshot   Download then create timestamped snapshot (no upsert)"
	@echo "  upsert-snapshot   Embed + upsert from most recent snapshot"
	@echo "  upsert-from       Embed + upsert from specific SNAPSHOT=path/to/dir_or_file"
	@echo "  eval              Run offline evaluation suite"
	@echo "  app               Run Streamlit UI"
	@echo "  up                Start local stack (Qdrant database)"
	@echo "  down              Stop local stack"
	@echo "  down-clean        Stop stack and remove named volumes (data loss)"
	@echo "  logs              Tail docker compose logs"
	@echo "  qstatus           Curl Qdrant /ready endpoint"
	@echo "  parse-text        Run LLM query parser against TEXT='<patient>'"
	@echo "  retrieve-text     Run retrieval pipeline on TEXT='<patient>'"
	@echo "  judge-text        Run judge pipeline on TEXT='<patient>'"
	@echo "  load-archive-qd   Download shared Qdrant snapshot and restore volume"

setup:
	uv sync
	@echo "Activate the virtualenv manually with: source .venv/bin/activate"

pipeline:
	uv run python -m ingest.main

pipeline-demo:
	uv run python -m ingest.main --demo

pipeline-snapshot:
	uv run python -m ingest.main --snapshot

ingest-snapshot:
	uv run python -m ingest.main --snapshot --prepare-only

upsert-snapshot:
	uv run python -m ingest.main --upsert-from-snapshot

upsert-from:
	@if [ -z "$(SNAPSHOT)" ]; then echo "Usage: make upsert-from SNAPSHOT=path/to/snapshot"; exit 1; fi
	uv run python -m ingest.main --upsert-from $(SNAPSHOT)

eval:
	uv run python -m eval.main

app:
	PYTHONPATH=. uv run streamlit run app/main.py

up:
	docker compose up -d --build
	@echo "TrialGPT: http://localhost:8501"
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"

down:
	docker compose down

down-clean:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

qstatus:
	curl -fsSL $${QDRANT_URL:-http://localhost:6333}/readyz || true

parse-text:
	@if [ -z "$(TEXT)" ]; then echo "Usage: make parse-text TEXT='patient description'"; exit 1; fi
	uv run python -m clinical_rag.query_parser --text "$(TEXT)"

retrieve-text:
	@if [ -z "$(TEXT)" ]; then echo "Usage: make retrieve-text TEXT='patient description'"; exit 1; fi
	uv run python -m clinical_rag.retrieval --text "$(TEXT)"

judge-text:
	@if [ -z "$(TEXT)" ]; then echo "Usage: make judge-text TEXT='patient description'"; exit 1; fi
	uv run python -m clinical_rag.judge --text "$(TEXT)"

load-archive-qd:
	@echo "Resolved Compose project name: $(PROJECT_LOWER)"
	@echo "Using volume: $(QDRANT_VOLUME)"
	@echo "Downloading Qdrant snapshot from https://storage.googleapis.com/clinicalrag_snap/qdrant_storage.tar.gz"
	@mkdir -p data
	@curl -fSL https://storage.googleapis.com/clinicalrag_snap/qdrant_storage.tar.gz -o data/qdrant_storage.tar.gz
	@echo "Stopping running stack (if any)..."
	@docker compose down || true
	@echo "Creating volume $(QDRANT_VOLUME) (if missing)..."
	@docker volume create $(QDRANT_VOLUME) >/dev/null 2>&1 || true
	@echo "Restoring snapshot into volume..."
	@docker run --rm -v $(QDRANT_VOLUME):/data -v "$(CURDIR)/data:/backup" busybox \
		sh -c "cd /data && rm -rf ./* && tar xzf /backup/qdrant_storage.tar.gz"
	@rm -f data/qdrant_storage.tar.gz
	@echo "Starting stack..."
	@$(MAKE) up
