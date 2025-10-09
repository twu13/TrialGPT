"""Centralized environment configuration.

Simple, dependency-light loader for core settings. Prefer this over
sprinkling os.getenv calls across modules.
"""

from dataclasses import dataclass
from datetime import date
import os
from typing import Optional


def _parse_date(name: str) -> Optional[date]:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        # Leave as None if value is not a valid ISO date; callers can warn if needed.
        return None


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    qdrant_url: str
    collection_name: str
    data_start_date: Optional[date]
    data_end_date: Optional[date]
    ctgov_api_base: str
    embedding_model_name: str
    llm_model_name: str
    allowed_statuses: list[str]


DEFAULT_ALLOWED_STATUSES = [
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
]

DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL_NAME = "gpt-4o"


def load_settings() -> Settings:
    """Load settings from environment with sensible defaults.

    Notes:
    - Allowed statuses are configured in code via DEFAULT_ALLOWED_STATUSES.
    - Embedding model name is configured in code via DEFAULT_EMBEDDING_MODEL_NAME.
    """
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=os.getenv("COLLECTION_NAME", "clinical_trials"),
        data_start_date=_parse_date("DATA_START_DATE"),
        data_end_date=_parse_date("DATA_END_DATE"),
        ctgov_api_base=os.getenv("CTGOV_API_BASE", "https://clinicaltrials.gov/api/v2"),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        llm_model_name=os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
        allowed_statuses=DEFAULT_ALLOWED_STATUSES,
    )


__all__ = ["Settings", "load_settings"]
