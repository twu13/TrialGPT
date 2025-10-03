"""Trial-level retrieval against Qdrant with optional filters.

Embeds a deterministic query text from the parser JSON and searches a
trial-level collection (one vector per trial). Applies payload filters
for gender compatibility, age bounds, and optional location facets.
Returns inclusion/exclusion bullets from the trial payload in a format
compatible with the judge and app layers.
"""

from functools import lru_cache
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models
from fastembed import TextEmbedding
from clinical_rag.config import load_settings

from clinical_rag.query_parser import build_query_text


_SETTINGS = load_settings()


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(url=_SETTINGS.qdrant_url)


@lru_cache(maxsize=1)
def _get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=_SETTINGS.embedding_model_name)


def build_filters(
    spec: Dict,
    *,
    recruiting_only: bool = True,
) -> Optional[http_models.Filter]:
    must: List[http_models.FieldCondition] = []

    # Recruiting status: handled upstream during ingest; no filter here to avoid redundancy

    # Gender compatibility: if patient sex present, allow trials with same sex or ALL
    sex = (spec.get("sex") or "").upper()
    if sex in ("MALE", "FEMALE"):
        must.append(
            http_models.FieldCondition(
                key="gender",
                match=http_models.MatchAny(any=[sex, "ALL"]),
            )
        )

    # Age bounds: ensure trial min_age <= patient age (skip max_age to avoid excluding trials with null max)
    age = spec.get("age")
    if isinstance(age, int):
        must.append(
            http_models.FieldCondition(
                key="min_age",
                range=http_models.Range(lte=age),
            )
        )
        # Optional: enforce max_age >= age if present in payload; omitted here to avoid excluding nulls

    # Location facets
    loc = spec.get("location") or {}
    if isinstance(loc, dict):
        city = (loc.get("city") or "").strip().lower() or None
        state = (loc.get("state") or "").strip().lower() or None
        country = (loc.get("country") or "").strip().lower() or None
        if city:
            must.append(
                http_models.FieldCondition(
                    key="location_cities",
                    match=http_models.MatchValue(value=city),
                )
            )
        if state:
            must.append(
                http_models.FieldCondition(
                    key="location_states",
                    match=http_models.MatchValue(value=state),
                )
            )
        if country:
            must.append(
                http_models.FieldCondition(
                    key="location_countries",
                    match=http_models.MatchValue(value=country),
                )
            )

    if not must:
        return None
    return http_models.Filter(must=must)


def _postfilter_by_max_age(spec: Dict, hits: List) -> List:
    """Filter out hits where patient's age exceeds trial max_age when max_age is set."""
    age = spec.get("age")
    if not isinstance(age, int):
        return hits
    kept: List = []
    for h in hits:
        p = h.payload or {}
        max_age = p.get("max_age")
        if max_age is None or (isinstance(max_age, int) and age <= max_age):
            kept.append(h)
    return kept


class _PayloadWrapper:
    """Lightweight object to mimic Qdrant hit interface used downstream (has .payload)."""

    def __init__(self, payload: Dict):
        self.payload = payload


def _search_trials(spec: Dict, *, max_trials: int):
    client = _get_client()
    collection = _SETTINGS.collection_name
    qtext = build_query_text(spec)
    qfilter = build_filters(spec)
    embedder = _get_embedder()
    vec = list(embedder.embed([qtext]))[0]
    hits = client.search(
        collection_name=collection,
        query_vector=vec,
        query_filter=qfilter,
        limit=max_trials,
        with_payload=True,
        with_vectors=False,
    )
    return _postfilter_by_max_age(spec, hits)[:max_trials]


def retrieve_with_exclusions(
    spec: Dict, *, max_trials: int = 5
) -> Dict[str, Dict[str, List]]:
    """Return up to max_trials, with inclusion/exclusion bullets reconstructed from trial payloads."""
    hits = _search_trials(spec, max_trials=max_trials)
    grouped: Dict[str, Dict[str, List]] = {}
    for h in hits:
        p = h.payload or {}
        nct = p.get("nct_id")
        if not nct:
            continue
        incl = p.get("inclusion_criteria") or []
        excl = p.get("exclusion_criteria") or []
        common = {
            "nct_id": nct,
            "trial_title": p.get("trial_title"),
            "overall_status": p.get("overall_status"),
            "gender": p.get("gender"),
            "min_age": p.get("min_age"),
            "max_age": p.get("max_age"),
            "conditions": p.get("conditions") or [],
            "interventions": p.get("interventions") or [],
            "locations": p.get("locations") or [],
            "location_cities": p.get("location_cities") or [],
            "location_states": p.get("location_states") or [],
            "location_countries": p.get("location_countries") or [],
            "phase": p.get("phase"),
            "study_type": p.get("study_type"),
            "url": p.get("url"),
        }
        incl_wrapped = [
            _PayloadWrapper(
                {**common, "chunk_id": f"{nct}:eligibility_inclusion:{i}", "text": t}
            )
            for i, t in enumerate(incl)
            if t
        ]
        excl_wrapped = [
            _PayloadWrapper(
                {**common, "chunk_id": f"{nct}:eligibility_exclusion:{i}", "text": t}
            )
            for i, t in enumerate(excl)
            if t
        ]
        if not incl_wrapped and not excl_wrapped:
            incl_wrapped = [
                _PayloadWrapper({**common, "chunk_id": f"{nct}:metadata", "text": None})
            ]
        grouped[nct] = {"info": common, "incl": incl_wrapped, "excl": excl_wrapped}
    return grouped


def _print_grouped(
    grouped: Dict[str, Dict[str, List]], *, bullets: bool = False
) -> None:
    for nct, ctx in grouped.items():
        incl = ctx.get("incl", [])
        excl = ctx.get("excl", [])
        title = None
        if incl:
            title = (incl[0].payload or {}).get("trial_title")
        elif excl:
            title = (excl[0].payload or {}).get("trial_title")
        print(f"{nct}\t(title={title or ''})\tincl={len(incl)}\texcl={len(excl)}")
        if bullets:
            for h in incl[:5]:
                p = h.payload or {}
                print(f"  + [{p.get('chunk_id')}] {p.get('text')}")
            for h in excl[:5]:
                p = h.payload or {}
                print(f"  - [{p.get('chunk_id')}] {p.get('text')}")


def main() -> None:
    import argparse, json, os, sys

    parser = argparse.ArgumentParser(description="Trial-level retrieval test CLI")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Free-text description (uses LLM parser)")
    g.add_argument("--spec-json", type=str, help="Raw JSON spec string (bypass parser)")
    g.add_argument(
        "--spec-file", type=str, help="Path to JSON spec file (bypass parser)"
    )
    parser.add_argument(
        "--max-trials", type=int, default=5, help="Max trials to retrieve"
    )
    parser.add_argument(
        "--print-bullets",
        action="store_true",
        help="Print up to 5 incl/excl bullets per trial",
    )
    args = parser.parse_args()

    # Build spec
    spec: Dict
    if args.spec_json:
        spec = json.loads(args.spec_json)
    elif args.spec_file:
        with open(args.spec_file, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        # Parse via LLM
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY required for --text parsing", file=sys.stderr)
            sys.exit(2)
        from clinical_rag.query_parser import parse as llm_parse

        spec = llm_parse(args.text)

    grouped = retrieve_with_exclusions(spec, max_trials=args.max_trials)
    _print_grouped(grouped, bullets=args.print_bullets)


if __name__ == "__main__":
    main()
