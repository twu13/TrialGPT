"""Trial-level retrieval against Qdrant with optional filters.

Embeds a deterministic query text from the parser JSON and searches a
trial-level collection (one vector per trial). Applies payload filters
for gender compatibility, age bounds, and optional location facets.
Returns inclusion/exclusion bullets from the trial payload in a format
compatible with the judge and app layers.
"""

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models
from fastembed import TextEmbedding
from clinical_rag.config import load_settings

from typing import OrderedDict as _OrderedDict

from clinical_rag.query_parser import build_query_components, build_query_text


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


_COMPONENT_WEIGHTS = {
    "conditions": 0.50,
    "medications": 0.25,
    "extra_terms": 0.25,
    "sex": 0.0,
}


def _weighted_query_vector(spec: Dict) -> List[float]:
    components: _OrderedDict[str, str] = build_query_components(spec)
    items: List[tuple[str, str, float]] = []
    for name, text in components.items():
        weight = _COMPONENT_WEIGHTS.get(name, 0.0)
        if not text or weight <= 0:
            continue
        items.append((name, text, weight))

    if not items:
        # Fallback to a single embedding of whatever query text is available
        qtext = build_query_text(spec) or " "
        embedder = _get_embedder()
        return list(embedder.embed([qtext]))[0]

    total_weight = sum(weight for _, _, weight in items)
    if total_weight <= 0:
        qtext = build_query_text(spec) or " "
        embedder = _get_embedder()
        return list(embedder.embed([qtext]))[0]

    embedder = _get_embedder()
    texts = [text for _, text, _ in items]
    raw_vectors = list(embedder.embed(texts))
    norm_weights = [weight / total_weight for _, _, weight in items]

    weighted_vector: List[float] | None = None
    for vec, norm_weight in zip(raw_vectors, norm_weights):
        scaled = [norm_weight * v for v in vec]
        if weighted_vector is None:
            weighted_vector = scaled
        else:
            weighted_vector = [a + b for a, b in zip(weighted_vector, scaled)]

    # weighted_vector cannot be None because items is non-empty
    return weighted_vector or []


@lru_cache(maxsize=1)
def get_location_facets(limit_per_scroll: int = 256) -> Dict[str, object]:
    """Return unique location options from Qdrant payloads.

    Results include sorted lists of countries, states keyed by country, and cities keyed by (country, state).
    """

    client = _get_client()
    collection = _SETTINGS.collection_name

    countries: set[str] = set()
    states_by_country: Dict[str, set[str]] = defaultdict(set)
    cities_by_region: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    next_offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=collection,
            limit=limit_per_scroll,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        if isinstance(scroll_result, tuple):
            points, next_offset = scroll_result
        else:  # pragma: no cover - compatibility path
            points = scroll_result[0]
            next_offset = scroll_result[1]

        if not points:
            break
        for point in points:
            payload = point.payload or {}
            for loc in payload.get("locations") or []:
                if not isinstance(loc, dict):
                    continue
                country = (loc.get("country") or "").strip().lower()
                state = (loc.get("state") or "").strip().lower()
                city = (loc.get("city") or "").strip().lower()
                if not country and not state and not city:
                    continue
                if country:
                    countries.add(country)
                if country and state:
                    states_by_country[country].add(state)
                    if city:
                        cities_by_region[(country, state)].add(city)
                elif country and city:
                    cities_by_region[(country, "")].add(city)
        if next_offset is None:
            break

    sorted_countries = sorted(countries)
    sorted_states = {
        country: sorted(values) for country, values in states_by_country.items()
    }
    sorted_cities = {key: sorted(values) for key, values in cities_by_region.items()}
    return {
        "countries": sorted_countries,
        "states_by_country": sorted_states,
        "cities_by_region": sorted_cities,
    }


def _search_trials(spec: Dict, *, max_trials: int):
    client = _get_client()
    collection = _SETTINGS.collection_name
    qfilter = build_filters(spec)
    vec = _weighted_query_vector(spec)
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
