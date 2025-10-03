"""ClinicalTrials.gov API client (v2 /studies).

Uses the v2 REST API at `/api/v2/studies`, paginating with `pageToken`
and filtering by last update window via Essie in `query.term`.

Notes:
- Requests only `protocolSection` to limit payload size.
- Normalizes studies into a common dict schema aligning with the PRD.
"""

from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
from clinical_rag.config import load_settings


def _build_endpoint_v2(base: str) -> str:
    base = base.rstrip("/")
    # Allow base to be either .../api or .../api/v2
    if base.endswith("/v2"):
        return f"{base}/studies"
    return f"{base}/v2/studies"


# Allowed overall statuses for efficient server-side filtering (from settings)
_SETTINGS = load_settings()
ALLOWED_STATUSES = set(_SETTINGS.allowed_statuses)


def _expr_last_update_range(start: Optional[date], end: Optional[date]) -> str:
    """Builds an expr range filter for last update posted date.

    Example:
      AREA[LastUpdatePostDate]RANGE[2024-01-01,2024-12-31]
    """
    start_s = start.isoformat() if start else ""
    end_s = end.isoformat() if end else datetime.utcnow().date().isoformat()
    return f"AREA[LastUpdatePostDate]RANGE[{start_s},{end_s}]"


def _age_to_years(value: str) -> Optional[int]:
    """Convert age strings like '65 Years' or '18 Years' to integer years.

    Returns None for 'N/A' or unparsable values.
    """
    if not value or value.upper() == "N/A":
        return None
    parts = value.strip().split()
    try:
        num = int(parts[0])
    except Exception:
        return None
    # Units could be Years, Months, Weeks, Days; we only map Years here.
    unit = parts[1].lower() if len(parts) > 1 else "years"
    if unit.startswith("year"):
        return num
    # Fallback: approximate conversions if ever needed
    if unit.startswith("month"):
        return max(0, round(num / 12))
    if unit.startswith("week"):
        return max(0, round(num / 52))
    if unit.startswith("day"):
        return max(0, round(num / 365))
    return None


def map_study_v2(study: Dict[str, Any]) -> Dict[str, object]:
    """Normalize a v2 /studies study into our common schema.

    Includes minimal, high-impact eligibility additions:
    - inclusion/exclusion split
    - raw and normalized ages, sex, healthy volunteers, std ages
    - study type
    - lean structured locations with geo
    - MeSH terms from derivedSection.conditionBrowseModule
    """
    ps = study.get("protocolSection", {}) or {}
    ident = ps.get("identificationModule", {}) or {}
    status = ps.get("statusModule", {}) or {}
    conds = ps.get("conditionsModule", {}) or {}
    design = ps.get("designModule", {}) or {}
    arms = ps.get("armsInterventionsModule", {}) or {}
    elig = ps.get("eligibilityModule", {}) or {}
    locs = ps.get("contactsLocationsModule", {}) or {}
    derived = study.get("derivedSection", {}) or {}

    nct_id = (ident.get("nctId") or "").strip()
    trial_title = ident.get("officialTitle") or ident.get("briefTitle") or ""
    overall_status = status.get("overallStatus")
    # phases is a list in v2; pick first or join
    phases = design.get("phases") or []
    phase = phases[0] if isinstance(phases, list) and phases else None
    conditions = conds.get("conditions") or []
    interventions: List[str] = []
    for itv in arms.get("interventions", []) or []:
        name = itv.get("name")
        if name:
            interventions.append(name)
    # Eligibility
    eligibility_md: str = elig.get("eligibilityCriteria") or ""
    min_age_raw = elig.get("minimumAge") or None
    max_age_raw = elig.get("maximumAge") or None
    min_age_years = _age_to_years(min_age_raw or "")
    max_age_years = _age_to_years(max_age_raw or "")
    sex = elig.get("sex")
    healthy_volunteers = elig.get("healthyVolunteers")
    std_ages = elig.get("stdAges") or []

    inclusion, exclusion = split_incl_excl(eligibility_md)

    # Compose locations
    locations_struct: List[Dict[str, Any]] = []
    for loc in locs.get("locations", []) or []:
        gp = loc.get("geoPoint") or {}
        locations_struct.append(
            {
                "city": loc.get("city"),
                "state": loc.get("state"),
                "country": loc.get("country"),
                "status": loc.get("status"),
                "lat": gp.get("lat"),
                "lon": gp.get("lon"),
            }
        )

    url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""

    # MeSH terms
    mesh_terms: List[str] = []
    cb = derived.get("conditionBrowseModule", {}) or {}
    for m in cb.get("meshes") or []:
        term = m.get("term")
        if term:
            mesh_terms.append(term)
    study_type = design.get("studyType") or None

    return {
        "nct_id": nct_id,
        "trial_title": trial_title,
        "overall_status": overall_status,
        "phase": phase,
        "conditions": conditions,
        "interventions": interventions,
        # New minimal additions for eligibility-driven matching
        "study_type": study_type,
        "eligibility_markdown": eligibility_md,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "min_age_raw": min_age_raw,
        "max_age_raw": max_age_raw,
        "min_age_years": min_age_years,
        "max_age_years": max_age_years,
        "sex": sex,
        "healthy_volunteers": healthy_volunteers,
        "std_ages": std_ages,
        "locations": locations_struct,
        "mesh_terms": mesh_terms,
        "url": url,
    }


def split_incl_excl(md: str) -> Tuple[List[str], List[str]]:
    """Heuristic splitter for inclusion/exclusion criteria markdown."""
    import re

    if not md:
        return [], []
    text = re.sub(r"\r", "", md)
    # Try Inclusion -> Exclusion path
    m = re.split(r"\n\s*exclusion criteria\s*:?\s*\n", text, flags=re.I)
    if len(m) == 2:
        inc_block, exc_block = m
    else:
        # Try reverse
        m = re.split(r"\n\s*inclusion criteria\s*:?\s*\n", text, flags=re.I)
        if len(m) == 2:
            pre, inc_block = m
            exc_split = re.split(
                r"\n\s*exclusion criteria\s*:?\s*\n", inc_block, flags=re.I
            )
            inc_block, exc_block = (
                (exc_split[0], exc_split[1]) if len(exc_split) == 2 else (inc_block, "")
            )
        else:
            inc_block, exc_block = text, ""

    bullet = r"^\s*(?:[-*•]|\d+\.|\([a-zA-Z]\))\s+(.+)$"
    inc = re.findall(bullet, inc_block, flags=re.M)
    exc = re.findall(bullet, exc_block, flags=re.M)
    return [s.strip() for s in inc], [s.strip() for s in exc]


def iter_study_fields(
    *,
    start: Optional[date],
    end: Optional[date],
    page_size: int = 100,
    timeout_s: float = 30.0,
    base_url: str = "https://clinicaltrials.gov/api/v2",
) -> Iterator[Dict[str, object]]:
    """Iterate normalized study records within a date window (v2 only)."""
    endpoint = _build_endpoint_v2(base_url)
    query_term = _expr_last_update_range(start, end)
    page_token: Optional[str] = None
    while True:
        params = {
            "format": "json",
            "markupFormat": "markdown",
            "pageSize": str(page_size),
            # Request protocolSection and condition browse meshes for MeSH terms
            # Use lowerCamel for derivedSection path per v2 docs; pipe-delimited list
            "fields": "ProtocolSection|derivedSection.conditionBrowseModule.meshes",
            "query.term": query_term,
            # Efficient server-side status filter
            "filter.overallStatus": "|".join(sorted(ALLOWED_STATUSES)),
        }
        if page_token:
            params["pageToken"] = page_token

        with httpx.Client(
            timeout=timeout_s, headers={"User-Agent": "clinical-rag/0.1"}
        ) as client:
            resp = client.get(endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()

        studies = data.get("studies", []) or []
        for st in studies:
            rec = map_study_v2(st)
            # Defensive client-side guard
            if rec.get("overall_status") in ALLOWED_STATUSES:
                yield rec

        page_token = data.get("nextPageToken")
        if not page_token:
            break
