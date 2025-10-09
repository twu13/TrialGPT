import json
from typing import Any, Iterable, List, Tuple

import streamlit as st

from clinical_rag.config import load_settings
from clinical_rag.query_parser import parse
from clinical_rag.retrieval import retrieve_with_exclusions, get_location_facets
from clinical_rag.judge import judge_grouped


@st.cache_data(show_spinner=False)
def _load_location_facets() -> dict:
    try:
        return get_location_facets()
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        st.warning(f"Unable to load location filters: {exc}")
        return {"countries": [], "states_by_country": {}, "cities_by_region": {}}


def _label_for(value: Any) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, str):
        value = value.strip()
        return value.title() if value else "Not specified"
    return str(value)


def _make_options(values: Iterable[Any]) -> List[Tuple[str, Any]]:
    opts: List[Tuple[str, Any]] = [("Not specified", None)]
    for val in values:
        if val is None:
            continue
        opts.append((_label_for(val), val))
    return opts


def _selectbox_with_reset(
    label: str,
    options: List[Tuple[str, Any]],
    *,
    key: str,
    disabled: bool = False,
) -> Any:
    if not options:
        options = [("Not specified", None)]
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[0]
    selected: Tuple[str, Any] = st.selectbox(  # type: ignore[assignment]
        label,
        options,
        key=key,
        format_func=lambda opt: opt[0] if opt else "Not specified",
        disabled=disabled,
    )
    return selected[1] if selected else None


def main() -> None:
    # ---------- Page config ----------
    st.set_page_config(
        page_title="TrialGPT",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # ---------- Styles ----------
    st.markdown(
        """
<style>
  /* --- Layout width --- */
  .block-container { max-width: 760px; margin: 0 auto; }

  /* --- Single centered search card --- */
  .search-card {
      background: #ffffff;
      padding: 1.5rem;
      border-radius: 14px;
      border: 1px solid #e9ecef;
      box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  }
  .search-card h1 { margin: 0 0 .25rem 0; font-size: 1.6rem; }
  .sub { color: #6c757d; margin-bottom: 1rem; }

  .search-tips {
      background-color: #f5faff;
      padding: .75rem 1rem;
      border-radius: 10px;
      border-left: 4px solid #2196f3;
      margin-top: .75rem;
      margin-bottom: .75rem;
  }

  .search-info {
      margin-top: 0.75rem;
  }

  .elig-ELIGIBLE { background: #d4edda; color: #155724; padding: 0.2rem 0.45rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
  .elig-INELIGIBLE { background: #f8d7da; color: #721c24; padding: 0.2rem 0.45rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
  .elig-POSSIBLY_ELIGIBLE { background: #d4edda; color: #155724; padding: 0.2rem 0.45rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }

  .badge { background: #eef2f7; color: #495057; padding: 0.15rem 0.45rem; border-radius: 6px; font-size: 0.75rem; font-weight: 650; margin-right: .35rem; display: inline-block; }
  .meta { color: #6c757d; font-size: .9rem; margin-top: .35rem; }

  /* --- Remove ONLY the rounded top "decoration" pill; keep Streamlit's header bar default --- */
  :where(div[data-testid="stDecoration"],
         div[data-testid="stDecorationContainer"]) {
      display: none !important;
      visibility: hidden !important;
      height: 0 !important;
      padding: 0 !important;
      margin: 0 !important;
      box-shadow: none !important;
      background: transparent !important;
      border: 0 !important;
  }

  /* Slightly reduce top padding to tighten the hero area (header stays default) */
  main .block-container { padding-top: 1rem; }

  

  /* Large centered loader shown inline (below UI) */
  .loading-box { display: flex; align-items: center; justify-content: center; padding: 1.5rem 0; }
  .spinner { width: 84px; height: 84px; border-radius: 50%; border: 10px solid #e9ecef; border-top-color: #667eea; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
""",
        unsafe_allow_html=True,
    )

    # ---------- Search Card (header + input + tips) ----------
    st.markdown("<h1>🏥 TrialGPT</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="sub">Find eligible trials on ClinicalTrials.gov that match your health needs.</p>',
        unsafe_allow_html=True,
    )

    location_facets = _load_location_facets()

    age_options = _make_options(range(0, 121))
    sel_age = _selectbox_with_reset("Age", age_options, key="filter_age")

    sex_options = _make_options(["FEMALE", "MALE"])
    sel_sex = _selectbox_with_reset("Sex", sex_options, key="filter_sex")

    country_options = _make_options(location_facets.get("countries", []))
    sel_country = _selectbox_with_reset(
        "Country", country_options, key="filter_country"
    )
    sel_country_key = (sel_country or "").lower()

    states = (
        location_facets.get("states_by_country", {}).get(sel_country_key, [])
        if sel_country_key
        else []
    )
    state_options = _make_options(states)
    sel_state = _selectbox_with_reset(
        "State / Province",
        state_options,
        key="filter_state",
        disabled=not states,
    )
    sel_state_key = (sel_state or "").lower() if sel_state else ""

    city_key = (sel_country_key, sel_state_key)
    cities = (
        location_facets.get("cities_by_region", {}).get(city_key, [])
        if sel_country_key
        else []
    )
    city_options = _make_options(cities)
    sel_city = _selectbox_with_reset(
        "City",
        city_options,
        key="filter_city",
        disabled=not cities,
    )

    with st.form(key="search_form", clear_on_submit=False):
        st.markdown(
            """
Describe the patient's condition(s), therapies of interest, and any additional preferences in the text box below. Submit when ready to search trials.
"""
        )

        search_query = st.text_area(
            label="Search query",  # accessibility-compliant
            placeholder="e.g., metastatic colorectal cancer; on cetuximab; prefers telemedicine and minimal clinic visits.",
            height=110,
            label_visibility="collapsed",  # hides the label from the UI
        )
        submitted = st.form_submit_button(
            "🔍 Search Clinical Trials", use_container_width=True, type="primary"
        )

    st.markdown(
        """
<div class="search-tips">
  <strong>💡 Tips:</strong> Any text format is acceptable (full sentences, bullets, etc.).
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------- Results (only after click) ----------
    if submitted:
        if not (search_query or "").strip():
            st.warning("Please enter a description to search.")
            return

        settings = load_settings()
        if not settings.openai_api_key:
            st.error("OPENAI_API_KEY not set. Judge requires an API key.")
            return

        # Large centered loading indicator shown inline (not blocking)
        loader = st.empty()
        loader.markdown(
            "<div class='loading-box'><div class='spinner'></div></div>",
            unsafe_allow_html=True,
        )

        spec = parse(search_query)

        if sel_age is not None:
            spec["age"] = int(sel_age)

        if sel_sex:
            spec["sex"] = sel_sex

        location_payload = {"city": None, "state": None, "country": None}
        if sel_country:
            location_payload["country"] = sel_country
        if sel_state:
            location_payload["state"] = sel_state
        if sel_city:
            location_payload["city"] = sel_city

        if any(location_payload.values()):
            spec["location"] = location_payload

        grouped = retrieve_with_exclusions(spec, max_trials=10)
        if not grouped:
            loader.empty()
            st.info(
                "No eligible trials found in the current dataset. Please relax your search criteria."
            )
            return
        judged = judge_grouped(spec, grouped)
        loader.empty()

        st.markdown("### 📋 Results")

        if not judged:
            st.info(
                "No eligible trials found in the current dataset. Please relax your search criteria."
            )
            return

        # Build normalized objects for consistent rendering
        normalized = []
        for j in judged:
            nct = j.get("nct_id")
            ctx = grouped.get(nct, {}) if isinstance(grouped, dict) else {}

            # representative payload (prefer first inclusion, else first exclusion)
            rep = None
            for h in ctx.get("incl", []) + ctx.get("excl", []):
                rep = h.payload or {}
                if rep:
                    break

            title = (rep or {}).get("trial_title") or nct or "Untitled Trial"
            url = (rep or {}).get("url")
            elig = (j.get("eligibility") or "").upper()
            status = (rep or {}).get("overall_status") or ""
            phase = (rep or {}).get("phase") or ""
            study_type = (rep or {}).get("study_type") or ""

            normalized.append(
                {
                    "nct": nct,
                    "title": title,
                    "elig": elig,
                    "status": status,
                    "phase": phase,
                    "study_type": study_type,
                    "url": url,
                    "ctx": ctx,
                    "judge": j,
                }
            )

        # Show POSSIBLY ELIGIBLE first; collapse INELIGIBLE by default
        possible = [x for x in normalized if x["elig"] == "POSSIBLY ELIGIBLE"]
        ineligible = [x for x in normalized if x["elig"] == "INELIGIBLE"]

        if not possible:
            st.info(
                "No eligible trials found in the current dataset. Please relax your search criteria."
            )

        def render_card(item):
            j = item["judge"]
            ctx = item["ctx"]

            with st.container(border=True):
                st.markdown(
                    f"**{item['title']}** "
                    f"<span class='elig-{item['elig'].replace(' ', '_')}'>{item['elig']}</span>",
                    unsafe_allow_html=True,
                )

                meta_bits = []
                if item["status"]:
                    meta_bits.append(f"<span class='badge'>{item['status']}</span>")
                if item["phase"]:
                    meta_bits.append(f"<span class='badge'>{item['phase']}</span>")
                if item["study_type"]:
                    meta_bits.append(f"<span class='badge'>{item['study_type']}</span>")
                if meta_bits:
                    st.markdown(
                        "<div class='meta'>" + " ".join(meta_bits) + "</div>",
                        unsafe_allow_html=True,
                    )

                if item["url"]:
                    st.caption(item["url"])

                # Locations (expander, semicolon-separated list of all sites)
                locs = (rep or {}).get("locations") or []
                with st.expander("Locations", expanded=False):
                    if isinstance(locs, list) and locs:
                        items = []
                        for l in locs:
                            if not isinstance(l, dict):
                                continue
                            city = (l.get("city") or "").strip()
                            state = (l.get("state") or "").strip()
                            country = (l.get("country") or "").strip()
                            parts = [p for p in [city, state, country] if p]
                            if parts:
                                items.append(", ".join(parts))
                        if items:
                            st.write("; ".join(items))
                        else:
                            st.write("No locations listed.")
                    else:
                        st.write("No locations listed.")

                explanation = j.get("explanation") or ""
                if explanation:
                    st.write("Explanation:")
                    st.write(explanation)

                # Full criteria expander
                with st.expander("All eligibility criteria", expanded=False):
                    col_in, col_ex = st.columns(2)
                    with col_in:
                        st.caption("Inclusion")
                        for h in ctx.get("incl", []):
                            p = h.payload or {}
                            if p.get("text"):
                                st.write(f"- {p.get('text')}")
                    with col_ex:
                        st.caption("Exclusion")
                        for h in ctx.get("excl", []):
                            p = h.payload or {}
                            if p.get("text"):
                                st.write(f"- {p.get('text')}")

        # Show possibly eligible first
        for item in possible:
            render_card(item)

        # Hidden ineligible section
        if ineligible:
            with st.expander(
                f"Show ineligible trials ({len(ineligible)})", expanded=False
            ):
                for item in ineligible:
                    render_card(item)

    # ---------- Footer ----------
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; color: #6c757d; padding: 1rem 0 2rem;">
  <p>This tool is for informational purposes only. Always consult a healthcare professional.</p>
  <p><small>Data sourced from ClinicalTrials.gov (Snapshot: September 19, 2025)</small></p>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
