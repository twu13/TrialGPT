import json
import streamlit as st

from clinical_rag.config import load_settings
from clinical_rag.query_parser import parse
from clinical_rag.retrieval import retrieve_with_exclusions
from clinical_rag.judge import judge_grouped


def main() -> None:
    # ---------- Page config ----------
    st.set_page_config(
        page_title="Clinical Trials Finder",
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

    with st.form(key="search_form", clear_on_submit=False):
        st.markdown(
            """
Please provide the following details (more information → better results):

- Age and sex (e.g., 30, male)
- Conditions and comorbidities (e.g., sickle cell disease; hypertension)
- Current medications (e.g., hydroxyurea)
- Location: city, state, country; optional radius (e.g., within 50 km)
- Preferences (extra terms): short phrases like "oral therapy", "telemedicine",
  "double-blind", "minimal clinic visits"
"""
        )
        search_query = st.text_area(
            label="Search query",  # accessibility-compliant
            placeholder="e.g., 30-year-old male with sickle cell disease; on hydroxyurea; in Boston, Massachusetts, United States within 50 km; prefers minimal clinic visits and telemedicine.",
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
        grouped = retrieve_with_exclusions(spec, max_trials=10)
        if not grouped:
            loader.empty()
            st.info(
                "No trials matched your criteria. Try broadening your description (e.g., expand distance or simplify meds)."
            )
            return
        judged = judge_grouped(spec, grouped)
        loader.empty()

        st.markdown("### 📋 Results")

        if not judged:
            st.info(
                "No trials matched your criteria. Try broadening your description (e.g., expand distance or simplify meds)."
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
  <p><small>Data sourced from ClinicalTrials.gov</small></p>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
