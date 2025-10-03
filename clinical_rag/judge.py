"""LLM-based eligibility judgment per trial, grounded in retrieved bullets.

CLI:
  uv run python -m clinical_rag.judge --text "..." --max-trials 5
"""

import argparse
import json
import os
import re
from functools import lru_cache
from typing import Dict, List

from openai import OpenAI

from clinical_rag.query_parser import parse
from clinical_rag.retrieval import retrieve_with_exclusions
from clinical_rag.prompts.judge_prompt import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI()


def _fmt_all_trials_context(
    grouped: Dict[str, Dict[str, List]], *, max_incl: int = 40, max_excl: int = 40
) -> str:
    parts: List[str] = []
    for nct_id, ctx in grouped.items():
        parts.append(f"TRIAL: {nct_id}")
        parts.append("Inclusion bullets:")
        for h in ctx.get("incl", [])[:max_incl]:
            p = h.payload or {}
            parts.append(f"- [{p.get('chunk_id')}] {p.get('text')}")
        parts.append("Exclusion bullets:")
        for h in ctx.get("excl", [])[:max_excl]:
            p = h.payload or {}
            parts.append(f"- [{p.get('chunk_id')}] {p.get('text')}")
        parts.append("")
    return "\n".join(parts)


def judge_from_text(
    text: str,
    *,
    max_trials: int = 10,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> List[Dict]:
    """Single LLM call that judges all shortlisted trials (up to max_trials) in one response."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM judging")

    spec = parse(text)
    grouped = retrieve_with_exclusions(spec, max_trials=max_trials)

    if verbose:
        print("[judge] Parsed spec:")
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        print("[judge] Selected trials and bullet counts:")
        for nct_id, ctx in grouped.items():
            print(
                f"  - {nct_id}: incl={len(ctx.get('incl', []))}, excl={len(ctx.get('excl', []))}"
            )

    client = _get_client()
    user_content = (
        "Patient spec JSON:\n"
        + json.dumps(spec, ensure_ascii=False)
        + "\n\n"
        + _fmt_all_trials_context(grouped)
    )
    schema_note = (
        "Return ONLY a JSON array of objects, each with keys: "
        "nct_id, eligibility ('POSSIBLY ELIGIBLE'|'INELIGIBLE'), explanation (string)."
    )
    if verbose:
        print("[judge] Prompt context to LLM (system + user):")
        print("--- SYSTEM ---")
        print(SYSTEM_PROMPT + "\n" + schema_note)
        print("--- USER ---")
        print(user_content)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + schema_note},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "[]"
    if verbose:
        print("[judge] Raw LLM output:")
        print(content)

    # Be robust to markdown code fences
    def _strip_fences(s: str) -> str:
        s = s.strip()
        # Remove leading/trailing ``` blocks if present
        if s.startswith("```") and s.endswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    parsed: List[Dict] = []
    candidates_to_try = [content, _strip_fences(content)]
    # Heuristic: try extracting between first '[' and last ']'
    try:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidates_to_try.append(content[start : end + 1])
    except Exception:
        pass
    for candidate in candidates_to_try:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                parsed = obj
                break
        except Exception:
            continue
    # Normalize to policy and sanitize keys
    data = []
    for item in parsed or []:
        try:
            elig_raw = str(item.get("eligibility", "")).strip().upper()
            item["eligibility"] = (
                "INELIGIBLE" if elig_raw == "INELIGIBLE" else "POSSIBLY ELIGIBLE"
            )
            if not isinstance(item.get("explanation"), str):
                item["explanation"] = str(item.get("explanation", ""))
        except Exception:
            item["eligibility"] = "POSSIBLY ELIGIBLE"
        data.append(item)
    # Optionally, we could post-validate that nct_ids exist in grouped
    return data


def judge_grouped(
    spec: Dict,
    grouped: Dict[str, Dict[str, List]],
    *,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> List[Dict]:
    """Judge already-retrieved trials (no parsing or retrieval inside).

    Accepts a parsed spec dict and a grouped map from retrieval.
    Returns an array of trial verdicts with keys:
    nct_id, eligibility, explanation, violated_inclusion_ids, blocking_exclusion_ids.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM judging")

    if verbose:
        print("[judge_grouped] Parsed spec:")
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        print("[judge_grouped] Trials to judge and bullet counts:")
        for nct_id, ctx in grouped.items():
            print(
                f"  - {nct_id}: incl={len(ctx.get('incl', []))}, excl={len(ctx.get('excl', []))}"
            )

    client = _get_client()
    user_content = (
        "Patient spec JSON:\n"
        + json.dumps(spec, ensure_ascii=False)
        + "\n\n"
        + _fmt_all_trials_context(grouped)
    )
    schema_note = (
        "Return ONLY a JSON array of objects, each with keys: "
        "nct_id, eligibility ('POSSIBLY ELIGIBLE'|'INELIGIBLE'), explanation (string)."
    )
    if verbose:
        print("[judge_grouped] Prompt context to LLM (system + user):")
        print("--- SYSTEM ---")
        print(SYSTEM_PROMPT + "\n" + schema_note)
        print("--- USER ---")
        print(user_content)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + schema_note},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "[]"
    if verbose:
        print("[judge_grouped] Raw LLM output:")
        print(content)

    def _strip_fences(s: str) -> str:
        s = s.strip()
        # Remove leading/trailing ``` blocks if present
        if s.startswith("```") and s.endswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    parsed: List[Dict] = []
    for candidate in [content, _strip_fences(content)]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                parsed = obj
                break
        except Exception:
            continue

    # Normalize to policy and sanitize keys
    data = []
    for item in parsed or []:
        try:
            elig_raw = str(item.get("eligibility", "")).strip().upper()
            item["eligibility"] = (
                "INELIGIBLE" if elig_raw == "INELIGIBLE" else "POSSIBLY ELIGIBLE"
            )
            if not isinstance(item.get("explanation"), str):
                item["explanation"] = str(item.get("explanation", ""))
        except Exception:
            item["eligibility"] = "POSSIBLY ELIGIBLE"
        data.append(item)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge trial eligibility using retrieved bullets"
    )
    parser.add_argument("--text", type=str, required=True, help="Patient description")
    parser.add_argument(
        "--max-trials", type=int, default=5, help="Max unique trials to judge"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print parsed spec and prompt context for debugging",
    )
    args = parser.parse_args()

    results = judge_from_text(
        args.text, max_trials=args.max_trials, verbose=args.verbose
    )
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
