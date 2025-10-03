"""
Generate a gold-standard retrieval dataset by sampling trials from the latest
snapshot and asking an LLM to write a realistic patient query that should
retrieve that specific trial (based on title, conditions, interventions, age,
sex, and location constraints).

Usage:
  uv run python -m eval.generate_gold \
    --num 1000 \
    --snapshot LATEST \
    --model gpt-4o-mini \
    --output eval/gold_retrieval.jsonl

Notes:
- Requires OPENAI_API_KEY for both query generation and parsing.
- Each generated query is parsed via clinical_rag.query_parser.parse and
  stored under the query_spec key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import argparse
import json
import os
import random
import time

from clinical_rag.query_parser import parse as parse_query


def _find_latest_snapshot(base: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    if not base.exists():
        raise FileNotFoundError(f"Snapshots base not found: {base}")
    for d in base.iterdir():
        if d.is_dir() and (d / "trials.jsonl").exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        raise FileNotFoundError("No snapshot with trials.jsonl found")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_trials(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _choose_age(
    min_age: Optional[int], max_age: Optional[int], *, rng: random.Random
) -> int:
    if isinstance(min_age, int) and isinstance(max_age, int) and max_age >= min_age:
        if max_age == min_age:
            return min_age
        return rng.randint(min_age, max_age)
    if isinstance(min_age, int):
        return min_age + min(20, max(0, rng.randint(0, 10)))
    if isinstance(max_age, int):
        base = max(0, max_age - 10)
        return rng.randint(base, max_age)
    return 40


def _choose_sex(trial_sex: Optional[str], *, rng: random.Random) -> Optional[str]:
    s = (trial_sex or "").upper()
    if s in ("MALE", "FEMALE"):
        return s.lower()
    if s == "ALL":
        return rng.choice(["male", "female"])  # either is fine
    return None


def _choose_location(
    locs: List[Dict], *, rng: random.Random
) -> Dict[str, Optional[str]]:
    if not isinstance(locs, list) or not locs:
        return {"city": None, "state": None, "country": None}
    cand = rng.choice([l for l in locs if isinstance(l, dict)] or [{}])
    return {
        "city": (cand.get("city") or None),
        "state": (cand.get("state") or None),
        "country": (cand.get("country") or None),
    }


def _build_llm_prompt(
    trial: Dict, age: int, sex_word: Optional[str], loc: Dict[str, Optional[str]]
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    title = (trial.get("trial_title") or "").strip()
    conds = ", ".join(trial.get("conditions") or [])
    itvs = ", ".join(trial.get("interventions") or [])
    phase = trial.get("phase") or ""
    stype = trial.get("study_type") or ""
    city = (loc.get("city") or "").strip()
    state = (loc.get("state") or "").strip()
    country = (loc.get("country") or "").strip()

    sys = (
        "You are generating concise patient descriptions to evaluate a clinical trials retrieval system.\n"
        "Return ONE short, natural English sentence (no bullets) describing a hypothetical patient whose case\n"
        "should retrieve the target trial. Include key signals (age, sex if applicable, primary condition(s),\n"
        "notable intervention(s)/drug/device terms, and a location token). Do NOT mention NCT IDs or quote\n"
        "the exact trial title; paraphrase naturally."
    )
    user = (
        "Target trial summary:\n"
        f"- Title: {title}\n"
        f"- Conditions: {conds}\n"
        f"- Interventions: {itvs}\n"
        f"- Phase: {phase}\n"
        f"- Study type: {stype}\n"
        f"- Age example: {age} years\n"
        f"- Sex example: {sex_word or 'any'}\n"
        f"- Location example tokens: {[t for t in [city, state, country] if t]}\n\n"
        "Write the single-sentence patient description now."
    )
    return sys, user


def _synth_query(
    trial: Dict, age: int, sex_word: Optional[str], loc: Dict[str, Optional[str]]
) -> str:
    parts = []
    parts.append(f"{age}-year-old")
    if sex_word in ("male", "female"):
        parts.append(sex_word)
    conds = trial.get("conditions") or []
    itvs = trial.get("interventions") or []
    if conds:
        parts.append("with " + ", ".join(conds[:2]))
    if itvs:
        parts.append("considering/receiving " + ", ".join(itvs[:2]))
    city = (loc.get("city") or "").strip()
    state = (loc.get("state") or "").strip()
    country = (loc.get("country") or "").strip()
    place = ", ".join([p for p in [city, state, country] if p])
    if place:
        parts.append(f"in {place}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gold-standard retrieval dataset from snapshot trials"
    )
    parser.add_argument(
        "--snapshot", type=str, default="LATEST", help="Snapshot dir or LATEST"
    )
    parser.add_argument(
        "--num", type=int, default=1000, help="Number of trials to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model id for generation and parsing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/gold_retrieval.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for query parsing")

    rng = random.Random(args.seed)

    base = Path("data") / "snapshots"
    snap_dir = (
        Path(args.snapshot)
        if args.snapshot != "LATEST"
        else _find_latest_snapshot(base)
    )
    trials_path = snap_dir / "trials.jsonl"
    if not trials_path.exists():
        raise FileNotFoundError(f"trials.jsonl not found in snapshot: {snap_dir}")

    trials = list(_load_trials(trials_path))
    if not trials:
        raise RuntimeError("No trials found in snapshot")

    rng.shuffle(trials)
    sample = trials[: max(0, min(args.num, len(trials)))]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI  # lazy import

    client = OpenAI()

    written = 0
    total = len(sample)
    start_ts = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, t in enumerate(sample, start=1):
            # Prepare guidance values
            min_age = t.get("min_age_years") or t.get("min_age")
            max_age = t.get("max_age_years") or t.get("max_age")
            age = _choose_age(
                min_age if isinstance(min_age, int) else None,
                max_age if isinstance(max_age, int) else None,
                rng=rng,
            )
            sex_word = _choose_sex(t.get("sex") or t.get("gender"), rng=rng)
            loc = _choose_location(t.get("locations") or [], rng=rng)

            # Build query via LLM with synth fallback on error
            sys, user = _build_llm_prompt(t, age, sex_word, loc)
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
                query = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                query = _synth_query(t, age, sex_word, loc)

            try:
                query_spec = parse_query(query, llm_model=args.model)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse generated query for {t.get('nct_id')}: {e}"
                ) from e

            row = {
                "query_spec": query_spec,
                "query": query,
                "nct_id": t.get("nct_id"),
                "snapshot": snap_dir.name,
                "title": t.get("trial_title"),
                "sex": sex_word,
                "age": age,
                "location": loc,
                "conditions": t.get("conditions") or [],
                "interventions": t.get("interventions") or [],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            # lightweight progress indicator
            if idx % 10 == 0 or idx == total:
                elapsed = max(1e-6, time.time() - start_ts)
                rate = idx / elapsed
                pct = (idx / total) * 100.0 if total else 100.0
                eta = (total - idx) / rate if rate > 0 else 0.0
                print(
                    f"\rProgress: {idx}/{total} ({pct:5.1f}%) | {rate:4.1f} qps | ETA {eta:5.1f}s",
                    end="",
                    flush=True,
                )
                # polite pacing for API
                time.sleep(0.05)

    # finalize progress line
    print()

    print(f"Gold dataset written: {out_path} (rows={written})")


if __name__ == "__main__":
    main()
