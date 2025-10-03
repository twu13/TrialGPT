"""Generate judge evaluation dataset by running retrieval and eligibility judgments.

Usage:
  uv run python -m eval.generate_judge \
    --input eval/gold_retrieval.jsonl \
    --output eval/gold_judge.jsonl \
    --k 10 \
    --model gpt-4o-mini

Each input row must include a `query_spec` field (dict or JSON string). For each
row the script:
  1. Reuses the parsed spec to retrieve top-K trials via
     `clinical_rag.retrieval.retrieve_with_exclusions`.
  2. Runs the grouped trials through `clinical_rag.judge.judge_grouped`, which
     performs a single LLM call to classify eligibility for all trials.
  3. Writes a JSONL record containing the original row fields, serialized
     retrieval context, and judge outputs.

The resulting dataset can be used to evaluate judge accuracy offline without
re-running retrieval or additional judgement calls.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from clinical_rag.retrieval import retrieve_with_exclusions
from clinical_rag.judge import judge_grouped


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_spec(row: Dict, *, row_index: int) -> Dict:
    spec = row.get("query_spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise ValueError(f"Row {row_index} has invalid query_spec JSON") from e
    if not isinstance(spec, dict):
        raise ValueError(f"Row {row_index} missing valid query_spec")
    return spec


def _serialize_bullets(bullets: Iterable, *, limit: Optional[int]) -> List[Dict]:
    serialized: List[Dict] = []
    bullet_list = bullets if isinstance(bullets, list) else list(bullets)
    if limit is not None and limit > 0:
        bullet_list = bullet_list[:limit]
    for b in bullet_list:
        payload = getattr(b, "payload", None) or {}
        serialized.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
            }
        )
    return serialized


def _representative_payload(ctx: Dict[str, List]) -> Dict:
    for h in ctx.get("incl", []) + ctx.get("excl", []):
        payload = getattr(h, "payload", None)
        if payload:
            return payload
    return {}


def _serialize_grouped(
    grouped: Dict[str, Dict[str, List]],
    *,
    max_incl: Optional[int],
    max_excl: Optional[int],
) -> List[Dict]:
    out: List[Dict] = []
    for nct_id, ctx in grouped.items():
        payload = _representative_payload(ctx)
        trial_info = {
            "nct_id": nct_id,
            "trial_title": payload.get("trial_title"),
            "overall_status": payload.get("overall_status"),
            "gender": payload.get("gender"),
            "min_age": payload.get("min_age"),
            "max_age": payload.get("max_age"),
            "conditions": payload.get("conditions") or [],
            "interventions": payload.get("interventions") or [],
            "locations": payload.get("locations") or [],
            "phase": payload.get("phase"),
            "study_type": payload.get("study_type"),
            "url": payload.get("url"),
            "inclusion_bullets": _serialize_bullets(
                ctx.get("incl", []), limit=max_incl
            ),
            "exclusion_bullets": _serialize_bullets(
                ctx.get("excl", []), limit=max_excl
            ),
        }
        out.append(trial_info)
    return out


def _rank_of(target: Optional[str], ranking: List[str]) -> Optional[int]:
    if not target:
        return None
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate judge evaluation dataset from gold retrieval queries"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eval/gold_retrieval.jsonl",
        help="Path to gold retrieval dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/gold_judge.jsonl",
        help="Output JSONL path for judge dataset",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of trials to retrieve per query"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model id for judge workflow",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of rows to process (0 = all)",
    )
    parser.add_argument(
        "--max-inclusion",
        type=int,
        default=40,
        help="Max inclusion bullets to store per trial (0 = all)",
    )
    parser.add_argument(
        "--max-exclusion",
        type=int,
        default=40,
        help="Max exclusion bullets to store per trial (0 = all)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Progress log frequency in rows"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for judge evaluation")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    processed = 0
    for _ in _iter_jsonl(input_path):
        total_rows += 1
        if args.limit and args.limit > 0 and total_rows >= args.limit:
            break
    if args.limit and args.limit > 0:
        total_rows = min(total_rows, args.limit)

    start_ts = time.time()
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if args.limit and processed >= args.limit:
                break

            spec = _load_spec(row, row_index=idx)
            grouped = retrieve_with_exclusions(spec, max_trials=args.k)
            ranking = list(grouped.keys())
            judge_output = judge_grouped(spec, grouped, model=args.model)

            serialized = {
                "query": row.get("query"),
                "query_spec": spec,
                "target_nct_id": row.get("nct_id"),
                "target_rank": _rank_of(row.get("nct_id"), ranking),
                "snapshot": row.get("snapshot"),
                "retrieval_k": args.k,
                "retrieved_nct_ids": ranking,
                "retrieved_trials": _serialize_grouped(
                    grouped,
                    max_incl=None if args.max_inclusion <= 0 else args.max_inclusion,
                    max_excl=None if args.max_exclusion <= 0 else args.max_exclusion,
                ),
                "judge_model": args.model,
                "judge_results": judge_output,
            }
            fout.write(json.dumps(serialized, ensure_ascii=False) + "\n")

            processed += 1
            if args.log_interval > 0 and processed % args.log_interval == 0:
                elapsed = max(1e-6, time.time() - start_ts)
                rate = processed / elapsed
                eta = (total_rows - processed) / rate if total_rows else 0.0
                pct = (processed / total_rows * 100.0) if total_rows else 100.0
                print(
                    f"Processed {processed}/{total_rows or '?'} ({pct:5.1f}%)"
                    f" | {rate:4.2f} rows/s | ETA {eta:6.1f}s",
                    end="\r",
                    flush=True,
                )

    elapsed = time.time() - start_ts
    print()
    print(
        f"Judge dataset written: {output_path} (rows={processed}, elapsed={elapsed:0.1f}s,"
        f" avg={(processed / elapsed) if elapsed else 0.0:0.2f} rows/s)"
    )


if __name__ == "__main__":
    main()
