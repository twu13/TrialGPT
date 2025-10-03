"""Orchestrate the end-to-end evaluation workflow.

Steps:
1. Generate gold retrieval queries (`eval.generate_gold`).
2. Execute the retrieval evaluation notebook.
3. Generate judge evaluation data (`eval.generate_judge`).
4. Execute the judge evaluation notebook.

Usage:
  uv run python -m eval.main [options]
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv
import nbformat
from nbclient import NotebookClient

from clinical_rag.config import load_settings


def _run_cli_module(module: str, argv: list[str]) -> None:
    """Invoke a module's CLI `main()` by temporarily swapping `sys.argv`."""

    old_argv = sys.argv
    sys.argv = [module.split(".")[-1], *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def _execute_notebook(path: Path, *, timeout: int, save_output: bool) -> None:
    """Run a notebook in-process via nbclient and optionally persist outputs."""

    nb_path = path.resolve()
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb, kernel_name="python3", timeout=timeout, allow_errors=False
    )
    client.execute()

    if save_output:
        executed_path = nb_path.with_suffix(".executed.ipynb")
        with executed_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation workflow end-to-end"
    )
    parser.add_argument(
        "--snapshot", default="LATEST", help="Snapshot directory or LATEST"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="Number of gold queries to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for gold query generation"
    )
    parser.add_argument(
        "--query-model",
        default="gpt-4o-mini",
        help="OpenAI model for gold query generation",
    )
    parser.add_argument(
        "--gold-output",
        default="eval/gold_retrieval.jsonl",
        help="Path for generated gold queries",
    )
    parser.add_argument(
        "--retrieval-notebook",
        default="eval/retrieval_eval.ipynb",
        help="Notebook to execute after gold generation",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini", help="OpenAI model for judge evaluation"
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Trials to retrieve per query when generating judge data",
    )
    parser.add_argument(
        "--judge-output",
        default="eval/gold_judge.jsonl",
        help="Path for judge evaluation dataset",
    )
    parser.add_argument(
        "--judge-notebook",
        default="eval/judge_eval.ipynb",
        help="Notebook to execute after judge data generation",
    )
    parser.add_argument(
        "--notebook-timeout",
        type=int,
        default=900,
        help="Execution timeout (seconds) for notebooks",
    )
    parser.add_argument(
        "--save-notebook-outputs",
        action="store_true",
        help="Persist executed notebook copies alongside originals",
    )

    args = parser.parse_args(argv)

    # Load environment variables for downstream modules (e.g., OpenAI/Qdrant creds)
    load_dotenv()

    settings = load_settings()
    print(
        "Running evaluation suite with collection '",
        settings.collection_name,
        "' and snapshot",
        args.snapshot,
    )

    # Step 1: generate gold retrieval dataset
    gold_args = [
        "--snapshot",
        args.snapshot,
        "--num",
        str(args.num_queries),
        "--seed",
        str(args.seed),
        "--model",
        args.query_model,
        "--output",
        args.gold_output,
    ]
    print("[1/4] Generating gold retrieval dataset...")
    _run_cli_module("eval.generate_gold", gold_args)

    # Step 2: execute retrieval evaluation notebook
    print("[2/4] Executing retrieval evaluation notebook...")
    _execute_notebook(
        Path(args.retrieval_notebook),
        timeout=args.notebook_timeout,
        save_output=args.save_notebook_outputs,
    )

    # Step 3: generate judge evaluation dataset
    judge_args = [
        "--input",
        args.gold_output,
        "--output",
        args.judge_output,
        "--k",
        str(args.retrieval_k),
        "--model",
        args.judge_model,
    ]
    print("[3/4] Generating judge evaluation dataset...")
    _run_cli_module("eval.generate_judge", judge_args)

    # Step 4: execute judge evaluation notebook
    print("[4/4] Executing judge evaluation notebook...")
    _execute_notebook(
        Path(args.judge_notebook),
        timeout=args.notebook_timeout,
        save_output=args.save_notebook_outputs,
    )

    print("Evaluation workflow completed successfully.")


if __name__ == "__main__":
    main()
