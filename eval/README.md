# Evaluation Overview

This directory contains the offline evaluation workflow for TrialGPT. The goal is to measure how well the retrieval pipeline surfaces the gold trial for each synthetic patient profile and how often the eligibility judge agrees with a secondary LLM audit.

## Datasets & Workflow

1. **Retrieval dataset generation (`generate_gold.py`)** – starting from the latest `data/snapshots/<timestamp>`, an LLM writes realistic patient descriptions for sampled trials and re-parses them into JSON specs. The resulting `gold_retrieval.jsonl` stores, per query, the originating trial ID and the parsed spec.
2. **Retrieval evaluation (`retrieval_eval.ipynb`)** – uses the production retrieval stack (`clinical_rag.retrieval.retrieve_with_exclusions`) against the current Qdrant collection and compares the gold trial’s rank against the retrieved list.
3. **Judge dataset generation (`generate_judge.py`)** – for each query, the application’s judge (`clinical_rag.judge.judge_grouped`) produces POSSIBLY_ELIGIBLE / INELIGIBLE labels alongside explanations. This creates `gold_judge.jsonl`.
4. **Judge evaluation (`judge_eval.ipynb`)** – an external LLM reviews the judged output for correctness, classifying each decision as FULLY_CORRECT, PARTIALLY_CORRECT, or INCORRECT.

All steps are orchestrated by `make eval`, which performs the generation, notebook execution (via `nbclient`), and summarises the tables below.

## Retrieval Metrics (K = 10)

| Metric | Value | Notes |
| --- | --- | --- |
| Hit@10 | 0.8310 | Fraction of queries where the gold trial appears in the top-10 |
| MRR@10 | 0.7804 | Mean reciprocal rank of the gold trial |
| Evaluated queries | 1,000 | Synthetic patient descriptions |
| Hits | 831 | Queries with gold trial retrieved within top-10 |

_Source: `eval/retrieval_eval.ipynb` (executed on snapshot 2025-09-19)._

## Judge (Eligibility) Audit

| Metric | Value |
| --- | --- |
| Fully correct | 0.0350 |
| Partially correct | 0.9310 |
| Incorrect | 0.0330 |
| Evaluated queries | 1,000 |

Interpretation:
- **Fully correct** – judge label matched the audit with no issues.
- **Partially correct** – mixed outcome (some correct trials, some missing context).
- **Incorrect** – judge misclassified the highlighted trial or failed to flag a clear exclusion.

_Source: `eval/judge_eval.ipynb` auditing the same 1,000-query snapshot._

## Reproducing

1. Clear the Qdrant database using `make down-clean` and run `make load-archive-qd` to load this exact snapshot.
3. Run `make eval` to regenerate the gold datasets and notebooks.
4. Open the notebooks in Jupyter/Quarto to inspect the results.
