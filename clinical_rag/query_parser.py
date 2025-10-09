"""Query Rewrite and Parsing Module (LLM-only).

Uses OpenAI Chat Completions with an externalized system prompt to parse
patient descriptions into a normalized JSON spec.

Usage (programmatic):
    from clinical_rag.query_parser import parse
    spec = parse("65 y/o male with diabetes, on metformin")

CLI (ad-hoc):
    uv run python -m clinical_rag.query_parser --text "..." [--model MODEL_ID]
"""

from collections import OrderedDict
from typing import Dict, Optional, Union, List
import argparse
import json
import os

from pydantic import BaseModel, Field
from openai import OpenAI

from clinical_rag.prompts.query_parser_prompt import SYSTEM_PROMPT
from clinical_rag.config import load_settings


_SETTINGS = load_settings()


class QuerySpec(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    extra_terms: list[str] = Field(default_factory=list)

    def to_dict(self) -> Dict:
        return self.model_dump()


def parse(text: str, *, llm_model: Optional[str] = None) -> Dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM parsing")

    client = OpenAI()
    model_to_use = llm_model or _SETTINGS.llm_model_name
    resp = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"

    def _coerce_json(raw: str) -> Dict:
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                "Query parser returned non-JSON content. Received: " + stripped
            )

    data = _coerce_json(content)
    for key in ("conditions", "medications", "extra_terms"):
        value = data.get(key)
        if value is None:
            data[key] = []
    spec = QuerySpec(**data).to_dict()
    return {**spec, "age": None, "sex": None, "location": None}


def build_query_components(spec: Union[Dict, QuerySpec]) -> "OrderedDict[str, str]":
    """Return ordered mapping of semantic query components."""
    if not isinstance(spec, QuerySpec):
        spec = QuerySpec(**spec)

    parts: "OrderedDict[str, str]" = OrderedDict()
    if spec.conditions:
        parts["conditions"] = "conditions:" + ", ".join(spec.conditions)
    if spec.medications:
        parts["medications"] = "meds:" + ", ".join(spec.medications)
    if spec.extra_terms:
        parts["extra_terms"] = "context:" + ", ".join(spec.extra_terms)
    return parts


def build_query_text(spec: Union[Dict, QuerySpec]) -> str:
    """Construct a deterministic semantic query string from a parsed spec."""
    components = build_query_components(spec)
    return " ".join(components.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a patient description into structured query JSON (LLM-only)"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Patient free-text input"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_SETTINGS.llm_model_name,
        help="OpenAI chat model id (defaults to LLM_MODEL_NAME or project default)",
    )
    args = parser.parse_args()

    spec = parse(args.text, llm_model=args.model)
    print(json.dumps(spec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
