"""Query Rewrite and Parsing Module (LLM-only).

Uses OpenAI Chat Completions with an externalized system prompt to parse
patient descriptions into a normalized JSON spec.

Usage (programmatic):
    from clinical_rag.query_parser import parse
    spec = parse("65 y/o male with diabetes, on metformin")

CLI (ad-hoc):
    uv run python -m clinical_rag.query_parser --text "..." [--model gpt-4o-mini]
"""

from typing import Dict, Optional, Union, List
import argparse
import json
import os

from pydantic import BaseModel, Field
from openai import OpenAI

from clinical_rag.prompts.query_parser_prompt import SYSTEM_PROMPT


class LocationPref(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class QuerySpec(BaseModel):
    age: int | None = Field(default=None)
    sex: str | None = Field(default=None)
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    extra_terms: list[str] = Field(default_factory=list)
    location: Optional[LocationPref] = None

    def to_dict(self) -> Dict:
        return self.model_dump()


def parse(text: str, *, llm_model: str = "gpt-4o-mini") -> Dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM parsing")

    client = OpenAI()
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    return QuerySpec(**data).to_dict()


def build_query_text(spec: Union[Dict, QuerySpec]) -> str:
    """Construct a deterministic semantic query string from a parsed spec.

    Rules:
    - Field order: age, sex, conditions, meds, comorbidities, location
    - Lowercase sex and location tokens, keep numbers as-is
    - Join multi-values with ", " in a stable order
    """
    if not isinstance(spec, QuerySpec):
        spec = QuerySpec(**spec)

    parts: List[str] = []
    if spec.age is not None:
        parts.append(f"age:{spec.age}")
    if spec.sex:
        parts.append(f"sex:{spec.sex.lower()}")
    if spec.conditions:
        parts.append("conditions:" + ", ".join(spec.conditions))
    if spec.medications:
        parts.append("meds:" + ", ".join(spec.medications))
    if spec.comorbidities:
        parts.append("comorbidities:" + ", ".join(spec.comorbidities))
    if spec.extra_terms:
        parts.append("context:" + ", ".join(spec.extra_terms))

    if spec.location:
        loc_tokens: List[str] = []
        if spec.location.city:
            loc_tokens.append(spec.location.city.lower())
        if spec.location.state:
            loc_tokens.append(spec.location.state.lower())
        if spec.location.country:
            loc_tokens.append(spec.location.country.lower())
        if loc_tokens:
            parts.append("location:" + " ".join(loc_tokens))
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a patient description into structured query JSON (LLM-only)"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Patient free-text input"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI chat model id"
    )
    args = parser.parse_args()

    spec = parse(args.text, llm_model=args.model)
    print(json.dumps(spec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
