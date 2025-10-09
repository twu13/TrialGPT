import json
import os

import pytest


def test_parse_llm_mock_basic(monkeypatch):
    # Import inside test to ensure monkeypatch targets the loaded module
    import clinical_rag.query_parser as qp

    # Ensure the code path uses LLM branch
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    expected = {
        "age": None,
        "sex": None,
        "conditions": ["metastatic breast cancer"],
        "medications": ["letrozole", "palbociclib"],
        "extra_terms": ["oral therapy", "minimal clinic visits"],
        "location": None,
    }

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, model, messages, temperature):
            return _Resp(json.dumps(expected))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _OpenAI:
        def __init__(self):
            self.chat = _Chat()

    # Patch the OpenAI client used in the module
    monkeypatch.setattr(qp, "OpenAI", _OpenAI)

    text = (
        "42-year-old female with metastatic breast cancer; currently taking letrozole "
        "and palbociclib; prefers oral therapy and minimal clinic visits; in New York City."
    )
    spec = qp.parse(text)

    assert spec == expected


def test_build_query_text_deterministic():
    from clinical_rag.query_parser import build_query_text

    spec = {
        "age": None,
        "sex": None,
        "conditions": ["metastatic breast cancer"],
        "medications": ["letrozole", "palbociclib"],
        "extra_terms": ["oral therapy", "telemedicine"],
        "location": None,
    }

    q = build_query_text(spec)
    assert "sex:" not in q
    assert "conditions:metastatic breast cancer" in q
    assert "meds:letrozole, palbociclib" in q
    assert "context:oral therapy, telemedicine" in q
