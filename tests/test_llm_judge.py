"""Offline tests for OpenAIJudge using MockLLMAdapter."""

from __future__ import annotations

import pytest

from htb.eval import Judge
from htb.eval.llm_judge import JUDGE_PROMPT_TEMPLATE, OpenAIJudge, _normalise_verdict
from htb.llm import MockLLMAdapter


def _mock_judge(fixed_response: str) -> OpenAIJudge:
    mock = MockLLMAdapter(default=fixed_response)
    return OpenAIJudge(llm=mock)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CORRECT", "CORRECT"),
        ("correct", "CORRECT"),
        ("  CORRECT  ", "CORRECT"),
        ("CORRECT.", "CORRECT"),
        ("Answer: CORRECT", "CORRECT"),
        ("correct answer", "CORRECT"),
        ("WRONG", "WRONG"),
        ("wrong", "WRONG"),
        ("INCORRECT", "WRONG"),
        ("NO", "WRONG"),
        ("", "WRONG"),
        ("maybe", "WRONG"),  # ambiguous -> conservative WRONG
    ],
)
def test_normalise_verdict(raw: str, expected: str) -> None:
    assert _normalise_verdict(raw) == expected


def test_openai_judge_satisfies_protocol():
    judge = _mock_judge("CORRECT")
    assert isinstance(judge, Judge)
    assert judge.name == "openai-judge"


def test_openai_judge_returns_correct_on_match():
    judge = _mock_judge("CORRECT")
    assert judge.judge("q", "apple", "apple pie") == "CORRECT"


def test_openai_judge_returns_wrong_on_mismatch():
    judge = _mock_judge("WRONG")
    assert judge.judge("q", "apple", "banana") == "WRONG"


def test_openai_judge_passes_expected_prompt_to_llm():
    mock = MockLLMAdapter(default="CORRECT")
    judge = OpenAIJudge(llm=mock)
    judge.judge("When did X happen?", "2024-06-15", "June 2024")
    assert len(mock.calls) == 1
    record = mock.calls[0]
    assert "Question: When did X happen?" in record.prompt
    assert "Gold answer: 2024-06-15" in record.prompt
    assert "Generated answer: June 2024" in record.prompt
    assert record.max_tokens == 8
    assert record.temperature == 0.0
    assert record.model == "gpt-4o"


def test_openai_judge_respects_empty_generated_answer():
    mock = MockLLMAdapter(default="WRONG")
    judge = OpenAIJudge(llm=mock)
    judge.judge("q", "gold", "")
    # Prompt must still be well-formed
    assert "Generated answer: (no answer)" in mock.calls[0].prompt


def test_openai_judge_with_openrouter_model():
    mock = MockLLMAdapter(default="CORRECT")
    judge = OpenAIJudge(llm=mock, model="openai/gpt-4o", temperature=0.2)
    judge.judge("q", "a", "b")
    record = mock.calls[0]
    assert record.model == "openai/gpt-4o"
    assert record.temperature == 0.2


def test_prompt_template_has_required_fields():
    """Template must include the three placeholders that format() requires."""
    for field in ("{question}", "{gold_answer}", "{generated_answer}"):
        assert field in JUDGE_PROMPT_TEMPLATE


def test_judge_robust_to_malformed_llm_output():
    """If the LLM returns garbage, judge falls back to WRONG (conservative)."""
    judge = _mock_judge("here is some long rambling text with no verdict token")
    assert judge.judge("q", "gold", "gen") == "WRONG"
