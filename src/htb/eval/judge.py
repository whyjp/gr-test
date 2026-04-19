"""Judges. Only offline-safe implementations live here; LLM judge is a stub."""

from __future__ import annotations

import re
from dataclasses import dataclass

from htb.eval.interfaces import Judgment

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


@dataclass(slots=True)
class KeywordMockJudge:
    """Deterministic offline judge: CORRECT iff the generated answer contains
    any of the gold answer's content tokens (ignoring stopwords).

    Intended ONLY for offline testing of the eval framework. Real benchmark
    runs must use an LLM judge.
    """

    name: str = "keyword-mock"
    min_overlap: int = 1
    stopwords: frozenset[str] = frozenset(
        {
            "a", "an", "and", "or", "the", "of", "to", "for", "in", "on",
            "at", "by", "with", "is", "are", "was", "were", "be", "been",
            "being", "it", "this", "that", "these", "those", "as", "from",
        }
    )

    def _content_tokens(self, s: str) -> set[str]:
        return {t for t in _tokens(s) if t not in self.stopwords}

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> Judgment:
        gold = self._content_tokens(gold_answer)
        gen = self._content_tokens(generated_answer)
        if not gold:
            return "CORRECT" if not gen else "WRONG"
        overlap = gold & gen
        return "CORRECT" if len(overlap) >= self.min_overlap else "WRONG"


@dataclass(slots=True)
class LLMJudgeStub:
    """Placeholder. Replaced in Phase 2+ with an OpenAI-based judge.
    Raises on use so it can't be silently called in offline tests.
    """

    name: str = "llm-judge-stub"

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> Judgment:
        raise NotImplementedError(
            "LLM judge not wired yet. Use KeywordMockJudge in offline tests "
            "or implement htb.eval.judge.OpenAILLMJudge once API access is available."
        )
