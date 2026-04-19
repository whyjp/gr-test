"""Protocols shared by systems and evaluation code.

A Pipeline is any object with ingest/retrieve/answer/reset — works for HippoRAG2,
GAAMA, and Hyper Triplet. A Judge scores a single generated answer against a gold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from htb.data import Conversation

Judgment = Literal["CORRECT", "WRONG"]


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    context: str
    word_count: int
    latency_ms: float = 0.0
    evidence_dia_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class AnswerResult:
    text: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class Pipeline(Protocol):
    """Minimum surface every system under test must implement."""

    name: str

    def reset(self) -> None: ...

    def ingest(self, conversation: Conversation) -> None: ...

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult: ...

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult: ...


@runtime_checkable
class Judge(Protocol):
    """Scores a single (question, gold, generated) triple."""

    name: str

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> Judgment: ...
