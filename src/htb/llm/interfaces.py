"""LLM adapter protocol.

Matches GAAMA's `gaama.adapters.interfaces.LLMAdapter` signature so the same
adapter can stand in for real API calls in both Hyper Triplet code and GAAMA
forked code during offline testing.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class CompletionKwargs(TypedDict, total=False):
    system: str | None
    max_tokens: int
    model: str | None
    temperature: float | None


@runtime_checkable
class LLMAdapter(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str: ...
