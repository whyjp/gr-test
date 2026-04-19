"""Offline MockLLMAdapter for smoke-testing extractor pipelines.

Usage:
    mock = MockLLMAdapter([
        (lambda p: "Extract facts and concepts" in p, canned_fact_generation_response()),
        (lambda p: "insight generation" in p,        canned_reflection_generation_response()),
    ])
    raw = mock.complete(prompt)

The adapter satisfies `htb.llm.LLMAdapter` and `gaama.adapters.interfaces.LLMAdapter`
(identical signatures). Use `make_gaama_mock()` or `make_hyper_triplet_mock()` for
ready-to-use presets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PromptMatcher = str | Callable[[str], bool]
CannedResponse = str | Callable[..., str]


@dataclass(slots=True, frozen=True)
class LLMCallRecord:
    prompt: str
    system: str | None
    max_tokens: int
    model: str | None
    temperature: float | None
    response: str
    matched_rule_index: int


@dataclass
class MockLLMAdapter:
    """Deterministic offline LLM adapter. Dispatches prompts to rules in order.

    Each rule is (matcher, responder). First matching rule wins.

    If no rule matches:
      - If `default` is set, it is returned.
      - Else, raises LookupError (so tests catch unexpected prompts).
    """

    rules: list[tuple[PromptMatcher, CannedResponse]] = field(default_factory=list)
    default: CannedResponse | None = None
    calls: list[LLMCallRecord] = field(default_factory=list)

    def add_rule(self, matcher: PromptMatcher, responder: CannedResponse) -> None:
        self.rules.append((matcher, responder))

    def _match(self, prompt: str) -> tuple[int, CannedResponse] | None:
        for i, (m, r) in enumerate(self.rules):
            if isinstance(m, str):
                if m in prompt:
                    return i, r
            else:
                if m(prompt):
                    return i, r
        return None

    @staticmethod
    def _render(responder: CannedResponse, prompt: str, kwargs: dict[str, Any]) -> str:
        if callable(responder):
            return responder(prompt, **kwargs)
        return responder

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        kwargs = {
            "system": system,
            "max_tokens": max_tokens,
            "model": model,
            "temperature": temperature,
        }
        match = self._match(prompt)
        if match is not None:
            idx, responder = match
            response = self._render(responder, prompt, kwargs)
        elif self.default is not None:
            idx = -1
            response = self._render(self.default, prompt, kwargs)
        else:
            raise LookupError(
                "MockLLMAdapter: no matching rule for prompt (first 120 chars):\n"
                f"{prompt[:120]!r}"
            )
        self.calls.append(
            LLMCallRecord(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                model=model,
                temperature=temperature,
                response=response,
                matched_rule_index=idx,
            )
        )
        return response


# ---------------------------------------------------------------------------
# Canned responses matching the known JSON schemas
# ---------------------------------------------------------------------------


def canned_fact_generation_response(
    facts: list[dict] | None = None,
    concepts: list[dict] | None = None,
) -> str:
    """Matches GAAMA `prompts/fact_generation.md` expected output.

    Default payload is a minimal 1-fact / 1-concept response so extractors
    can be end-to-end tested.
    """
    payload = {
        "facts": facts
        if facts is not None
        else [
            {
                "fact_text": "Melanie enjoys painting lake sunrises",
                "belief": 0.9,
                "source_episode_ids": ["ep-mock-1"],
                "concepts": ["painting_hobby"],
            }
        ],
        "concepts": concepts
        if concepts is not None
        else [
            {
                "concept_label": "painting_hobby",
                "episode_ids": ["ep-mock-1"],
            }
        ],
    }
    return json.dumps(payload)


def canned_reflection_generation_response(
    reflections: list[dict] | None = None,
) -> str:
    """Matches GAAMA `prompts/reflection_generation.md` expected output."""
    payload = {
        "reflections": reflections
        if reflections is not None
        else [
            {
                "reflection_text": "Melanie draws creative inspiration from natural scenery",
                "belief": 0.75,
                "source_fact_ids": ["fact-mock-1"],
            }
        ],
    }
    return json.dumps(payload)


def canned_node_set_generation_response(
    node_sets: list[dict] | None = None,
) -> str:
    """Matches the Hyper Triplet `prompts/node_set_generation.md` expected output.

    Default payload is a single node set with fact + full qualifier set.
    """
    payload = {
        "node_sets": node_sets
        if node_sets is not None
        else [
            {
                "fact": {
                    "subject": "Melanie",
                    "predicate": "painted",
                    "object": "a lake sunrise",
                },
                "source_episode_ids": ["ep-mock-1"],
                "belief": 0.9,
                "qualifiers": {
                    "location": "cabin by the lake",
                    "participants": ["Melanie"],
                    "activity_type": "painting",
                    "time_reference": "summer 2023",
                    "mood": "peaceful",
                    "topic": "artistic_creation",
                },
            }
        ],
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


# Distinctive substrings from each prompt; identified by reading the upstream
# markdown templates in external/gaama/prompts/ and our new node_set template.
_FACT_GEN_MARKER = "Extract facts and concepts from conversation episodes"
_REFLECTION_GEN_MARKER = "insight generation system"
_NODE_SET_GEN_MARKER = "node_set"


def make_gaama_mock(
    facts: list[dict] | None = None,
    concepts: list[dict] | None = None,
    reflections: list[dict] | None = None,
) -> MockLLMAdapter:
    """Preset: ready to stand in for GAAMA's LLM during offline tests.

    Responds to fact_generation and reflection_generation prompts with the
    canned JSON shapes. Raises on unrecognised prompts.
    """
    return MockLLMAdapter(
        rules=[
            (_FACT_GEN_MARKER, canned_fact_generation_response(facts, concepts)),
            (_REFLECTION_GEN_MARKER, canned_reflection_generation_response(reflections)),
        ],
    )


def make_hyper_triplet_mock(
    node_sets: list[dict] | None = None,
    reflections: list[dict] | None = None,
) -> MockLLMAdapter:
    """Preset: ready to stand in for Hyper Triplet's LLM.

    Responds to node_set_generation and reflection_generation prompts.
    """
    return MockLLMAdapter(
        rules=[
            (_NODE_SET_GEN_MARKER, canned_node_set_generation_response(node_sets)),
            (_REFLECTION_GEN_MARKER, canned_reflection_generation_response(reflections)),
        ],
    )
