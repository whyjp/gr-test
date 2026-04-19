"""LLMNodeSetExtractor — one LLM call producing hyper-relational node_sets.

Replaces GAAMA's `LLMFactExtractor.extract_facts()` (fact+concept) with a
single call that emits typed qualifier pairs bound atomically to each fact.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from htb.llm import LLMAdapter
from systems.hyper_triplet.types import NodeSet

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "node_set_generation.md"

_JSON_FENCE_RES = (
    re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL),
    re.compile(r"^```\s*\n?(.*?)\n?```\s*$", re.DOTALL),
)

_MAX_JSON_RETRIES = 2


def _strip_json_block(text: str) -> str:
    text = text.strip()
    for pattern in _JSON_FENCE_RES:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return text


def render_prompt(template: str, substitutions: dict[str, str]) -> str:
    result = template
    for key, value in substitutions.items():
        result = result.replace("{{" + key + "}}", value)
    return result


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    id: str
    text: str
    session_date: str = ""


@dataclass(frozen=True, slots=True)
class FactRef:
    id: str
    text: str


def _format_episodes(eps: Sequence[EpisodeRef]) -> str:
    if not eps:
        return "(none)"
    lines = []
    for ep in eps:
        if ep.session_date:
            lines.append(f"[{ep.id}] [{ep.session_date}] {ep.text}")
        else:
            lines.append(f"[{ep.id}] {ep.text}")
    return "\n".join(lines)


def _format_facts(facts: Sequence[FactRef]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(f"[{f.id}] {f.text}" for f in facts)


def _format_qualifiers_by_type(qmap: dict[str, Sequence[str]]) -> str:
    if not qmap or not any(values for values in qmap.values()):
        return "(none)"
    lines = []
    for qtype, values in qmap.items():
        clean = [v for v in values if v]
        if not clean:
            continue
        lines.append(f"{qtype}:")
        for v in clean:
            lines.append(f"  - {v}")
    return "\n".join(lines) if lines else "(none)"


@dataclass
class LLMNodeSetExtractor:
    llm: LLMAdapter
    max_tokens: int = 4000
    prompt_template: str | None = None  # defaults to file at PROMPT_PATH

    def __post_init__(self) -> None:
        if self.prompt_template is None:
            self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def extract_node_sets(
        self,
        new_episodes: Sequence[EpisodeRef],
        related_episodes: Sequence[EpisodeRef] = (),
        existing_facts: Sequence[FactRef] = (),
        existing_qualifiers_by_type: dict[str, Sequence[str]] | None = None,
    ) -> list[NodeSet]:
        prompt = render_prompt(
            self.prompt_template or "",
            {
                "new_episodes": _format_episodes(new_episodes),
                "related_episodes": _format_episodes(related_episodes),
                "existing_facts": _format_facts(existing_facts),
                "existing_qualifiers_by_type": _format_qualifiers_by_type(
                    existing_qualifiers_by_type or {}
                ),
            },
        )
        raw = self.llm.complete(prompt, max_tokens=self.max_tokens)
        data = self._parse_response(raw)
        raw_sets = data.get("node_sets") or []
        return self._validate_node_sets(raw_sets)

    def _parse_response(self, raw: str) -> dict:
        stripped = _strip_json_block(raw or "")
        if not stripped:
            return {"node_sets": []}
        parse_error: str = "parse failed"
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"node_sets": data}
        except json.JSONDecodeError as exc:
            logger.warning("Initial JSON parse failed: %s", exc)
            parse_error = str(exc)
        return self._retry_parse(raw or "", parse_error)

    def _retry_parse(self, original: str, error: str) -> dict:
        last_error = error
        last_raw = original
        for _attempt in range(1, _MAX_JSON_RETRIES + 1):
            retry_prompt = (
                "Your previous response could not be parsed as valid JSON.\n\n"
                f"Error: {last_error}\n\n"
                f"Previous output (first 1000 chars):\n{last_raw[:1000]}\n\n"
                'Return ONLY valid JSON: {"node_sets": [...]}. No markdown fences.'
            )
            try:
                raw = self.llm.complete(retry_prompt, max_tokens=self.max_tokens)
            except Exception as exc:
                last_error = f"LLM call failed: {exc}"
                continue
            stripped = _strip_json_block(raw or "")
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    return data
                if isinstance(data, list):
                    return {"node_sets": data}
            except json.JSONDecodeError as exc:
                last_error = str(exc)
                last_raw = raw or ""
                continue
        logger.error("Giving up on JSON parse after %d retries", _MAX_JSON_RETRIES)
        return {"node_sets": []}

    @staticmethod
    def _validate_node_sets(items: list) -> list[NodeSet]:
        valid: list[NodeSet] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                logger.warning("node_sets[%d] is not a dict: %r", i, type(item))
                continue
            try:
                valid.append(NodeSet.model_validate(item))
            except ValidationError as exc:
                logger.warning("node_sets[%d] failed validation: %s", i, exc)
        return valid
