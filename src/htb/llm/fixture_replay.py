"""Fixture-replay LLM adapter — returns canned node_set responses keyed by
which chunk's episode markers appear in the rendered prompt.

Lets us drive the full Hyper Triplet pipeline end-to-end with realistic
hand-crafted extraction data (no API, deterministic).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from htb.llm.mock import MockLLMAdapter


def load_fixture(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _chunk_matcher(marker: str) -> Callable[[str], bool]:
    """Prompt matcher keyed on a chunk's unique dia_id marker.

    The extractor renders episodes as `[ep-D1:3] [date] text`, so checking
    for `D1:3]` works reliably.
    """

    def _match(prompt: str) -> bool:
        return f"{marker}]" in prompt or f" {marker} " in prompt

    return _match


def build_replay_mock(
    fixture: dict[str, Any],
    *,
    default_empty_node_sets: bool = True,
) -> MockLLMAdapter:
    """Build a MockLLMAdapter whose rules replay gold node_sets from a fixture.

    Each chunk entry in `fixture["chunks"]` needs:
      - `marker`: a unique episode id (e.g. "D1:3") that appears in the prompt
        only for that chunk
      - `gold_node_sets`: list of node_set dicts
    """
    rules: list[tuple] = []
    for chunk in fixture.get("chunks", []):
        marker = chunk["marker"]
        gold = chunk["gold_node_sets"]
        response = json.dumps({"node_sets": gold})
        rules.append((_chunk_matcher(marker), response))

    default = json.dumps({"node_sets": []}) if default_empty_node_sets else None
    return MockLLMAdapter(rules=rules, default=default)
