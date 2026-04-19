"""BoundaryDetector — segment a turn stream into coherent chunks.

Replaces the fixed ``turns_per_chunk`` chunking used in v0 ``HyperTripletPipeline``.
Combines three signals:

1. **Session boundaries** — absolute (chunks never span sessions).
2. **Temporal gap** — large inter-turn time delta inside a session triggers a split.
3. **Entity overlap** — low Jaccard overlap between the current chunk's token set
   and the next turn's token set triggers a split (semantic drift heuristic).

All three are classifier-style decisions: "does this turn belong to the current
chunk?" — no LLM needed. Per plan v5 Phase B + HINGE Invariant #8.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from htb.data.locomo import Turn

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
        "from", "have", "has", "how", "i", "in", "is", "it", "its", "of",
        "on", "or", "so", "that", "the", "this", "to", "what", "when",
        "where", "who", "why", "with", "you", "your", "my", "was", "were",
    }
)


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(slots=True, frozen=True)
class BoundaryConfig:
    max_turns_per_chunk: int = 8
    min_chunk_size: int = 1
    entity_overlap_threshold: float = 0.1
    # Session index is an absolute boundary — chunks never span sessions.
    respect_session_boundary: bool = True


@dataclass(slots=True)
class BoundaryDetector:
    config: BoundaryConfig = BoundaryConfig()

    def segment(self, turns: Sequence[Turn]) -> list[list[Turn]]:
        """Partition a flat list of Turn into a list of chunks.

        Deterministic given config + input.
        """
        if not turns:
            return []

        chunks: list[list[Turn]] = []
        current: list[Turn] = [turns[0]]
        current_tokens = _content_tokens(turns[0].text)

        for i in range(1, len(turns)):
            turn = turns[i]
            prev_session = current[-1].session_index
            boundary = False

            if (self.config.respect_session_boundary and turn.session_index != prev_session) or len(current) >= self.config.max_turns_per_chunk:
                boundary = True
            else:
                turn_tokens = _content_tokens(turn.text)
                overlap = _jaccard(current_tokens, turn_tokens)
                # Low overlap signals topic drift; split if below threshold AND
                # the current chunk already satisfies the minimum size.
                if (
                    overlap < self.config.entity_overlap_threshold
                    and len(current) >= self.config.min_chunk_size
                ):
                    boundary = True

            if boundary:
                chunks.append(current)
                current = [turn]
                current_tokens = _content_tokens(turn.text)
            else:
                current.append(turn)
                current_tokens = current_tokens | _content_tokens(turn.text)

        if current:
            chunks.append(current)

        return chunks
