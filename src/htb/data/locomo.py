"""LoCoMo-10 loader.

Actual schema differs from the plan doc. See MEMORY project_locomo10_schema.md.

Top-level is a list of 10 sample dicts:
  {
    "sample_id": "conv-26",
    "conversation": {
        "speaker_a", "speaker_b",
        "session_<N>_date_time", "session_<N>": [turn, ...], ...
    },
    "qa": [{"question", "answer", "evidence", "category"}, ...],
    "event_summary": {...}?, "observation": {...}?, "session_summary": {...}?,
  }
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")
DIA_ID_EXTRACT_RE = re.compile(r"D:?\s*(\d+)\s*:\s*(\d+)")

QA_CATEGORIES_BENCHMARK: frozenset[int] = frozenset({1, 2, 3, 4})
QA_CATEGORY_ADVERSARIAL: int = 5


def normalize_dia_ids(raw: str) -> list[str]:
    """Return canonical D<N>:<M> ids extracted from a possibly-messy evidence string.

    Real LoCoMo-10 evidence strings include:
      - multiple refs joined by space, semicolon, or comma ("D9:1 D4:4")
      - zero-padded turn index ("D30:05")
      - typo with extra colon ("D:11:26")
      - purely malformed tokens ("D") — silently dropped

    Returns an empty list if no canonical id can be recovered.
    """
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in DIA_ID_EXTRACT_RE.finditer(raw):
        sess = int(m.group(1))
        turn = int(m.group(2))
        cid = f"D{sess}:{turn}"
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: str
    dia_id: str
    text: str
    session_index: int

    @property
    def turn_index(self) -> int:
        m = DIA_ID_RE.match(self.dia_id)
        if not m:
            raise ValueError(f"malformed dia_id: {self.dia_id!r}")
        return int(m.group(2))


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    date_time: str
    turns: tuple[Turn, ...]


class QAPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: Any  # stringified on access; may be str | int | list in raw data
    evidence: tuple[str, ...] = Field(default_factory=tuple)
    category: int
    adversarial_answer: str | None = None

    @property
    def gold_answer_text(self) -> str:
        a = self.answer
        if isinstance(a, str):
            return a
        if isinstance(a, (int, float)):
            return str(a)
        if isinstance(a, list):
            return " | ".join(str(x) for x in a)
        return str(a)

    @property
    def evidence_dia_ids(self) -> tuple[str, ...]:
        """Canonical dia_ids parsed from possibly-messy raw evidence strings."""
        out: list[str] = []
        seen: set[str] = set()
        for raw in self.evidence:
            for cid in normalize_dia_ids(raw):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return tuple(out)


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    qa: tuple[QAPair, ...]
    event_summary: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    session_summary: dict[str, Any] | None = None

    def iter_turns(self) -> Iterator[Turn]:
        for s in self.sessions:
            yield from s.turns

    def turn_by_dia_id(self, dia_id: str) -> Turn | None:
        for t in self.iter_turns():
            if t.dia_id == dia_id:
                return t
        return None

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


def _parse_turn(raw: dict[str, Any], session_index: int) -> Turn:
    return Turn(
        speaker=raw["speaker"],
        dia_id=raw["dia_id"],
        text=raw["text"],
        session_index=session_index,
    )


def _parse_sessions(conv: dict[str, Any]) -> tuple[Session, ...]:
    session_indices: list[int] = []
    for k in conv:
        m = SESSION_KEY_RE.match(k)
        if m:
            session_indices.append(int(m.group(1)))
    session_indices.sort()

    sessions: list[Session] = []
    for idx in session_indices:
        turns_raw = conv[f"session_{idx}"]
        date_time = conv.get(f"session_{idx}_date_time", "")
        turns = tuple(_parse_turn(t, idx) for t in turns_raw)
        sessions.append(Session(index=idx, date_time=date_time, turns=turns))
    return tuple(sessions)


def _parse_qa(raw: dict[str, Any]) -> QAPair:
    return QAPair(
        question=raw["question"],
        answer=raw.get("answer"),
        evidence=tuple(raw.get("evidence") or ()),
        category=int(raw["category"]),
        adversarial_answer=raw.get("adversarial_answer"),
    )


def _parse_sample(raw: dict[str, Any]) -> Conversation:
    conv = raw["conversation"]
    sessions = _parse_sessions(conv)
    qa = tuple(_parse_qa(q) for q in raw["qa"])
    return Conversation(
        sample_id=raw["sample_id"],
        speaker_a=conv["speaker_a"],
        speaker_b=conv["speaker_b"],
        sessions=sessions,
        qa=qa,
        event_summary=raw.get("event_summary"),
        observation=raw.get("observation"),
        session_summary=raw.get("session_summary"),
    )


def load_locomo10(path: str | Path) -> list[Conversation]:
    """Load locomo10.json and return a list of 10 Conversation objects.

    Raises ValueError if the file doesn't look like LoCoMo-10.
    """
    raw = orjson.loads(Path(path).read_bytes())
    if not isinstance(raw, list):
        raise ValueError(f"expected top-level list, got {type(raw).__name__}")
    if len(raw) != 10:
        raise ValueError(f"expected 10 conversations, got {len(raw)}")
    return [_parse_sample(s) for s in raw]


def iter_qa_excluding_adversarial(
    conversations: Iterable[Conversation],
) -> Iterator[tuple[Conversation, QAPair]]:
    """Yield (conversation, qa) pairs with category != 5 (adversarial)."""
    for conv in conversations:
        for qa in conv.qa:
            if qa.category in QA_CATEGORIES_BENCHMARK:
                yield conv, qa
