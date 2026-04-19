"""Typed models for Hyper Triplet node_sets.

v5 adds explicit L0/L1/L2/L3 layer views (per
`docs/hyper-triplet-implementation-plan-v5.md`) on top of the original
`Fact` + `Qualifiers` shape. The legacy shape is preserved for
backwards compatibility; new code should access layered views via
`NodeSet.l0 / l1 / l2 / l3`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QualifierType = Literal[
    "location",
    "participant",
    "activity_type",
    "time_reference",
    "mood",
    "topic",
]

QUALIFIER_TYPES: tuple[QualifierType, ...] = (
    "location",
    "participant",
    "activity_type",
    "time_reference",
    "mood",
    "topic",
)

# LLM-facing key for participants is plural (a list); internal `participant`
# type name is singular (one node per participant).
_LLM_TO_INTERNAL_KEY = {
    "location": "location",
    "participants": "participant",
    "activity_type": "activity_type",
    "time_reference": "time_reference",
    "mood": "mood",
    "topic": "topic",
}


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str
    predicate: str
    object: str

    def to_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


class Qualifiers(BaseModel):
    model_config = ConfigDict(frozen=True)

    location: str | None = None
    participants: tuple[str, ...] = ()
    activity_type: str | None = None
    time_reference: str | None = None
    mood: str | None = None
    topic: str | None = None

    @field_validator("participants", mode="before")
    @classmethod
    def _coerce_participants(cls, v):
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,)
        return tuple(v)

    def iter_typed_values(self) -> Iterator[tuple[QualifierType, str]]:
        """Yield (qualifier_type, value) pairs for every non-null value.

        Participants fan out: one tuple per person. Empty strings are skipped.
        """
        if self.location and self.location.strip():
            yield "location", self.location.strip()
        for p in self.participants:
            if p and p.strip():
                yield "participant", p.strip()
        if self.activity_type and self.activity_type.strip():
            yield "activity_type", self.activity_type.strip()
        if self.time_reference and self.time_reference.strip():
            yield "time_reference", self.time_reference.strip()
        if self.mood and self.mood.strip():
            yield "mood", self.mood.strip()
        if self.topic and self.topic.strip():
            yield "topic", self.topic.strip()


# ---------------------------------------------------------------------------
# v5 layer views — read-only projections of NodeSet into its 4 functional layers
# ---------------------------------------------------------------------------


class L0Fact(BaseModel):
    """Core atomic fact layer: the (subject, predicate, object) triple plus
    relation-level qualifiers (confidence, per-edge validity etc.)."""

    model_config = ConfigDict(frozen=True)

    subject: str
    predicate: str
    object: str
    edge_qualifiers: dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


class L1TemporalImportance(BaseModel):
    """Temporal + importance metadata layer."""

    model_config = ConfigDict(frozen=True)

    timestamp: str | None = None
    time_reference: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    duration_days: int | None = None
    importance: float = 0.0
    belief: float = 1.0


class L2Context(BaseModel):
    """Context / environment layer: location, participants, activity, mood."""

    model_config = ConfigDict(frozen=True)

    location: str | None = None
    participants: tuple[str, ...] = ()
    activity_type: str | None = None
    mood: str | None = None


class L3Auxiliary(BaseModel):
    """Auxiliary / derived layer: topic, community id, embedding pointer, source ref."""

    model_config = ConfigDict(frozen=True)

    topic: str | None = None
    community_id: str | None = None
    embedding_ref: str | None = None
    source_ref: str | None = None


def _compute_ns_id(fact: Fact) -> str:
    """Deterministic ns_id from the fact triple text. Matches `fact_node_id`
    hashing scheme so the same fact gets a stable identifier across runs."""
    key = fact.to_text().strip().lower()
    return "ns-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NodeSet — v5-extended with ns_id + importance + layer views
# ---------------------------------------------------------------------------


class NodeSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact: Fact
    source_episode_ids: tuple[str, ...] = Field(default_factory=tuple)
    belief: float = 1.0
    qualifiers: Qualifiers = Field(default_factory=Qualifiers)
    # v5 additions (optional so existing construction still works):
    importance: float = 0.0
    ns_id: str | None = None

    @field_validator("belief", mode="before")
    @classmethod
    def _clamp_belief(cls, v):
        try:
            b = float(v)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, b))

    @field_validator("source_episode_ids", mode="before")
    @classmethod
    def _coerce_source_ids(cls, v):
        if v is None:
            return ()
        return tuple(v)

    @field_validator("importance", mode="before")
    @classmethod
    def _clamp_importance(cls, v):
        try:
            i = float(v)
        except (TypeError, ValueError):
            return 0.0
        # importance is unbounded in theory but we clip to [0, 1e6] to catch runaway values
        return max(0.0, min(1e6, i))

    @property
    def effective_ns_id(self) -> str:
        """Returns the explicit ns_id if set, else a deterministic hash of the fact."""
        return self.ns_id or _compute_ns_id(self.fact)

    # --- layer views (v5) ---

    @property
    def l0(self) -> L0Fact:
        return L0Fact(
            subject=self.fact.subject,
            predicate=self.fact.predicate,
            object=self.fact.object,
        )

    @property
    def l1(self) -> L1TemporalImportance:
        return L1TemporalImportance(
            time_reference=self.qualifiers.time_reference,
            importance=self.importance,
            belief=self.belief,
        )

    @property
    def l2(self) -> L2Context:
        return L2Context(
            location=self.qualifiers.location,
            participants=self.qualifiers.participants,
            activity_type=self.qualifiers.activity_type,
            mood=self.qualifiers.mood,
        )

    @property
    def l3(self) -> L3Auxiliary:
        return L3Auxiliary(topic=self.qualifiers.topic)


def merge_key(qualifier_type: QualifierType, value: str) -> tuple[QualifierType, str]:
    """Canonical MERGE key: (type, case-folded stripped value).

    Two qualifier values merge into one graph node iff their merge_keys match.
    """
    return qualifier_type, value.strip().lower()
