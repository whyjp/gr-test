"""Typed models for Hyper Triplet node_sets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

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


class NodeSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact: Fact
    source_episode_ids: tuple[str, ...] = Field(default_factory=tuple)
    belief: float = 1.0
    qualifiers: Qualifiers = Field(default_factory=Qualifiers)

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


def merge_key(qualifier_type: QualifierType, value: str) -> tuple[QualifierType, str]:
    """Canonical MERGE key: (type, case-folded stripped value).

    Two qualifier values merge into one graph node iff their merge_keys match.
    """
    return qualifier_type, value.strip().lower()
