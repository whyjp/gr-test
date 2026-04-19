"""Offline schema validation for LoCoMo-10 loader.

Skipped automatically if data/locomo10.json is not present; run
`bash scripts/fetch-locomo10.sh` first.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from htb.data import (
    Conversation,
    QAPair,
    Session,
    Turn,
    iter_qa_excluding_adversarial,
    load_locomo10,
    normalize_dia_ids,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="data/locomo10.json missing; run scripts/fetch-locomo10.sh",
)


@pytest.fixture(scope="module")
def convs() -> list[Conversation]:
    return load_locomo10(DATA_PATH)


def test_ten_conversations(convs: list[Conversation]):
    assert len(convs) == 10


def test_sample_ids_unique_and_prefixed(convs: list[Conversation]):
    ids = [c.sample_id for c in convs]
    assert len(set(ids)) == 10
    assert all(cid.startswith("conv-") for cid in ids)


def test_sessions_nonempty_and_ordered(convs: list[Conversation]):
    for c in convs:
        assert len(c.sessions) >= 1
        indices = [s.index for s in c.sessions]
        assert indices == sorted(indices)
        for s in c.sessions:
            assert len(s.turns) > 0
            assert isinstance(s, Session)


def test_turn_fields(convs: list[Conversation]):
    for c in convs:
        for t in c.iter_turns():
            assert isinstance(t, Turn)
            assert t.speaker in {c.speaker_a, c.speaker_b}
            assert t.dia_id.startswith("D")
            assert ":" in t.dia_id
            assert t.text
            assert t.turn_index >= 1


def test_total_turns_matches_inspection(convs: list[Conversation]):
    total = sum(c.n_turns for c in convs)
    assert total == 5882  # matches ad-hoc inspection


def test_qa_fields(convs: list[Conversation]):
    for c in convs:
        for q in c.qa:
            assert isinstance(q, QAPair)
            assert q.question
            assert q.category in {1, 2, 3, 4, 5}


def test_qa_total_and_benchmark_subset(convs: list[Conversation]):
    total = sum(len(c.qa) for c in convs)
    benchmark = sum(1 for _ in iter_qa_excluding_adversarial(convs))
    assert total == 1986
    assert benchmark == 1540  # 1,986 minus 446 adversarial


def test_qa_category_distribution(convs: list[Conversation]):
    dist = Counter(q.category for c in convs for q in c.qa)
    assert dist[1] == 282
    assert dist[2] == 321
    assert dist[3] == 96
    assert dist[4] == 841
    assert dist[5] == 446


def test_normalize_dia_ids_handles_real_edge_cases():
    # multi-ref with semicolon
    assert normalize_dia_ids("D8:6; D9:17") == ["D8:6", "D9:17"]
    # multi-ref with whitespace
    assert normalize_dia_ids("D9:1 D4:4 D4:6") == ["D9:1", "D4:4", "D4:6"]
    # typo with extra colon
    assert normalize_dia_ids("D:11:26") == ["D11:26"]
    # zero-padding on turn index
    assert normalize_dia_ids("D30:05") == ["D30:5"]
    # purely malformed token
    assert normalize_dia_ids("D") == []
    # empty
    assert normalize_dia_ids("") == []
    # dedup
    assert normalize_dia_ids("D1:1 D1:1") == ["D1:1"]


# Known LoCoMo-10 dataset errata: evidence dia_ids that pass normalization
# but genuinely don't exist in the conversation (session doesn't have that many turns).
# Keep this list tight — if it grows, investigate.
KNOWN_EVIDENCE_ERRATA: set[tuple[str, str]] = {
    ("conv-42", "D10:19"),  # session_10 only has 16 turns
    ("conv-47", "D4:36"),   # session_4 only has 25 turns
}


def test_evidence_dia_ids_resolve_to_real_turns(convs: list[Conversation]):
    """Category 1-4 QA: normalized evidence dia_ids should point to actual turns,
    except for a small known set of LoCoMo-10 dataset errata.

    Adversarial (cat 5) evidence may reference turns that don't exist by design.
    Raw evidence strings are messy (multi-ref joins, zero-padding, typos); the loader
    exposes `.evidence_dia_ids` which returns cleaned canonical ids.
    """
    unresolved: set[tuple[str, str]] = set()
    empty_evidence: list[tuple[str, tuple[str, ...]]] = []
    for c in convs:
        for q in c.qa:
            if q.category == 5:
                continue
            normalized = q.evidence_dia_ids
            if q.evidence and not normalized:
                empty_evidence.append((c.sample_id, q.evidence))
            for cid in normalized:
                if c.turn_by_dia_id(cid) is None:
                    unresolved.add((c.sample_id, cid))
    unexpected = unresolved - KNOWN_EVIDENCE_ERRATA
    assert not unexpected, f"unexpected unresolved ids (not in known errata): {unexpected}"
    # At most the single 'D' token may fail to normalize; tolerate up to 2 for safety.
    assert len(empty_evidence) <= 2, f"too many unnormalizable evidences: {empty_evidence}"


def test_frozen_models_are_hashable(convs: list[Conversation]):
    t = next(convs[0].iter_turns())
    # pydantic frozen=True makes instances hashable
    _ = {t}
