"""Tests for BoundaryDetector."""

from __future__ import annotations

from htb.data.locomo import Turn
from systems.hyper_triplet.boundary_detector import (
    BoundaryConfig,
    BoundaryDetector,
    _content_tokens,
    _jaccard,
)


def _mk_turn(session_index: int, i: int, text: str) -> Turn:
    return Turn(speaker="A", dia_id=f"D{session_index}:{i}", text=text, session_index=session_index)


def test_empty_input_returns_empty():
    assert BoundaryDetector().segment([]) == []


def test_single_turn_returns_single_chunk():
    t = [_mk_turn(1, 1, "hello world")]
    chunks = BoundaryDetector().segment(t)
    assert len(chunks) == 1
    assert chunks[0] == t


def test_session_boundary_always_splits():
    turns = [
        _mk_turn(1, 1, "same topic alpha"),
        _mk_turn(1, 2, "same topic alpha continued"),
        _mk_turn(2, 1, "same topic alpha new session"),  # session change forces split
    ]
    chunks = BoundaryDetector().segment(turns)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert chunks[1][0].session_index == 2


def test_max_turns_per_chunk_caps_chunk_size():
    turns = [_mk_turn(1, i, f"same word token {i % 2}") for i in range(1, 11)]
    detector = BoundaryDetector(config=BoundaryConfig(max_turns_per_chunk=3))
    chunks = detector.segment(turns)
    # 10 turns / 3 per chunk = 4 chunks (3+3+3+1)
    assert len(chunks) == 4
    assert all(len(c) <= 3 for c in chunks)


def test_topic_drift_causes_split():
    turns = [
        _mk_turn(1, 1, "painting art colors canvas"),
        _mk_turn(1, 2, "painting art canvas sunset"),
        _mk_turn(1, 3, "completely different topic football tournament"),
    ]
    detector = BoundaryDetector(
        config=BoundaryConfig(entity_overlap_threshold=0.3, max_turns_per_chunk=10)
    )
    chunks = detector.segment(turns)
    # The third turn has zero overlap with the "painting" chunk -> split
    assert len(chunks) == 2
    assert "painting" in chunks[0][0].text
    assert "football" in chunks[1][0].text


def test_high_overlap_keeps_chunks_together():
    turns = [
        _mk_turn(1, 1, "painting art colors"),
        _mk_turn(1, 2, "painting art sunset"),
        _mk_turn(1, 3, "painting art canvas"),
    ]
    # Overlap threshold low, so high-overlap content stays grouped
    detector = BoundaryDetector(
        config=BoundaryConfig(entity_overlap_threshold=0.1, max_turns_per_chunk=10)
    )
    chunks = detector.segment(turns)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_content_tokens_filters_stopwords():
    assert _content_tokens("the quick brown fox") == {"quick", "brown", "fox"}
    assert _content_tokens("") == frozenset()


def test_jaccard_extremes():
    assert _jaccard(frozenset(), frozenset()) == 1.0
    assert _jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert _jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
    assert _jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1.0 / 3.0


def test_locomo_fixture_segmentation():
    """Smoke test on real conv-26 session 1 turns."""
    from pathlib import Path

    from htb.data import load_locomo10

    data_path = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
    if not data_path.exists():
        import pytest

        pytest.skip("data/locomo10.json missing")
    convs = load_locomo10(data_path)
    conv26 = next(c for c in convs if c.sample_id == "conv-26")
    session1_turns = conv26.sessions[0].turns
    chunks = BoundaryDetector(config=BoundaryConfig(max_turns_per_chunk=8)).segment(
        list(session1_turns)
    )
    # Session 1 has 18 turns; with max=8, expect at least 3 chunks
    assert len(chunks) >= 3
    # Every chunk respects session boundary (session 1 only)
    assert all(all(t.session_index == 1 for t in chunk) for chunk in chunks)
    # Turn count preserved
    assert sum(len(c) for c in chunks) == len(session1_turns)


def test_min_chunk_size_prevents_micro_splits():
    turns = [
        _mk_turn(1, 1, "only word foo"),
        _mk_turn(1, 2, "completely different bar"),
    ]
    # With min_chunk_size=2 and low overlap, the first turn alone can't become
    # its own chunk; the drift heuristic is suppressed until it has 2 turns.
    detector = BoundaryDetector(
        config=BoundaryConfig(min_chunk_size=2, entity_overlap_threshold=0.5)
    )
    chunks = detector.segment(turns)
    assert len(chunks) == 1  # drift allowed only after size >= 2
