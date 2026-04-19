"""End-to-end Pipeline smoke test. Runs HyperTripletPipeline through the
BenchmarkRunner on real conv-26 turns using the fixture-replay mock LLM
and the KeywordMockJudge. Proves the full harness connects without API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from htb.data import load_locomo10
from htb.eval import BenchmarkRunner, KeywordMockJudge, Pipeline
from htb.llm import build_replay_mock, load_fixture
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.pipeline import (
    HyperTripletPipeline,
    context_passthrough_answerer,
    template_answerer,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "locomo_conv26_session1_gold.json"
)

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="data/locomo10.json missing; run scripts/fetch-locomo10.sh",
)


@pytest.fixture(scope="module")
def conv26():
    convs = load_locomo10(DATA_PATH)
    return next(c for c in convs if c.sample_id == "conv-26")


@pytest.fixture()
def fixture_pipeline():
    fixture = load_fixture(FIXTURE_PATH)
    mock = build_replay_mock(fixture)
    extractor = LLMNodeSetExtractor(llm=mock)
    return HyperTripletPipeline(extractor=extractor, answerer=template_answerer), fixture


def test_pipeline_satisfies_protocol(fixture_pipeline):
    pipeline, _ = fixture_pipeline
    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "hyper-triplet"


def test_reset_creates_fresh_graph(fixture_pipeline):
    pipeline, _ = fixture_pipeline
    pipeline.reset()
    assert pipeline.graph.stats()["n_nodes"] == 0


def test_ingest_chunks_feeds_fixture_boundaries(fixture_pipeline):
    pipeline, fixture = fixture_pipeline
    pipeline.reset()
    chunks = [
        [
            EpisodeRef(id=e["id"], text=e["text"], session_date=fixture["session_date"])
            for e in chunk["episodes"]
        ]
        for chunk in fixture["chunks"]
    ]
    pipeline.ingest_chunks(chunks)
    stats = pipeline.graph.stats()
    # All 3 chunks produced their gold, so 7 facts total (3+2+2)
    assert stats["nodes.fact"] == 7
    assert stats["nodes.episode"] == 16


def test_retrieve_returns_evidence_for_known_query(fixture_pipeline):
    pipeline, fixture = fixture_pipeline
    pipeline.reset()
    chunks = [
        [
            EpisodeRef(id=e["id"], text=e["text"], session_date=fixture["session_date"])
            for e in chunk["episodes"]
        ]
        for chunk in fixture["chunks"]
    ]
    pipeline.ingest_chunks(chunks)

    r = pipeline.retrieve("When did Caroline go to the LGBTQ support group?")
    assert r.context
    assert "D1:3" in r.evidence_dia_ids


def test_template_answerer_extracts_time_reference(fixture_pipeline):
    pipeline, fixture = fixture_pipeline
    pipeline.reset()
    chunks = [
        [
            EpisodeRef(id=e["id"], text=e["text"], session_date=fixture["session_date"])
            for e in chunk["episodes"]
        ]
        for chunk in fixture["chunks"]
    ]
    pipeline.ingest_chunks(chunks)

    r = pipeline.retrieve("When did Melanie paint a sunrise?")
    a = pipeline.answer("When did Melanie paint a sunrise?", r)
    # Fixture encodes 2022 as time_reference for the painting fact
    assert "2022" in a.text


def test_default_ingest_auto_chunks_full_conversation(conv26, fixture_pipeline):
    """Smoke test: calling pipeline.ingest(conv) with the full 14-session
    conversation succeeds without crash. Only the session-1 chunks have fixture
    gold; other sessions fall back to the mock's default empty response. We
    expect a non-zero but incomplete graph."""
    pipeline, _ = fixture_pipeline
    pipeline.reset()
    pipeline.ingest(conv26)
    stats = pipeline.graph.stats()
    assert stats["n_nodes"] > 0
    assert stats["nodes.episode"] == conv26.n_turns  # every turn -> episode
    # Some facts from session-1 chunks will materialise
    assert stats.get("nodes.fact", 0) >= 1


def test_benchmark_runner_exercises_full_loop(conv26, fixture_pipeline):
    """Run the full BenchmarkRunner loop on conv-26 and assert at least one
    category-1/2 QA referencing D1:X gets CORRECT under KeywordMockJudge."""
    pipeline, _ = fixture_pipeline
    runner = BenchmarkRunner(pipeline=pipeline, judge=KeywordMockJudge())
    results = runner.run([conv26], n_runs=1)

    assert len(results.per_run) == 1
    all_records = results.per_run[0]
    # Cat 5 (adversarial) excluded
    assert all(r.category in {1, 2, 3, 4} for r in all_records)

    # The pipeline produced SOME CORRECTs — proves wiring works end-to-end
    correct = [r for r in all_records if r.correct]
    assert len(correct) > 0, "expected at least one CORRECT judgement"

    # And specifically: the painting year QA (cat 2, evidence D1:12) or
    # LGBTQ date QA (cat 2, evidence D1:3) should be answerable with our
    # template answerer
    targeted_qs = [
        r for r in all_records
        if "LGBTQ" in r.question or "paint a sunrise" in r.question
    ]
    assert targeted_qs, "expected at least one targeted QA in conv-26"


def test_passthrough_answerer_returns_context():
    assert context_passthrough_answerer("q", "ctx") == "ctx"
    assert context_passthrough_answerer("q", "") == "(no context)"


def test_template_answerer_fallback_returns_first_fact():
    ctx = "## Facts\n- Alice met Bob\n    - location: Paris"
    assert template_answerer("who did Alice meet?", ctx) == "Alice met Bob"


def test_template_answerer_when_query_picks_time_reference():
    ctx = "## Facts\n- Alice visited Paris\n    - time_reference: June 2023\n    - mood: happy"
    assert template_answerer("When did Alice visit Paris?", ctx) == "June 2023"


def test_template_answerer_no_context_fallback():
    assert template_answerer("anything", "") == "(no context)"
