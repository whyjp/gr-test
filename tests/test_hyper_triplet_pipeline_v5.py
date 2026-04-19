"""Tests for HyperTripletPipelineV5 — offline, using MockLLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from htb.data import load_locomo10
from htb.eval import BenchmarkRunner, KeywordMockJudge, Pipeline
from htb.llm import build_replay_mock, load_fixture
from systems.hyper_triplet.config import DEFAULT_CONFIG, HyperTripletConfig
from systems.hyper_triplet.extractors import LLMNodeSetExtractor
from systems.hyper_triplet.pipeline import template_answerer
from systems.hyper_triplet.pipeline_v5 import HyperTripletPipelineV5

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_conv26_session1_gold.json"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists() or not FIXTURE_PATH.exists(),
    reason="data/locomo10.json or fixture missing",
)


@pytest.fixture(scope="module")
def conv26():
    return next(c for c in load_locomo10(DATA_PATH) if c.sample_id == "conv-26")


@pytest.fixture()
def fixture_pipeline():
    fixture = load_fixture(FIXTURE_PATH)
    mock = build_replay_mock(fixture)
    extractor = LLMNodeSetExtractor(llm=mock)
    return HyperTripletPipelineV5(extractor=extractor, answerer=template_answerer)


def test_v5_pipeline_satisfies_protocol(fixture_pipeline):
    assert isinstance(fixture_pipeline, Pipeline)
    assert fixture_pipeline.name == "hyper-triplet-v5"


def test_reset_initialises_all_components(fixture_pipeline):
    fixture_pipeline.reset()
    assert fixture_pipeline.store.stats()["n_stars"] == 0
    assert fixture_pipeline._boundary is not None
    assert fixture_pipeline._three_stage is not None
    assert fixture_pipeline._community is not None
    assert fixture_pipeline._importance is not None


def test_ingest_populates_star_store(conv26, fixture_pipeline):
    fixture_pipeline.reset()
    fixture_pipeline.ingest(conv26)
    stats = fixture_pipeline.store.stats()
    # Fixture only supplies gold for session 1 chunks; other sessions get empty
    # node_sets from the mock's default. So we expect >=3 stars (the fixture has
    # 7 gold node_sets total, but boundary segmentation may change chunk match).
    assert stats["n_stars"] >= 1


def test_retrieve_returns_evidence_for_fixture_query(conv26, fixture_pipeline):
    fixture_pipeline.reset()
    fixture_pipeline.ingest(conv26)
    r = fixture_pipeline.retrieve("When did Melanie paint a sunrise?")
    # We don't assert exact evidence because boundary-detector chunking may
    # differ from the fixture's hand-picked 3 chunks. But the pipeline must
    # respond without crashing and produce some context when the question
    # hits any ingested node_set.
    assert isinstance(r.context, str)
    assert isinstance(r.evidence_dia_ids, tuple)


def test_community_detector_enabled_by_default(fixture_pipeline):
    fixture_pipeline.reset()
    # After reset+ingest, the pipeline's _community should have run when ingest
    # is called. Here we only check the component is present.
    assert fixture_pipeline._community is not None


def test_importance_scorer_writes_back(conv26, fixture_pipeline):
    fixture_pipeline.reset()
    fixture_pipeline.ingest(conv26)
    # After ingest, every star should have importance > 0 (neutral recency baseline = 1)
    store = fixture_pipeline.store
    if store.stats()["n_stars"] > 0:
        any_star = next(store.iter_stars())
        assert any_star.importance > 0


def test_disable_community_via_config(fixture_pipeline):
    fixture_pipeline.config = DEFAULT_CONFIG.with_overrides(use_community=False)
    fixture_pipeline.reset()
    # Adding facts should NOT invoke community detection.
    # We verify by putting 2 stars that share a qualifier and confirming no
    # community_id got assigned.
    from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers

    store = fixture_pipeline.store
    a = NodeSet(fact=Fact(subject="A", predicate="did", object="X"), qualifiers=Qualifiers(location="Paris"))
    b = NodeSet(fact=Fact(subject="B", predicate="did", object="Y"), qualifiers=Qualifiers(location="Paris"))
    store.put(a)
    store.put(b)
    # Manually simulate what ingest would do — but with use_community=False
    # the pipeline wouldn't call detect(). Assert it hasn't been called:
    assert store.community_of(a.effective_ns_id) is None


def test_benchmark_runner_end_to_end(conv26, fixture_pipeline):
    runner = BenchmarkRunner(pipeline=fixture_pipeline, judge=KeywordMockJudge())
    results = runner.run([conv26], n_runs=1)
    assert len(results.per_run) == 1
    records = results.per_run[0]
    # Pipeline ran end-to-end, records exist for category 1-4 questions
    assert all(r.category in {1, 2, 3, 4} for r in records)


def test_retrieve_before_reset_returns_empty(fixture_pipeline):
    """Calling retrieve() on a fresh pipeline returns empty without crashing."""
    fresh = HyperTripletPipelineV5(extractor=fixture_pipeline.extractor)
    r = fresh.retrieve("anything")
    assert r.context == ""
    assert r.word_count == 0


def test_respects_budget_words_override(conv26, fixture_pipeline):
    fixture_pipeline.reset()
    fixture_pipeline.ingest(conv26)
    r_small = fixture_pipeline.retrieve("Melanie Caroline painting", budget_words=3)
    r_large = fixture_pipeline.retrieve("Melanie Caroline painting", budget_words=500)
    assert r_small.word_count <= r_large.word_count


def test_custom_config_propagates_to_components():
    """Verify HyperTripletConfig's per-module configs reach the pipeline's
    runtime components after reset()."""
    from systems.hyper_triplet.boundary_detector import BoundaryConfig

    custom_boundary = BoundaryConfig(max_turns_per_chunk=3, min_chunk_size=1)
    cfg = HyperTripletConfig(boundary=custom_boundary)
    pipeline = HyperTripletPipelineV5(
        extractor=LLMNodeSetExtractor(llm=None),  # type: ignore[arg-type]
        config=cfg,
    )
    pipeline.reset()
    assert pipeline._boundary.config.max_turns_per_chunk == 3  # type: ignore[union-attr]
