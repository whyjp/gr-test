"""Tests for result serialisation + summary generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from htb.data.locomo import Conversation, QAPair, Session, Turn
from htb.eval import (
    AblationRunner,
    BenchmarkRunner,
    KeywordMockJudge,
    LoadedResult,
    RunMetadata,
    SystemResult,
    format_summary_markdown,
    load_all_in_dir,
    load_system_result,
    save_ablation_sweep,
    save_system_result,
    serialize_system_result,
    write_summary,
)
from htb.eval.interfaces import AnswerResult, RetrievalResult


def _mk_conv() -> Conversation:
    t = Turn(speaker="A", dia_id="D1:1", text="hi", session_index=1)
    s = Session(index=1, date_time="t", turns=(t,))
    return Conversation(
        sample_id="c1",
        speaker_a="A",
        speaker_b="B",
        sessions=(s,),
        qa=tuple(
            QAPair(question=q, answer=a, category=c)
            for q, a, c in [("q1", "apple", 1), ("q2", "banana", 2)]
        ),
    )


@dataclass
class OraclePipeline:
    name: str = "oracle"
    _map: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self._map = {}

    def ingest(self, conversation: Conversation) -> None:
        for q in conversation.qa:
            self._map[q.question] = q.gold_answer_text

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        return RetrievalResult(context=self._map.get(query, ""), word_count=1)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=retrieval.context)


def _run_oracle() -> SystemResult:
    runner = BenchmarkRunner(pipeline=OraclePipeline(), judge=KeywordMockJudge())
    rr = runner.run([_mk_conv()], n_runs=1)
    return SystemResult(name="oracle", run_results=rr)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_serialize_system_result_shape():
    sys_res = _run_oracle()
    md = RunMetadata(dataset="locomo10", system="oracle", seed=42)
    payload = serialize_system_result(sys_res, md)
    assert payload["metadata"]["system"] == "oracle"
    assert payload["metadata"]["seed"] == 42
    assert payload["n"] == 2
    assert payload["accuracy"] == 1.0
    assert "1" in payload["per_category"]
    assert "2" in payload["per_category"]
    assert len(payload["records"]) == 2


def test_save_and_load_round_trip(tmp_path: Path):
    sys_res = _run_oracle()
    md = RunMetadata(dataset="locomo10", system="oracle", seed=42)
    path = save_system_result(sys_res, md, tmp_path)
    assert path.exists()
    assert path.name == "locomo10_oracle_42.json"

    loaded = load_system_result(path)
    assert isinstance(loaded, LoadedResult)
    assert loaded.metadata.system == "oracle"
    assert loaded.metadata.seed == 42
    assert loaded.accuracy == 1.0
    assert loaded.n == 2
    assert len(loaded.records) == 2
    assert 1 in loaded.per_category
    assert 2 in loaded.per_category


def test_load_all_in_dir_orders_deterministically(tmp_path: Path):
    sys_res = _run_oracle()
    save_system_result(sys_res, RunMetadata("locomo10", "a", 1), tmp_path)
    save_system_result(sys_res, RunMetadata("locomo10", "b", 2), tmp_path)
    save_system_result(sys_res, RunMetadata("locomo10", "a", 2), tmp_path)
    loaded = load_all_in_dir(tmp_path)
    names_seeds = [(r.metadata.system, r.metadata.seed) for r in loaded]
    # Sorted by filename: locomo10_a_1, locomo10_a_2, locomo10_b_2
    assert names_seeds == [("a", 1), ("a", 2), ("b", 2)]


def test_load_all_in_dir_ignores_non_json(tmp_path: Path):
    (tmp_path / "README.md").write_text("not a result", encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    sys_res = _run_oracle()
    save_system_result(sys_res, RunMetadata("locomo10", "a", 1), tmp_path)
    loaded = load_all_in_dir(tmp_path)
    # broken.json skipped, README.md not a json file and not loaded
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def test_format_summary_markdown_contains_headers(tmp_path: Path):
    sys_res = _run_oracle()
    for seed in (42, 1337):
        save_system_result(sys_res, RunMetadata("locomo10", "oracle", seed), tmp_path)
    loaded = load_all_in_dir(tmp_path)
    md = format_summary_markdown(loaded)
    assert "## Overall accuracy by system" in md
    assert "## Per-category accuracy" in md
    assert "oracle" in md


def test_write_summary_writes_markdown_file(tmp_path: Path):
    sys_res = _run_oracle()
    save_system_result(sys_res, RunMetadata("locomo10", "oracle", 42), tmp_path)
    path = write_summary(tmp_path)
    assert path.exists()
    assert path.name == "summary.md"
    text = path.read_text(encoding="utf-8")
    assert "oracle" in text


def test_summary_handles_empty_dir(tmp_path: Path):
    md = format_summary_markdown([])
    # No systems rows, but headers present
    assert "## Overall accuracy" in md


# ---------------------------------------------------------------------------
# Ablation sweep integration
# ---------------------------------------------------------------------------


def test_save_ablation_sweep_creates_one_file_per_preset(tmp_path: Path):
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OraclePipeline(),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_community", "no_importance"),
    )
    sweep = runner.run([_mk_conv()], n_runs=1)
    paths = save_ablation_sweep(
        sweep,
        dataset="locomo10",
        seed=42,
        results_dir=tmp_path,
        extract_model="gpt-4o-mini",
    )
    assert len(paths) == 3
    assert {p.name for p in paths} == {
        "locomo10_baseline_42.json",
        "locomo10_no_community_42.json",
        "locomo10_no_importance_42.json",
    }


def test_ablation_sweep_json_carries_notes(tmp_path: Path):
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OraclePipeline(),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_community"),
    )
    sweep = runner.run([_mk_conv()], n_runs=1)
    save_ablation_sweep(sweep, dataset="locomo10", seed=42, results_dir=tmp_path)
    no_community_path = tmp_path / "locomo10_no_community_42.json"
    payload = json.loads(no_community_path.read_text(encoding="utf-8"))
    assert "community" in payload["metadata"]["notes"].lower()


def test_make_run_metadata_from_env(monkeypatch):
    from htb.eval import make_run_metadata_from_env

    monkeypatch.setenv("EXTRACT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
    md = make_run_metadata_from_env(dataset="locomo10", system="oracle", seed=42)
    assert md.extract_model == "gpt-4o-mini"
    assert md.judge_model == "gpt-4o"
