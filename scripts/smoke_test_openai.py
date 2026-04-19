"""Smoke test: HyperTripletPipelineV5 end-to-end against the real OpenAI API.

Runs ingest + retrieve + answer on ONE LoCoMo conversation with a tiny QA
subset (category 1-2, first 5 only) so cost stays under ~$0.10 USD.

Usage:
    uv run --extra llm python scripts/smoke_test_openai.py

Reads .env at repo root for OPENAI_API_KEY. Prints a compact per-QA table
and an overall accuracy summary. Writes ``results/smoke_test.json`` for
later inspection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from htb.data import load_locomo10  # noqa: E402
from htb.eval import KeywordMockJudge  # noqa: E402
from htb.llm import OpenAIAdapter  # noqa: E402
from systems.hyper_triplet.config import HyperTripletConfig  # noqa: E402
from systems.hyper_triplet.extractors import LLMNodeSetExtractor  # noqa: E402
from systems.hyper_triplet.pipeline import template_answerer  # noqa: E402
from systems.hyper_triplet.pipeline_v5 import HyperTripletPipelineV5  # noqa: E402


SAMPLE_ID = "conv-26"
MAX_QA = 5
CATEGORIES = {1, 2}


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY missing from .env", file=sys.stderr)
        return 1

    data_path = REPO_ROOT / "data" / "locomo10.json"
    if not data_path.exists():
        print(f"[error] dataset missing at {data_path}; run scripts/fetch-locomo10.sh", file=sys.stderr)
        return 1

    convs = load_locomo10(data_path)
    conv = next(c for c in convs if c.sample_id == SAMPLE_ID)
    qa_subset = [q for q in conv.qa if q.category in CATEGORIES][:MAX_QA]
    print(f"[smoke] sample={conv.sample_id} turns={conv.n_turns} qa={len(qa_subset)}")

    extract_model = os.environ.get("EXTRACT_MODEL", "gpt-4o-mini")
    print(f"[smoke] LLM={extract_model}")
    extractor_llm = OpenAIAdapter(default_model=extract_model)
    extractor = LLMNodeSetExtractor(llm=extractor_llm)

    pipeline = HyperTripletPipelineV5(
        extractor=extractor,
        answerer=template_answerer,
        config=HyperTripletConfig(),
    )
    judge = KeywordMockJudge()

    t0 = time.time()
    print("[smoke] ingesting...")
    pipeline.reset()
    pipeline.ingest(conv)
    ingest_time = time.time() - t0
    stats = pipeline.store.stats()
    print(f"[smoke] ingest_time={ingest_time:.1f}s  store={stats}")

    records: list[dict] = []
    correct = 0
    for i, qa in enumerate(qa_subset):
        t_q = time.time()
        r = pipeline.retrieve(qa.question, budget_words=1000)
        ans = pipeline.answer(qa.question, r)
        verdict = judge.judge(qa.question, qa.gold_answer_text, ans.text)
        is_correct = verdict == "CORRECT"
        correct += int(is_correct)
        elapsed = time.time() - t_q
        records.append(
            {
                "i": i,
                "category": qa.category,
                "question": qa.question,
                "gold": qa.gold_answer_text,
                "answer": ans.text[:200],
                "evidence": list(r.evidence_dia_ids),
                "verdict": verdict,
                "elapsed_s": round(elapsed, 2),
            }
        )
        flag = "OK " if is_correct else "X  "
        print(
            f"  {flag} cat{qa.category} ({elapsed:4.1f}s)  {qa.question[:70]}"
        )
        print(f"       gold: {qa.gold_answer_text[:100]}")
        print(f"       ans:  {ans.text[:100]}")

    total_time = time.time() - t0
    accuracy = correct / len(qa_subset) if qa_subset else 0.0
    print()
    print(
        f"[smoke] total_time={total_time:.1f}s  "
        f"accuracy={accuracy:.2%}  correct={correct}/{len(qa_subset)}"
    )

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "smoke_test.json"
    out_path.write_text(
        json.dumps(
            {
                "sample_id": conv.sample_id,
                "n_qa": len(qa_subset),
                "accuracy": accuracy,
                "ingest_time_s": round(ingest_time, 2),
                "total_time_s": round(total_time, 2),
                "store_stats": stats,
                "extract_model": extract_model,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[smoke] results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
