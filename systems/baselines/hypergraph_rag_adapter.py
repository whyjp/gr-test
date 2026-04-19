"""HyperGraphRAG adapter (stub)."""

from __future__ import annotations

from dataclasses import dataclass

from systems.baselines.base import REPO_ROOT, BaselineAdapter


@dataclass
class HyperGraphRAGAdapter(BaselineAdapter):
    name: str = "hypergraph_rag"

    def __post_init__(self) -> None:
        self.external_path = REPO_ROOT / "external" / "hypergraph_rag"

    def _readiness_hint(self) -> str:
        return (
            "To wire: (1) set OPENAI_API_KEY; (2) add external/hypergraph_rag to "
            "sys.path; (3) instantiate hypergraphrag.HyperGraphRAG(working_dir=..); "
            "(4) override ingest() to call .insert([conv_text]) for each "
            "conversation; retrieve() maps to .query(question). LoCoMo dialogue "
            "input must be concatenated per session or per conversation — prompt "
            "tuning may be needed since upstream targets narrative medical / legal "
            "documents."
        )
