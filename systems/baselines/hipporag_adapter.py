"""HippoRAG 2 adapter (stub)."""

from __future__ import annotations

from dataclasses import dataclass

from systems.baselines.base import REPO_ROOT, BaselineAdapter


@dataclass
class HippoRAGAdapter(BaselineAdapter):
    name: str = "hipporag"

    def __post_init__(self) -> None:
        self.external_path = REPO_ROOT / "external" / "hipporag"

    def _readiness_hint(self) -> str:
        return (
            "To wire: (1) set OPENAI_API_KEY; (2) add external/hipporag to sys.path; "
            "(3) instantiate HippoRAG from main_openai.py; (4) override ingest() to "
            "feed passages (each session = passage). Caveat: HippoRAG 2's own "
            "evaluation is on MuSiQue / HotpotQA / 2Wiki — LoCoMo is NOT in their "
            "paper. The 69.9% headline in our baseline-numbers.md is from HippoRAG "
            "1 via GAAMA's LoCoMo port. Options: (a) use HippoRAG 2 as a new data "
            "point, (b) use GAAMA's inline HippoRAG port to match the 69.9% number."
        )
