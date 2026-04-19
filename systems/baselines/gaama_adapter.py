"""GAAMA adapter (stub)."""

from __future__ import annotations

from dataclasses import dataclass

from systems.baselines.base import REPO_ROOT, BaselineAdapter


@dataclass
class GaamaAdapter(BaselineAdapter):
    name: str = "gaama"

    def __post_init__(self) -> None:
        self.external_path = REPO_ROOT / "external" / "gaama"
        # Populated by a wire-up function in Phase F that imports from
        # gaama.api.AgenticMemorySDK + gaama.services.*.

    def _readiness_hint(self) -> str:
        return (
            "To wire: (1) set OPENAI_API_KEY; (2) add external/gaama to sys.path; "
            "(3) instantiate gaama.api.AgenticMemorySDK; (4) override ingest() to "
            "call AgenticMemorySDK.ingest(...) + .create(CreateOptions(agent_id, "
            "task_id)); retrieve() via sdk.retrieve(RetrieveOptions(budget=..)). "
            "See external/gaama/evals/locomo/run_create_ltm.py for the canonical "
            "flow."
        )
