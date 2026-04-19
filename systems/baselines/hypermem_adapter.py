"""HyperMem adapter (stub) — EverOS submodule."""

from __future__ import annotations

from dataclasses import dataclass

from systems.baselines.base import REPO_ROOT, BaselineAdapter


@dataclass
class HyperMemAdapter(BaselineAdapter):
    name: str = "hypermem"

    def __post_init__(self) -> None:
        self.external_path = (
            REPO_ROOT / "external" / "everos" / "methods" / "HyperMem"
        )

    def _readiness_hint(self) -> str:
        return (
            "To wire: (1) set OPENROUTER_API_KEY (paper default) or OPENAI_API_KEY; "
            "(2) add external/everos/methods/HyperMem/hypermem to sys.path; "
            "(3) stand up local vLLM Qwen3-Embedding-4B + Qwen3-Reranker-4B OR "
            "point at DeepInfra as fallback (see env.template); (4) run the 6-stage "
            "pipeline: hypermem.main.eval with stages 1-6 per conversation. "
            "User's 3080 Ti may not hold both Qwen3-4B models concurrently at FP16 "
            "— prefer DeepInfra fallback for initial runs."
        )
