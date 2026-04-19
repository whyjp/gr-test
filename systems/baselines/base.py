"""Base class + exceptions shared by every baseline Pipeline adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from htb.data.locomo import Conversation
from htb.eval.interfaces import AnswerResult, RetrievalResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class PipelineNotReadyError(RuntimeError):
    """Raised when a baseline adapter is asked to run but its upstream
    dependencies / API credentials are not wired yet.

    Includes a human-readable hint for what the operator needs to do.
    """


@dataclass
class BaselineAdapter:
    """Common skeleton for all baseline Pipeline adapters.

    Concrete subclasses implement reset/ingest/retrieve/answer against their
    specific upstream (GAAMA SDK, REST API, etc.). The base class enforces
    the Pipeline protocol shape and supplies consistent not-ready errors.
    """

    name: str = "baseline"
    external_path: Path = field(default_factory=lambda: REPO_ROOT / "external")
    # Subclasses flip to True once they successfully connect to their upstream.
    ready: bool = False
    # Context from the most recent ingest — subclasses may override semantics.
    _last_ingested_conv_id: str | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Pipeline protocol (to be overridden — defaults surface a clear error)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop any in-memory / in-store state for the adapter."""
        self._last_ingested_conv_id = None

    def ingest(self, conversation: Conversation) -> None:
        raise PipelineNotReadyError(self._not_ready_hint("ingest"))

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        raise PipelineNotReadyError(self._not_ready_hint("retrieve"))

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        raise PipelineNotReadyError(self._not_ready_hint("answer"))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _not_ready_hint(self, op: str) -> str:
        hint = self._readiness_hint()
        return (
            f"[{self.name}] cannot {op}() — adapter is not ready. {hint} "
            "Phase F wires each baseline once the upstream is configured."
        )

    def _readiness_hint(self) -> str:
        """Override per adapter to describe exactly what needs to be set up
        for this baseline (API key, docker-compose, Python deps, etc.)."""
        return "Subclass did not override _readiness_hint."

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready,
            "external_path": str(self.external_path),
            "last_ingested_conv_id": self._last_ingested_conv_id,
        }
