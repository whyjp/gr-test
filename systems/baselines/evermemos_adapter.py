"""EverMemOS adapter (stub) — REST client at localhost:1995."""

from __future__ import annotations

import os
from dataclasses import dataclass

from systems.baselines.base import REPO_ROOT, BaselineAdapter


@dataclass
class EverMemOSAdapter(BaselineAdapter):
    name: str = "evermemos"
    api_base_url: str = "http://localhost:1995"
    api_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.external_path = (
            REPO_ROOT / "external" / "everos" / "methods" / "evermemos"
        )
        env_url = os.environ.get("EVERMEMOS_API_URL")
        if env_url:
            self.api_base_url = env_url

    def _readiness_hint(self) -> str:
        return (
            "To wire: (1) bring up docker-compose at "
            "external/everos/methods/evermemos via scripts/evermemos-up.sh "
            "(MongoDB + ES + Milvus + Redis); (2) configure "
            "methods/evermemos/.env with OPENAI_API_KEY + VECTORIZE_API_KEY; "
            "(3) `make run` inside that directory to start the REST server on "
            "port 1995; (4) override ingest() to POST /api/v1/memories and "
            "retrieve() to POST /api/v1/memories/search. Pipeline maps one "
            "LoCoMo conversation to a single tenant+agent scope (per the "
            "run_create_ltm.py pattern)."
        )
