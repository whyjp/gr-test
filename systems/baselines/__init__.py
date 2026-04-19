"""Baseline Pipeline adapters for external systems.

Each adapter wraps one cloned system in ``external/`` to satisfy the
``htb.eval.interfaces.Pipeline`` protocol so MultiSystemRunner can
compare them under identical conditions.

Current status: **stubs**. Adapters are Pipeline-shaped and import cleanly
(offline-safe) but raise ``PipelineNotReadyError`` in ingest / retrieve /
answer until API credentials + upstream deps are wired in Phase F.

Adapters:
- GaamaAdapter          — external/gaama
- HyperGraphRAGAdapter  — external/hypergraph_rag
- HippoRAGAdapter       — external/hipporag (HippoRAG 2)
- HyperMemAdapter       — external/everos/methods/HyperMem
- EverMemOSAdapter      — external/everos/methods/evermemos (REST at :1995)
"""

from htb.data.locomo import Conversation
from systems.baselines.base import BaselineAdapter, PipelineNotReadyError
from systems.baselines.evermemos_adapter import EverMemOSAdapter
from systems.baselines.gaama_adapter import GaamaAdapter
from systems.baselines.hipporag_adapter import HippoRAGAdapter
from systems.baselines.hypergraph_rag_adapter import HyperGraphRAGAdapter
from systems.baselines.hypermem_adapter import HyperMemAdapter

__all__ = [
    "BaselineAdapter",
    "Conversation",
    "EverMemOSAdapter",
    "GaamaAdapter",
    "HippoRAGAdapter",
    "HyperGraphRAGAdapter",
    "HyperMemAdapter",
    "PipelineNotReadyError",
]
