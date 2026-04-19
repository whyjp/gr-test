"""Central configuration for the Hyper Triplet pipeline.

Per plan v5 Phase E and `my-own-test-design-spec.md` §10.2, every
hyperparameter lives here — no magic numbers in module bodies.
Composes the per-module configs (Boundary / Importance / Stage1 /
Stage2 / Stage3 / Community / PPR) plus 11 principle-level ablation
toggles derived from plan v5 + `grouping-principle-integration.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from systems.hyper_triplet.boundary_detector import BoundaryConfig
from systems.hyper_triplet.community_detector import CommunityConfig
from systems.hyper_triplet.importance_scorer import ImportanceConfig
from systems.hyper_triplet.retrieval_ppr import PPRConfig
from systems.hyper_triplet.retrieval_stages import (
    Stage1Broad,
    Stage2Rank,
    Stage3Exact,
)


@dataclass(slots=True, frozen=True)
class HyperTripletConfig:
    """Top-level config composing every hyperparameter and every ablation flag.

    All config dataclasses are frozen / immutable so sharing one instance
    across systems in a MultiSystemRunner is safe.
    """

    # --- per-module configs ---
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    stage1: Stage1Broad = field(default_factory=Stage1Broad)
    stage2: Stage2Rank = field(default_factory=Stage2Rank)
    stage3: Stage3Exact = field(default_factory=Stage3Exact)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    ppr: PPRConfig = field(default_factory=PPRConfig)

    # --- principle-level ablation toggles (plan v5 Phase E + Inv #8) ---
    use_node_set: bool = True
    use_layer_separation: bool = True
    use_hyper_edge: bool = True
    use_star_storage: bool = True
    use_stage1: bool = True
    use_hybrid_index: bool = True
    use_community: bool = True
    use_importance: bool = True
    use_boundary_detector: bool = True
    use_ontology_axis: bool = True
    # HINGE invariant #8 explicitly violated when True — for ablation only,
    # never in production. Enables an extra reflection-generation LLM call
    # whose output is stored as authoritative memory (GAAMA-style).
    use_gaama_style_reflection: bool = False

    def with_overrides(self, **kwargs) -> HyperTripletConfig:
        """Return a new config with the given fields overridden.

        Convenience for ablation presets and for MultiSystemRunner variants.
        """
        return replace(self, **kwargs)

    @property
    def retrieval_pipeline_mode(self) -> str:
        if not self.use_stage1:
            return "no_stage1"
        return "full"


DEFAULT_CONFIG = HyperTripletConfig()
