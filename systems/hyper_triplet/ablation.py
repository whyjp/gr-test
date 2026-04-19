"""Ablation presets — 11 named HyperTripletConfig variants.

Per plan v5 Phase E, each preset flips specific HINGE / plan-v5 invariants
off (or, uniquely for `gaama_style_reflection_on`, flips Invariant #8 on in
the WRONG direction — enabling an encoder-style reflection pass to measure
the ceiling loss).

All presets inherit from BASELINE (the default HyperTripletConfig) and
override only the fields that differ, so future config extensions
automatically cascade.

HINGE invariant mapping (docs/hinge-north-star.md §6):

| Preset | Invariant tested |
|---|---|
| no_node_set | #1 atomicity |
| no_layer_separation | spec-layer principle |
| no_hyper_edge | #3 qualifier typing |
| no_star_storage | #6 first-class qualifiers in storage |
| no_stage1 | spec pipeline stage-1 |
| no_hybrid_index | retrieval signal combination |
| no_community | L3 high-level grouping |
| no_importance | L1 temporal/importance layer |
| no_boundary_detector | boundary-aware ingest |
| no_ontology_axis | grouping-principle 3rd axis |
| gaama_style_reflection_on | #8 LLM-as-classifier-only (DELIBERATE violation) |
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from systems.hyper_triplet.config import DEFAULT_CONFIG, HyperTripletConfig
from systems.hyper_triplet.retrieval_stages import Stage1Broad

ABLATION_NAMES: tuple[str, ...] = (
    "baseline",
    "no_node_set",
    "no_layer_separation",
    "no_hyper_edge",
    "no_star_storage",
    "no_stage1",
    "no_hybrid_index",
    "no_community",
    "no_importance",
    "no_boundary_detector",
    "no_ontology_axis",
    "gaama_style_reflection_on",
)


@dataclass(slots=True, frozen=True)
class AblationPreset:
    name: str
    description: str
    config: HyperTripletConfig
    hinge_invariant_violated: str  # short label for reports


def _preset(name: str, description: str, invariant: str, **overrides) -> AblationPreset:
    return AblationPreset(
        name=name,
        description=description,
        config=DEFAULT_CONFIG.with_overrides(**overrides),
        hinge_invariant_violated=invariant,
    )


def all_presets() -> dict[str, AblationPreset]:
    """Return every ablation preset keyed by name."""
    presets: dict[str, AblationPreset] = {
        "baseline": AblationPreset(
            name="baseline",
            description="HINGE-faithful default — all invariants preserved.",
            config=DEFAULT_CONFIG,
            hinge_invariant_violated="none",
        ),
        "no_node_set": _preset(
            "no_node_set",
            "Independent triple extraction instead of atomic node_set bundle.",
            invariant="#1 atomicity",
            use_node_set=False,
        ),
        "no_layer_separation": _preset(
            "no_layer_separation",
            "Collapse L0/L1/L2/L3 into one flat Qualifiers bag.",
            invariant="spec layer principle",
            use_layer_separation=False,
        ),
        "no_hyper_edge": _preset(
            "no_hyper_edge",
            "Remove edge qualifiers; edges become plain (src, type, dst) triples.",
            invariant="#3 qualifier typing",
            use_hyper_edge=False,
        ),
        "no_star_storage": _preset(
            "no_star_storage",
            "Flat graph — no per-ns star index. O(n) retrieval per star.",
            invariant="#6 qualifiers first-class",
            use_star_storage=False,
        ),
        "no_stage1": AblationPreset(
            name="no_stage1",
            description="Skip Stage1Broad — every star enters Stage2Rank directly.",
            config=DEFAULT_CONFIG.with_overrides(use_stage1=False),
            hinge_invariant_violated="spec pipeline stage-1",
        ),
        "no_hybrid_index": _preset(
            "no_hybrid_index",
            "Stage1 uses fact text only, no context/community expansion.",
            invariant="retrieval signal combination",
            stage1=Stage1Broad(
                top_n=DEFAULT_CONFIG.stage1.top_n,
                include_fact_text=True,
                expand_via_community=False,
                min_overlap_tokens=DEFAULT_CONFIG.stage1.min_overlap_tokens,
            ),
            use_hybrid_index=False,
        ),
        "no_community": _preset(
            "no_community",
            "Skip community detection; stage1 expansion degrades to nothing.",
            invariant="L3 high-level grouping",
            use_community=False,
            stage1=replace(DEFAULT_CONFIG.stage1, expand_via_community=False),
        ),
        "no_importance": _preset(
            "no_importance",
            "Zero out Stage2Rank.importance_weight; ranking by temporal signal only.",
            invariant="L1 temporal/importance layer",
            use_importance=False,
            stage2=replace(DEFAULT_CONFIG.stage2, importance_weight=0.0),
        ),
        "no_boundary_detector": _preset(
            "no_boundary_detector",
            "Fixed-turn chunking; no entity-drift split inside a session.",
            invariant="boundary-aware ingest",
            use_boundary_detector=False,
        ),
        "no_ontology_axis": _preset(
            "no_ontology_axis",
            "Drop L3 ontology_type + ontology_properties; 2-axis grouping only.",
            invariant="grouping-principle 3rd axis",
            use_ontology_axis=False,
        ),
        "gaama_style_reflection_on": _preset(
            "gaama_style_reflection_on",
            "DELIBERATELY violate Invariant #8 by running a GAAMA-style "
            "reflection generator over already-extracted facts. Used to "
            "measure the encoder-style grouping ceiling loss (empirically "
            "~14pp on LoCoMo per grouping-node-principle.md §2).",
            invariant="#8 LLM-as-classifier-only (DELIBERATE violation)",
            use_gaama_style_reflection=True,
        ),
    }
    return presets


def get_preset(name: str) -> AblationPreset:
    """Return the named preset or raise KeyError."""
    presets = all_presets()
    if name not in presets:
        raise KeyError(
            f"Unknown ablation preset: {name}. Available: {sorted(presets.keys())}"
        )
    return presets[name]


def diff_from_baseline(preset: AblationPreset) -> dict[str, object]:
    """Return the subset of HyperTripletConfig fields that differ from baseline.

    Useful for compact reporting in ablation tables.
    """
    diff: dict[str, object] = {}
    for f in fields(DEFAULT_CONFIG):
        baseline_value = getattr(DEFAULT_CONFIG, f.name)
        preset_value = getattr(preset.config, f.name)
        if baseline_value != preset_value:
            diff[f.name] = preset_value
    return diff
