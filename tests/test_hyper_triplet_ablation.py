"""Tests for config.py + ablation.py preset machinery."""

from __future__ import annotations

import pytest

from systems.hyper_triplet.ablation import (
    ABLATION_NAMES,
    AblationPreset,
    all_presets,
    diff_from_baseline,
    get_preset,
)
from systems.hyper_triplet.config import DEFAULT_CONFIG, HyperTripletConfig


def test_default_config_is_frozen_and_composable():
    cfg = HyperTripletConfig()
    # Overrides return a new instance, not mutation
    cfg2 = cfg.with_overrides(use_community=False)
    assert cfg.use_community is True
    assert cfg2.use_community is False
    assert cfg is not cfg2


def test_retrieval_pipeline_mode_reflects_stage1_toggle():
    cfg = HyperTripletConfig()
    assert cfg.retrieval_pipeline_mode == "full"
    cfg_no_s1 = cfg.with_overrides(use_stage1=False)
    assert cfg_no_s1.retrieval_pipeline_mode == "no_stage1"


def test_all_presets_named():
    presets = all_presets()
    assert set(presets.keys()) == set(ABLATION_NAMES)
    assert len(ABLATION_NAMES) == 12  # baseline + 11 ablations


def test_baseline_preset_is_unmodified():
    baseline = get_preset("baseline")
    assert baseline.config == DEFAULT_CONFIG
    assert baseline.hinge_invariant_violated == "none"


def test_every_non_baseline_preset_differs_from_baseline():
    for name, preset in all_presets().items():
        if name == "baseline":
            continue
        diff = diff_from_baseline(preset)
        assert diff, f"preset {name!r} has no config diff from baseline"


def test_no_community_disables_use_community_and_expansion():
    preset = get_preset("no_community")
    assert preset.config.use_community is False
    assert preset.config.stage1.expand_via_community is False


def test_no_importance_zeros_stage2_importance_weight():
    preset = get_preset("no_importance")
    assert preset.config.use_importance is False
    assert preset.config.stage2.importance_weight == 0.0


def test_no_stage1_sets_pipeline_mode():
    preset = get_preset("no_stage1")
    assert preset.config.retrieval_pipeline_mode == "no_stage1"


def test_no_hybrid_index_disables_community_expansion():
    preset = get_preset("no_hybrid_index")
    assert preset.config.stage1.expand_via_community is False


def test_gaama_style_reflection_is_deliberate_violation():
    """Inv #8 is flipped on (WRONG direction) to measure ceiling loss."""
    preset = get_preset("gaama_style_reflection_on")
    assert preset.config.use_gaama_style_reflection is True
    # Everything else stays baseline
    assert preset.config.use_node_set is True
    assert preset.config.use_community is True
    assert "DELIBERATE" in preset.hinge_invariant_violated.upper()


def test_get_preset_raises_on_unknown():
    with pytest.raises(KeyError):
        get_preset("no_such_preset")


def test_diff_from_baseline_shape():
    diff = diff_from_baseline(get_preset("no_community"))
    assert "use_community" in diff
    assert diff["use_community"] is False


def test_presets_are_frozen_dataclasses():
    preset = get_preset("no_community")
    try:
        preset.name = "mutated"  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("AblationPreset should be frozen")


def test_every_preset_has_invariant_label():
    for name, preset in all_presets().items():
        assert isinstance(preset, AblationPreset)
        assert preset.hinge_invariant_violated
        assert isinstance(preset.description, str) and preset.description
        assert preset.name == name


def test_config_all_toggles_true_by_default_except_gaama_reflection():
    """Invariant #8 violation is off by default; everything else is on."""
    from dataclasses import fields as _fields

    cfg = DEFAULT_CONFIG
    flags = {
        f.name: getattr(cfg, f.name)
        for f in _fields(cfg)
        if f.name.startswith("use_")
    }
    assert flags["use_gaama_style_reflection"] is False
    for k, v in flags.items():
        if k == "use_gaama_style_reflection":
            continue
        assert v is True, f"default config has {k}={v}, expected True"
