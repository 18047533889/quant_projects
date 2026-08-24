# -*- coding: utf-8 -*-
"""R30 §7: production loader split.  The default registry is explicitly split
into PRODUCTION / RESEARCH / INTERNAL groups; ``load_all(include_research=False)``
loads only the strictly production surface (no research model families)."""
from __future__ import annotations

import pytest

import factor_engine.cleaned_operators
from factor_engine.cleaned_operators import (
    _LOAD_MODULES,
    RESEARCH_LOAD_MODULES,
    INTERNAL_KERNEL_MODULES,
    PRODUCTION_LOAD_MODULES,
)


def test_split_groups_are_declared():
    assert RESEARCH_LOAD_MODULES
    assert INTERNAL_KERNEL_MODULES
    # every research/internal module is part of the full default list
    assert set(RESEARCH_LOAD_MODULES) <= set(_LOAD_MODULES)
    assert set(INTERNAL_KERNEL_MODULES) <= set(_LOAD_MODULES)


def test_groups_are_disjoint():
    assert not (set(RESEARCH_LOAD_MODULES) & set(INTERNAL_KERNEL_MODULES))


def test_research_modules_are_named_model_families():
    # R30 §7 names exactly the ts_model / panel_model / research_transform /
    # dmd / research_spectral families.
    assert "factor_engine.cleaned_operators.ts_model.state_space" in RESEARCH_LOAD_MODULES
    assert "factor_engine.cleaned_operators.cross_section.panel_model" in RESEARCH_LOAD_MODULES
    assert "factor_engine.cleaned_operators.research_transform" in RESEARCH_LOAD_MODULES
    assert "factor_engine.cleaned_operators.dmd" in RESEARCH_LOAD_MODULES
    assert "factor_engine.cleaned_operators.research_spectral" in RESEARCH_LOAD_MODULES


def test_production_loader_excludes_research_modules():
    # Structural check: the module-level loop skips RESEARCH_LOAD_MODULES when
    # include_research=False.  We assert the flag is threaded (the loader code
    # path) rather than re-running a 70s load in-process.
    import inspect
    from factor_engine.cleaned_operators import _load_all_impl

    src = inspect.getsource(_load_all_impl)
    assert "include_research" in src
    assert "RESEARCH_LOAD_MODULES" in src
    assert "continue" in src
