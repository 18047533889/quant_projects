# -*- coding: utf-8 -*-
"""R10 #1/#3 regression tests.

``DataSourceBuildContext`` must thread the engine's run-mode / production /
mining policy into every DataAccessSource built by ``storage.factory``
(recursively through Composite / LongTable / Intraday children), so a production
run gets the hard gates ON without the YAML author remembering them per-source.
And (R10 #3) an explicit ``strict_unknown_fields=False`` must NEVER downgrade a
production run back to fail-open — production is a floor.
"""
from __future__ import annotations

import pytest

from factor_engine.storage.factory import DataSourceBuildContext, build_data_source


def _ctx(**kw) -> DataSourceBuildContext:
    return DataSourceBuildContext(**kw)


def test_research_context_leaves_source_research():
    src = build_data_source(
        {"type": "data_access", "dataset": "test_daily"},
        build_context=_ctx(run_mode="research"),
    )
    assert src.production is False
    assert src.strict_unknown_fields is False
    assert src.enforce_mining_gate is False


def test_production_context_injects_gates():
    src = build_data_source(
        {"type": "data_access", "dataset": "test_daily"},
        build_context=_ctx(
            run_mode="production",
            enforce_mining_gate=True,
            snapshot_policy="snapshot_now_only",
            coverage_policy=0.7,
        ),
    )
    assert src.production is True
    assert src.strict_unknown_fields is True
    assert src.enforce_mining_gate is True
    assert src.snapshot_now_only is True
    assert src.mining_coverage_threshold == 0.7


def test_production_is_a_floor_explicit_false_cannot_downgrade():
    # R10 #3: even an explicit strict_unknown_fields=False under a production
    # run must stay fail-closed.
    src = build_data_source(
        {
            "type": "data_access",
            "dataset": "test_daily",
            "strict_unknown_fields": False,
        },
        build_context=_ctx(run_mode="production"),
    )
    assert src.production is True
    assert src.strict_unknown_fields is True


def test_research_explicit_strict_can_go_stricter():
    src = build_data_source(
        {"type": "data_access", "dataset": "test_daily"},
        build_context=_ctx(run_mode="research"),
    )
    # explicit strict at the SOURCE level (not context) tightens research.
    src2 = build_data_source(
        {"type": "data_access", "dataset": "test_daily", "strict_unknown_fields": True},
        build_context=_ctx(run_mode="research"),
    )
    assert src2.strict_unknown_fields is True
    assert src2.production is True or src2.strict_unknown_fields  # direct gate on


def test_per_source_run_mode_override():
    # a source may override to research even under a production context
    src = build_data_source(
        {"type": "data_access", "dataset": "test_daily", "run_mode": "research"},
        build_context=_ctx(run_mode="production"),
    )
    # production context propagates production by default, but a per-source
    # run_mode research + production=None keeps production off.
    assert src.run_mode == "research"
    assert src.production is False


def test_composite_children_inherit_production_context():
    src = build_data_source(
        {
            "type": "composite",
            "anchor": "prices",
            "anchor_column": "close",
            "sources": {
                "prices": {"type": "data_access", "dataset": "test_daily"},
                "fund": {
                    "type": "long_table",
                    "source": {"type": "data_access", "dataset": "test_valuation"},
                },
            },
        },
        build_context=_ctx(run_mode="production", enforce_mining_gate=True),
    )
    # composite children are DataAccessSource (or wrapped long-table) instances
    def _collect(s, out):
        inner = getattr(s, "inner", None)
        if inner is not None and inner is not s:
            _collect(inner, out)
        if type(s).__name__ == "DataAccessSource" or hasattr(s, "strict_unknown_fields"):
            out.append(s)

    sources = []
    for name, child in src.sources.items():
        _collect(child, sources)
    assert sources, "expected child sources"
    for child in sources:
        assert child.production is True, f"{child.dataset} production"
        assert child.strict_unknown_fields is True, f"{child.dataset} strict"
        assert child.enforce_mining_gate is True, f"{child.dataset} mining gate"


def test_unknown_option_still_rejected():
    with pytest.raises(ValueError, match="Unsupported options"):
        build_data_source(
            {"type": "data_access", "dataset": "test_daily", "bogus_key": 1},
            build_context=_ctx(run_mode="production"),
        )
