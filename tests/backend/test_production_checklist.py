# -*- coding: utf-8
"""Production checklist 完整性 + winsorize 参数校验。"""
from __future__ import annotations

import pytest

from factor_engine.backend.aggregation_spec import sum_all_null_is_null
from factor_engine.backend.calendar_spec import calendar_ops_require_explicit_spec
from factor_engine.backend.count_semantics import canonical_count_is_expanding_non_null
from factor_engine.backend.cross_section_spec import is_effectively_zero_std, zscore_zero_std_epsilon
from factor_engine.backend.cumulative_state_spec import cumulative_batch_only_phase1
from factor_engine.backend.fill_semantics import ffill_production_requires_limit
from factor_engine.backend.group_key_spec import null_group_outputs_null
from factor_engine.backend.group_spec import group_std_singleton_is_null
from factor_engine.backend.lag_spec import ts_delay_is_row_delay
from factor_engine.backend.ordering_spec import (
    duplicate_ts_inst_keys_forbidden,
    production_output_sorted_by_ts_inst,
)
from factor_engine.backend.pairwise_alignment import pairwise_alignment_contract_wired
from factor_engine.backend.pairwise_spec import beta_uses_pairwise_var
from factor_engine.backend.plan_params import PlanParamError, parse_winsorize_quantiles
from factor_engine.backend.production_checklist import PRODUCTION_CHECKLIST, checklist_complete_for_phase1
from factor_engine.backend.production_signature import param_allowed
from factor_engine.backend.universe_spec import empty_universe_preserves_keys
from factor_engine.planner.logical_plan import PlanNode


def test_production_checklist_phase1_tracks_wiring():
    from factor_engine.backend.production_checklist import PHASE1_COMPLETE_STATUSES, checklist_summary

    summary = checklist_summary()
    assert summary.get("defined", 0) > 0
    assert "gate_enforced" in PHASE1_COMPLETE_STATUSES
    assert "parity_tested" in PHASE1_COMPLETE_STATUSES


def test_core_contract_flags():
    assert ts_delay_is_row_delay()
    assert duplicate_ts_inst_keys_forbidden()
    assert production_output_sorted_by_ts_inst()
    assert sum_all_null_is_null()
    assert beta_uses_pairwise_var()
    assert null_group_outputs_null()
    assert group_std_singleton_is_null()
    assert canonical_count_is_expanding_non_null()
    assert empty_universe_preserves_keys()
    assert calendar_ops_require_explicit_spec()
    assert cumulative_batch_only_phase1()
    assert ffill_production_requires_limit()
    assert pairwise_alignment_contract_wired()


def test_pairwise_alignment_contract_flags():
    from factor_engine.backend.pairwise_alignment import AlignmentMode, pairwise_alignment_spec_for

    for canonical in ("ts_corr", "ts_cov", "ts_beta", "ts_regression_slope"):
        spec = pairwise_alignment_spec_for(canonical)
        assert spec.mode == AlignmentMode.EXACT_ALIGNMENT
        assert spec.date_axis is True
        assert spec.instrument_axis is True
        assert spec.universe_snapshot is True
        assert "date_axis" in spec.checked_axes()
        assert "universe_snapshot" in spec.checked_axes()


def test_zscore_zero_std_epsilon():
    assert is_effectively_zero_std(0.0)
    assert is_effectively_zero_std(1e-15, scale=1.0)
    assert not is_effectively_zero_std(1e-6, scale=1.0)
    assert zscore_zero_std_epsilon(scale=1e6) >= 1e-2


@pytest.mark.parametrize(
    "lo,hi,err",
    [
        (0.95, 0.05, "不能大于"),
        (-0.1, 0.9, "必须 >="),
        (0.5, 1.5, "必须 <="),
        (0.5, 0.5, "不能相等"),
    ],
)
def test_winsorize_invalid_params_rejected(lo, hi, err):
    node = PlanNode(
        op="winsorize",
        inputs=[PlanNode(op="column", inputs=[], attrs={"name": "x"})],
        attrs={"lower": lo, "upper": hi},
    )
    with pytest.raises(PlanParamError, match=err):
        parse_winsorize_quantiles(node)


def test_winsorize_symmetric_a():
    node = PlanNode(
        op="winsorize",
        inputs=[PlanNode(op="column", inputs=[], attrs={"name": "x"})],
        attrs={"a": 0.05},
    )
    lo, hi = parse_winsorize_quantiles(node)
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(0.95)


def test_winsorize_single_positional_is_lower_bound():
    node = PlanNode(
        op="winsorize",
        inputs=[
            PlanNode(op="column", inputs=[], attrs={"name": "x"}),
            PlanNode(op="literal", inputs=[], attrs={"value": 0.25}),
        ],
        attrs={},
    )
    lo, hi = parse_winsorize_quantiles(node)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.95)


def test_scale_signature_finite_only():
    assert param_allowed("scale", "to", value=1.0) == "production"
    assert param_allowed("scale", "to", value=float("nan")) == "forbidden"
    assert param_allowed("bfill", "method") == "forbidden"
