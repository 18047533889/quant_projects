# -*- coding: utf-8
"""Composite lowering：高级算子 → 基础 DAG。"""
from __future__ import annotations

import pytest

from factor_engine.planner.composite_lowering import (
    collect_plan_ops,
    composite_dual_backend_capable,
    lower_composite_operators,
    lowered_primitives,
    register_lowering,
)
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.optimizer import Optimizer


def _col(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def test_mom_lowers_to_ts_delta():
    plan = PlanNode(op="MOM", inputs=[_col()], attrs={"window": 5})
    out = lower_composite_operators(plan)
    assert out.op == "ts_delta"
    assert out.inputs[0].op == "column"


def test_roc_lowers_to_ts_pct_times_100():
    plan = PlanNode(op="ROC", inputs=[_col()], attrs={"window": 3})
    out = lower_composite_operators(plan)
    assert out.op == "multiply"
    assert out.inputs[0].op == "ts_pct"
    assert out.inputs[1].op == "literal"
    assert out.inputs[1].attrs["value"] == 100.0


def test_bollinger_upper_lowers_to_mean_std():
    plan = PlanNode(
        op="BollingerUpper",
        inputs=[_col(), PlanNode(op="literal", attrs={"value": 20}, inputs=[]), PlanNode(op="literal", attrs={"value": 2.0}, inputs=[])],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "add"
    assert out.inputs[0].op == "ts_mean"
    assert out.inputs[1].op == "multiply"
    assert out.inputs[1].inputs[0].op == "literal"
    assert out.inputs[1].inputs[1].op == "ts_std"


def test_bollinger_lower_lowers_to_subtract():
    plan = PlanNode(op="BollingerLower", inputs=[_col()], attrs={"window": 10, "std_dev": 1.5})
    out = lower_composite_operators(plan)
    assert out.op == "subtract"
    assert out.inputs[0].op == "ts_mean"


def test_dpo_lowers_with_shifted_mean():
    plan = PlanNode(
        op="DPO",
        inputs=[_col(), PlanNode(op="literal", attrs={"value": 20}, inputs=[])],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "subtract"
    delayed = out.inputs[1]
    assert delayed.op == "ts_delay"
    assert delayed.inputs[1].op == "literal"
    assert delayed.inputs[1].attrs["value"] == 11


def test_optimizer_applies_composite_before_rewrite():
    col = _col()
    plan = PlanNode(op="MOM", inputs=[col], attrs={"window": 2})
    out = Optimizer().optimize(plan)
    assert out.op == "ts_delta"


def test_lowered_primitives_for_mom():
    prims = lowered_primitives("MOM")
    assert prims is not None
    assert prims == ("ts_delta",)


def test_composite_dual_backend_capable_mom():
    from factor_engine.cleaned_operators import load_all

    load_all()
    # MOM lowers to ts_delta, but capability is evidence-backed: the current
    # primitive evidence does not certify ts_delta for both production backends.
    assert lowered_primitives("MOM") == ("ts_delta",)
    assert composite_dual_backend_capable("MOM") is False


def test_collect_plan_ops_after_lowering():
    plan = PlanNode(op="ROC", inputs=[_col()], attrs={"window": 1})
    lowered = lower_composite_operators(plan)
    ops = collect_plan_ops(lowered)
    assert "ts_pct" in ops
    assert "multiply" in ops
    assert "ROC" not in ops


def test_obv_lowers_with_coalesce_on_sign():
    plan = PlanNode(
        op="OBV",
        inputs=[_col("close"), _col("volume")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "cum_sum"
    inner = out.inputs[0]
    assert inner.op == "multiply"
    assert inner.inputs[0].op == "fillna_const"


def test_obv_lowers_to_cum_sum_chain():
    plan = PlanNode(
        op="OBV",
        inputs=[_col("close"), _col("volume")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "cum_sum"
    inner = out.inputs[0]
    assert inner.op == "multiply"
    assert inner.inputs[0].op == "fillna_const"
    assert inner.inputs[0].inputs[0].op == "sign"
    assert inner.inputs[0].inputs[0].inputs[0].op == "ts_delta"


def test_stochastic_k_lowers_to_min_max_div():
    plan = PlanNode(
        op="StochasticK",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={"window": 14},
    )
    out = lower_composite_operators(plan)
    assert out.op == "multiply"
    assert out.inputs[0].op == "safe_div_null"


def test_stochastic_d_lowers_to_mean_of_k():
    plan = PlanNode(
        op="StochasticD",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={"window": 14},
    )
    out = lower_composite_operators(plan)
    assert out.op == "ts_mean"
    assert out.inputs[0].op == "multiply"


def test_bollinger_bands_lowers_to_ts_mean():
    plan = PlanNode(op="BollingerBands", inputs=[_col()], attrs={"window": 20})
    out = lower_composite_operators(plan)
    assert out.op == "ts_mean"


def test_operating_margin_lowers_to_protected_div():
    plan = PlanNode(
        op="operating_margin",
        inputs=[_col("oi"), _col("rev")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"


def test_quick_ratio_lowers_to_subtract_and_div():
    plan = PlanNode(
        op="quick_ratio",
        inputs=[_col("ca"), _col("inv"), _col("cl")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"
    assert out.inputs[0].op == "subtract"


def test_micro_spread_lowers_to_protected_div():
    plan = PlanNode(
        op="micro_spread",
        inputs=[_col("high"), _col("low"), _col("close")],
        attrs={},
    )
    out = lower_composite_operators(plan)
    assert out.op == "safe_div_null"
    assert out.inputs[0].op == "subtract"


def test_list_composite_lowerings_batch_one_size():
    from factor_engine.planner.composite_lowering import list_composite_lowerings

    names = list_composite_lowerings()
    assert len(names) >= 16
    for name in (
        "MOM",
        "ROC",
        "OBV",
        "StochasticK",
        "StochasticD",
        "BollingerBands",
        "operating_margin",
        "micro_spread",
    ):
        assert name in names


def test_removed_obv_is_not_claimed_dual_backend_capable():
    from factor_engine.cleaned_operators import load_all

    load_all()
    assert composite_dual_backend_capable("OBV") is False


# ---------------------------------------------------------------------------
# R6 P0-04 / P0-05 / P0-06 composite-lowering hardening
# ---------------------------------------------------------------------------

def test_lowered_primitives_macd_fires_ema_branches():
    # The static probe must use declared deps (fast/slow/signal) with distinct
    # representative values so MACD's fast < slow branch actually fires.
    prims = lowered_primitives("MACD_line")
    assert prims is not None
    assert "ts_ema" in prims


def test_lowered_primitives_bollinger_reads_std_dev():
    prims = lowered_primitives("BollingerUpper")
    assert prims is not None
    assert "ts_std" in prims


def test_declared_parameter_defaults_drive_probe_not_fake_stub():
    # P0-06: probe uses the composite's registered defaults / declared deps —
    # MACD_signal must expand to the EMA-of-EMA chain, not a fabricated window=3.
    prims = lowered_primitives("MACD_signal")
    assert "ts_ema" in prims


def test_duplicate_lowering_rejected_with_both_sources():
    from factor_engine.planner.composite_lowering import (
        _LOWERING_SOURCES,
        CompositeLoweringDuplicateError,
    )

    # Re-registering an existing composite without a declared replacement fails
    # and names the original source.
    with pytest.raises(CompositeLoweringDuplicateError):

        @register_lowering("MOM")
        def _dup_mom(node):
            return node

    assert _LOWERING_SOURCES.get("MOM") == "factor_engine.planner.lowerings.technical"


def test_declared_replacement_authorises_reregister():
    from factor_engine.planner.composite_lowering import declare_lowering_replacement

    declare_lowering_replacement("__r6_test_canonical__", "test override")
    register_lowering("__r6_test_canonical__")(lambda node: node)
    register_lowering("__r6_test_canonical__")(lambda node: node)  # second OK


def test_optimizer_rejects_fractional_window_before_lowering():
    # P0-04: Composite(window=5.9) must be rejected at planning, never lowered to
    # ts_mean(x, 5).  Use a pure PlanNode so no DSL/field plumbing is needed.
    plan = PlanNode(op="MOM", inputs=[_col()], attrs={"window": 5.9})
    from factor_engine.planner.optimizer import Optimizer

    with pytest.raises(Exception):
        Optimizer().optimize(plan, production=True)


def test_optimizer_accepts_integral_window_unchanged():
    plan = PlanNode(op="MOM", inputs=[_col()], attrs={"window": 5})
    out = Optimizer().optimize(plan)
    assert out.op == "ts_delta"


def test_strict_float_rejects_bool_and_nan_params():
    from factor_engine.planner.lowerings import _helpers as H

    with pytest.raises(ValueError):
        H.strict_float(True, "std_dev")
    with pytest.raises(ValueError):
        H.strict_float(float("nan"), "std_dev")
    with pytest.raises(ValueError):
        H.strict_float(float("inf"), "std_dev")
    assert H.strict_float(2.5, "std_dev") == 2.5


def test_canonicalize_parameter_values_normalises_proportional_weights():
    """R13 NEW-P0-18: weight normalization is gated on the operator's declared
    ``ParamSpec.equivalence == "positive_scale"`` — never inferred from the name.

    Without a canonical context (no declared equivalence) a ``weights`` vector is
    left verbatim; with the declaration, proportional vectors collapse to one
    unit-sum key.
    """
    from factor_engine.planner.canonicalize_params import canonicalize_parameter_values
    from types import SimpleNamespace

    # No declared equivalence -> no name-based normalization (R13 NEW-P0-18).
    a = canonicalize_parameter_values({"weights": [1.0, 1.0, 1.0]})
    b = canonicalize_parameter_values({"weights": [0.1, 0.1, 0.1]})
    assert a != b
    assert a == {"weights": (1.0, 1.0, 1.0)}

    # Declared equivalence="positive_scale" -> unit-sum normalized hash key.
    import factor_engine.planner.canonicalize_params as cmod

    def _fake_contract(canon):
        spec = SimpleNamespace(equivalence="positive_scale")
        meta = SimpleNamespace(param_specs={"weights": spec}, param_names=["weights"])
        return (object(), meta, meta.param_specs, {})

    original = cmod._operator_contract
    cmod._operator_contract = _fake_contract
    try:
        x = canonicalize_parameter_values({"weights": [1.0, 1.0, 1.0]}, canonical="op")
        y = canonicalize_parameter_values({"weights": [0.1, 0.1, 0.1]}, canonical="op")
    finally:
        cmod._operator_contract = original
    assert x == y
    assert abs(sum(x["weights"]) - 1.0) < 1e-9

    # R13 NEW-P0-19: a declared scale-equivalent vector with a non-numeric
    # element is REJECTED whole, never silently filtered to a shorter vector.
    import pytest
    from factor_engine.backend.operator_errors import OperatorParameterError

    cmod._operator_contract = _fake_contract
    try:
        with pytest.raises(OperatorParameterError, match="non-numeric"):
            canonicalize_parameter_values(
                {"weights": [1.0, "bad", 2.0]}, canonical="op"
            )
    finally:
        cmod._operator_contract = original
