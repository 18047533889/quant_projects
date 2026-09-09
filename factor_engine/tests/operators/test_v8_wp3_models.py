import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.optimizer import Optimizer
from factor_engine.planner.canonicalize_params import canonicalize_parameter_values

def col(name):
    return PlanNode("column", attrs={"name": name})


def test_entropy_uses_one_extendable_template_set():
    from factor_engine.cleaned_operators.sequence_complexity import _sample_entropy
    from factor_engine.cleaned_operators.ts_model.complexity import _pseudocount_sample_entropy
    from factor_engine.cleaned_operators.ts_model.complexity import _sample_entropy as old_sample_entropy
    values = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    tolerance = 0.2 * np.std(values)
    assert _sample_entropy(values, 2, tolerance) == pytest.approx(0.0)
    assert old_sample_entropy(values, 2, 0.2, 6) == pytest.approx(0.0)
    assert _pseudocount_sample_entropy(values, 2, 0.2, 6) == pytest.approx(0.0)


def test_entropy_boundary_policies():
    from factor_engine.cleaned_operators.sequence_complexity import _sample_entropy
    from factor_engine.cleaned_operators.ts_model.complexity import _pseudocount_sample_entropy

    assert np.isnan(_sample_entropy(np.arange(3.0), 2, 1.0))  # no extendable pair (B=0)
    assert np.isnan(_sample_entropy(np.array([0., 0., 1., 0.]), 1, 0.0))  # B>0, A=0
    assert np.isnan(_sample_entropy(np.ones(6), 2, -1.0))
    assert np.isnan(_pseudocount_sample_entropy(np.ones(6), 2, 0.2, 6))
    assert np.isnan(_pseudocount_sample_entropy(np.array([0., 1., 2., 3.]), 2, 0.2, 4))


def test_entropy_public_winner_signature():
    op = OperatorRegistry.get("ts_sample_entropy")
    assert op.__class__.__module__.endswith("sequence_complexity")
    assert op.metadata.param_names == [
        "x", "window", "embedding_dim", "tolerance_scale", "min_periods"
    ]


def test_wp3_catalog_versions_follow_central_authority_after_bootstrap():
    from factor_engine.backend.operator_semantic_version import OPERATOR_SEMANTIC_VERSIONS
    from factor_engine.cleaned_operators.layer_governance import _canonical_semantic_version
    from factor_engine.cleaned_operators.ts_model import ar_meanrev, dynamic_regression

    affected = {
        "ts_sample_entropy", "ts_pseudocount_sample_entropy",
        "ts_market_liquidity_beta", "ts_industry_liquidity_beta",
            *ar_meanrev._AR_CONFIGURED_HISTORY_CANONICALS,
            *dynamic_regression._MULTI_CONFIGURED_HISTORY_CANONICALS,
    }
    assert len(affected) == 34
    for canonical in affected:
        assert OperatorRegistry._catalog[canonical]["semantic_version"] == f"{OPERATOR_SEMANTIC_VERSIONS[canonical]}.0"
    assert _canonical_semantic_version("ts_sample_entropy", "1.0") == "2.0"
    assert _canonical_semantic_version("unknown_custom_operator", "7.5") == "7.5"


def test_optimizer_rejects_impossible_ar():
    plan = PlanNode("ts_ar_forecast", inputs=(col("close"),), attrs={"window": 3, "order": 1})
    with pytest.raises(OperatorParameterError, match="INSUFFICIENT_CONFIGURED_HISTORY"):
        Optimizer().lower_only(plan)


def test_optimizer_rejects_impossible_multi_regression():
    plan = PlanNode(
        "ts_multi_regression_r2", inputs=(col("y"), col("x")),
        attrs={"window": 4, "min_periods": 1, "add_intercept": True},
    )
    with pytest.raises(OperatorParameterError, match="INSUFFICIENT_CONFIGURED_HISTORY"):
        Optimizer().lower_only(plan)


@pytest.mark.parametrize("window,order,valid", [(3, 1, False), (4, 1, True), (7, 3, False), (8, 3, True)])
def test_ar_configured_history_boundary(window, order, valid):
    from factor_engine.cleaned_operators.ts_model.ar_meanrev import validate_ar_configured_history
    if valid:
        assert validate_ar_configured_history(window, order) == window
    else:
        with pytest.raises(ValueError, match="INSUFFICIENT_CONFIGURED_HISTORY"):
            validate_ar_configured_history(window, order)


@pytest.mark.parametrize("window,valid", [(9, False), (10, True)])
def test_multi_configured_history_boundary(window, valid):
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import validate_multi_configured_history
    if valid:
        assert validate_multi_configured_history(window, 1, 1, True) == window
    else:
        with pytest.raises(ValueError, match="INSUFFICIENT_CONFIGURED_HISTORY"):
            validate_multi_configured_history(window, 1, 1, True)


def test_noncoeff_output_ignores_selector():
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": np.arange(20.0)}, index=idx)
    y = 2.0 * x
    op = OperatorRegistry.get("ts_multi_regression_r2")
    out = op.calculate(
        y, x, window=10, coefficient_index=999, min_periods=1,
        add_intercept=False,
    )
    assert out["A"].iloc[-1] == pytest.approx(1.0)
    assert op.metadata.deprecated_ignored_params == ("coefficient_index",)

    positional = op.calculate(y, x, None, None, None, 10, 999, 1, False)
    assert positional["A"].iloc[-1] == pytest.approx(1.0)
    with pytest.raises(OperatorParameterError):
        op.calculate(y, x, window=10, unknown_selector=1)


def test_noncoeff_selector_is_absent_from_effective_identity():
    left = canonicalize_parameter_values(
        {"window": 20, "coefficient_index": 0}, canonical="ts_multi_regression_r2"
    )
    right = canonicalize_parameter_values(
        {"window": 20, "coefficient_index": 999}, canonical="ts_multi_regression_r2"
    )
    assert left == right == {"window": 20}
    assert canonicalize_parameter_values(
        {"window": 20, "coefficient_index": 0}, canonical="ts_multi_regression_coeff"
    ) != canonicalize_parameter_values(
        {"window": 20, "coefficient_index": 1}, canonical="ts_multi_regression_coeff"
    )


def test_polars_liquidity_beta_high_offset_matches_centered_reference():
    increments = 0.02 + np.sin(np.arange(100.0)) * 0.005
    liquidity = 1e12 + np.cumsum(increments)
    observed_delta = np.diff(liquidity, prepend=np.nan)
    own_return = 3.0 * observed_delta
    y = pl.DataFrame({"A": own_return})
    x = pl.DataFrame({"A": liquidity})
    op = OperatorRegistry.get("ts_market_liquidity_beta", backend="polars")

    result = op.calculate(y, x, window=60)["A"][-1]

    xv, yv = observed_delta[-60:], own_return[-60:]
    xc, yc = xv - np.mean(xv), yv - np.mean(yv)
    reference = np.dot(xc, yc) / np.dot(xc, xc)
    assert result == pytest.approx(reference, abs=1e-10)
    assert "native" in op.metadata.tags
    assert op._physical_spec.execution_kind.value == "polars_native_expr"


def test_polars_liquidity_beta_pairwise_finite_zero_variance_and_axes():
    n = 40
    delta_a = np.linspace(1e-9, 4e-9, n)
    level_a = 1e6 + np.cumsum(delta_a)
    observed_a = np.diff(level_a, prepend=np.nan)
    liquidity = pl.DataFrame({
        "A": level_a,
        "B": 1e6 + 2.0 * np.arange(n),
    })
    returns = pl.DataFrame({"A": 2 * observed_a, "B": np.arange(n, dtype=float)})
    returns = returns.with_columns(pl.when(pl.arange(0, n) == 25).then(None).otherwise(pl.col("A")).alias("A"))
    op = OperatorRegistry.get("ts_market_liquidity_beta", backend="polars")
    out = op.calculate(returns, liquidity, window=20)
    assert out.columns == ["A", "B"]
    assert out.height == n
    assert out["A"][-1] == pytest.approx(2.0, rel=1e-6)
    assert out["B"][-1] is None or np.isnan(out["B"][-1])


def test_physical_spec_accepts_contract_instances_across_module_reload():
    import importlib
    from types import SimpleNamespace

    import factor_engine.backend.contracts as contracts
    from factor_engine.backend.polars_backend_kind import (
        PolarsImplementationKind, polars_backend_kind,
    )

    before_reload = contracts.PhysicalImplementationSpec(
        canonical="probe", backend="polars",
        execution_kind=contracts.ExecutionKind.POLARS_NATIVE_EXPR,
    )
    reloaded = importlib.reload(contracts)
    after_reload = reloaded.PhysicalImplementationSpec(
        canonical="probe", backend="polars",
        execution_kind=reloaded.ExecutionKind.POLARS_NATIVE_EXPR,
    )

    assert polars_backend_kind(SimpleNamespace(_physical_spec=before_reload)) is PolarsImplementationKind.POLARS_NATIVE
    assert polars_backend_kind(SimpleNamespace(_physical_spec=after_reload)) is PolarsImplementationKind.POLARS_NATIVE


def test_physical_spec_rejects_lookalikes_and_unrelated_contracts():
    from dataclasses import fields, make_dataclass
    from types import SimpleNamespace

    import factor_engine.backend.contracts as contracts
    from factor_engine.backend.polars_backend_kind import get_physical_spec

    genuine = contracts.PhysicalImplementationSpec(
        canonical="probe", backend="polars",
        execution_kind=contracts.ExecutionKind.POLARS_NATIVE_EXPR,
    )
    contract_fields = fields(contracts.PhysicalImplementationSpec)
    lookalike_type = make_dataclass(
        "PhysicalImplementationSpec",
        [(field.name, field.type) for field in contract_fields],
        namespace={"__module__": "factor_engine.backend.contracts"},
    )
    lookalike = lookalike_type(*(getattr(genuine, field.name) for field in contract_fields))

    assert get_physical_spec(SimpleNamespace(_physical_spec=lookalike)) is None
    assert get_physical_spec(
        SimpleNamespace(_physical_spec=contracts.PhysicalImplementationID("probe"))
    ) is None
