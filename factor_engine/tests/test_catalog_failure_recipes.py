import pytest

from factor_engine.tools.catalog_failure_recipes import (
    migrate_catalog_failure_formula,
)


def test_migration_is_opt_in():
    formula = (
        "rank(ts_ordinal_irreversibility(ret, window=120, order=3, "
        "delay=1, min_periods=5))"
    )

    result = migrate_catalog_failure_formula(formula, logic="time irreversibility")

    assert result.formula == formula
    assert result.changes == ()


@pytest.mark.parametrize(
    ("formula", "logic", "expected", "change_fragment"),
    [
        (
            "rank(ts_ordinal_irreversibility(ret, window=120, order=3, "
            "delay=1, min_periods=5))",
            "收益时间反演不可逆性",
            "rank(ts_ordinal_irreversibility(ret, window=120, order=3, "
            "delay=1, min_patterns=5))",
            "SEMANTIC_REDESIGN ts_ordinal_irreversibility",
        ),
        (
            "rank(neg(ts_extreme_cluster_ratio(ret, 60, 2.0, 0.0, 5, 5)))",
            "极端收益集中后的过度反应修复",
            "rank(neg(ts_extreme_cluster_ratio(ret, 60, 'quantile', 0.9, "
            "'absolute', 5)))",
            "SEMANTIC_REDESIGN ts_extreme_cluster_ratio",
        ),
        (
            "rank(ts_mmd_rbf_shift(ret, window=60))",
            "收益60日RBF-MMD分布变化",
            "rank(ts_mmd_rbf_shift(ret, recent_window=20, old_window=40))",
            "SEMANTIC_REDESIGN ts_mmd_rbf_shift",
        ),
        (
            "rank(ts_tail_ratio(ret, 60, 2.0, 0.0, 5))",
            "高尾部占比但低绝对收益熵表示风险高度集中",
            "rank(ts_tail_ratio(ret, 60, 0.05, 0.95, 5))",
            "SEMANTIC_REDESIGN ts_tail_ratio",
        ),
        (
            "rank(KeltnerPosition(high, low, close, close, 20, 2.0))",
            "通道中的相对位置和通道宽度共同刻画趋势状态",
            "rank(KeltnerPosition(high, low, close, 20, 20, 2.0))",
            "SEMANTIC_REDESIGN KeltnerPosition",
        ),
    ],
)
def test_reviewed_catalog_failure_recipe_is_rewritten(
    formula, logic, expected, change_fragment
):
    # Defect: an exact reviewed legacy call survives with an invalid canonical
    # signature, or is changed without an explicit audit record.
    result = migrate_catalog_failure_formula(formula, logic=logic, enabled=True)

    assert result.formula == expected
    assert len(result.changes) == 1
    assert change_fragment in result.changes[0]
    assert migrate_catalog_failure_formula(
        result.formula, logic=logic, enabled=True
    ).changes == ()


@pytest.mark.parametrize(
    ("formula", "logic"),
    [
        (
            "ts_extreme_cluster_ratio(ret, 60, 2.0, 0.0, 5, 5)",
            "generic signal",
        ),
        ("ts_extreme_cluster_ratio(ret, 20, 'quantile', 0.9, 'absolute', 3)", "extreme clustering"),
        ("ts_mmd_rbf_shift(ret, window=61)", "MMD distribution shift"),
        ("ts_mmd_rbf_shift(ret, window=60)", "generic signal"),
        ("ts_tail_ratio(ret, 60, 2.0, 0.0, 6)", "tail concentration"),
        ("KeltnerPosition(high, low, close, open, 20, 2.0)", "channel position"),
        (
            "ts_ordinal_irreversibility(ret, window=5, order=4, delay=2, "
            "min_periods=5)",
            "ordinal irreversibility",
        ),
    ],
)
def test_ambiguous_or_unsupported_lookalike_fails_closed(formula, logic):
    # Defect: a broad alias silently invents semantics for a call outside the
    # reviewed shape or without corroborating factor logic.
    result = migrate_catalog_failure_formula(formula, logic=logic, enabled=True)

    assert result.formula == formula
    assert result.changes == ()


def test_multiple_nested_calls_preserve_unrelated_source_text():
    formula = (
        "add(ts_ordinal_irreversibility(ret, window=120, order=3, delay=1, "
        "min_periods=5), ts_mmd_rbf_shift(ret, window=60))"
    )
    logic = "时间反演不可逆性与RBF-MMD分布变化联合"

    result = migrate_catalog_failure_formula(formula, logic=logic, enabled=True)

    assert result.formula == (
        "add(ts_ordinal_irreversibility(ret, window=120, order=3, delay=1, "
        "min_patterns=5), ts_mmd_rbf_shift(ret, recent_window=20, "
        "old_window=40))"
    )
    assert len(result.changes) == 2


def test_invalid_and_oversized_inputs_are_bounded():
    with pytest.raises(SyntaxError):
        migrate_catalog_failure_formula("ts_tail_ratio(", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_failure_formula("x" * 65_537, enabled=True)


@pytest.mark.parametrize(
    ("source", "logic", "expected_op"),
    [
        (
            "ts_ordinal_irreversibility(ret, window=120, order=3, delay=1, "
            "min_periods=5)",
            "ordinal irreversibility",
            "ts_ordinal_irreversibility",
        ),
        (
            "ts_extreme_cluster_ratio(ret, 60, 2.0, 0.0, 5, 5)",
            "extreme return clustering",
            "ts_extreme_cluster_ratio",
        ),
        (
            "ts_mmd_rbf_shift(ret, window=60)",
            "MMD distribution shift",
            "ts_mmd_rbf_shift",
        ),
        (
            "ts_tail_ratio(ret, 60, 2.0, 0.0, 5)",
            "tail ratio concentration",
            "ts_tail_ratio",
        ),
        (
            "KeltnerPosition(high, low, close, close, 20, 2.0)",
            "Keltner channel position",
            "divide",
        ),
    ],
)
def test_migrated_recipe_parses_and_lowers_against_real_contract(
    source, logic, expected_op
):
    # Defect: a textually plausible rewrite still violates the registered DSL
    # contract and fails during real planning.
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    formula = migrate_catalog_failure_formula(
        source, logic=logic, enabled=True
    ).formula
    ir = Analyzer().lower(DSLParser(surface="compat_research").parse(formula)).ir

    assert ir.op == expected_op


def test_redesigned_parameters_execute_on_small_real_panels():
    # Defect: repaired parameters compile but remain invalid at the actual
    # pandas operator boundary.
    import numpy as np
    import pandas as pd

    import factor_engine.cleaned_operators.alpha_language_distribution  # noqa: F401
    import factor_engine.cleaned_operators.robust_tail  # noqa: F401
    import factor_engine.cleaned_operators.state_geometry  # noqa: F401
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.technical.indicators_v2 import (
        KeltnerPosition,
    )

    values = np.sin(np.arange(140, dtype=float) / 7.0) + np.arange(140) / 100.0
    panel = pd.DataFrame({"A": values})
    ordinal = OperatorRegistry.get(
        "ts_ordinal_irreversibility", backend="pandas_numpy"
    ).calculate(panel, window=120, order=3, delay=1, min_patterns=5)
    cluster = OperatorRegistry.get(
        "ts_extreme_cluster_ratio", backend="pandas_numpy"
    ).calculate(
        panel, window=60, threshold="quantile", q=0.9,
        side="absolute", min_periods=5
    )
    mmd = OperatorRegistry.get(
        "ts_mmd_rbf_shift", backend="pandas_numpy"
    ).calculate(panel, recent_window=20, old_window=40)
    tail = OperatorRegistry.get("ts_tail_ratio", backend="pandas_numpy").calculate(
        panel, window=60, q_low=0.05, q_high=0.95, min_periods=5
    )
    high = panel + 1.0
    low = panel - 1.0
    keltner = KeltnerPosition(high, low, panel, 20, 20, 2.0)

    for result in (ordinal, cluster, mmd, tail, keltner):
        assert np.isfinite(result.iloc[-1, 0])


def test_overlapping_reviewed_calls_fail_closed_as_one_formula():
    # Defect: independently applying old AST spans for a parent and child call
    # corrupts source offsets or emits audit records for a discarded child edit.
    formula = (
        "ts_extreme_cluster_ratio(ts_mmd_rbf_shift(ret, window=60), "
        "60, 2.0, 0.0, 5, 5)"
    )
    logic = "extreme cluster of an MMD distribution shift"

    result = migrate_catalog_failure_formula(formula, logic=logic, enabled=True)

    assert result.formula == formula
    assert result.changes == ()
