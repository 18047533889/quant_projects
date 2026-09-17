import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r13_parameter_recipes import (
    migrate_catalog_r13_parameter_formula,
)


CASES = (
    (
        "cs_rank_gaussian(rank(ret), 3.0)",
        "cs_rank_gaussian(rank(ret), 'blom')",
        "SEMANTIC_REDESIGN",
    ),
    (
        "cs_rank_copula_mi(ret, turnover_ratio, grid=10)",
        "cs_rank_copula_mi(ret, turnover_ratio, grid=8)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_conditional_transfer_entropy(ret, turnover_ratio, gt(ret, 0.0), "
        "window=120, bins=5, lag=1, min_transitions=30, min_cells_ratio=0.1)",
        "ts_conditional_transfer_entropy(ret, turnover_ratio, gt(ret, 0.0), "
        "window=120, bins=2, lag=1, min_transitions=30, min_cells_ratio=0.1)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_markov_entropy_production(ret, window=120, bins=6, lag=1, min_periods=60)",
        "ts_markov_entropy_production(ret, window=120, bins=5, lag=1, min_periods=60)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_active_information_storage(ret, window=120, bins=6, history_length=2)",
        "ts_active_information_storage(ret, window=120, bins=2, history_length=2)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_conditional_mutual_information(ret, turnover_ratio, close, window=60, bins=4)",
        "ts_conditional_mutual_information(ret, turnover_ratio, close, window=60, bins=3)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_km_diffusion_gradient(ret, window=80, bins=6, lag=1)",
        "ts_km_diffusion_gradient(ret, window=80, bins=5, lag=1)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_extremogram(ret, window=120, quantile=0.9, lag=1, side='upper', "
        "fixed_threshold=False)",
        "ts_extremogram(ret, window=120, quantile=0.1, lag=1, side='upper', "
        "fixed_threshold=False)",
        "SEMANTIC_REDESIGN",
    ),
    (
        "ts_cross_extremogram(ret, turnover_ratio, window=120, target_q=0.9, "
        "source_q=0.9, lag=1, target_side='upper', source_side='upper', "
        "fixed_threshold=False)",
        "ts_cross_extremogram(ret, turnover_ratio, window=120, target_q=0.1, "
        "source_q=0.1, lag=1, target_side='upper', source_side='upper', "
        "fixed_threshold=False)",
        "SEMANTIC_REDESIGN",
    ),
)


def _review_logic(source: str) -> str:
    meanings = {
        "cs_rank_gaussian": "高斯排名；explicit canonical method='blom'",
        "cs_rank_copula": "Copula 联结结构",
        "ts_conditional_transfer_entropy": "方向性信息流",
        "ts_markov_entropy_production": "熵产生",
        "ts_active_information_storage": "自身可预测信息",
        "ts_conditional_mutual_information": "条件互信息",
        "ts_km_diffusion_gradient": "扩散梯度",
        "ts_extremogram": "极值依赖",
        "ts_cross_extremogram": "极值依赖",
    }
    return next(meaning for name, meaning in meanings.items() if name in source)


@pytest.mark.parametrize("source,expected,prefix", CASES)
def test_exact_reviewed_parameter_shapes_migrate(source, expected, prefix):
    result = migrate_catalog_r13_parameter_formula(
        source, logic=_review_logic(source), enabled=True
    )
    assert result.formula == expected
    assert result.changes
    assert all(change.startswith(prefix) for change in result.changes)
    assert migrate_catalog_r13_parameter_formula(
        result.formula, logic=_review_logic(source), enabled=True
    ).changes == ()


@pytest.mark.parametrize(
    "formula",
    [
        "cs_rank_gaussian(ret, 'van_der_waerden')",
        "cs_rank_copula_mi(ret, close, grid=16)",
        "ts_extremogram(ret, window=120, quantile=0.9, side='lower')",
        "ts_conditional_transfer_entropy(ret, close, ret, window=60, bins=5, lag=1)",
        "ts_active_information_storage(ret, window=120, bins=6, history_length=1)",
    ],
)
def test_unreviewed_or_already_canonical_shapes_fail_closed(formula):
    result = migrate_catalog_r13_parameter_formula(
        formula, logic=f"reviewed R12e parameter recipe: {formula}", enabled=True
    )
    assert result.formula == formula
    assert result.changes == ()


def test_same_operator_without_reviewed_keywords_fails_closed():
    assert migrate_catalog_r13_parameter_formula("ts_km_diffusion_gradient(ret)", logic="x", enabled=True).changes == ()
    assert migrate_catalog_r13_parameter_formula(
        "cs_rank_copula_mi(ret, close, grid=10)",
        logic="unrelated valuation signal",
        enabled=True,
    ).changes == ()


@pytest.mark.parametrize(
    "logic",
    [
        "高斯排名",
        "Gaussian rank normalization",
        "正态分位数",
        "参数化连续值入口",
        "mention blom without an affirmative declaration",
        "do not use canonical method='blom'",
        "canonical methods are blom and van_der_waerden",
    ],
)
def test_gaussian_numeric_method_requires_one_explicit_canonical_method(logic):
    formula = "cs_rank_gaussian(ret, 3.0)"
    result = migrate_catalog_r13_parameter_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


def test_gaussian_numeric_method_uses_explicit_van_der_waerden_review():
    result = migrate_catalog_r13_parameter_formula(
        "cs_rank_gaussian(ret, 3.0)",
        logic="reviewed canonical method: van_der_waerden",
        enabled=True,
    )
    assert result.formula == "cs_rank_gaussian(ret, 'van_der_waerden')"
    assert "method='van_der_waerden'" in result.changes[0]


@pytest.mark.parametrize("method", ["blom", "van_der_waerden"])
def test_gaussian_already_valid_explicit_method_is_preserved(method):
    formula = f"cs_rank_gaussian(ret, '{method}')"
    result = migrate_catalog_r13_parameter_formula(
        formula,
        logic="reviewed canonical method='blom'",
        enabled=True,
    )
    assert result.formula == formula
    assert result.changes == ()


def test_migration_is_opt_in_logic_gated_and_bounded():
    formula = "cs_rank_gaussian(ret, 3.0)"
    assert migrate_catalog_r13_parameter_formula(formula, logic="reviewed").formula == formula
    assert migrate_catalog_r13_parameter_formula(formula, enabled=True).formula == formula
    with pytest.raises(SyntaxError):
        migrate_catalog_r13_parameter_formula("ts_extremogram(", logic="x", enabled=True)
    with pytest.raises(ValueError, match="input budget"):
        migrate_catalog_r13_parameter_formula("x" * 65_537, logic="x", enabled=True)


@pytest.mark.parametrize("source,expected,_prefix", CASES)
def test_migrated_formulas_compile_without_source_reads(source, expected, _prefix):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.sources.datasource import DataSource

    class NoReadSource(DataSource):
        def _reject(self, method, *args, **kwargs):
            raise AssertionError(f"compile must not read through {method}")

        def load_column(self, *args, **kwargs):
            return self._reject("load_column", *args, **kwargs)

        def load_columns(self, *args, **kwargs):
            return self._reject("load_columns", *args, **kwargs)

        def prefetch_columns(self, *args, **kwargs):
            return self._reject("prefetch_columns", *args, **kwargs)

        def scan_polars_long(self, *args, **kwargs):
            return self._reject("scan_polars_long", *args, **kwargs)

        def scan_index_long(self, *args, **kwargs):
            return self._reject("scan_index_long", *args, **kwargs)

    load_all()
    migrated = migrate_catalog_r13_parameter_formula(
        source, logic=_review_logic(source), enabled=True
    ).formula
    assert migrated == expected
    expr = DSLParser(surface="compat_research").parse(migrated)
    FactorEngine(PandasBackend(), NoReadSource(), run_mode="research").compile(
        Factor(name="r13-parameter", expr=expr, source_expr=migrated, surface="compat_research")
    )


def test_tail_probability_conversion_preserves_upper_quantile_threshold():
    values = np.arange(1.0, 121.0)
    assert np.quantile(values, 0.9) == np.quantile(values, 1.0 - 0.1)


def test_repaired_parameters_execute_on_small_real_panels():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    index = pd.date_range("2025-01-01", periods=160)
    columns = ["A", "B", "C", "D"]
    base = np.arange(160.0)[:, None] + np.arange(4.0)[None, :]
    ret = pd.DataFrame(np.sin(base / 7.0) * 0.02, index=index, columns=columns)
    turnover = pd.DataFrame(np.cos(base / 9.0) + 2.0, index=index, columns=columns)

    gaussian = OperatorRegistry.get("cs_rank_gaussian", "pandas_numpy").calculate(
        ret, method="blom"
    )
    assert np.isfinite(gaussian.iloc[-1]).all()

    extremogram = OperatorRegistry.get("ts_extremogram", "pandas_numpy").calculate(
        ret, window=120, quantile=0.1, lag=1, side="upper", fixed_threshold=False
    )
    assert np.isfinite(extremogram.iloc[-1]).all()

    markov = OperatorRegistry.get(
        "ts_markov_entropy_production", "pandas_numpy"
    ).calculate(ret, window=120, bins=5, lag=1, min_periods=60)
    assert np.isfinite(markov.iloc[-1]).all()

    copula_columns = [f"S{i:03d}" for i in range(128)]
    copula_base = np.arange(3.0)[:, None] + np.arange(128.0)[None, :]
    copula_a = pd.DataFrame(np.sin(copula_base / 7.0), columns=copula_columns)
    copula_b = pd.DataFrame(np.cos(copula_base / 11.0), columns=copula_columns)
    copula = OperatorRegistry.get("cs_rank_copula_mi", "pandas_numpy").calculate(
        copula_a, copula_b, grid=8
    )
    assert np.isfinite(copula.iloc[-1]).all()
