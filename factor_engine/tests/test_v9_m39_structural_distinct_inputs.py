import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.group_spectrum import _group_spectrum_series
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.field import FieldRef
from factor_engine.expr.canonical import canonical_expression
from factor_engine.identity.serializer import canonical_ast_text
from factor_engine.ir.analyzer import (
    Analyzer, DistinctInputContractError, _canonical_expression_identity,
)


@pytest.fixture(scope="module", autouse=True)
def bootstrap_registered_contracts():
    from factor_engine.cleaned_operators import load_all
    load_all()


def _call(f1, f2, f3):
    return CleanedCall(
        "group_feature_mode_share",
        args=(f1, f2, f3, ColumnRef("industry")),
        kwargs=(("breadth_window", 2), ("group_schema_version", "v1")),
    )


def test_same_expression_is_rejected_before_leaf_resolution(monkeypatch):
    analyzer = Analyzer()

    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("leaf resolution / IO boundary was reached")

    monkeypatch.setattr(analyzer, "_resolve_field_ir", unexpected_io)
    repeated = ColumnRef("turnover")
    with pytest.raises(DistinctInputContractError, match="f1.*f2.*same structural"):
        analyzer.lower(_call(repeated, repeated, ColumnRef("momentum")))


def test_distinct_contract_is_catalog_visible_only_on_spectrum_outputs():
    declared = (("f1", "f2", "f3"),)
    for name in (
        "group_feature_mode_share", "group_feature_effective_rank",
        "group_feature_mode_localization", "group_feature_spectral_gap",
        "group_feature_second_mode_localization",
    ):
        assert OperatorRegistry.get(name).metadata.distinct_input_groups == declared
        assert OperatorRegistry.catalog()[name]["distinct_input_groups"] == declared
    for diagnostic in (
        "group_feature_valid_member_count", "group_feature_coverage_ratio",
    ):
        assert OperatorRegistry.get(diagnostic).metadata.distinct_input_groups == ()


def test_field_identity_includes_catalog_hash():
    common = dict(
        name="close", field_id="price.close", canonical_name="close",
        table="Daily", source_name="close",
    )
    # A catalog revision is a different structural field identity even when all
    # transport labels are equal.
    first = FieldRef(**common, catalog_hash="catalog-a")
    second = FieldRef(**common, catalog_hash="catalog-b")
    assert _canonical_expression_identity(first) != _canonical_expression_identity(second)
    from factor_engine.identity.serializer import canonical_ast_text
    from factor_engine.expr.canonical import canonical_expression
    assert canonical_ast_text(first) == canonical_expression(first)
    assert canonical_ast_text(second) == canonical_expression(second)
    assert canonical_expression(first) != canonical_expression(second)
    assert canonical_ast_text(first) == canonical_expression(first)
    assert canonical_ast_text(second) == canonical_expression(second)
    assert canonical_expression(first) != canonical_expression(second)


def test_identity_reuses_operator_and_parameter_alias_authorities():
    x = ColumnRef("close")
    alias_form = CleanedCall("mean", args=(x,), kwargs=(("d", 5),))
    canonical_form = CleanedCall("ts_mean", args=(x,), kwargs=(("window", 5),))
    assert _canonical_expression_identity(alias_form) == _canonical_expression_identity(
        canonical_form
    )


def test_runtime_allows_distinct_fields_with_equal_numeric_values_and_is_prefix_stable():
    rng = np.random.default_rng(39)
    prefix = rng.normal(size=(3, 12))
    f3 = rng.normal(size=(3, 12))
    feats = np.stack((prefix, prefix.copy(), f3), axis=2)
    groups = np.full((3, 12), "A", dtype=object)

    short = _group_spectrum_series(feats, groups, "mode_share", breadth_window=2)
    future = np.concatenate((feats, rng.normal(size=(1, 12, 3))), axis=0)
    extended_groups = np.full((4, 12), "A", dtype=object)
    long = _group_spectrum_series(future, extended_groups, "mode_share", breadth_window=2)

    np.testing.assert_allclose(short, long[:3], equal_nan=True)
    assert np.isfinite(short[1:]).any()


def test_raw_dataframe_call_does_not_invent_expression_identity():
    rng = np.random.default_rng(390)
    index = pd.date_range("2026-01-01", periods=3)
    columns = [f"s{i}" for i in range(12)]
    same = pd.DataFrame(rng.normal(size=(3, 12)), index=index, columns=columns)
    third = pd.DataFrame(rng.normal(size=(3, 12)), index=index, columns=columns)
    groups = pd.DataFrame("A", index=index, columns=columns)

    # Unknown research DataFrames expose values but no trustworthy structural
    # field identity.  Equal values must therefore remain admissible.
    op = OperatorRegistry.get("group_feature_mode_share", backend="pandas_numpy", mode="research")
    result = op.calculate(
        same, same.copy(), third, groups, breadth_window=2,
    )
    assert result.shape == same.shape
    assert result.notna().any().any()
    pd.testing.assert_frame_equal(
        result.iloc[:2],
        op.calculate(same.iloc[:2], same.iloc[:2].copy(), third.iloc[:2], groups.iloc[:2], breadth_window=2),
    )
