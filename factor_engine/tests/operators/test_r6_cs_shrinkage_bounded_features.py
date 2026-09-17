import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.api.dsl_parser import DSLParser
from factor_engine.backend.polars_backend_kind import get_physical_spec
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


def _features(count=5):
    rng = np.random.default_rng(919)
    index = pd.date_range("2024-01-01", periods=6)
    columns = [f"S{i}" for i in range(14)]
    latent = rng.normal(size=(6, 14))
    return [
        pd.DataFrame(
            (0.2 + 0.1 * i) * latent + rng.normal(size=(6, 14)),
            index=index, columns=columns,
        )
        for i in range(count)
    ]


def _oracle(features, shrinkage):
    values = [feature.to_numpy(dtype=float) for feature in features]
    out = np.full_like(values[0], np.nan)
    for row in range(out.shape[0]):
        matrix = np.column_stack([value[row] for value in values])
        mean = matrix.mean(axis=0)
        cov = np.atleast_2d(np.cov(matrix.T))
        shrunk = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
        inverse = np.linalg.inv(shrunk + 1e-12 * np.eye(len(features)))
        centered = matrix - mean
        out[row] = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse, centered))
    return out


def test_five_feature_dsl_and_pandas_match_independent_oracle():
    formula = (
        "cs_shrinkage_mahalanobis(f1,f2,f3,f4,f5,shrinkage=0.2)"
    )
    expr = DSLParser(surface="compat_research").parse(formula)
    assert expr.op == "cs_shrinkage_mahalanobis"
    assert len(expr.args) == 5
    assert dict(expr.kwargs)["shrinkage"] == 0.2

    features = _features()
    operator = OperatorRegistry.get(expr.op, "pandas_numpy", mode="research")
    actual = operator.calculate(*features, shrinkage=0.2)
    np.testing.assert_allclose(actual, _oracle(features, 0.2), rtol=1e-12, atol=1e-12)


def test_legacy_four_features_and_fifth_positional_shrinkage_are_unchanged():
    features = _features(4)
    operator = OperatorRegistry.get(
        "cs_shrinkage_mahalanobis", "pandas_numpy", mode="research"
    )
    legacy = operator.calculate(*features, 0.2)
    keyword = operator.calculate(*features, shrinkage=0.2)
    named = operator.calculate(
        f1=features[0], f2=features[1], f3=features[2], f4=features[3],
        shrinkage=0.2,
    )
    np.testing.assert_allclose(legacy, keyword, equal_nan=True)
    np.testing.assert_allclose(named, keyword, equal_nan=True)
    with pytest.raises(ValueError, match="both as legacy fifth positional and keyword"):
        operator.calculate(*features, 0.2, shrinkage=0.3)


def test_five_feature_polars_delegate_matches_pandas_and_contract():
    features = _features()
    pandas_op = OperatorRegistry.get(
        "cs_shrinkage_mahalanobis", "pandas_numpy", mode="research"
    )
    polars_op = OperatorRegistry.get(
        "cs_shrinkage_mahalanobis", "polars", mode="research"
    )
    assert polars_op.metadata.param_names == [
        "f1", "f2", "f3", "f4", "f5", "shrinkage"
    ]
    assert polars_op.metadata.mixed_params == ("f5",)
    assert get_physical_spec(polars_op) is not None
    expected = pandas_op.calculate(*features, shrinkage=0.2)
    polars_features = [pl.from_pandas(feature) for feature in features]
    actual = polars_op.calculate(*polars_features, shrinkage=0.2)
    np.testing.assert_allclose(
        np.asarray(actual.to_numpy(), dtype=float), expected.to_numpy(dtype=float),
        rtol=1e-12, atol=1e-12,
    )


def test_feature_bound_rejects_six_panels_without_dropping_input():
    operator = OperatorRegistry.get(
        "cs_shrinkage_mahalanobis", "pandas_numpy", mode="research"
    )
    with pytest.raises(ValueError, match="multiple values for parameter 'shrinkage'"):
        operator.calculate(*_features(6), shrinkage=0.2)
