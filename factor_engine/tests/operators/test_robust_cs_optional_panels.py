"""Optional robust-CS panels must remain optional through registry wrappers."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()


NAMES = (
    "cs_mahalanobis_distance", "cs_knn_distance", "cs_local_density_score",
    "cs_robust_mahalanobis_mad", "cs_actual_lof_score", "cs_relative_density_ratio",
)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_optional_feature_wrapper_matches_kernel(name, count):
    from factor_engine.cleaned_operators.cross_section import robust_cs
    rng = np.random.default_rng(84)
    panels = [pd.DataFrame(rng.normal(size=(3, 32))) for _ in range(count)]
    op = OperatorRegistry.get(name, "pandas_numpy", mode="research")
    actual = op.calculate(*panels)
    expected = op._calculate_series(*panels)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isfinite(np.asarray(actual)).any()
    for key in ("f2", "f3", "f4"):
        assert op.metadata.param_specs[key].default is None


def test_required_features_and_upper_bound_remain_enforced():
    panel = pd.DataFrame(np.arange(64).reshape(2, 32))
    shrink = OperatorRegistry.get("cs_shrinkage_mahalanobis", "pandas_numpy", mode="research")
    with pytest.raises(ValueError, match="missing required"):
        shrink.calculate(panel, panel, panel)
    op = OperatorRegistry.get("cs_robust_mahalanobis_mad", "pandas_numpy", mode="research")
    with pytest.raises(ValueError, match="missing required"):
        op.calculate()
    with pytest.raises(ValueError, match="at most"):
        op.calculate(*([panel] * 5))


def _edge_case_panels():
    rng = np.random.default_rng(90210)
    columns = [f"asset_{i:02d}" for i in range(32)]
    panels = [pd.DataFrame(rng.normal(size=(3, 32)), columns=columns) for _ in range(4)]
    for panel in panels:
        panel.iloc[:, 1] = panel.iloc[:, 0]  # ties
    panels[0].iloc[0, 3] = np.nan
    panels[1].iloc[1, 7] = np.nan
    panels[2].iloc[2, 11] = np.nan
    return panels


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_registered_polars_delegate_matches_pandas_for_optional_panels(name, count):
    panels = _edge_case_panels()[:count]
    polars_panels = [pl.from_pandas(panel) for panel in panels]
    pandas_op = OperatorRegistry.get(name, "pandas_numpy", mode="research")
    polars_op = OperatorRegistry.get(name, "polars", mode="research")

    assert polars_op is not None
    spec = polars_op._physical_spec
    assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
    assert spec.supports_lazy is False
    assert "polars_native" not in polars_op.metadata.tags

    expected = pandas_op.calculate(*panels)
    actual = polars_op.calculate(*polars_panels)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12,
                               equal_nan=True)
    assert actual.columns == list(expected.columns)
    assert actual.shape == expected.shape
    assert np.isfinite(actual.to_numpy()).any()


@pytest.mark.parametrize("backend", ("pandas_numpy", "polars"))
@pytest.mark.parametrize("count", (1, 2, 3))
def test_shrinkage_keeps_four_required_features(backend, count):
    op = OperatorRegistry.get("cs_shrinkage_mahalanobis", backend, mode="research")
    specs = op.metadata.param_specs
    for key in ("f1", "f2", "f3", "f4"):
        assert key not in specs or not specs[key].has_default
    assert specs["f5"].default is None
    assert specs["shrinkage"].default == 0.1

    with pytest.raises(ValueError, match="missing required"):
        op.calculate(*_edge_case_panels()[:count])
