"""Production registry routes the distinct lagged robust recipe to FE."""
import sys

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_smoothing import (
    ROBUST_EWMA_RECIPE, execute_robust_ewma, get_fe_composite_executor,
)
from factor_preprocess.errors import GovernanceError
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.transforms.smoothing import robust_ewma


def _frame():
    rng = np.random.default_rng(20261002)
    frame = pd.DataFrame({
        "asset_id": np.tile(["a", "b"], 30),
        "date": np.repeat(np.arange(30, dtype=np.int64), 2),
        "value": rng.normal(size=60),
    }, index=np.arange(60) // 2)
    frame.iloc[14, frame.columns.get_loc("value")] = np.nan
    frame.iloc[22, frame.columns.get_loc("value")] = np.inf
    return frame


@pytest.mark.parametrize("min_periods", [0, 1, 3])
def test_registry_uses_fe_and_preserves_oracle_and_prefix(min_periods):
    registry = get_default_registry()
    meta = registry.get("robust_ewma")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_equivalent_semantics == ROBUST_EWMA_RECIPE
    assert get_fe_composite_executor("robust_ewma", ROBUST_EWMA_RECIPE) is execute_robust_ewma
    frame = _frame()
    before = frame.copy(deep=True)
    kwargs = dict(halflife=5.0, winsor_std=2.0, min_periods=min_periods)
    result = registry.get_execution("robust_ewma")(frame, **kwargs)
    expected = robust_ewma(frame, **kwargs)
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-14, equal_nan=True)
    assert result.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, before)
    prefix = registry.get_execution("robust_ewma")(frame.iloc[:35], **kwargs)
    np.testing.assert_allclose(prefix, result.iloc[:35], rtol=2e-13, atol=2e-14, equal_nan=True)


def test_missing_fe_fails_closed_with_explicit_research_fallback(monkeypatch):
    registry = get_default_registry()
    monkeypatch.setitem(sys.modules, "factor_engine.backend.long_robust_ewm", None)
    with pytest.raises(GovernanceError):
        registry.get_execution("robust_ewma")
    frame = _frame()
    actual = registry.get_execution("robust_ewma", allow_research=True)(frame, halflife=5.0)
    expected = robust_ewma(frame, halflife=5.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
