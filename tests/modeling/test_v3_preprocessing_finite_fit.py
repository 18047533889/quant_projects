import numpy as np
import pandas as pd
import pytest

from modeling.dataset import PanelDataset
from modeling.artifact import FrozenPreprocessing
from modeling.trainer import PreprocessingSpec, fit_preprocessing


def panel(values):
    return PanelDataset(pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(values)),
        "stock": ["A"] * len(values), "feature": values,
    }), feature_cols=["feature"], label_col=None)


@pytest.mark.parametrize("steps", [("imputer",), ("winsor",), ("standardize",),
                                   ("imputer", "winsor", "standardize")])
def test_fitted_statistics_ignore_all_nonfinite_values(steps):
    spec = PreprocessingSpec(steps=steps, winsor_quantiles=(0.0, 1.0))
    finite_missing = fit_preprocessing(panel([2., 4., 6., np.nan, np.nan]), spec)
    source = panel([2., 4., 6., np.inf, -np.inf])
    before = source.frame.copy(deep=True)
    fitted = fit_preprocessing(source, spec)
    assert fitted.state_hash() == finite_missing.state_hash()
    pd.testing.assert_frame_equal(source.frame, before)
    np.testing.assert_allclose(fitted.transform(np.array([[2.], [4.], [6.]])),
                               finite_missing.transform(np.array([[2.], [4.], [6.]])))


def test_imputer_uses_independent_finite_mean_not_zero_after_infinity():
    fitted = fit_preprocessing(panel([2., 4., 6., np.inf]), PreprocessingSpec(steps=("imputer",)))
    np.testing.assert_allclose(fitted.transform(np.array([[np.nan], [np.inf], [-np.inf]])),
                               np.full((3, 1), 4.))


def test_finite_fit_version_changes_identity_without_rewriting_legacy_state():
    legacy = FrozenPreprocessing([{"kind": "imputer", "means": [4.]}])
    original_hash = legacy.state_hash()
    fitted = fit_preprocessing(panel([2., 4., 6.]), PreprocessingSpec(steps=("imputer",)))
    assert fitted.steps[0]["fit_semantics_version"] == "finite-only-v2"
    assert fitted.state_hash() != original_hash
    assert legacy.state_hash() == original_hash
    np.testing.assert_array_equal(legacy.transform(np.array([[np.nan]])), [[4.]])
