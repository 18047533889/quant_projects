"""Search domains must bind to real kernels, not merely existing names."""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.eligibility.engine import TreatmentEligibilityEngine
from factor_preprocess.registry.transforms import create_default_registry


def profile(turnover=0.5):
    return FactorProfileArtifact(factor_id="p", factor_version="1", semantic_family="PRICE_VOLUME",
        source_type="price", update_frequency="daily", natural_horizon=20,
        time_behavior={"raw_turnover": turnover})


@pytest.mark.parametrize("endpoint", [0, 1])
def test_kama_search_bounds_execute_without_keyword_translation(endpoint):
    registry = create_default_registry()
    space = TreatmentEligibilityEngine(registry=registry).build_search_space(profile())
    params = {key: int(bounds[endpoint]) for key, bounds in space.allowed_transform_ids["kama"].items()}
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100),
                          "asset_id": "A", "value": np.arange(100, dtype=float)})
    result = registry.get_execution("kama")(frame, **params)
    assert result.notna().sum() > 50
    assert np.isnan(result.iloc[0])


def test_trimming_smoothing_does_not_force_slower_iir_response():
    engine = TreatmentEligibilityEngine()
    full = engine.build_search_space(profile()).allowed_transform_ids["one_sided_iir_lowpass"]["alpha"]
    trimmed = engine.build_search_space(profile(0.05)).allowed_transform_ids["one_sided_iir_lowpass"]["alpha"]
    # Larger alpha weights the new observation MORE, hence less smoothing.
    assert trimmed[0] > full[0]
    assert trimmed[1] == full[1]
