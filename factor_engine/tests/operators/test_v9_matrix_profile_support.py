import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.mark.parametrize("canonical", ["ts_matrix_profile_motif_frequency", "ts_matrix_profile_neighbor_dispersion"])
@pytest.mark.parametrize("history", [25, 26])
def test_impossible_three_candidate_domain_rejects_before_kernel(canonical, history, monkeypatch):
    load_all()
    op = OperatorRegistry.get(canonical, backend="pandas_numpy", mode="research")
    def forbidden(*args, **kwargs):
        pytest.fail("impossible candidate support reached numerical kernel")
    monkeypatch.setattr(op, "_calculate_series", forbidden)
    with pytest.raises(ValueError, match="three historical candidates"):
        op.calculate(pd.DataFrame({"A":np.arange(80.)}), window=60, subsequence_length=20, history=history)


@pytest.mark.parametrize("canonical", ["ts_matrix_profile_motif_frequency", "ts_matrix_profile_neighbor_dispersion", "ts_matrix_profile_novelty", "ts_matrix_profile_motif_age"])
def test_exact_support_boundary_is_usable_and_single_candidate_controls_unchanged(canonical):
    load_all()
    op = OperatorRegistry.get(canonical, backend="pandas_numpy", mode="research")
    history = 25 if canonical.endswith(("novelty", "age")) else 27
    x = pd.DataFrame({"A":np.random.default_rng(933).normal(size=80)})
    full = op.calculate(x, window=60, subsequence_length=20, history=history)
    assert full.A.notna().sum() > 0
    prefix = op.calculate(x.iloc[:65], window=60, subsequence_length=20, history=history)
    pd.testing.assert_frame_equal(full.iloc[:65], prefix)
