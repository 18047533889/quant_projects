import numpy as np
import pytest
from factor_engine.cleaned_operators import binned_response


@pytest.mark.parametrize("kernel", [
    binned_response._monotonicity_series, binned_response._curvature_series,
])
def test_infeasible_binning_rejects_before_output_allocation(monkeypatch, kernel):
    x = np.zeros((2, 1))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("invalid parameter domain must reject before allocation")

    monkeypatch.setattr(binned_response.np, "full", forbidden_allocation)
    with pytest.raises(ValueError, match="INFEASIBLE_PARAMETER_DOMAIN"):
        kernel(x, x, window=5, bins=4, min_per_bin=3)
