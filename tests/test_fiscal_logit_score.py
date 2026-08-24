# -*- coding: utf-8 -*-
"""Tests for fiscal_logit_score operator."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.fundamental.fiscal_logit_score_op import pd_fiscal_logit_score


def _panel(data, index=None, columns=None):
    """Helper to create test panels."""
    if index is None:
        index = pd.date_range("2020-01-01", periods=len(data), freq="D")
    if columns is None:
        columns = [f"A{i}" for i in range(len(data[0]))]
    return pd.DataFrame(data, index=index, columns=columns)


class TestFiscalLogitScore:
    def test_basic_logit_score(self):
        """Basic logit transformation and z-score."""
        x = _panel([[0.5], [0.5], [0.5], [0.5], [0.7]])
        period_id = _panel([["2020Q1"], ["2020Q2"], ["2020Q3"], ["2020Q4"], ["2021Q1"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # Prior values all 0.5, logit(0.5) = 0
        # Current 0.7, logit(0.7) = log(0.7/0.3) ≈ 0.847
        # z = (0.847 - 0) / 0 = undefined (std = 0)
        assert np.isnan(result.iloc[-1, 0])

    def test_varying_margins(self):
        """Varying margin values."""
        x = _panel([[0.4], [0.5], [0.6], [0.45], [0.55], [0.7]])
        period_id = _panel([["2020Q1"], ["2020Q2"], ["2020Q3"], ["2020Q4"], ["2021Q1"], ["2021Q2"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # Should produce finite z-score
        assert np.isfinite(result.iloc[-1, 0])

    def test_boundary_clamping(self):
        """Values outside [0, 1] should be clamped."""
        x = _panel([[0.0], [1.0], [0.5], [0.3], [0.7]])
        period_id = _panel([["2020Q1"], ["2020Q2"], ["2020Q3"], ["2020Q4"], ["2021Q1"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # 0.0 -> 0.001, 1.0 -> 0.999
        # Should be finite and reasonable
        assert abs(result.iloc[-1, 0]) < 2.0  # Should be finite and reasonable

    def test_min_periods_requirement(self):
        """Insufficient periods."""
        x = _panel([[0.4], [0.5], [0.6]])
        period_id = _panel([["2020Q1"], ["2020Q2"], ["2020Q3"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # Only 3 periods, need 4
        assert np.isnan(result.iloc[-1, 0])

    def test_typical_margin_ratios(self):
        """Test with typical margin ratios in [0, 1]."""
        x = _panel([[0.15], [0.18], [0.12], [0.20], [0.16], [0.25]])
        period_id = _panel([["2020Q1"], ["2020Q2"], ["2020Q3"], ["2020Q4"], ["2021Q1"], ["2021Q2"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # Should produce finite z-score
        assert np.isfinite(result.iloc[-1, 0])

    def test_parameter_validation(self):
        """Test parameter validation."""
        x = _panel([[0.5]])
        period_id = _panel([["2020Q1"]])

        # Invalid revision policy
        with pytest.raises(ValueError, match="revision_policy"):
            pd_fiscal_logit_score(x, period_id, revision_policy="invalid")

        # Negative periods
        with pytest.raises(ValueError):
            pd_fiscal_logit_score(x, period_id, periods=-1)

        # Bool periods
        with pytest.raises(TypeError):
            pd_fiscal_logit_score(x, period_id, periods=True)

    def test_multicolumn(self):
        """Multiple instruments."""
        x = _panel([[0.3, 0.7], [0.4, 0.6], [0.35, 0.65], [0.45, 0.55], [0.5, 0.5]])
        period_id = _panel([["2020Q1", "2020Q1"], ["2020Q2", "2020Q2"], ["2020Q3", "2020Q3"],
                           ["2020Q4", "2020Q4"], ["2021Q1", "2021Q1"]])
        result = pd_fiscal_logit_score(x, period_id, periods=8, min_periods=4)
        # Both columns should produce results
        assert result.shape == x.shape
        assert np.isfinite(result.iloc[-1, 0])
        assert np.isfinite(result.iloc[-1, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
