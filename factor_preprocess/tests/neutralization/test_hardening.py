"""
Pinning tests for neutralization hardening:

1. Column-name collision — a values-frame column sharing an exposure name
   must not silently regress on values-frame data (suffixes=("", "_exp")
   keeps the left column under its original name).
2. Integer-dtype value columns — insufficient-data dates must produce NaN,
   not the INT64_MIN integer sentinel from np.full_like on int dtype.
3. Rank guard — an exactly-determined fit must produce NaN, not ~0
   residuals masquerading as a fully neutralized factor.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.neutralization.ols import ols_neutralize
from factor_preprocess.neutralization.regularized import ridge_neutralize
from factor_preprocess.neutralization._alignment import (
    merged_exposure_cols,
    nan_residuals,
)


def _frames(collision: bool, seed=0, n_dates=3, n_assets=12):
    """values + exposures where y = 2*size + noise (size is the exposure).

    With ``collision=True`` the exposures frame also carries a column named
    ``value`` holding a DIFFERENT signal (pure noise uncorrelated with y's
    driver); correct neutralization regresses on the exposure ``size``
    (aliased ``size_exp`` when it collides), never on the values-frame
    ``value`` column itself.
    """
    rng = np.random.RandomState(seed)
    rows_v, rows_e = [], []
    for d in range(n_dates):
        size = rng.randn(n_assets)
        noise = rng.randn(n_assets) * 0.1
        y = 2.0 * size + noise
        decoy = rng.randn(n_assets)  # values-frame decoy only if collision
        for a in range(n_assets):
            rows_v.append({"date": d, "asset_id": f"A{a}", "value": y[a]})
            row_e = {"date": d, "asset_id": f"A{a}", "size": size[a]}
            if collision:
                row_e["value"] = decoy[a]
            rows_e.append(row_e)
    return pd.DataFrame(rows_v), pd.DataFrame(rows_e)


class TestColumnNameCollision:
    def test_merged_exposure_cols_aliases_colliding_names(self):
        _, exposures = _frames(collision=True)
        values, _ = _frames(collision=False)
        # values frame carries "value" too, so the exposure's "value" is
        # aliased even though "size" is not.
        cols = merged_exposure_cols(exposures, values, "date", "asset_id")
        assert cols == ["size", "value_exp"]

        # Now values frame also carries "size" → must alias to size_exp
        values2 = values.rename(columns={"value": "size"})
        exposures2 = exposures  # has size + value
        cols2 = merged_exposure_cols(exposures2, values2, "date", "asset_id")
        assert cols2 == ["size_exp", "value"]

    def test_ols_collision_regresses_on_exposure_not_values(self):
        values, exposures = _frames(collision=True, seed=1)
        # exposures frame's "value" column is renamed to "value_exp" by the
        # suffixes merge; ols must use it (the decoy), not values' "value".
        # Regression on the true driver (size) must shrink variance vs raw.
        res = ols_neutralize(
            values, exposures, min_observations=5, value_col="value"
        )
        assert len(res) == len(values)
        raw_var = np.var(values["value"].values)
        res_var = np.nanvar(res.values)
        # y = 2*size + 0.1*noise → residual variance ≈ 0.01, well under 4+
        assert res_var < raw_var / 2

    def test_ridge_collision(self):
        values, exposures = _frames(collision=True, seed=2)
        res = ridge_neutralize(values, exposures, alpha=1.0, min_observations=5)
        assert len(res) == len(values)
        assert np.nanvar(res.values) < np.var(values["value"].values) / 2


class TestIntegerDtypeSentinel:
    def test_nan_residuals_float_on_int_input(self):
        y = np.array([1, 2, 3], dtype=np.int64)
        out = nan_residuals(y)
        assert out.dtype == np.float64
        assert np.all(np.isnan(out))

    def test_ols_insufficient_data_int_values_are_nan_not_sentinel(self):
        # Integer-dtype y with too few valid observations per date.
        rng = np.random.RandomState(3)
        rows_v, rows_e = [], []
        for d in range(2):
            size = rng.randn(4)
            for a in range(4):
                rows_v.append({"date": d, "asset_id": f"A{a}", "value": int(a)})
                rows_e.append({"date": d, "asset_id": f"A{a}", "size": size[a]})
        values = pd.DataFrame(rows_v)
        exposures = pd.DataFrame(rows_e)
        assert values["value"].dtype.kind == "i"

        res = ols_neutralize(values, exposures, min_observations=50)
        # Must be all NaN — never the INT64_MIN sentinel (-2**63)
        assert np.all(np.isnan(res.values))
        vals = res.values.astype(np.float64)
        assert np.all(np.isnan(vals) | (np.abs(vals) < 1e12))


class TestRankGuard:
    def test_exactly_determined_fit_yields_nan_not_zero_residual(self):
        # n_valid == n_exposures + intercept → exactly determined; the old
        # guard admitted it and residuals collapsed to ~0.
        rng = np.random.RandomState(4)
        rows_v, rows_e = [], []
        for d in range(2):
            size = rng.randn(2)  # 2 assets == 1 exposure + intercept
            y = 2.0 * size + 0.1 * rng.randn(2)
            for a in range(2):
                rows_v.append({"date": d, "asset_id": f"A{a}", "value": y[a]})
                rows_e.append({"date": d, "asset_id": f"A{a}", "size": size[a]})
        values = pd.DataFrame(rows_v)
        exposures = pd.DataFrame(rows_e)

        res = ols_neutralize(values, exposures, min_observations=1)
        assert np.all(np.isnan(res.values))

    def test_ridge_exactly_determined_yields_nan(self):
        rng = np.random.RandomState(5)
        rows_v, rows_e = [], []
        for d in range(2):
            size = rng.randn(2)  # 2 assets == 1 exposure + intercept
            y = 2.0 * size + 0.1 * rng.randn(2)
            for a in range(2):
                rows_v.append({"date": d, "asset_id": f"A{a}", "value": y[a]})
                rows_e.append({"date": d, "asset_id": f"A{a}", "size": size[a]})
        values = pd.DataFrame(rows_v)
        exposures = pd.DataFrame(rows_e)

        res = ridge_neutralize(values, exposures, min_observations=1)
        assert np.all(np.isnan(res.values))
