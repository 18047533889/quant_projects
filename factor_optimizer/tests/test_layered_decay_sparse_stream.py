import numpy as np
import pandas as pd

from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
from factor_optimizer.adapters.layered_decay import _assign_daily_quantiles
from factor_optimizer.adapters.layered_decay_long import _execute_sparse_layered_decay_validated
from factor_preprocess.transforms.layered_decay import layered_decay


def _dense_oracle(frame, half_lives):
    panel = frame.pivot(index="date", columns="asset_id", values="value").sort_index()
    x = np.where(np.isfinite(panel.to_numpy(dtype=float)), panel.to_numpy(dtype=float), np.nan)
    bins = _assign_daily_quantiles(x)
    dense = layered_decay(x, bins, half_lives, allow_research=True)
    rows = panel.index.get_indexer(frame["date"])
    cols = panel.columns.get_indexer(frame["asset_id"])
    return pd.Series(dense[rows, cols], index=frame.index, name="value")


def test_sparse_stream_matches_dense_oracle_with_gaps_ties_and_shuffled_rows():
    rng = np.random.default_rng(3302)
    records = []
    for date in range(9):
        for asset in range(37):
            if (date * 7 + asset * 3) % 8 == 0:
                continue
            value = float((asset % 5) - 2) if date % 3 == 0 else float(rng.normal())
            if (date + asset) % 19 == 0:
                value = np.nan
            if (date + asset) % 23 == 0:
                value = np.inf
            records.append((date, asset, value))
    frame = pd.DataFrame(records, columns=["date", "asset_id", "value"])
    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame.index = np.arange(len(frame)) % 13
    lives = tuple(np.linspace(1., 60., 20))
    expected = _dense_oracle(frame, lives)
    actual = _execute_sparse_layered_decay_validated(frame, lives)
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    plan = LayeredDecayPlan(lives, "train:stream")
    try:
        plan.execute(frame, allow_research=True)
    except ValueError as exc:
        assert "monotone" in str(exc)
    else:
        raise AssertionError("plan input remains monotone per asset")
    ordered = frame.sort_values(["asset_id", "date"], kind="stable")
    np.testing.assert_array_equal(plan.execute(ordered, allow_research=True).to_numpy(),
                                  _dense_oracle(ordered, lives).to_numpy())


def test_sparse_stream_preserves_lag_resets_at_missing_union_date_and_causality():
    rows = []
    # At least 20 assets per populated date, with asset 0 absent at one date.
    for date in range(5):
        for asset in range(24):
            if asset == 0 and date == 2:
                continue
            rows.append((date, asset, float(asset + date)))
    frame = pd.DataFrame(rows, columns=["date", "asset_id", "value"])
    lives = (5.,) * 20
    baseline = _execute_sparse_layered_decay_validated(frame, lives)
    assert np.isnan(baseline[frame.date.eq(0)]).all()
    # Missing at date 2 resets asset 0, so date 3 cannot inherit its date 1 state.
    assert np.isnan(baseline[(frame.date.eq(3)) & frame.asset_id.eq(0)].iloc[0])
    changed = frame.copy()
    changed.loc[changed.date.eq(4), "value"] *= -1000.
    after = _execute_sparse_layered_decay_validated(changed, lives)
    np.testing.assert_array_equal(after[frame.date.lt(4)], baseline[frame.date.lt(4)])


def test_sparse_stream_categorical_keys_have_real_values_and_reject_duplicates():
    dates = np.repeat([0, 1, 2], 24)
    assets = np.tile(np.arange(24), 3)
    frame = pd.DataFrame({
        "date": pd.Categorical(dates, categories=[0, 1, 2, 99]),
        "asset_id": pd.Categorical(assets, categories=list(range(24)) + [999]),
        "value": assets.astype(float) + dates * 0.25,
    })
    plan = LayeredDecayPlan((2.,) * 20, "train:categorical")
    result = plan.execute(frame, allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), _dense_oracle(frame, (2.,) * 20).to_numpy())
    assert np.isfinite(result[frame.date.eq(2)]).any()
    duplicate = pd.concat([frame, frame.iloc[[0]]])
    try:
        plan.execute(duplicate, allow_research=True)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate date/asset identities must be rejected")


def test_sparse_stream_matches_oracle_across_forced_quantile_blocks():
    dates = np.repeat(np.arange(8), 24)
    assets = np.tile(np.arange(24), 8)
    values = (assets * 7 + dates * 11).astype(float)
    values[(assets + dates) % 13 == 0] = np.nan
    frame = pd.DataFrame({"date": dates, "asset_id": assets, "value": values})
    lives = tuple(np.linspace(1., 60., 20))
    bytes_per_date = len(np.unique(assets)) * 52 + 20 * 1024
    actual = _execute_sparse_layered_decay_validated(
        frame, lives, working_bytes=bytes_per_date * 2
    )
    expected = _dense_oracle(frame, lives)
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
