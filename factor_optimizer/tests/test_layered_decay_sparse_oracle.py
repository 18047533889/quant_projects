import numpy as np
import pandas as pd

from factor_optimizer.adapters.layered_decay import LayeredDecayPlan, _assign_daily_quantiles
from factor_optimizer.adapters.layered_decay_long import _execute_sparse_layered_decay_validated


def _legacy_recurrence(x, bins, half_lives):
    alpha = -np.expm1(-np.log(2.) / np.asarray(half_lives, float))
    rho = 1. - alpha
    numerator = np.zeros((x.shape[1], len(alpha)))
    mass = np.zeros_like(numerator)
    scale = np.zeros(x.shape[1])
    out = np.full(x.shape, np.nan)
    for t in range(1, len(x)):
        source, layer = x[t - 1], bins[t - 1]
        good = np.isfinite(source) & (layer >= 0)
        numerator *= rho
        mass *= rho
        numerator[~good] = 0.
        mass[~good] = 0.
        scale[~good] = 0.
        assets = np.flatnonzero(good)
        if not len(assets):
            continue
        new_scale = np.maximum(scale[assets], np.abs(source[assets]))
        ratio = np.divide(scale[assets], new_scale, out=np.zeros(len(assets)), where=new_scale > 0)
        numerator[assets] *= ratio[:, None]
        scale[assets] = new_scale
        normalized = np.divide(source[assets], new_scale, out=np.zeros(len(assets)), where=new_scale > 0)
        selected = layer[assets]
        numerator[assets, selected] += alpha[selected] * normalized
        mass[assets, selected] += alpha[selected]
        weighted = numerator[assets].sum(axis=1) / mass[assets].sum(axis=1)
        out[t, assets] = np.clip(weighted, -1., 1.) * new_scale
    return out


def test_sparse_stream_matches_independent_pre_refactor_oracle():
    rng = np.random.default_rng(7712)
    rows = []
    for date in range(12):
        for asset in range(43):
            if (date * 11 + asset * 5) % 13 == 0:
                continue
            value = float(asset % 4) if date % 4 == 0 else float(rng.normal())
            if (date + asset) % 29 == 0:
                value = np.nan
            if (date + asset) % 31 == 0:
                value = np.inf
            rows.append((date, asset, value))
    frame = pd.DataFrame(rows, columns=["date", "asset_id", "value"])
    frame = frame.sort_values(["asset_id", "date"], kind="stable")
    panel = frame.pivot(index="date", columns="asset_id", values="value").sort_index()
    x = np.where(np.isfinite(panel.to_numpy(dtype=float)), panel.to_numpy(dtype=float), np.nan)
    bins = _assign_daily_quantiles(x)
    half_lives = tuple(np.linspace(1., 60., 20))
    dense = _legacy_recurrence(x, bins, half_lives)
    rows_at = panel.index.get_indexer(frame["date"])
    cols_at = panel.columns.get_indexer(frame["asset_id"])
    expected = dense[rows_at, cols_at]
    actual = _execute_sparse_layered_decay_validated(frame, half_lives).to_numpy()
    np.testing.assert_array_equal(actual, expected)
    via_plan = LayeredDecayPlan(half_lives, "train:legacy-oracle").execute(
        frame, allow_research=True
    ).to_numpy()
    np.testing.assert_array_equal(via_plan, expected)
