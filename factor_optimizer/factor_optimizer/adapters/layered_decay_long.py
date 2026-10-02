"""Sparse long-panel driver for the shared research NumPy layered-decay state."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _execute_sparse_layered_decay_validated(
        frame: pd.DataFrame, half_lives, *, working_bytes=64 * 1024 * 1024
) -> pd.Series:
    """Execute exact QE daily bins with O(assets×layers + input rows) storage.

    Global dates present in the input define the recurrence clock. Missing asset
    rows reset state at the next observed union date, matching a dense pivot with
    absent cells filled as NaN. Results retain caller row order and index.
    """
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float, name="value")
    from factor_preprocess.transforms.layered_decay_state import LayeredDecayState
    from quant_evaluator.metrics.quantile import assign_quantiles_batch

    date_codes, dates = pd.factorize(frame["date"], sort=True)
    asset_codes, assets = pd.factorize(frame["asset_id"], sort=False)
    order = np.argsort(date_codes, kind="stable")
    ordered_dates = date_codes[order]
    boundaries = np.r_[0, np.flatnonzero(ordered_dates[1:] != ordered_dates[:-1]) + 1, len(order)]
    values = frame["value"].to_numpy(dtype=float, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    output = np.full(len(frame), np.nan)
    state = LayeredDecayState(len(assets), half_lives)
    if (isinstance(working_bytes, bool) or not isinstance(working_bytes, int)
            or working_bytes < 1):
        raise ValueError("working_bytes must be a positive integer")
    previous_values = previous_bins = None
    bytes_per_date = max(1, len(assets)) * 52 + 20 * 1024
    if len(dates) and bytes_per_date > working_bytes:
        raise ValueError("one date exceeds the layered-decay quantile working-set budget")
    dates_per_block = max(1, min(len(dates), working_bytes // bytes_per_date))
    for first_date in range(0, len(dates), dates_per_block):
        last_date = min(first_date + dates_per_block, len(dates))
        first_row, last_row = boundaries[first_date], boundaries[last_date]
        block_rows = order[first_row:last_row]
        block_dates = ordered_dates[first_row:last_row] - first_date
        x_block = np.full((last_date - first_date, len(assets)), np.nan)
        x_block[block_dates, asset_codes[block_rows]] = values[block_rows]
        bins_block = assign_quantiles_batch(x_block, n_quantiles=20)
        for date_code in range(first_date, last_date):
            local_date = date_code - first_date
            row_start, row_stop = boundaries[date_code:date_code + 2]
            rows = order[row_start:row_stop]
            current_assets = asset_codes[rows]
            if previous_values is not None:
                good = np.flatnonzero(np.isfinite(previous_values) & (previous_bins >= 0))
                if len(good) == len(assets) and len(state.active_assets) == len(assets):
                    by_asset = state.step(previous_values, previous_bins)
                else:
                    by_asset = state._step_sparse_trusted(
                        previous_values[good], previous_bins[good], good
                    )
                output[rows] = by_asset[current_assets]
            previous_values = x_block[local_date]
            previous_bins = bins_block[local_date]
        previous_values = previous_values.copy()
        previous_bins = previous_bins.copy()
        del x_block, bins_block
    return pd.Series(output, index=frame.index, name="value")
