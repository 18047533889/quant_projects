"""Rebuild candidate TRAIN input after failed shared-frame execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebuild_train_frame(time_axis, asset_axis, baseline_values):
    """Restore ordered axes and detached baseline values, on failure only.

    The caller supplies the original batch axes and independent baseline
    array, never the potentially corrupted candidate dataframe.
    """
    values = np.asarray(baseline_values)
    times = np.asarray(time_axis.values)
    assets = np.asarray(asset_axis.values)
    if (values.ndim != 2 or times.ndim != 1 or assets.ndim != 1
            or values.shape[0] > len(times) or values.shape[1] != len(assets)):
        raise ValueError("TRAIN baseline shape does not match original axes")
    return pd.DataFrame({
        "date": np.repeat(times[:values.shape[0]], values.shape[1]),
        "asset_id": np.tile(assets, values.shape[0]),
        "value": values.ravel(),
    }, copy=True)
