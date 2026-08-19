"""
Shared result-alignment helpers for per-date neutralizers.

The neutralizers merge ``values`` with ``exposures`` on (date, asset) and
must return residuals aligned to the ORIGINAL ``values`` row order.  Merging
resets/loses the row index, and reindexing by the merged frame's index
misaligns rows whenever ``values.index`` is non-unique or the merged frame
has extra rows (duplicate (date, asset) keys in exposures).  The safe
pattern — already used by ``neutralization.ols`` — carries a positional row
key through the merge and aligns on it; these helpers give every
neutralizer the same behavior.
"""
import numpy as np
import pandas as pd

_ROW_KEY_PREFIX = "__row_position__"


def merged_exposure_cols(
    exposures: pd.DataFrame,
    values: pd.DataFrame,
    date_col: str,
    asset_col: str,
) -> list:
    """Exposure column names as they appear in the merged frame.

    ``merge(..., suffixes=("", "_exp"))`` keeps a values-frame column under
    its original name when it collides with an exposure name; selecting the
    un-suffixed name would then regress on values-frame data.  Any exposure
    column that also exists in ``values`` must be read via its ``_exp`` alias.
    """
    cols = [c for c in exposures.columns if c not in (date_col, asset_col)]
    if not cols:
        raise ValueError("No exposure columns found")
    return [f"{c}_exp" if c in values.columns else c for c in cols]


def nan_residuals(y: np.ndarray) -> np.ndarray:
    """All-NaN residual array matching ``y``'s shape.

    ``np.full_like(y, np.nan)`` on an integer-dtype ``y`` produces the
    integer sentinel (INT64_MIN), not NaN — a garbage extreme value that
    flows downstream looking like a signal.
    """
    return np.full(y.shape, np.nan, dtype=np.float64)


def add_position_key(values: pd.DataFrame) -> tuple:
    """Return (copy of values with a unique positional key column, key name)."""
    row_key = _ROW_KEY_PREFIX
    while row_key in values.columns:
        row_key = f"_{row_key}"
    out = values.copy()
    out[row_key] = np.arange(len(out), dtype=np.intp)
    return out, row_key


def align_residuals_to_values(
    result_frames: list,
    row_key: str,
    values_index: pd.Index,
    value_name: str = "residual",
) -> pd.Series:
    """Concatenate per-date result frames and align to ``values_index`` positionally.

    ``result_frames`` entries must carry the positional key as their index
    (see ``add_position_key``).  Rows absent from the merged frame become
    NaN; duplicated positions keep the first occurrence.
    """
    n = len(values_index)
    if not result_frames:
        return pd.Series(np.nan, index=values_index, name=value_name)
    all_results = pd.concat(result_frames)
    # A duplicate (date, asset) key in ``exposures`` multiplies merged rows,
    # so one positional key can appear more than once; keep the first
    # occurrence so the result always has exactly one row per input row.
    all_results = all_results[~all_results.index.duplicated(keep="first")]
    residuals = all_results[value_name].reindex(range(n))
    residuals.index = values_index
    residuals.name = value_name
    return residuals
