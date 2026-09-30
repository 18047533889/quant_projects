"""Date-local tail saturation through FactorEngine's canonical winsorize."""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

_MAX_BATCH_CELLS = 1_000_000


def execute_fe_tail_saturation(frame: pd.DataFrame, *, quantile: float,
                               side: str) -> pd.Series:
    """Apply FE row-wise winsorization to each observed date cross-section.

    The optimizer's long panel can be sparse across dates. Because this
    operation is date-local, similar-width date rows can be batched without
    constructing a global date-by-asset panel. Local integer columns preserve
    every input observation, including the caller's row order.
    """
    if (isinstance(quantile, (bool, np.bool_))
            or not isinstance(quantile, numbers.Real)
            or not np.isfinite(quantile) or not 0.5 < quantile < 1.0):
        raise ValueError("quantile must be finite in (0.5, 1)")
    if side not in {"top", "bottom", "both"}:
        raise ValueError("side must be top, bottom, or both")
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float, name="value")

    if side == "top":
        lower, upper = 0.0, float(quantile)
    elif side == "bottom":
        lower, upper = 1.0 - float(quantile), 1.0
    else:
        lower, upper = 1.0 - float(quantile), float(quantile)

    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
    except ImportError as exc:
        raise RuntimeError(
            "TAIL_SATURATION is FE-owned, but FactorEngine is unavailable"
        ) from exc

    operator = OperatorRegistry.get("winsorize", backend="pandas_numpy", mode="any")
    if operator is None:
        raise RuntimeError("FactorEngine canonical 'winsorize' has no pandas_numpy backend")

    raw = frame["value"].to_numpy(dtype=float, copy=True)
    saturated = np.full(len(frame), np.nan, dtype=float)
    by_padded_width = {}
    for positions in frame.groupby("date", sort=False, observed=True).indices.values():
        positions = np.asarray(positions, dtype=np.intp)
        width = len(positions)
        padded_width = (
            (1 << (width - 1).bit_length()) if width <= _MAX_BATCH_CELLS else width
        )
        by_padded_width.setdefault(padded_width, []).append((positions, width))

    # Row-wise winsorize is independent across dates. Batch only dates with
    # similar observed row counts, padding with NaN (which FE excludes from
    # quantiles). Power-of-two width buckets limit padding to <2x, ordinal
    # columns avoid an asset pivot, and the target caps each batch's cells.
    for padded_width, date_rows in by_padded_width.items():
        rows_per_batch = max(1, _MAX_BATCH_CELLS // padded_width)
        columns = np.arange(padded_width)
        for start in range(0, len(date_rows), rows_per_batch):
            rows_batch = date_rows[start:start + rows_per_batch]
            values = np.full((len(rows_batch), padded_width), np.nan, dtype=float)
            for row, (positions, width) in enumerate(rows_batch):
                values[row, :width] = raw[positions]
            panel = pd.DataFrame(values, columns=columns)
            result = operator.calculate(panel, lower=lower, upper=upper)
            if (not isinstance(result, pd.DataFrame)
                    or result.shape != panel.shape
                    or not result.index.equals(panel.index)
                    or not result.columns.equals(panel.columns)):
                raise ValueError("FactorEngine 'winsorize' changed the cross-sectional axes")
            matrix = result.to_numpy(dtype=float, copy=False)
            for row, (positions, width) in enumerate(rows_batch):
                saturated[positions] = matrix[row, :width]
    return pd.Series(saturated, index=frame.index, name="value")


__all__ = ["execute_fe_tail_saturation"]
