from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar / value conversion
# ---------------------------------------------------------------------------
def _safe_value(v: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python primitives."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _dict_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make all values in a flat dict JSON-safe."""
    return {k: _safe_value(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Series / DataFrame → columnar dict  (with daily downsampling)
# ---------------------------------------------------------------------------
def _index_to_strings(idx: pd.Index) -> list:
    """Convert an index to a list of string representations."""
    if hasattr(idx, "strftime"):
        return idx.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    return [str(x) for x in idx]


def _safe_values_bulk(arr: np.ndarray) -> list:
    """Vectorised safe-value conversion for a numpy array."""
    out = arr.astype(object)
    mask = np.isnan(arr.astype(float, copy=False)) if arr.dtype.kind == "f" else np.zeros(len(arr), dtype=bool)
    out[mask] = None
    return out.tolist()


def _series_to_dict(s: pd.Series) -> Dict[str, list]:
    """Columnar representation: { index: [...], values: [...] }."""
    return {
        "index": _index_to_strings(s.index),
        "values": _safe_values_bulk(s.to_numpy(dtype=np.float64, na_value=np.nan)),
    }


def _dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Columnar representation: { index: [...], columns: { col: [...] } }."""
    columns = {}
    for col in df.columns:
        columns[col] = _safe_values_bulk(df[col].to_numpy(dtype=np.float64, na_value=np.nan))
    return {
        "index": _index_to_strings(df.index),
        "columns": columns,
    }


# ---------------------------------------------------------------------------
# Daily downsampling helpers
# ---------------------------------------------------------------------------
def _downsample_holding_pnl_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample minute-level holding PnL DataFrame to daily frequency.

    - pnl:           daily sum
    - cum_pnl:       last value per day (end-of-day equity)
    - net_position:  last value per day
    - long_active:   daily mean (fraction of day long)
    - short_active:  daily mean (fraction of day short)
    - transaction_cost (if present): daily sum
    """
    if df.empty:
        return df

    idx = pd.to_datetime(df.index)
    day_key = idx.date

    agg_spec = {
        "pnl": "sum",
        "cum_pnl": "last",
        "net_position": "last",
        "long_active": "mean",
        "short_active": "mean",
    }
    if "transaction_cost" in df.columns:
        agg_spec["transaction_cost"] = "sum"

    daily = df.groupby(day_key).agg(agg_spec)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# High-level serialisation for evaluation payloads
# ---------------------------------------------------------------------------
def serialize_horizon_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Serialise a single horizon evaluation payload to JSON-safe dict.

    Series data is downsampled to daily frequency.
    """
    return {
        "summary": _dict_safe(payload["summary"]),
        "daily_ic": _series_to_dict(payload["daily_ic"]),
        "daily_rank_ic": _series_to_dict(payload["daily_rank_ic"]),
        "layered_single_period": _series_to_dict(payload["layered_single_period"]),
        "holding_pnl": _dataframe_to_dict(
            _downsample_holding_pnl_daily(payload["holding_pnl"]),
        ),
        "holding_stats": _dict_safe(payload["holding_stats"]),
        "holding_pnl_with_cost": _dataframe_to_dict(
            _downsample_holding_pnl_daily(payload["holding_pnl_with_cost"]),
        ),
        "holding_stats_with_cost": _dict_safe(payload["holding_stats_with_cost"]),
    }


def serialize_status_result(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Serialise all status-group results to JSON-safe dict.

    Series data is downsampled to daily frequency.
    """
    out = {}
    for status_val, payload in results.items():
        out[status_val] = {
            "n_bars": payload["n_bars"],
            "ic_summary": _dict_safe(payload["ic_summary"]),
            "rank_ic_summary": _dict_safe(payload["rank_ic_summary"]),
            "daily_ic": _series_to_dict(payload["daily_ic"]),
            "daily_rank_ic": _series_to_dict(payload["daily_rank_ic"]),
            "holding_pnl": _dataframe_to_dict(
                _downsample_holding_pnl_daily(payload["holding_pnl"]),
            ),
            "holding_stats": _dict_safe(payload["holding_stats"]),
            "holding_pnl_with_cost": _dataframe_to_dict(
                _downsample_holding_pnl_daily(payload["holding_pnl_with_cost"]),
            ),
            "holding_stats_with_cost": _dict_safe(payload["holding_stats_with_cost"]),
            "holding_total_return": _safe_value(payload["holding_total_return"]),
            "holding_total_return_with_cost": _safe_value(
                payload["holding_total_return_with_cost"],
            ),
        }
    return out


def to_json(data: Dict[str, Any], **kwargs) -> str:
    """Dump a JSON-safe dict to a JSON string."""
    return json.dumps(data, ensure_ascii=False, **kwargs)
