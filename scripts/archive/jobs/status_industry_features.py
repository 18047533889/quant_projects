"""Pure, local industry aggregation helpers for status features."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_ALIASES = {
    "date": ("observation_date", "TradeDate", "trade_date", "date"),
    "industry": ("industry", "industry_code", "sw_l1", "industry_name"),
    "return": ("ret_1d", "Return", "return", "daily_return"),
    "above_ma": ("above_ma", "above_ma_ratio", "above_ma_20", "pct_above_ma_20"),
    "ma_gap": ("ma_gap_20d", "ma_gap_20", "ma_gap"),
    "weight": ("market_cap", "mkt_cap", "amount", "Amount", "weight"),
}


def _rows(data: Any) -> list[Mapping[str, Any]]:
    if hasattr(data, "to_pylist"):
        return list(data.to_pylist())
    if isinstance(data, Mapping):
        return [data]
    return list(data)


def _value(row: Mapping[str, Any], field: str, columns: Mapping[str, str] | None) -> Any:
    if columns and field in columns:
        return row.get(columns[field])
    for name in _ALIASES[field]:
        if name in row:
            return row[name]
    return None


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _valid(values: Iterable[Any]) -> list[float]:
    return [x for x in (_number(value) for value in values) if not math.isnan(x)]


def _mean(values: Iterable[Any]) -> float:
    values = _valid(values)
    return sum(values) / len(values) if values else math.nan


def _std(values: Iterable[Any]) -> float:
    values = _valid(values)
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _share_weights(values: Iterable[Any]) -> list[float]:
    weights = [x for x in _valid(values) if x >= 0]
    total = sum(weights)
    return [x / total for x in weights] if total > 0 else []


def _hhi(shares: Iterable[float]) -> float:
    shares = list(shares)
    return sum(x * x for x in shares) if shares else math.nan


def _entropy(shares: Iterable[float], normalize: bool = False) -> float:
    shares = [x for x in shares if x > 0]
    if not shares:
        return math.nan
    value = -sum(x * math.log(x) for x in shares)
    return value / math.log(len(shares)) if normalize and len(shares) > 1 else value


def aggregate_industry_features(
    data: Any,
    *,
    columns: Mapping[str, str] | None = None,
    weight_field: str | None = None,
    return_field: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate by supplied ``(date, industry)`` observations.

    Input can be a PyArrow Table or an iterable of row mappings. Null and empty
    industry labels are retained. NaNs are excluded from each statistic, but
    their group is retained. No membership is inferred across dates.
    """
    rows = _rows(data)
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_value(row, "date", columns), _value(row, "industry", columns))].append(row)

    result: list[dict[str, Any]] = []
    for (date, industry), members in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        value_columns = dict(columns or {})
        if return_field:
            value_columns["return"] = return_field
        returns = [_value(row, "return", value_columns) for row in members]
        above = [_value(row, "above_ma", columns) for row in members]
        if all(value is None for value in above):
            gaps = [_number(_value(row, "ma_gap", columns)) for row in members]
            above = [1.0 if not math.isnan(x) and x > 0 else 0.0 if not math.isnan(x) else math.nan for x in gaps]
        valid_returns = _valid(returns)
        valid_above = _valid(above)
        weights = [_value(row, "weight", {**(columns or {}), "weight": weight_field} if weight_field else columns) for row in members]
        numeric_weights = [_number(x) for x in weights]
        weighted_pairs = [(v, w) for v, w in zip(returns, numeric_weights) if not math.isnan(_number(v)) and not math.isnan(w) and w >= 0]
        weighted_above = [(v, w) for v, w in zip(above, numeric_weights) if not math.isnan(_number(v)) and not math.isnan(w) and w >= 0]
        total_weight = sum(w for _, w in weighted_pairs)
        total_above_weight = sum(w for _, w in weighted_above)
        mean = _mean(valid_returns)
        ordered = sorted(valid_returns)
        p10 = ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)] if ordered else math.nan
        p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else math.nan
        shares = _share_weights([max(x, 0.0) for x in valid_returns])
        result.append({
            "observation_date": date,
            "industry": industry,
            "n_members": len(members),
            "n_valid_returns": len(valid_returns),
            "advancing_ratio": sum(x > 0 for x in valid_returns) / len(valid_returns) if valid_returns else math.nan,
            "advancing_ratio_weighted": sum(w for v, w in weighted_pairs if v > 0) / total_weight if total_weight > 0 else math.nan,
            "return_mean": mean,
            "return_dispersion": _std(valid_returns),
            "return_mad": _mean(abs(x - mean) for x in valid_returns) if valid_returns else math.nan,
            "return_p90_p10": p90 - p10 if valid_returns else math.nan,
            "above_ma_ratio": sum(x > 0 for x in valid_above) / len(valid_above) if valid_above else math.nan,
            "above_ma_ratio_weighted": sum(w for v, w in weighted_above if v > 0) / total_above_weight if total_above_weight > 0 else math.nan,
            "leadership_hhi": _hhi(shares),
            "leadership_entropy": _entropy(shares),
            "leadership_entropy_normalized": _entropy(shares, normalize=True),
        })
    return result


def aggregate_industry_status_features(data: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return aggregate_industry_features(data, **kwargs)


def to_pyarrow_table(rows: Sequence[Mapping[str, Any]]):
    import pyarrow as pa
    return pa.Table.from_pylist(list(rows))


__all__ = ["aggregate_industry_features", "aggregate_industry_status_features", "to_pyarrow_table"]
