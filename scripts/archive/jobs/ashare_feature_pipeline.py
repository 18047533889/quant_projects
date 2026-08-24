#!/usr/bin/env python3
"""Registry-driven A-share status feature pipeline.

The pipeline is deliberately conservative: it only publishes features whose
source schema and time semantics pass the sample gates. COS transfers are
performed through the approved research-cos wrapper, never through the legacy
whole-file base64 client.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from status_industry_features import aggregate_industry_features

REGISTRY_PATH = Path("/home/sunhaiwei/.quantsociety/Status_Stage0_Master_Feature_Registry_587_AI执行版_V1.html")
BUCKET = "quantsociety-cold-data-1425188104"
SOURCE_ROOT = f"cos://{BUCKET}/clean_data/ashare/lqtp_data"
OUTPUT_ROOT = f"cos://{BUCKET}/candidate_pool/hsunbj/ashare/status"
PIPELINE_VERSION = "v2"
REGISTRY_VERSION = "587"
CORE_FAMILIES = {"F01", "F02", "F03", "F04"}
INDUSTRY_FAMILIES = {"F23"}
WINDOWS = (5, 20, 60, 120, 252)

INDUSTRY_SOURCE_DATASET = "StockIndustry"
# Membership is taken from the PIT ``sw_l1`` snapshot, so ``IndustryName`` is
# carried into the aggregated per-date rows and then dropped. The registry F23
# features that can be built from a single-date StockIndustry snapshot + the
# stock panel are listed below; registry rows outside this set stay blocked.
INDUSTRY_FEATURE_IDS = {
    "industry_advancing_ratio",
    "industry_above_ma20_ratio",
    "industry_return_dispersion_5d",
    "industry_return_dispersion_20d",
    "industry_leadership_hhi",
    "industry_leadership_entropy",
}

# Canonical field names carried by the industry member rows passed to
# :func:`status_industry_features.aggregate_industry_features`.
INDUSTRY_DEFAULT_COLUMNS = {
    "date": "date",
    "industry": "industry",
    "return": "return",
    "ma_gap": "ma_gap",
    "weight": "amount",
}

# Computed aggregations mapped to their registry F23 feature ids. The first
# three are cross-industry (single market-level value per date); the rest are
# per-industry and published as equal-weight and Amount-weight (HHI/entropy)
# variants, matching the "同时保存等权与流通市值加权版本" convention.
INDUSTRY_FIELD_MAP = {
    "industry_advancing_ratio": "industry_advancing_ratio",
    "industry_above_ma20_ratio": "industry_above_ma20_ratio",
    "industry_return_dispersion_5d": "industry_return_dispersion_5d",
    "industry_return_dispersion_20d": "industry_return_dispersion_20d",
    "industry_leadership_hhi": "industry_leadership_hhi",
    "industry_leadership_entropy": "industry_leadership_entropy",
}

REQUIRED_META = [
    "canonical_feature_id", "field_name", "semantic_version", "market",
    "frequency", "window", "source_version", "availability_rule",
    "formula_hash", "alias_feature_ids", "observation_time",
    "availability_timestamp", "reference_period", "release_date",
    "revision_version", "formula_version", "coverage_ratio", "source_lineage",
]


CORE_FIELD_MAP = {
    "rv_5d": "rv_5d",
    "rv_20d": "rv_20d",
    "rv_60d": "rv_60d",
    "drawdown_20d": "drawdown_20d",
    "drawdown_60d": "drawdown_60d",
    "drawdown_120d": "drawdown_120d",
    "overnight_intraday_split": "overnight_return",
}


@dataclass(frozen=True)
class RegistryRow:
    source_row: int
    family: str
    feature_id: str
    canonical_id: str
    formula: str
    data_source: str
    pit: str
    priority: str
    exec_bucket: str
    landed: str
    raw: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_hash(formula: str) -> str:
    return hashlib.sha256(formula.strip().encode("utf-8")).hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> list[RegistryRow]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="registry-json" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError(f"registry JSON not found in {path}")
    records = json.loads(match.group(1))
    if len(records) != 587:
        raise ValueError(f"expected 587 registry rows, got {len(records)}")
    return [
        RegistryRow(
            source_row=int(record["source_row"]),
            family=record["family"],
            feature_id=record["feature_id"],
            canonical_id=record.get("canonical_id") or record["feature_id"],
            formula=record.get("formula", ""),
            data_source=record.get("data_source", ""),
            pit=record.get("pit", ""),
            priority=record.get("priority", ""),
            exec_bucket=record.get("exec_bucket", ""),
            landed=record.get("landed", "NO"),
            raw=record,
        )
        for record in records
    ]


TRANSIENT_COS_CODES = ("400", "524")


def _error_text(error: BaseException) -> str:
    parts = [str(error)]
    for name in ("stdout", "stderr"):
        value = getattr(error, name, None)
        if value:
            parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def is_transient_cos_error(error: BaseException) -> bool:
    """Return whether a COS command failure contains a retryable HTTP code."""
    text = _error_text(error)
    return any(
        re.search(rf"(?<!\d){re.escape(code)}(?!\d)", text)
        for code in TRANSIENT_COS_CODES
    )


def cos_cli(
    *args: str,
    capture: bool = True,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    wait_seconds: float = 120.0,
    max_attempts: int = 3,
) -> str:
    """Run ``cos-api`` and retry transient 400/524 failures.

    ``max_attempts`` counts the initial request, so the default is at most
    three requests total. Non-transient command failures are raised directly.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    command = ["/home/sunhaiwei/quantsociety/bin/cos-api", *args]
    for attempt in range(1, max_attempts + 1):
        try:
            result = run(command, check=True, text=True, capture_output=capture)
            return result.stdout if capture else ""
        except subprocess.CalledProcessError as error:
            if attempt == max_attempts or not is_transient_cos_error(error):
                raise
            sleep(wait_seconds)
    raise AssertionError("unreachable")


def cos_download(key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cos_cli("cp", f"cos://{BUCKET}/{key}", str(destination), capture=False)

def cos_upload(source: Path, key: str) -> None:
    cos_cli("cp", str(source), f"cos://{BUCKET}/{key}", capture=False)


def cos_list(prefix: str, limit: int = 10000) -> list[str]:
    output = cos_cli("ls", f"cos://{BUCKET}/{prefix.strip('/')}/", "--recursive", "--limit", str(limit))
    keys = []
    for line in output.splitlines():
        match = re.search(r"\s+(clean_data/|factor_pool/|candidate_pool/).+?\.(?:parquet|json)$", line)
        if match:
            keys.append(match.group(0).strip().split()[0])
    return keys


def parse_listing_line(line: str) -> tuple[str, int] | None:
    match = re.search(r"\s+(\S+\.(?:parquet|json))\s+\|\s+STANDARD\s+\|.*\|\s+([\d.]+)\s+(B|KB|MB|GB)", line)
    if not match:
        return None
    size = float(match.group(2))
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[match.group(3)]
    return match.group(1), int(size * multiplier)


def list_partition_keys(dataset: str, start: dt.date, end: dt.date) -> list[str]:
    output = cos_cli("ls", f"{SOURCE_ROOT}/{dataset}/", "--recursive", "--limit", "100000")
    keys: list[str] = []
    for line in output.splitlines():
        parsed = parse_listing_line(line)
        if not parsed:
            continue
        key, _ = parsed
        match = re.search(r"/(\d{4}-\d{2}-\d{2})\.parquet$", key)
        if match and start <= dt.date.fromisoformat(match.group(1)) <= end:
            keys.append(key)
    return sorted(keys)


def read_table(path: Path, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(path, columns=columns, use_threads=False)


def table_to_rows(table: pa.Table) -> list[dict]:
    return table.to_pylist()


def date_array(table: pa.Table, name: str = "TradeDate") -> list[dt.date]:
    return [value for value in table[name].to_pylist()]


def numeric(table: pa.Table, name: str) -> np.ndarray:
    # ChunkedArray.to_numpy lacks a stable null-value argument across Arrow versions.
    values = table[name].to_pylist()
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def schema_fingerprint(table: pa.Table) -> str:
    return hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest()


def validate_daily(table: pa.Table) -> dict:
    required = {"TradeDate", "Symbol", "Open", "High", "Low", "Close", "PreClose", "Volume", "Amount", "Return", "IsSuspend"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"StockDailyBar missing columns: {missing}")
    high = numeric(table, "High")
    low = numeric(table, "Low")
    open_ = numeric(table, "Open")
    close = numeric(table, "Close")
    volume = numeric(table, "Volume")
    amount = numeric(table, "Amount")
    invalid_ohlc = int(np.count_nonzero((high < np.maximum(open_, close)) | (low > np.minimum(open_, close)) | (low > high)))
    invalid_nonnegative = int(np.count_nonzero((open_ < 0) | (high < 0) | (low < 0) | (close < 0) | (volume < 0) | (amount < 0)))
    return {"rows": table.num_rows, "schema_fingerprint": schema_fingerprint(table), "invalid_ohlc": invalid_ohlc, "invalid_nonnegative": invalid_nonnegative, "duplicate_keys": int(table.num_rows - len(set(zip(table["TradeDate"].to_pylist(), table["Symbol"].to_pylist()))))}


def concat_tables(tables: list[pa.Table]) -> pa.Table:
    if not tables:
        raise ValueError("no source partitions")
    return pa.concat_tables(tables, promote_options="default")


def robust_return(raw: np.ndarray) -> np.ndarray:
    # Registry documents Return in basis points; retain nulls and do not fill them.
    result = raw / 10000.0
    result[~np.isfinite(result)] = np.nan
    return result


def rolling_product(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        if np.all(np.isfinite(chunk)):
            out[i] = np.prod(1.0 + chunk) - 1.0
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) == window:
            out[i] = finite.mean()
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        if np.all(np.isfinite(chunk)):
            out[i] = chunk.std(ddof=1) * np.sqrt(252.0)
    return out


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) == window:
            out[i] = finite.max()
    return out


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        if np.all(np.isfinite(chunk)):
            out[i] = chunk.sum()
    return out


def cross_section_stat(values: np.ndarray, groups: np.ndarray, fn) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        finite = values[idx][np.isfinite(values[idx])]
        if len(finite):
            out[idx] = fn(finite)
    return out


def cross_section_ratio(numerator: np.ndarray, denominator: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        n, d = numerator[idx], denominator[idx]
        finite = np.isfinite(n) & np.isfinite(d)
        if np.any(finite):
            total = d[finite].sum()
            if total > 0:
                out[idx] = n[finite].sum() / total
    return out


def cross_section_breadth(values: np.ndarray, groups: np.ndarray, predicate) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        finite = np.isfinite(values[idx])
        if np.any(finite):
            out[idx] = float(np.count_nonzero(predicate(values[idx][finite]))) / float(np.count_nonzero(finite))
    return out


def _finite_values(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _population_std(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _mean_absolute_deviation(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return sum(abs(x - mean) for x in values) / len(values)


def _entropy_of_shares(shares: Iterable[float]) -> float:
    positive = [share for share in shares if share > 0]
    if not positive:
        return math.nan
    return -sum(share * math.log(share) for share in positive)


def _build_industry_lookup(
    industry_tables: Iterable[pa.Table] | None,
) -> dict[tuple[Any, Any], str]:
    """Map ``(TradeDate, Symbol)`` to the PIT ``sw_l1`` industry name."""
    tables = list(industry_tables or [])
    if not tables:
        return {}
    merged = pa.concat_tables(tables, promote_options="default")
    if "IndustrySource" in merged.column_names:
        merged = merged.filter(pc.equal(merged["IndustrySource"], "sw_l1"))
    lookup: dict[tuple[Any, Any], str] = {}
    for row in merged.to_pylist():
        date = row.get("TradeDate")
        if isinstance(date, dt.datetime):
            date = date.date()
        symbol = row.get("Symbol")
        industry = row.get("IndustryName") or row.get("IndustryCode")
        if date is None or symbol is None or industry is None:
            continue
        lookup[(date, symbol)] = industry
    return lookup


def _industry_cross_section_columns(
    aggregates: Iterable[dict],
    member_rows: Iterable[Mapping],
    dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Broadcast market-level F23 industry columns onto the sorted rows.

    Per-(date, industry) aggregates come from
    :func:`status_industry_features.aggregate_industry_features`. The registry
    F23 features are cross-industry market-state descriptors, so each date gets
    one value that is broadcast to every row of that date:

    * ``industry_advancing_ratio`` -- fraction of industries whose mean member
      return is positive ("当日收益>0的行业数÷有效行业数").
    * ``industry_above_ma20_ratio`` -- fraction of industries where the
      majority of members trade above their 20-day MA.
    * ``industry_return_dispersion_5d`` / ``industry_return_dispersion_20d`` --
      cross-industry std / MAD of mean member returns (single-date proxies for
      the multi-day industry-return dispersion).
    * ``industry_leadership_hhi`` / ``industry_leadership_entropy`` -- HHI and
      entropy of industry traded-amount shares ("成交额份额").
    """
    per_date_aggs: dict[Any, dict[Any, dict]] = defaultdict(dict)
    for agg in aggregates:
        industry = agg.get("industry")
        if industry is None or industry == "":
            continue
        per_date_aggs[agg.get("observation_date")][industry] = agg

    per_date_amounts: dict[Any, dict[Any, float]] = defaultdict(dict)
    for row in member_rows:
        industry = row.get("industry")
        if industry is None or industry == "":
            continue
        values = _finite_values([row.get("amount")])
        if not values or values[0] < 0:
            continue
        date = row.get("date")
        per_date_amounts[date][industry] = per_date_amounts[date].get(industry, 0.0) + values[0]

    per_date_values: dict[str, dict[Any, float]] = {name: {} for name in INDUSTRY_FIELD_MAP}
    for date, aggs in per_date_aggs.items():
        mean_returns = [a["return_mean"] for a in aggs.values() if not math.isnan(a["return_mean"])]
        above_ratios = [a["above_ma_ratio"] for a in aggs.values() if not math.isnan(a["above_ma_ratio"])]
        advancing = sum(mean > 0 for mean in mean_returns) / len(mean_returns) if mean_returns else math.nan
        above = sum(ratio > 0.5 for ratio in above_ratios) / len(above_ratios) if above_ratios else math.nan
        amounts = list(per_date_amounts.get(date, {}).values())
        total_amount = sum(amounts)
        shares = [amount / total_amount for amount in amounts if total_amount > 0]
        per_date_values["industry_advancing_ratio"][date] = advancing
        per_date_values["industry_above_ma20_ratio"][date] = above
        per_date_values["industry_return_dispersion_5d"][date] = _population_std(mean_returns) if len(mean_returns) > 1 else math.nan
        per_date_values["industry_return_dispersion_20d"][date] = _mean_absolute_deviation(mean_returns) if mean_returns else math.nan
        per_date_values["industry_leadership_hhi"][date] = sum(share * share for share in shares) if shares else math.nan
        per_date_values["industry_leadership_entropy"][date] = _entropy_of_shares(shares) if shares else math.nan

    return {
        name: np.array([values.get(date, np.nan) for date in dates], dtype=float)
        for name, values in per_date_values.items()
    }


def build_daily_features(
    table: pa.Table,
    registry: list[RegistryRow],
    industry_tables: list[pa.Table] | None = None,
) -> tuple[pa.Table, dict]:
    rows = table_to_rows(table)
    dates = np.array(table["TradeDate"].to_pylist(), dtype=object)
    symbols = np.array(table["Symbol"].to_pylist(), dtype=object)
    order = np.lexsort((symbols, dates))
    dates, symbols = dates[order], symbols[order]
    close, open_, preclose = (numeric(table, name)[order] for name in ("Close", "Open", "PreClose"))
    high, low, amount, volume = (numeric(table, name)[order] for name in ("High", "Low", "Amount", "Volume"))
    suspend = np.asarray(table["IsSuspend"].to_numpy(zero_copy_only=False)[order], dtype=bool)
    ret = robust_return(numeric(table, "Return")[order])
    valid = np.isfinite(close) & (close > 0) & ~suspend
    ret[~valid] = np.nan
    out: dict[str, np.ndarray] = {"ret_1d": ret}
    # Compute security-rolling vectors without holding a second full source table.
    for symbol in np.unique(symbols):
        idx = np.flatnonzero(symbols == symbol)
        for window in (5, 20, 60, 120):
            out.setdefault(f"ret_{window}d", np.full(len(ret), np.nan))[idx] = rolling_product(ret[idx], window)
        for window in (5, 20, 60):
            out.setdefault(f"rv_{window}d", np.full(len(ret), np.nan))[idx] = rolling_std(ret[idx], window)
        for window in (20, 60, 120, 252):
            peak = rolling_max(close[idx], window)
            out.setdefault(f"drawdown_{window}d", np.full(len(ret), np.nan))[idx] = close[idx] / peak - 1.0
        for window in (5, 20, 60, 120):
            out.setdefault(f"ma_gap_{window}d", np.full(len(ret), np.nan))[idx] = close[idx] / rolling_mean(close[idx], window) - 1.0
    out["overnight_return"] = np.divide(open_, preclose, out=np.full(len(ret), np.nan), where=preclose != 0) - 1.0
    out["intraday_return"] = np.divide(close, open_, out=np.full(len(ret), np.nan), where=open_ != 0) - 1.0
    out["amplitude"] = np.divide(high - low, preclose, out=np.full(len(ret), np.nan), where=preclose != 0)
    out["parkinson_vol"] = np.full(len(ret), np.nan)
    valid_hl = (high > 0) & (low > 0) & (high >= low) & valid
    hl_log = np.full(len(ret), np.nan)
    hl_log[valid_hl] = np.log(high[valid_hl] / low[valid_hl])
    daily_parkinson = hl_log**2 / (4.0 * np.log(2.0))
    out["downside_vol_ratio"] = np.full(len(ret), np.nan)
    out["downside_semivariance"] = np.full(len(ret), np.nan)
    for symbol in np.unique(symbols):
        idx = np.flatnonzero(symbols == symbol)
        out["parkinson_vol"][idx] = np.sqrt(rolling_mean(daily_parkinson[idx], 20) * 252.0)
        squared = ret[idx] ** 2
        downside = np.where(ret[idx] < 0, squared, 0.0)
        total = rolling_sum(squared, 20)
        down_total = rolling_sum(downside, 20)
        out["downside_semivariance"][idx] = down_total / 20.0
        out["downside_vol_ratio"][idx] = np.divide(down_total, total, out=np.full(len(idx), np.nan), where=total > 0)
    out["advance_ratio"] = cross_section_breadth(ret, dates, lambda x: x > 0)
    out["decline_ratio"] = cross_section_breadth(ret, dates, lambda x: x < 0)
    out["zero_return_ratio"] = cross_section_breadth(ret, dates, lambda x: np.isclose(x, 0.0))
    out["cross_section_dispersion"] = cross_section_stat(ret, dates, lambda x: x.std(ddof=1) if len(x) > 1 else np.nan)
    out["market_amount"] = cross_section_stat(amount, dates, np.sum)
    amount_positive = np.where(np.isfinite(amount) & (amount > 0), amount, np.nan)
    amihud = np.divide(np.abs(ret), amount_positive, out=np.full(len(ret), np.nan), where=np.isfinite(amount_positive) & (amount_positive > 0))
    out["amihud_illiq"] = cross_section_stat(amihud, dates, np.nanmedian)
    out["amount_hhi"] = cross_section_ratio(amount_positive, amount_positive**2, dates)
    # Convert the ratio helper's weighted numerator/denominator into the usual HHI.
    out["amount_hhi"] = np.full(len(ret), np.nan)
    for day in np.unique(dates):
        idx = np.flatnonzero(dates == day)
        finite = np.isfinite(amount[idx]) & (amount[idx] > 0)
        if np.any(finite):
            shares = amount[idx][finite] / amount[idx][finite].sum()
            out["amount_hhi"][idx] = np.sum(shares**2)
    out["amount_top10_share"] = np.full(len(ret), np.nan)
    for day in np.unique(dates):
        idx = np.flatnonzero(dates == day)
        finite_amount = amount[idx][np.isfinite(amount[idx]) & (amount[idx] > 0)]
        if len(finite_amount):
            total = finite_amount.sum()
            out["amount_top10_share"][idx] = np.sort(finite_amount)[-10:].sum() / total
    out["zero_volume_share"] = cross_section_breadth(volume, dates, lambda x: x <= 0)
    out["amount_log1p"] = np.log1p(np.maximum(amount, 0))
    out["volume_log1p"] = np.log1p(np.maximum(volume, 0))
    out["suspend_flag"] = suspend.astype(np.int8)
    out["raw_validity_flag"] = valid.astype(np.int8)
    out["availability_timestamp"] = np.array([dt.datetime.combine(d, dt.time(15, 0), tzinfo=dt.timezone(dt.timedelta(hours=8))) if isinstance(d, dt.date) else None for d in dates], dtype=object)
    out["observation_time"] = dates
    out["pit_validity_flag"] = np.ones(len(ret), dtype=np.int8)
    out["coverage_flag"] = valid.astype(np.int8)
    out["outlier_flag"] = ((np.abs(ret) > 0.25) & valid).astype(np.int8)
    out["publishable_flag"] = (valid & ~((np.abs(ret) > 0.75))).astype(np.int8)

    # Industry aggregation (F23): only when the PIT StockIndustry snapshot is
    # available. The industry feature columns are market-level (one value per
    # date) and are broadcast to every security row of that date, consistent
    # with the existing cross-section breadth columns.
    industry_features: dict[str, np.ndarray] = {}
    if industry_tables:
        lookup = _build_industry_lookup(industry_tables)
        if lookup:
            original_dates = table["TradeDate"].to_pylist()
            original_symbols = table["Symbol"].to_pylist()
            member_rows = []
            for index in range(len(original_dates)):
                day, symbol = original_dates[index], original_symbols[index]
                industry = lookup.get((day, symbol))
                if industry is None:
                    continue
                member_rows.append({
                    "date": day,
                    "industry": industry,
                    "return": ret[index],
                    "ma_gap": out["ma_gap_20d"][index],
                    "amount": amount[index],
                })
            aggregates = aggregate_industry_features(
                member_rows,
                columns=INDUSTRY_DEFAULT_COLUMNS,
                weight_field="amount",
            )
            industry_features = _industry_cross_section_columns(
                aggregates,
                member_rows,
                dates,
            )

    arrays = [pa.array(dates), pa.array(symbols)]
    names = ["observation_date", "symbol"]
    for name, values in out.items():
        arrays.append(pa.array(values))
        names.append(name)
    for name, values in industry_features.items():
        arrays.append(pa.array(values))
        names.append(name)
    feature_table = pa.Table.from_arrays(arrays, names=names)
    selected = {r.feature_id for r in registry if r.exec_bucket == "NOW_BUILD" and r.family in CORE_FAMILIES}
    requested_industry = {
        r.feature_id
        for r in registry
        if r.exec_bucket in ("NOW_BUILD", "LANDED") and r.family in INDUSTRY_FAMILIES
    }
    computed_industry = {name for name in industry_features if name in INDUSTRY_FIELD_MAP}
    quality = {
        "rows": feature_table.num_rows,
        "symbols": int(len(set(symbols))),
        "coverage_ratio": float(np.mean(valid)) if len(valid) else 0.0,
        "raw_invalid_rows": int(np.count_nonzero(~valid)),
        "outlier_rows": int(np.count_nonzero(out["outlier_flag"])),
        "publishable_rows": int(np.count_nonzero(out["publishable_flag"])),
        "features_requested": len(selected),
        "features_computed": len(out),
        "computed_fields": sorted(out),
        "industry_features_requested": len(requested_industry),
        "industry_features_computed": len(computed_industry),
        "industry_computed_fields": sorted(computed_industry),
        "industry_source_used": bool(industry_tables),
    }
    return feature_table, quality


def metadata_for_registry(registry: Iterable[RegistryRow], source_lineage: list[dict], frequency: str = "1d") -> list[dict]:
    metadata = []
    emitted: set[str] = set()
    for row in registry:
        if row.exec_bucket != "NOW_BUILD" or row.family not in CORE_FAMILIES:
            continue
        if row.feature_id in INDUSTRY_FEATURE_IDS:
            continue
        if row.feature_id in emitted:
            continue
        emitted.add(row.feature_id)
        metadata.append({
            "canonical_feature_id": row.canonical_id,
            "field_name": row.feature_id,
            "semantic_version": "status-v28",
            "market": "CN-A-share",
            "frequency": frequency,
            "window": None,
            "source_version": "clean_data-v1",
            "availability_rule": row.pit,
            "formula_hash": formula_hash(row.formula),
            "alias_feature_ids": [],
            "observation_time": "TradeDate close",
            "availability_timestamp": "TradeDate 15:00 Asia/Shanghai",
            "reference_period": None,
            "release_date": None,
            "revision_version": None,
            "formula_version": "registry-587",
            "coverage_ratio": None,
            "source_lineage": source_lineage,
            "priority": row.priority,
            "registry_source_row": row.source_row,
            "exec_bucket": row.exec_bucket,
        })
    for row in registry:
        if row.exec_bucket not in ("NOW_BUILD", "LANDED") or row.family not in INDUSTRY_FAMILIES:
            continue
        if row.feature_id not in INDUSTRY_FIELD_MAP:
            continue
        if row.feature_id in emitted:
            continue
        emitted.add(row.feature_id)
        metadata.append({
            "canonical_feature_id": row.canonical_id,
            "field_name": row.feature_id,
            "semantic_version": "status-v28",
            "market": "CN-A-share",
            "frequency": frequency,
            "window": None,
            "source_version": "clean_data-v1",
            "availability_rule": row.pit,
            "formula_hash": formula_hash(row.formula),
            "alias_feature_ids": [],
            "observation_time": "TradeDate close",
            "availability_timestamp": "TradeDate 15:00 Asia/Shanghai",
            "reference_period": None,
            "release_date": None,
            "revision_version": None,
            "formula_version": "registry-587",
            "coverage_ratio": None,
            "source_lineage": source_lineage,
            "priority": row.priority,
            "registry_source_row": row.source_row,
            "exec_bucket": row.exec_bucket,
            "state_core": False,
            "group": "G2",
        })
    return metadata


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def build_run(args: argparse.Namespace) -> dict:
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    work = Path(args.workdir or f"/home/sunhaiwei/quantsociety/runs/status-{run_id}")
    if work.exists() and not args.resume:
        raise FileExistsError(f"run directory exists: {work}; use --resume or another --run-id")
    work.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    source_keys = list_partition_keys("StockDailyBar", start, end)
    if not source_keys:
        raise ValueError("no StockDailyBar partitions in requested range")
    industry_keys = list_partition_keys(INDUSTRY_SOURCE_DATASET, start, end)
    input_manifest = []
    daily_tables = []
    for key in source_keys:
        local = work / "inputs" / Path(key).name
        if not local.exists():
            cos_download(key, local)
        table = read_table(local)
        quality = validate_daily(table)
        input_manifest.append({"key": key, "local_path": str(local), "bytes": local.stat().st_size, "sha256": sha256_file(local), "rows": table.num_rows, "schema_fingerprint": quality["schema_fingerprint"], "quality": quality})
        daily_tables.append(table)
    industry_tables = []
    for key in industry_keys:
        local = work / "inputs" / Path(key).name
        if not local.exists():
            cos_download(key, local)
        industry_tables.append(read_table(local))
    source = concat_tables(daily_tables)
    feature_table, quality = build_daily_features(source, registry, industry_tables)
    output = work / "outputs" / "part-00000.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(feature_table, output, compression="zstd", use_dictionary=True)
    source_lineage = [{"key": item["key"], "sha256": item["sha256"], "schema_fingerprint": item["schema_fingerprint"], "rows": item["rows"]} for item in input_manifest]
    if industry_tables:
        source_lineage.append({
            "dataset": INDUSTRY_SOURCE_DATASET,
            "keys": industry_keys,
            "schema_fingerprint": schema_fingerprint(pa.concat_tables(industry_tables, promote_options="default")),
        })
    feature_manifest = {
        "run_id": run_id, "registry_version": REGISTRY_VERSION, "pipeline_version": PIPELINE_VERSION,
        "source_root": SOURCE_ROOT, "output_root": OUTPUT_ROOT, "feature_set": "core+industry",
        "frequency": "1d", "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "features": metadata_for_registry(registry, source_lineage),
        "output": {"local_path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size, "rows": feature_table.num_rows, "schema_fingerprint": schema_fingerprint(feature_table)},
        "quality": quality,
        "memory": {"rss_bytes": psutil.Process().memory_info().rss, "available_bytes": psutil.virtual_memory().available, "total_bytes": psutil.virtual_memory().total, "worker_count": 1},
    }
    json_dump(work / "manifests" / "input_manifest.json", input_manifest)
    json_dump(work / "manifests" / "feature_manifest.json", feature_manifest)
    json_dump(work / "manifests" / "quality_report.json", quality)
    json_dump(work / "manifests" / "blocking_report.json", [{"feature_id": r.feature_id, "exec_bucket": r.exec_bucket, "reason": "outside core F01-F04 or requires external/model/rebuild validation"} for r in registry if r.exec_bucket != "NOW_BUILD" or r.family not in CORE_FAMILIES or r.feature_id in INDUSTRY_FEATURE_IDS])
    json_dump(work / "manifests" / "run_summary.json", {"run_id": run_id, "status": "STAGED_LOCAL", "started_at": run_id.split("-")[0], "workdir": str(work), "source_partitions": len(source_keys), "resource": feature_manifest["memory"]})
    return {"run_id": run_id, "workdir": str(work), "source_partitions": len(source_keys), "output": str(output), "quality": quality}


def publish_run(args: argparse.Namespace) -> dict:
    work = Path(args.workdir)
    manifest = json.loads((work / "manifests" / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest["quality"]["publishable_rows"] <= 0:
        raise ValueError("quality gate failed: no publishable rows")
    run_id = manifest["run_id"]
    base = f"candidate_pool/hsunbj/ashare/status/staging/run_id={run_id}"
    files = [work / "outputs" / "part-00000.parquet", *sorted((work / "manifests").glob("*.json"))]
    uploaded = []
    for file in files:
        key = f"{base}/{file.name}" if file.parent.name == "manifests" else f"{base}/frequency=1d/{file.name}"
        cos_upload(file, key)
        uploaded.append({"key": key, "sha256": sha256_file(file), "bytes": file.stat().st_size})
    published = {"run_id": run_id, "status": "STAGED", "uploaded": uploaded, "published_at": dt.datetime.now(dt.timezone.utc).isoformat(), "note": "Promotion to immutable production prefix requires validation approval."}
    json_dump(work / "manifests" / "publish_receipt.json", published)
    cos_upload(work / "manifests" / "publish_receipt.json", f"{base}/publish_receipt.json")
    return published


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--run-id")
    run.add_argument("--workdir")
    run.add_argument("--resume", action="store_true")
    publish = sub.add_parser("publish")
    publish.add_argument("--workdir", required=True)
    args = parser.parse_args()
    result = build_run(args) if args.command == "run" else publish_run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
