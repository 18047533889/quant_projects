"""输入输出加载与 schema 校验。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：执行磁盘契约校验，并在计算前统一表结构与时间列。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import (
    COL_ASSET,
    COL_DATETIME,
    COL_DECISION_TS,
    COL_FACTOR_ID,
    COL_IS_ACTIVE,
    COL_IS_TRADABLE,
    COL_KNOWLEDGE_TS,
    REQUIRED_MARKET_COLUMNS_BASE,
    REQUIRED_PURE_FACTOR_COLUMNS,
    REQUIRED_UNIVERSE_COLUMNS,
)
from .validation import Severity, ValidationBuffer


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。

    入参：
        path: JSON 文件路径。

    出参：
        dict[str, Any]：解析后的 JSON 对象。
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_table(path: Path) -> pd.DataFrame:
    """加载表格文件。

    入参：
        path: 表格文件路径，支持 `.parquet`/`.pq`/`.csv`。

    出参：
        pd.DataFrame：加载后的表格。

    异常：
        ValueError：文件后缀不在支持列表时抛出。
    """
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table file format: {path}")


def _load_table_checked(path: Path, name: str, vb: ValidationBuffer) -> pd.DataFrame | None:
    """加载外部输入表并将文件级错误记录到验证缓冲区。

    入参：
        path: 输入文件绝对路径。
        name: 输入对象名（如 market_data / universe）。
        vb: 验证缓冲区。

    出参：
        pd.DataFrame | None：加载成功返回 DataFrame；失败返回 `None`。

    边界处理：
        - 文件缺失：记录 `<name>_file_missing` 错误。
        - 文件不可读/解析失败：记录 `<name>_file_unreadable` 错误。
    """
    if not path.exists():
        vb.add(
            Severity.ERROR,
            f"{name}_file_missing",
            f"{name} file not found: {path}",
        )
        return None
    try:
        return _load_table(path)
    except Exception as exc:  # pragma: no cover - 失败路径由调用方测试覆盖
        vb.add(
            Severity.ERROR,
            f"{name}_file_unreadable",
            f"{name} file cannot be loaded: {path}",
            details={"error": str(exc)},
        )
        return None


def _check_required_columns(df: pd.DataFrame, required: tuple[str, ...], name: str, vb: ValidationBuffer) -> None:
    """检查必需列是否齐全并记录错误。

    入参：
        df: 待校验 DataFrame。
        required: 必需列名集合。
        name: 表名（用于错误信息）。
        vb: 验证缓冲区。

    出参：
        无。若缺列则向 `vb` 写 ERROR。
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        vb.add(Severity.ERROR, "missing_columns", f"{name} missing required columns: {missing}")


def _check_unique_keys(df: pd.DataFrame, keys: list[str], name: str, vb: ValidationBuffer) -> None:
    """检查主键唯一性并记录 ERROR。"""
    dup = int(df.duplicated(keys).sum())
    if dup:
        vb.add(
            Severity.ERROR,
            f"{name}_duplicate_keys",
            f"{name} has {dup} duplicate rows on {keys}",
        )


def _standardize_market_schema(df: pd.DataFrame) -> pd.DataFrame:
    """将 market_data 别名映射为标准字段（FID §5.1）。

    仅用于 **market_data**（forward return），与 PureFactor 无关。
    PureFactor 上游格式见 `purification_pure_factor_base/{factor_id}/data.parquet`，已是标准列名。

    映射：`ticker->asset`，`trade_date->datetime`，`vw->vwap`，`o->open`。
    """
    out = df.copy()
    rename_map: dict[str, str] = {}
    if COL_ASSET not in out.columns and "ticker" in out.columns:
        rename_map["ticker"] = COL_ASSET
    if COL_DATETIME not in out.columns and "trade_date" in out.columns:
        rename_map["trade_date"] = COL_DATETIME
    if "vwap" not in out.columns and "vw" in out.columns:
        rename_map["vw"] = "vwap"
    if "open" not in out.columns and "o" in out.columns:
        rename_map["o"] = "open"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _status_is_passed(status: Any) -> bool:
    """判断 manifest 校验状态是否为通过。"""
    return str(status).strip().lower() in {"passed", "ok", "success"}


def normalize_datetime_column(df: pd.DataFrame, col: str = COL_DATETIME) -> pd.DataFrame:
    """将时间列解析并归一化到日级日期。

    入参：
        df: 原始 DataFrame。
        col: 时间列名，默认 `datetime`。

    出参：
        pd.DataFrame：丢弃无法解析时间后，并把时间归一化到日级的副本。
    """
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out.dropna(subset=[col]).copy()
    out[col] = out[col].dt.normalize()
    return out


def load_pure_factor_input(pure_factor_dir: Path, vb: ValidationBuffer) -> pd.DataFrame | None:
    """加载并校验一个 PureFactor 输入目录。

    入参：
        pure_factor_dir: 包含 `candidate.json`、`data.parquet`、`manifest.json` 的目录。
        vb: 接收 schema 与磁盘契约错误的验证缓冲区。

    标准列：`datetime, asset, factor_value, factor_id, knowledge_ts, decision_ts`。
    参考：`database/tier1/purification_pure_factor_base/alpha_101_001/data.parquet`。
    仅 `datetime` 归一化到日级；`decision_ts` 缺失时默认等于 `datetime`。

    校验项：
        - PureFactor 三件套文件必须存在。
        - Candidate 必须包含 Gateway/Assetization/Purification 段。
        - `manifest.artifact_type` 必须是 `PureFactor`。
        - folder/manifest/parquet 的 factor_id 必须一致；
          candidate 中若提供 `factor_id`，也必须一致。
        - `manifest.files` 必须引用 `candidate.json` 与 `data.parquet`。
        - `manifest.validation.status` 必须为 passed/ok/success。
        - parquet 必须是长表，且包含 `REQUIRED_PURE_FACTOR_COLUMNS`。
        - 主键 `(datetime, asset, factor_id)` 必须唯一。
    """
    pure_factor_dir = pure_factor_dir.resolve()
    candidate_path = pure_factor_dir / "candidate.json"
    data_path = pure_factor_dir / "data.parquet"
    manifest_path = pure_factor_dir / "manifest.json"
    missing = [p.name for p in (candidate_path, data_path, manifest_path) if not p.exists()]
    if missing:
        vb.add(
            Severity.ERROR,
            "pure_factor_missing_files",
            f"Missing PureFactor files under {pure_factor_dir}: {missing}",
        )
        return None

    candidate = _read_json(candidate_path)
    manifest = _read_json(manifest_path)
    for sec in ("Gateway", "Assetization", "Purification"):
        if sec not in candidate:
            vb.add(Severity.ERROR, "candidate_sections_missing", f"candidate.json missing section: {sec}")
            return None

    if manifest.get("artifact_type") != "PureFactor":
        vb.add(
            Severity.ERROR,
            "manifest_artifact_type_invalid",
            "manifest.artifact_type must be PureFactor",
        )
        return None

    files = manifest.get("files") or {}
    if files.get("candidate") != "candidate.json" or files.get("data") != "data.parquet":
        vb.add(
            Severity.ERROR,
            "manifest_files_invalid",
            "manifest.files must reference candidate.json and data.parquet",
            details={"files": files},
        )
        return None

    validation = manifest.get("validation") or {}
    if not _status_is_passed(validation.get("status")):
        vb.add(
            Severity.ERROR,
            "manifest_validation_failed",
            "manifest.validation.status must be passed/ok/success",
            details={"status": validation.get("status")},
        )
        return None

    df = _load_table(data_path)
    if isinstance(df.columns, pd.MultiIndex):
        vb.add(Severity.ERROR, "pure_factor_not_long_table", "PureFactor parquet must be long table")
        return None
    _check_required_columns(df, REQUIRED_PURE_FACTOR_COLUMNS, "pure_factor", vb)
    if vb.has_error():
        return None

    folder_factor_id = pure_factor_dir.name
    candidate_factor_id = str(candidate.get("factor_id") or "")
    manifest_factor_id = str(manifest.get("factor_id") or "")
    parquet_factor_ids = sorted({str(x) for x in df[COL_FACTOR_ID].dropna().unique()})
    if len(parquet_factor_ids) != 1:
        vb.add(
            Severity.ERROR,
            "factor_id_ambiguous",
            "pure_factor parquet must contain exactly one factor_id",
            details={"parquet_factor_ids": parquet_factor_ids},
        )
        return None
    parquet_factor_id = parquet_factor_ids[0]
    ids = {
        "folder": folder_factor_id,
        "candidate": candidate_factor_id,
        "manifest": manifest_factor_id,
        "parquet": parquet_factor_id,
    }
    if len({v for v in ids.values() if v}) != 1:
        vb.add(
            Severity.ERROR,
            "factor_id_mismatch",
            "factor_id mismatch across folder/candidate/manifest/parquet",
            details=ids,
        )
        return None

    if COL_DECISION_TS not in df.columns:
        df[COL_DECISION_TS] = df[COL_DATETIME]

    pk = [COL_DATETIME, COL_ASSET, COL_FACTOR_ID]
    dup = int(df.duplicated(pk).sum())
    if dup:
        vb.add(Severity.ERROR, "duplicate_keys", f"pure_factor has {dup} duplicate rows on {pk}")
        return None
    return normalize_datetime_column(df, COL_DATETIME)


def load_market_data(market_data_path: Path, vb: ValidationBuffer) -> pd.DataFrame | None:
    """加载并校验用于 forward return 的行情数据。

    入参：
        market_data_path: 行情表路径。
        vb: 接收缺列错误的验证缓冲区。

    出参：
        pd.DataFrame | None：必需列齐全时返回归一化行情表，否则返回 `None`。

    说明：
        本函数仅检查基础列 `datetime`、`asset`；
        `vwap`/`open` 的价格列规则由 `forward_returns.pick_price_column` 负责。
    """
    df = _load_table_checked(market_data_path.resolve(), "market_data", vb)
    if df is None:
        return None
    df = _standardize_market_schema(df)
    _check_required_columns(df, REQUIRED_MARKET_COLUMNS_BASE, "market_data", vb)
    if vb.has_error():
        return None
    out = normalize_datetime_column(df, COL_DATETIME)
    _check_unique_keys(out, [COL_DATETIME, COL_ASSET], "market_data", vb)
    if vb.has_error():
        return None
    return out


def load_universe(universe_path: Path, vb: ValidationBuffer) -> pd.DataFrame | None:
    """加载并校验 active/tradable universe 表。

    入参：
        universe_path: universe 表路径。
        vb: 接收缺列错误的验证缓冲区。

    出参：
        pd.DataFrame | None：必需列齐全时返回归一化 universe，否则返回 `None`。

    必需列：
        `datetime`、`asset`、`is_active`、`is_tradable`。
    """
    df = _load_table_checked(universe_path.resolve(), "universe", vb)
    if df is None:
        return None
    _check_required_columns(df, REQUIRED_UNIVERSE_COLUMNS, "universe", vb)
    if vb.has_error():
        return None
    out = normalize_datetime_column(df, COL_DATETIME)
    _check_unique_keys(out, [COL_DATETIME, COL_ASSET], "universe", vb)
    if vb.has_error():
        return None
    return out


def _parse_bool_strict(series: pd.Series, *, col_name: str, vb: ValidationBuffer) -> pd.Series | None:
    """将布尔列按显式词典严格解析，不接受隐式 truthy/falsy。"""
    normalized = series.astype("string").str.strip().str.lower()
    truthy = {"true", "1", "t", "yes", "y"}
    falsy = {"false", "0", "f", "no", "n"}

    is_true = normalized.isin(truthy)
    is_false = normalized.isin(falsy)
    parsed = pd.Series(np.nan, index=series.index, dtype=float)
    parsed.loc[is_true] = 1.0
    parsed.loc[is_false] = 0.0

    invalid = ~(is_true | is_false)
    if int(invalid.sum()) > 0:
        examples = series.loc[invalid].dropna().astype(str).head(5).tolist()
        vb.add(
            Severity.ERROR,
            "universe_bool_parse_failed",
            f"universe column `{col_name}` contains invalid boolean values",
            count=int(invalid.sum()),
            details={"column": col_name, "examples": examples},
        )
        return None
    return parsed.astype(bool)


def ensure_bool_universe(universe: pd.DataFrame, vb: ValidationBuffer) -> pd.DataFrame | None:
    """将 universe 的 active/tradable 字段按显式词典解析为布尔类型。

    入参：
        universe: 包含 `is_active` 与 `is_tradable` 的 DataFrame。

    出参：
        pd.DataFrame | None：成功返回布尔化副本；若存在非法取值返回 `None` 并写 ERROR。
    """
    out = universe.copy()
    parsed_active = _parse_bool_strict(out[COL_IS_ACTIVE], col_name=COL_IS_ACTIVE, vb=vb)
    parsed_tradable = _parse_bool_strict(out[COL_IS_TRADABLE], col_name=COL_IS_TRADABLE, vb=vb)
    if parsed_active is None or parsed_tradable is None:
        return None
    out[COL_IS_ACTIVE] = parsed_active
    out[COL_IS_TRADABLE] = parsed_tradable
    return out


def build_universe_from_factor_assets(factor_exposure: pd.DataFrame) -> pd.DataFrame:
    """当 universe 缺失时，用因子暴露中的资产全集构造兜底 universe。

    入参：
        factor_exposure: PureFactor 暴露表。

    出参：
        pd.DataFrame：最小 universe 表，字段为
        `datetime, asset, is_active, is_tradable`。

    边界处理：
        - 自动去重 `(datetime, asset)`。
        - 丢弃 datetime 或 asset 缺失行。
        - 兜底表统一设为 `is_active=True` 与 `is_tradable=True`。
    """
    out = factor_exposure[[COL_DATETIME, COL_ASSET]].dropna(subset=[COL_DATETIME, COL_ASSET]).drop_duplicates()
    out = out.copy()
    out[COL_IS_ACTIVE] = True
    out[COL_IS_TRADABLE] = True
    return out.reset_index(drop=True)

