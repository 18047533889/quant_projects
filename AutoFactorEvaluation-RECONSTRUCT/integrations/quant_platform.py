"""AutoFactorEvaluation 到仓库根 ``factor_engine`` / ``data_access`` 的唯一桥接层。

业务模块不得再自行拼接 ``sys.path``、直接扫描行情 Parquet，或引用目录内复制的
FactorEngine。所有生产计算、行情读取、快照 lineage 与因子 staging 写入均从这里进入。
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa


@dataclass(frozen=True)
class MarketDataSpec:
    market: str
    dataset: str
    time_column: str
    instrument_column: str
    fields: dict[str, str]


_MARKET_SPECS: dict[str, MarketDataSpec] = {
    "ashare": MarketDataSpec(
        market="ashare",
        dataset="ashare_stock_daily",
        time_column="TradeDate",
        instrument_column="Symbol",
        fields={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "preclose": "PreClose",
            "volume": "Volume",
            "amount": "Amount",
            "ret": "Return",
            "vwap": "Vwap",
        },
    ),
    "us": MarketDataSpec(
        market="us",
        dataset="us_stock_daily",
        time_column="TradeDate",
        instrument_column="Ticker",
        fields={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "preclose": "PreClose",
            "volume": "Volume",
            "amount": "Amount",
            "ret": "Ret",
            "vwap": "VWAP",
        },
    ),
}

_MARKET_ALIASES = {
    "a_share": "ashare",
    "a-share": "ashare",
    "cn": "ashare",
    "china": "ashare",
    "us_stock": "us",
    "usa": "us",
    "": "ashare",
}


def _find_quant_projects_root() -> Path:
    configured = os.environ.get("QUANT_PROJECTS_ROOT", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    for candidate in candidates:
        if (
            (candidate / "factor_engine" / "runtime" / "engine.py").is_file()
            and (candidate / "data_access" / "store.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "无法定位 quant_projects 根目录；请设置 QUANT_PROJECTS_ROOT，且目录中必须包含 "
        "factor_engine/ 与 data_access/"
    )


def bootstrap_quant_platform() -> Path:
    """一次性引导当前 monorepo 的 FactorEngine 平铺包与 DataAccess。"""
    root = _find_quant_projects_root()
    factor_engine_root = root / "factor_engine"
    # FactorEngine 内部仍使用 ``from api...`` / ``from runtime...`` 平铺导入，
    # factor_engine 根必须位于仓库根之前；集中在此处理，业务模块不再自行改 sys.path。
    for value in (str(factor_engine_root), str(root)):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return root


bootstrap_quant_platform()

from api.dsl_parser import parse_factor  # noqa: E402
from backend.context import ExecutionContext  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402
from storage.sources.data_access_source import DataAccessSource  # noqa: E402
from storage.sources.datasource import DataSource  # noqa: E402


@dataclass
class FactorExecution:
    factor_name: str
    result: pd.Series
    snapshot_id: str | None
    plan: Any
    analysis: Any
    runtime: dict[str, Any]


class InMemoryFrameSource(DataSource):
    """把长表或 MultiIndex DataFrame 适配为当前 FactorEngine ``DataSource``。"""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        time_column: str | None = None,
        instrument_column: str | None = None,
    ) -> None:
        self._columns: dict[str, pd.Series] = {}
        self.data_snapshot_id = "in_memory:empty"
        if frame is None:
            return
        self._load_frame(
            frame,
            time_column=time_column,
            instrument_column=instrument_column,
        )

    @staticmethod
    def _axis_columns(frame: pd.DataFrame) -> tuple[str, str]:
        time_candidates = ("datetime", "TradeDate", "trade_date", "align_time", "timestamp")
        instrument_candidates = ("asset", "Symbol", "Ticker", "ticker", "symbol")
        time_col = next((c for c in time_candidates if c in frame.columns), None)
        instrument_col = next((c for c in instrument_candidates if c in frame.columns), None)
        if time_col is None or instrument_col is None:
            raise ValueError(
                "内存行情必须含时间列和标的列；支持 "
                f"time={time_candidates}, instrument={instrument_candidates}"
            )
        return time_col, instrument_col

    def _load_frame(
        self,
        frame: pd.DataFrame,
        *,
        time_column: str | None,
        instrument_column: str | None,
    ) -> None:
        if isinstance(frame.index, pd.MultiIndex):
            panel = frame.copy()
            if panel.index.nlevels != 2:
                raise ValueError("FactorEngine 内存源要求二级 MultiIndex(time, instrument)")
            panel = panel.sort_index()
        else:
            tc, ic = self._axis_columns(frame)
            time_column = time_column or tc
            instrument_column = instrument_column or ic
            panel = frame.copy()
            panel[time_column] = pd.to_datetime(panel[time_column], errors="raise")
            if panel.duplicated([time_column, instrument_column]).any():
                dup = panel.loc[
                    panel.duplicated([time_column, instrument_column], keep=False),
                    [time_column, instrument_column],
                ].head(10)
                raise ValueError(f"内存行情存在重复主键:\n{dup}")
            panel = panel.set_index([time_column, instrument_column]).sort_index()
        self._columns = {str(col): panel[col] for col in panel.columns}
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(panel.index.to_frame(index=False), index=False)
            .to_numpy()
            .tobytes()
        ).hexdigest()[:20]
        self.data_snapshot_id = f"in_memory:{digest}"

    def load_column(self, name: str) -> pd.Series:
        try:
            return self._columns[name]
        except KeyError as exc:
            raise KeyError(
                f"列 {name!r} 不存在；可用列={sorted(self._columns)}"
            ) from exc

    def load_columns(self, names: list[str]) -> dict[str, pd.Series]:
        return {name: self.load_column(name) for name in names}

    def prefetch_columns(self, names: list[str]) -> None:
        for name in names:
            self.load_column(name)

    def close(self) -> None:
        self._columns.clear()


def get_market_data_spec(
    market: str | None,
    *,
    dataset: str | None = None,
    fields: Mapping[str, str] | None = None,
) -> MarketDataSpec:
    normalized = str(market or "").strip().lower()
    normalized = _MARKET_ALIASES.get(normalized, normalized)
    if normalized not in _MARKET_SPECS:
        raise ValueError(f"不支持的市场 {market!r}；当前支持 {sorted(_MARKET_SPECS)}")
    base = _MARKET_SPECS[normalized]
    return MarketDataSpec(
        market=base.market,
        dataset=str(dataset or base.dataset),
        time_column=base.time_column,
        instrument_column=base.instrument_column,
        fields={**base.fields, **dict(fields or {})},
    )


def _manifest_date(manifest: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = manifest.get(name)
        if value not in (None, ""):
            return str(value)
    data = manifest.get("data_source")
    if isinstance(data, Mapping):
        for name in names:
            value = data.get(name)
            if value not in (None, ""):
                return str(value)
    return None


def _manifest_instruments(manifest: Mapping[str, Any]) -> list[str] | None:
    value = manifest.get("instrument_filter")
    if value is None and isinstance(manifest.get("data_source"), Mapping):
        value = manifest["data_source"].get("instrument_filter")
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def build_factor_engine_config(
    manifest: Mapping[str, Any],
    *,
    backend: str = "pandas",
    run_mode: str = "research",
) -> dict[str, Any]:
    """返回可审计的当前 FactorEngine/DataAccess 配置字典（不落临时 YAML）。"""
    data_cfg = manifest.get("data_source") if isinstance(manifest.get("data_source"), Mapping) else {}
    spec = get_market_data_spec(
        str(manifest.get("market", "ashare")),
        dataset=data_cfg.get("dataset") or manifest.get("dataset"),
        fields=data_cfg.get("fields") if isinstance(data_cfg.get("fields"), Mapping) else None,
    )
    return {
        "factor": {
            "name": str(manifest.get("candidate_id") or manifest.get("factor_id") or "autofactor"),
            "expr": str(manifest.get("formula", "")),
            "freq": str(manifest.get("frequency_bucket", "1d")),
            "universe": manifest.get("universe_id"),
            "description": str(manifest.get("description", "")),
        },
        "data_source": {
            "type": "data_access",
            "dataset": spec.dataset,
            "fields": dict(spec.fields),
            "start_date": _manifest_date(manifest, "start_date", "start"),
            "end_date": _manifest_date(manifest, "end_date", "end"),
            "instrument_filter": _manifest_instruments(manifest),
            "read_auto": True,
            "params": dict(data_cfg.get("params") or {}),
        },
        "backend": {"type": str(backend)},
        "run": {"mode": str(run_mode)},
    }


def _assert_dsl_expression(formula: str, expression_type: str | None = None) -> None:
    if str(expression_type or "dsl").lower() in {"python", "code"}:
        raise ValueError(
            "AutoFactorEvaluation 生产接入禁止执行任意 Python 因子代码；"
            "请先转换为 FactorEngine DSL 并通过 Gateway 白名单"
        )
    if not str(formula or "").strip():
        raise ValueError("因子公式不能为空")


def validate_factor_formula(
    formula: str,
    *,
    name: str = "autofactor_validation",
    freq: str = "1d",
    universe: str | None = None,
    backend: str = "pandas",
    run_mode: str = "research",
) -> tuple[Any, Any, Any]:
    """用仓库根最新 FactorEngine 做真实 parser/analyzer/planner 编译。"""
    _assert_dsl_expression(formula)
    factor = parse_factor(
        formula,
        name=name,
        freq=freq,
        universe=universe,
        description="AutoFactorEvaluation validation",
    )
    source = InMemoryFrameSource()
    engine = FactorEngine(build_backend(backend), source, run_mode=run_mode)
    plan, analysis = engine.compile(
        factor,
        pit_enforce=run_mode == "production",
        pit_forbid_forward_fill=run_mode == "production",
    )
    return factor, plan, analysis


def _coerce_result_series(value: Any, *, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.rename(name)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise TypeError(f"因子结果 DataFrame 必须只有一列，收到 {value.shape[1]} 列")
        return value.iloc[:, 0].rename(name)
    if hasattr(value, "to_pandas"):
        converted = value.to_pandas()
        if isinstance(converted, pd.Series):
            return converted.rename(name)
    raise TypeError(f"FactorEngine 返回类型不受支持: {type(value).__name__}")


def execute_factor_formula(
    formula: str,
    *,
    factor_name: str,
    market: str = "ashare",
    dataset: str | None = None,
    fields: Mapping[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    backend: str = "pandas",
    run_mode: str = "research",
    freq: str = "1d",
    universe: str | None = None,
    description: str | None = None,
    expression_type: str = "dsl",
) -> FactorExecution:
    """以 DataAccessSource 读取登记数据集，并由当前 FactorEngine 执行一条 DSL。"""
    _assert_dsl_expression(formula, expression_type)
    spec = get_market_data_spec(market, dataset=dataset, fields=fields)
    source = DataAccessSource(
        dataset=spec.dataset,
        fields=dict(spec.fields),
        start_date=start_date,
        end_date=end_date,
        instrument_filter=list(instrument_filter) if instrument_filter else None,
        read_auto=True,
        params=dict(params or {}),
    )
    production = str(run_mode).lower() == "production"
    try:
        snapshot_id = source.refresh_snapshot(force=True)
        factor = parse_factor(
            formula,
            name=factor_name,
            freq=freq,
            universe=universe,
            description=description,
        )
        engine = FactorEngine(
            backend=build_backend(backend),
            data_source=source,
            run_mode=run_mode,
        )
        plan, analysis = engine.compile(
            factor,
            pit_enforce=production,
            pit_forbid_forward_fill=production,
        )
        runtime = engine.run(
            factor,
            plan=plan,
            analysis=analysis,
            input_dq_check=production,
            input_dq_strict=True,
            auto_warmup=production,
            trim_warmup=True,
            market=spec.market,
            pit_enforce=production,
            pit_forbid_forward_fill=production,
        )
        series = _coerce_result_series(runtime.get("result"), name=factor_name)
        if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
            raise ValueError("FactorEngine 结果必须为 MultiIndex(datetime, asset) Series")
        series = series.sort_index()
        return FactorExecution(
            factor_name=factor_name,
            result=series,
            snapshot_id=source.data_snapshot_id or snapshot_id,
            plan=plan,
            analysis=analysis,
            runtime=runtime,
        )
    finally:
        source.close()


def execute_factor_on_frame(
    formula: str,
    frame: pd.DataFrame,
    *,
    factor_name: str = "autofactor_frame",
    backend: str = "pandas",
    run_mode: str = "research",
    time_column: str | None = None,
    instrument_column: str | None = None,
) -> FactorExecution:
    """测试、调试与外部输入场景：同一最新引擎在内存长表上执行。"""
    _assert_dsl_expression(formula)
    source = InMemoryFrameSource(
        frame,
        time_column=time_column,
        instrument_column=instrument_column,
    )
    factor = parse_factor(formula, name=factor_name, freq="1d", universe=None)
    engine = FactorEngine(build_backend(backend), source, run_mode=run_mode)
    plan, analysis = engine.compile(factor)
    runtime = engine.run(factor, plan=plan, analysis=analysis)
    series = _coerce_result_series(runtime.get("result"), name=factor_name).sort_index()
    return FactorExecution(
        factor_name=factor_name,
        result=series,
        snapshot_id=source.data_snapshot_id,
        plan=plan,
        analysis=analysis,
        runtime=runtime,
    )


def load_market_frame(
    *,
    market: str = "ashare",
    dataset: str | None = None,
    fields: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, Any]:
    """经 DataAccess 读取行情并标准化为 ``datetime/asset`` 长表。"""
    spec = get_market_data_spec(market, dataset=dataset)
    logical_fields = list(fields or spec.fields.keys())
    missing = sorted(set(logical_fields) - set(spec.fields))
    if missing:
        raise ValueError(f"市场 {spec.market} 未登记逻辑字段: {missing}")
    from data_access import get_store

    store = get_store()
    ds = store.get_dataset(spec.dataset)
    physical = [spec.fields[name] for name in logical_fields]
    columns = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
    read = store.read_result(
        spec.dataset,
        columns=columns,
        time_range=(start_date, end_date) if start_date or end_date else None,
        instrument_filter=list(instrument_filter) if instrument_filter else None,
        **dict(params or {}),
    )
    frame = read.table.to_pandas()
    rename = {
        ds.time_column: "datetime",
        ds.instrument_column: "asset",
        **{spec.fields[name]: name for name in logical_fields},
    }
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame.duplicated(["datetime", "asset"]).any():
        raise ValueError(f"DataAccess 数据集 {spec.dataset} 返回重复 (datetime, asset) 主键")
    frame = frame.sort_values(["asset", "datetime"]).reset_index(drop=True)
    return frame, read.snapshot


def _factor_series_to_staging_frame(
    series: pd.Series,
    *,
    factor_version: str,
    snapshot_id: str | None,
) -> pd.DataFrame:
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("写入因子湖要求 MultiIndex(datetime, asset) Series")
    frame = series.rename("value").reset_index()
    frame.columns = ["datetime", "asset", "value"]
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["asset"] = frame["asset"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype(float)
    frame["calc_time"] = datetime.now(timezone.utc).isoformat()
    frame["factor_version"] = str(factor_version)
    frame["data_snapshot_id"] = str(snapshot_id or "unknown")
    finite = np.isfinite(frame["value"].to_numpy(dtype=float, copy=False))
    frame["is_valid"] = finite.astype(np.int64)
    frame["invalid_reason"] = np.where(finite, "", "non_finite")
    frame["year"] = frame["datetime"].dt.year.astype(np.int32)
    return frame


def materialize_factor_to_staging(
    series: pd.Series,
    *,
    factor_id: str,
    factor_version: str = "1",
    snapshot_id: str | None = None,
    publish: bool = False,
) -> dict[str, Any]:
    """通过 DataAccess 幂等 upsert 到 ``factor_lake_staging``，可选原子发布。"""
    from data_access import get_store

    frame = _factor_series_to_staging_frame(
        series,
        factor_version=factor_version,
        snapshot_id=snapshot_id,
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    store = get_store()
    staging = store.upsert(
        "factor_lake_staging",
        table,
        factor_id=factor_id,
        upsert_on=["datetime", "asset"],
        partition_by=["year"],
    )
    output: dict[str, Any] = {
        "dataset": "factor_lake_staging",
        "factor_id": factor_id,
        "rows": int(len(frame)),
        "valid_rows": int(frame["is_valid"].sum()),
        "snapshot_id": snapshot_id,
        "staging": staging,
    }
    if publish:
        output["publish"] = store.publish_from_staging(
            "factor_lake_staging",
            "factor_lake",
            factor_id=factor_id,
        )
    return output
