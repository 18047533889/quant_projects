"""Assetization 计算入口。

所有正式行情读取走 ``data_access``，所有 DSL 执行走仓库根最新 ``factor_engine``。
目录内旧 FactorEngine 不再参与运行。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from integrations.quant_platform import (
    execute_factor_formula,
    execute_factor_on_frame,
    materialize_factor_to_staging,
)


class FormulaEvaluator:
    """兼容旧 API 的内存 DataFrame 求值器，底层仍是最新 FactorEngine。"""

    def __init__(self, *, backend: str = "pandas", run_mode: str = "research") -> None:
        self.backend = backend
        self.run_mode = run_mode

    def evaluate(self, formula: str, market_df: pd.DataFrame):
        execution = execute_factor_on_frame(
            formula,
            market_df,
            factor_name="autofactor_frame_eval",
            backend=self.backend,
            run_mode=self.run_mode,
        )
        return execution.result.to_numpy(dtype=float, copy=False)


def run_assetization(
    manifest: Mapping[str, Any],
    *,
    market_data_root: str | None = None,
    cache_root: str | None = None,
    dataset: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    backend: str = "pandas",
    run_mode: str = "research",
    materialize_staging: bool = True,
    publish: bool = False,
) -> dict[str, Any]:
    """运行单因子资产化，并可幂等写入 ``factor_lake_staging``。

    ``market_data_root`` / ``cache_root`` 仅为旧调用兼容参数；正式路径由
    ``data_access/config/datasets.yaml`` 与 DataSnapshot 决定。
    """
    del market_data_root, cache_root
    manifest = dict(manifest)
    run_id = f"as_{datetime.now():%Y%m%d_%H%M%S_%f}"
    checked_at = datetime.now(timezone.utc).isoformat()
    formula, expression_type = _extract_formula(manifest)
    candidate_id = str(manifest.get("candidate_id") or "unknown")
    market = str(manifest.get("market") or "ashare")
    data_cfg = manifest.get("data_source") if isinstance(manifest.get("data_source"), Mapping) else {}
    resolved_dataset = dataset or data_cfg.get("dataset") or manifest.get("dataset")
    resolved_start = start_date or _manifest_date(manifest, "start_date", "start")
    resolved_end = end_date or _manifest_date(manifest, "end_date", "end")
    resolved_instruments = list(instrument_filter) if instrument_filter else _manifest_instruments(manifest)
    fields = data_cfg.get("fields") if isinstance(data_cfg.get("fields"), Mapping) else None
    params = data_cfg.get("params") if isinstance(data_cfg.get("params"), Mapping) else None

    execution = execute_factor_formula(
        formula,
        factor_name=candidate_id,
        market=market,
        dataset=resolved_dataset,
        fields=fields,
        start_date=resolved_start,
        end_date=resolved_end,
        instrument_filter=resolved_instruments,
        params=params,
        backend=backend,
        run_mode=run_mode,
        freq=str(manifest.get("frequency_bucket", "1d")),
        universe=manifest.get("universe_id"),
        description=str(manifest.get("description", "")),
        expression_type=expression_type,
    )

    from assetization.scripts.registry import extract_coordinates, generate_factor_id

    coordinates = extract_coordinates(manifest)
    factor_id = generate_factor_id(
        coordinates,
        seed={"formula": formula, "candidate_id": candidate_id, "coordinates": coordinates},
    )
    staging_summary = None
    if materialize_staging:
        staging_summary = materialize_factor_to_staging(
            execution.result,
            factor_id=factor_id,
            factor_version=str(manifest.get("factor_version", "1")),
            snapshot_id=execution.snapshot_id,
            publish=publish,
        )

    daily = _split_by_day(execution.result)
    return {
        "Status": "PASS",
        "RunId": run_id,
        "CheckedAt": checked_at,
        "Steps": {
            "ExtractFormula": {
                "status": "PASS",
                "expression_type": expression_type,
                "formula_preview": formula[:120],
            },
            "RunEngine": {
                "status": "PASS",
                "total_rows": len(execution.result),
                "non_null": int(execution.result.notna().sum()),
                "backend": backend,
                "run_mode": run_mode,
            },
            "DataAccess": {
                "status": "PASS",
                "dataset": resolved_dataset,
                "snapshot_id": execution.snapshot_id,
                "start_date": resolved_start,
                "end_date": resolved_end,
            },
            "MaterializeStaging": {
                "status": "PASS" if staging_summary is not None else "SKIPPED",
                "summary": staging_summary,
            },
        },
        "factor_id": factor_id,
        "candidate_id": candidate_id,
        "expression_type": expression_type,
        "data_snapshot_id": execution.snapshot_id,
        "daily_values": daily,
        "series": execution.result,
        "staging": staging_summary,
    }


def _extract_formula(manifest: Mapping[str, Any]) -> tuple[str, str]:
    expression_type = str(manifest.get("expression_type", "dsl")).lower()
    if expression_type in {"python", "code"}:
        raise ValueError(
            "Assetization production path 禁止执行任意 Python code；"
            "请将候选转换为 FactorEngine DSL"
        )
    formula = str(manifest.get("formula") or manifest.get("expr") or "").strip()
    if not formula:
        raise ValueError("manifest 缺少 formula/expr")
    return formula, "dsl"


def _manifest_date(manifest: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = manifest.get(key)
        if value not in (None, ""):
            return str(value)
    data_cfg = manifest.get("data_source")
    if isinstance(data_cfg, Mapping):
        for key in keys:
            value = data_cfg.get(key)
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
    return [str(item) for item in value]


def _split_by_day(series: pd.Series) -> dict[str, pd.Series]:
    if not isinstance(series.index, pd.MultiIndex):
        return {"full": series}
    timestamps = pd.to_datetime(series.index.get_level_values(0), errors="raise")
    normalized = timestamps.normalize()
    daily: dict[str, pd.Series] = {}
    for day in sorted(normalized.unique()):
        mask = normalized == day
        daily[pd.Timestamp(day).strftime("%Y-%m-%d")] = series.loc[mask]
    return daily
