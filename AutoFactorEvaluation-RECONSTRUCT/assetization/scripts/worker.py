"""Assetization worker：Gateway package → FactorEngine/DataAccess → staging package。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .compute import run_assetization

logger = logging.getLogger("assetization.worker")


def process_factor_dir(
    factor_dir: str | Path,
    *,
    market_data_path: str | Path | None = None,
    market_dataset: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    instrument_filter: Sequence[str] | None = None,
    output_base: str | Path | None = None,
    temp_base: str | Path | None = None,
    backend: str = "pandas",
    run_mode: str = "research",
    write_factor_staging: bool = True,
    publish: bool = False,
) -> tuple[str, str]:
    """处理一个 Gateway Pass 因子。

    ``market_data_path`` 仅保留旧调用兼容；行情由 ``market_dataset`` 或 manifest
    中的 DataAccess dataset 决定。``output_base`` 由后续 router 使用，本 worker
    只写 ``temp_base`` 与 DataAccess factor staging。
    """
    del market_data_path, output_base
    factor_dir = Path(factor_dir)
    afv_path = factor_dir / "afv.json"
    if not afv_path.is_file():
        raise FileNotFoundError(f"因子目录缺少 afv.json: {factor_dir}")
    if temp_base is None:
        raise ValueError("assetization: temp_base 必须从外部配置传入")

    afv = json.loads(afv_path.read_text(encoding="utf-8"))
    gateway = afv.get("Gateway", {})
    gateway_status = str(gateway.get("Label") or gateway.get("Status") or "").upper()
    if gateway_status not in {"PASS", "PASSED"}:
        raise ValueError(
            f"因子未通过 Gateway: candidate={afv.get('candidate_id', factor_dir.name)}, "
            f"status={gateway_status or 'missing'}"
        )
    manifest_path = factor_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    merged: dict[str, Any] = {**manifest, **afv}
    merged.setdefault("candidate_id", factor_dir.name)
    if market_dataset:
        merged["dataset"] = market_dataset

    result = run_assetization(
        merged,
        dataset=market_dataset,
        start_date=start_date,
        end_date=end_date,
        instrument_filter=instrument_filter,
        backend=backend,
        run_mode=run_mode,
        materialize_staging=write_factor_staging,
        publish=publish,
    )
    factor_id = str(result["factor_id"])
    temp_dir = Path(temp_base) / factor_id
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    row_count = 0
    for date_str, daily_series in result["daily_values"].items():
        frame = _daily_series_frame(daily_series, factor_id=factor_id)
        row_count += len(frame)
        frame.to_parquet(data_dir / f"{date_str}.parquet", index=False)

    enriched_afv = dict(afv)
    enriched_afv["factor_id"] = factor_id
    enriched_afv["Assetization"] = {
        "Label": "Materialized",
        "status": "passed",
        "factor_id": factor_id,
        "run_id": result["RunId"],
        "materialized_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "row_count": row_count,
        "data_snapshot_id": result.get("data_snapshot_id"),
        "factor_lake_staging": result.get("staging"),
        "formula": merged.get("formula"),
        "candidate_id": merged.get("candidate_id"),
        "backend": backend,
        "run_mode": run_mode,
    }
    (temp_dir / "afv.json").write_text(
        json.dumps(enriched_afv, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (temp_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "factor_id": factor_id,
                "data_snapshot_id": result.get("data_snapshot_id"),
                "factor_lake_staging": result.get("staging"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Assetization 完成 factor_id=%s rows=%d snapshot=%s temp=%s",
        factor_id,
        row_count,
        result.get("data_snapshot_id"),
        temp_dir,
    )
    return factor_id, str(temp_dir)


def _daily_series_frame(series: pd.Series, *, factor_id: str) -> pd.DataFrame:
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("daily_values 必须为 MultiIndex(datetime, asset) Series")
    frame = series.rename("factor_value").reset_index()
    frame.columns = ["TradeDate", "Symbol", "factor_value"]
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"], errors="raise")
    frame["Symbol"] = frame["Symbol"].astype(str)
    frame["factor_id"] = factor_id
    return frame[["TradeDate", "Symbol", "factor_id", "factor_value"]]
