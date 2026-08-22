"""Efficient readers for the local Barra-lite B/f package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    normalized.index = index.normalize()
    normalized.columns = pd.Index(map(str, normalized.columns), name="asset")
    normalized.index.name = "date"
    return normalized.sort_index()


def weights_to_long(weights: pd.DataFrame, epsilon: float = 1e-12) -> pd.DataFrame:
    """Convert a wide weight matrix to sparse ``date, asset, weight`` rows."""
    normalized = _normalize_frame(weights).fillna(0.0)
    sparse = normalized.where(normalized.abs() > epsilon)
    long = sparse.stack().dropna().rename("weight").reset_index()
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long["asset"] = long["asset"].astype(str)
    long["weight"] = long["weight"].astype(float)
    return long


@dataclass(slots=True)
class ExposureAggregation:
    exposure: pd.DataFrame
    invested_exposure: pd.DataFrame
    coverage: pd.DataFrame
    missing: pd.DataFrame


class RiskModelStore:
    """Read and aggregate one precomputed Barra-lite package.

    Exposure aggregation stays in long form inside DuckDB.  This avoids
    materializing a date x asset x factor cube for the full-A packages.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"风险模型缺少 manifest: {manifest_path}")
        self.manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if self.manifest.get("product") != "barra_lite":
            raise ValueError("风险模型 manifest.product 必须为 barra_lite")
        specs = list(self.manifest.get("factors") or [])
        self.factors = [str(item["id"]) for item in specs]
        self.style_factors = [
            str(item["id"]) for item in specs if item.get("type") == "continuous"
        ]
        self.industry_factors = [
            str(item["id"]) for item in specs if item.get("type") == "dummy"
        ]
        if not self.factors or len(set(self.factors)) != len(self.factors):
            raise ValueError("风险模型因子列表必须非空且唯一")
        if not self.style_factors or not self.industry_factors:
            raise ValueError("风险模型必须同时包含连续风格和行业 dummy")

        files = self.manifest.get("files") or {}
        self.exposure_path = self.root / str(files.get("exposure", ""))
        factor_returns_name = files.get("factor_returns")
        if not factor_returns_name:
            raise ValueError("manifest.files 缺少 factor_returns；该风险包不能做 B/f 分析")
        self.factor_returns_path = self.root / str(factor_returns_name)
        for path in [self.exposure_path, self.factor_returns_path]:
            if not path.is_file():
                raise FileNotFoundError(f"风险模型文件不存在: {path}")
        self._validate_schema(
            self.exposure_path,
            {"date", "asset", "factor_id", "exposure"},
        )
        self._validate_schema(
            self.factor_returns_path,
            {"date", "factor_id", "factor_return"},
        )
        quality = self.manifest.get("quality") or {}
        date_min = quality.get("date_min")
        date_max = quality.get("date_max")
        if date_min is None or date_max is None:
            connection = duckdb.connect()
            try:
                path = self._sql_path(self.exposure_path)
                date_min, date_max = connection.execute(
                    f"SELECT MIN(date), MAX(date) FROM read_parquet('{path}')"
                ).fetchone()
            finally:
                connection.close()
        self.date_min = pd.Timestamp(date_min).normalize()
        self.date_max = pd.Timestamp(date_max).normalize()

    @staticmethod
    def _validate_schema(path: Path, required: set[str]) -> None:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        missing = required - columns
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(sorted(missing))}")

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", self.root.name))

    @staticmethod
    def _sql_path(path: Path) -> str:
        return path.as_posix().replace("'", "''")

    def aggregate_exposure(
        self,
        weights: pd.DataFrame,
        *,
        missing_policy: str = "report_unknown",
        epsilon: float = 1e-12,
    ) -> ExposureAggregation:
        """Calculate ``w'B`` and explicit model-coverage diagnostics."""
        policy = str(missing_policy).lower()
        if policy not in {"error", "report_unknown", "renormalize"}:
            raise ValueError("missing_policy 仅支持 error、report_unknown、renormalize")
        normalized = _normalize_frame(weights).fillna(0.0)
        dates = normalized.index
        long = weights_to_long(normalized, epsilon=epsilon)
        factor_columns = pd.Index(self.factors, name="factor_id")
        if long.empty:
            empty = pd.DataFrame(0.0, index=dates, columns=factor_columns)
            coverage = pd.DataFrame(
                {
                    "invested_weight": 0.0,
                    "total_abs_weight": 0.0,
                    "matched_weight": 0.0,
                    "matched_abs_weight": 0.0,
                    "unknown_weight": 0.0,
                    "coverage_weight": 1.0,
                    "missing_asset_count": 0,
                },
                index=dates,
            )
            return ExposureAggregation(empty, empty.copy(), coverage, long)

        connection = duckdb.connect()
        try:
            connection.register("portfolio_weights", long)
            exposure_path = self._sql_path(self.exposure_path)
            connection.execute(
                f"""
                CREATE TEMP VIEW risk_exposure AS
                SELECT date, asset, factor_id, exposure
                FROM read_parquet('{exposure_path}')
                """
            )
            aggregated = connection.execute(
                """
                SELECT
                    w.date,
                    e.factor_id,
                    SUM(w.weight * e.exposure) AS exposure
                FROM portfolio_weights w
                JOIN risk_exposure e
                  ON w.date = e.date AND w.asset = e.asset
                GROUP BY 1, 2
                ORDER BY 1, 2
                """
            ).df()
            coverage = connection.execute(
                """
                WITH pairs AS (
                    SELECT DISTINCT date, asset, 1 AS matched
                    FROM risk_exposure
                )
                SELECT
                    w.date,
                    SUM(w.weight) AS invested_weight,
                    SUM(ABS(w.weight)) AS total_abs_weight,
                    SUM(CASE WHEN p.matched = 1 THEN w.weight ELSE 0 END) AS matched_weight,
                    SUM(CASE WHEN p.matched = 1 THEN ABS(w.weight) ELSE 0 END) AS matched_abs_weight,
                    SUM(CASE WHEN p.matched IS NULL THEN w.weight ELSE 0 END) AS unknown_weight,
                    SUM(CASE WHEN p.matched IS NULL THEN 1 ELSE 0 END) AS missing_asset_count
                FROM portfolio_weights w
                LEFT JOIN pairs p
                  ON w.date = p.date AND w.asset = p.asset
                GROUP BY 1
                ORDER BY 1
                """
            ).df()
            missing = connection.execute(
                """
                WITH pairs AS (
                    SELECT DISTINCT date, asset
                    FROM risk_exposure
                )
                SELECT w.*
                FROM portfolio_weights w
                ANTI JOIN pairs p
                  ON w.date = p.date AND w.asset = p.asset
                ORDER BY w.date, w.asset
                """
            ).df()
        finally:
            connection.close()

        if not missing.empty and policy == "error":
            sample = missing.iloc[0]
            raise ValueError(
                f"{sample['date']} {sample['asset']} 缺少风险暴露；"
                f"共 {len(missing)} 条非零权重无法匹配"
            )

        if aggregated.empty:
            exposure = pd.DataFrame(0.0, index=dates, columns=factor_columns)
        else:
            aggregated["date"] = pd.to_datetime(aggregated["date"])
            exposure = (
                aggregated.pivot(index="date", columns="factor_id", values="exposure")
                .reindex(index=dates, columns=factor_columns)
                .fillna(0.0)
            )

        coverage["date"] = pd.to_datetime(coverage["date"])
        coverage = coverage.set_index("date").reindex(dates)
        totals = normalized.sum(axis=1)
        total_abs = normalized.abs().sum(axis=1)
        coverage["invested_weight"] = coverage["invested_weight"].fillna(totals)
        coverage["total_abs_weight"] = coverage["total_abs_weight"].fillna(total_abs)
        for column in [
            "matched_weight",
            "matched_abs_weight",
            "unknown_weight",
            "missing_asset_count",
        ]:
            coverage[column] = coverage[column].fillna(0.0)
        coverage["coverage_weight"] = np.where(
            coverage["total_abs_weight"].abs() > epsilon,
            coverage["matched_abs_weight"] / coverage["total_abs_weight"],
            1.0,
        )
        coverage["missing_asset_count"] = coverage["missing_asset_count"].astype(int)

        invested_denominator = coverage["invested_weight"].replace(0.0, np.nan)
        invested_exposure = exposure.div(invested_denominator, axis=0).fillna(0.0)
        if policy == "renormalize":
            denominator = coverage["matched_weight"].replace(0.0, np.nan)
            exposure = exposure.div(denominator, axis=0).fillna(0.0)
        missing["date"] = pd.to_datetime(missing["date"]) if not missing.empty else missing.get("date")
        return ExposureAggregation(exposure, invested_exposure, coverage, missing)

    def load_factor_returns(self, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        """Load f for exactly the requested dates and manifest factor order."""
        requested = pd.DatetimeIndex(dates).normalize()
        if requested.empty:
            return pd.DataFrame(index=requested, columns=self.factors, dtype=float)
        connection = duckdb.connect()
        date_frame = pd.DataFrame({"date": requested.strftime("%Y-%m-%d")})
        try:
            connection.register("requested_dates", date_frame)
            path = self._sql_path(self.factor_returns_path)
            frame = connection.execute(
                f"""
                SELECT f.date, f.factor_id, f.factor_return
                FROM read_parquet('{path}') f
                JOIN requested_dates d USING(date)
                ORDER BY f.date, f.factor_id
                """
            ).df()
        finally:
            connection.close()
        frame["date"] = pd.to_datetime(frame["date"])
        result = (
            frame.pivot(index="date", columns="factor_id", values="factor_return")
            .reindex(index=requested, columns=self.factors)
        )
        return result
