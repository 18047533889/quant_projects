"""真实 Parquet 数据集上的因子冒烟测试（集成验证用）。

对 ``DATASET_SPECS`` 中各数据集构造简易因子并执行 ``FactorEngine.run``，
用于 CI / 本地快速验证 DSL 与数据接入是否正常。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean, zscore
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource


@dataclass(frozen=True)
class DatasetSpec:
    """单个 Parquet 数据集的列映射与候选因子列。"""

    name: str
    relative_path: str
    timestamp_col: str
    instrument_col: str
    candidate_columns: tuple[str, ...]
    timestamp_unit: str | None = None


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="fundamentals/balance_sheet",
        relative_path="fundamentals/balance_sheet",
        timestamp_col="period_end",
        instrument_col="tickers",
        candidate_columns=("total_assets", "total_liabilities", "total_equity"),
    ),
    DatasetSpec(
        name="fundamentals/cash_flow_statement",
        relative_path="fundamentals/cash_flow_statement",
        timestamp_col="period_end",
        instrument_col="tickers",
        candidate_columns=(
            "net_cash_from_operating_activities",
            "net_cash_from_investing_activities",
            "net_income",
        ),
    ),
    DatasetSpec(
        name="fundamentals/financials_ratios",
        relative_path="fundamentals/financials_ratios",
        timestamp_col="date",
        instrument_col="ticker",
        candidate_columns=("price_to_earnings", "return_on_equity", "debt_to_equity"),
    ),
    DatasetSpec(
        name="fundamentals/income_statement",
        relative_path="fundamentals/income_statement",
        timestamp_col="period_end",
        instrument_col="tickers",
        candidate_columns=("revenue", "operating_income", "net_income_loss_attributable_common_shareholders"),
    ),
    DatasetSpec(
        name="fundamentals/short_interest",
        relative_path="fundamentals/short_interest",
        timestamp_col="settlement_date",
        instrument_col="ticker",
        candidate_columns=("short_interest", "days_to_cover", "avg_daily_volume"),
    ),
    DatasetSpec(
        name="fundamentals/short_volume",
        relative_path="fundamentals/short_volume",
        timestamp_col="date",
        instrument_col="ticker",
        candidate_columns=("short_volume", "total_volume", "short_volume_ratio"),
    ),
    DatasetSpec(
        name="fundamentals/stocks_floats",
        relative_path="fundamentals/stocks_floats",
        timestamp_col="effective_date",
        instrument_col="ticker",
        candidate_columns=("free_float", "free_float_percent", "outstanding_shares"),
    ),
    DatasetSpec(
        name="us_stocks_sip/day_aggs_v1",
        relative_path="us_stocks_sip/day_aggs_v1",
        timestamp_col="window_start",
        instrument_col="ticker",
        candidate_columns=("close", "volume", "transactions"),
        timestamp_unit="ns",
    ),
    DatasetSpec(
        name="us_stocks_sip/minute_aggs_v1",
        relative_path="us_stocks_sip/minute_aggs_v1",
        timestamp_col="window_start",
        instrument_col="ticker",
        candidate_columns=("close", "volume", "transactions"),
        timestamp_unit="ns",
    ),
    DatasetSpec(
        name="us_stocks_sip/quotes_v1",
        relative_path="us_stocks_sip/quotes_v1",
        timestamp_col="sip_timestamp",
        instrument_col="ticker",
        candidate_columns=("bid_price", "ask_price", "bid_size"),
        timestamp_unit="ns",
    ),
    DatasetSpec(
        name="us_stocks_sip/trades_v1",
        relative_path="us_stocks_sip/trades_v1",
        timestamp_col="sip_timestamp",
        instrument_col="ticker",
        candidate_columns=("price", "size", "exchange"),
        timestamp_unit="ns",
    ),
)


class MultiParquetSeriesSource(DataSource):
    """从目录下多个 Parquet 文件读取列并规范为 MultiIndex Series。"""

    def __init__(
        self,
        root: str | Path,
        timestamp_col: str,
        instrument_col: str,
        max_files: int = 2,
        timestamp_unit: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """配置根目录、时间/标的列名及可选日期过滤。"""
        self.root = Path(root)
        self.timestamp_col = timestamp_col
        self.instrument_col = instrument_col
        self.max_files = max_files
        self.timestamp_unit = timestamp_unit
        self.start_date = start_date
        self.end_date = end_date
        self._column_cache: dict[str, Any] = {}

    def _selected_files(self) -> list[Path]:
        return sorted(self.root.rglob("*.parquet"))

    @staticmethod
    def _normalize_instrument(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if not value:
                return "UNKNOWN"
            return str(value[0])
        return str(value)

    def load_column(self, name: str):
        """通过 DuckDB 读取列并规范为 ``(timestamp, instrument)`` MultiIndex Series。"""
        import pandas as pd
        from data_access.core.engine import get_shared_engine

        if name in self._column_cache:
            return self._column_cache[name]

        all_files = self._selected_files()
        if not all_files:
            raise FileNotFoundError(f"No parquet files found under {self.root}")

        cols = [self.timestamp_col, self.instrument_col, name]
        col_clause = ", ".join(f'"{c}"' for c in cols)
        sql = f"SELECT {col_clause} FROM read_parquet(?, union_by_name=true)"
        engine = get_shared_engine()

        # 优先走批量：一次读前 max_files 个文件（语义兼容旧版「读到 max_files 个成功 frame 就停」
        # 的最常见路径 —— 多数时候前 max_files 个文件都是好的）。
        batch_files = all_files[: self.max_files] if self.max_files else all_files
        try:
            table = engine.execute_arrow(sql, [[str(p) for p in batch_files]])
            df = table.to_pandas(self_destruct=True, split_blocks=True)
        except Exception:  # pragma: no cover - depends on raw data quality
            # 批量失败：逐文件读，累计 max_files 个成功就停，保留老语义
            frames: list[pd.DataFrame] = []
            read_errors: list[str] = []
            for path in all_files:
                try:
                    table = engine.execute_arrow(sql, [str(path)])
                    frames.append(
                        table.to_pandas(self_destruct=True, split_blocks=True)
                    )
                    if self.max_files and len(frames) >= self.max_files:
                        break
                except Exception as exc:
                    read_errors.append(f"{path}: {exc}")

            if not frames:
                preview = "\n".join(read_errors[:5])
                raise RuntimeError(
                    f"No readable parquet file for column '{name}' under {self.root}. Errors:\n{preview}"
                )
            df = pd.concat(frames, ignore_index=True)

        if self.timestamp_unit:
            ts = pd.to_datetime(df[self.timestamp_col], unit=self.timestamp_unit, utc=True, errors="coerce")
            ts = ts.dt.tz_convert(None)
        else:
            ts = pd.to_datetime(df[self.timestamp_col], utc=True, errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)

        df["timestamp"] = ts.dt.normalize()
        df["instrument"] = df[self.instrument_col].map(self._normalize_instrument)

        series = df.set_index(["timestamp", "instrument"])[name].sort_index()
        if self.start_date:
            series = series[series.index.get_level_values("timestamp") >= self.start_date]
        if self.end_date:
            series = series[series.index.get_level_values("timestamp") <= self.end_date]
        self._column_cache[name] = series
        return series


def _build_factors(columns: list[str], prefix: str) -> list[Factor]:
    main = columns[0]
    factors = [
        Factor(name=f"{prefix}_rank_{main}", expr=rank(col(main))),
        Factor(name=f"{prefix}_zscore_{main}", expr=zscore(col(main))),
        Factor(name=f"{prefix}_tsmean3_rank_{main}", expr=rank(ts_mean(col(main), 3))),
    ]

    if len(columns) > 1:
        aux = columns[1]
        factors.append(
            Factor(
                name=f"{prefix}_spread_{main}_vs_{aux}",
                expr=zscore(col(main) / (col(aux) + 1e-9)),
            )
        )

    return factors


def run_dataset_factor_smoke(
    massive_parquet_root: str | Path,
    spec: DatasetSpec,
    max_files: int = 2,
) -> dict[str, Any]:
    """对单个数据集运行 rank/zscore/ts_mean 等冒烟因子并汇总结果。"""
    import pandas as pd

    dataset_root = Path(massive_parquet_root) / spec.relative_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    source = MultiParquetSeriesSource(
        root=dataset_root,
        timestamp_col=spec.timestamp_col,
        instrument_col=spec.instrument_col,
        max_files=max_files,
        timestamp_unit=spec.timestamp_unit,
    )

    factors = _build_factors(list(spec.candidate_columns), prefix=spec.name.replace("/", "_"))
    engine = FactorEngine(backend=PandasBackend(), data_source=source)

    results: list[dict[str, Any]] = []
    for factor in factors:
        try:
            out = engine.run(factor)
            series = out["result"].sort_index()
            assert isinstance(series, pd.Series)
            results.append(
                {
                    "factor": factor.name,
                    "ok": True,
                    "rows": int(series.shape[0]),
                    "non_nan_ratio": float(series.notna().mean()) if len(series) else 0.0,
                    "lookback": int(out["analysis"].lookback),
                }
            )
        except Exception as exc:  # pragma: no cover - integration path
            results.append(
                {
                    "factor": factor.name,
                    "ok": False,
                    "error": str(exc),
                }
            )

    return {
        "dataset": spec.name,
        "dataset_root": str(dataset_root),
        "factor_count": len(factors),
        "results": results,
    }


def run_all_dataset_smokes(massive_parquet_root: str | Path, max_files: int = 2) -> list[dict[str, Any]]:
    """对 ``DATASET_SPECS`` 中全部数据集依次执行冒烟测试。"""
    return [run_dataset_factor_smoke(massive_parquet_root, spec, max_files=max_files) for spec in DATASET_SPECS]
