"""
massive_cleaning_framework —— raw_massive_data 的通用清洗框架

语义：按原始文件树 **逐个文件** 读 → 清洗 → 写到 cleaned_massive_data。
每个输出文件 1:1 对应一个输入文件（不做跨文件聚合）。

为什么不走 data_access.store.read_arrow：
    data_access 的读接口是「按 dataset 级别 glob 聚合读」，返回一张大 Arrow
    Table。本框架的核心诉求是「每个文件独立清洗，保证输入输出 1:1」，跨文件
    聚合读完全不符合语义，且会占大量内存。因此本文件在 .data_access_allowlist
    里被登记为豁免，继续用 pd.read_parquet(single_file)。

下游消费者：
    清洗后的数据通过 data_access 的 `us_stocks_sip_day_aggs`（已登记）读取；
    raw 端如需直读，用 `us_stocks_sip_raw` 或 `massive_ticks`（PR4 起已登记）。
    任何 analytics 代码都不应再拼 massive_parquet 的路径，直接用 store。

维护人：quant 基础平台组
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


LOGGER = logging.getLogger("massive_cleaning_framework")


@dataclass
class SourceConfig:
    source: str
    path_pattern: str
    dataset_type: str
    frequency: str
    enabled: bool
    ticker_column: str | None
    align_time_column: str | None
    time_columns: list[str]
    time_format: str | None
    primary_key_columns: list[str]
    timezone: str = "unknown"
    array_columns_to_explode: list[str] = field(default_factory=list)
    dedup_strategy: str = "strict_primary_key"
    allow_null_ticker: bool = False
    allow_null_align_time: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], defaults: dict[str, Any]) -> "SourceConfig":
        merged = dict(defaults)
        merged.update(raw)
        return cls(
            source=merged["source"],
            path_pattern=merged["path_pattern"],
            dataset_type=merged["dataset_type"],
            frequency=merged["frequency"],
            enabled=bool(merged.get("enabled", True)),
            ticker_column=merged.get("ticker_column"),
            align_time_column=merged.get("align_time_column"),
            time_columns=list(merged.get("time_columns") or []),
            time_format=merged.get("time_format"),
            primary_key_columns=list(merged.get("primary_key_columns") or []),
            timezone=merged.get("timezone", "unknown"),
            array_columns_to_explode=list(merged.get("array_columns_to_explode") or []),
            dedup_strategy=merged.get("dedup_strategy", "strict_primary_key"),
            allow_null_ticker=bool(merged.get("allow_null_ticker", False)),
            allow_null_align_time=bool(merged.get("allow_null_align_time", False)),
            notes=merged.get("notes", ""),
        )


def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_config(config_path: Path) -> list[SourceConfig]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "sources" not in payload:
        raise ValueError(f"Invalid cleaning config: {config_path}")
    defaults = payload.get("defaults", {})
    return [SourceConfig.from_dict(item, defaults) for item in payload["sources"]]


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.where(normalized.notna() & (normalized != ""), pd.NA)


def parse_align_time(series: pd.Series, time_format: str | None) -> pd.Series:
    if time_format is None:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    if time_format in {"iso_date", "iso_datetime"}:
        return pd.to_datetime(series, errors="coerce", utc=True)

    if time_format.startswith("unix_"):
        unit = time_format.split("_", 1)[1]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.where(numeric != 0)
        return pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit)

    raise ValueError(f"Unsupported time_format: {time_format}")


def resolve_repo_root(script_path: Path) -> Path:
    for candidate in [script_path.resolve().parent, *script_path.resolve().parents]:
        if (candidate / "raw_data_layer").exists() and (candidate / "factor_layer").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Unable to resolve repository root from script location.")


def resolve_raw_files(raw_root: Path, source_config: SourceConfig) -> list[Path]:
    return sorted(raw_root.glob(source_config.path_pattern))


def relative_output_path(raw_file: Path, raw_root: Path, clean_root: Path) -> Path:
    return clean_root / raw_file.relative_to(raw_root)


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def explode_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df
    for column in columns:
        if column in result.columns:
            result = result.explode(column, ignore_index=True)
    return result


def build_primary_key(df: pd.DataFrame, source_config: SourceConfig) -> tuple[pd.Series, list[str], list[str]]:
    present_columns = ensure_columns_exist(df, source_config.primary_key_columns)
    missing_columns = [column for column in source_config.primary_key_columns if column not in present_columns]

    if not present_columns:
        fallback = pd.Series(range(len(df)), index=df.index, dtype="int64").astype("string")
        return fallback.map(lambda value: f"{source_config.source}:{value}"), present_columns, missing_columns

    key_frame = df[present_columns].copy()
    for column in present_columns:
        if pd.api.types.is_datetime64_any_dtype(key_frame[column]):
            key_frame[column] = key_frame[column].astype("string")
        else:
            key_frame[column] = key_frame[column].astype("string").fillna("<NA>")

    hashes = pd.util.hash_pandas_object(key_frame, index=False).astype("uint64").astype("string")
    primary_key = hashes.map(lambda value: f"{source_config.source}:{value}")
    return primary_key, present_columns, missing_columns


def deduplicate(df: pd.DataFrame, source_config: SourceConfig, present_pk_columns: list[str]) -> pd.DataFrame:
    if not present_pk_columns:
        return df

    keep = "first"
    if source_config.dedup_strategy == "keep_latest_by_align_time" and "align_time" in df.columns:
        df = df.sort_values("align_time", kind="stable")
        keep = "last"

    return df.drop_duplicates(subset=present_pk_columns, keep=keep, ignore_index=True)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    standard_columns = [
        "source",
        "dataset_type",
        "frequency",
        "ticker",
        "align_time",
        "primary_key",
        "primary_key_columns_used",
        "align_time_source_column",
        "ticker_source_column",
        "timezone",
        "notes",
    ]
    front = [column for column in standard_columns if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df[front + rest]


def clean_dataframe(df: pd.DataFrame, source_config: SourceConfig, raw_file: Path = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """清洗DataFrame数据（不读取文件）"""
    original_rows = len(df)
    return _perform_cleaning(df, source_config, original_rows, raw_file)


def clean_one_file(raw_file: Path, source_config: SourceConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(raw_file)
    original_rows = len(df)
    return _perform_cleaning(df, source_config, original_rows, raw_file)


def _perform_cleaning(df: pd.DataFrame, source_config: SourceConfig, original_rows: int, raw_file: Path) -> tuple[pd.DataFrame, dict[str, Any]]:

    df = explode_columns(df, source_config.array_columns_to_explode)

    if source_config.ticker_column and source_config.ticker_column in df.columns:
        if source_config.ticker_column == "ticker":
            df["ticker"] = normalize_ticker_series(df["ticker"])
        else:
            df["ticker"] = normalize_ticker_series(df[source_config.ticker_column])
    else:
        df["ticker"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if source_config.align_time_column and source_config.align_time_column in df.columns:
        df["align_time"] = parse_align_time(df[source_config.align_time_column], source_config.time_format)
    else:
        df["align_time"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    primary_key, present_pk_columns, missing_pk_columns = build_primary_key(df, source_config)
    df["primary_key"] = primary_key

    df["source"] = source_config.source
    df["dataset_type"] = source_config.dataset_type
    df["frequency"] = source_config.frequency
    df["primary_key_columns_used"] = ", ".join(present_pk_columns)
    df["align_time_source_column"] = source_config.align_time_column
    df["ticker_source_column"] = source_config.ticker_column
    df["timezone"] = source_config.timezone
    df["notes"] = source_config.notes

    if not source_config.allow_null_ticker:
        before = len(df)
        df = df[df["ticker"].notna()].copy()
        dropped = before - len(df)
    else:
        dropped = 0

    if not source_config.allow_null_align_time:
        before = len(df)
        df = df[df["align_time"].notna()].copy()
        dropped_align_time = before - len(df)
    else:
        dropped_align_time = 0

    df = deduplicate(df, source_config, present_pk_columns)
    df = reorder_columns(df)

    summary = {
        "source": source_config.source,
        "raw_file": str(raw_file),
        "rows_in": original_rows,
        "rows_out": len(df),
        "dropped_null_ticker": dropped,
        "dropped_null_align_time": dropped_align_time,
        "missing_primary_key_columns": missing_pk_columns,
    }
    return df, summary


def process_source(
    raw_root: Path,
    clean_root: Path,
    source_config: SourceConfig,
    limit_files: int | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    raw_files = resolve_raw_files(raw_root, source_config)
    if limit_files is not None:
        raw_files = raw_files[:limit_files]

    if not raw_files:
        LOGGER.warning("No raw files matched for %s", source_config.source)
        return []

    summaries: list[dict[str, Any]] = []
    for raw_file in raw_files:
        output_file = relative_output_path(raw_file, raw_root, clean_root)
        if output_file.exists() and not overwrite:
            LOGGER.info("Skip existing %s", output_file)
            continue

        cleaned_df, summary = clean_one_file(raw_file, source_config)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_parquet(output_file, index=False)
        summaries.append(summary)

        if summary["missing_primary_key_columns"]:
            LOGGER.warning(
                "Source %s missing configured primary key columns %s in file %s",
                source_config.source,
                summary["missing_primary_key_columns"],
                raw_file,
            )

    return summaries


def write_run_summary(clean_root: Path, summaries: list[dict[str, Any]]) -> Path:
    summary_path = clean_root / "_cleaning_run_summary.json"
    payload = {
        "file_count": len(summaries),
        "sources": sorted({item["source"] for item in summaries}),
        "files": summaries,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic cleaner for Massive raw parquet datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "data_source_cleaning_config.yaml",
        help="Path to the cleaning YAML config.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Massive data root. Defaults to ../massive_parquet relative to the repository root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override raw data root. Defaults to <data-root>/raw_massive_data.",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=None,
        help="Override cleaned data root. Defaults to <data-root>/cleaned_massive_data.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Process only selected source entries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process only the first N files matched for each selected source.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cleaned parquet files if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    repo_root = resolve_repo_root(Path(__file__))
    data_root = (args.data_root.resolve() if args.data_root else (repo_root.parent / "massive_parquet").resolve())
    raw_root = args.raw_root.resolve() if args.raw_root else data_root / "raw_massive_data"
    clean_root = args.clean_root.resolve() if args.clean_root else data_root / "cleaned_massive_data"

    if not args.config.exists():
        raise SystemExit(f"Missing config file: {args.config}")
    if not raw_root.exists():
        raise SystemExit(f"Missing raw root directory: {raw_root}")

    clean_root.mkdir(parents=True, exist_ok=True)

    source_configs = load_config(args.config)
    selected_sources = set(args.sources or [])
    if selected_sources:
        source_configs = [config for config in source_configs if config.source in selected_sources]
    else:
        source_configs = [config for config in source_configs if config.enabled]

    all_summaries: list[dict[str, Any]] = []
    for source_config in source_configs:
        LOGGER.info("Processing source %s", source_config.source)
        all_summaries.extend(
            process_source(
                raw_root=raw_root,
                clean_root=clean_root,
                source_config=source_config,
                limit_files=args.limit_files,
                overwrite=args.overwrite,
            )
        )

    summary_path = write_run_summary(clean_root, all_summaries)
    print(
        json.dumps(
            {
                "processed_sources": sorted({item["source"] for item in all_summaries}),
                "processed_files": len(all_summaries),
                "summary_path": str(summary_path),
                "raw_root": str(raw_root),
                "clean_root": str(clean_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()