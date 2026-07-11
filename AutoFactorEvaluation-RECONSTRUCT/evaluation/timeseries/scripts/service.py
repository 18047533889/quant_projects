"""时序服务入口编排。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：串联输入加载、目标收益计算、PIT 对齐、核心计算与产物落盘。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .align import pit_align
from .config import TimeseriesConfig, TimeseriesPaths, validate_timeseries_config
from .emit import write_json, write_manifest, write_parquet
from .forward_returns import compute_or_load_forward_returns
from .ic import compute_daily_ic_series
from .io import (
    build_universe_from_factor_assets,
    ensure_bool_universe,
    load_market_data,
    load_pure_factor_input,
    load_universe,
)
from .logging import build_run_logger
from .portfolio import attach_cost_and_path, compute_turnover_series
from .quantile import compute_quantile_panels
from .schemas import (
    COL_FACTOR_ID,
)
from .validation import Severity, ValidationBuffer

def _close_run_logger_handlers(logger) -> None:
    """仅关闭当前 run logger 的 handlers，避免影响全局 logging。"""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


@dataclass
class TimeseriesRunResult:
    """单次 4.1 时序评估运行结果。

    字段：
        out_dir: 本次 factor/eval_run 产物所在目录。
        manifest_path: `manifest.json` 路径（用于下游文件发现与运行追溯）。
        validation_report_path: `validation_report.json` 路径（用于诊断与审计）。
        artifacts: 逻辑产物名到物理路径的映射。
    """

    out_dir: Path
    manifest_path: Path
    validation_report_path: Path
    artifacts: dict[str, Path]


def _resolve_factor_id(pure_factor: pd.DataFrame, input_factor_id: str | None, vb: ValidationBuffer) -> str | None:
    """解析并校验本次运行唯一的 factor_id。

    入参：
        pure_factor: 从磁盘读取的 PureFactor 暴露表。
        input_factor_id: 调用方可选传入的 factor_id；若传入，必须与数据内唯一值一致。
        vb: 记录 mismatch/ambiguous 错误的验证缓冲区。

    出参：
        str | None：成功时返回唯一 factor_id；失败返回 `None`。
    """
    ids = sorted(set(str(x) for x in pure_factor[COL_FACTOR_ID].dropna().unique().tolist()))
    if input_factor_id is not None:
        if len(ids) != 1 or ids[0] != str(input_factor_id):
            vb.add(
                Severity.ERROR,
                "factor_id_mismatch",
                "factor_id parameter does not match pure factor data",
                details={"input_factor_id": input_factor_id, "data_factor_ids": ids},
            )
            return None
        return str(input_factor_id)
    if len(ids) != 1:
        vb.add(
            Severity.ERROR,
            "factor_id_ambiguous",
            "cannot infer unique factor_id from pure factor data",
            details={"data_factor_ids": ids},
        )
        return None
    return ids[0]


def run_timeseries_evaluation(
    *,
    eval_run_id: str,
    pure_factor_dir: Path,
    market_data_path: Path,
    universe_path: Path,
    factor_id: str | None = None,
    config: TimeseriesConfig | None = None,
    paths: TimeseriesPaths | None = None,
) -> TimeseriesRunResult:
    """执行一次完整的 4.1 时序评估（FID §1.1）。

    入参：
        eval_run_id: 本次评估运行唯一 ID。
        pure_factor_dir: PureFactor 三件套目录（candidate/data/manifest）。
        market_data_path: 市场行情表路径。
        universe_path: universe 表路径（外部已落盘输入，4.1 仅消费不生成）。
        factor_id: 可选预期 factor_id；传入时必须与 PureFactor 数据一致。
        config: 数值与行为配置；为空时使用 `TimeseriesConfig()`。
        paths: 输出/缓存/日志路径配置；为空时使用 `TimeseriesPaths()`。

    出参：
        TimeseriesRunResult：返回输出目录、manifest、validation report 与各产物路径。
        其中下游主消费对象是各 parquet；manifest 与 validation report
        分别承载可追溯元信息与诊断信息。

    异常：
        RuntimeError：当输入校验、factor_id 解析、目标收益计算或 PIT 对齐失败时抛出。
        若 universe 文件存在但不合法，会直接失败；仅在文件缺失时使用兜底 universe。

    工作流：
        加载 PureFactor -> 加载 market/universe -> 计算或命中 forward return 缓存
        -> PIT 对齐 -> 计算 IC/分层/组合序列 -> 写出 parquet/json 与 manifest。
    """
    cfg = config or TimeseriesConfig()
    if paths is None:
        raise ValueError("TimeseriesPaths 必须从外部传入，无代码级默认值")
    p = paths

    output_base = p.resolved_output_dir()
    cache_dir = p.resolved_cache_dir()
    log_dir = p.resolved_log_dir()

    logger = build_run_logger(log_dir, eval_run_id)
    vb = ValidationBuffer()
    vb.bind_logger(logger)
    logger.info("Timeseries run started")
    try:
        pure_factor = load_pure_factor_input(pure_factor_dir, vb)
        if pure_factor is None:
            raise RuntimeError("Pure factor input validation failed")
        logger.info("Pure factor loaded | rows=%s | dir=%s", len(pure_factor), pure_factor_dir)

        inferred_factor_id = _resolve_factor_id(pure_factor, factor_id, vb)
        if inferred_factor_id is None:
            raise RuntimeError("factor_id resolution failed")
        logger.info("factor_id resolved | factor_id=%s", inferred_factor_id)

        if not validate_timeseries_config(cfg, vb):
            raise RuntimeError("config validation failed")
        logger.info(
            "Config validated | horizons=%s | n_quantiles=%s | weighting=%s | rank_scope=%s",
            cfg.horizons,
            cfg.n_quantiles,
            cfg.weighting,
            cfg.rank_scope,
        )

        out_dir = output_base / inferred_factor_id / eval_run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        market_data = load_market_data(market_data_path, vb)
        if market_data is None:
            raise RuntimeError("market_data missing")
        logger.info("Market data loaded | rows=%s | path=%s", len(market_data), market_data_path)

        resolved_universe_path = universe_path.resolve()
        if resolved_universe_path.exists():
            u = load_universe(resolved_universe_path, vb)
            if u is None:
                raise RuntimeError("universe validation failed")
            universe = ensure_bool_universe(u, vb)
            if universe is None:
                raise RuntimeError("universe bool parsing failed")
            logger.info("Universe loaded | rows=%s | path=%s", len(universe), resolved_universe_path)
        else:
            vb.add(
                Severity.WARNING,
                "universe_fallback_from_factor_assets",
                "universe file missing; fallback to all assets from factor exposure",
                details={"universe_path": str(resolved_universe_path)},
            )
            universe = build_universe_from_factor_assets(pure_factor)
            logger.info("Universe fallback from factor assets | rows=%s", len(universe))

        if vb.has_error():
            raise RuntimeError("input validation failed")

        forward_return, price_field, cache_hit = compute_or_load_forward_returns(
            market_data,
            cfg.horizons,
            cache_dir,
            vb,
            eps=cfg.eps,
        )
        if forward_return is None or price_field is None:
            raise RuntimeError("forward return computation failed")
        logger.info("Forward return ready | price_field=%s | cache_hit=%s", price_field, cache_hit)

        aligned = pit_align(pure_factor, forward_return, universe, cfg.timezone_policy, vb)
        if vb.has_error():
            write_json(vb.to_report_dict(eval_run_id=eval_run_id, factor_id=inferred_factor_id), out_dir / "validation_report.json")
            raise RuntimeError("alignment validation failed")
        logger.info("Aligned panel rows=%s", len(aligned))

        daily_ic, daily_rank_ic = compute_daily_ic_series(aligned, cfg, inferred_factor_id, eval_run_id, vb)
        logger.info("IC series computed | ic_rows=%s | rank_ic_rows=%s", len(daily_ic), len(daily_rank_ic))
        quantile_df, tb_df, ls_gross_df, cov_df, weight_rows = compute_quantile_panels(
            aligned,
            universe,
            cfg,
            inferred_factor_id,
            eval_run_id,
            vb,
        )
        logger.info(
            "Quantile panels computed | quantile_rows=%s | top_minus_bottom_rows=%s | coverage_rows=%s",
            len(quantile_df),
            len(tb_df),
            len(cov_df),
        )
        turnover_df = compute_turnover_series(weight_rows, cfg.horizons, inferred_factor_id, eval_run_id)
        ls_df = attach_cost_and_path(ls_gross_df, turnover_df, cfg)
        logger.info("Turnover & long-short series ready | turnover_rows=%s | long_short_rows=%s", len(turnover_df), len(ls_df))

        artifacts = {
            "daily_ic": out_dir / "daily_ic.parquet",
            "daily_rank_ic": out_dir / "daily_rank_ic.parquet",
            "quantile_backtest": out_dir / "quantile_backtest.parquet",
            "top_minus_bottom": out_dir / "top_minus_bottom_series.parquet",
            "long_short_returns": out_dir / "long_short_returns.parquet",
            "turnover_series": out_dir / "turnover_series.parquet",
            "coverage_series": out_dir / "coverage_series.parquet",
            # 诊断与审计文件：不影响下游直接读取 parquet。
            "validation_report": out_dir / "validation_report.json",
        }

        write_parquet(daily_ic, artifacts["daily_ic"])
        write_parquet(daily_rank_ic, artifacts["daily_rank_ic"])
        write_parquet(quantile_df, artifacts["quantile_backtest"])
        write_parquet(tb_df, artifacts["top_minus_bottom"])
        write_parquet(ls_df, artifacts["long_short_returns"])
        write_parquet(turnover_df, artifacts["turnover_series"])
        write_parquet(cov_df, artifacts["coverage_series"])
        write_json(vb.to_report_dict(eval_run_id=eval_run_id, factor_id=inferred_factor_id), artifacts["validation_report"])
        logger.info("Artifacts written | count=%s | out_dir=%s", len(artifacts), out_dir)

        manifest_path = write_manifest(
            out_dir,
            factor_id=inferred_factor_id,
            eval_run_id=eval_run_id,
            environment=cfg.environment,
            price_field=price_field,
            cache_hit=cache_hit,
            timezone_policy=cfg.timezone_policy,
            config=asdict(cfg),
            artifacts=artifacts,
        )
        logger.info("Manifest written | manifest_path=%s", manifest_path)

        logger.info("Timeseries run finished")
        return TimeseriesRunResult(
            out_dir=out_dir,
            manifest_path=manifest_path,
            validation_report_path=artifacts["validation_report"],
            artifacts=artifacts,
        )
    finally:
        _close_run_logger_handlers(logger)

