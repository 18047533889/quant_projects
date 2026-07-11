"""Evaluation 总模块 pipeline：串联 timeseries → indicator → label。

模块: evaluation
对应文档: evaluation/README.md

职责
----
对单个"因子资产"提供因子评估一条龙服务：

1. timeseries(4.1)：基于因子暴露与历史行情计算绩效时序（IC/分层/多空/换手/覆盖率）。
2. indicator(4.2)：消费绩效时序，计算统计指标矩阵、评分卡与 Dev4 评估摘要。
3. label(4.3)：依据评估摘要对因子贴标，并给出"入库推荐路由信息"。

设计原则：保持各子模块独立、文件级交接，不重写其内部逻辑；本模块只做编排与
轻量数据适配（把 4.1 产物组织为 4.2 期望的 bundle）。

边界（重要）：
- 本模块**不执行**真正的入库落地（不把因子物化进目标 tier 目录、不更新共享标签
  注册表）。label 步骤以"路由信息计算模式"（``materialize=False``）运行，仅返回
  标签与入库推荐路由信息；具体入库由更高级别调用管道统一负责。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.indicator import run_stage4_2
from evaluation.label import (
    LabelModuleConfig,
    default_label_module_config,
    run_label_pipeline,
)
from evaluation.label import DeepSeekClient
from evaluation.timeseries import (
    TimeseriesConfig,
    TimeseriesPaths,
    TimeseriesRunResult,
    run_timeseries_evaluation,
)

# 4.1 产物逻辑名 → 4.2 输入契约序列名 + 源文件。
# 说明：4.2 的 ``quantile_return_panel`` 需要 ``net_return`` 列，而 4.1 的
# ``quantile_backtest`` 仅含 ``gross_return``（分位层不计成本，见 timeseries FID §2.4）。
# 因此本模块在构造 bundle 时为分位面板补 ``net_return = gross_return``（4.1 无分位级
# 成本，二者口径等价），其余序列按 horizon 过滤后直接引用。
_TS_ARTIFACT_TO_SERIES: dict[str, str] = {
    "daily_rank_ic": "rank_ic_series",
    "daily_ic": "ic_series",
    "quantile_backtest": "quantile_return_panel",
    "top_minus_bottom": "top_minus_bottom_series",
    "long_short_returns": "long_short_return_series",
    "turnover_series": "turnover_series",
    "coverage_series": "coverage_series",
}


@dataclass(frozen=True)
class EvaluationPipelineResult:
    """评估 pipeline 一条龙运行结果。

    字段：
        factor_id: 当前因子 ID。
        eval_run_id: 本次评估运行唯一 ID。
        timeseries: 4.1 时序评估结果（产物路径与 manifest 索引）。
        indicator_by_horizon: horizon → 4.2 指标计算结果 dict（含 summary_scorecard、
            dev4_evaluation_summary、metrics_matrix、output_dir 等）。
        primary_horizon: 用于贴标/路由决策的主 horizon。
        summary_scorecard: 主 horizon 的评分卡。
        evaluation_summary: 主 horizon 的 Dev4 评估摘要（驱动 label 决策的输入）。
        tag_package: label 模块产出的完整标签包。
        admission_decision: 准入决策（含目标 tier）。
        route_record: 路由记录（含 ``target_factor_dir`` 入库推荐路由信息）。
        lifecycle_event: 生命周期事件。
        work_dir: 本次 pipeline 中间产物目录（各 horizon 的 bundle 与指标输出）。
    """

    factor_id: str
    eval_run_id: str
    timeseries: TimeseriesRunResult
    indicator_by_horizon: dict[int, dict[str, Any]]
    primary_horizon: int
    summary_scorecard: dict[str, Any]
    evaluation_summary: dict[str, Any]
    tag_package: dict[str, Any]
    admission_decision: dict[str, Any]
    route_record: dict[str, Any]
    lifecycle_event: dict[str, Any]
    work_dir: Path

    @property
    def route_recommendation(self) -> str:
        """评估模块对该因子的入库推荐方向（如 ``tier3a_core``）。"""
        return str(self.evaluation_summary.get("route_recommendation", ""))

    @property
    def target_tier(self) -> str:
        """路由决策得到的目标 tier（如 ``Tier3A``）。"""
        return str(self.admission_decision.get("tier", ""))

    @property
    def target_factor_dir(self) -> str:
        """入库推荐目标目录（路由信息，非实际落盘动作）。"""
        return str(self.route_record.get("target_factor_dir", ""))


def _horizons_of(config: TimeseriesConfig) -> list[int]:
    """返回去重升序的 horizon 列表。"""
    return sorted({int(h) for h in config.horizons})


def _write_horizon_bundle(
    artifacts: dict[str, Path],
    horizon: int,
    bundle_dir: Path,
) -> Path:
    """把 4.1 多 horizon 产物过滤为单 horizon，并组织成 4.2 输入 bundle。

    入参：
        artifacts: timeseries 产物逻辑名 → parquet 路径。
        horizon: 目标 horizon（4.2 要求单 horizon bundle，见 indicator FID §7）。
        bundle_dir: 本 horizon 的 bundle 输出目录。

    出参：
        Path：写出的 `performance_series_bundle.json` 路径。
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, str] = {}

    for artifact_name, series_name in _TS_ARTIFACT_TO_SERIES.items():
        src = artifacts.get(artifact_name)
        if src is None or not Path(src).exists():
            continue
        df = pd.read_parquet(src)
        if "horizon" in df.columns:
            df = df.loc[df["horizon"].astype("int64") == int(horizon)].copy()
        if df.empty:
            continue
        if series_name == "quantile_return_panel" and "net_return" not in df.columns:
            # 4.1 分位层不计成本，补 net_return = gross_return（口径等价）。
            df["net_return"] = df["gross_return"]
        out_path = bundle_dir / f"{series_name}.parquet"
        df.to_parquet(out_path, index=False)
        bundle[series_name] = str(out_path.resolve())

    # daily_rank_ic 同时含 rank_ic_t 与 kendall_tau_t，可复用为 kendall_tau_series。
    rank_ic_path = bundle.get("rank_ic_series")
    if rank_ic_path is not None:
        rank_df = pd.read_parquet(rank_ic_path)
        if "kendall_tau_t" in rank_df.columns:
            bundle["kendall_tau_series"] = rank_ic_path

    bundle_path = bundle_dir / "performance_series_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle_path


def _build_pipeline_logger(log_dir: Path, eval_run_id: str) -> logging.Logger:
    """为单次 pipeline 编排创建文件 logger。

    入参：
        log_dir: 编排日志目录（默认 ``database/log/evaluation``）。
        eval_run_id: 本次评估运行 ID，用于区分不同运行的日志行。

    出参：
        logging.Logger：绑定到 ``<log_dir>/evaluation_pipeline.log`` 的 logger。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"evaluation.pipeline.{eval_run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "evaluation_pipeline.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _close_logger_handlers(logger: logging.Logger) -> None:
    """关闭并移除本次 pipeline logger 的 handlers，避免文件句柄泄漏。"""
    for handler in list(logger.handlers):
        try:
            handler.close()
        finally:
            logger.removeHandler(handler)


def run_evaluation_pipeline(
    *,
    eval_run_id: str,
    pure_factor_dir: str | Path,
    market_data_path: str | Path,
    universe_path: str | Path,
    factor_id: str | None = None,
    timeseries_config: TimeseriesConfig | None = None,
    timeseries_paths: TimeseriesPaths | None = None,
    label_config: LabelModuleConfig | None = None,
    deepseek_client: DeepSeekClient | None = None,
    primary_horizon: int | None = None,
    work_dir: str | Path | None = None,
    operator: str = "evaluation.pipeline",
) -> EvaluationPipelineResult:
    """对单个因子资产执行 timeseries → indicator → label 一条龙评估。

    入参（上游因子资产 + 市场基本信息）：
        eval_run_id: 本次评估运行唯一 ID（用于子目录与产物追溯）。
        pure_factor_dir: 因子资产目录（PureFactor 三件套：candidate.json 含因子表达式/
            配置、data.parquet 含历史因子暴露值、manifest.json），同时作为 label 的
            ``source_factor_dir``。
        market_data_path: 历史行情数据路径（至少 datetime/asset + vwap 或 open）。
        universe_path: 市场基本信息路径（datetime/asset/is_active/is_tradable，
            可选 industry/mcap），提供行业与可交易约束、市值加权所需信息。
        factor_id: 可选预期 factor_id；传入时必须与因子资产数据一致。
        timeseries_config: 4.1 计算配置；为空时用 ``TimeseriesConfig()``。
        timeseries_paths: 4.1 输出/缓存/日志路径；为空时用 ``TimeseriesPaths()``。
        label_config: label 模块路径与环境配置；为空时用 ``default_label_module_config()``。
        deepseek_client: 可选语义贴标客户端；为空时 label 使用 ``config.deepseek`` 真实客户端。
            离线编排可注入具备 ``request_semantic_tags`` 的确定性客户端。
        primary_horizon: 用于贴标/路由的主 horizon；为空时取 horizons 升序首位。
        work_dir: pipeline 中间产物目录；为空时落在 timeseries 输出目录下的
            ``_evaluation_pipeline`` 子目录。
        operator: 写入生命周期/路由记录的操作者名称。

    出参：
        EvaluationPipelineResult：聚合 4.1 产物、各 horizon 的 4.2 指标，以及主 horizon
        的评分卡、评估摘要、标签包与入库推荐路由信息。

    异常：
        RuntimeError：timeseries 阶段输入/对齐校验失败时抛出。
        ValueError / KeyError：indicator 或 label 阶段契约校验失败时抛出。

    说明（边界）：
        本函数仅产出"统计指标 + 标签 + 入库推荐路由信息"，不执行真正入库落地
        （label 以 ``materialize=False`` 运行）。

    日志：
        编排级日志写入 ``<database/log/evaluation>/evaluation_pipeline.log``（评估日志
        根目录取自 ``label_config.log_dir`` 的父目录），记录 pipeline 启停与各阶段完成；
        各子模块仍写各自日志：timeseries → ``timeseries/<eval_run_id>.log``，indicator →
        ``indicator/stage4_2_metrics.log``（pipeline 已为其显式注入该目录，逐 horizon 追加），
        label → ``label/label_pipeline.log``。
    """
    ts_config = timeseries_config or TimeseriesConfig()
    horizons = _horizons_of(ts_config)
    chosen_horizon = int(primary_horizon) if primary_horizon is not None else horizons[0]
    if chosen_horizon not in horizons:
        raise ValueError(
            f"primary_horizon={chosen_horizon} not in configured horizons {horizons}"
        )

    if label_config is None:
        raise ValueError(
            "label_config 不能为空，请通过配置文件指定 label 参数。"
            "参考: all_configs/auto_factor_evaluation/config.yaml"
        )
    active_label_config = label_config
    # 评估各子模块的日志同根存放：复用 label 配置解析出的
    # ``database/log/evaluation`` 作为评估日志根目录，使 timeseries/indicator/label/
    # pipeline 四类日志落在同一处，路径与文档约定一致。
    evaluation_log_root = active_label_config.log_dir.parent
    indicator_log_dir = evaluation_log_root / "indicator"
    logger = _build_pipeline_logger(evaluation_log_root, eval_run_id)
    logger.info(
        "Evaluation pipeline started | eval_run_id=%s | factor_id=%s | horizons=%s | primary_horizon=%s",
        eval_run_id,
        factor_id,
        horizons,
        chosen_horizon,
    )
    try:
        # 1) 4.1 时序评估。
        ts_result = run_timeseries_evaluation(
            eval_run_id=eval_run_id,
            pure_factor_dir=Path(pure_factor_dir),
            market_data_path=Path(market_data_path),
            universe_path=Path(universe_path),
            factor_id=factor_id,
            config=ts_config,
            paths=timeseries_paths,
        )
        logger.info("Timeseries stage done | out_dir=%s", ts_result.out_dir)

        base_work_dir = (
            Path(work_dir)
            if work_dir is not None
            else ts_result.out_dir / "_evaluation_pipeline"
        )
        base_work_dir.mkdir(parents=True, exist_ok=True)

        # 2) 逐 horizon 组织 bundle 并运行 4.2 指标计算。
        indicator_by_horizon: dict[int, dict[str, Any]] = {}
        for horizon in horizons:
            bundle_dir = base_work_dir / f"indicator_input_h{horizon}"
            _write_horizon_bundle(ts_result.artifacts, horizon, bundle_dir)
            indicator_out = base_work_dir / f"indicator_output_h{horizon}"
            indicator_by_horizon[horizon] = run_stage4_2(
                input_dir=bundle_dir,
                output_dir=indicator_out,
                log_dir=indicator_log_dir,
            )
            logger.info("Indicator stage done | horizon=%s | output_dir=%s", horizon, indicator_out)

        primary_result = indicator_by_horizon[chosen_horizon]
        # indicator 的 summary_scorecard 是带 factor_id/eval_run_id/horizon 的包装结构，
        # 内层 ``summary_scorecard`` 才是指标卡；这里直接暴露内层指标卡，便于下游读取。
        summary_payload = primary_result["summary_scorecard"]
        summary_scorecard = dict(summary_payload.get("summary_scorecard", summary_payload))
        evaluation_summary = dict(primary_result["dev4_evaluation_summary"])
        evaluation_summary_path = (
            base_work_dir / f"indicator_output_h{chosen_horizon}" / "dev4_evaluation_summary.json"
        )

        # 3) label 贴标 + 路由（仅路由信息，不物化入库）。
        label_result = run_label_pipeline(
            source_factor_dir=Path(pure_factor_dir),
            evaluation_summary_path=evaluation_summary_path,
            config=active_label_config,
            operator=operator,
            deepseek_client=deepseek_client,
            materialize=False,
        )
        logger.info(
            "Label stage done | factor_id=%s | route_recommendation=%s | target_tier=%s",
            label_result["factor_id"],
            evaluation_summary.get("route_recommendation"),
            label_result["admission_decision"].get("tier"),
        )

        logger.info("Evaluation pipeline finished | eval_run_id=%s", eval_run_id)
        return EvaluationPipelineResult(
            factor_id=str(label_result["factor_id"]),
            eval_run_id=eval_run_id,
            timeseries=ts_result,
            indicator_by_horizon=indicator_by_horizon,
            primary_horizon=chosen_horizon,
            summary_scorecard=summary_scorecard,
            evaluation_summary=evaluation_summary,
            tag_package=label_result["tag_package"],
            admission_decision=label_result["admission_decision"],
            route_record=label_result["route_record"],
            lifecycle_event=label_result["lifecycle_event"],
            work_dir=base_work_dir,
        )
    except Exception:
        logger.exception("Evaluation pipeline failed | eval_run_id=%s", eval_run_id)
        raise
    finally:
        _close_logger_handlers(logger)


__all__ = ["EvaluationPipelineResult", "run_evaluation_pipeline"]
