"""
Evaluation 工作进程

处理 Purification 产出的纯化因子:
    1. 读取 afv.json 获取元信息
    2. 加载因子数据（data/ 目录按日 parquet）
    3. 加载行情数据（StockDailyBar 按日 parquet）
    4. 执行评估流水线：timeseries → indicator → label
    5. 更新 afv.json（追加 Evaluation 段）
    6. 写入 tier0/evaluation_temp
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import EvaluationConfig

logger = logging.getLogger("evaluation.worker")

# 因子数据列名映射（Purification 输出 → Evaluation 内部格式）
_FACTOR_COLUMN_MAP = {
    "TradeDate": "datetime",
    "Symbol": "asset",
}

# 行情数据列名映射（StockDailyBar 格式 → Evaluation 格式）
_MARKET_COLUMN_MAP = {
    "TradeDate": "datetime",
    "Symbol": "asset",
}


def _normalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 Purification 产出的列名映射为 Evaluation 内部列名。"""
    return df.rename(columns=_FACTOR_COLUMN_MAP)


def _denormalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 Evaluation 内部列名映射回 Purification 产出的列名。"""
    inv_map = {v: k for k, v in _FACTOR_COLUMN_MAP.items()}
    return df.rename(columns=inv_map)


def _load_factor_data(factor_dir: Path) -> pd.DataFrame:
    """从因子目录加载数据（data/ 下的所有 parquet 文件）。

    Args:
        factor_dir: 因子目录路径（含 data/ 子目录）。

    Returns:
        合并后的长表 DataFrame，列名已标准化为 datetime/asset/factor_id/factor_value。
    """
    data_dir = factor_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"因子目录缺少 data/ 子目录: {factor_dir}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"data/ 下无 parquet 文件: {data_dir}")

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning("读取 %s 失败: %s", f, e)

    if not dfs:
        raise ValueError(f"无法读取任何 parquet 文件: {data_dir}")

    result = pd.concat(dfs, ignore_index=True)
    # 标准化列名
    result = _normalize_factor_columns(result)
    return result


def _load_market_data_all(market_data_path: Path) -> pd.DataFrame:
    """加载 StockDailyBar 目录下所有日频行情数据（使用全局缓存）。

    Args:
        market_data_path: StockDailyBar 目录路径。

    Returns:
        合并后的行情 DataFrame，列名已标准化。
    """
    from cache_utils import load_market_data_cached

    df = load_market_data_cached(
        market_data_path=market_data_path,
        rename_columns=_MARKET_COLUMN_MAP,
    )
    # 去重
    result = df.drop_duplicates(
        subset=["datetime", "asset"]
    ).reset_index(drop=True)
    return result


def _build_universe_from_market(market_df: pd.DataFrame) -> pd.DataFrame:
    """从行情数据构建 universe 表。

    将行情数据中的 (datetime, asset) 对视为可交易资产，
    生成 is_active=True, is_tradable=True 的 universe。

    Args:
        market_df: 行情 DataFrame（含 datetime, asset 列）。

    Returns:
        universe DataFrame。
    """
    universe = market_df[["datetime", "asset"]].drop_duplicates().copy()
    universe["is_active"] = True
    universe["is_tradable"] = True
    return universe


def process_factor_dir(
    factor_dir: str | Path,
    *,
    config: EvaluationConfig | None = None,
    temp_base: str | Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """处理单个待评估因子目录。

    Args:
        factor_dir: tier1/purification_pure_factor_base 中的因子目录路径。
        config: EvaluationConfig 实例。默认自动加载。
        temp_base: tier0/evaluation_temp 路径。默认从配置读取。

    Returns:
        (factor_id, temp_dir, route_info): 因子 ID、临时输出目录、路由信息。
    """
    import sys as _sys
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))

    cfg = config or EvaluationConfig()
    factor_dir = Path(factor_dir)

    # 确定临时输出路径
    tb = Path(temp_base) if temp_base else cfg.temp_base
    if not tb.exists():
        raise FileNotFoundError(
            f"Evaluation 临时目录不存在: {tb}\n"
            "  → 请确保该路径已在配置文件中声明且已被创建"
        )

    # 读取 afv.json
    afv_path = factor_dir / "afv.json"
    if not afv_path.exists():
        raise FileNotFoundError(f"因子目录缺少 afv.json: {factor_dir}")

    afv = json.loads(afv_path.read_text(encoding="utf-8"))
    factor_id = afv.get("factor_id", factor_dir.name)
    candidate_id = afv.get("candidate_id", factor_dir.name)

    # 检查是否已评估
    if "Evaluation" in afv:
        logger.info("因子 %s 已评估，跳过: Label=%s",
                     factor_id, afv["Evaluation"].get("Label", "?"))
        return factor_id, str(tb / factor_id), afv.get("Evaluation", {})

    logger.info("Evaluation 开始: factor_id=%s", factor_id)

    # 读取 manifest.json
    manifest_path = factor_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # ========== 1. 加载因子数据 ==========
    df_factor = _load_factor_data(factor_dir)
    logger.info("  加载因子数据: %d 行", len(df_factor))

    # 确保必需的列存在
    for col in ["factor_id", "factor_value"]:
        if col not in df_factor.columns:
            raise ValueError(f"因子数据缺少列: {col}")

    # 补全时间列
    if "knowledge_ts" not in df_factor.columns:
        df_factor["knowledge_ts"] = df_factor["datetime"]
    if "decision_ts" not in df_factor.columns:
        df_factor["decision_ts"] = df_factor["datetime"]

    # ========== 2. 加载行情数据 ==========
    market_data_path = cfg.market_data_path
    logger.info("  加载行情数据: %s", market_data_path)
    market_df = _load_market_data_all(market_data_path)
    logger.info("  行情数据: %d 行, 日期范围 %s ~ %s",
                 len(market_df),
                 market_df["datetime"].min().date(),
                 market_df["datetime"].max().date())

    # ========== 3. 构建 universe ==========
    universe_df = _build_universe_from_market(market_df)
    logger.info("  Universe: %d 条记录", len(universe_df))

    # ========== 4. 创建临时工作目录（整合为单文件 data.parquet）==========
    # 使用 factor_id 作为目录名，以满足 load_pure_factor_input 的文件夹名校验
    import tempfile as _tf
    _tmp_root = Path(_tf.mkdtemp(prefix="eval_"))
    tmp_dir = _tmp_root / factor_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 4a. 写入单文件 data.parquet（timeseries 模块期望的格式）
        eval_data_path = tmp_dir / "data.parquet"
        df_factor.to_parquet(eval_data_path, index=False)

        # 4b. 写入单文件 market_data.parquet（统一列名格式）
        market_df_out = market_df.rename(columns={
            "TradeDate": "datetime",
            "Symbol": "asset",
        })
        # 将所有列名转为小写（Vwap→vwap, Open→open 等）
        market_df_out.columns = [c.lower() for c in market_df_out.columns]
        eval_market_path = tmp_dir / "market_data.parquet"
        market_df_out.to_parquet(eval_market_path, index=False)

        # 4c. 写入单文件 universe.parquet
        eval_universe_path = tmp_dir / "universe.parquet"
        universe_df.to_parquet(eval_universe_path, index=False)

        # 4d. 写入 candidate.json（afv.json 的副本）
        eval_candidate_path = tmp_dir / "candidate.json"
        eval_candidate_path.write_text(
            json.dumps(afv, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 4e. 写入 manifest.json
        eval_manifest_path = tmp_dir / "manifest.json"
        eval_manifest = {
            **manifest,
            "artifact_type": "PureFactor",
            "files": {"candidate": "candidate.json", "data": "data.parquet"},
            "validation": {"status": "passed"},
        }
        eval_manifest_path.write_text(
            json.dumps(eval_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ========== 5. 运行评估 pipeline ==========
        eval_run_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{factor_id[:8]}"

        from evaluation.pipeline import run_evaluation_pipeline
        from evaluation.timeseries import TimeseriesConfig, TimeseriesPaths


        # 构建 TimeseriesConfig
        ts_config = TimeseriesConfig(
            horizons=cfg.horizons,
            min_assets=cfg.min_assets,
            n_quantiles=cfg.n_quantiles,
            factor_direction=cfg.factor_direction,
            weighting=cfg.weighting,
            cost_bps=cfg.cost_bps,
        )

        # 构建 TimeseriesPaths：使用全局共享 forward return 缓存
        from cache_utils import get_forward_return_cache_dir
        ts_paths = TimeseriesPaths(
            project_root=_PROJECT_ROOT,
            output_dir=tmp_dir / "timeseries_output",
            cache_dir=get_forward_return_cache_dir(),
            log_dir=tmp_dir / "timeseries_logs",
        )

        # 构建 label config（全部路径从 ConfigManager 获取，无内部默认值）
        from config_manager import ConfigManager as _CM
        from evaluation.label.scripts.schemas import LabelModuleConfig
        _shared_cfg = _CM()
        label_config = LabelModuleConfig(
            project_root=_PROJECT_ROOT,
            tag_policy_path=_shared_cfg.label_config_file("tag_policy"),
            route_policy_path=_shared_cfg.label_config_file("route_policy"),
            label_registry_path=_shared_cfg.label_config_file("label_registry"),
            state_dir=_shared_cfg.path("evaluation_label_cache"),
            log_dir=_shared_cfg.logging_directory / "evaluation" / "label",
            tier1_pure_factor_base_dir=_shared_cfg.path("purification_pure_factor_base"),
            tier2_fix_base_dir=_shared_cfg.output_tier("tier2_incubator"),
            tier2x_llm_mutation_base_dir=_shared_cfg.output_tier("tier2x_optimization_factory"),
            tier3a_core_base_dir=_shared_cfg.output_tier("tier3a_core"),
            tier3b_satellite_base_dir=_shared_cfg.output_tier("tier3b_satellite"),
            tier3c_feature_material_base_dir=_shared_cfg.output_tier("tier3c_feature"),
            tier3d_operation_storage_base_dir=_shared_cfg.output_tier("tier3d_optimized_reserve"),
            tier4_anti_sample_base_dir=_shared_cfg.output_tier("tier4_archive"),
        )

        # 读取 .env 配置 DeepSeek API key 和模型
        _env_path = _shared_cfg.label_config_file("deepseek_env_file")
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()

        from evaluation.label.scripts.deepseek_client import DeepSeekClient
        from evaluation.label.scripts.schemas import DeepSeekConfig

        # 使用 deepseek-v4-flash 模型
        _deepseek_cfg = DeepSeekConfig(
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="DEEPSEEK_V4_FLASH_API_KEY",
        )
        deepseek_client = DeepSeekClient(_deepseek_cfg)
        label_config.deepseek = _deepseek_cfg

        logger.info("  执行评估流水线（DeepSeek 模型: %s）...", _deepseek_cfg.model)
        result = run_evaluation_pipeline(
            eval_run_id=eval_run_id,
            pure_factor_dir=tmp_dir,
            market_data_path=eval_market_path,
            universe_path=eval_universe_path,
            factor_id=factor_id,
            timeseries_config=ts_config,
            timeseries_paths=ts_paths,
            deepseek_client=deepseek_client,
            label_config=label_config,
            primary_horizon=cfg.horizons[0],
            work_dir=tmp_dir / "pipeline_work",
            operator="evaluation.worker",
        )

        logger.info("  评估完成: factor_id=%s route_recommendation=%s target_tier=%s",
                     result.factor_id,
                     result.route_recommendation,
                     result.target_tier)

        # ========== 6. 写出到临时路径（仅保留 data/ + afv.json + manifest.json）==========
        temp_dir = tb / factor_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 6a. 复制原始的 data/ 目录（评估不修改因子值）
        src_data_dir = factor_dir / "data"
        dst_data_dir = temp_dir / "data"
        if src_data_dir.exists():
            if dst_data_dir.exists():
                shutil.rmtree(dst_data_dir)
            shutil.copytree(src_data_dir, dst_data_dir)

        # 6b. 更新 afv.json（追加 Evaluation 段，集中存储所有评估标识）
        evaluated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 提取精简的关键指标
        scorecard = result.summary_scorecard or {}
        key_metrics = {}
        for _k in ["rank_ic_mean", "rank_ic_ir", "rank_ic_win_rate",
                     "top_minus_bottom_mean", "long_short_sharpe",
                     "turnover", "universe_coverage_mean"]:
            if _k in scorecard:
                key_metrics[_k] = scorecard[_k]

        # 提取标签概要
        tag_pkg = result.tag_package or {}
        tags_summary = {}
        rule_tags = tag_pkg.get("rule_tags", {})
        if rule_tags:
            for _cat in ["performance_tags", "action_tags"]:
                _vals = rule_tags.get(_cat, [])
                if _vals:
                    tags_summary[_cat] = _vals
        deepseek_tags = tag_pkg.get("deepseek_tags", {})
        if deepseek_tags.get("primary_label"):
            tags_summary["semantic_label"] = deepseek_tags["primary_label"]
            tags_summary["confidence"] = deepseek_tags.get("confidence", 0)

        afv["Evaluation"] = {
            "Label": "Evaluated",
            "eval_run_id": eval_run_id,
            "evaluated_at": evaluated_at,
            "input_pure_factor_uri": str(factor_dir),
            "route_recommendation": result.route_recommendation,
            "target_tier": result.target_tier,
            "horizons": cfg.horizons,
            "key_metrics": key_metrics,
            "tags": tags_summary,
            "admission_reason": result.admission_decision.get("reason", ""),
            "row_count": len(df_factor),
        }

        (temp_dir / "afv.json").write_text(
            json.dumps(afv, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 6c. 更新 manifest.json（轻量）
        eval_manifest_out = {
            **manifest,
            "factor_id": factor_id,
            "eval_run_id": eval_run_id,
            "created_at": evaluated_at,
            "producer": "evaluation",
            "artifact_type": "EvaluatedFactor",
            "route_recommendation": result.route_recommendation,
            "target_tier": result.target_tier,
            "row_count": len(df_factor),
            "validation": {"status": "passed"},
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(eval_manifest_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("  ✅ 完成: factor_id=%s temp=%s route=%s",
                     factor_id, temp_dir, result.route_recommendation)

        # 打包路由信息返回
        route_info = {
            "route_recommendation": result.route_recommendation,
            "target_tier": result.target_tier,
            "admission_decision": result.admission_decision,
        }
        return factor_id, str(temp_dir), route_info
    finally:
        # 清理临时工作目录
        shutil.rmtree(_tmp_root, ignore_errors=True)
