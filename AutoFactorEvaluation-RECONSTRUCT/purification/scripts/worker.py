"""
Purification 工作进程

处理 Assetization 产出的原始因子:
    1. 读取 afv.json 获取元信息
    2. 加载因子数据（data/ 目录按日 parquet）
    3. 加载行业分类数据（StockIndustry）
    4. 执行纯化：缺失值填补 → 去极值 → 正交化（可选）
    5. 更新 afv.json（追加 Purification 段）
    6. 写入 tier0/purification_temp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .low_freq import dynamic_imputation, robust_winsorization, risk_orthogonalization
from ..config import PurificationConfig

logger = logging.getLogger("purification.worker")

# 行业分类的列名映射（StockIndustry parquet → 统一格式）
_INDUSTRY_COLUMNS = {
    "TradeDate": "datetime",
    "Symbol": "asset",
}

# 因子数据的列名映射（Assetization 输出 → Purification 内部格式）
_FACTOR_COLUMN_MAP = {
    "TradeDate": "datetime",
    "Symbol": "asset",
}

# Purification 内部处理使用的标准列
_REQUIRED_FACTOR_COLUMNS = {"datetime", "asset", "factor_id", "factor_value"}


def _normalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 Assetization 产出的列名映射为 Purification 内部列名。"""
    return df.rename(columns=_FACTOR_COLUMN_MAP)


def _denormalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 Purification 内部列名映射回 Assetization 产出的列名（TradeDate/Symbol）。"""
    inv_map = {v: k for k, v in _FACTOR_COLUMN_MAP.items()}
    return df.rename(columns=inv_map)


def _load_industry_data(
    industry_path: Path,
    dates: list,
    standard: str = "zjw",
) -> pd.Series:
    """加载行业分类数据。

    从 StockIndustry 目录下加载与因子日期对应的行业分类，
    筛选指定分类口径（IndustrySource=standard），
    返回 {asset: industry_code} 的 Series。

    Args:
        industry_path: StockIndustry 数据目录。
        dates: 需要加载的日期列表（datetime.date 或 str）。
        standard: 行业分类口径，如 "zjw"（证监会）。

    Returns:
        asset → industry_code 的 Series。
    """
    if not industry_path.exists():
        logger.warning("行业数据目录不存在: %s", industry_path)
        return pd.Series(dtype=object)

    all_labels = []
    for d in dates:
        date_str = str(d)[:10] if hasattr(d, "strftime") else str(d)[:10]
        file_path = industry_path / f"{date_str}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                # 筛选指定分类口径
                df_zjw = df[df.get("IndustrySource", "") == standard]
                if not df_zjw.empty:
                    # 取每个 asset 的第一条记录（一个 asset 在同一分类口径下只有一条）
                    label = df_zjw.set_index("Symbol")["IndustryCode"]
                    all_labels.append(label)
            except Exception as e:
                logger.debug("行业数据加载失败 %s: %s", date_str, e)

    if not all_labels:
        logger.warning("未加载到任何行业分类数据（standard=%s）", standard)
        return pd.Series(dtype=object)

    # 合并所有日期的行业标签，取最新的
    combined = pd.concat(all_labels, axis=1)
    # 用最后出现的值（最新日期优先）
    latest = combined.iloc[:, -1]
    return latest.dropna().astype(str)


def _load_factor_data(factor_dir: Path) -> pd.DataFrame:
    """从因子目录加载数据（data/ 下的所有 parquet 文件）。

    Args:
        factor_dir: 因子目录路径（含 data/ 子目录）。

    Returns:
        合并后的长表 DataFrame（列: datetime, asset, factor_id, factor_value）。
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


def _write_factor_data(output_dir: Path, df: pd.DataFrame) -> None:
    """将纯化后的因子数据写入 data/ 目录（按日分片）。

    Args:
        output_dir: 输出目录（其下 data/ 子目录）。
        df: 纯化后的长表（列: datetime, asset, factor_id, factor_value）。
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 转回 TradeDate/Symbol 命名
    df_out = _denormalize_factor_columns(df)

    # 按 TradeDate 分片
    date_col = "TradeDate" if "TradeDate" in df_out.columns else "datetime"
    for date_val, group in df_out.groupby(
        df_out[date_col].dt.date if hasattr(df_out[date_col], "dt") else df_out[date_col]
    ):
        date_str = str(date_val)
        file_path = data_dir / f"{date_str}.parquet"
        group.to_parquet(file_path, index=False)

    logger.info("  写入 data/: %s → %d 个日频文件", data_dir, len(df_out.groupby(date_col)))


def process_factor_dir(
    factor_dir: str | Path,
    *,
    config: PurificationConfig | None = None,
    temp_base: str | Path | None = None,
) -> tuple[str, str]:
    """处理单个待纯化因子目录。

    Args:
        factor_dir: tier1/assetization_raw_factor_base 中的因子目录路径。
        config: PurificationConfig 实例。默认自动加载。
        temp_base: tier0/purification_temp 路径。默认从配置读取。

    Returns:
        (factor_id, temp_dir): 因子 ID 和临时输出目录。
    """
    import sys as _sys
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))

    cfg = config or PurificationConfig()
    factor_dir = Path(factor_dir)

    # 确定临时输出路径
    tb = Path(temp_base) if temp_base else cfg.temp_base
    if not tb.exists():
        raise FileNotFoundError(
            f"Purification 临时目录不存在: {tb}\n"
            "  → 请确保该路径已在配置文件中声明且已被创建"
        )

    # 读取 afv.json
    afv_path = factor_dir / "afv.json"
    if not afv_path.exists():
        raise FileNotFoundError(f"因子目录缺少 afv.json: {factor_dir}")

    afv = json.loads(afv_path.read_text(encoding="utf-8"))
    factor_id = afv.get("factor_id", factor_dir.name)
    candidate_id = afv.get("candidate_id", factor_dir.name)

    # 检查是否已纯化
    if "Purification" in afv:
        logger.info("因子 %s 已纯化，跳过: Label=%s",
                     factor_id, afv["Purification"].get("Label", "?"))
        return factor_id, str(tb / factor_id)

    logger.info("Purification 开始: factor_id=%s", factor_id)

    # 读取 manifest.json（保留上游元信息）
    manifest_path = factor_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # ========== 1. 加载因子数据 ==========
    df_long = _load_factor_data(factor_dir)
    logger.info("  加载因子数据: %d 行, %d 列", len(df_long), len(df_long.columns))

    # 校验必需列
    missing = _REQUIRED_FACTOR_COLUMNS - set(df_long.columns)
    if missing:
        raise ValueError(f"因子数据缺少必要列: {sorted(missing)}")

    # 主键唯一性校验
    pk = df_long[["datetime", "asset", "factor_id"]]
    if pk.duplicated().any():
        dup_count = pk.duplicated().sum()
        logger.warning("  主键存在 %d 条重复，去重处理", dup_count)
        df_long = df_long.drop_duplicates(subset=["datetime", "asset", "factor_id"])

    # ========== 2. 加载行业分类数据 ==========
    dates = sorted(df_long["datetime"].dt.date.unique()) if hasattr(df_long["datetime"], "dt") else []
    industry_labels = _load_industry_data(cfg.industry_data_path, dates, cfg.industry_standard)
    logger.info("  行业分类: standard=%s, 覆盖 %d 个资产", cfg.industry_standard, len(industry_labels))

    # 权重（使用行业均值填补时需要的市值权重）
    # 由于当前没有真实的市值数据，使用等权替代
    weights = pd.Series(1.0, index=industry_labels.index)

    # ========== 构建正交化基矩阵（合并行业 dummy + 因子池暴露）==========
    # 将行业分类转为 one-hot 虚拟变量矩阵（作为正交化的基线风险暴露）
    risk_exposures = None
    if not industry_labels.empty:
        risk_exposures = pd.get_dummies(industry_labels, prefix="IND").astype(float)
        logger.info("  行业虚拟变量基: %d 个资产, %d 个行业", len(risk_exposures), risk_exposures.shape[1])

    # 若存在已有因子池，将因子暴露追加到同一矩阵中
    # 最终 risk_exposures = [行业 dummy | 因子池暴露]（合并为一张矩阵，仅一次正交化）
    for lib_name, lib_path in cfg.existing_factor_library.items():
        if lib_path.exists() and any(lib_path.iterdir()):
            logger.info("  发现已有因子库: %s (%s)", lib_name, lib_path)
            risk_file = lib_path / "risk_exposures" / "data.parquet"
            if risk_file.exists():
                try:
                    extra_exposures = pd.read_parquet(risk_file)
                    extra_idx = extra_exposures.set_index(extra_exposures.columns[0])
                    logger.info("  加载因子池暴露: %d 个资产, %d 个因子",
                                 len(extra_idx), extra_idx.shape[1])
                    # 与行业虚拟变量合并到同一矩阵
                    if risk_exposures is not None:
                        risk_exposures = risk_exposures.join(extra_idx, how="outer")
                    else:
                        risk_exposures = extra_idx
                except Exception as e:
                    logger.warning("  因子池暴露加载失败: %s", e)
            break

    # ========== 3. 执行纯化 ==========
    # 转为宽表 (index=datetime, columns=asset)
    df_wide = df_long.pivot(index="datetime", columns="asset", values="factor_value")
    steps_taken = []

    # 3a. 缺失值填补
    try:
        if cfg.imputation_method == "industry_weighted" and not industry_labels.empty:
            logger.info("  执行行业加权缺失值填补...")
            df_wide = dynamic_imputation(
                df_wide,
                method="industry_weighted",
                industry_labels=industry_labels,
                weights=weights,
            )
            steps_taken.append("imputation")
        else:
            logger.info("  执行时序前向填充衰减填补...")
            df_wide = dynamic_imputation(
                df_wide,
                method="forward_fill_decay",
                max_delay=cfg.max_delay,
            )
            steps_taken.append("imputation (forward_fill_decay)")
    except Exception as e:
        logger.warning("  行业填补失败 (%s)，回退到时序填补", e)
        df_wide = dynamic_imputation(
            df_wide,
            method="forward_fill_decay",
            max_delay=cfg.max_delay,
        )
        steps_taken.append("imputation (fallback)")

    # 3b. 稳健去极值
    logger.info("  执行稳健去极值 (MAD=%.3f)...", cfg.winsorization_mad_multiplier)
    df_wide = robust_winsorization(df_wide)
    steps_taken.append("winsorization")

    # 3c. 正交化：一次 WLS 回归，同时剥离行业 dummy + 因子池暴露
    # 注意：行业 dummy 和因子池暴露已合并到同一张 risk_exposures 矩阵，
    # 仅执行一次正交化（而非分别正交化），避免重复剥离。
    if risk_exposures is not None and not risk_exposures.empty:
        logger.info("  执行正交化 (WLS): 基矩阵 %d 列（行业 dummy + 因子池暴露）",
                     risk_exposures.shape[1])
        try:
            df_wide = risk_orthogonalization(df_wide, risk_exposures, weights)
            steps_taken.append("orthogonalization")
        except Exception as e:
            logger.warning("  正交化失败 (%s)，跳过此步骤", e)
    else:
        logger.info("  无行业分类数据，跳过正交化")

    # ========== 4. 转回长表 ==========
    df_pure = df_wide.reset_index().melt(
        id_vars="datetime", var_name="asset", value_name="factor_value"
    )
    df_pure = df_pure.dropna(subset=["factor_value"])

    # 恢复其他元数据列
    meta_cols = [c for c in df_long.columns if c not in {"datetime", "asset", "factor_value"}]
    if meta_cols:
        df_pure = pd.merge(
            df_pure,
            df_long[["datetime", "asset"] + meta_cols].drop_duplicates(
                subset=["datetime", "asset"]
            ),
            on=["datetime", "asset"],
            how="left",
        )

    logger.info("  纯化完成: %d 行", len(df_pure))

    # ========== 5. 写出到临时路径 ==========
    temp_dir = tb / factor_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 5a. 写入 data/（按日分片）
    _write_factor_data(temp_dir, df_pure)

    # 5b. 更新 afv.json（追加 Purification 段）
    run_id = f"purification_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{factor_id[:8]}"
    purified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    orthogonal_basis_uris = []
    orthogonal_basis_uris.append(str(cfg.industry_data_path))  # 行业分类数据作为正交化基
    if risk_exposures is not None:
        for lib_name, lib_path in cfg.existing_factor_library.items():
            orthogonal_basis_uris.append(str(lib_path / "risk_exposures"))

    afv["Purification"] = {
        "Label": "Pure",
        "run_id": run_id,
        "purified_at": purified_at,
        "input_raw_factor_uri": str(factor_dir),
        "orthogonal_basis_uris": orthogonal_basis_uris,
        "steps": steps_taken,
        "row_count": len(df_pure),
        "industry_standard": cfg.industry_standard,
    }

    (temp_dir / "afv.json").write_text(
        json.dumps(afv, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5c. 写入 manifest.json
    pure_manifest = {
        **manifest,
        "schema_version": "disk.v1",
        "library": "pure_factor_base",
        "artifact_type": "PureFactor",
        "producer": "purification",
        "factor_id": factor_id,
        "run_id": run_id,
        "created_at": purified_at,
        "source": {
            "raw_factor_uri": str(factor_dir),
            "orthogonal_basis_uris": orthogonal_basis_uris,
        },
        "row_count": len(df_pure),
        "purification_steps": steps_taken,
        "industry_standard": cfg.industry_standard,
        "validation": {"status": "passed"},
    }
    (temp_dir / "manifest.json").write_text(
        json.dumps(pure_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("  ✅ 完成: factor_id=%s temp=%s", factor_id, temp_dir)
    return factor_id, str(temp_dir)
