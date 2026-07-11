"""
AutoFactorEvaluation 统一配置管理器（v2 — 三基座设计）

外部传参全部集中在一个 config.yaml 中，按以下分类:

  1. 基座路径（3个）：candidate_pool, factor_pool, local_tmp
  2. 数据源路径：market_data, industry_data
  3. DeepSeek API 密钥
  4. 模块参数（暂保留，后续整理）

所有非路径类参数直接从配置文件字段读取，无代码级默认值。
路径类参数通过 path_convention 的三棵树（CANDIDATE_TREE / FACTOR_TREE / LOCAL_TREE）自动推导子路径。

使用方式:
    export AFVCONFIG=/path/to/config_dir
    config = ConfigManager()

    # 或显式传入
    config = ConfigManager(config_dir="/path/to/config_dir")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from path_convention import (
    resolve_candidate as _resolve_candidate,
    resolve_factor as _resolve_factor,
    resolve_local as _resolve_local,
    CANDIDATE_TREE, FACTOR_TREE, LOCAL_TREE,
)

# 默认查找路径：项目内或项目外同级
_PROJECT_ROOT = Path(__file__).resolve().parent
_INTERNAL_CFG = _PROJECT_ROOT / "all_configs" / "auto_factor_evaluation"
_EXTERNAL_CFG = _PROJECT_ROOT.parent / "all_configs" / "auto_factor_evaluation"
_DEFAULT_CONFIG_DIR = _INTERNAL_CFG if _INTERNAL_CFG.exists() else _EXTERNAL_CFG

_DEFAULT_LOCAL_TMP = "/tmp/auto_factor_evaluation_tmp"


class ConfigError(Exception):
    """配置错误。"""
    pass


class ConfigManager:
    """统一配置管理器（v2）。

    三大基座路径:
      - candidate_pool: COS 候选因子池（子路径: candidate/, archive/）
      - factor_pool:    COS 持久化存储（子路径: tier1~tier4）
      - local_tmp:      本地临时工作区（子路径: tier0、缓存、日志）

    所有路径自动推导，无需逐一声明。
    """

    def __init__(self, config_dir: str | Path | None = None):
        # ── 确定配置目录 ──
        if config_dir is not None:
            self.config_dir = Path(config_dir)
        else:
            env_dir = os.environ.get("AFVCONFIG")
            self.config_dir = Path(env_dir) if env_dir else _DEFAULT_CONFIG_DIR

        # ── 加载 YAML ──
        master_path = self.config_dir / "config.yaml"
        if not master_path.exists():
            raise FileNotFoundError(
                f"主配置文件不存在: {master_path}\n"
                f"请确保配置目录正确，或创建 {master_path}"
            )
        with open(master_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        if not self._cfg:
            raise ValueError(f"配置文件为空: {master_path}")

        # ── 读取三大基座 ──
        self._candidate_pool: str | None = self._cfg.get("candidate_pool")
        self._factor_pool: str | None = self._cfg.get("factor_pool")
        self._local_tmp: str | None = self._cfg.get("local_tmp")

        # ── 自动 ensure COS 端目录 ──
        if self._candidate_pool:
            self._ensure_cos_dirs("candidate_pool", self._candidate_pool)
        if self._factor_pool:
            self._ensure_cos_dirs("factor_pool", self._factor_pool)

    # ============================================================
    # COS 目录自动创建
    # ============================================================

    @staticmethod
    def _ensure_cos_dirs(kind: str, base_path: str):
        try:
            from cos_utils import ensure_candidate_pool_dirs, ensure_factor_pool_dirs
            fn = ensure_candidate_pool_dirs if kind == "candidate_pool" else ensure_factor_pool_dirs
            fn(base_path)
        except Exception:
            pass

    # ============================================================
    # 内部工具
    # ============================================================

    def _require(self, *keys: str) -> Any:
        val: Any = self._cfg
        for k in keys:
            if not isinstance(val, dict) or k not in val:
                raise KeyError(f"config.yaml 缺少必填字段: {'/'.join(keys)} (缺失: {k})")
            val = val[k]
        return val

    # ============================================================
    # 基座路径属性
    # ============================================================

    @property
    def candidate_pool(self) -> str | None:
        """候选因子池 COS 基座。"""
        return self._candidate_pool

    @property
    def factor_pool(self) -> str | None:
        """因子持久化 COS 基座。"""
        return self._factor_pool

    @property
    def local_tmp(self) -> str:
        """本地临时基座（默认 /tmp/auto_factor_evaluation_tmp）。"""
        return self._local_tmp or _DEFAULT_LOCAL_TMP

    # ============================================================
    # 路径推导
    # ============================================================

    def path(self, key: str) -> Any:
        """获取逻辑路径。

        按顺序查找: candidate_tree → factor_tree → local_tree → 抛 KeyError。
        """
        if self._candidate_pool is not None and key in CANDIDATE_TREE:
            return _resolve_candidate(self._candidate_pool, key)
        if self._factor_pool is not None and key in FACTOR_TREE:
            return _resolve_factor(self._factor_pool, key)
        if key in LOCAL_TREE:
            return _resolve_local(self.local_tmp, key)
        raise KeyError(
            f"未知路径逻辑键名: {key!r}。"
            f"可用键名: {', '.join(sorted({**CANDIDATE_TREE, **FACTOR_TREE, **LOCAL_TREE}))}"
        )

    def output_tier(self, key: str) -> Any:
        """获取 tier 输出路径（仅 factor_pool 相关）。"""
        if self._factor_pool is not None and key in FACTOR_TREE:
            return _resolve_factor(self._factor_pool, key)
        raise KeyError(
            f"未知 tier 路径键名: {key!r}。"
            f"可用: {', '.join(sorted(FACTOR_TREE))}"
        )

    # ============================================================
    # 数据源路径
    # ============================================================

    @property
    def market_data_path(self) -> Path:
        """行情数据表路径。

        返回配置中 market_data.path 的值（用户直接提供表路径，如
        cos://bucket/.../StockDailyBar/ 或 /local/path/StockDailyBar/）。
        """
        return Path(self._require("market_data", "path"))

    @property
    def market_data_forward_return_field(self) -> str:
        """用于计算 forward return 的字段名（如 Vwap）。"""
        return self._require("market_data", "forward_return_field")

    @property
    def market_data_adjustment_field(self) -> str:
        """后复权调整使用的字段名（如 Factor）。"""
        return self._require("market_data", "adjustment_field")

    @property
    def industry_data_path(self) -> Path:
        """行业分类数据表路径。"""
        return Path(self._require("industry_data", "path"))

    @property
    def industry_source_field(self) -> str:
        """行业中用于识别分类口径的字段名（如 IndustrySource）。"""
        return self._require("industry_data", "source_field")

    @property
    def industry_code_field(self) -> str:
        """行业中表示行业代码的字段名（如 IndustryCode）。"""
        return self._require("industry_data", "code_field")

    @property
    def industry_standard(self) -> str:
        """行业分类口径（如 sws2021）。"""
        return self._require("industry_data", "standard")

    # ============================================================
    # DeepSeek 密钥
    # ============================================================

    @property
    def deepseek_api_key(self) -> str | None:
        """DeepSeek API 密钥。"""
        return self._cfg.get("deepseek", {}).get("api_key")

    # ============================================================
    # 日志
    # ============================================================

    @property
    def logging_directory(self) -> Path:
        """日志根目录：{local_tmp}/logs/。"""
        return _resolve_local(self.local_tmp, "pipeline_log").parent  # type: ignore[return-value]

    # ============================================================
    # 模块参数（暂保留，后续整理）
    # ============================================================

    def gateway_param(self, key: str) -> Any:
        return self._require("gateway", key)

    def purification_param(self, key: str) -> Any:
        return self._require("purification", key)

    def evaluation_param(self, key: str) -> Any:
        return self._require("evaluation", key)

    def service_param(self, key: str) -> Any:
        return self._require("service", key)

    # ============================================================
    # 原始配置
    # ============================================================

    def raw(self) -> dict:
        return dict(self._cfg)
