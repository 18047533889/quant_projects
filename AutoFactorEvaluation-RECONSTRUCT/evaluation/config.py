"""
Evaluation 配置加载模块

从 evaluation_config.yaml 加载配置，提供统一访问接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "evaluation" / "configs"


class EvaluationConfig:
    """Evaluation 模块配置。"""

    # route_recommendation → tier 子目录名的映射
    ROUTE_MAP: dict[str, str] = {
        "tier3a_core": "3a_core_production_base",
        "tier3b_satellite": "3b_satellite_production_base",
        "tier3c_feature": "3c_feature_matierial_base",
        "tier3d_optimized_reserve": "3d_operation_storage_base",
        "tier2_incubator": "2_fix_base",
        "tier2x_optimization_factory": "2x_llm_mutation_base",
        "tier4_archive": "anti_sample_base",
    }

    def __init__(self, config_dir: str | Path | None = None):
        if config_dir is None:
            # 默认配置目录：项目级共享配置
            _candidate = _PROJECT_ROOT / "all_configs" / "auto_factor_evaluation"
            if not _candidate.exists():
                _candidate = _PROJECT_ROOT.parent / "all_configs" / "auto_factor_evaluation"
            config_dir = _candidate
        else:
            config_dir = Path(config_dir)
        config_path = config_dir / "evaluation_config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Evaluation 配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

    @staticmethod
    def _resolve(p: str) -> Path:
        """路径解析：绝对路径直接使用，相对路径基于项目根。"""
        pp = Path(p)
        return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()

    @property
    def pure_factor_base(self) -> Path:
        """上游 PureFactor 基座路径。"""
        return self._resolve(self._raw["paths"]["pure_factor_base"])

    @property
    def temp_base(self) -> Path:
        """评估临时路径（tier0）。"""
        return self._resolve(self._raw["paths"]["temp_base"])

    @property
    def market_data_path(self) -> Path:
        """行情数据路径（StockDailyBar 目录）。"""
        return self._resolve(self._raw["market_data_path"])

    @property
    def horizons(self) -> list[int]:
        """前向收益率计算口径（horizon 天数列表）。"""
        return self._raw["evaluation"]["horizons"]

    @property
    def min_assets(self) -> int:
        return self._raw["evaluation"]["min_assets"]

    @property
    def n_quantiles(self) -> int:
        return self._raw["evaluation"]["n_quantiles"]

    @property
    def factor_direction(self) -> int:
        return self._raw["evaluation"]["factor_direction"]

    @property
    def weighting(self) -> str:
        return self._raw["evaluation"]["weighting"]

    @property
    def cost_bps(self) -> float:
        return self._raw["evaluation"]["cost_bps"]

    @property
    def output_tiers(self) -> dict[str, Path]:
        """所有输出 tier 的路径映射。"""
        out = self._raw["paths"]["output"]
        return {k: self._resolve(v) for k, v in out.items()}

    def target_dir_for_recommendation(self, recommendation: str) -> Path | None:
        """根据 route_recommendation 获取目标目录路径。

        优先使用 config.yaml 中 output_tiers 的显式配置；
        无配置时由 ROUTE_MAP + project_root 推导。

        Args:
            recommendation: route_recommendation 值（如 "tier3a_core"）。

        Returns:
            目标目录的 Path，若映射不存在则返回 None。
        """
        # 优先从配置的 output_tiers 获取
        out = self._raw.get("paths", {}).get("output", {})
        if recommendation in out:
            return self._resolve(out[recommendation])

        # 回退：从 ROUTE_MAP + project_root 推导
        sub_dir = self.ROUTE_MAP.get(recommendation)
        if sub_dir is None:
            return None
        from config_manager import ConfigManager
        db_root = ConfigManager().database_root
        tier_num = recommendation[4]
        return (db_root / f"tier{tier_num}" / sub_dir).resolve()

    @property
    def logging_level(self) -> str:
        return self._raw["logging"]["level"]
