"""
Purification 配置加载模块

从 purification_config.yaml 加载配置，提供统一访问接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "purification" / "configs"


class PurificationConfig:
    """Purification 模块配置。"""

    def __init__(self, config_dir: str | Path | None = None):
        if config_dir is None:
            # 默认配置目录：项目级共享配置
            _candidate = _PROJECT_ROOT / "all_configs" / "auto_factor_evaluation"
            if not _candidate.exists():
                _candidate = _PROJECT_ROOT.parent / "all_configs" / "auto_factor_evaluation"
            config_dir = _candidate
        else:
            config_dir = Path(config_dir)
        config_path = config_dir / "purification_config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Purification 配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

    @staticmethod
    def _resolve(p: str) -> Path:
        """路径解析：绝对路径直接使用，相对路径基于项目根。"""
        pp = Path(p)
        return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()

    @property
    def raw_factor_base(self) -> Path:
        """上游 RawFactor 基座路径。"""
        return self._resolve(self._raw["paths"]["raw_factor_base"])

    @property
    def temp_base(self) -> Path:
        """纯化临时路径（tier0）。"""
        return self._resolve(self._raw["paths"]["temp_base"])

    @property
    def pure_factor_base(self) -> Path:
        """纯化因子输出基座（tier1）。"""
        return self._resolve(self._raw["paths"]["pure_factor_base"])

    @property
    def industry_data_path(self) -> Path:
        """行业分类数据路径。"""
        return self._resolve(self._raw["industry_data"]["path"])

    @property
    def industry_standard(self) -> str:
        """行业分类口径。"""
        return self._raw["industry_data"]["standard"]

    @property
    def imputation_method(self) -> str:
        """缺失值填补方法。"""
        return self._raw["purification"]["imputation_method"]

    @property
    def max_delay(self) -> int:
        """时序填补最大延迟期数。"""
        return self._raw["purification"]["max_delay"]

    @property
    def decay_rate(self) -> float:
        """前向填充衰减因子。"""
        return self._raw["purification"]["decay_rate"]

    @property
    def winsorization_mad_multiplier(self) -> float:
        """去极值 MAD 乘数。"""
        return self._raw["purification"]["winsorization_mad_multiplier"]

    @property
    def existing_factor_library(self) -> dict[str, Path]:
        """现有因子库路径（正交化依赖）。"""
        lib = self._raw["paths"]["existing_factor_library"]
        return {k: self._resolve(v) for k, v in lib.items()}

    @property
    def logging_level(self) -> str:
        return self._raw["logging"]["level"]

    @property
    def logging_file(self) -> str:
        return str(self._resolve(self._raw["logging"]["file"]))
