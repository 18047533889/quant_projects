"""4.1 时序模块运行配置与默认路径。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：声明模块输入/输出路径契约与计算参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .validation import Severity, ValidationBuffer

ALLOWED_WEIGHTING = {"equal", "cap_weighted"}
ALLOWED_RANK_SCOPE = {"global", "sector_relative"}
ALLOWED_TIMEZONE_POLICY = {"reject_mixed", "allow_naive"}
ALLOWED_ENVIRONMENTS = {"production", "sample", "mock"}


@dataclass
class TimeseriesPaths:
    """单次时序评估运行的路径契约。

    所有路径必须从外部传入，无代码级默认值。缺失即报错。
    """

    output_dir: Path
    cache_dir: Path
    log_dir: Path

    def resolved_output_dir(self) -> Path:
        """返回时序评估产物根目录。"""
        return self.output_dir

    def resolved_cache_dir(self) -> Path:
        """返回 forward return 缓存目录。"""
        return self.cache_dir

    def resolved_log_dir(self) -> Path:
        """返回运行日志目录。"""
        return self.log_dir


@dataclass
class TimeseriesConfig:
    """4.1 时序评估的数值与行为配置。"""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 20])
    min_assets: int = 30
    n_quantiles: int = 5
    factor_direction: int = 1
    weighting: str = "equal"  # 可选：equal、cap_weighted
    rank_scope: str = "global"  # 可选：global、sector_relative
    cost_bps: float = 0.0
    eps: float = 1e-8
    timezone_policy: str = "reject_mixed"  # 可选：reject_mixed、allow_naive
    environment: str = "production"  # 可选：production、sample、mock


def validate_timeseries_config(config: TimeseriesConfig, vb: ValidationBuffer) -> bool:
    """校验运行配置，发现非法取值时 fail-fast（FID §2.7）。

    说明：
        - 对显式枚举参数做白名单校验；
        - 对数值参数做最小约束（含 `cost_bps >= 0`）；
        - `horizons` 在校验通过后会规范化为去重升序正整数列表。
    """
    ok = True

    if not config.horizons:
        vb.add(Severity.ERROR, "config_horizons_empty", "horizons must not be empty")
        ok = False
    else:
        try:
            normalized_h = sorted({int(h) for h in config.horizons})
        except Exception:
            vb.add(Severity.ERROR, "config_horizons_invalid", "horizons must be integer list")
            ok = False
            normalized_h = []
        if normalized_h and any(h <= 0 for h in normalized_h):
            vb.add(Severity.ERROR, "config_horizons_non_positive", "horizons must be positive integers")
            ok = False
        if normalized_h:
            config.horizons = normalized_h

    if int(config.min_assets) <= 0:
        vb.add(Severity.ERROR, "config_min_assets_invalid", "min_assets must be > 0")
        ok = False
    if int(config.n_quantiles) <= 0:
        vb.add(Severity.ERROR, "config_n_quantiles_invalid", "n_quantiles must be > 0")
        ok = False
    if float(config.eps) <= 0:
        vb.add(Severity.ERROR, "config_eps_invalid", "eps must be > 0")
        ok = False
    if float(config.cost_bps) < 0:
        vb.add(Severity.ERROR, "config_cost_bps_invalid", "cost_bps must be >= 0")
        ok = False
    if int(config.factor_direction) not in {-1, 1}:
        vb.add(Severity.ERROR, "config_factor_direction_invalid", "factor_direction must be -1 or 1")
        ok = False

    if config.weighting not in ALLOWED_WEIGHTING:
        vb.add(
            Severity.ERROR,
            "config_weighting_invalid",
            "weighting must be one of equal/cap_weighted",
            details={"weighting": config.weighting, "allowed": sorted(ALLOWED_WEIGHTING)},
        )
        ok = False
    if config.rank_scope not in ALLOWED_RANK_SCOPE:
        vb.add(
            Severity.ERROR,
            "config_rank_scope_invalid",
            "rank_scope must be one of global/sector_relative",
            details={"rank_scope": config.rank_scope, "allowed": sorted(ALLOWED_RANK_SCOPE)},
        )
        ok = False
    if config.timezone_policy not in ALLOWED_TIMEZONE_POLICY:
        vb.add(
            Severity.ERROR,
            "config_timezone_policy_invalid",
            "timezone_policy must be one of reject_mixed/allow_naive",
            details={"timezone_policy": config.timezone_policy, "allowed": sorted(ALLOWED_TIMEZONE_POLICY)},
        )
        ok = False
    if config.environment not in ALLOWED_ENVIRONMENTS:
        vb.add(
            Severity.ERROR,
            "config_environment_invalid",
            "environment must be one of production/sample/mock",
            details={"environment": config.environment, "allowed": sorted(ALLOWED_ENVIRONMENTS)},
        )
        ok = False

    return ok
