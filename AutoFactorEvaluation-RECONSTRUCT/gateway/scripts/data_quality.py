"""
数据质量检测模块 - 集成版

包含:
- DistributionDriftDetector: 分布漂移检测
- LiquidityAnomalyDetector: 流动性异常检测
- ClockJitterValidator: 时钟抖动验证
- DataQualityChecker: 数据质量检测器主类
- DataQualityRunner: 数据质量检测运行器
"""

"""
数据质量检测器
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy import stats


class DistributionDriftDetector:
    """分布漂移检测器 (KS检验 + Wasserstein距离)"""

    def __init__(self, config: Dict[str, Any]):
        self.ks_threshold = config.get("ks_p_value_threshold", 0.05)
        self.wasserstein_threshold = config.get("wasserstein_threshold", 1.0)

    def detect(self, current: pd.Series, historical: pd.Series) -> Tuple[float, float, float]:
        drift_score = 0.0

        # KS检验
        _, ks_p_value = stats.ks_2samp(current.dropna(), historical.dropna())

        # Wasserstein距离
        wasserstein_dist = self._wasserstein_distance(current, historical)

        if ks_p_value < self.ks_threshold:
            drift_score += 0.5
        if wasserstein_dist > self.wasserstein_threshold:
            drift_score += 0.5

        return ks_p_value, wasserstein_dist, drift_score

    def _wasserstein_distance(self, p: pd.Series, q: pd.Series) -> float:
        p_vals, q_vals = p.dropna().values, q.dropna().values
        if len(p_vals) == 0 or len(q_vals) == 0:
            return 0.0
        quantiles = np.linspace(0, 1, 100)
        return np.mean(np.abs(np.quantile(p_vals, quantiles) - np.quantile(q_vals, quantiles)))


class LiquidityAnomalyDetector:
    """流动性异常检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.spread_threshold = config.get("spread_spike_threshold", 2.0)
        self.lob_threshold = config.get("lob_gap_threshold", 0.05)

    def detect(self, data: pd.DataFrame) -> Tuple[int, int, float]:
        spread_spikes = 0
        lob_gaps = 0
        anomaly_score = 0.0

        # Spread突变检测
        if 'spread' in data.columns:
            spread = data['spread']
            z_scores = np.abs((spread - spread.mean()) / (spread.std() + 1e-8))
            spread_spikes = int((z_scores > self.spread_threshold).sum())
            if spread_spikes > 0:
                anomaly_score += min(0.5, spread_spikes / 100)

        # LOB缺失检测
        if 'bid_volume' in data.columns and 'ask_volume' in data.columns:
            zero_ratio = ((data['bid_volume'] + data['ask_volume']) == 0).mean()
            if zero_ratio > self.lob_threshold:
                lob_gaps = int(zero_ratio * len(data))
                anomaly_score += min(0.3, zero_ratio)

        return spread_spikes, lob_gaps, anomaly_score


class ClockJitterValidator:
    """时钟跳动验证器"""

    def __init__(self, config: Dict[str, Any]):
        pass

    def validate(self, timestamps: pd.Series) -> Tuple[int, int, float]:
        if len(timestamps) < 2 or 'timestamp' not in timestamps.name:
            return 0, 0, 0.0

        diff = pd.to_datetime(timestamps).diff()
        out_of_order = int((diff < pd.Timedelta(0)).sum())
        validity_score = min(0.7, out_of_order / len(timestamps)) if out_of_order > 0 else 0.0

        return out_of_order, 0, validity_score

"""
数据质量检测主逻辑
"""

import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from .data_quality import DistributionDriftDetector, LiquidityAnomalyDetector, ClockJitterValidator


@dataclass
class QualityReport:
    overall_score: float
    is_passed: bool
    label: str  # "Pass" or "Temp"
    reason: str
    ks_p_value: float = 1.0
    wasserstein_distance: float = 0.0
    distribution_drift_score: float = 0.0
    spread_spike_count: int = 0
    lob_gap_count: int = 0
    liquidity_anomaly_score: float = 0.0
    timestamp_out_of_order: int = 0
    overall_quality_score: float = 0.0


class DataQualityChecker:
    """数据质量检测器主类"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("data_quality")

        self.drift_detector = DistributionDriftDetector(config.get("drift", {}))
        self.liquidity_detector = LiquidityAnomalyDetector(config.get("liquidity", {}))
        self.clock_validator = ClockJitterValidator(config.get("clock", {}))

        self.pass_threshold = config.get("pass_threshold", 0.7)

    def check(self, market_data_path: str) -> QualityReport:
        """执行数据质量检测"""
        self.logger.info(f"Loading market data from: {market_data_path}")

        # 加载数据
        data = self._load_data(market_data_path)
        if data is None:
            return QualityReport(
                overall_score=0.0, is_passed=False, label="Temp",
                reason="Failed to load market data"
            )

        # 执行各项检测
        ks_pv, wass_dist, drift_score = self.drift_detector.detect(
            data.get('close', data.iloc[:, 0]),
            data.get('close', data.iloc[:, 0]).iloc[:len(data) // 2]
        )
        spread_spikes, lob_gaps, liq_score = self.liquidity_detector.detect(data)
        out_of_order, _, clock_score = self.clock_validator.validate(data.get('timestamp', pd.Series()))

        # 综合评分
        overall = self._calc_score(drift_score, liq_score, clock_score)
        is_passed = overall >= self.pass_threshold
        label = "Pass" if is_passed else "Temp"
        reason = self._build_reason(drift_score, liq_score, clock_score, ks_pv, spread_spikes)

        return QualityReport(
            overall_score=overall,
            is_passed=is_passed,
            label=label,
            reason=reason,
            ks_p_value=ks_pv,
            wasserstein_distance=wass_dist,
            distribution_drift_score=drift_score,
            spread_spike_count=spread_spikes,
            lob_gap_count=lob_gaps,
            liquidity_anomaly_score=liq_score,
            timestamp_out_of_order=out_of_order,
            overall_quality_score=overall,
        )

    def _load_data(self, path: str) -> Optional[pd.DataFrame]:
        try:
            p = Path(path)
            if p.is_file() and p.suffix == '.parquet':
                return pd.read_parquet(p)
            elif p.is_dir():
                files = list(p.glob("*.parquet"))
                if files:
                    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            return None
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return None

    def _calc_score(self, drift: float, liq: float, clock: float) -> float:
        weights = {'drift': 0.4, 'liquidity': 0.3, 'clock': 0.3}
        drift_score = 1 - min(1.0, drift)
        liq_score = 1 - min(1.0, liq)
        clock_score = 1 - min(1.0, clock)
        return (weights['drift'] * drift_score +
                weights['liquidity'] * liq_score +
                weights['clock'] * clock_score)

    def _build_reason(self, drift: float, liq: float, clock: float,
                      ks_pv: float, spikes: int) -> str:
        reasons = []
        if drift > 0.3:
            reasons.append(f"distribution drift (KS p={ks_pv:.3f})")
        if liq > 0.2:
            reasons.append(f"liquidity anomaly ({spikes} spikes)")
        if clock > 0.1:
            reasons.append("clock jitter detected")
        return "; ".join(reasons) if reasons else "Quality score below threshold"

"""
数据质量检测运行器 - 独立组件
"""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import yaml


class DataQualityRunner:
    """
    数据质量检测运行器

    功能:
    1. 读取网关输出的 candidate.json
    2. 执行数据质量检测
    3. 追加 DataQuality 段落
    4. 写回原文件或输出到新位置
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.checker = DataQualityChecker(self.config)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """加载配置"""
        default = {
            "drift": {"ks_p_value_threshold": 0.05, "wasserstein_threshold": 1.0},
            "liquidity": {"spread_spike_threshold": 2.0, "lob_gap_threshold": 0.05},
            "clock": {},
            "pass_threshold": 0.7,
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_cfg = yaml.safe_load(f)
                    if user_cfg:
                        default.update(user_cfg)
            except Exception:
                pass

        return default

    def check(self, market_data_path: str) -> "QualityReport":
        """
        快捷方法：仅执行数据质量检测，返回 QualityReport。
        供 GatewayCore 集成使用（无需操作 candidate.json）。

        Args:
            market_data_path: 市场数据路径

        Returns:
            QualityReport 对象
        """
        return self.checker.check(market_data_path)

    def process(
            self,
            candidate_json_path: Path,
            market_data_path: str,
            output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        处理单个候选因子：读取 candidate.json → 数据质量检测 → 追加 DataQuality 段

        Args:
            candidate_json_path: candidate.json 路径
            market_data_path: 市场数据路径
            output_path: 输出路径（默认覆盖原文件）

        Returns:
            更新后的 candidate.json 内容
        """
        # 1. 读取 candidate.json
        with open(candidate_json_path, 'r', encoding='utf-8') as f:
            candidate = json.load(f)

        # 2. 执行数据质量检测
        report = self.checker.check(market_data_path)

        # 3. 构建 DataQuality 段落
        run_id = f"dq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        dq_segment = {
            "Label": report.label,
            "Reason": report.reason,
            "run_id": run_id,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "distribution_drift_score": report.distribution_drift_score,
            "ks_p_value": report.ks_p_value,
            "wasserstein_distance": report.wasserstein_distance,
            "liquidity_anomaly_score": report.liquidity_anomaly_score,
            "spread_spike_count": report.spread_spike_count,
            "timestamp_out_of_order": report.timestamp_out_of_order,
            "overall_quality_score": report.overall_score,
        }

        # 4. 追加到 candidate.json
        candidate["DataQuality"] = dq_segment

        # 5. 写回
        out_path = output_path or candidate_json_path
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(candidate, f, indent=2, ensure_ascii=False)

        return candidate

    def process_batch(
            self,
            gateway_pass_base: Path,
            market_data_path: str,
            duplicate_base: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        批量处理所有网关通过的因子

        Args:
            gateway_pass_base: 网关输出目录
            market_data_path: 市场数据路径
            duplicate_base: 重复因子输出目录（数据检测 Temp 的因子移到这里）

        Returns:
            处理统计
        """
        stats = {"total": 0, "pass": 0, "temp": 0, "errors": 0}

        # 遍历所有候选目录
        for candidate_dir in Path(gateway_pass_base).iterdir():
            if not candidate_dir.is_dir():
                continue

            candidate_json = candidate_dir / "candidate.json"
            if not candidate_json.exists():
                continue

            stats["total"] += 1

            try:
                # 执行检测
                result = self.process(candidate_json, market_data_path)

                if result["DataQuality"]["Label"] == "Pass":
                    stats["pass"] += 1
                else:
                    stats["temp"] += 1
                    # 如果是 Temp，移动到 duplicate_base
                    if duplicate_base:
                        dest_dir = duplicate_base / candidate_dir.name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(candidate_json), str(dest_dir / "candidate.json"))
                        shutil.rmtree(candidate_dir)

            except Exception as e:
                stats["errors"] += 1

        return stats


def run_data_quality(
        candidate_json_path: str,
        market_data_path: str,
        output_path: Optional[str] = None,
        config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：运行数据质量检测

    Args:
        candidate_json_path: candidate.json 文件路径
        market_data_path: 市场数据路径（Parquet）
        output_path: 输出路径（可选）
        config_path: 配置文件路径（可选）

    Returns:
        更新后的 candidate 字典
    """
    config_path_obj = Path(config_path) if config_path else None
    runner = DataQualityRunner(config_path_obj)
    return runner.process(
        Path(candidate_json_path),
        market_data_path,
        Path(output_path) if output_path else None
    )
