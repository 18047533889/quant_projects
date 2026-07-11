"""
数据质量检测运行器 - 独立组件
"""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import yaml

from .checker import DataQualityChecker


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