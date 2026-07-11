"""
Evaluation 路由模块

监控 tier0/evaluation_temp，根据 Evaluation 段中的 route_recommendation
将因子路由到对应的目标 tier 目录。

路由映射:
    tier3a_core           → database/tier3/3a_core_production_base/
    tier3b_satellite      → database/tier3/3b_satellite_production_base/
    tier3c_feature        → database/tier3/3c_feature_matierial_base/
    tier3d_optimized_reserve → database/tier3/3d_operation_storage_base/
    tier2_incubator       → database/tier2/2_fix_base/
    tier2x_optimization_factory → database/tier2/2x_llm_mutation_base/
    tier4_archive         → database/tier4/anti_sample_base/
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import EvaluationConfig

logger = logging.getLogger("evaluation.router")


class EvaluationRouter:
    """Evaluation 路由监控器。"""

    def __init__(
        self,
        temp_base: str | Path,
        config: EvaluationConfig | None = None,
    ):
        """
        Args:
            temp_base: tier0/evaluation_temp 路径。
            config: EvaluationConfig 实例。默认自动加载。
        """
        self.temp_base = Path(temp_base)
        self.config = config or EvaluationConfig()
        self._processed: set[str] = set()

    def scan_and_route(self, max_items: int = 0) -> list[dict]:
        """扫描评估临时库，根据路由推荐分发到对应 tier。

        Args:
            max_items: 最大处理数量，0 表示全部。

        Returns:
            路由记录列表。
        """
        records = []
        if not self.temp_base.exists():
            return records

        for item in sorted(self.temp_base.iterdir()):
            if not item.is_dir():
                continue
            if item.name in self._processed:
                continue
            if max_items and len(records) >= max_items:
                break

            record = self._route_one(item)
            if record:
                records.append(record)
                self._processed.add(item.name)

        return records

    def _route_one(self, temp_dir: Path) -> Optional[dict]:
        """路由单个因子目录。"""
        afv_path = temp_dir / "afv.json"
        if not afv_path.exists():
            logger.warning("评估目录缺少 afv.json，跳过: %s", temp_dir)
            return None

        try:
            afv = json.loads(afv_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("afv.json 解析失败 %s: %s", temp_dir, e)
            return None

        factor_id = afv.get("factor_id", temp_dir.name)
        evaluation = afv.get("Evaluation", {})
        label = evaluation.get("Label", "Evaluated")
        candidate_id = afv.get("candidate_id", temp_dir.name)

        # 获取路由推荐
        route_recommendation = evaluation.get("route_recommendation", "")
        if not route_recommendation:
            logger.warning("因子 %s 缺少 route_recommendation，跳过路由", factor_id)
            return None

        # 解析目标目录
        target_dir = self.config.target_dir_for_recommendation(route_recommendation)
        if target_dir is None:
            logger.warning("因子 %s route_recommendation=%s 无对应目标目录，跳过",
                           factor_id, route_recommendation)
            return None

        # 目标：target_dir / factor_id
        target = target_dir / factor_id
        target.mkdir(parents=True, exist_ok=True)

        # 直接转移整个目录（原子操作，避免中间库残留）
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(temp_dir), str(target))

        logger.info("路由: %s → %s (route=%s)", candidate_id, target, route_recommendation)
        return {
            "candidate_id": candidate_id,
            "factor_id": factor_id,
            "label": label,
            "route_recommendation": route_recommendation,
            "target": str(target),
        }
