"""
Gateway 路由进程模块

监控 tier0/gateway_temp/，根据 afv.json 中的 Gateway 标签将因子目录路由到目标库。

路由规则:
    Gateway.Label == "Pass"       → tier1/gateway_pass_base/
    Gateway.Label == "Temp"       → tier1/gateway_temp_base/
    Gateway.Label == "Duplicated" → tier1/gateway_duplicated_base/
    Gateway.Label == "Rejected"   → tier4/anti_sample_base/gateway/
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .io_utils import IOManager

logger = logging.getLogger("gateway.router")

# 路由映射：Gateway.Label → 目标库目录名（相对于数据库根）
ROUTE_MAP = {
    "Pass":       ("tier1", "gateway_pass_base"),
    "Temp":       ("tier1", "gateway_temp_base"),
    "Duplicated": ("tier1", "gateway_duplicated_base"),
    "Rejected":   ("tier4", "anti_sample_base", "gateway"),
}

# 需要缓存报告的标签
CACHE_LABELS = {"Duplicated"}


class FactorRouter:
    """因子路由监控器。

    持续扫描 tier0/gateway_temp/，对每个因子目录：
    1. 读取 afv.json 获取 Gateway.Label
    2. 根据标签将目录路由到目标库
    3. 若为 Duplicated，确保缓存报告已存在
    """

    def __init__(self, temp_base: str | Path, db_root: str | Path):
        """
        Args:
            temp_base: tier0/gateway_temp/ 的路径。
            db_root: database/ 根目录的路径。
        """
        self.temp_base = Path(temp_base)
        self.db_root = Path(db_root)
        self._processed: set[str] = set()

    def scan_and_route(self, max_items: int = 0) -> list[dict]:
        """扫描临时库一次，路由所有待处理的因子。

        Args:
            max_items: 最大处理数量，0 表示全部。

        Returns:
            路由记录列表。
        """
        records = []
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

    def _route_one(self, factor_dir: Path) -> Optional[dict]:
        """路由单个因子目录。"""
        afv_path = factor_dir / "afv.json"
        if not afv_path.exists():
            logger.warning("因子目录缺少 afv.json，跳过: %s", factor_dir)
            return None

        try:
            afv = json.loads(afv_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("afv.json 解析失败 %s: %s", factor_dir, e)
            return None

        gateway = afv.get("Gateway", {})
        label = gateway.get("Label", "Rejected")
        reason = gateway.get("Reason", "")
        run_id = gateway.get("run_id", "")
        candidate_id = afv.get("candidate_id", factor_dir.name)

        # 确定路由目标
        route_key = label if label in ROUTE_MAP else "Rejected"
        route_parts = ROUTE_MAP[route_key]
        target_base = self.db_root.joinpath(*route_parts)

        # 执行路由
        try:
            target_dir = IOManager.route_factor(factor_dir, target_base)
            logger.info(
                "路由完成: candidate_id=%s label=%s → %s",
                candidate_id, label, target_dir,
            )
        except Exception as e:
            logger.error("路由失败 %s: %s", factor_dir, e)
            return None

        return {
            "candidate_id": candidate_id,
            "label": label,
            "reason": reason,
            "run_id": run_id,
            "source": str(factor_dir),
            "target": str(target_dir),
        }

    @staticmethod
    def run_loop(temp_base: str | Path, db_root: str | Path, interval: float = 2.0):
        """持续监控并路由（阻塞式循环）。

        Args:
            temp_base: tier0/gateway_temp/ 路径。
            db_root: database/ 根目录路径。
            interval: 扫描间隔（秒）。
        """
        router = FactorRouter(temp_base, db_root)
        logger.info("路由监控启动: temp=%s db=%s", temp_base, db_root)
        try:
            while True:
                records = router.scan_and_route()
                for r in records:
                    logger.info("  → %s (%s)", r["candidate_id"], r["label"])
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("路由监控停止")
