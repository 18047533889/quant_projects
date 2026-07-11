"""
Purification 路由模块

监控 tier0/purification_temp，将因子路由到 tier1/purification_pure_factor_base。
路由时保持目录名不变（factor_id）。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("purification.router")


class PurificationRouter:
    """Purification 路由监控器。"""

    def __init__(self, temp_base: str | Path, output_base: str | Path):
        """
        Args:
            temp_base: tier0/purification_temp 路径。
            output_base: tier1/purification_pure_factor_base 路径。
        """
        self.temp_base = Path(temp_base)
        self.output_base = Path(output_base)
        self._processed: set[str] = set()

    def scan_and_route(self, max_items: int = 0) -> list[dict]:
        """扫描纯化临时库，路由所有待处理的因子。

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
            logger.warning("纯化目录缺少 afv.json，跳过: %s", temp_dir)
            return None

        try:
            afv = json.loads(afv_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("afv.json 解析失败 %s: %s", temp_dir, e)
            return None

        factor_id = afv.get("factor_id", temp_dir.name)
        purification = afv.get("Purification", {})
        label = purification.get("Label", "Pure")
        candidate_id = afv.get("candidate_id", temp_dir.name)

        # 目标目录：pure_factor_base / {factor_id}
        target = self.output_base / factor_id
        target.mkdir(parents=True, exist_ok=True)

        # 直接转移整个目录（原子操作，避免中间库残留）
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(temp_dir), str(target))

        logger.info("路由: %s → %s (Label=%s)", candidate_id, target, label)
        return {
            "candidate_id": candidate_id,
            "factor_id": factor_id,
            "label": label,
            "target": str(target),
        }
