"""
Assetization 路由模块

监控 tier0/assetization_temp，将因子路由到 tier1/assetization_raw_factor_base。
路由时将目录名从 candidate_id 改为 factor_id（从 afv.json 读取）。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetization.router")


class AssetizationRouter:
    """Assetization 路由监控器。"""

    def __init__(self, temp_base: str | Path, output_base: str | Path):
        """
        Args:
            temp_base: tier0/assetization_temp 路径。
            output_base: tier1/assetization_raw_factor_base 路径。
        """
        self.temp_base = Path(temp_base)
        self.output_base = Path(output_base)
        self._processed: set[str] = set()

    def scan_and_route(self, max_items: int = 0) -> list[dict]:
        """扫描资产化临时库，路由所有待处理的因子。

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
            logger.warning("因子目录缺少 afv.json，跳过: %s", temp_dir)
            return None

        try:
            afv = json.loads(afv_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("afv.json 解析失败 %s: %s", temp_dir, e)
            return None

        factor_id = afv.get("factor_id", temp_dir.name)
        assetization = afv.get("Assetization", {})
        label = assetization.get("Label", "Materialized")

        # 目标目录使用 factor_id
        target_dir = self.output_base / factor_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # 直接转移整个目录
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(temp_dir), str(target_dir))

        logger.info("路由: %s → %s (%s)", temp_dir.name, target_dir, label)
        return {
            "candidate_id": temp_dir.name,
            "factor_id": factor_id,
            "label": label,
            "target": str(target_dir),
        }
