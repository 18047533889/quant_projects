"""GTJA-191 catalog 加载：对外默认 185 条可投递因子。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.paths import PACKAGE_ROOT

CATALOG_PATH = PACKAGE_ROOT / "dsl" / "gtja191_dsl_catalog.json"
DELIVERABLE_COUNT = 185
SOURCE_FORMULA_COUNT = 191


def load_catalog() -> dict[str, dict[str, Any]]:
    """加载完整 catalog（含内部 stub / benchmark 条目）。"""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def is_deliverable(item: dict[str, Any]) -> bool:
    return bool(item.get("valid")) and not bool(item.get("delivery_excluded"))


def deliverable_catalog(catalog: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """185 条可投递因子（factor_engine 落值 / manifest 默认集合）。"""
    catalog = catalog or load_catalog()
    return {name: item for name, item in catalog.items() if is_deliverable(item)}


def deliverable_names(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    names = sorted(deliverable_catalog(catalog).keys())
    if len(names) != DELIVERABLE_COUNT:
        raise ValueError(f"expected {DELIVERABLE_COUNT} deliverable factors, got {len(names)}")
    return names
