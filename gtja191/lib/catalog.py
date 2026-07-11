"""GTJA-191 catalog 加载：对外默认 185 条可投递因子。"""
from __future__ import annotations

import json
from typing import Any

from lib.dsl_normalize import normalize_operator_names
from lib.paths import PACKAGE_ROOT

CATALOG_PATH = PACKAGE_ROOT / "dsl" / "gtja191_dsl_catalog.json"
DELIVERABLE_COUNT = 185
SOURCE_FORMULA_COUNT = 191

# 经源公式逐项核对的转换修正。
_AUDITED_FORMULA_OVERRIDES: dict[str, str] = {
    # REGBETA(..., SEQUENCE(n)) 是“对时间位置回归”，不是对 lag(close) 回归。
    "gtja191_alpha_021": "ts_time_slope(ts_mean(close, 6), 6)",
    "gtja191_alpha_116": "ts_time_slope(close, 20)",
    "gtja191_alpha_147": "ts_time_slope(ts_mean(close, 12), 12)",
    # 源公式为 rank(corr(sum(mid,19), sum(mean(volume,40),19),13)) 的五次方；
    # 历史转换误把 13 写成 power(13)，并把 5 当成 corr 窗口。
    "gtja191_alpha_056": (
        "rank(open - ts_min(open, 12)) < "
        "rank(power(rank(ts_corr(ts_sum((high + low) / 2, 19), "
        "ts_sum(ts_mean(volume, 40), 19), 13)), 5))"
    ),
}


def _normalize_item(name: str, item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    formula = _AUDITED_FORMULA_OVERRIDES.get(name, str(out.get("dsl_formula", "")))
    out["dsl_formula"] = normalize_operator_names(formula)
    return out


def load_catalog() -> dict[str, dict[str, Any]]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GTJA catalog root must be a mapping")
    if len(raw) != SOURCE_FORMULA_COUNT:
        raise ValueError(
            f"expected {SOURCE_FORMULA_COUNT} source formulas, got {len(raw)}"
        )
    return {str(name): _normalize_item(str(name), dict(item)) for name, item in raw.items()}


def is_deliverable(item: dict[str, Any]) -> bool:
    return bool(item.get("valid")) and not bool(item.get("delivery_excluded"))


def deliverable_catalog(
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    catalog = catalog or load_catalog()
    return {name: item for name, item in catalog.items() if is_deliverable(item)}


def deliverable_names(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    names = sorted(deliverable_catalog(catalog).keys())
    if len(names) != DELIVERABLE_COUNT:
        raise ValueError(f"expected {DELIVERABLE_COUNT} deliverable factors, got {len(names)}")
    return names
