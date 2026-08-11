# -*- coding: utf-8 -*-
"""R40 #25 —— 参数域认证证据**全维度完备性**审计。

背景
----
:class:`runtime.parameter_domain_store.CertificationKey` 已把 8 轴身份编码
（canonical / semantic_version / backend / execution_variant / source_context /
parameter_point / dtype / grain），但 oracle 生成的证据**全维度完备性**从未被
断言——一个算子可能只认证了 ``pandas_numpy/reference/daily/memory/float64``
一个组合，却在 production 被当作"已认证"。

本脚本提供可复用的 ``check_*`` 函数（可被 ``scripts/audit_r40_hard_gates.py``
与 tests 复用）：

    - :func:`check_evidence_axis_completeness`：每条认证点的 8 轴身份必须完整
      （parameter_point 允许空——零参算子）；
    - :func:`check_evidence_cross_product_completeness`：per-canonical 必须覆盖
      canonical × parameter-domain × backend × execution_variant ×
      grain × source_context 的全部**必需轴**组合（缺一即 incomplete）。

必需轴默认值：``backend={pandas_numpy, polars, duckdb_sql}``、
``execution_variant={reference}``、``grain={daily}``、
``source_context={production, research, memory}``、``dtype={float64}``。
这些默认值偏严格——production SQL 下推（R40 #61）要求 canonical 同时白名单 +
参数域认证，因此 duckdb_sql 轴缺失会被本审计标记。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FE_ROOT = Path(__file__).resolve().parents[1]

#: 必需轴的默认覆盖要求（可被调用方覆盖）。
REQUIRED_AXES: dict[str, frozenset[str]] = {
    "backend": frozenset({"pandas_numpy", "polars", "duckdb_sql"}),
    "execution_variant": frozenset({"reference"}),
    "grain": frozenset({"daily"}),
    "source_context": frozenset({"production", "research", "memory"}),
    "dtype": frozenset({"float64"}),
}

#: CertificationKey 的 8 轴（parameter_point 允许空——零参算子）。
_IDENTITY_AXES = (
    "canonical",
    "semantic_version",
    "backend",
    "execution_variant",
    "source_context",
    "dtype",
    "grain",
)


def _axis_of(point: Mapping[str, Any], axis: str) -> str:
    v = point.get(axis)
    return "" if v is None else str(v)


def check_evidence_axis_completeness(
    points: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """每条认证点的 7 个标量轴（除 parameter_point）必须非空。

    返回 ``{"incomplete": [(canonical, axis)], "complete": bool}``。
    ``parameter_point`` 允许空（零参算子合法）；其余任一轴缺失/空 → incomplete。
    """
    incomplete: list[tuple[str, str]] = []
    for point in points:
        canon = _axis_of(point, "canonical")
        for axis in _IDENTITY_AXES:
            if not _axis_of(point, axis):
                incomplete.append((canon or "<anon>", axis))
    return {"incomplete": incomplete, "complete": not incomplete}


def required_axis_cross_product(
    *,
    required: Mapping[str, frozenset[str]] | None = None,
) -> tuple[frozenset[tuple[str, ...]], tuple[str, ...]]:
    """构造必需轴的笛卡尔积（默认全轴）。返回 ``(combos, axis_names)``。"""
    req = required or REQUIRED_AXES
    axis_names = tuple(req.keys())
    import itertools

    combos: set[tuple[str, ...]] = set()
    for combo in itertools.product(*(sorted(req[a]) for a in axis_names)):
        combos.add(tuple(combo))
    return frozenset(combos), axis_names


def check_evidence_cross_product_completeness(
    points: Iterable[Mapping[str, Any]],
    *,
    required: Mapping[str, frozenset[str]] | None = None,
    required_canonicals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """per-canonical 覆盖 canonical × parameter-domain × backend × variant ×
    grain × source context 的全部必需轴组合。

    对每个 canonical（passed 的认证点）计算已覆盖的轴组合集合；与必需笛卡尔积
    比对，缺失组合记为 incomplete。返回：:

        {
          "canonicals": {canon: {"covered": n, "missing": [...], "ok": bool}},
          "missing_total": int,
          "all_complete": bool,
          "required_cross_product": n,
        }

    ``required_canonicals`` 显式给定时，缺失 canonical（完全无认证点）也会被
    标记为 incomplete（缺全部必需组合）。
    """
    req = required or REQUIRED_AXES
    combos, axis_names = required_axis_cross_product(required=req)
    covered: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for point in points:
        if not point.get("passed", True):
            continue
        canon = _axis_of(point, "canonical")
        if not canon:
            continue
        combo = tuple(_axis_of(point, a) for a in axis_names)
        covered[canon].add(combo)
    canonicals = required_canonicals or sorted(covered.keys())
    per: dict[str, Any] = {}
    missing_total = 0
    for canon in canonicals:
        got = covered.get(canon, set())
        missing = sorted(combos - got)
        per[canon] = {
            "covered": len(got),
            "missing": list(missing),
            "ok": not missing,
        }
        missing_total += len(missing)
    return {
        "canonicals": per,
        "missing_total": missing_total,
        "all_complete": missing_total == 0,
        "required_cross_product": len(combos),
    }


def load_store_points(path: str | Path) -> list[dict[str, Any]]:
    """从 R37 参数域证据 JSON 读认证点列表。"""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    points = data.get("certified_points") or data.get("points") or []
    return [dict(x) for x in points if isinstance(x, dict)]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    path = (
        Path(argv[0])
        if argv
        else FE_ROOT / "docs" / "evidence" / "r37" / "R37_PARAMETER_DOMAIN_STORE.json"
    )
    points = load_store_points(path)
    axis = check_evidence_axis_completeness(points)
    cross = check_evidence_cross_product_completeness(points)
    print(f"evidence={path}")
    print(f"points={len(points)}")
    print(f"axis_complete={axis['complete']} incomplete={axis['incomplete'][:10]}")
    print(f"cross_product_all_complete={cross['all_complete']} "
          f"missing={cross['missing_total']} required={cross['required_cross_product']}")
    for canon, rec in sorted(cross["canonicals"].items()):
        if not rec["ok"]:
            print(f"  MISSING {canon}: covered={rec['covered']} "
                  f"missing={rec['missing'][:6]} total_missing={len(rec['missing'])}")
    return 0 if (axis["complete"] and cross["all_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
