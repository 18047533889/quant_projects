#!/usr/bin/env python3
"""校验 manifest.json 字段名与 domain 是否符合 canonical 注册表。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"
CANON_PATH = Path(__file__).resolve().parents[1] / "enums" / "canonical_data_fields.json"

if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_canonical() -> dict[str, Any]:
    if not CANON_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {CANON_PATH}；请先运行: python3 factor_engine/scripts/build_canonical_fields.py"
        )
    return json.loads(CANON_PATH.read_text(encoding="utf-8"))


def _operator_names() -> set[str]:
    from api.operator_registry import build_dsl_allowlist

    return set(build_dsl_allowlist().keys())


def _column_entries(canonical: dict[str, Any], market: str) -> dict[str, dict[str, Any]]:
    """逻辑列名 → 列元数据（跨表合并，后写覆盖前写）。"""
    market_body = canonical.get("markets", {}).get(market, {})
    out: dict[str, dict[str, Any]] = {}
    for section in ("tables_local", "tables_massive_external"):
        tables = market_body.get(section, {}) or {}
        for _table, body in tables.items():
            for _phys, meta in (body.get("columns") or {}).items():
                names = {
                    meta.get("canonical"),
                    meta.get("physical"),
                    meta.get("formula_us_dsl"),
                    meta.get("lqtp_alias_expands_to"),
                }
                for alias in meta.get("lqtp_aliases") or []:
                    names.add(alias)
                for name in names:
                    if name:
                        out[str(name)] = meta
    return out


def _forbidden_columns(canonical: dict[str, Any], market: str) -> set[str]:
    forbidden: set[str] = set()
    for meta in _column_entries(canonical, market).values():
        if meta.get("formula_usage") == "forbidden":
            for key in ("canonical", "physical", "formula_us_dsl"):
                if meta.get(key):
                    forbidden.add(str(meta[key]))
    return forbidden


def _referenced_columns_from_formula(formula: str) -> set[str]:
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer

    expr = parse_expr(formula)
    return set(Analyzer().lower(expr).referenced_columns)


def _validate_manifest(path: Path, canonical: dict[str, Any], operators: set[str]) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: JSON 解析失败: {exc}"]

    market = str(data.get("market") or "").strip()
    if not market:
        errors.append(f"{path}: 缺少 market")
        return errors

    formula = str(data.get("formula") or "").strip()
    if not formula:
        errors.append(f"{path}: 缺少 formula")
        return errors

    domain_root = str(data.get("domain_root") or "price_volume").strip()
    domain_scopes = (
        canonical.get("domain_scopes", {})
        .get("markets", {})
        .get(market, {})
        .get(domain_root, {})
    )
    primary_tables = set(data.get("mining_scope", {}).get("primary_tables") or [])
    expected_primary = set(domain_scopes.get("primary_tables") or [])
    if expected_primary and primary_tables - expected_primary:
        extra = sorted(primary_tables - expected_primary)
        errors.append(
            f"{path}: primary_tables {extra} 不属于 domain_root={domain_root!r}"
        )

    col_index = _column_entries(canonical, market)
    forbidden = _forbidden_columns(canonical, market)

    try:
        refs = _referenced_columns_from_formula(formula)
    except Exception as exc:
        errors.append(f"{path}: 公式解析失败: {exc}")
        return errors

    for col in sorted(refs):
        if col in operators:
            errors.append(f"{path}: '{col}' 是算子名，不能作为数据列引用")
            continue
        if col in forbidden:
            errors.append(f"{path}: 列 '{col}' 标记为 forbidden，不得出现在公式中")
            continue
        meta = col_index.get(col)
        if meta is None:
            errors.append(f"{path}: 列 '{col}' 未在 canonical_data_fields.json 注册")
            continue
        usage = meta.get("formula_usage")
        if usage == "auxiliary_only":
            errors.append(f"{path}: 列 '{col}' 仅 auxiliary_tables，不能作主信号")
        elif usage == "filter_only":
            errors.append(f"{path}: 列 '{col}' 仅 filter，不能作主信号")
        signal_domain = meta.get("signal_domain")
        if signal_domain and signal_domain != domain_root and usage == "primary_signal":
            errors.append(
                f"{path}: 列 '{col}' signal_domain={signal_domain!r} "
                f"与 campaign domain_root={domain_root!r} 不一致"
            )
    return errors


def _iter_manifests(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"路径不存在: {target}")
    pattern = "**/manifest.json" if recursive else "manifest.json"
    return sorted(target.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 manifest 字段与 domain")
    parser.add_argument("path", type=Path, help="manifest.json 或 campaign 目录")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="扫描目录下全部 manifest.json",
    )
    args = parser.parse_args()

    canonical = _load_canonical()
    operators = _operator_names()
    manifests = _iter_manifests(args.path, args.recursive)
    if not manifests:
        print(f"未找到 manifest: {args.path}", file=sys.stderr)
        return 2

    all_errors: list[str] = []
    for mf in manifests:
        all_errors.extend(_validate_manifest(mf, canonical, operators))

    if all_errors:
        print("FAIL:", file=sys.stderr)
        for line in all_errors:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"OK: {len(manifests)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
