from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GTJA = ROOT / "gtja191"
AUTO = ROOT / "AutoFactorEvaluation-RECONSTRUCT"


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_pit_policies() -> None:
    path = ROOT / "factor_engine" / "cleaned_operators" / "operator_policy.py"
    replace_exact(
        path,
        '''    "maximum": {"scope": "elementwise", "pit_safe": True},
    "minimum": {"scope": "elementwise", "pit_safe": True},
''',
        '''    "maximum": {"scope": "elementwise", "pit_safe": True},
    "minimum": {"scope": "elementwise", "pit_safe": True},
    "flex_max": {"scope": "elementwise", "pit_safe": True},
    "flex_min": {"scope": "elementwise", "pit_safe": True},
''',
        "PIT policies for flex extrema",
    )


def patch_gtja_tests() -> None:
    path = GTJA / "tests" / "test_gtja191.py"
    replace_exact(
        path,
        '''        self.assertIn("ts_max((high + low + close) / 3 - close, 3)", dsl)
        self.assertIn("ts_min((high + low + close) / 3 - close, 3)", dsl)
''',
        '''        self.assertIn("ts_max(col('vwap') - close, 3)", dsl)
        self.assertIn("ts_min(col('vwap') - close, 3)", dsl)
        self.assertNotIn("(high + low + close) / 3", dsl)
''',
        "GTJA alpha007 VWAP assertions",
    )


def run_main(path: Path, argv: list[str] | None = None) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path), *(argv or [])]
        namespace = runpy.run_path(str(path))
        result = namespace["main"]()
        if result not in (None, 0):
            raise RuntimeError(f"{path} returned {result}")
    finally:
        sys.argv = old_argv


def rebuild_assets() -> None:
    run_main(GTJA / "scripts" / "convert_and_build_delivery.py")
    run_main(GTJA / "scripts" / "generate_materialize_configs.py")
    run_main(AUTO / "scripts" / "generate_gtja185_pack.py")


def verify_assets() -> None:
    sys.path.insert(0, str(GTJA))
    from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog

    catalog = deliverable_catalog()
    if len(catalog) != DELIVERABLE_COUNT or DELIVERABLE_COUNT != 185:
        raise RuntimeError(f"deliverable catalog count mismatch: {len(catalog)}")
    bad = []
    for name, item in catalog.items():
        source = str(item.get("source_formula") or "")
        formula = str(item["dsl_formula"])
        if "VWAP" in source.upper() and "col('vwap')" not in formula and 'col("vwap")' not in formula:
            bad.append(name)
    if bad:
        raise RuntimeError(f"VWAP semantics still stale: {bad[:10]}")

    raw_catalog = json.loads((GTJA / "dsl" / "gtja191_dsl_catalog.json").read_text(encoding="utf-8"))
    stale_catalog = [
        name
        for name, item in raw_catalog.items()
        if "VWAP" in str(item.get("source_formula") or "").upper()
        and "col('vwap')" not in str(item.get("dsl_formula") or "")
        and 'col("vwap")' not in str(item.get("dsl_formula") or "")
    ]
    if stale_catalog:
        raise RuntimeError(f"raw catalog VWAP semantics stale: {stale_catalog[:10]}")

    configs = sorted((GTJA / "examples" / "materialize").glob("gtja191_alpha_*.yaml"))
    if len(configs) != 185:
        raise RuntimeError(f"expected 185 materialize configs, got {len(configs)}")
    stale_configs = [
        path.name
        for path in configs
        if "(high + low + close) / 3" in path.read_text(encoding="utf-8")
    ]
    if stale_configs:
        raise RuntimeError(f"materialize configs still contain VWAP proxy: {stale_configs[:10]}")

    pack = json.loads((AUTO / "factor_packs" / "gtja185_pack.json").read_text(encoding="utf-8"))
    if len(pack.get("factors") or []) != 185:
        raise RuntimeError("AutoFactorEvaluation GTJA185 pack count mismatch")


def main() -> None:
    patch_pit_policies()
    patch_gtja_tests()
    rebuild_assets()
    verify_assets()
    print("GTJA185 canonical catalog, configs, candidate pool and factor pack rebuilt")


if __name__ == "__main__":
    main()
