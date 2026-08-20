# -*- coding: utf-8 -*-
"""R37 Phase 0: Baseline / Inventory。

输出（docs/evidence/r37/）：
    R37_BASELINE.json                     基线 HEAD、组件 hash、关键状态
    R37_CURRENT_OPERATOR_INVENTORY.parquet production canonical 逐算子 inventory
    R37_CURRENT_RESOURCE_ARCH.json         资源权威现状（唯一 HostResourceCoordinator?）
    R37_CURRENT_FE_DA_BOUNDARY.json        FE×DA 边界（capability/protocol）

R37 §0：执行前读取最新 main；基线 c2309dbd；若 HEAD 已变化先记录新 SHA。
本脚本每次运行都绑定审计时点 current HEAD——不再沿用旧 SHA 的 inventory。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from backend.cleaned_bridge import ensure_cleaned_loaded  # noqa: E402
from backend.evidence_provenance import current_commit_sha  # noqa: E402

OUT = Path("docs/evidence/r37")
OUT.mkdir(parents=True, exist_ok=True)


def _component_tree_hash(subdir: str) -> str:
    from backend.evidence_provenance import _tree_hash

    root = Path(subdir)
    if root.is_dir():
        return _tree_hash(root, ("*.py",))
    return "missing"


def build_operator_inventory() -> list[dict]:
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.operator_surface import classify_canonical

    rows = []
    for canon in sorted(factor_production_targets()):
        try:
            surface = classify_canonical(canon)
        except Exception:
            surface = "unknown"
        try:
            backends = ",".join(sorted(OperatorRegistry.backends_for(canon))) or "none"
        except Exception:
            backends = "unknown"
        try:
            from backend.production_signature import signature_for

            sig = signature_for(canon)
            typed = sig is not None and bool(getattr(sig, "params", ()))
        except Exception:
            typed = False
        rows.append({
            "canonical": canon,
            "surface": surface,
            "backends": backends,
            "has_typed_signature": typed,
        })
    return rows


def build_resource_arch() -> dict:
    import inspect

    arch: dict = {
        "modules": {},
        "single_authority": False,
    }
    probes = {
        "host_resource_coordinator": "runtime.host_resource_coordinator",
        "resource_broker": "runtime.resource_broker",
        "resource_governor": "runtime.resource_governor",
        "resource_autopilot": "runtime.resource_autopilot",
        "resource_calibration_store": "runtime.resource_calibration_store",
        "resource_monitor": "runtime.resource_monitor",
        "auto_shard_planner": "runtime.auto_shard_planner",
        "governed_buffer_store": "runtime.buffer_store",
        "change_impact_dag": "runtime.change_impact",
    }
    for name, modname in probes.items():
        try:
            mod = __import__(modname, fromlist=["*"])
            classes = [n for n, o in inspect.getmembers(mod, inspect.isclass)
                       if o.__module__ == mod.__name__]
            arch["modules"][name] = {"importable": True, "classes": classes}
        except Exception as e:
            arch["modules"][name] = {"importable": False, "error": str(e)}
    # 是否存在唯一 HostResourceCoordinator singleton？
    try:
        from runtime.host_resource_coordinator import HostResourceCoordinator

        has_singleton = hasattr(HostResourceCoordinator, "instance")
        arch["host_resource_coordinator_class"] = HostResourceCoordinator.__name__
        arch["has_singleton"] = has_singleton
        arch["single_authority"] = has_singleton
    except Exception as e:
        arch["host_resource_coordinator_class"] = None
        arch["error"] = str(e)
    return arch


def build_fe_da_boundary() -> dict:
    import inspect

    boundary: dict = {"fe_da_modules": {}, "capability_handshake": False}
    for name, modname in {
        "dataaccess_capabilities": "dataaccess.capabilities",
        "dataaccess_resource_bridge": "dataaccess.runtime.resource_bridge",
        "dataaccess_governor": "dataaccess.runtime.resource_governor",
        "fe_prepared_batch_session": "runtime.batch_service",
        "fe_batch_data_request": "runtime.batch_service",
    }.items():
        try:
            mod = __import__(modname, fromlist=["*"])
            classes = [n for n, o in inspect.getmembers(mod, inspect.isclass)
                       if o.__module__ == mod.__name__]
            funcs = [n for n, o in inspect.getmembers(mod, inspect.isfunction)
                     if o.__module__ == mod.__name__]
            boundary["fe_da_modules"][name] = {"importable": True,
                                               "classes": classes[:20],
                                               "funcs": funcs[:20]}
        except Exception as e:
            boundary["fe_da_modules"][name] = {"importable": False, "error": str(e)}
    try:
        from dataaccess.capabilities import DataAccessCapabilities  # type: ignore

        boundary["capability_handshake"] = True
    except Exception:
        boundary["capability_handshake"] = False
    return boundary


def main() -> int:
    ensure_cleaned_loaded()
    head = current_commit_sha()

    inv = build_operator_inventory()
    counts = Counter(r["surface"] for r in inv)
    typed = sum(1 for r in inv if r["has_typed_signature"])

    baseline = {
        "task_book": "FactorEngine_R37_最新HEAD剩余系统性风险终审_..._20260811.md",
        "task_book_baseline_sha": "c2309dbd4abe73945d8f1c97b402b8b4cbdc75b6",
        "audit_time_sha": head,
        "created_at": "2026-08-11",
        "producer_script": "scripts/generate_r37_baseline.py",
        "operator_inventory": {
            "total_production_canonicals": len(inv),
            "with_typed_signature": typed,
            "without_typed_signature": len(inv) - typed,
            "by_surface": dict(counts),
        },
        "resource_arch": build_resource_arch(),
        "fe_da_boundary": build_fe_da_boundary(),
    }

    # parquet inventory（回退 CSV）
    try:
        import polars as pl  # type: ignore

        df = pl.DataFrame(inv)
        df.write_parquet(OUT / "R37_CURRENT_OPERATOR_INVENTORY.parquet")
        baseline["operator_inventory"]["parquet_written"] = True
    except Exception as e:
        import csv

        with (OUT / "R37_CURRENT_OPERATOR_INVENTORY.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(inv[0].keys()))
            w.writeheader()
            for r in inv:
                w.writerow(r)
        baseline["operator_inventory"]["parquet_written"] = False
        baseline["operator_inventory"]["parquet_error"] = str(e)

    (OUT / "R37_BASELINE.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "R37_CURRENT_RESOURCE_ARCH.json").write_text(
        json.dumps(baseline["resource_arch"], indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "R37_CURRENT_FE_DA_BOUNDARY.json").write_text(
        json.dumps(baseline["fe_da_boundary"], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[r37-baseline] head={head} canonicals={len(inv)} typed={typed} "
          f"surfaces={dict(counts)}")
    print(f"[r37-baseline] single_resource_authority="
          f"{baseline['resource_arch'].get('single_authority')}")
    print(f"[r37-baseline] fe_da_capability_handshake="
          f"{baseline['fe_da_boundary'].get('capability_handshake')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
