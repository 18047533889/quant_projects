# -*- coding: utf-8
"""算子 production 认证流水线（parity → evidence → manifest）。"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_JSON = FE_ROOT / "evidence" / "primitive_verified.json"
COMPOSITE_EVIDENCE_JSON = FE_ROOT / "evidence" / "composite_verified.json"


class CertStage(str, Enum):
    POLARS_REFERENCE = "polars_reference_parity"
    DUCKDB_REFERENCE = "duckdb_reference_parity"
    DUCKDB_REAL_SQL = "duckdb_real_sql_verified"
    POLARS_EDGE = "polars_edge_verified"
    DUCKDB_EDGE = "duckdb_edge_verified"
    NO_FALLBACK = "no_fallback_verified"


@dataclass
class CertStageResult:
    stage: CertStage
    passed: bool
    detail: str = ""


@dataclass
class CertificationReport:
    canonical: str
    stages: list[CertStageResult] = field(default_factory=list)
    wrote_evidence: bool = False
    manifest_ok: bool = False

    @property
    def ok(self) -> bool:
        return all(s.passed for s in self.stages) and self.manifest_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "ok": self.ok,
            "stages": {s.stage.value: {"passed": s.passed, "detail": s.detail} for s in self.stages},
            "wrote_evidence": self.wrote_evidence,
            "manifest_ok": self.manifest_ok,
        }


# 阶段 → pytest 选择器（须与 parity case 名称一致）
_STAGE_PYTEST: dict[CertStage, list[str]] = {
    CertStage.POLARS_REFERENCE: [
        "tests/backend_parity/test_production_core_triple_parity.py",
        "tests/backend_parity/test_production_safe_bulk_parity.py",
        "tests/backend_parity/test_batch2_rolling_triple_parity.py",
    ],
    CertStage.DUCKDB_REAL_SQL: [
        "tests/backend_parity/test_production_core_triple_parity.py",
        "tests/backend_parity/test_production_safe_bulk_parity.py",
        "tests/backend_parity/test_batch2_rolling_triple_parity.py",
    ],
    CertStage.POLARS_EDGE: [
        "tests/backend_parity/test_p0_edge_cases_triple_parity.py",
        "tests/backend_parity/test_batch2_rolling_triple_parity.py",
    ],
    CertStage.DUCKDB_EDGE: [
        "tests/backend_parity/test_p0_edge_cases_triple_parity.py",
        "tests/backend_parity/test_batch2_rolling_triple_parity.py",
    ],
    CertStage.NO_FALLBACK: ["tests/backend_parity/test_polars_long_no_pandas_path.py"],
}


def _pytest_for_stage(stage: CertStage, canon: str) -> tuple[bool, str]:
    files = _STAGE_PYTEST.get(stage, [])
    if not files:
        return True, "skip"
    expr = canon.replace("_", "\\_")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *[f"{f}::{expr}" if "::" not in f else f for f in files],
        "-k",
        canon,
        "--tb=line",
    ]
    flat_files = []
    for f in files:
        flat_files.append(f)
    cmd = [sys.executable, "-m", "pytest", "-q", *flat_files, "-k", canon, "--tb=line"]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    env["PYTHONPATH"] = f"{root}:{fe}"
    proc = subprocess.run(
        cmd,
        cwd=str(FE_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return True, "passed"
    if "no tests ran" in out.lower() or "deselected" in out.lower():
        return False, f"无 parity case: 先在测试模块添加 {canon!r} case"
    return False, out.strip()[-500:]


def _load_evidence(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_evidence_key(canon: str, stage: CertStage, *, composite: bool = False) -> None:
    path = COMPOSITE_EVIDENCE_JSON if composite else EVIDENCE_JSON
    data = _load_evidence(path)
    key = stage.value
    if composite:
        mapping = {
            CertStage.POLARS_REFERENCE: "reference_to_lowered_pandas",
            CertStage.DUCKDB_REAL_SQL: "lowered_duckdb_real_sql",
            CertStage.POLARS_EDGE: "lowered_polars_native",
            CertStage.DUCKDB_EDGE: "edge_verified",
        }
        key = mapping.get(stage, stage.value)
    items = sorted(set(data.get(key, [])) | {canon})
    data[key] = items
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_certification(
    canonical: str,
    *,
    write_evidence: bool = False,
    refresh_manifest: bool = True,
) -> CertificationReport:
    """运行单算子认证流水线。"""
    from backend.phase1_scope import assert_phase1_freeze, is_phase1_in_scope
    from planner.composite_lowering import has_composite_lowering

    canon = canonical.strip()
    report = CertificationReport(canonical=canon)
    if not is_phase1_in_scope(canon):
        report.stages.append(
            CertStageResult(CertStage.POLARS_REFERENCE, False, "不在 phase1 冻结范围")
        )
        return report

    is_composite = has_composite_lowering(canon)
    stages = [
        CertStage.POLARS_REFERENCE,
        CertStage.DUCKDB_REAL_SQL,
        CertStage.POLARS_EDGE,
        CertStage.DUCKDB_EDGE,
        CertStage.NO_FALLBACK,
    ]
    if is_composite:
        stages = [
            CertStage.POLARS_REFERENCE,
            CertStage.DUCKDB_REAL_SQL,
            CertStage.POLARS_EDGE,
        ]

    for stage in stages:
        ok, detail = _pytest_for_stage(stage, canon)
        report.stages.append(CertStageResult(stage, ok, detail))
        if ok and write_evidence:
            _write_evidence_key(canon, stage, composite=is_composite)

    report.wrote_evidence = write_evidence

    if refresh_manifest:
        proc = subprocess.run(
            [
                sys.executable,
                str(FE_ROOT / "scripts" / "export_operator_manifest.py"),
                "--out",
                str(FE_ROOT / "benchmarks" / "operator_manifest.json"),
            ],
            cwd=str(FE_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": f"{FE_ROOT.parent}:{FE_ROOT}"},
            capture_output=True,
            text=True,
        )
        report.manifest_ok = proc.returncode == 0
    else:
        report.manifest_ok = True

    if write_evidence and report.manifest_ok:
        contract = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/backend/test_primitive_evidence_contract.py",
                "tests/backend/test_composite_evidence_contract.py",
            ],
            cwd=str(FE_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": f"{FE_ROOT.parent}:{FE_ROOT}"},
            capture_output=True,
            text=True,
        )
        if contract.returncode != 0:
            report.manifest_ok = False
            report.stages.append(
                CertStageResult(
                    CertStage.NO_FALLBACK,
                    False,
                    "evidence contract 失败（JSON 须与测试 case 一致，不可手填）",
                )
            )

    return report
