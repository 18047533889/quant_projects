# -*- coding: utf-8 -*-
"""R34 evidence framework：GateResult / EvidenceHeader / current-head 绑定。

在 ``backend/evidence_provenance`` 的既有溯源设施之上扩展，不重复制造同义
模块（R34 §189）。三个不变量：

1. ``executed_cases > 0`` 才允许 PASS —— 禁止硬编码 ``gates["x"] = True``；
2. artifact 必须绑定当前 HEAD 与 working-tree hash —— 旧 evidence 不得沿用；
3. 可重复计算 —— 同 SHA + 同 env 下 artifact 内容稳定。

``scan_hardcoded_true_gates`` 用 AST 扫描审计脚本，把字面量 True/False gate
赋值全部暴露出来（negative control），再由调用方判定每个是否是真检查。
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

FE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FE_ROOT.parent

GateStatus = Literal["PASS", "FAIL", "NOT_RUN", "N/A"]


@dataclass(frozen=True)
class GateResult:
    """一个 hard gate 的可验证结果。

    PASS 只允许出现在 ``executed_cases > 0`` 且 ``failed_cases == 0`` 时。
    缺少 executed case 的 gate 一律 NOT_RUN —— 即使代码路径返回 True。
    """

    gate_id: str
    status: GateStatus
    executed_cases: int
    failed_cases: int
    evidence_files: tuple[str, ...] = ()
    evidence_hashes: tuple[str, ...] = ()
    commit_sha: str = ""
    details: dict = field(default_factory=dict)

    @classmethod
    def not_run(cls, gate_id: str, reason: str, commit_sha: str = "") -> "GateResult":
        return cls(gate_id, "NOT_RUN", 0, 0, details={"reason": reason}, commit_sha=commit_sha)

    @classmethod
    def from_cases(
        cls, gate_id: str, cases: list[bool], evidence_files: tuple[str, ...] = (),
        commit_sha: str = "", details: dict | None = None,
    ) -> "GateResult":
        """从 case 级布尔结果聚合一个 gate。

        case 为空 -> NOT_RUN；有 FAIL -> FAIL；全 PASS 且 case 数 > 0 -> PASS。
        """
        executed = len(cases)
        failed = int(any(not c for c in cases))
        if executed == 0:
            status: GateStatus = "NOT_RUN"
        elif failed:
            status = "FAIL"
        else:
            status = "PASS"
        return cls(
            gate_id=gate_id, status=status, executed_cases=executed,
            failed_cases=failed, evidence_files=evidence_files,
            commit_sha=commit_sha, details=details or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "executed_cases": self.executed_cases,
            "failed_cases": self.failed_cases,
            "evidence_files": list(self.evidence_files),
            "evidence_hashes": list(self.evidence_hashes),
            "commit_sha": self.commit_sha,
            "details": self.details,
        }


@dataclass(frozen=True)
class EvidenceHeader:
    """R34 §4 统一 evidence header。任一字段不匹配 => evidence invalid。"""

    commit_sha: str
    dirty_tree_hash: str
    operator_registry_hash: str
    operator_surface_hash: str
    operator_semantic_hash: str
    field_catalog_hash: str
    dataaccess_contract_hash: str
    planner_hash: str
    backend_hash: str
    runtime_hash: str
    test_source_hash: str
    golden_source_hash: str
    dependency_lock_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "commit_sha": self.commit_sha,
            "dirty_tree_hash": self.dirty_tree_hash,
            "operator_registry_hash": self.operator_registry_hash,
            "operator_surface_hash": self.operator_surface_hash,
            "operator_semantic_hash": self.operator_semantic_hash,
            "field_catalog_hash": self.field_catalog_hash,
            "dataaccess_contract_hash": self.dataaccess_contract_hash,
            "planner_hash": self.planner_hash,
            "backend_hash": self.backend_hash,
            "runtime_hash": self.runtime_hash,
            "test_source_hash": self.test_source_hash,
            "golden_source_hash": self.golden_source_hash,
            "dependency_lock_hash": self.dependency_lock_hash,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EvidenceHeader):
            return False
        return self.to_dict() == other.to_dict()


def current_commit_sha() -> str:
    from backend.evidence_provenance import current_commit_sha as _cur

    return _cur()


def _tree_hash(root: Path, patterns: tuple[str, ...]) -> str:
    from backend.evidence_provenance import _tree_hash as _th

    return _th(root, patterns)


def _source_hash(path: Path) -> str:
    from backend.evidence_provenance import _source_hash as _sh

    return _sh(path)


def _compute_payload_hash(payload: Any) -> str:
    from backend.evidence_provenance import compute_payload_hash

    return compute_payload_hash(payload)


def _git(*args: str) -> str | None:
    try:
        return (
            subprocess_check_output(args)
        )
    except Exception:
        return None


def subprocess_check_output(args: list[str]) -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(args, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def dirty_tree_hash() -> str:
    """Working-tree-bound hash（未提交改动也会改变它）。

    用 git 的 clean-filter digest 逐文件计算 factor_engine 下全部 tracked
    py/json/csv —— 与 ``evidence_provenance._tree_hash`` 同一机制，因此任何
    unstaged 编辑都会令 evidence 失效。
    """
    return _tree_hash(FE_ROOT, ("*.py", "*.json", "*.csv"))


def component_hashes() -> dict[str, str]:
    """计算 EvidenceHeader 的 13 个组成 hash。"""
    cleaned = FE_ROOT / "cleaned_operators"
    fields = FE_ROOT / "fields"
    planner = FE_ROOT / "planner"
    backend = FE_ROOT / "backend"
    runtime = FE_ROOT / "runtime"
    tests = FE_ROOT / "tests"

    # operator semantic：同一份 semantic contract hash（与 primitive evidence 共用）
    try:
        from backend.evidence_provenance import semantic_hashes_for

        semantic_hash = _compute_payload_hash(
            {
                "policy": _source_hash(cleaned / "operator_policy.py"),
                "signature": _source_hash(backend / "production_signature.py"),
                "semantics": _compute_payload_hash(
                    {
                        "numeric": _source_hash(backend / "numeric_semantics.py"),
                        "cross_section": _source_hash(backend / "cross_section_spec.py"),
                    }
                ),
            }
        )
    except Exception:
        semantic_hash = ""

    # DataAccess：优先 hash monorepo 内 dataaccess/ 树，否则用安装版本
    da_hash = ""
    da_root = REPO_ROOT / "dataaccess"
    if da_root.is_dir():
        da_hash = _tree_hash(da_root, ("*.py",))
    else:
        try:
            import data_access  # type: ignore

            da_hash = _compute_payload_hash({"version": str(getattr(data_access, "__version__", "?"))})
        except Exception:
            da_hash = "missing"

    lock = _source_hash(FE_ROOT / "pyproject.toml")
    for extra in ("requirements.lock", "uv.lock", "poetry.lock", "requirements.txt"):
        p = REPO_ROOT / extra
        if p.is_file():
            lock = _compute_payload_hash(
                {"pyproject": lock, extra: _source_hash(p)}
            )
            break

    return {
        "commit_sha": current_commit_sha(),
        "dirty_tree_hash": dirty_tree_hash(),
        "operator_registry_hash": _tree_hash(cleaned, ("*.py",)),
        "operator_surface_hash": _source_hash(cleaned / "operator_surface.py"),
        "operator_semantic_hash": semantic_hash,
        "field_catalog_hash": _tree_hash(fields, ("*.py",)),
        "dataaccess_contract_hash": da_hash,
        "planner_hash": _tree_hash(planner, ("*.py",)),
        "backend_hash": _tree_hash(backend, ("*.py",)),
        "runtime_hash": _tree_hash(runtime, ("*.py",)),
        "test_source_hash": _tree_hash(tests, ("*.py",)),
        "golden_source_hash": _tree_hash(tests / "operator_golden", ("*.py", "*.csv", "*.json", "*.parquet")),
        "dependency_lock_hash": lock,
    }


def current_evidence_header() -> EvidenceHeader:
    hashes = component_hashes()
    return EvidenceHeader(
        commit_sha=hashes["commit_sha"],
        dirty_tree_hash=hashes["dirty_tree_hash"],
        operator_registry_hash=hashes["operator_registry_hash"],
        operator_surface_hash=hashes["operator_surface_hash"],
        operator_semantic_hash=hashes["operator_semantic_hash"],
        field_catalog_hash=hashes["field_catalog_hash"],
        dataaccess_contract_hash=hashes["dataaccess_contract_hash"],
        planner_hash=hashes["planner_hash"],
        backend_hash=hashes["backend_hash"],
        runtime_hash=hashes["runtime_hash"],
        test_source_hash=hashes["test_source_hash"],
        golden_source_hash=hashes["golden_source_hash"],
        dependency_lock_hash=hashes["dependency_lock_hash"],
    )


# ---------------------------------------------------------------------------
# Hardcoded-True / presence-only gate 扫描（P0-002 / R34-ZERO_HARDCODED_TRUE_GATES）
# ---------------------------------------------------------------------------

def scan_hardcoded_true_gates(paths: list[Path]) -> list[dict[str, Any]]:
    """AST 扫描，返回 audit 脚本里字面量 gate 赋值。

    每个结果：``{"file": ..., "line": ..., "literal": True/False, "node": ...}``。
    调用方把这些拿去判定是真检查（try/except 探针、aggregate case）还是假 gate。
    """
    findings: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                _scan_assign(node, path, findings)
            elif isinstance(node, ast.AugAssign):
                continue
    return findings


def _scan_assign(node: ast.Assign, path: Path, findings: list[dict[str, Any]]) -> None:
    # gates["X"] = <literal> or gates[var] = <literal>
    value_literal = None
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
        value_literal = node.value.value
    if value_literal is None:
        return
    for target in node.targets:
        if _is_gates_subscript(target) or _is_gates_attr(target):
            findings.append(
                {
                    "file": str(path.relative_to(FE_ROOT)),
                    "line": node.lineno,
                    "literal": value_literal,
                    "assign_type": type(node).__name__,
                }
            )


def _is_gates_subscript(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        value = node.value
        while isinstance(value, ast.Subscript):
            value = value.value
        if isinstance(value, ast.Name):
            return value.id in {"gates", "results", "outcomes", "gate_results"}
        if isinstance(value, ast.Attribute) and value.attr in {"gates", "results"}:
            return True
    return False


def _is_gates_attr(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr in {"passed", "ok", "status"}:
        # e.g. ``gates["x"]["passed"] = True`` handled above; attribute form:
        # ``result.passed = True`` — flag only when leftmost is a gates-like name.
        base = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        return isinstance(base, ast.Name) and base.id in {"gate", "g"}
    return False


# ---------------------------------------------------------------------------
# Evidence freshness（P0-001）
# ---------------------------------------------------------------------------

def stale_evidence_report() -> dict[str, Any]:
    """枚举已知证据 artifact 的绑定 SHA，判定是否 current-HEAD / ancestor。"""
    head = current_commit_sha()
    report: dict[str, Any] = {"current_head": head, "artifacts": {}}

    def _record(name: str, path: Path, sha: str | None) -> None:
        bound = bool(sha)
        current = bound and sha == head
        report["artifacts"][name] = {
            "path": str(path.relative_to(FE_ROOT)) if path else "",
            "bound_sha": sha or "",
            "bound": bound,
            "current_head": current,
        }

    _record("factor_operator_verified", FE_ROOT / "evidence" / "factor_operator_verified.json",
            _json_sha(FE_ROOT / "evidence" / "factor_operator_verified.json", "commit_sha"))
    _record("primitive_verified", FE_ROOT / "evidence" / "primitive_verified.json",
            _json_sha(FE_ROOT / "evidence" / "primitive_verified.json", None))
    for round_name in ("r28", "r30", "r31", "r32"):
        dirp = FE_ROOT / "docs" / "evidence" / round_name
        for f in sorted(dirp.glob("*.json")):
            _record(f"{round_name}/{f.stem}", f, _json_sha(f, None))
    return report


def _json_sha(path: Path, key: str | None) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if key:
        return str(data.get(key) or "") or None
    # 无 commit 字段 -> 视为不绑定
    if isinstance(data, dict) and any(
        k in data for k in ("commit_sha", "commit", "sha", "head", "HEAD", "evidence_head")
    ):
        for k in ("commit_sha", "commit", "sha", "head", "HEAD", "evidence_head"):
            if k in data and str(data[k]):
                return str(data[k])
    return None


def evidence_store_path() -> Path:
    return FE_ROOT / "docs" / "evidence" / "r34"
