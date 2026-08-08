# -*- coding: utf-8 -*-
"""因子运行血缘（Run Lineage）记录。"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RunLineage:
    """单次因子运行的血缘记录（AST、算子 catalog、DQ、行数等）。"""

    run_id: str
    factor_id: str
    factor_name: str
    ast_hash: str
    operator_catalog_hash: str
    field_catalog_hash: str = ""
    expression: str | None = None
    lookback: int = 0
    referenced_columns: list[str] = field(default_factory=list)
    dq_passed: bool | None = None
    row_count: int | None = None
    non_null_count: int | None = None
    # P1-024: universe-mask coverage recorded at materialization.  ``coverage_mask``
    # is a compact serializable summary of the applied universe mask; None when no
    # universe mask was applied (full-universe use).
    coverage_mask: dict[str, Any] | None = None
    coverage_ratio: float | None = None
    drop_reason: str | None = None
    engine_version: str = "factor_engine"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典，便于 catalog / JSON 持久化。"""
        return asdict(self)


def new_run_id() -> str:
    """生成新的运行 UUID（hex，无连字符）。"""
    return uuid.uuid4().hex


def build_run_lineage(
    *,
    factor_id: str,
    factor_name: str,
    ast_hash: str,
    operator_catalog_hash: str,
    field_catalog_hash: str | None = None,
    expression: str | None = None,
    lookback: int = 0,
    referenced_columns: set[str] | list[str] = (),
    result=None,
    dq_report=None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
    coverage_mask: dict[str, Any] | None = None,
    coverage_ratio: float | None = None,
    drop_reason: str | None = None,
) -> RunLineage:
    """从分析结果与执行输出组装 :class:`RunLineage`。"""
    row_count = len(result) if result is not None else None
    non_null = int(result.notna().sum()) if result is not None and hasattr(result, "notna") else None
    dq_passed = dq_report.passed if dq_report is not None else None
    if field_catalog_hash is None:
        try:
            from fields import compute_field_catalog_hash

            field_catalog_hash = compute_field_catalog_hash()
        except Exception:
            field_catalog_hash = ""
    return RunLineage(
        run_id=run_id or new_run_id(),
        factor_id=factor_id,
        factor_name=factor_name,
        ast_hash=ast_hash,
        operator_catalog_hash=operator_catalog_hash,
        field_catalog_hash=field_catalog_hash,
        expression=expression,
        lookback=int(lookback),
        referenced_columns=sorted(referenced_columns),
        dq_passed=dq_passed,
        row_count=row_count,
        non_null_count=non_null,
        coverage_mask=coverage_mask,
        coverage_ratio=coverage_ratio,
        drop_reason=drop_reason,
        extra=extra or {},
    )


def hash_data_source_config(config: dict[str, Any]) -> str:
    """对数据源配置做稳定 SHA256，用作 ``data_snapshot_id``。"""
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_git_commit_hash(*, repo_root: str | None = None) -> str | None:
    """尽力采集当前代码 git commit（用于 lineage 可复现）。"""
    import subprocess
    from pathlib import Path

    candidates: list[Path] = []
    if repo_root:
        candidates.append(Path(repo_root))
    try:
        from workspace_paths import quant_projects_root

        candidates.append(quant_projects_root())
    except Exception:
        pass

    for root in candidates:
        git_dir = root / ".git"
        if not git_dir.exists():
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            return commit or None
    return None
