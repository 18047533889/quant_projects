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
    run_id: str
    factor_id: str
    factor_name: str
    ast_hash: str
    operator_catalog_hash: str
    expression: str | None
    lookback: int
    referenced_columns: list[str]
    dq_passed: bool | None = None
    row_count: int | None = None
    non_null_count: int | None = None
    engine_version: str = "factor_engine"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_run_id() -> str:
    return uuid.uuid4().hex


def build_run_lineage(
    *,
    factor_id: str,
    factor_name: str,
    ast_hash: str,
    operator_catalog_hash: str,
    expression: str | None,
    lookback: int,
    referenced_columns: set[str] | list[str],
    result=None,
    dq_report=None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunLineage:
    row_count = len(result) if result is not None else None
    non_null = int(result.notna().sum()) if result is not None and hasattr(result, "notna") else None
    dq_passed = dq_report.passed if dq_report is not None else None
    return RunLineage(
        run_id=run_id or new_run_id(),
        factor_id=factor_id,
        factor_name=factor_name,
        ast_hash=ast_hash,
        operator_catalog_hash=operator_catalog_hash,
        expression=expression,
        lookback=int(lookback),
        referenced_columns=sorted(referenced_columns),
        dq_passed=dq_passed,
        row_count=row_count,
        non_null_count=non_null,
        extra=extra or {},
    )


def hash_data_source_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
