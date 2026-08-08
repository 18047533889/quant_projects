# -*- coding: utf-8 -*-
"""因子运行血缘（Run Lineage）记录。"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
    # Review-8 #483: ``result.notna().sum()`` on a DataFrame returns a Series
    # and ``int(Series)`` raises.  Flatten via ``np.asarray`` so panel-native
    # DataFrame results, Series results, and empty results all count cells.
    non_null = (
        int(np.asarray(result.notna()).sum())
        if result is not None and hasattr(result, "notna")
        else None
    )
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


_SECRET_KEY_FRAGMENTS = (
    "password",
    "token",
    "secret",
    "access_key",
    "secret_key",
    "session_token",
    "dsn",
    "credential",
    "authorization",
    "apikey",
    "api_key",
    "private_key",
)


def _redact_config_value(key: str, value: Any) -> Any:
    """Review-8 #452: redact secrets before hashing so credentials never enter
    the factor-lake catalog identity."""
    lowered = str(key).lower()
    if any(frag in lowered for frag in _SECRET_KEY_FRAGMENTS) and value is not None:
        return "<redacted>"
    return value


def _canonicalize_config_value(value: Any, key: str = "") -> Any:
    """Recursive canonical serialization of a data-source config.

    * dict / list / tuple recurse (secrets redacted);
    * str / int / bool / None pass through;
    * float must be finite — NaN/Inf are rejected (they would form a
      non-standard, un-round-trippable identity, Review-8 #484);
    * datetime / date / pydantic dump to ISO/JSON;
    * anything else raises instead of leaking an address-bearing ``str()``.
    """
    if isinstance(value, dict):
        return {
            k: _canonicalize_config_value(_redact_config_value(str(k), v), str(k))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_config_value(v, key) for v in value]
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        if not np.isfinite(num):
            raise ValueError(
                f"data source config contains non-finite value at key {key!r}"
            )
        return num
    if hasattr(value, "model_dump"):  # pydantic
        try:
            return _canonicalize_config_value(value.model_dump(mode="json"), key)
        except Exception:
            pass
    if hasattr(value, "isoformat"):  # datetime / date
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"data source config contains unsupported object {type(value).__name__} "
        f"at key {key!r}"
    )


def hash_data_source_config(config: dict[str, Any]) -> str:
    """对数据源配置做稳定 SHA256，用作 ``data_snapshot_id``。

    Review-8 #484: canonicalizes first (finite floats, ISO datetimes, secret
    redaction, ``allow_nan=False``); unsupported objects raise rather than
    leaking an address-bearing ``str()`` into the identity.
    """
    canonical = _canonicalize_config_value(config)
    payload = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
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
