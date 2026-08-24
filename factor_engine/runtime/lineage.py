# -*- coding: utf-8 -*-
"""因子运行血缘（Run Lineage）记录。"""

from __future__ import annotations

import hashlib
import json
import os
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


def build_engine_version() -> dict[str, str]:
    """R32-P0-042: 实际可复现的 engine/build 版本 —— 不只是 "factor_engine"。

    记录 package_version / git SHA / DataAccess version / Python / NumPy /
    Pandas / Polars / DuckDB / PyArrow / SciPy。任一采集失败记为 "unknown"
    （不伪造）。
    """
    out: dict[str, str] = {}

    def _ver(mod: str) -> str:
        try:
            m = __import__(mod)
            return str(getattr(m, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            return "unknown"

    out["package_version"] = _ver("factor_engine") or "unknown"
    try:
        import factor_engine  # noqa: F401

        out["package_version"] = str(getattr(factor_engine, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        out["package_version"] = "unknown"
    git_sha = resolve_git_commit_hash()
    out["git_sha"] = git_sha or "unknown"
    out["dataaccess_version"] = _ver("data_access") or _ver("dataaccess")
    out["python"] = str(__import__("sys").version.split()[0])
    for mod in ("numpy", "pandas", "polars", "duckdb", "pyarrow", "scipy"):
        out[mod] = _ver(mod)
    return out


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
        # R32-P0-041: production lineage 的 field catalog hash 必须 fail-closed。
        # 写空字符串会让「field catalog 计算失败」伪装成「无字段依赖」，下游
        # identity/evidence 校验拿到假值。失败直接抛，绝不下写 ""。
        from factor_engine.fields import compute_field_catalog_hash

        field_catalog_hash = compute_field_catalog_hash()
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
        # R32-P0-042: engine_version 记录实际可复现版本（package/build SHA /
        # DataAccess / Python / 关键库），不再是硬编码 "factor_engine"。
        engine_version=json.dumps(build_engine_version(), sort_keys=True),
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


def _sanitize_uri_value(value: str) -> str:
    """R32-P0-037/038: value-aware URI sanitizer —— 只去 credential。

    不能把整个 DSN 替换为 ``<redacted>``：那会把不同 scheme/host/port/database/
    dataset 的数据源折成同一个 source hash（身份碰撞）。正确做法是 parse URI，
    只 redact：
      - netloc 中的 password（``user:pass@host`` → ``user:<redacted>@host``）；
      - query 中 token/password/credential 类参数；
    保留 scheme / host / port / database / path / dataset / semantic options。

    非 URI 字符串原样返回（无 secret 结构可拆）；任何解析失败也原样返回。
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    if not isinstance(value, str) or not value.strip():
        return value
    try:
        parts = urlsplit(value)
    except (ValueError, TypeError):
        return value
    if not parts.scheme:
        return value
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            netloc = f"{user}:<redacted>@{host}"
        else:
            netloc = f"{userinfo}:<redacted>@{host}"
    query = parts.query
    if query:
        try:
            qs = parse_qsl(query, keep_blank_values=True)
        except (ValueError, TypeError):
            qs = []
        qs = [
            (
                k,
                "<redacted>"
                if any(f in str(k).lower() for f in _SECRET_KEY_FRAGMENTS)
                else v,
            )
            for k, v in qs
        ]
        query = urlencode(qs)
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _redact_config_value(key: str, value: Any) -> Any:
    """Review-8 #452 + R32-P0-037/038: redact secrets before hashing.

    - 键名命中 secret 片段：str 值走 URI sanitizer（保留 scheme/host/db，只去
      credential）；非 str 值整体 ``<redacted>``（无结构可拆）；
    - 普通字符串值也经 ``_canonicalize_config_value`` 的 URI 检查（P0-038：
      credential 藏在普通 url 值里时按值识别）。
    """
    lowered = str(key).lower()
    if value is None:
        return value
    if any(frag in lowered for frag in _SECRET_KEY_FRAGMENTS):
        if isinstance(value, str):
            return _sanitize_uri_value(value)
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
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        # R32-P0-038: credential 藏在普通 url 值里（非 secret 键名）时按值识别。
        return _sanitize_uri_value(value)
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


#: R32-P0-039: source config 中可被误当作 source identity 的相对路径键名。
_PATH_IDENTITY_KEYS = frozenset({
    "path", "root", "dir", "directory", "data_dir", "dataset_dir", "file",
})


def audit_source_path_identity(config: dict[str, Any], *, cwd: str | None = None) -> dict[str, Any]:
    """R32-P0-039: 审计 source config 中 CWD-relative path 的碰撞风险。

    relative ``./data`` 字符串跨 working directory 会碰撞成同一 source identity。
    返回：
      - ``relative_paths``：发现的相对路径键值；
      - ``canonical_logical_ids``：config 中可作为 logical dataset id 的键
        （dataset / dataset_id / table / manifest_snapshot）—— 优先用它们做
        source identity；
      - ``has_relative_collision_risk``：是否存在相对路径且无 logical id 兜底。
    """
    relative_paths: list[dict[str, str]] = []
    logical_ids: dict[str, str] = {}
    resolved_cwd = cwd or os.getcwd()

    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in {"dataset", "dataset_id", "table", "manifest_snapshot", "snapshot_id"} and isinstance(v, str):
                    logical_ids[key] = v
                _walk(v, f"{prefix}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _walk(item, f"{prefix}[{i}]")
        elif isinstance(obj, str) and obj.strip():
            lower = prefix.rsplit(".", 1)[-1].lower() if prefix else ""
            if lower in _PATH_IDENTITY_KEYS and not os.path.isabs(obj) and "://" not in obj:
                relative_paths.append({"key": prefix, "value": obj})

    _walk(config)
    has_logical_id = bool(logical_ids)
    return {
        "relative_paths": relative_paths,
        "canonical_logical_ids": logical_ids,
        "has_logical_id": has_logical_id,
        "has_relative_collision_risk": bool(relative_paths) and not has_logical_id,
        "cwd": resolved_cwd,
    }


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
