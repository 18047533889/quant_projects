# -*- coding: utf-8 -*-
"""Correct COS object-layout handling for event, period and sparse tables."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from data_access.core.exceptions import ValidationError
from .cos_contract import COS_DATASET_CONTRACTS

_RECURSIVE_LAYOUTS = frozenset({"period_files", "event_files", "sparse_files"})
_COMPLETE_MARKER = ".data_access_complete.json"
_installed = False
_originals: dict[str, Any] = {}


def _allow_full_sync() -> bool:
    """P1-6 FIX: Validate boolean env var at read time."""
    raw = os.environ.get("DATA_ACCESS_ALLOW_FULL_COS_SYNC", "").strip().lower()
    if not raw:
        return False
    allowed = {"0", "1", "true", "false", "yes", "no", "on", "off"}
    if raw not in allowed:
        raise ValueError(
            f"DATA_ACCESS_ALLOW_FULL_COS_SYNC='{raw}' is invalid; "
            f"allowed values: {', '.join(sorted(allowed))}"
        )
    return raw in {"1", "true", "yes", "on"}


def _local_dir(spec: Any) -> Path:
    from data_access.cos import mirror
    return mirror._local_table_dir(spec)


def _marker_path(spec: Any) -> Path:
    return _local_dir(spec) / _COMPLETE_MARKER


def _parquet_files(spec: Any) -> list[Path]:
    root = _local_dir(spec)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file() and p.stat().st_size > 0)


def _inventory_hash(root: Path, files: list[Path]) -> str:
    """#P1-1 本地 parquet 清单指纹：(相对路径|size|mtime_ns) 的 sha256。

    marker 必须绑定本地对象清单，否则上游新增文件后「永久 complete」——本地一旦
    出现 marker 里没有的新文件/变更，指纹即 mismatch → 视为 stale，强制重同步。
    """
    import hashlib

    digest = hashlib.sha256()
    for p in files:
        try:
            st = p.stat()
            rel = p.relative_to(root)
        except (OSError, ValueError):
            continue
        digest.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def _write_complete_marker(spec: Any, *, source: str) -> None:
    root = _local_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    files = _parquet_files(spec)
    payload = {
        "source": source,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "layout": spec.layout,
        "cos_prefix": spec.cos_prefix,
        "table": spec.table,
        # #P1-1 RemoteSnapshot 字段：object_count / inventory_hash 让 marker 不再是
        # 「永久 complete」，本地/上游清单变化 → 指纹 mismatch → stale。
        "object_count": len(files),
        "inventory_hash": _inventory_hash(root, files),
    }
    _marker_path(spec).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _complete_marker_valid(spec: Any) -> bool:
    """#P1-1 marker 存在 + parquet 存在 + 本地清单指纹与 marker 一致才算 complete。

    指纹 mismatch（新文件被同步进来但 marker 没更新）→ stale，要求重新 full sync，
    避免「今天 sync 完，明天上游加文件，本地永远 complete 不更新」。
    """
    marker = _marker_path(spec)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    files = _parquet_files(spec)
    if not files:
        return False
    recorded_hash = payload.get("inventory_hash")
    if isinstance(recorded_hash, str) and recorded_hash:
        if _inventory_hash(_local_dir(spec), files) != recorded_hash:
            return False
    return True


def _full_sync(spec: Any, *, source: str) -> list[Path]:
    from data_access.cos import mirror
    root = _local_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    mirror._run_cos_cli(["sync", mirror._cos_table_uri(spec) + "/", str(root) + "/"])
    files = _parquet_files(spec)
    if not files:
        raise ValidationError(f"COS full sync completed without parquet files: {mirror._cos_table_uri(spec)}")
    _write_complete_marker(spec, source=source)
    return files


def _require_full_sync_opt_in(dataset_name: str) -> None:
    if _allow_full_sync():
        return
    raise ValidationError(
        f"COS 数据集 {dataset_name!r} 的物理布局不能由 decision time 枚举文件。"
        "请优先使用 DATA_ACCESS_COS_REMOTE_BACKEND=httpfs；"
        "确需 CLI/本地整表同步时显式设置 DATA_ACCESS_ALLOW_FULL_COS_SYNC=1。"
    )


def _patched_sync_dataset(dataset_name: str, *, time_range: tuple[Any, Any] | None = None, full: bool = False) -> None:
    from data_access.cos import mirror
    spec = mirror.mirror_spec_for_dataset(dataset_name)
    if spec is None or spec.layout not in _RECURSIVE_LAYOUTS:
        return _originals["mirror.sync_dataset"](dataset_name, time_range=time_range, full=full)
    if not full:
        _require_full_sync_opt_in(dataset_name)
    _full_sync(spec, source="mirror")


def _patched_ensure_local_mirror(dataset_name: str, *, time_range: tuple[Any, Any] | None = None) -> None:
    from data_access.cos import mirror
    if mirror.skip_cos_mirror():
        return
    spec = mirror.mirror_spec_for_dataset(dataset_name)
    if spec is None or spec.layout not in _RECURSIVE_LAYOUTS:
        return _originals["mirror.ensure_local_mirror"](dataset_name, time_range=time_range)
    files = _parquet_files(spec)
    if _complete_marker_valid(spec):
        return
    if files:
        raise ValidationError(
            f"COS 本地镜像 {dataset_name!r} 含递归事件/期间文件但没有完整同步 marker；"
            "无法证明查询范围完整。请执行显式 full sync 或切换 remote/httpfs。"
        )
    _require_full_sync_opt_in(dataset_name)
    _full_sync(spec, source="mirror")


def _month_globs(spec: Any, time_range: tuple[Any, Any] | None) -> list[str]:
    from data_access.cos import remote
    base = remote._s3_table_base(spec)
    remote.authorize_s3_path(base + "/")
    if time_range is None:
        return [f"{base}/*.parquet"]
    start = remote._parse_date(time_range[0])
    end = remote._parse_date(time_range[1])
    if start is None or end is None:
        return [f"{base}/*.parquet"]
    if start > end:
        return []
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return [f"{base}/{year:04d}-{month:02d}-*.parquet" for year, month in months]


def _patched_build_remote_paths(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    ds: Any = None,
    params: Any = None,
) -> list[str]:
    # #P0-1 新版 canonical 签名带 ds/params；旧 patch 必须原样透传，否则
    # install_cos_storage_runtime 会把 ParametricDataset generic remote 弄残。
    from data_access.cos import remote
    spec = remote.mirror_spec_for_dataset(dataset_name)
    if spec is None:
        return _originals["remote.build_remote_paths"](
            dataset_name, time_range=time_range, ds=ds, params=params
        )
    if spec.layout == "daily_parquet":
        # Exact day lists include weekends and holidays. Month globs preserve
        # object pruning while the structured predicate enforces the exact range.
        return _month_globs(spec, time_range)
    if spec.layout not in _RECURSIVE_LAYOUTS:
        return _originals["remote.build_remote_paths"](
            dataset_name, time_range=time_range, ds=ds, params=params
        )
    base = remote._s3_table_base(spec)
    remote.authorize_s3_path(base + "/")
    # Availability time cannot infer period_end/event filenames. DuckDB filters
    # the wildcard scan by the explicit semantic clock column.
    return [f"{base}/**/*.parquet"]


def _patched_local_complete(dataset_name: str, *, time_range: tuple[Any, Any] | None) -> bool:
    from data_access.cos import remote
    spec = remote.mirror_spec_for_dataset(dataset_name)
    if spec is None or spec.layout not in _RECURSIVE_LAYOUTS:
        return _originals["remote.local_mirror_complete_for_range"](dataset_name, time_range=time_range)
    return _complete_marker_valid(spec)


def _patched_materialize_remote_via_cli(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    ds: Any = None,
    params: Any = None,
) -> list[str]:
    # #P0-1 与 _patched_build_remote_paths 同：ds/params 原样透传 canonical。
    from data_access.cos import remote
    spec = remote.mirror_spec_for_dataset(dataset_name)
    if spec is None or spec.layout not in _RECURSIVE_LAYOUTS:
        return _originals["remote.materialize_remote_via_cli"](
            dataset_name, time_range=time_range, ds=ds, params=params
        )
    cache_spec = remote._cache_mirror_spec(spec)
    files = _parquet_files(cache_spec)
    if _complete_marker_valid(cache_spec):
        return [str(path) for path in files]
    _require_full_sync_opt_in(dataset_name)
    files = _full_sync(cache_spec, source="remote_cli_cache")
    return [str(path) for path in files]


def install_cos_storage_runtime() -> None:
    global _installed
    if _installed:
        return
    from data_access.cos import mirror, remote
    for name, contract in COS_DATASET_CONTRACTS.items():
        layout = contract.storage_layout
        if layout not in _RECURSIVE_LAYOUTS:
            continue
        spec = mirror.DATASET_MIRROR_REGISTRY.get(name)
        if spec is not None:
            # 保存原始 spec，uninstall 才能精确还原（#P0 收官：install 不能永久
            # 改全局 registry，否则测试/同进程后续读全部被污染成递归布局）。
            _originals.setdefault("mirror.registry_entries", {})
            _originals["mirror.registry_entries"].setdefault(name, spec)
            mirror.DATASET_MIRROR_REGISTRY[name] = replace(spec, layout=layout)
    _originals.update({
        "mirror.sync_dataset": mirror.sync_dataset,
        "mirror.ensure_local_mirror": mirror.ensure_local_mirror,
        "remote.build_remote_paths": remote.build_remote_paths,
        "remote.local_mirror_complete_for_range": remote.local_mirror_complete_for_range,
        "remote.materialize_remote_via_cli": remote.materialize_remote_via_cli,
    })
    mirror.sync_dataset = _patched_sync_dataset
    mirror.ensure_local_mirror = _patched_ensure_local_mirror
    remote.build_remote_paths = _patched_build_remote_paths
    remote.local_mirror_complete_for_range = _patched_local_complete
    remote.materialize_remote_via_cli = _patched_materialize_remote_via_cli
    _installed = True


def uninstall_cos_storage_runtime() -> None:
    """#P0 收官：还原 install 造成的全部全局副作用（registry + 各 patch）。

    让 install 可以安全地用于测试/短期模式——测试结束时调用，后续代码不被
    递归布局 / 月份 glob patch 污染。幂等：未 install 时 no-op。
    """
    global _installed
    if not _installed:
        return
    from data_access.cos import mirror, remote
    entries = _originals.get("mirror.registry_entries", {})
    for name, spec in entries.items():
        mirror.DATASET_MIRROR_REGISTRY[name] = spec
    if "mirror.sync_dataset" in _originals:
        mirror.sync_dataset = _originals["mirror.sync_dataset"]
    if "mirror.ensure_local_mirror" in _originals:
        mirror.ensure_local_mirror = _originals["mirror.ensure_local_mirror"]
    if "remote.build_remote_paths" in _originals:
        remote.build_remote_paths = _originals["remote.build_remote_paths"]
    if "remote.local_mirror_complete_for_range" in _originals:
        remote.local_mirror_complete_for_range = _originals[
            "remote.local_mirror_complete_for_range"
        ]
    if "remote.materialize_remote_via_cli" in _originals:
        remote.materialize_remote_via_cli = _originals[
            "remote.materialize_remote_via_cli"
        ]
    _originals.clear()
    _installed = False


__all__ = ["install_cos_storage_runtime"]
