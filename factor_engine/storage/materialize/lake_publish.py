"""因子湖 staging → published 发布（带人工审批门）。"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from storage.exceptions import FactorNotFoundError


class PublishNotApprovedError(PermissionError):
    """因子湖发布未获审批时抛出。
    
    参数:
        无
    """


def is_publish_approved(*, approve: bool = False) -> bool:
    """检查发布是否已获审批。
    
    参数:
        approve: 是否显式审批发布（可选）
    
    返回:
        bool
    """
    if approve:
        return True
    return os.environ.get("QUANT_PUBLISH_APPROVED", "").lower() in (
        "1",
        "true",
        "yes",
    )


def require_publish_approval(*, approve: bool = False) -> None:
    """要求发布审批，未通过则抛错。
    
    参数:
        approve: 是否显式审批发布（可选）
    
    返回:
        无
    """
    if not is_publish_approved(approve=approve):
        raise PublishNotApprovedError(
            "因子湖发布需要显式审批：传入 approve=True 或设置 QUANT_PUBLISH_APPROVED=1"
        )


def _resolve_staging_factor_dir(factor_id: str) -> Path:
    """解析 staging 中因子目录路径。
    
    参数:
        factor_id: 因子唯一标识
    
    返回:
        Path
    """
    from data_access import get_store

    store = get_store()
    return store.resolve_dataset_path("factor_lake_staging", factor_id=factor_id)


def _count_parquet_rows(root: Path) -> int:
    """统计目录下 Parquet 文件总行数。
    
    参数:
        root: 根目录路径
    
    返回:
        int
    """
    import pyarrow.parquet as pq

    total = 0
    if not root.exists():
        return 0
    for pq_file in root.rglob("*.parquet"):
        total += pq.read_metadata(pq_file).num_rows
    return total


def sync_local_factor_to_staging(
    *,
    factor_id: str,
    lake_root: str | Path,
) -> dict[str, Any]:
    """将本地因子湖目录原子复制到 staging。
    
    参数:
        factor_id: 因子唯一标识（可选）
        lake_root: 因子湖根目录（可选）
    
    返回:
        dict[str, Any]
    """
    source = Path(lake_root) / "factors" / factor_id
    if not source.exists():
        raise FileNotFoundError(f"本地因子目录不存在: {source}")

    staging_dir = _resolve_staging_factor_dir(factor_id)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = staging_dir.parent / f".{factor_id}.staging_tmp_{uuid.uuid4().hex[:8]}"
    backup_dir = staging_dir.parent / f".{factor_id}.staging_bak_{uuid.uuid4().hex[:8]}"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(source, tmp_dir)
        if staging_dir.exists():
            os.replace(staging_dir, backup_dir)
        os.replace(tmp_dir, staging_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir.exists() and not staging_dir.exists():
            os.replace(backup_dir, staging_dir)
        raise

    return {
        "factor_id": factor_id,
        "source_dir": str(source),
        "staging_dir": str(staging_dir),
        "rows": _count_parquet_rows(staging_dir),
    }


def staging_has_data(factor_id: str) -> bool:
    """检查 staging 是否已有因子数据。
    
    参数:
        factor_id: 因子唯一标识
    
    返回:
        bool
    """
    staging_dir = _resolve_staging_factor_dir(factor_id)
    return _count_parquet_rows(staging_dir) > 0


def publish_factor_lake(
    *,
    factor_id: str,
    lake_root: str | Path | None = None,
    approve: bool = False,
    sync_from_local: bool = True,
    reconcile: bool = True,
    data_source_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """审批后将 staging 晋升为 published 因子湖。
    
    参数:
        factor_id: 因子唯一标识（可选）
        lake_root: 因子湖根目录（可选）
        approve: 是否显式审批发布（可选）
        sync_from_local: 见函数签名（可选）
        reconcile: 见函数签名（可选）
        data_source_config: 见函数签名（可选）
    
    返回:
        dict[str, Any]
    """
    require_publish_approval(approve=approve)

    from runtime.snapshot_reconcile import reconcile_data_snapshot
    from storage.catalog import FactorCatalog
    from workspace_paths import default_factor_lake_root

    root = Path(lake_root or default_factor_lake_root())
    catalog = FactorCatalog(root / "_catalog.sqlite")
    if catalog.get_factor_info(factor_id) is None:
        raise FactorNotFoundError(f"因子 '{factor_id}' 未在 catalog 注册")

    if reconcile:
        report = reconcile_data_snapshot(
            factor_id=factor_id,
            lake_root=root,
            data_source_config=data_source_config,
            catalog=catalog,
        )
        if not report["ok"]:
            raise ValueError(
                "data_snapshot_id 对账失败: "
                + ", ".join(report["mismatches"])
            )

    sync_summary = None
    if sync_from_local:
        if staging_has_data(factor_id):
            sync_summary = {
                "factor_id": factor_id,
                "skipped": True,
                "reason": "staging already populated (direct upsert path)",
            }
        else:
            sync_summary = sync_local_factor_to_staging(
                factor_id=factor_id,
                lake_root=root,
            )

    from data_access import get_store

    store = get_store()
    runs = catalog.list_runs(factor_id, limit=1)
    run_id = runs[0]["run_id"] if runs else None
    publish_result = store.publish_from_staging(
        "factor_lake_staging",
        "factor_lake",
        factor_id=factor_id,
    )
    return {
        "factor_id": factor_id,
        "approved": True,
        "run_id": run_id,
        "sync": sync_summary,
        "publish": publish_result,
    }
