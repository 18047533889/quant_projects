"""因子湖 staging → published 发布（带人工审批门）。"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from factor_engine.storage.exceptions import FactorNotFoundError


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


def advance_published_watermark(
    *,
    factor_id: str,
    catalog: Any,
    store: Any,
) -> dict[str, Any] | None:
    """publish 成功后推进权威水位线（staging-only 时 defer 的水位线在此 commit）。

    #收官轮 P0（Integration）：staging-only 物化不再在 materialize() 里推进权威
    水位线（只算到 STAGED）；只有 ``publish_from_staging`` 成功晋升后这里才把
    FactorCatalog 的正式水位线推进到 PUBLISHED。发布失败（抛异常）时不调用 → 水位线
    保持原样，增量调度器不会误以为「已正式提交」而 skip。

    返回更新后的 watermark dict；published 目录不可读/无数据时返回 None（水位线不动）。
    """
    lake_dir = store.resolve_dataset_path("factor_lake", factor_id=factor_id)
    if not lake_dir.is_dir():
        return None
    import pyarrow.parquet as pq

    start_dt = end_dt = None
    rows = 0
    for f in sorted(lake_dir.rglob("*.parquet")):
        if f.name.startswith("."):
            continue
        try:
            table = pq.read_table(f, columns=["datetime"])
        except Exception:
            try:
                table = pq.read_table(f)
            except Exception:  # pragma: no cover - 单文件不可读，跳过
                continue
        if "datetime" not in table.column_names:
            continue
        rows += table.num_rows
        try:
            import pyarrow.compute as pac

            extrema = pac.min_max(table.column("datetime"))
            mn = extrema.as_py()["min"]
            mx = extrema.as_py()["max"]
        except Exception:  # pragma: no cover - 无法取 min/max，跳过该文件
            continue
        if mn is None or mx is None:
            continue
        if start_dt is None or mn < start_dt:
            start_dt = mn
        if end_dt is None or mx > end_dt:
            end_dt = mx
    if start_dt is None or end_dt is None:
        return None
    import pandas as pd

    start_date = pd.Timestamp(start_dt).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_dt).strftime("%Y-%m-%d")
    existing = catalog.get_watermark(factor_id) or {}
    if existing.get("start_date") and existing["start_date"] < start_date:
        start_date = existing["start_date"]
    if existing.get("end_date") and existing["end_date"] > end_date:
        end_date = existing["end_date"]
    catalog.update_watermark(
        factor_id=factor_id,
        start_date=start_date,
        end_date=end_date,
        row_count=rows,
    )
    return catalog.get_watermark(factor_id)


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

    from factor_engine.runtime.snapshot_reconcile import reconcile_data_snapshot
    from factor_engine.storage.catalog import FactorCatalog
    from factor_engine.util.workspace_paths import default_factor_lake_root

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
    # #收官轮 P0：publish 是唯一把权威水位线推进到 PUBLISHED 的入口——成功晋升后才
    # 调用；`store.publish_from_staging` 抛异常时不会走到这里，水位线保持原样
    # （staging 写成功但 publish 失败 → 权威水位线 unchanged）。
    publish_result = store.publish_from_staging(
        "factor_lake_staging",
        "factor_lake",
        factor_id=factor_id,
    )
    watermark = advance_published_watermark(
        factor_id=factor_id,
        catalog=catalog,
        store=store,
    )
    return {
        "factor_id": factor_id,
        "approved": True,
        "run_id": run_id,
        "sync": sync_summary,
        "publish": publish_result,
        "watermark": watermark,
    }
