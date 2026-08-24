"""因子写目标 Protocol：local / staging / clickhouse 统一契约。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FactorWriteTarget(Protocol):
    """因子落盘目标协议。
    
    参数:
        无
    """

    name: str

    def write_factor_frame(
        self,
        factor_id: str,
        frame: pd.DataFrame,
        *,
        upsert_on: list[str] | None = None,
        partition_by: list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """写入已规范化的长表 DataFrame（datetime, asset, value + 可选 metadata）。
        
        参数:
            factor_id: 因子唯一标识
            frame: 长表 DataFrame
            upsert_on: 见函数签名（可选）
            partition_by: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """


class LocalParquetWriteTarget:
    """写入本地因子湖分区 Parquet。
    
    参数:
        lake_root: 因子湖根目录（可选）
    """

    name = "local"

    def __init__(self, lake_root: str | Path | None = None) -> None:
        """初始化实例。
        
        参数:
            lake_root: 因子湖根目录（可选）
        
        返回:
            无
        """
        from factor_engine.storage.materializer import ParquetMaterializer

        self._materializer = ParquetMaterializer(lake_root=lake_root)

    @property
    def lake_root(self) -> Path:
        """lake_root。
        
        参数:
            无
        
        返回:
            Path
        """
        return self._materializer.lake_root

    def write_factor_frame(
        self,
        factor_id: str,
        frame: pd.DataFrame,
        *,
        upsert_on: list[str] | None = None,
        partition_by: list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """write_factor_frame。

        参数:
            factor_id: 因子唯一标识
            frame: 长表 DataFrame
            upsert_on: 见函数签名（可选）
            partition_by: 见函数签名（可选）

        返回:
            dict[str, Any]

        #收官轮 P0：production 禁止 direct-local 因子湖直写——绕过 DataAccess
        staging→publish 的原子发布/journal/快照 manifest，直写文件时崩溃/半写
        无恢复。production 必须走 ``staging`` 写目标 + ``publish_factor_lake``
        审批发布。research 保留（legacy 快速落盘路径）。
        """
        try:
            from factor_engine.runtime.production_policy import is_production_mode

            if is_production_mode():
                raise ValueError(
                    f"factor_id={factor_id!r}: production 禁止 direct-local "
                    "factor-lake write（绕过 DataAccess staging→publish 原子发布/"
                    "snapshot manifest）。请用 staging 写目标 + publish_factor_lake。"
                )
        except ValueError:
            raise
        except Exception:
            pass
        del upsert_on, params
        from factor_engine.storage.partition_policy import PartitionPolicy, attach_partition_columns, iter_partition_groups

        df = frame.copy()
        if df.empty:
            return {"factor_id": factor_id, "rows_written": 0, "target": self.name}
        df["asset"] = df["asset"].astype("string")
        policy = PartitionPolicy.from_config(partition_columns=partition_by)
        work_df = attach_partition_columns(df, policy)
        factor_dir = self._materializer.lake_root / "factors" / str(factor_id)
        rows = 0
        for part_values, grp in iter_partition_groups(work_df, policy):
            self._materializer._upsert_partition(
                factor_dir,
                part_values,
                grp.reset_index(drop=True),
                policy=policy,
            )
            rows += len(grp)
        return {
            "factor_id": factor_id,
            "rows_written": rows,
            "target": self.name,
            "lake_root": str(self._materializer.lake_root),
        }


class StagingWriteTarget:
    """经 data_access upsert 写入 staging。
    
    参数:
        dataset: data_access 数据集名称（可选）
    """

    name = "staging"

    def __init__(self, dataset: str = "factor_lake_staging") -> None:
        """初始化实例。
        
        参数:
            dataset: data_access 数据集名称（可选）
        
        返回:
            无
        """
        self.dataset = str(dataset or "factor_lake_staging")

    def write_factor_frame(
        self,
        factor_id: str,
        frame: pd.DataFrame,
        *,
        upsert_on: list[str] | None = None,
        partition_by: list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """write_factor_frame。
        
        参数:
            factor_id: 因子唯一标识
            frame: 长表 DataFrame
            upsert_on: 见函数签名（可选）
            partition_by: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """
        import pyarrow as pa

        from data_access import get_store

        out = frame.copy()
        if "year" not in out.columns:
            out["year"] = pd.to_datetime(out["datetime"]).dt.year.astype("int32")
        keys = upsert_on or ["datetime", "asset"]
        parts = partition_by or ["year"]
        table = pa.Table.from_pandas(out, preserve_index=False)
        store = get_store()
        result = store.upsert(
            self.dataset,
            table,
            upsert_on=keys,
            partition_by=parts,
            factor_id=factor_id,
            **params,
        )
        return {"dataset": self.dataset, "factor_id": factor_id, "target": self.name, **result}


class ClickHouseWriteTarget:
    """经 ClickHouseMaterializer 写入 ClickHouse。
    
    参数:
        table: ClickHouse 表名（可选）
        host: 见函数签名（可选）
        port: 见函数签名（可选）
        database: 见函数签名（可选）
        username: 见函数签名（可选）
        password: 见函数签名（可选）
        secure: 见函数签名（可选）
    """

    name = "clickhouse"

    def __init__(
        self,
        *,
        table: str = "factor_values",
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
    ) -> None:
        """初始化实例。
        
        参数:
            table: ClickHouse 表名（可选）
            host: 见函数签名（可选）
            port: 见函数签名（可选）
            database: 见函数签名（可选）
            username: 见函数签名（可选）
            password: 见函数签名（可选）
            secure: 见函数签名（可选）
        
        返回:
            无
        """
        self.table = table
        self._ch_overrides = {
            k: v
            for k, v in {
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "secure": secure,
            }.items()
            if v is not None
        }

    def write_factor_frame(
        self,
        factor_id: str,
        frame: pd.DataFrame,
        *,
        upsert_on: list[str] | None = None,
        partition_by: list[str] | None = None,
        factor_version: str = "",
        data_snapshot_id: str | None = None,
        ensure_table: bool = True,
        **params: Any,
    ) -> dict[str, Any]:
        """write_factor_frame。
        
        参数:
            factor_id: 因子唯一标识
            frame: 长表 DataFrame
            upsert_on: 见函数签名（可选）
            partition_by: 见函数签名（可选）
            factor_version: 见函数签名（可选）
            data_snapshot_id: 见函数签名（可选）
            ensure_table: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """
        del upsert_on, partition_by, params
        from factor_engine.storage.clickhouse_materializer import ClickHouseMaterializer

        if frame.empty:
            return {"factor_id": factor_id, "rows_written": 0, "target": self.name}
        work = frame.copy()
        work = work.set_index(["datetime", "asset"])
        series = work["value"]
        series.index = series.index.set_names(["timestamp", "instrument"])
        ch = ClickHouseMaterializer(table=self.table, **self._ch_overrides)
        summary = ch.materialize(
            factor_id=factor_id,
            result=series,
            factor_version=factor_version,
            data_snapshot_id=data_snapshot_id,
            ensure_table=ensure_table,
        )
        return {
            "factor_id": summary.factor_id,
            "rows_written": summary.rows_written,
            "table": summary.table,
            "database": summary.database,
            "target": self.name,
        }

    def write_factor_series(
        self,
        factor_id: str,
        result: pd.Series,
        *,
        factor_version: str = "",
        data_snapshot_id: str | None = None,
        ensure_table: bool = True,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_metadata: bool = True,
        **params: Any,
    ) -> dict[str, Any]:
        """从 MultiIndex Series 写入 ClickHouse（dual_write / 纯 CH 路径）。
        
        参数:
            factor_id: 因子唯一标识
            result: 因子计算结果 Series
            factor_version: 见函数签名（可选）
            data_snapshot_id: 见函数签名（可选）
            ensure_table: 见函数签名（可选）
            dq_check: 见函数签名（可选）
            dq_strict: 见函数签名（可选）
            dq_thresholds: 见函数签名（可选）
            preserve_invalid_rows: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
            write_metadata: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """
        del params
        from factor_engine.storage.clickhouse_materializer import ClickHouseMaterializer

        if getattr(result, "empty", False):
            return {"factor_id": factor_id, "rows_written": 0, "target": self.name}
        ch = ClickHouseMaterializer(table=self.table, **self._ch_overrides)
        summary = ch.materialize(
            factor_id=factor_id,
            result=result,
            factor_version=factor_version,
            data_snapshot_id=data_snapshot_id,
            ensure_table=ensure_table,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
        )
        out = {
            "factor_id": summary.factor_id,
            "rows_written": summary.rows_written,
            "table": summary.table,
            "database": summary.database,
            "target": self.name,
        }
        if summary.dq_report is not None:
            out["dq_report"] = summary.dq_report
        return out


#: #收官轮 P0（Integration）：write_target 严格枚举。未知/拼写错误的目标
#: （如 ``"locla"``）必须在**任何 side effect 之前**拒绝——否则 ``write_local=False
#: ∧ write_staging=False`` 却照样注册因子/计算 active_df/更新水位线/返回
#: rows_written>0，出现「metadata 说已提交、物理数据根本不存在」。
_ALLOWED_WRITE_TARGETS = frozenset({
    "local",
    "parquet",
    "staging",
    "factor_lake_staging",
    "both",
    "clickhouse",
    "ch",
    "staging_clickhouse",
})


def normalize_write_target(name: str) -> dict[str, Any]:
    """严格解析 ``write_target`` → ``{local, staging, clickhouse}`` 布尔集合。

    - ``local`` / ``parquet`` → 本地 lake
    - ``staging`` / ``factor_lake_staging`` / ``staging:<dataset>`` → staging 数据集
    - ``both`` → local + staging 双写
    - ``clickhouse`` / ``ch`` → ClickHouse
    - ``staging_clickhouse`` → staging + ClickHouse

    未知 / 拼写错误目标 → ``ValueError``（在注册因子 / 写文件 / 更新水位线等任何
    side effect 之前 fail-closed）。
    """
    raw = str(name or "local").strip().lower()
    if not raw:
        raw = "local"
    if raw.startswith("staging:"):
        dataset = raw.split(":", 1)[1].strip()
        if not dataset:
            raise ValueError(f"write_target 'staging:<dataset>' 需要非空 dataset，收到 {name!r}")
        return {
            "local": False,
            "staging": True,
            "clickhouse": False,
            "staging_dataset": dataset,
        }
    if raw in {"local", "parquet"}:
        return {"local": True, "staging": False, "clickhouse": False}
    if raw in {"staging", "factor_lake_staging"}:
        return {"local": False, "staging": True, "clickhouse": False}
    if raw == "both":
        return {"local": True, "staging": True, "clickhouse": False}
    if raw in {"clickhouse", "ch"}:
        return {"local": False, "staging": False, "clickhouse": True}
    if raw == "staging_clickhouse":
        return {"local": False, "staging": True, "clickhouse": True}
    raise ValueError(
        f"未知 FactorWriteTarget {name!r}；可用 local / staging / staging:<dataset> "
        f"/ both / clickhouse / staging_clickhouse"
    )


def resolve_write_target(
    name: str,
    *,
    lake_root: str | Path | None = None,
    staging_dataset: str = "factor_lake_staging",
    clickhouse_table: str = "factor_values",
    **ch_overrides: Any,
) -> FactorWriteTarget:
    """按名称解析因子写目标实例。

    参数:
        name: 逻辑列名
        lake_root: 因子湖根目录（可选）
        staging_dataset: 见函数签名（可选）
        clickhouse_table: 见函数签名（可选）

    返回:
        FactorWriteTarget
    """
    # #收官轮 P0：统一走严格枚举（未知目标在这里拒绝），不再逐分支 if 放行。
    flags = normalize_write_target(name)
    if flags["local"] and not flags["staging"] and not flags["clickhouse"]:
        return LocalParquetWriteTarget(lake_root=lake_root)
    if flags["staging"] and not flags["local"] and not flags["clickhouse"]:
        return StagingWriteTarget(flags.get("staging_dataset") or staging_dataset)
    if flags["clickhouse"] and not flags["local"] and not flags["staging"]:
        return ClickHouseWriteTarget(table=clickhouse_table, **ch_overrides)
    # ``both`` / ``staging_clickhouse`` 是多目标组合，单 target 实例无法表达，
    # 由 ParquetMaterializer / orchestrator 消费 normalize_write_target 自行编排。
    raise ValueError(
        f"write_target {name!r} 是多目标组合（both / staging_clickhouse），"
        "不能解析为单个 FactorWriteTarget；请由 materialize orchestrator 处理。"
    )
