# -*- coding: utf-8
"""ClickHouse 双写服务：从 FactorEngine 抽离，供 materialize / reconcile 共用。

NEW-P0-59 / NEW-P0-60 / NEW-P1-61（R13 审计收口）：

- ``DualWriteRequest`` 携带 **ONE** unified ``semantic_identity_digest``（full
  SHA-256），Parquet / catalog / ClickHouse / matrix 从它派生各自的 16 位显示
  前缀——杜绝 ``ast_hash[:16]`` 与 ``identity_digest[:16]`` 的跨存储 split-brain；
- ``MaterializationDelta`` 统一携带 ``upserts`` + ``tombstones`` +
  ``semantic_identity_digest`` + ``generation``，使 Parquet 的 deleted_keys
  （NaN/deleted）能传播到 ClickHouse；
- ``generation`` / ``transaction_id`` 是 Parquet staging、ClickHouse 写入、
  watermark commit 共享的幂等重试 token。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from logging_utils import get_logger

logger = get_logger("runtime.dual_write_service")


# ---------------------------------------------------------------------------
# 统一双写增量 / 请求契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializationDelta:
    """统一物化增量：所有 sink（Parquet / ClickHouse / matrix）消费同一份。

    NEW-P0-60：Parquet 的 tombstone（deleted_keys → NaN/deleted）必须传播到
    ClickHouse——双写接口只收 ``upserts`` 时历史删除永远不会到 CH。统一 delta
    携带 ``tombstones`` 后，CH 把删除键写成 NaN 行覆盖旧有限值。

    NEW-P1-61：``generation`` / ``transaction_id`` 是 Parquet staging、
    ClickHouse 写入、watermark commit 共享的幂等重试 token——「CH 成功但
    watermark commit 失败」后按同一 token 重试不会重复写。
    """

    upserts: Any  # pd.Series, MultiIndex(timestamp, instrument)
    tombstones: tuple = ()
    semantic_identity_digest: str | None = None  # full SHA-256
    generation: str | None = None
    transaction_id: str | None = None
    data_snapshot_id: str | None = None

    @property
    def has_tombstones(self) -> bool:
        return bool(self.tombstones)

    def factor_version(self, ast_hash: str) -> str:
        """两个 sink 共用的 16 位显示版本（从 full digest 派生）。"""
        return str(self.semantic_identity_digest or ast_hash)[:16]


@dataclass(frozen=True)
class DualWriteRequest:
    """一次双写请求：携带 ONE unified 语义身份 digest（full）+ tombstone + token。

    NEW-P0-59：Parquet 与 ClickHouse 的 ``factor_version`` 必须同源——都从这
    一个 full digest 派生 16 位前缀，杜绝 ``ast_hash[:16]`` 与
    ``identity_digest[:16]`` 的 split-brain（universe=CSI300 vs CSI500 必须给
    出不同且一致的两侧版本）。
    """

    factor_id: str
    upserts: Any  # pd.Series, MultiIndex(timestamp, instrument)
    ast_hash: str
    write_target: str = "clickhouse"
    semantic_identity_digest: str | None = None
    tombstones: tuple = ()
    generation: str | None = None
    transaction_id: str | None = None
    data_snapshot_id: str | None = None
    clickhouse_table: str | None = None

    @property
    def factor_version(self) -> str:
        """权威 16 位显示版本：full semantic digest 前缀，缺省 ast_hash 前缀。"""
        return str(self.semantic_identity_digest or self.ast_hash)[:16]

    def to_delta(self) -> MaterializationDelta:
        return MaterializationDelta(
            upserts=self.upserts,
            tombstones=self.tombstones,
            semantic_identity_digest=self.semantic_identity_digest,
            generation=self.generation,
            transaction_id=self.transaction_id,
            data_snapshot_id=self.data_snapshot_id,
        )


def _series_with_tombstones(
    result: pd.Series,
    tombstones: tuple,
) -> pd.Series:
    """把 (datetime, asset) 删除键转成 NaN 行，追加到结果 Series。

    与 Parquet ``_append_tombstones`` 同构：CH 的 ReplacingMergeTree 对
    (timestamp, instrument, factor_id) dedup 取新行，NaN 行会覆盖旧有限值，
    读取端把该格映射为 NaN/deleted。
    """
    if not tombstones:
        return result
    names = getattr(result.index, "names", None)
    try:
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(k[0]), str(k[1])) for k in tombstones],
            names=names or ["timestamp", "instrument"],
        )
    except Exception as exc:  # pragma: no cover - 删除键格式非法
        raise ValueError(
            f"tombstones 必须是 (datetime, asset) 二元组，收到 {tombstones!r}: {exc}"
        ) from exc
    dtype = getattr(result, "dtype", "float32")
    tomb = pd.Series(np.nan, index=idx, dtype=dtype)
    merged = pd.concat([result, tomb])
    # dedup keep="last"：同一键的 NaN 覆盖旧有限值（与 Parquet upsert 同语义）。
    if merged.index.duplicated().any():
        merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def append_clickhouse_to_summary(
    summary: dict[str, Any],
    *,
    factor_id: str,
    result: pd.Series,
    ast_hash: str,
    write_target: str,
    factor_version: str | None = None,
    data_snapshot_id: str | None = None,
    clickhouse_table: str | None = None,
    preserve_invalid_rows: bool = False,
    value_dtype: str = "float32",
    write_metadata: bool = True,
    ensure_table: bool = True,
    dq_check: bool = False,
    dq_strict: bool = True,
    dq_thresholds=None,
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_username: str | None = None,
    ch_password: str | None = None,
    ch_secure: bool | None = None,
    semantic_identity_digest: str | None = None,
    tombstones: list[tuple] | tuple | None = None,
    generation: str | None = None,
    transaction_id: str | None = None,
    delta: MaterializationDelta | None = None,
) -> dict[str, Any]:
    """Parquet/staging 落盘后追加 ClickHouse 写入。

    #收官轮 P0：``factor_version`` 由 orchestrator 传入**同一份** canonical
    ``semantic_identity.identity_digest()[:16]``（Parquet/catalog/ClickHouse/matrix
    共用）。缺省才回退 ``ast_hash[:16]`` —— 否则公式不变但 universe/freq/PIT/source
    变化时，Parquet=新 version、catalog=新 version、ClickHouse=旧 version，跨存储
    split-brain，ReplacingMergeTree 会错误地以旧 version 新 row 覆盖新 version。

    NEW-P0-59：优先从统一 ``semantic_identity_digest``（full SHA-256）派生显示
    前缀；NEW-P0-60：``tombstones`` / ``delta`` 携带的删除键先并入 ``result``，
    使 CH 与 Parquet 一样把历史删除映射为 NaN；NEW-P1-61：``generation`` /
    ``transaction_id`` 透传到 summary，供 watermark commit 幂等重试。
    """
    from storage.exceptions import DualWriteError
    from storage.write_targets import ClickHouseWriteTarget

    effective_delta = delta or MaterializationDelta(
        upserts=result,
        tombstones=tuple(tombstones or ()),
        semantic_identity_digest=semantic_identity_digest,
        generation=generation,
        transaction_id=transaction_id,
        data_snapshot_id=data_snapshot_id,
    )
    # NEW-P0-59：统一版本来源——full digest 前缀 > 显式 factor_version > ast_hash
    # 前缀。调用方只要传了统一 digest（DualWriteRequest / delta），CH 一律从它
    # 派生 16 位显示版本，绝不回退到可能失配的 ``factor_version`` / ``ast_hash``。
    full_digest = effective_delta.semantic_identity_digest
    if full_digest:
        ch_version = str(full_digest)[:16]
    else:
        ch_version = str(factor_version or ast_hash)[:16]
    # NEW-P0-60：把删除键先转成 NaN 行再写 CH（与 Parquet tombstone 同构）。
    ch_result_series = _series_with_tombstones(
        effective_delta.upserts, effective_delta.tombstones
    )

    target = ClickHouseWriteTarget(
        table=clickhouse_table or "factor_values",
        host=ch_host,
        port=ch_port,
        database=ch_database,
        username=ch_username,
        password=ch_password,
        secure=ch_secure,
    )
    partial = dict(summary)
    partial["write_target"] = str(write_target).lower()
    partial["primary_write_completed"] = True
    try:
        ch_result = target.write_factor_series(
            factor_id,
            ch_result_series,
            factor_version=ch_version,
            data_snapshot_id=data_snapshot_id,
            ensure_table=ensure_table,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
        )
    except Exception as exc:
        partial["partial_write"] = True
        partial["clickhouse_error"] = str(exc)
        raise DualWriteError(
            f"ClickHouse 写入失败（主存储可能已成功）: factor_id={factor_id}, error={exc}",
            summary=partial,
            cause=exc,
        ) from exc

    merged = dict(partial)
    merged["clickhouse"] = {
        "factor_id": ch_result["factor_id"],
        "table": ch_result["table"],
        "rows_written": ch_result["rows_written"],
        "database": ch_result["database"],
    }
    if ch_result.get("dq_report") is not None:
        merged["clickhouse_dq_report"] = ch_result["dq_report"]
    # NEW-P1-61：统一 generation/transaction token 贯穿到 commit 路径。
    if effective_delta.generation is not None:
        merged["run_generation"] = effective_delta.generation
    if effective_delta.transaction_id is not None:
        merged["transaction_id"] = effective_delta.transaction_id
    merged["partial_write"] = False
    return merged


def dual_write_clickhouse(
    materializer,
    summary: dict[str, Any],
    *,
    factor_id: str,
    result: pd.Series,
    ast_hash: str,
    write_target: str,
    factor_version: str | None = None,
    data_snapshot_id: str | None = None,
    clickhouse_table: str | None = None,
    preserve_invalid_rows: bool = False,
    value_dtype: str = "float32",
    write_metadata: bool = True,
    ensure_table: bool = True,
    dq_check: bool = False,
    dq_strict: bool = True,
    dq_thresholds=None,
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_username: str | None = None,
    ch_password: str | None = None,
    ch_secure: bool | None = None,
    semantic_identity_digest: str | None = None,
    tombstones: list[tuple] | tuple | None = None,
    generation: str | None = None,
    transaction_id: str | None = None,
    delta: MaterializationDelta | None = None,
    request: DualWriteRequest | None = None,
) -> dict[str, Any]:
    """ClickHouse 双写；defer watermark 时成功 commit、失败 abort。

    NEW-P0-59/P0-60/P1-61：可整体传 ``request: DualWriteRequest``（携带统一
    full digest + tombstones + generation/transaction token），或分别传
    ``semantic_identity_digest`` / ``tombstones`` / ``generation`` /
    ``transaction_id``。
    """
    from storage.exceptions import DualWriteError

    req = request
    if req is not None:
        factor_id = req.factor_id
        result = req.upserts
        ast_hash = req.ast_hash
        write_target = req.write_target
        data_snapshot_id = data_snapshot_id or req.data_snapshot_id
        delta = req.to_delta()
    deferred = bool(summary.get("watermark_deferred"))
    try:
        merged = append_clickhouse_to_summary(
            summary,
            factor_id=factor_id,
            result=result,
            ast_hash=ast_hash,
            factor_version=factor_version,
            write_target=write_target,
            data_snapshot_id=data_snapshot_id,
            clickhouse_table=clickhouse_table,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
            ensure_table=ensure_table,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            semantic_identity_digest=semantic_identity_digest,
            tombstones=tombstones,
            generation=generation,
            transaction_id=transaction_id,
            delta=delta,
        )
    except DualWriteError as exc:
        if deferred:
            materializer.abort_deferred_materialization(
                exc.summary or summary,
                error=str(exc.cause or exc),
            )
        raise

    if deferred:
        return materializer.commit_deferred_materialization(merged)
    return merged
