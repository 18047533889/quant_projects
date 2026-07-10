# -*- coding: utf-8
"""因子结果规范化：Parquet / ClickHouse / staging 共享的 DQ + 清洗 + 长表转换。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from storage.catalog import compute_ir_hash
from storage.materializer import MaterializeMetadata, ParquetMaterializer


def prepare_factor_dataframe(
    result: pd.Series,
    *,
    ir_node: Any | None = None,
    ast_hash: str | None = None,
    data_snapshot_id: str | None = None,
    dq_check: bool = False,
    dq_strict: bool = True,
    dq_thresholds=None,
    preserve_invalid_rows: bool = False,
    value_dtype: str = "float32",
    write_metadata: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """规范化因子结果为长表并可选执行 DQ。
    
    参数:
        result: 因子计算结果 Series
        ir_node: 因子 IR 树根节点（可选）
        ast_hash: 因子 AST 哈希（可选）
        data_snapshot_id: 见函数签名（可选）
        dq_check: 见函数签名（可选）
        dq_strict: 见函数签名（可选）
        dq_thresholds: 见函数签名（可选）
        preserve_invalid_rows: 见函数签名（可选）
        value_dtype: 见函数签名（可选）
        write_metadata: 见函数签名（可选）
    
    返回:
        tuple[pd.DataFrame, dict[str, Any] | None, str]
    """
    if ast_hash is None:
        if ir_node is not None:
            ast_hash = compute_ir_hash(ir_node)
        else:
            ast_hash = "__no_hash_provided__"

    dq_report = None
    if dq_check:
        from runtime.dq_gates import assert_factor_dq

        dq_report = assert_factor_dq(
            result,
            thresholds=dq_thresholds,
            raise_on_fail=dq_strict,
            preserve_invalid_rows=preserve_invalid_rows,
        )

    meta = None
    if write_metadata:
        meta = MaterializeMetadata(
            calc_time=datetime.now(timezone.utc).isoformat(),
            factor_version=ast_hash[:16],
            data_snapshot_id=data_snapshot_id,
        )

    df = ParquetMaterializer._normalize_to_long_table(
        result,
        metadata=meta,
        value_dtype=value_dtype,
    )
    df = ParquetMaterializer._clean(df, preserve_invalid_rows=preserve_invalid_rows)
    dq_dict = dq_report.to_dict() if dq_report is not None else None
    return df, dq_dict, ast_hash
