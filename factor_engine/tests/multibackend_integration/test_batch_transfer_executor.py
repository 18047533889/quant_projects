# -*- coding: utf-8 -*-
"""Regression tests for R21-TRANSFER-EXECUTOR.

Verifies that BatchTransferOptimizer maps each TransferTransform to a real
executor (no fake no-op), unknown targets fail closed with TypedTransferError,
batch transfer propagates typed failures instead of swallowing them, the
DuckDB connection is returned as a resident handle, Arrow→Polars goes directly
(not via to_pandas), and semantic preservation is validated after each transfer.

Ported from root tests/multibackend/test_batch_transfer_executor.py; imports
adapted to factor_engine authority (TransferTransform + typed transfer errors
live in runtime.multibackend.batch_transfer_optimizer).
"""
from __future__ import annotations

import os

# 线程环境变量必须在导入 polars/pyarrow 之前设置（单线程，避免 xdist 干扰）。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from factor_engine.runtime.multibackend.batch_transfer_optimizer import (
    BatchTransferOptimizer,
    SemanticMismatchError,
    SemanticSnapshot,
    TransferExecutor,
    TransferInputTypeError,
    TransferTransform,
    TypedTransferError,
    UnknownTransferTargetError,
    _assert_semantic_preservation,
)
from factor_engine.planning.transfer_edge import UnsupportedTransferTransform
from factor_engine.planning.transfer_edge import UnsupportedTransferTransform


@pytest.fixture(autouse=True)
def _thread_env():
    """确保 polars/pyarrow 在单线程下运行（串行 pytest）。"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    yield


def _arrow_table():
    return pa.table(
        {
            "date": pa.array(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "instrument": pa.array([1, 2, 3], type=pa.int64()),
            "value": pa.array([1.5, 2.5, None], type=pa.float64()),
        }
    )


class TestTransferExecutorRealConversions:
    """每个 TransferTransform 必须执行真实转换（非 no-op）。"""

    def test_arrow_to_polars_direct(self):
        """Arrow → Polars 必须直接 pl.from_arrow，不得经 to_pandas 中转。"""
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.ARROW_TO_POLARS, _arrow_table())
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 3)
        # 直接路径：结果应保留 Arrow 的 null（不因 to_pandas 丢失）
        assert out["value"].null_count() == 1

    def test_polars_to_pandas(self):
        ex = TransferExecutor()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = ex.execute(TransferTransform.POLARS_TO_PANDAS, df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_pandas_to_polars(self):
        ex = TransferExecutor()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        out = ex.execute(TransferTransform.PANDAS_TO_POLARS, df)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 2)

    def test_polars_to_numpy(self):
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.POLARS_TO_NUMPY, pl.Series([1, 2, 3]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_numpy_to_polars(self):
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.NUMPY_TO_POLARS, np.array([1, 2, 3]))
        assert isinstance(out, pl.Series)
        assert out.shape == (3,)

    def test_duckdb_to_arrow(self):
        import duckdb

        ex = TransferExecutor()
        conn = duckdb.connect()
        conn.register("t", _arrow_table())
        rel = conn.table("t")
        out = ex.execute(TransferTransform.DUCKDB_TO_ARROW, rel)
        assert isinstance(out, pa.Table)
        assert out.num_rows == 3

    def test_q_to_arrow_accepts_arrow(self):
        """Q→Arrow：已就绪的 Arrow 数据原样返回（真实转换，非 no-op）。"""
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.Q_TO_ARROW, _arrow_table())
        assert isinstance(out, pa.Table)
        assert out.num_rows == 3

    def test_clickhouse_to_arrow(self):
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.CLICKHOUSE_TO_ARROW, _arrow_table())
        assert isinstance(out, pa.Table)
        assert out.num_rows == 3


class TestUnknownTargetFailsClosed:
    """未知 target / 无真实 executor 的 transform 必须抛 TypedTransferError。"""

    def test_unknown_transform_raises(self):
        ex = TransferExecutor()
        with pytest.raises(UnknownTransferTargetError):
            ex.execute(TransferTransform.WIDE_TO_LONG, pd.DataFrame({"a": [1]}))

    def test_unknown_backend_raises_on_register(self):
        b = BatchTransferOptimizer()
        with pytest.raises(UnsupportedTransferTransform):
            b.register_transfer_request(
                "t", pd.DataFrame({"a": [1]}), "src", "totally_unknown"
            )

    def test_wrong_input_type_raises_typed_error(self):
        ex = TransferExecutor()
        with pytest.raises(TransferInputTypeError):
            ex.execute(TransferTransform.ARROW_TO_POLARS, "not-a-table")

    def test_typed_error_is_factor_engine_error(self):
        from factor_engine.runtime.exceptions import FactorEngineError

        assert issubclass(TypedTransferError, FactorEngineError)


class TestBatchTransferPropagatesFailure:
    """批处理不得吞错返回 batch_id；必须向上传播 typed failure。"""

    def test_batch_failure_propagates(self):
        b = BatchTransferOptimizer(batch_threshold=2)
        b.register_transfer_request(
            "t1", _arrow_table(), "duckdb", "polars",
            transform=TransferTransform.ARROW_TO_POLARS,
        )
        with pytest.raises(TransferInputTypeError):
            b.register_transfer_request(
                "t2", "not-a-table", "duckdb", "polars",
                transform=TransferTransform.ARROW_TO_POLARS,
            )
        # 失败被记录，不静默成功
        assert b.metrics().failures >= 1

    def test_batch_success_converts_all(self):
        b = BatchTransferOptimizer(batch_threshold=2)
        b.register_transfer_request(
            "t1", _arrow_table(), "duckdb", "polars",
            transform=TransferTransform.ARROW_TO_POLARS,
        )
        batch_id = b.register_transfer_request(
            "t2", _arrow_table(), "duckdb", "polars",
            transform=TransferTransform.ARROW_TO_POLARS,
        )
        assert batch_id is not None
        assert b.metrics().total_tables_transferred == 2


class TestDuckDBResidentHandle:
    """_batch_to_duckdb 的连接必须作为常驻句柄返回，不泄漏。"""

    def test_injected_conn_is_resident(self):
        import duckdb

        conn = duckdb.connect()
        b = BatchTransferOptimizer(duckdb_conn=conn)
        assert b.resident_duckdb_conn is conn

    def test_conn_usable_after_batch(self):
        import duckdb

        conn = duckdb.connect()
        b = BatchTransferOptimizer(duckdb_conn=conn, batch_threshold=1)
        b.register_transfer_request(
            "t1", _arrow_table(), "duckdb", "arrow",
            transform=TransferTransform.DUCKDB_TO_ARROW,
        )
        # 连接仍存活且可查询（所有权由持方负责，非泄漏）
        assert b.resident_duckdb_conn is conn
        assert conn.execute("SELECT 1").fetchone() == (1,)


class TestSemanticPreservation:
    """传输后必须校验语义保留（date/instrument/order/dtype/null/timezone/grain）。"""

    def test_arrow_to_pandas_preserves_null_and_rows(self):
        ex = TransferExecutor()
        out = ex.execute(TransferTransform.ARROW_TO_PANDAS, _arrow_table())
        assert out.shape == (3, 3)
        assert out["value"].isna().sum() == 1

    def test_semantic_snapshot_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
        snap = SemanticSnapshot.from_dataframe(df)
        assert snap.num_rows == 3
        assert snap.num_cols == 2
        assert snap.null_counts == (1, 0)

    def test_category_to_numeric_flagged(self):
        before = SemanticSnapshot.from_dataframe(
            pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
        )
        after = SemanticSnapshot.from_dataframe(pd.DataFrame({"c": [1, 2, 1]}))
        with pytest.raises(SemanticMismatchError):
            _assert_semantic_preservation(
                before, after, TransferTransform.ARROW_TO_PANDAS
            )

    def test_row_count_mismatch_flagged(self):
        before = SemanticSnapshot.from_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
        after = SemanticSnapshot.from_dataframe(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(SemanticMismatchError):
            _assert_semantic_preservation(
                before, after, TransferTransform.PANDAS_TO_POLARS
            )

    def test_timezone_mismatch_flagged(self):
        before = SemanticSnapshot.from_dataframe(
            pd.DataFrame(
                {"t": pd.date_range("2024-01-01", periods=3, tz="Asia/Shanghai")}
            )
        )
        after = SemanticSnapshot.from_dataframe(
            pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3)})
        )
        with pytest.raises(SemanticMismatchError):
            _assert_semantic_preservation(
                before, after, TransferTransform.PANDAS_TO_POLARS
            )

    def test_grain_mismatch_flagged(self):
        before_df = pd.DataFrame({"a": [1, 2, 3]})
        before_df.attrs = {"grain": "daily"}
        after_df = pd.DataFrame({"a": [1, 2, 3]})
        after_df.attrs = {"grain": "minute"}
        before = SemanticSnapshot.from_dataframe(before_df)
        after = SemanticSnapshot.from_dataframe(after_df)
        with pytest.raises(SemanticMismatchError):
            _assert_semantic_preservation(
                before, after, TransferTransform.PANDAS_TO_POLARS
            )
