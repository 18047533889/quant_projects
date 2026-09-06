# -*- coding: utf-8 -*-
"""R14 #3：``execute_materialize`` 构造**唯一** ``MaterializationDelta`` 供 CH 消费。

外部 AI 复查第二轮 P0：``MaterializationDelta``（upserts/tombstones/
semantic_identity_digest/generation/transaction_id）底层能力已就绪，但实际
``execute_materialize`` 调 ClickHouse 时没传 tombstones——源行删除后 Parquet
写了 tombstone、ClickHouse 旧有限值仍存在，跨存储 split-brain。

本文件断言 orchestrator 主链：``materialize_incremental(deleted_keys=...)`` →
``execute_materialize`` → 单个 delta → CH 收到的 Series 中删除键为 NaN。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.api import rank
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _data() -> dict[str, pd.Series]:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {"close": pd.Series(np.arange(6, dtype=float) + 1.0, index=idx)}


def _fake_ch_target(captured: dict):
    """捕获 CH 收到的 series / factor_version 的假 ClickHouseWriteTarget。"""

    class _Fake:
        def __init__(self, **kw):
            captured["kwargs"] = kw

        def write_factor_series(self, factor_id, series, factor_version=None,
                                data_snapshot_id=None, **kw):
            captured["series"] = series
            captured["factor_version"] = factor_version
            return {
                "factor_id": factor_id,
                "table": "t",
                "rows_written": len(series),
                "database": "default",
            }

    return _Fake


def test_r14_execute_materialize_delta_tombstones_reach_ch(tmp_path, monkeypatch):
    import factor_engine.storage.write_targets as write_targets

    captured: dict = {}
    monkeypatch.setattr(
        write_targets, "ClickHouseWriteTarget", _fake_ch_target(captured)
    )
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f = Factor(name="a", expr=rank(col("close")))
    deleted_key = (pd.Timestamp("2024-01-02"), "A")
    out = eng.materialize_incremental(
        f,
        factor_id="a",
        since="2024-01-01",
        end_date="2024-01-03",
        lake_root=tmp_path / "lake",
        write_target="clickhouse",
        deleted_keys=[deleted_key],
    )
    # CH 确实被调用，且收到含 tombstone 的 series
    assert "series" in captured
    ch_series = captured["series"]
    # (2024-01-02, A) 在 CH 侧是 NaN（tombstone 覆盖旧有限值，与 Parquet 同构）
    assert ch_series.loc[deleted_key] != ch_series.loc[deleted_key]  # NaN
    # 其余键保持有限值（未受 tombstone 波及）
    assert ch_series.loc[(pd.Timestamp("2024-01-03"), "B")] > 0
    # 同一事件正常 commit
    assert out["materialization"]["primary_write_completed"] is True


def test_r14_execute_materialize_delta_parity_parquet_and_ch(tmp_path, monkeypatch):
    """CH 版本与 catalog/Parquet 同源：delta 携带 full semantic digest → CH 侧
    16 位前缀 == catalog 注册的 factor_version（不是 ast_hash 前缀）。

    R14 #3 + #4 联查：orchestrator 只构造一个 delta，CH 从 delta 的
    ``semantic_identity_digest`` 派生显示版本——若 orchestrator 没把 digest 塞进
    delta（仍只传 ``factor_version``），CH 侧可能与 catalog 的权威版本失配。
    """
    import factor_engine.storage.write_targets as write_targets

    captured: dict = {}
    monkeypatch.setattr(
        write_targets, "ClickHouseWriteTarget", _fake_ch_target(captured)
    )
    eng = FactorEngine(
        backend=PandasBackend(), data_source=InMemorySeriesSource(data=_data())
    )
    f = Factor(name="a", expr=rank(col("close")))
    eng.materialize_incremental(
        f,
        factor_id="a",
        since="2024-01-01",
        end_date="2024-01-03",
        lake_root=tmp_path / "lake",
        write_target="clickhouse",
    )
    ch_version = captured["factor_version"]
    assert isinstance(ch_version, str) and len(ch_version) == 16
    # catalog 权威存 full digest；CH 显示版本必须 == catalog full digest 的 16 位
    # 前缀——即 CH 从 delta 携带的同一份 full semantic digest 派生，不是 ast_hash。
    from factor_engine.storage.materializer import ParquetMaterializer

    catalog = ParquetMaterializer(lake_root=tmp_path / "lake").catalog
    cat_version = catalog.get_factor_info("a")["factor_version"]
    assert len(cat_version) == 64  # full SHA-256 digest 入 catalog
    assert ch_version == cat_version[:16]  # CH 与 catalog 同源
