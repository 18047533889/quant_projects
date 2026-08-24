# -*- coding: utf-8 -*-
"""R44 节点级增量：Arrow 状态块存储（``runtime.state_block_store``）测试。

覆盖：
1. ``StateBlock`` 与逐 instrument ``StateCheckpoint`` 的互转 + Arrow IPC 往返；
2. ``StateBlockStore`` 单块读写 / 缺失 / schema 不匹配 fail-closed；
3. ``write_generation`` 多分片原子性（中途崩溃 → 读者仍见完整旧代）；
4. ``StatefulCheckpointStoreAdapter`` 作为现有 JSON 存储的 drop-in（bootstrap
   后 resume 与 full-history 完全一致）；
5. ``garbage_collect`` 只保留最新一代。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.state_block_store import (
    StateBlock,
    StateBlockStore,
    StatefulCheckpointStoreAdapter,
)
from factor_engine.runtime.stateful_incremental import try_stateful_segmented_incremental
from stateful_runtime import execute_stateful_segment
from factor_engine.ir.nodes import IRNode


class _PanelSource:
    """Minimal DataSource stub returning MultiIndex (timestamp, instrument) series."""

    def __init__(self, panel: pd.DataFrame):
        self._panel = panel

    def load_column(self, name: str) -> pd.Series:
        stacked = self._panel.stack()
        stacked.index = stacked.index.set_names(["timestamp", "instrument"])
        return stacked.rename(name)


def _ema_ir(span: int = 3) -> IRNode:
    return IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})


def _close_panel(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "A": np.arange(n, dtype=float) + 10.0,
            "B": np.arange(n, dtype=float) * 2.0 + 1.0,
        },
        index=idx,
    )


def _build_checkpoints(panel: pd.DataFrame) -> list:
    """对每个 instrument 跑一次 ``execute_stateful_segment`` 生成真实 checkpoint。"""
    checkpoints = []
    for inst in panel.columns:
        res = execute_stateful_segment(
            "ts_ema", {"x": panel[inst].to_numpy(dtype=float)},
            timestamps=panel.index, instrument=inst,
            input_identity={"dataset": "unit"}, params={"span": 3},
            starts_at_dataset_origin=True,
        )
        checkpoints.append(res.checkpoint)
    return checkpoints


# ---------------------------------------------------------------------------
# 1. StateBlock 互转 + Arrow IPC 往返
# ---------------------------------------------------------------------------
def test_state_block_roundtrip_preserves_state_and_identity() -> None:
    panel = _close_panel(40)
    checkpoints = _build_checkpoints(panel)
    block = StateBlock.from_checkpoints("f1::ts_ema", checkpoints)

    # 身份字段保留。
    assert block.as_of == checkpoints[0].as_of
    assert block.schema_version == checkpoints[0].state_schema_version
    assert block.semantic_version == checkpoints[0].semantic_version
    assert block.instruments == ("A", "B")
    assert block.input_fingerprints == tuple(c.input_fingerprint for c in checkpoints)

    # Arrow IPC 往返。
    table = block.to_arrow()
    restored = StateBlock.from_arrow(table)
    assert restored.as_of == block.as_of
    assert restored.schema_version == block.schema_version
    assert restored.instruments == block.instruments
    assert restored.input_fingerprints == block.input_fingerprints
    assert restored.semantic_version == block.semantic_version

    # 解回逐 instrument checkpoint，state 值与原始一致。
    rebuilt = restored.to_checkpoints()
    assert len(rebuilt) == len(checkpoints)
    for original, back in zip(checkpoints, rebuilt):
        assert back.instrument == original.instrument
        assert back.as_of == original.as_of
        assert back.state["last_timestamp"] == original.state["last_timestamp"]
        # ts_ema 的 ema 是 EwmState dict —— 逐字段比对。
        for key in ("weighted_avg", "old_wt", "valid_count"):
            np.testing.assert_allclose(
                back.state["ema"][key], original.state["ema"][key], equal_nan=True
            )


def test_state_block_union_of_missing_fields() -> None:
    # 两个 instrument 的 state 字段不同 → 并集，缺失字段为 None。
    from stateful_contract import StateCheckpoint

    ck_a = StateCheckpoint(
        operator="ts_ema", instrument="A", as_of="2024-01-01T00:00:00+00:00",
        state_schema_version="ema_state.v2", semantic_version="2.0",
        input_fingerprint="fp", state={"ema": {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 1}, "last_timestamp": "2024-01-01T00:00:00+00:00"},
    )
    ck_b = StateCheckpoint(
        operator="ts_ema", instrument="B", as_of="2024-01-01T00:00:00+00:00",
        state_schema_version="ema_state.v2", semantic_version="2.0",
        input_fingerprint="fp", state={"ema": {"weighted_avg": 2.0, "old_wt": 1.0, "valid_count": 2}, "extra": 5, "last_timestamp": "2024-01-01T00:00:00+00:00"},
    )
    block = StateBlock.from_checkpoints("k", [ck_a, ck_b])
    assert set(block.state_arrays.keys()) == {"ema", "extra", "last_timestamp"}
    # A 缺 extra → None。
    assert block.state_arrays["extra"][0] is None
    assert block.state_arrays["extra"][1] == 5
    # 往返后仍保持。
    restored = StateBlock.from_arrow(block.to_arrow())
    assert restored.state_arrays["extra"][0] is None
    assert restored.state_arrays["extra"][1] == 5


# ---------------------------------------------------------------------------
# 2. StateBlockStore 单块读写 / fail-closed
# ---------------------------------------------------------------------------
def test_write_block_load_block_roundtrip(tmp_path) -> None:
    store = StateBlockStore(root=tmp_path)
    block = StateBlock.from_checkpoints(
        "f1::ts_ema", _build_checkpoints(_close_panel(10)), generation="gen-000001"
    )
    store.write_block(block)
    loaded = store.load_block("f1::ts_ema")
    assert loaded is not None
    assert loaded.as_of == block.as_of
    assert loaded.instruments == block.instruments
    assert loaded.state_arrays["ema"][0] == block.state_arrays["ema"][0]
    assert store.latest_as_of("f1::ts_ema") == block.as_of
    blocks = store.list_blocks("f1::ts_ema")
    assert len(blocks) == 1
    assert blocks[0]["generation"] == "gen-000001"


def test_load_block_fails_closed_for_missing_and_schema_mismatch(tmp_path) -> None:
    store = StateBlockStore(root=tmp_path)
    block = StateBlock.from_checkpoints(
        "f1::ts_ema", _build_checkpoints(_close_panel(10)), generation="gen-000001"
    )
    store.write_block(block)
    # 缺失 key。
    assert store.load_block("missing_key") is None
    # schema 不匹配（请求一个不存在的 schema 版本）。
    assert store.load_block("f1::ts_ema", schema_version="nope_state.v9") is None
    # 不存在的 generation。
    assert store.load_block("f1::ts_ema", generation="gen-999999") is None


# ---------------------------------------------------------------------------
# 3. write_generation 原子性
# ---------------------------------------------------------------------------
def test_write_generation_atomicity_on_crash(tmp_path) -> None:
    store = StateBlockStore(root=tmp_path)
    panel = _close_panel(10)
    cks = _build_checkpoints(panel)
    # 两个分片：A 与 B 各一个块。
    block_a = StateBlock.from_checkpoints("f1::ts_ema", [cks[0]])
    block_b = StateBlock.from_checkpoints("f1::ts_ema", [cks[1]])

    # 第一代成功发布。
    gen1 = store.write_generation("f1::ts_ema", cks[0].as_of, {"A": block_a, "B": block_b})
    assert store.current_generation("f1::ts_ema") == gen1
    gen1_blocks = store.load_generation("f1::ts_ema", gen1)
    assert set(gen1_blocks.keys()) == {"A", "B"}

    # 第二代：让第二个分片写失败（注入异常）→ 读者仍见完整旧代。
    class _Boom(Exception):
        pass

    original_new_file = StateBlockStore._generation_dir

    def _boom_generation_dir(self, key, schema, as_of, gen):
        # 只对第二代（gen-000002）的 B 分片写失败：让目录不可写。
        if gen == "gen-000002":
            raise _Boom("simulated crash mid-write")
        return original_new_file(self, key, schema, as_of, gen)

    StateBlockStore._generation_dir = _boom_generation_dir
    try:
        with pytest.raises(_Boom):
            store.write_generation("f1::ts_ema", cks[0].as_of, {"A": block_a, "B": block_b})
    finally:
        StateBlockStore._generation_dir = original_new_file

    # 当前代仍是旧代，且旧代完整可读。
    assert store.current_generation("f1::ts_ema") == gen1
    still = store.load_generation("f1::ts_ema", gen1)
    assert set(still.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# 4. Adapter drop-in：bootstrap 后 resume 与 full-history 完全一致
# ---------------------------------------------------------------------------
def test_adapter_bootstrap_and_resume_match_full_history_parity(tmp_path) -> None:
    n, split = 40, 25
    panel = _close_panel(n)

    full_ref: dict[str, np.ndarray] = {}
    for inst in panel.columns:
        res = execute_stateful_segment(
            "ts_ema", {"x": panel[inst].to_numpy(dtype=float)},
            timestamps=panel.index, instrument=inst,
            input_identity={"dataset": "unit"}, params={"span": 3},
            starts_at_dataset_origin=True,
        )
        full_ref[inst] = res.values

    block_store = StateBlockStore(root=tmp_path)
    store = StatefulCheckpointStoreAdapter(block_store=block_store)

    bootstrap = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(), source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert bootstrap is not None
    boot_series, mode = bootstrap
    assert mode["bootstrap"] is True
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        np.column_stack([full_ref["A"][: split + 1], full_ref["B"][: split + 1]]),
        equal_nan=True,
    )

    tail = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(),
        source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, mode2 = tail
    assert mode2["bootstrap"] is False
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        np.column_stack([full_ref["A"][split:], full_ref["B"][split:]]),
        equal_nan=True,
    )


def test_adapter_missing_checkpoint_falls_back_to_none(tmp_path) -> None:
    store = StatefulCheckpointStoreAdapter(block_store=StateBlockStore(root=tmp_path))
    panel = _close_panel(10)
    attempt = try_stateful_segmented_incremental(
        factor_id="f1", ir=_ema_ir(), source=_PanelSource(panel),
        store=store, start=panel.index[5], end=panel.index[-1], bootstrap=False,
    )
    assert attempt is None


# ---------------------------------------------------------------------------
# 5. garbage_collect 只保留最新一代
# ---------------------------------------------------------------------------
def test_garbage_collect_keeps_newest_generation(tmp_path) -> None:
    store = StateBlockStore(root=tmp_path)
    panel = _close_panel(10)
    cks = _build_checkpoints(panel)
    block = StateBlock.from_checkpoints("f1::ts_ema", cks)

    gen1 = store.write_generation("f1::ts_ema", cks[0].as_of, {"all": block})
    gen2 = store.write_generation("f1::ts_ema", cks[0].as_of, {"all": block})
    gen3 = store.write_generation("f1::ts_ema", cks[0].as_of, {"all": block})
    assert store.current_generation("f1::ts_ema") == gen3

    removed = store.garbage_collect("f1::ts_ema", keep=1)
    assert removed == 2
    assert store.current_generation("f1::ts_ema") == gen3
    # 旧代已删，新代仍完整。
    assert store.load_generation("f1::ts_ema", gen3) is not None
    with pytest.raises(ValueError):
        store.load_generation("f1::ts_ema", gen1)
