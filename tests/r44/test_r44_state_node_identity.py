# -*- coding: utf-8 -*-
"""R44: node-level incremental FactorEngine — StateNodeIdentity / node plan /
interior-stateful executor tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.runtime.stateful_incremental import (
    NodeIncrementalMode,
    compute_state_node_identity,
    plan_node_level_incremental,
    shared_state_resume,
    try_interior_stateful_incremental,
)
from stateful_runtime import execute_stateful_segment


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


def _rank_ema_ir(span: int = 3) -> IRNode:
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})
    return IRNode(op="rank", inputs=(ema,))


def _ema_plus_close_ir(span: int = 3) -> IRNode:
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})
    close = IRNode(op="column", attrs={"name": "close"})
    return IRNode(op="add", inputs=(ema, close))


# ---------------------------------------------------------------------------
# 1. StateNodeIdentity
# ---------------------------------------------------------------------------

def test_state_node_identity_stable_and_shared_across_factor_ids() -> None:
    a1 = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope1")
    a2 = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope1")
    assert a1 == a2
    assert a1.stable_key() == a2.stable_key()
    # 不同 factor_id 不参与身份 —— 相同 canonical/params/source_scope 共享 key。
    b = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope1")
    assert a1.stable_key() == b.stable_key()


def test_state_node_identity_differs_on_params() -> None:
    a = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope1")
    b = compute_state_node_identity("ts_ema", {"span": 30}, source_scope="scope1")
    assert a.stable_key() != b.stable_key()


def test_state_node_identity_differs_on_source_scope() -> None:
    a = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope1")
    b = compute_state_node_identity("ts_ema", {"span": 20}, source_scope="scope2")
    assert a.stable_key() != b.stable_key()


def test_state_node_identity_differs_on_input_column() -> None:
    # R45 碰撞修复：ts_ema(close,20) 与 ts_ema(volume,20) 必须不同身份。
    a = compute_state_node_identity(
        "ts_ema", {"span": 20}, source_scope="scope1", input_columns=("close",)
    )
    b = compute_state_node_identity(
        "ts_ema", {"span": 20}, source_scope="scope1", input_columns=("volume",)
    )
    assert a.stable_key() != b.stable_key()


def test_state_node_identity_binds_context_fields() -> None:
    # R45：calendar_version / timezone / universe / price basis / field /
    # operator contract / numeric semantics / missing policy 任一不同 => 不同身份。
    base = dict(source_scope="scope1", input_columns=("close",))
    a = compute_state_node_identity("ts_ema", {"span": 20}, **base)
    variants = [
        dict(calendar_version="v2"),
        dict(timezone="Asia/Shanghai"),
        dict(universe_membership_hash="u2"),
        dict(price_basis_policy="pb2"),
        dict(field_contract_digests={"close": "d2"}),
        dict(operator_semantic_contract_digest="osc2"),
        dict(numeric_semantics_hash="ns2"),
        dict(missing_support_policy="ms2"),
    ]
    for v in variants:
        b = compute_state_node_identity("ts_ema", {"span": 20}, **{**base, **v})
        assert a.stable_key() != b.stable_key(), v


def test_state_node_identity_none_context_distinct_from_empty() -> None:
    # R45：None 上下文按 None 参与哈希（不跳过），与空字符串上下文不同。
    a = compute_state_node_identity(
        "ts_ema", {"span": 20}, source_scope="scope1",
        input_columns=("close",), calendar_version=None,
    )
    b = compute_state_node_identity(
        "ts_ema", {"span": 20}, source_scope="scope1",
        input_columns=("close",), calendar_version="",
    )
    assert a.stable_key() != b.stable_key()


# ---------------------------------------------------------------------------
# 2. shared_state_resume
# ---------------------------------------------------------------------------

def test_shared_state_resume_identical_for_shared_ema() -> None:
    panel = _close_panel(20)
    src1 = _PanelSource(panel)
    src2 = _PanelSource(panel)
    store = StatefulCheckpointStore(root="/tmp/r44_shared_store")
    ids = shared_state_resume(
        canonical="ts_ema", params={"span": 20}, sources=[src1, src2], store=store
    )
    assert len(ids) == 2
    assert ids[0].stable_key() == ids[1].stable_key()


# ---------------------------------------------------------------------------
# 3. plan_node_level_incremental
# ---------------------------------------------------------------------------

def test_plan_supportable_rank_ema() -> None:
    ir = _rank_ema_ir(span=20)
    plan = plan_node_level_incremental(ir, factor_id="f1", source_scope="scope1")
    assert plan.supportable is True
    assert plan.unsupported_reason is None
    modes = {n.op: n.mode for n in plan.nodes}
    assert modes["ts_ema"] == NodeIncrementalMode.RESTORE_STATE
    assert modes["rank"] in (NodeIncrementalMode.LOAD_TAIL, NodeIncrementalMode.LOAD_TODAY)
    assert "ts_ema" in [n.op for n in plan.nodes if n.mode == NodeIncrementalMode.RESTORE_STATE]


def test_plan_unsupported_interior_shape() -> None:
    # 一个 FULL_REPLAY 节点（KAMA，无 checkpoint restore）与 checkpointed 兄弟共存。
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": 3})
    kama = IRNode(op="KAMA", inputs=(IRNode(op="column", attrs={"name": "close"}),))
    ir = IRNode(op="add", inputs=(ema, kama))
    plan = plan_node_level_incremental(ir, factor_id="f1", source_scope="scope1")
    assert plan.supportable is False
    assert plan.unsupported_reason is not None


# ---------------------------------------------------------------------------
# 4. try_interior_stateful_incremental
# ---------------------------------------------------------------------------

def _full_reference(ir: IRNode, panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """用 execute_stateful_segment 全历史参考（stateful 节点）+ 下游解释器。"""
    # 定位 IR 中的 ts_ema 节点（可能不是根的直接子节点）。
    ema = None

    def find(node: IRNode) -> None:
        nonlocal ema
        if ema is not None:
            return
        if node.op == "ts_ema":
            ema = node
            return
        for child in node.inputs:
            find(child)

    find(ir)
    assert ema is not None
    ref: dict[str, np.ndarray] = {}
    for inst in panel.columns:
        res = execute_stateful_segment(
            "ts_ema", {"x": panel[inst].to_numpy(dtype=float)},
            timestamps=panel.index, instrument=inst,
            input_identity={"dataset": "unit"}, params={"span": ema.attrs["span"]},
            starts_at_dataset_origin=True,
        )
        ref[inst] = res.values
    return ref


def test_interior_stateful_bootstrap_then_resume_parity(tmp_path) -> None:
    n, split = 20, 12
    panel = _close_panel(n)
    ir = _rank_ema_ir(span=3)
    store = StatefulCheckpointStore(root=tmp_path)

    # 全历史参考：rank(ema(close,3)) —— 每日期截面 rank。
    ema_ref = _full_reference(ir, panel)
    ema_panel = pd.DataFrame(
        {inst: ema_ref[inst] for inst in panel.columns}, index=panel.index
    )
    valid = np.isfinite(ema_panel.to_numpy(dtype=float))
    masked = ema_panel.where(valid)
    r = masked.rank(axis=1, method="average")
    n_valid = pd.Series(valid.sum(axis=1), index=ema_panel.index)
    denom = (n_valid - 1).replace(0, np.nan)
    full_rank = r.sub(1, axis=0).div(denom, axis=0)
    singleton = valid & np.broadcast_to((n_valid <= 1).to_numpy()[:, None], full_rank.shape)
    full_rank = full_rank.where(~singleton, 0.5).where(valid, np.nan)

    # Bootstrap over [0, split] (inclusive).
    boot = try_interior_stateful_incremental(
        factor_id="f1", ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert boot is not None
    boot_series, mode = boot
    assert mode["mode"] == "interior_stateful"
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        full_rank.iloc[: split + 1].to_numpy(),
        equal_nan=True,
    )

    # Incremental resume over [split, n-1].
    tail = try_interior_stateful_incremental(
        factor_id="f1", ir=ir, source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, mode2 = tail
    assert mode2["bootstrap"] is False
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        full_rank.iloc[split:].to_numpy(),
        equal_nan=True,
    )


def test_interior_stateful_ema_plus_close_parity(tmp_path) -> None:
    n, split = 20, 12
    panel = _close_panel(n)
    ir = _ema_plus_close_ir(span=3)
    store = StatefulCheckpointStore(root=tmp_path)

    ema_ref = _full_reference(ir, panel)
    ema_panel = pd.DataFrame(
        {inst: ema_ref[inst] for inst in panel.columns}, index=panel.index
    )
    full = ema_panel + panel  # ema + close

    boot = try_interior_stateful_incremental(
        factor_id="f2", ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert boot is not None
    boot_series, _ = boot
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        full.iloc[: split + 1].to_numpy(),
        equal_nan=True,
    )

    tail = try_interior_stateful_incremental(
        factor_id="f2", ir=ir, source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, _ = tail
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        full.iloc[split:].to_numpy(),
        equal_nan=True,
    )


def test_interior_stateful_unsupported_downstream_returns_none(tmp_path) -> None:
    # 下游含不受支持算子（ts_mean）→ 返回 None（回退 full replay，绝不错误值）。
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": 3})
    ir = IRNode(op="ts_mean", inputs=(ema,), attrs={"window": 5})
    panel = _close_panel(20)
    store = StatefulCheckpointStore(root=tmp_path)
    result = try_interior_stateful_incremental(
        factor_id="f3", ir=ir, source=_PanelSource(panel),
        store=store, start=panel.index[0], end=panel.index[-1], bootstrap=True,
    )
    assert result is None


# ---------------------------------------------------------------------------
# 5. R45: interior-from-root eval — rank(ema(close)+volume) returns full rank.
# ---------------------------------------------------------------------------

def _close_volume_panel(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "A": np.arange(n, dtype=float) + 10.0,
            "B": np.arange(n, dtype=float) * 2.0 + 1.0,
            "volume": np.arange(n, dtype=float) * 3.0 + 5.0,
        },
        index=idx,
    )


def _rank_ema_plus_volume_ir(span: int = 3) -> IRNode:
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})
    volume = IRNode(op="column", attrs={"name": "volume"})
    add = IRNode(op="add", inputs=(ema, volume))
    return IRNode(op="rank", inputs=(add,))


def test_interior_stateful_rank_of_ema_plus_volume_parity(tmp_path) -> None:
    # R45 #3: 从真实因子根求值 —— rank(ema(close)+volume) 返回完整 rank 结果，
    # 而非只返回 add(ema, volume)。
    n, split = 20, 12
    panel = _close_volume_panel(n)
    ir = _rank_ema_plus_volume_ir(span=3)
    store = StatefulCheckpointStore(root=tmp_path)

    # 全历史参考：rank(ema(close,3) + volume)。load_column 返回整块 panel 的
    # 堆叠（A/B/volume 三列即三个 instrument），故 add 覆盖全部 3 列。
    ema_ref = _full_reference(ir, panel)
    ema_panel = pd.DataFrame(
        {inst: ema_ref[inst] for inst in panel.columns}, index=panel.index
    )
    add_panel = ema_panel + panel
    valid = np.isfinite(add_panel.to_numpy(dtype=float))
    masked = add_panel.where(valid)
    r = masked.rank(axis=1, method="average")
    n_valid = pd.Series(valid.sum(axis=1), index=add_panel.index)
    denom = (n_valid - 1).replace(0, np.nan)
    full_rank = r.sub(1, axis=0).div(denom, axis=0)
    singleton = valid & np.broadcast_to((n_valid <= 1).to_numpy()[:, None], full_rank.shape)
    full_rank = full_rank.where(~singleton, 0.5).where(valid, np.nan)

    boot = try_interior_stateful_incremental(
        factor_id="f4", ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert boot is not None
    boot_series, mode = boot
    assert mode["mode"] == "interior_stateful"
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        full_rank.iloc[: split + 1].to_numpy(),
        equal_nan=True,
    )

    tail = try_interior_stateful_incremental(
        factor_id="f4", ir=ir, source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, mode2 = tail
    assert mode2["bootstrap"] is False
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        full_rank.iloc[split:].to_numpy(),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# 6. R45: multi-stateful DAG — ts_ema + RSI_WILDER resume both.
# ---------------------------------------------------------------------------

def _ema_plus_rsi_ir(span: int = 3, rsi_window: int = 3) -> IRNode:
    ema = IRNode(op="ts_ema", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"span": span})
    rsi = IRNode(op="RSI_WILDER", inputs=(IRNode(op="column", attrs={"name": "close"}),), attrs={"window": rsi_window})
    return IRNode(op="add", inputs=(ema, rsi))


def test_plan_supportable_multi_stateful() -> None:
    # R45 #2: 多 stateful 节点（ts_ema + RSI_WILDER）不再因 composite state 回退。
    ir = _ema_plus_rsi_ir()
    plan = plan_node_level_incremental(ir, factor_id="f5", source_scope="scope1")
    assert plan.supportable is True
    assert plan.unsupported_reason is None
    assert len(plan.stateful_node_ids) == 2
    stateful_ops = [n.op for n in plan.nodes if n.mode == NodeIncrementalMode.RESTORE_STATE]
    assert "ts_ema" in stateful_ops
    assert "RSI_WILDER" in stateful_ops


def test_interior_stateful_multi_stateful_parity(tmp_path) -> None:
    # R45 #2: ts_ema + RSI_WILDER 两个 stateful 节点各自恢复，再注入 DAG。
    n, split = 20, 12
    panel = _close_panel(n)
    ir = _ema_plus_rsi_ir(span=3, rsi_window=3)
    store = StatefulCheckpointStore(root=tmp_path)

    # 全历史参考：ema(close,3) + RSI_WILDER(close,3)。
    ema_ref = _full_reference(ir, panel)
    ema_panel = pd.DataFrame(
        {inst: ema_ref[inst] for inst in panel.columns}, index=panel.index
    )
    rsi_ref: dict[str, np.ndarray] = {}
    for inst in panel.columns:
        res = execute_stateful_segment(
            "RSI_WILDER", {"x": panel[inst].to_numpy(dtype=float)},
            timestamps=panel.index, instrument=inst,
            input_identity={"dataset": "unit"}, params={"window": 3},
            starts_at_dataset_origin=True,
        )
        rsi_ref[inst] = res.values
    rsi_panel = pd.DataFrame(
        {inst: rsi_ref[inst] for inst in panel.columns}, index=panel.index
    )
    full = ema_panel + rsi_panel

    boot = try_interior_stateful_incremental(
        factor_id="f5", ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    assert boot is not None
    boot_series, mode = boot
    assert mode["mode"] == "interior_stateful"
    np.testing.assert_allclose(
        boot_series.unstack(level="instrument").to_numpy(),
        full.iloc[: split + 1].to_numpy(),
        equal_nan=True,
    )

    tail = try_interior_stateful_incremental(
        factor_id="f5", ir=ir, source=_PanelSource(panel.iloc[split:]),
        store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
    )
    assert tail is not None
    tail_series, mode2 = tail
    assert mode2["bootstrap"] is False
    np.testing.assert_allclose(
        tail_series.unstack(level="instrument").to_numpy(),
        full.iloc[split:].to_numpy(),
        equal_nan=True,
    )
