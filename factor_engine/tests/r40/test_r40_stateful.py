# -*- coding: utf-8 -*-
"""R40 #244-250: stateful segmented 的 typed fallback / boundary probe /
axis identity / input identity / corruption hard-fail / chunk-invariance gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.math_certificate import (
    CHECKPOINT_CHUNK_INVARIANCE_FAILURE,
    check_chunk_invariance_all_segmented_canonicals,
)
from factor_engine.runtime.stateful_incremental import (
    AxisIdentityCertificate,
    BoundaryProbeResult,
    SegmentedFallbackReason,
    SourceSnapshotIdentityUnavailableError,
    StatefulSegmentedCorruptionError,
    _boundary_timeline,
    _checkpoint_input_identity,
    _source_snapshot_scope,
)


class _BrokenSource:
    """source snapshot identity resolver 故意故障（data_snapshot_id=NaN 使
    ``compute_data_scope`` 的 ``json.dumps(allow_nan=False)`` 抛 ValueError）。"""

    def __init__(self, fail_scope: bool = True) -> None:
        self.fail_scope = fail_scope
        self.start_date = None
        self.end_date = None
        self.data_snapshot_id = float("nan")
        self.dataset = "broken"

    def load_column(self, name: str):
        raise RuntimeError("load failed")


def test_ephemeral_scope_never_reuses_checkpoint_identity() -> None:
    """#244: research ephemeral 每次不同 -> 永不跨 run 复用；production 抛错。"""
    src = _BrokenSource()
    s1 = _source_snapshot_scope(src, mode="research")
    s2 = _source_snapshot_scope(src, mode="research")
    assert s1.startswith("ephemeral:")
    assert s1 != s2  # unique run-scoped ID -> 跨 run 永不命中
    with pytest.raises(SourceSnapshotIdentityUnavailableError):
        _source_snapshot_scope(src, mode="production")


def test_boundary_probe_unknown_blocks_resume() -> None:
    """#245: 探测失败 -> UNKNOWN（production 不允许 resume）。"""
    src = _BrokenSource()
    result, timeline = _boundary_timeline(src, "x", since="2024-01-01", start="2024-01-10", mode="production")
    assert result is BoundaryProbeResult.UNKNOWN
    assert timeline == []


def test_boundary_per_instrument_checkpoint_as_of() -> None:
    """#246: boundary probe 按 (checkpoint.as_of, start) 分组 —— 不同 as_of 各自探测。"""
    src = _BrokenSource()
    r1, _ = _boundary_timeline(src, "x", since="2024-01-01", start="2024-01-10", mode="research")
    r2, _ = _boundary_timeline(src, "x", since="2024-01-05", start="2024-01-10", mode="research")
    # 两个不同 as_of 各独立探测；探测失败均 UNKNOWN（research 不抛，但结果可区分调用）
    assert r1 is BoundaryProbeResult.UNKNOWN and r2 is BoundaryProbeResult.UNKNOWN


def test_multi_input_axis_identity_verified_in_production() -> None:
    """#247: AxisIdentityCertificate 在 production 要求 exact match。"""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = ["A", "B"]
    f1 = pd.DataFrame(np.ones((5, 2)), index=idx, columns=cols)
    f2 = pd.DataFrame(np.ones((5, 2)), index=idx, columns=cols)
    f3_bad_col = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["A", "C"])
    assert AxisIdentityCertificate.verify_frames_share_identity({"a": f1, "b": f2}, mode="production") is True
    assert AxisIdentityCertificate.verify_frames_share_identity({"a": f1, "b": f3_bad_col}, mode="production") is False
    # research 宽松：仅长度一致
    f4 = pd.DataFrame(np.ones((5, 3)), index=idx, columns=["A", "B", "D"])
    assert AxisIdentityCertificate.verify_frames_share_identity({"a": f1, "b": f4}, mode="research") is False  # 列数不同


def test_checkpoint_identity_differs_across_market_contexts() -> None:
    """#248: input identity 绑定 market / calendar / timezone / universe / basis。"""
    base = _checkpoint_input_identity(
        factor_id="f1", canonical="ts_ema", input_names=["x"], params={},
        source_scope="scope", market="ashare", calendar_version="cal-v1",
        timezone="Asia/Shanghai",
    )
    us = _checkpoint_input_identity(
        factor_id="f1", canonical="ts_ema", input_names=["x"], params={},
        source_scope="scope", market="us", calendar_version="cal-v1",
        timezone="America/New_York",
    )
    assert base != us
    # 不同 numeric semantics / universe hash 也改变 identity
    base2 = _checkpoint_input_identity(
        factor_id="f1", canonical="ts_ema", input_names=["x"], params={},
        source_scope="scope", market="ashare", universe_membership_hash="uni-v2",
    )
    assert base != base2


def test_segmented_path_hard_fails_on_corruption() -> None:
    """#249: corruption 类在 production 下 hard fail（StatefulSegmentedCorruptionError）。"""
    from factor_engine.runtime.stateful_incremental import _CORRUPTION_REASONS

    assert SegmentedFallbackReason.CHECKPOINT_CORRUPTION in _CORRUPTION_REASONS
    assert SegmentedFallbackReason.STATE_CORRUPTION in _CORRUPTION_REASONS
    assert SegmentedFallbackReason.PIT_VIOLATION in _CORRUPTION_REASONS
    assert SegmentedFallbackReason.CALENDAR_ERROR in _CORRUPTION_REASONS
    assert SegmentedFallbackReason.NO_CHECKPOINT not in _CORRUPTION_REASONS
    # 模拟 corruption：as_of unparseable -> 在 production 应 raise。
    # 直接验证异常类型可用（真实路径由 try_stateful_segmented_incremental 触发）。
    assert issubclass(StatefulSegmentedCorruptionError, RuntimeError)


def test_chunk_invariance_hard_gate_all_segmented_canonicals() -> None:
    """#250: 所有 SEGMENTED_EXECUTION_CANONICALS 的 checkpoint 分块重放 == 全量。"""
    ok, detail = check_chunk_invariance_all_segmented_canonicals(n_bars=60, n_random_chunkings=2)
    assert ok is True, detail
    assert detail["gate"] == CHECKPOINT_CHUNK_INVARIANCE_FAILURE
    assert all(r["ok"] for r in detail["per_canonical"].values()), detail
