# -*- coding: utf-8 -*-
"""R21-FE-ZEROCOPY-OWNERSHIP：typed buffer-ownership 契约 + 零拷贝硬化测试。

覆盖（audit R21-FE-ZEROCOPY-AUDIT 的 GAP-1..4）：
  (a) GAP-1  typed ``BufferOwnership`` 枚举（READ_ONLY / OWNED_MUTABLE /
      BORROWED_READ_ONLY）挂在 ``BufferRef.ownership`` 上，替换自由字符串
      ``"shared"``；``to_dict()`` 序列化兼容（``READ_ONLY`` → ``"shared"``）。
      policy：caller buffer 默认 READ_ONLY，除非显式转移所有权；对 caller
      buffer 的零拷贝共享必须 READ_ONLY。
  (b) GAP-2a  caller buffer 不被算子变异——快照 float ndarray，跑热核
      （``_standardize`` vector path），断言输入逐字节不变。
  (c) GAP-2b  零拷贝路径 == 引用拷贝路径——同一算子经共享 buffer 与显式拷贝
      得到逐位一致结果。
  (d) GAP-2c  不同执行顺序 → 相同输出（复用 r38 stage-by-stage parity 思路）。
  (e) GAP-3  共享只读视图 ``flags.writeable=False``——意外原地写大声失败
      （ValueError），caller 数据完好。
  (f) GAP-4  两个算子共享同一输入 buffer、交换执行顺序 → 输出一致且输入不变。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("FACTOR_ENGINE_USE_NUMBA", "0")
os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


def _panel(n_day: int = 40, n_stk: int = 6, seed: int = 42, nan_frac: float = 0.05):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_day, n_stk))
    mask = rng.random(size=a.shape) < nan_frac
    a[mask] = np.nan
    return pd.DataFrame(a, columns=[f"s{i}" for i in range(n_stk)])


# ---------------------------------------------------------------------------
# (a) GAP-1：typed BufferOwnership 枚举 + BufferRef 接线 + 序列化兼容
# ---------------------------------------------------------------------------


def test_buffer_ownership_enum_typed_members():
    from factor_engine.runtime.buffer_ref import BufferRef
    from factor_engine.runtime.ownership import BufferOwnership

    # 三个 typed 成员，且是 str 枚举（JSON 可序列化）。
    assert {m.name for m in BufferOwnership} == {
        "READ_ONLY", "OWNED_MUTABLE", "BORROWED_READ_ONLY",
    }
    assert issubclass(BufferOwnership, str)
    # 默认契约：caller buffer 只读（序列化兼容旧 "shared" 自由字符串）。
    ref = BufferRef(representation="numpy_block")
    assert ref.ownership is BufferOwnership.READ_ONLY
    assert ref.ownership.value == "shared"
    # to_dict() 序列化兼容：旧 JSON 仍是 "shared"。
    assert ref.to_dict()["ownership"] == "shared"


def test_buffer_ref_typed_ownership_roundtrip_to_dict():
    from factor_engine.runtime.buffer_ref import BufferRef
    from factor_engine.runtime.ownership import BufferOwnership

    owned = BufferRef(
        representation="numpy_block",
        location=np.arange(4.0),
        ownership=BufferOwnership.OWNED_MUTABLE,
    )
    assert owned.to_dict()["ownership"] == "owned_mutable"
    borrowed = BufferRef(
        representation="numpy_block",
        ownership=BufferOwnership.BORROWED_READ_ONLY,
    )
    assert borrowed.to_dict()["ownership"] == "borrowed_read_only"


def test_factor_block_ref_uses_typed_ownership():
    from factor_engine.runtime import factor_block_ref as fbr
    from factor_engine.runtime.ownership import BufferOwnership

    idx = pd.date_range("2024-01-01", periods=4)
    vals = np.arange(8.0).reshape(4, 2)
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    # values_ref 零拷贝共享 caller buffer → BORROWED_READ_ONLY。
    assert ref.values_ref.ownership is BufferOwnership.BORROWED_READ_ONLY
    # validity 掩码是新建的 owned 数组 → OWNED_MUTABLE。
    vals2 = np.ones((4, 2), dtype="float64")
    vals2[0, 0] = np.nan
    ref2 = fbr.build_factor_block(["a", "b"], vals2, idx, dtype="float64")
    assert ref2.validity_ref is not None
    assert ref2.validity_ref.ownership is BufferOwnership.OWNED_MUTABLE


# ---------------------------------------------------------------------------
# (b) GAP-2a：caller buffer 不被算子变异（热核 _standardize vector path）
# ---------------------------------------------------------------------------


def test_standardize_kernel_does_not_mutate_caller_buffer():
    """快照 float ndarray → 跑 ``_standardize`` → 输入逐字节不变。"""
    from factor_engine.cleaned_operators.vector_path import _standardize

    rng = np.random.default_rng(123)
    f1 = rng.normal(size=2000)
    f2 = rng.normal(size=2000) * 3.0 + 1.0
    snap = np.concatenate([f1, f2]).tobytes()
    z = _standardize(f1, f2)
    assert z is not None
    z1, z2 = z
    assert np.allclose(z1, (f1 - f1.mean()) / np.std(f1), atol=1e-12)
    assert np.allclose(z2, (f2 - f2.mean()) / np.std(f2), atol=1e-12)
    # caller 输入逐字节不变（算子只分配新数组，绝不原地写输入）。
    assert np.concatenate([f1, f2]).tobytes() == snap


def test_panel_operator_does_not_mutate_caller_frame():
    """单 dtype float frame 经 ``to_numpy(dtype=float)`` 是 VIEW —— 快照后跑
    算子，断言 caller frame 逐位不变（视图写会直接污染 caller）。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    df = _panel(n_day=60, n_stk=5, nan_frac=0.0)
    arr = df.to_numpy(dtype=float)
    assert np.shares_memory(arr, df.to_numpy(dtype=float))
    snap = arr.tobytes()
    op = OperatorRegistry.get("ts_mean", "pandas_numpy")
    out = op.calculate(df, window=5)
    assert out.shape == df.shape
    assert arr.tobytes() == snap, "operator must not write through the to_numpy view"
    assert df.to_numpy(dtype=float).tobytes() == snap


# ---------------------------------------------------------------------------
# (c) GAP-2b：零拷贝路径 == 引用拷贝路径（同算子、同输入）
# ---------------------------------------------------------------------------


def test_shared_vs_copied_buffer_identical_results():
    """同一算子：共享 buffer（零拷贝视图）vs 显式拷贝 → 逐位一致。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=120)
    base = pd.DataFrame(rng.normal(size=(120, 4)), index=idx, columns=list("ABCD"))
    # 共享视图路径（零拷贝：shares_memory 为真）。
    shared = base.to_numpy(dtype=float)
    assert np.shares_memory(shared, base.to_numpy(dtype=float))
    shared_frame = pd.DataFrame(shared, index=idx, columns=list("ABCD"))
    # 显式拷贝路径（参考）。
    copied = pd.DataFrame(base.to_numpy(dtype=float).copy(), index=idx, columns=list("ABCD"))

    op = OperatorRegistry.get("ts_std", "pandas_numpy")
    out_shared = op.calculate(shared_frame, window=10)
    out_copied = op.calculate(copied, window=10)
    # NaN 位置一致 + 数值逐位一致（NaN != NaN，故用 equal_nan 比较）。
    assert np.array_equal(
        out_shared.to_numpy(dtype=float),
        out_copied.to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.shares_memory(shared_frame.to_numpy(dtype=float), base.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# (d) GAP-2c / GAP-4：执行顺序无关性 —— 共享输入 buffer 的算子重排
# ---------------------------------------------------------------------------


def _engine(dates: int = 30):
    from factor_engine.api import col, rank, ts_mean, ts_std
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dts = pd.bdate_range("2024-01-02", periods=dates)
    idx = pd.MultiIndex.from_product(
        [dts, ["A", "B", "C"]], names=["timestamp", "instrument"]
    )
    rng = np.random.default_rng(11)
    close = pd.Series(rng.normal(size=len(idx)), index=idx)
    src = InMemorySeriesSource({"close": close})
    return (
        FactorEngine(data_source=src, backend=PandasBackend()),
        close,
    )


def test_reordered_operators_share_one_input_buffer_identical_and_input_unchanged():
    """GAP-4：两个算子共享同一输入 buffer（run_many CSE 共享 'close' 列），
    交换因子顺序 → 每个因子输出逐位一致，且 source 'close' 输入逐字节不变。"""
    from factor_engine.api import col, rank, ts_mean, ts_std
    from factor_engine.api.factor import Factor

    engine, close = _engine(dates=30)
    f1 = Factor(name="f1", expr=rank(ts_mean(col("close"), 5)))
    f2 = Factor(name="f2", expr=ts_std(col("close"), 5))
    snap = close.to_numpy(dtype=float).tobytes()

    out_a = engine.run_many([f1, f2], perf=None)
    assert close.to_numpy(dtype=float).tobytes() == snap
    out_b = engine.run_many([f2, f1], perf=None)
    assert close.to_numpy(dtype=float).tobytes() == snap

    r1a = out_a["results"]["f1"]
    r1b = out_b["results"]["f1"]
    r2a = out_a["results"]["f2"]
    r2b = out_b["results"]["f2"]
    pd.testing.assert_series_equal(r1a, r1b)
    pd.testing.assert_series_equal(r2a, r2b)
    assert close.to_numpy(dtype=float).tobytes() == snap


def test_stage_by_stage_parity_same_shared_buffer():
    """GAP-2c：复用 r38 stage-by-stage 思路 —— barrier 子树逐 stage 执行并物化
    到 shared buffer 后，与整 root 一次性执行数值等价（同一输入 buffer 被多
    stage 共享，顺序不改变结果）。"""
    from factor_engine.planner.physical_lowerer import extract_stage_subplans

    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.buffer_store import GovernedBufferStore

    from factor_engine.api import col, cs_rank, ts_mean
    from factor_engine.api.factor import Factor

    engine, _close = _engine(dates=40)
    f = Factor(name="nested", expr=cs_rank(ts_mean(col("close"), 10)))
    node, _analysis = engine.compile(f)
    ctx = engine._make_context(shared_result_cache={})
    ref_plan, stages = extract_stage_subplans(node, factor_name="f")
    assert stages, "期望至少一个 barrier stage"
    store = ctx.shared_buffers or GovernedBufferStore({})
    if ctx.shared_buffers is None:
        ctx = ExecutionContext(
            data_source=ctx.data_source,
            shared_result_cache={},
            shared_buffers=store,
        )
    backend = engine.backend
    for sid, sub in stages:
        val = backend.execute(sub, ctx)
        assert val is not None, f"stage {sid} 执行结果为空"
        store.put(sid, val)
    staged = backend.execute(ref_plan, ctx)
    full = backend.execute(node, ctx)
    pd.testing.assert_series_equal(
        staged.sort_index(), full.sort_index(), check_names=False
    )


# ---------------------------------------------------------------------------
# (e) GAP-3：共享只读视图 flags.writeable=False —— 意外原地写大声失败
# ---------------------------------------------------------------------------


def test_shared_readonly_view_rejects_inplace_write():
    """make_readonly_shared_view 返回零拷贝 WRITEABLE=False 视图：原地写抛
    ValueError，caller 数据完好。"""
    from factor_engine.runtime.ownership import make_readonly_shared_view

    a = np.arange(12.0).reshape(3, 4)
    view = make_readonly_shared_view(a)
    assert np.shares_memory(a, view)
    assert view.flags.writeable is False
    assert a.flags.writeable is True, "caller 自己的数组必须仍可写"
    with pytest.raises(ValueError):
        view[0, 0] = 99.0
    assert a[0, 0] == 0.0


def test_factor_block_ref_shared_values_view_writeable_false():
    """build_factor_block 的 values_ref.location 是共享只读视图：意外原地写
    大声失败，caller 数组完好；to_arrow_table / to_parquet 仍正常工作。"""
    from factor_engine.runtime import factor_block_ref as fbr

    idx = pd.date_range("2024-01-01", periods=4)
    vals = np.arange(8.0).reshape(4, 2)
    ref = fbr.build_factor_block(["a", "b"], vals, idx, dtype="float64")
    loc = ref.values_ref.location
    assert np.shares_memory(loc, vals)
    assert loc.flags.writeable is False
    with pytest.raises(ValueError):
        loc[0, 0] = 99.0
    assert vals[0, 0] == 0.0
    table = ref.to_arrow_table()
    assert table.num_rows == 4
    path = ref.to_parquet("/tmp/r21_fe_block.parquet")
    back = pd.read_parquet(path)
    assert list(back["a"]) == [0.0, 2.0, 4.0, 6.0]


def test_readonly_guard_fails_loudly_on_single_dtype_frame_view():
    """单 dtype float frame 的 to_numpy(dtype=float) 是 VIEW；把共享视图置为
    WRITEABLE=False 后，意外原地写必须大声失败，而不是静默污染 caller frame。"""
    from factor_engine.runtime.ownership import make_readonly_shared_view

    df = _panel(n_day=10, n_stk=3, nan_frac=0.0)
    arr = df.to_numpy(dtype=float)
    assert np.shares_memory(arr, df.to_numpy(dtype=float))
    guard = make_readonly_shared_view(arr)
    assert guard.flags.writeable is False
    assert np.shares_memory(guard, df.to_numpy(dtype=float))
    snap = df.to_numpy(dtype=float).tobytes()
    with pytest.raises(ValueError):
        guard[0, 0] = 123.0
    assert df.to_numpy(dtype=float).tobytes() == snap
