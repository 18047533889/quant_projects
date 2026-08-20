# -*- coding: utf-8
"""R20 compile-chain semantic preservation (R20-001..013 / R20-457..459).

Covers the rolling CSE / SQL lowering / composite lowering / lowering hash /
fastpath rewrite audit items:

- R20-001 — rolling CSE ``plan_ref`` inherits the shared subtree's output
  semantic contract;
- R20-002 — every rolling-CSE rewrite preserves ``semantic_attrs`` / ``node_id``
  (single ``rewrite_node`` constructor);
- R20-003 — ``verify_plan_ref_semantics`` / ``verify_plan_refs_semantics``;
- R20-004 — ``SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0`` and the before / after
  output-semantic-digest invariant;
- R20-005..008 — ``rolling_semantic_key`` keys on every panel input's full
  ``structural_key`` (no first-column-only over-sharing);
- R20-009 — adversarial corpus: no false sharing across transforms / params /
  price basis;
- R20-010..013 — ``ROLLING_OPS`` derived from the registry, consistent with the
  bootstrap base set, window parsed via the declared history param;
- R20-457 — rolling CSE keeps each root's output semantic digest;
- R20-458 — ``ts_mean(log(close),20)`` never shares with ``ts_mean(close,20)``;
- R20-459 — SQL ``materialized_series`` carries the extracted subtree's semantic
  attrs; composite lowering propagates the composite's output contract.
"""
from __future__ import annotations

# Fast registry bring-up: registering just the rolling-op modules is O(1s) and
# sufficient for the derivation tests, unlike the full (and currently
# concurrent-session-broken) ``load_all()``.
try:  # pragma: no cover - collection-time registry warm-up
    import cleaned_operators.common.time_series  # noqa: E402,F401
    import cleaned_operators.common.statistics  # noqa: E402,F401
except Exception:  # pragma: no cover - concurrent session may break a module
    pass

import pytest

from planner.logical_plan import PlanNode
from planner.plan_hash import structural_key
from planner.cse import apply_cse
from planner import rolling_cache as rolling_cache_mod
from planner.rolling_cache import (
    _BASE_ROLLING_OPS,
    _derive_rolling_ops,
    _window_from_attrs,
    is_rolling_operator,
    refresh_rolling_ops,
)
from planner.rolling_cse import (
    SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE,
    apply_rolling_cse,
    assert_rolling_cse_semantics_preserved,
    rolling_cse_output_semantic_digests,
    rolling_semantic_key,
    verify_plan_ref_semantics,
    verify_plan_refs_semantics,
)
from planner.composite_lowering import lower_composite_operators


# ---------------------------------------------------------------------------
# 构造 helpers
# ---------------------------------------------------------------------------
def _col(name: str = "close", **source_keys: object) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name, **source_keys}, inputs=[])


def _lit(value: object) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": value}, inputs=[])


def _ts_mean(x: PlanNode, window: int, semantic_attrs=None, **attrs: object) -> PlanNode:
    return PlanNode(
        op="ts_mean",
        inputs=[x, _lit(window)],
        attrs=attrs,
        semantic_attrs=semantic_attrs or {},
    )


def _ts_std(x: PlanNode, window: int, semantic_attrs=None, **attrs: object) -> PlanNode:
    return PlanNode(
        op="ts_std",
        inputs=[x, _lit(window)],
        attrs=attrs,
        semantic_attrs=semantic_attrs or {},
    )


def _walk(root: PlanNode):
    for c in root.inputs:
        yield from _walk(c)
    yield root


def _find_ref(root: PlanNode) -> PlanNode | None:
    for n in _walk(root):
        if n.op == "plan_ref":
            return n
    return None


def _has_plan_ref(root: PlanNode) -> bool:
    return _find_ref(root) is not None


_SEM = {"unit": "return", "grain": "daily", "price_basis": "RAW"}


# ---------------------------------------------------------------------------
# R20-001: plan_ref 继承被共享子树的 output semantic contract
# ---------------------------------------------------------------------------
def test_r20_001_plan_ref_carries_semantic_attrs():
    a = PlanNode(op="ts_std", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    b = PlanNode(op="ts_std_dev", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    new_roots, shared = apply_rolling_cse([a, b])
    ref = _find_ref(new_roots[1])
    assert ref is not None
    assert dict(ref.semantic_attrs) == _SEM
    # the shared subtree root keeps the same contract
    sid = ref.attrs["sid"]
    assert dict(shared[sid].semantic_attrs) == _SEM


# ---------------------------------------------------------------------------
# R20-002: rolling CSE rewrite 保留 semantic_attrs / node_id
# ---------------------------------------------------------------------------
def test_r20_002_rewrite_preserves_semantic_attrs_on_normal_nodes():
    col = _col()
    std = PlanNode(op="ts_std", inputs=[col, _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    std_dev = PlanNode(op="ts_std_dev", inputs=[col, _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    root_a = PlanNode(op="add", inputs=[std, _col("v")], attrs={}, semantic_attrs=dict(_SEM))
    root_b = PlanNode(op="add", inputs=[std_dev, _col("w")], attrs={}, semantic_attrs=dict(_SEM))
    new_roots, _ = apply_rolling_cse([root_a, root_b])
    for root in new_roots:
        for n in _walk(root):
            if n.op not in {"column", "literal"}:
                assert dict(n.semantic_attrs) == _SEM, f"semantic lost at {n.op}"
    ref = _find_ref(new_roots[1])
    assert ref is not None
    assert dict(ref.semantic_attrs) == _SEM


# ---------------------------------------------------------------------------
# R20-003: verify_plan_ref_semantics
# ---------------------------------------------------------------------------
def test_r20_003_verify_plan_ref_semantics_ok():
    a = PlanNode(op="ts_std", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    b = PlanNode(op="ts_std_dev", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    new_roots, shared = apply_rolling_cse([a, b])
    ref = _find_ref(new_roots[1])
    assert ref is not None
    assert verify_plan_ref_semantics(ref, shared)
    assert verify_plan_refs_semantics(new_roots, shared)


def test_r20_003_verify_plan_ref_semantics_rejects_bad_sid():
    ref = PlanNode(op="plan_ref", attrs={"sid": "nonexistent"}, inputs=[])
    assert not verify_plan_ref_semantics(ref, {})
    # a shared node stored under a key that is NOT its structural key: any
    # plan_ref whose sid == that key must be rejected (sid binds semantics)
    node = PlanNode(
        op="ts_mean", inputs=[_col()], attrs={"window": 5},
        semantic_attrs={"unit": "return"},
    )
    shared_wrong_key = {"abc": node}
    bad = PlanNode(
        op="plan_ref", attrs={"sid": "abc"}, inputs=[],
        semantic_attrs={"unit": "return"},
    )
    assert not verify_plan_ref_semantics(bad, shared_wrong_key)
    # store under the CORRECT structural key; a mismatched semantic digest fails
    correct_sid = structural_key(node)
    shared = {correct_sid: node}
    mismatch = PlanNode(
        op="plan_ref", attrs={"sid": correct_sid}, inputs=[],
        semantic_attrs={"unit": "price"},  # != shared node's unit=return
    )
    assert not verify_plan_ref_semantics(mismatch, shared)
    # a fully consistent reference passes
    ok = PlanNode(
        op="plan_ref", attrs={"sid": correct_sid}, inputs=[],
        semantic_attrs={"unit": "return"},
    )
    assert verify_plan_ref_semantics(ok, shared)


# ---------------------------------------------------------------------------
# R20-004 / R20-457: rolling CSE 前后 output semantic digest 一致
# ---------------------------------------------------------------------------
def test_r20_457_semantic_digest_preserved_across_cse_stages():
    col = _col()
    # ts_std vs ts_std_dev: structurally different, semantically equivalent —
    # rolling CSE merges them into a plan_ref, and the output semantic digest of
    # every root must be identical before structural CSE / after structural CSE /
    # after rolling CSE.
    std = PlanNode(op="ts_std", inputs=[col, _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    std_dev = PlanNode(
        op="ts_std_dev", inputs=[col, _lit(20)], attrs={},
        semantic_attrs=dict(_SEM),
    )
    root_a = PlanNode(op="add", inputs=[std, _col("v")], attrs={}, semantic_attrs=dict(_SEM))
    root_b = PlanNode(op="add", inputs=[std_dev, _col("w")], attrs={}, semantic_attrs=dict(_SEM))
    before = [root_a, root_b]

    struct_roots, struct_shared = apply_cse(before)
    roll_roots, roll_shared = apply_rolling_cse(struct_roots, existing_shared=struct_shared)

    assert rolling_cse_output_semantic_digests(before) == rolling_cse_output_semantic_digests(struct_roots)
    assert rolling_cse_output_semantic_digests(struct_roots) == rolling_cse_output_semantic_digests(roll_roots)
    assert_rolling_cse_semantics_preserved(before, roll_roots)
    assert SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0
    # a plan_ref was actually created (ts_std_dev -> shared ts_std subtree)
    ref = _find_ref(roll_roots[1])
    assert ref is not None
    assert dict(ref.semantic_attrs) == _SEM
    # and the produced plan_refs are all verifiable against the shared table
    assert verify_plan_refs_semantics(roll_roots, roll_shared)


def test_r20_004_rolling_cse_does_not_lose_semantic_attrs_when_shared():
    a = PlanNode(op="ts_std", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    b = PlanNode(op="ts_std_dev", inputs=[_col(), _lit(20)], attrs={}, semantic_attrs=dict(_SEM))
    new_roots, shared = apply_rolling_cse([a, b])
    ref = _find_ref(new_roots[1])
    assert ref is not None
    assert verify_plan_ref_semantics(ref, shared)
    assert_rolling_cse_semantics_preserved([a, b], new_roots)


# ---------------------------------------------------------------------------
# R20-005..008: rolling_semantic_key 包含每个 panel 输入的完整 structural_key
# ---------------------------------------------------------------------------
def test_r20_005_log_transform_not_shared():
    close = _col()
    plain = _ts_mean(close, 20)
    logged = _ts_mean(PlanNode(op="log", inputs=[close], attrs={}), 20)
    assert rolling_semantic_key(plain) is not None
    assert rolling_semantic_key(plain) != rolling_semantic_key(logged)
    new_roots, shared = apply_rolling_cse([plain, logged])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_r20_006_corr_second_input_transform_not_shared():
    close, vol = _col(), _col("volume")
    corr1 = PlanNode(op="ts_corr", inputs=[close, vol, _lit(20)], attrs={})
    corr2 = PlanNode(
        op="ts_corr",
        inputs=[PlanNode(op="log", inputs=[close], attrs={}), vol, _lit(20)],
        attrs={},
    )
    assert rolling_semantic_key(corr1) != rolling_semantic_key(corr2)
    new_roots, _ = apply_rolling_cse([corr1, corr2])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])


def test_r20_007_divide_transform_not_shared():
    close = _col()
    inner = _ts_mean(close, 5)
    div = PlanNode(op="divide", inputs=[close, inner], attrs={})
    a = _ts_mean(close, 20)
    b = _ts_mean(div, 20)
    assert rolling_semantic_key(a) != rolling_semantic_key(b)
    new_roots, _ = apply_rolling_cse([a, b])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])


def test_r20_008_key_contains_full_input_structural_keys():
    close = _col()
    k = rolling_semantic_key(_ts_mean(close, 20))
    assert k is not None
    assert '"canonical":"ts_mean"' in k
    # the panel input's structural key must appear in the payload
    assert structural_key(close) in k


# ---------------------------------------------------------------------------
# R20-009: adversarial CSE corpus —— 一律不得误共享
# ---------------------------------------------------------------------------
def test_r20_009_same_window_different_min_periods_not_shared():
    a = _ts_mean(_col(), 20, min_periods=1)
    b = _ts_mean(_col(), 20, min_periods=20)
    assert rolling_semantic_key(a) != rolling_semantic_key(b)
    new_roots, shared = apply_rolling_cse([a, b])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_r20_009_same_window_different_ddof_not_shared():
    a = _ts_std(_col(), 20, ddof=0)
    b = _ts_std(_col(), 20, ddof=1)
    assert rolling_semantic_key(a) != rolling_semantic_key(b)
    new_roots, shared = apply_rolling_cse([a, b])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_r20_009_same_input_different_price_basis_not_shared():
    a = _ts_mean(_col(), 20, semantic_attrs={"price_basis": "RAW"})
    b = _ts_mean(_col(), 20, semantic_attrs={"price_basis": "CONTINUOUS"})
    assert rolling_semantic_key(a) != rolling_semantic_key(b)
    new_roots, shared = apply_rolling_cse([a, b])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_r20_009_same_window_different_scalar_param_not_shared():
    # a rolling op with a second scalar literal (half-life-like) must key on it
    a = PlanNode(op="ts_decay_exp_window", inputs=[_col(), _lit(10), _lit(0.5)], attrs={})
    b = PlanNode(op="ts_decay_exp_window", inputs=[_col(), _lit(10), _lit(0.9)], attrs={})
    assert rolling_semantic_key(a) != rolling_semantic_key(b)
    new_roots, shared = apply_rolling_cse([a, b])
    assert not _has_plan_ref(new_roots[0])
    assert not _has_plan_ref(new_roots[1])
    assert len(shared) == 0


def test_r20_009_shared_only_when_fully_equivalent():
    # two FULLY equivalent rolling subtrees (incl. output semantics) DO share
    a = _ts_mean(_col(), 20, semantic_attrs=dict(_SEM))
    b = _ts_mean(_col(), 20, semantic_attrs=dict(_SEM))
    assert rolling_semantic_key(a) == rolling_semantic_key(b)
    new_roots, shared = apply_rolling_cse([a, b])
    assert len(shared) >= 1


# ---------------------------------------------------------------------------
# R20-010..013: ROLLING_OPS 派生集合一致性与窗口解析
# ---------------------------------------------------------------------------
def test_r20_010_rolling_ops_derived_set_consistency():
    # registry is warmed up by the module-level time_series/statistics imports;
    # a full load_all is avoided (slow + concurrent-session-broken).
    from cleaned_operators.registry import OperatorRegistry

    refreshed = refresh_rolling_ops()
    derived = _derive_rolling_ops()
    assert derived, "rolling-op modules failed to register; registry is empty"
    # ROLLING_OPS must equal the freshly derived canonical set (memoization,
    # not a hand-written list).
    assert set(refreshed) == set(derived)
    assert set(rolling_cache_mod.ROLLING_OPS) == set(derived)
    # every bootstrap base op must be present through its canonical resolution
    for op in _BASE_ROLLING_OPS:
        canonical = OperatorRegistry.resolve_canonical(op)
        if canonical in OperatorRegistry.list_canonical():
            assert canonical in rolling_cache_mod.ROLLING_OPS or op in rolling_cache_mod.ROLLING_OPS
    # every op in ROLLING_OPS must itself be a rolling operator
    for op in rolling_cache_mod.ROLLING_OPS:
        assert is_rolling_operator(op)


def test_r20_010_is_rolling_operator_resolves_aliases():
    refresh_rolling_ops()
    assert is_rolling_operator("ts_mean")
    assert is_rolling_operator("ts_std_dev")  # alias -> ts_std
    assert is_rolling_operator("ts_correlation")  # alias -> ts_corr
    assert not is_rolling_operator("log")
    assert not is_rolling_operator("add")
    assert not is_rolling_operator("column")


def test_r20_010_derived_set_extends_beyond_base():
    # The registry-derived set must cover ops that were never in the hand-written
    # bootstrap list (proves derivation, not just re-hashing the base set).
    from cleaned_operators.registry import OperatorRegistry

    refresh_rolling_ops()
    derived = set(rolling_cache_mod.ROLLING_OPS)
    # every bootstrap base op (alias or canonical) resolves into the derived set
    base_canonicals = {OperatorRegistry.resolve_canonical(op) for op in _BASE_ROLLING_OPS}
    for op in _BASE_ROLLING_OPS:
        assert OperatorRegistry.resolve_canonical(op) in derived
    extra = derived - base_canonicals
    assert extra, "derived rolling set must extend beyond the bootstrap base set"
    for op in sorted(extra)[:3]:
        assert is_rolling_operator(op)


def test_r20_011_window_parsing_uses_declared_history_param():
    # ts_mean declares window-ish param -> read "window" / fall back to d/n
    assert _window_from_attrs("ts_mean", {"window": 20}) == 20
    assert _window_from_attrs("ts_mean", {"d": 20}) == 20
    # ts_delay declares "n" as its lag param
    assert _window_from_attrs("ts_delay", {"n": 5}) == 5
    assert _window_from_attrs("ts_delta", {"n": 3}) == 3


# ---------------------------------------------------------------------------
# R20-459: SQL materialized_series / composite lowering 语义保留
# ---------------------------------------------------------------------------
def test_r20_459_sql_materialized_series_carries_semantic_attrs(monkeypatch):
    from planner import sql_lowerer

    col = _col()
    mean = PlanNode(
        op="ts_mean", inputs=[col], attrs={"window": 20},
        semantic_attrs=dict(_SEM),
    )
    root = PlanNode(
        op="divide", inputs=[mean, col], attrs={},
        semantic_attrs=dict(_SEM),
    )
    # Force the ts_mean subtree to be treated as SQL-extractable without needing
    # a fully-loaded SQL registry (concurrent-session proof).
    monkeypatch.setattr(
        sql_lowerer,
        "_is_sql_capable",
        lambda node, *, mode="production": False,
    )
    monkeypatch.setattr(
        sql_lowerer,
        "_is_extractable_sql_subtree",
        lambda node, *, mode="production": node.op == "ts_mean",
    )
    pp = sql_lowerer.lower_to_physical_plan(root, mode="production")
    assert not pp.fully_sql

    placeholder = None
    for n in _walk(pp.root):
        if n.op == "materialized_series":
            placeholder = n
    assert placeholder is not None
    sid = placeholder.attrs["sid"]
    extracted = pp.sql_subtrees[sid]
    # materialized_series semantic attrs == extracted subtree root semantic attrs
    assert dict(placeholder.semantic_attrs) == dict(extracted.semantic_attrs)
    assert placeholder.semantic_attrs["price_basis"] == "RAW"
    # the rebuilt enclosing node also keeps its semantic contract
    assert pp.root.semantic_attrs["unit"] == "return"


def test_r20_459_composite_lowering_propagates_semantic_attrs():
    col = _col()
    macd = PlanNode(
        op="MACD_line",
        inputs=[col],
        attrs={"fast": 12, "slow": 26, "signal": 9},
        semantic_attrs=dict(_SEM),
    )
    lowered = lower_composite_operators(macd)
    assert lowered.op == "subtract"
    assert dict(lowered.semantic_attrs) == _SEM
    assert lowered.semantic_attrs["price_basis"] == "RAW"
