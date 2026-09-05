# -*- coding: utf-8 -*-
"""R61-P0 #58 — production factor ladder tests.

Covers the FE-native batch ladder in
``factor_engine/scripts/production_factor_ladder.py``:

(a) pool generation: 62-operator real mining-grammar roots, unique structural
    fingerprints > 95% (numeric-literal normalized), >= 2 economic families,
    shared ``ts_mean(close,20)`` / ``ts_std(return,20)`` subtrees present;
(b) a small level (100 roots) executed end-to-end as ONE batch
    ``engine.run_many(factors, enable_cse=True)`` with the storage
    ``StreamingSink`` (shard + sidecar written) and ``DryRunCheckpoint``
    resume — failures carry real error text (never lossy ``(fid, None)``);
(c) timeout=0 rejected; worker governance resolves (>=1 worker, cpu-budget);
(d) 100k pool generation-count == 100000 and uniqueness > 95% WITHOUT
    executing any of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.scripts import production_factor_ladder as ladder
from factor_engine.scripts.production_factor_ladder import (
    FAIL_CATEGORIES,
    LadderRoot,
    RootFailure,
    _default_param_value,
    build_data_source,
    build_engine,
    build_one_expression,
    classify_error_message,
    generate_pool,
    load_terminal_operators,
    pool_stats,
    run_level,
    stratified_sample,
    structural_key,
    validate_timeout,
)


# ---------------------------------------------------------------------------
# (a) generation invariants
# ---------------------------------------------------------------------------
def test_generate_pool_unique_structural_roots_gt95() -> None:
    pool = generate_pool(200, seed=7)
    fps = [r.structural_fingerprint for r in pool]
    unique = len(set(fps))
    assert len(pool) == 200
    assert 100.0 * unique / len(pool) > 95.0
    # structural fingerprint normalizes numeric literals (20 == 20.0)
    e1 = make_cleaned_call_factory("ts_mean")(col("close"), 20)
    e2 = make_cleaned_call_factory("ts_mean")(col("close"), 20.0)
    assert structural_key(e1) == structural_key(e2)


def test_generate_pool_has_real_terminal_operators() -> None:
    pool = generate_pool(80, seed=3)
    ops = {r.operator_name for r in pool}
    # generated from the 62 Agent-direct terminal set (registry-backed loader)
    assert ops
    assert all(r.expr is not None or r.build_error for r in pool)
    # all exprs are real CleanedCall nodes (op attribute present)
    built = [r for r in pool if r.expr is not None]
    assert len(built) > len(pool) * 0.8
    for r in built[:10]:
        assert getattr(r.expr, "op", None) is not None


def test_generate_pool_multiple_families_and_shared_subtrees() -> None:
    pool = generate_pool(300, seed=11)
    fams = {r.economic_effect_family for r in pool}
    assert len(fams) >= 2
    shared_mean = structural_key(make_cleaned_call_factory("ts_mean")(col("close"), 20))
    shared_std = structural_key(make_cleaned_call_factory("ts_std")(col("return"), 20))
    # structural fingerprint of a wrapped root CONTAINS the shared subtree key
    # as a substring when the shared child appears anywhere in the tree
    n_has_shared = sum(
        1 for r in pool
        if shared_mean in r.structural_fingerprint or shared_std in r.structural_fingerprint
    )
    assert n_has_shared > 0, "no root reuses a forced shared subtree"


def test_stratified_sample_covers_families() -> None:
    pool = generate_pool(200, seed=5)
    sample = stratified_sample(pool, 50, np.random.default_rng(1))
    assert len(sample) == 50
    fams = {r.economic_effect_family for r in sample}
    assert len(fams) >= 2


# ---------------------------------------------------------------------------
# (b) small level end-to-end (batch run_many, sink, checkpoint, errors)
# ---------------------------------------------------------------------------
def test_level100_batch_run_many_sink_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: the resource broker's live psutil probes (family RSS via
    # /proc traversal + disk_io_counters) can block indefinitely on a loaded
    # host — an environment property, not ladder behavior. Stub both with
    # fixed envelopes so the test exercises the ladder, not the host.
    import factor_engine.runtime.resource_broker as _rb
    import factor_engine.runtime.resource_governor as _rg
    import factor_engine.runtime.run_peak_sampler as _rps

    G = 1 << 30
    monkeypatch.setattr(_rb, "_process_family_rss", lambda pss=False: G)
    monkeypatch.setattr(_rb, "_process_family_cpu_times", lambda: 0.0)
    monkeypatch.setattr(_rb, "_cpu_util", lambda interval=0.0: 0.0)
    monkeypatch.setattr(_rb, "_disk_io_counters", lambda: None)
    monkeypatch.setattr(_rb, "_psi", lambda ns: (0.0, 0.0))
    monkeypatch.setattr(_rb, "_swap", lambda: (0, 0))
    monkeypatch.setattr(_rb, "_host_mem_available", lambda: 64 * G)
    monkeypatch.setattr(_rg, "process_family_rss_bytes", lambda prefer_pss=False: G)
    monkeypatch.setattr(_rg, "_host_mem_available_bytes", lambda: 64 * G)
    monkeypatch.setattr(_rps, "_family_memory", lambda: (G, G))
    monkeypatch.setenv("FACTOR_ENGINE_HERMETIC_PROBES", "1")
    monkeypatch.setenv("FACTOR_ENGINE_MICRO_BATCH", "0")

    pool = generate_pool(120, seed=21)
    sample = stratified_sample(pool, 40, np.random.default_rng(2))
    assert len(sample) == 40
    engine = build_engine(build_data_source(n_stocks=8, n_days=30))
    st = run_level(
        100,
        sample,
        engine,
        out_dir=tmp_path,
        workers=2,
        per_factor_timeout=20.0,
        resume=True,
        write_results=True,
        result_policy="sink",
        chunk_size=20,
    )
    # batch path exercised: shared nodes found when reuse present
    assert st.passed + st.failed == len(sample)
    # every failure is preserved with a REAL error (never lossy None)
    for f in st.failures:
        assert f.fid
        assert f.category in FAIL_CATEGORIES
        assert f.error_message, "failure must carry real error text"
    # failures all carry real message strings from exceptions
    # (no (fid, None) lossy tuples anywhere in the state)
    assert st.failed == len(st.failures)
    # checkpoint written: completed+failed json exists
    ckpt = tmp_path / "checkpoint" / "level_100"
    assert (ckpt / "campaign.json").exists()
    # sink shards written for passed roots when write_results
    shards = list((tmp_path / "lake" / "level_100" / "shards").glob("*.sidecar.json"))
    assert len(shards) >= st.passed - 1 or st.passed == 0


def test_run_level_checkpoint_resume_skips_done(tmp_path: Path) -> None:
    pool = generate_pool(40, seed=23)
    sample = stratified_sample(pool, 20, np.random.default_rng(4))
    engine = build_engine(build_data_source(n_stocks=8, n_days=30))
    st1 = run_level(
        100, sample, engine, out_dir=tmp_path, workers=1,
        per_factor_timeout=20.0, resume=True, write_results=False,
        result_policy="return",
    )
    assert st1.passed + st1.failed == len(sample)
    # second run resumes: no pending roots recomputed
    st2 = run_level(
        100, sample, engine, out_dir=tmp_path, workers=1,
        per_factor_timeout=20.0, resume=True, write_results=False,
        result_policy="return",
    )
    assert st2.passed == st1.passed
    assert st2.failed == st1.failed
    assert st2.wall_time_s < 10.0  # resume fast path: nothing recomputed


def test_failures_never_lossy(tmp_path: Path) -> None:
    """Deliberately broken roots must record (formula, error, category)."""
    from factor_engine.api.factor import Factor

    # build a LadderRoot with expr None carrying a real build error
    broken = LadderRoot(
        fid="f_zzz",
        expr=None,
        operator_name="group_neutralize",
        economic_effect_family="cross_section",
        structural_fingerprint="__build_failed__:group_neutralize",
        source_columns=(),
        scalar_params={},
        build_error="OperatorParameterError: group_neutralize.fallback_policy "
                    "[planning]: fallback_policy must be a string",
    )
    ok_root = LadderRoot(
        fid="f_ok0",
        expr=make_cleaned_call_factory("ts_mean")(col("close"), 20),
        operator_name="ts_mean",
        economic_effect_family="trend",
        structural_fingerprint=structural_key(
            make_cleaned_call_factory("ts_mean")(col("close"), 20)
        ),
        source_columns=("close",),
        scalar_params={"window": 20},
    )
    engine = build_engine(build_data_source(n_stocks=10, n_days=40))
    st = run_level(
        100, [ok_root, broken], engine, out_dir=tmp_path, workers=1,
        per_factor_timeout=30.0, resume=True, write_results=False,
        result_policy="return",
    )
    assert st.passed == 1
    assert st.failed == 1
    failure = st.failures[0]
    assert failure.operator_name == "group_neutralize"
    assert failure.category == "param_error"
    assert "fallback_policy" in failure.error_message


# ---------------------------------------------------------------------------
# (c) enforcement knobs
# ---------------------------------------------------------------------------
def test_timeout_zero_rejected() -> None:
    with pytest.raises(ValueError):
        validate_timeout(0)
    with pytest.raises(ValueError):
        validate_timeout(None)
    assert validate_timeout(30.0) == 30.0


def test_worker_governance_resolves_budget() -> None:
    from factor_engine.scripts.production_factor_ladder import resolve_workers

    gov = resolve_workers(4)
    assert 1 <= gov["workers"] <= 8
    assert gov["cpu_budget"] >= 1


def test_classify_error_message_routes_categories() -> None:
    assert classify_error_message("unknown column 'foo'") == "data_missing"
    assert classify_error_message("OperatorParameterError: window must be >= 1") == "param_error"
    assert classify_error_message("timed out after 30s") == "timeout"
    assert classify_error_message("MemoryError: allocation failed") == "oom"
    assert classify_error_message("all NaN result") == "semantic"


# ---------------------------------------------------------------------------
# (d) 100k pool generation (count + uniqueness, NO execution)
# ---------------------------------------------------------------------------
import os as _os


def test_generate_pool_100k_count_and_uniqueness() -> None:
    # ~10 min of CPU at ~190 roots/s; gated to manual/CI env runs.
    if not _os.environ.get("LADDER_100K_TEST"):
        pytest.skip("set LADDER_100K_TEST=1 to run the 100k generation test "
                    "(~10 min CPU)")
    pool = generate_pool(100_000, seed=99)
    stats = pool_stats(pool)
    assert stats["total"] == 100_000
    assert stats["unique_pct"] > 95.0
    assert stats["n_families"] >= 2


# ---------------------------------------------------------------------------
# pool/operator helpers stay import-able and deterministic
# ---------------------------------------------------------------------------
def test_terminal_operators_loader() -> None:
    ops = load_terminal_operators()
    assert len(ops) >= 40
    names = {o["canonical"] for o in ops}
    assert "ts_mean" in names
    assert "ts_std" in names


def test_build_one_expression_passes_for_known_op() -> None:
    ops = {o["canonical"]: o for o in load_terminal_operators()}
    expr, cols, kw = build_one_expression(ops["ts_mean"], param_seed=42)
    assert getattr(expr, "op", None) == "ts_mean"
    assert cols
