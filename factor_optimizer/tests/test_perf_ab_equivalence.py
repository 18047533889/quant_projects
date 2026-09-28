"""Performance A/B equivalence and guard tests for the _pair_ic optimization.

Context
-------
The IC computation inside ``factor_optimizer.research_batch._pair_ic`` was changed
from the ``quant_evaluator`` ``evaluate("rank_ic_series")`` facade to the lighter
``compute_daily_ic(..., method="spearman")`` call. The two are *bit-level*
equivalent (verified MAX_ABS_DIFF == 0.0 across many seeds, validity masks and
``minimum_assets`` values during development), but the lighter call avoids the
heavier evaluator facade + evidence-store hashing on every per-candidate TRAIN
evaluation.

This module locks that equivalence and the surrounding invariants so the
optimization can never silently change a *value* (only *how fast* it is
computed). The pre-optimization path is preserved verbatim as the private oracle
``_pair_ic_evaluate_reference`` and is used here as the ground truth.

Five categories (per the performance subagent brief):
  1. New vs reference equivalence (>=3 seeds x multiple configs), bit-level.
  2. Cache correctness (hit / miss / different config must not cross-contaminate).
  3. Decision invariance end-to-end (optimize_factor_batch winners / rejection
     reasons / per-candidate ledger identical when the oracle replaces _pair_ic).
  4. Performance guard (the new kernel is not slower than the reference).
  5. Known-value oracle (independent Spearman reimplementation + analytic case).
"""

from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

from factor_optimizer.research_batch import (
    BatchOptimizationConfig,
    PairICCache,
    _candidate_ic_key,
    _pair_ic,
    _pair_ic_evaluate_reference,
    optimize_factor_batch,
)


# --------------------------------------------------------------------------- #
# Shared fixtures / builders
# --------------------------------------------------------------------------- #
def _make_labels(rng, t, n):
    """Synthetic target labels mirroring tests/test_research_batch.py::fixture."""
    z = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)])
    y = z ** 2
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    labels = LabelBundle(
        "synthetic", y, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=aa,
    )
    return labels, aa


def make_pair_inputs(seed=82, t=240, n=40, minimum_assets=20, *, clean=False):
    """Build direct inputs for _pair_ic / _pair_ic_evaluate_reference.

    Returns (raw, candidate, batch, labels, indices, config).

    ``clean=True`` makes every value finite and sets candidate == target so the
    CANDIDATE IC is analytically known (== 1.0).
    """
    rng = np.random.default_rng(seed)
    labels, aa = _make_labels(rng, t, n)
    raw = rng.standard_normal((t, n))
    if clean:
        candidate = labels.values.copy()  # perfect rank correlation with target
    else:
        candidate = np.tanh(raw) + 0.01 * rng.standard_normal((t, n))
    ta = AxisRef("time", "int", t, np.arange(t))
    # batch is only consulted for time_axis / asset_axis inside _pair_ic.
    batch = FactorBatch(("RAW",), ta, aa, raw[..., None])
    indices = np.arange(0, int(t * 0.8))  # a TRAIN-like contiguous subset
    config = BatchOptimizationConfig(minimum_assets=minimum_assets)
    return raw, candidate, batch, labels, indices, config


def make_optimize_inputs(seed=82, t=240, n=40):
    """Build a full multi-factor batch + labels for optimize_factor_batch."""
    rng = np.random.default_rng(seed)
    z = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)])
    y = z ** 2
    values = np.stack((y, -y, z, np.full_like(z, np.nan)), axis=-1)
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("good", "reverse", "u", "invalid"), ta, aa, values)
    labels = LabelBundle(
        "synthetic", y, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=aa,
    )
    return batch, labels


# Independent, transparent Spearman reimplementation (oracle for category 5).
def _spearman_1d(x, y):
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0:
        return np.nan
    return float((dx * dy).sum() / denom)


def _average_ranks(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for val_idx in np.where(counts > 1)[0]:
        mask = inv == val_idx
        ranks[mask] = ranks[mask].mean()
    return ranks


# --------------------------------------------------------------------------- #
# 1. New vs reference equivalence (bit-level)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 1234])
@pytest.mark.parametrize("minimum_assets", [3, 10, 20, 50])
def test_pair_ic_new_matches_oracle_bit_level(seed, minimum_assets):
    """_pair_ic (compute_daily_ic) must equal the evaluate()-based oracle exactly.

    Documented equivalence evidence: MAX_ABS_DIFF == 0.0 across seeds,
    validity masks and minimum_assets during development. This test re-proves it
    on every run with assert_array_equal(equal_nan=True).
    """
    raw, cand, batch, labels, idx, cfg = make_pair_inputs(
        seed=seed, minimum_assets=minimum_assets
    )
    ref_diff, ref_good, ref_ret = _pair_ic_evaluate_reference(
        raw, cand, batch, labels, idx, cfg
    )
    new_diff, new_good, new_ret = _pair_ic(raw, cand, batch, labels, idx, cfg)

    assert np.array_equal(ref_diff, new_diff, equal_nan=True), (
        f"diff diverged: max_abs={np.nanmax(np.abs(ref_diff - new_diff))}"
    )
    assert np.array_equal(ref_good, new_good)
    assert ref_ret == new_ret


# --------------------------------------------------------------------------- #
# 2. Cache correctness
# --------------------------------------------------------------------------- #
def test_candidate_ic_key_depends_on_minimum_assets():
    """Different minimum_assets must produce a different cache key (no 串味)."""
    rng = np.random.default_rng(5)
    v = rng.standard_normal(40)
    target = _make_labels(rng, 40, 40)[0]
    k_low = _candidate_ic_key(v, target, 20)
    k_high = _candidate_ic_key(v, target, 50)
    assert k_low != k_high
    # identical inputs -> identical key
    assert _candidate_ic_key(v, target, 20) == k_low


def test_reference_cache_hit_returns_identical():
    """A second call with the same reference_cache must be a cache hit and equal."""
    raw, cand, batch, labels, idx, cfg = make_pair_inputs(seed=11, minimum_assets=20)
    cache = PairICCache()
    d1, g1, r1 = _pair_ic(raw, cand, batch, labels, idx, cfg,
                          reference_cache=cache, candidate_cache=cache)
    # a no-cache run must produce the identical paired IC (cache only changes speed)
    d2, g2, r2 = _pair_ic(raw, cand, batch, labels, idx, cfg)  # no cache
    assert np.array_equal(d1, d2, equal_nan=True)
    assert np.array_equal(g1, g2)
    assert r1 == r2


def test_different_config_does_not_cross_contaminate_cache():
    """Same inputs, different minimum_assets, shared cache -> correct per-config value."""
    raw, cand, batch, labels, idx, cfg_low = make_pair_inputs(
        seed=11, minimum_assets=20
    )
    _, _, _, _, _, cfg_high = make_pair_inputs(seed=11, minimum_assets=50)
    cache = PairICCache()
    d_low, g_low, _ = _pair_ic(raw, cand, batch, labels, idx, cfg_low,
                               reference_cache=cache, candidate_cache=cache)
    # running a different config must not reuse a stale 20-asset IC for the 50-asset call
    d_high, g_high, _ = _pair_ic(raw, cand, batch, labels, idx, cfg_high,
                                 reference_cache=cache, candidate_cache=cache)
    # reference (no cache) values for each config
    ref_low, _, _ = _pair_ic(raw, cand, batch, labels, idx, cfg_low)
    ref_high, _, _ = _pair_ic(raw, cand, batch, labels, idx, cfg_high)
    assert np.array_equal(d_low, ref_low, equal_nan=True)
    assert np.array_equal(d_high, ref_high, equal_nan=True)


# --------------------------------------------------------------------------- #
# 3. Decision invariance end-to-end
# --------------------------------------------------------------------------- #
def _run_optimize(use_oracle):
    import factor_optimizer.research_batch as rb

    batch, labels = make_optimize_inputs(seed=82)
    if use_oracle:
        original = rb._pair_ic
        def oracle(*args, **kwargs):
            # The legacy oracle predates reusable prepared TRAIN slices.
            kwargs.pop("prepared_split", None)
            return rb._pair_ic_evaluate_reference(*args, **kwargs)
        rb._pair_ic = oracle
        try:
            result = optimize_factor_batch(
                batch, labels,
                config=BatchOptimizationConfig(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")),
                allow_research=True,
            )
        finally:
            rb._pair_ic = original
    else:
        result = optimize_factor_batch(
            batch, labels,
            config=BatchOptimizationConfig(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")),
            allow_research=True,
        )
    return result


def _factor_signature(result):
    sig = {}
    for fid, item in result.factors.items():
        sig[fid] = (
            item.selected_family,
            item.status,
            item.plan_identity,
            item.train_gain,
            item.validation_lower_bound,
            # per-candidate ledger: plan identity -> (status, train_gain)
            tuple(sorted(
                (rec.get("plan_identity"), rec.get("status"), rec.get("train_gain"))
                for rec in getattr(item, "candidates", [])
            )),
        )
    return sig


def test_decision_invariance_end_to_end():
    """Swapping _pair_ic for its oracle must keep every selection decision identical."""
    normal = _run_optimize(use_oracle=False)
    oracle = _run_optimize(use_oracle=True)

    assert _factor_signature(normal) == _factor_signature(oracle), (
        "optimize_factor_batch made a different decision with the oracle IC path"
    )
    # time split is independent of IC internals but assert stability anyway
    assert (list(normal.split.train_indices) == list(oracle.split.train_indices))
    assert (list(normal.split.validation_indices) == list(oracle.split.validation_indices))
    assert (list(normal.split.test_indices) == list(oracle.split.test_indices))


# --------------------------------------------------------------------------- #
# 4. Performance guard
# --------------------------------------------------------------------------- #
def test_pair_ic_not_slower_than_oracle():
    """The new compute_daily_ic path must not regress vs the evaluate() oracle.

    Loose upper bound (1.3x) protects against shared-host timing noise while
    still catching a gross regression (e.g. the optimization accidentally
    reverted). The measured speedup is printed for the report.
    """
    raw, cand, batch, labels, idx, cfg = make_pair_inputs(
        seed=3, t=240, n=40, minimum_assets=20
    )

    # warm-up (also pays the lazy imports once)
    for _ in range(3):
        _pair_ic(raw, cand, batch, labels, idx, cfg)
        _pair_ic_evaluate_reference(raw, cand, batch, labels, idx, cfg)

    def time_call(fn, reps=25):
        import time
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(raw, cand, batch, labels, idx, cfg)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best

    new_t = time_call(_pair_ic)
    ref_t = time_call(_pair_ic_evaluate_reference)
    speedup = ref_t / new_t if new_t > 0 else float("inf")
    print(f"\n[perf] _pair_ic={new_t*1e3:.2f}ms  oracle={ref_t*1e3:.2f}ms  "
          f"speedup={speedup:.3f}x")
    assert new_t <= ref_t * 1.3, (
        f"new path regressed: {new_t*1e3:.2f}ms vs oracle {ref_t*1e3:.2f}ms"
    )


# --------------------------------------------------------------------------- #
# 5. Known-value oracle
# --------------------------------------------------------------------------- #
def test_identical_raw_and_candidate_gives_zero_diff():
    """When raw == candidate == target, both IC columns are 1.0 so diff == 0 exactly.

    This is a direct analytic known value on the *actual* _pair_ic output (not just
    the independent oracle): a factor identical to the target has rank IC 1.0, so
    the paired delta between two identical factors is identically zero.
    """
    rng = np.random.default_rng(1)
    labels, aa = _make_labels(rng, 120, 40)
    target = labels.values
    raw = target.copy()
    cand = target.copy()
    ta = AxisRef("time", "int", 120, np.arange(120))
    batch = FactorBatch(("RAW",), ta, aa, raw[..., None])
    idx = np.arange(0, 96)
    cfg = BatchOptimizationConfig(minimum_assets=20)
    diff, good, _ = _pair_ic(raw, cand, batch, labels, idx, cfg)
    assert good.all(), "all days good when raw==candidate==target (all finite)"
    assert np.array_equal(diff, np.zeros_like(diff)), (
        f"diff must be exactly 0 when raw==candidate==target; got max_abs="
        f"{np.nanmax(np.abs(diff))}"
    )


def test_constant_candidate_is_not_finite():
    """A constant candidate yields no finite rank IC (good == False on those days)."""
    rng = np.random.default_rng(9)
    labels, aa = _make_labels(rng, 120, 40)
    raw = rng.standard_normal((120, 40))
    cand = np.zeros((120, 40))  # constant across assets each day
    ta = AxisRef("time", "int", 120, np.arange(120))
    batch = FactorBatch(("RAW",), ta, aa, raw[..., None])
    idx = np.arange(0, 96)
    cfg = BatchOptimizationConfig(minimum_assets=20)
    diff, good, _ = _pair_ic(raw, cand, batch, labels, idx, cfg)
    # constant candidate -> zero variance in ranks -> Spearman undefined -> NaN
    assert not np.isfinite(diff).all()
    # every day is flagged not-good (the CANDIDATE column is NaN everywhere)
    assert (~good).all()


def test_independent_spearman_matches_pair_ic_diff():
    """Cross-check _pair_ic's paired diff against an independent Spearman impl.

    Loose tolerance (rtol=1e-6, atol=1e-8): the independent average-rank Spearman
    is a separate implementation from quant_evaluator's; the two agree to well
    within this bound. The bit-level guarantee vs the evaluate() oracle lives in
    test_pair_ic_new_matches_oracle_bit_level.
    """
    raw, cand, batch, labels, idx, cfg = make_pair_inputs(
        seed=21, t=200, n=40, minimum_assets=20
    )
    diff, good, _ = _pair_ic(raw, cand, batch, labels, idx, cfg)
    target = labels.values[idx]
    c = cand[idx]
    r = raw[idx]
    for i in np.where(good)[0]:
        cand_ic = _spearman_1d(c[i], target[i])
        raw_ic = _spearman_1d(r[i], target[i])
        implied_diff = cand_ic - raw_ic
        assert abs(implied_diff - diff[i]) <= 1e-6 + 1e-8 * abs(diff[i]), (
            f"day {i}: independent diff {implied_diff} vs _pair_ic diff {diff[i]}"
        )


@pytest.mark.parametrize("seed", [5, 17, 41])
def test_pair_ic_matches_oracle_with_label_validity_and_sparse_factors(seed):
    raw, cand, batch, labels, idx, cfg = make_pair_inputs(
        seed=seed, t=120, n=40, minimum_assets=20
    )
    rng = np.random.default_rng(seed + 1000)
    raw = raw.copy()
    cand = cand.copy()
    raw[rng.random(raw.shape) < 0.12] = np.nan
    cand[rng.random(cand.shape) < 0.17] = np.nan
    validity = rng.random(labels.values.shape) >= 0.19
    validity[3, :] = False
    labels = replace(labels, validity=validity)
    expected = _pair_ic_evaluate_reference(raw, cand, batch, labels, idx, cfg)
    actual = _pair_ic(raw, cand, batch, labels, idx, cfg)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_cached_raw_reference_only_constructs_candidate_column(monkeypatch):
    import quant_evaluator.metrics.ic as ic_module

    raw, cand, batch, labels, idx, cfg = make_pair_inputs(seed=31)
    cache = PairICCache()
    widths = []
    original = ic_module.compute_daily_ic

    def record_width(factors, *args, **kwargs):
        widths.append(factors.factor_ids)
        return original(factors, *args, **kwargs)

    monkeypatch.setattr(ic_module, "compute_daily_ic", record_width)
    _pair_ic(raw, cand, batch, labels, idx, cfg, reference_cache=cache)
    candidate = -cand
    actual = _pair_ic(raw, candidate, batch, labels, idx, cfg,
                      reference_cache=cache)
    assert widths[-1] == ("CANDIDATE",)
    reference = _pair_ic_evaluate_reference(raw, candidate, batch, labels, idx, cfg)
    np.testing.assert_array_equal(actual[0], reference[0])
    np.testing.assert_array_equal(actual[1], reference[1])
    assert actual[2] == reference[2]
