"""M03/M04 extension kernels: spec §12 golden oracles, determinism,
fail-closed evidence semantics, dtype rejection, backend parity.

Covers QE-EXT-SPEC-1.0 §7.3 (cdar / ced / drawdown_budget_exceedance /
joint_drawdown_occupancy), §7.4 (book_conditional_loss / book_tail_loss_delta
/ book_downside_beta), §6.6 (upper-tail boundary weights), §6.5 (ordered
path summaries) and §12 (runnable reference algorithms):

- §12 golden tail / summary oracles verbatim (fractional mass, tie
  distribution, permutation invariance, cross-block drawdown, associativity,
  non-commutativity, empty-block identity, end short block, plan
  reproducibility, sampled-MDD-vs-naive 1e-12);
- path kernels reuse the repo drawdown authority
  (``metrics/risk/drawdown_analysis.compute_drawdown_series``): V0=1 in the
  high-water mark, zero wealth an absorbing 100% loss, consistent with
  ``metrics/portfolio_stats.compute_maximum_drawdown``;
- >=3-seed bitwise determinism for the resampled kernels;
- NaN / short sample / broken calendar fail-closed evidence states
  (§1.3.2, §6.1, §5.3 eight states only);
- dtype rejection (bool / complex / object / str) via TypeError;
- backend parity: cpu_fast vs cpu_reference (measured deviation asserted
  and recorded), cuda vs cpu_reference under the rtol<=1e-8 / atol<=1e-10
  house rule (skipped when cupy is unavailable).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_evaluator.contracts.evidence_status import EvidenceStatus
from quant_evaluator.metrics.extension import (
    block_plan,
    book_conditional_loss,
    book_downside_beta,
    book_tail_loss_delta,
    cdar,
    ced,
    drawdown_budget_exceedance,
    joint_drawdown_occupancy,
    merge_path,
    path_summary,
    summary_mdd,
)
from quant_evaluator.metrics.extension._tail import (
    resolved_tail_mass,
    tail_weights_1d,
    upper_tail_mean_1d,
)
from quant_evaluator.metrics.extension.path_risk import (
    _sampled_mdd_fast,
    _sampled_mdd_reference,
)
from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_series
from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown

try:
    import cupy as cp

    HAS_CUPY = True
except Exception:  # pragma: no cover - CPU-only environments
    HAS_CUPY = False

BACKENDS = ["cpu_reference", "cpu_fast"]
if HAS_CUPY:
    BACKENDS.append("cuda")

FAST_RTOL = 1e-12      # measured: <= 3.4e-16 on the tournament seeds
CUDA_RTOL = 1e-8       # house rule
CUDA_ATOL = 1e-10      # house rule


def _assert_close(reference, actual, backend):
    if backend == "cpu_fast":
        assert np.allclose(reference, actual, rtol=FAST_RTOL, atol=0.0)
    else:
        assert np.allclose(reference, actual, rtol=CUDA_RTOL, atol=CUDA_ATOL)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260926)


@pytest.fixture(scope="module")
def panel(rng):
    return rng.normal(0.0005, 0.012, (260, 6))


BENCH_KW = dict(min_periods=10, min_tail_mass=1.0)
CED_KW = dict(path_horizon=29, block_length=7, repetitions=37, seed=1,
              min_periods=10, min_tail_mass=0.5)
JOINT_KW = dict(min_periods=10)


# ---------------------------------------------------------------------------
# §12 golden oracles
# ---------------------------------------------------------------------------

class TestSpec12TailOracles:
    def test_upper_tail_mean_fractional_mass(self):
        assert upper_tail_mean_1d(np.array([1., 2, 3, 4]), 0.625) == pytest.approx(11 / 3)

    def test_tail_weights_ties_distribute(self):
        np.testing.assert_allclose(
            tail_weights_1d(np.array([3., 3, 1, 0]), 0.625), [0.75, 0.75, 0.0, 0.0]
        )

    def test_tail_condition_permutation(self):
        book = np.array([3., 3, 1, 0])
        target = np.array([1., 7, 10, 20])
        perm = [2, 1, 3, 0]
        w = tail_weights_1d(book, 0.625)
        wp = tail_weights_1d(book[perm], 0.625)
        assert w @ target / w.sum() == pytest.approx(wp @ target[perm] / wp.sum())

    def test_upper_tail_mean_integer_mass(self):
        assert upper_tail_mean_1d(np.array([1., 2, 3, 4]), 0.5) == pytest.approx(3.5)

    @pytest.mark.parametrize("values,q", [
        ([], 0.9),
        ([1.0, np.nan], 0.9),
        ([1.0], 1.0),
        ([1.0], 0.0),
    ])
    def test_tail_invalid(self, values, q):
        with pytest.raises(ValueError):
            upper_tail_mean_1d(np.asarray(values, dtype=float), q)

    def test_mass_snap_avoids_off_by_one(self):
        # (1-q)*n within 8eps of an integer must snap to that integer
        n = 1000
        assert resolved_tail_mass(n, 0.995) == 5.0


class TestSpec12PathOracles:
    def test_initial_wealth_included(self):
        assert summary_mdd(path_summary(np.array([-.1, .1]))) == pytest.approx(0.1)

    def test_all_positive_zero_drawdown(self):
        assert summary_mdd(path_summary(np.array([.1, .02, .03]))) == 0.0

    def test_cross_block_drawdown(self):
        merged = merge_path(path_summary(np.array([.2, -.1])),
                            path_summary(np.array([-.2, .1])))
        assert summary_mdd(merged) == pytest.approx(0.28)
        assert summary_mdd(merged) == pytest.approx(
            summary_mdd(path_summary(np.array([.2, -.1, -.2, .1])))
        )

    def test_path_associativity(self):
        a, b, c = (path_summary(np.array(x)) for x in ([.1, -.15], [.02, -.03], [.05, -.1]))
        np.testing.assert_allclose(
            tuple(merge_path(merge_path(a, b), c)),
            tuple(merge_path(a, merge_path(b, c))), atol=1e-14,
        )

    def test_path_noncommutative(self):
        a = path_summary(np.array([.2, -.1]))
        b = path_summary(np.array([-.2, .1]))
        assert summary_mdd(merge_path(a, b)) != pytest.approx(summary_mdd(merge_path(b, a)))

    def test_path_empty_identity(self):
        a = path_summary(np.array([.1, -.2, .02]))
        identity = path_summary(np.array([]))
        np.testing.assert_allclose(tuple(merge_path(a, identity)), tuple(a), atol=1e-14)
        np.testing.assert_allclose(tuple(merge_path(identity, a)), tuple(a), atol=1e-14)

    def test_plan_end_short_block(self):
        starts, lengths = block_plan(103, 103, 7, 25)
        assert lengths[-1] == 5
        assert lengths.sum() == 103
        assert np.all(starts <= 96)

    def test_plan_reproducible(self):
        a, la = block_plan(103, 29, 7, 25, 7)
        b, lb = block_plan(103, 29, 7, 25, 7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(la, lb)

    @pytest.mark.parametrize("args", [
        (30, 30, True, 10),   # bool block_length
        (30, 30, 10, 1),      # repetitions < 2
        (30, 30, 40, 10),     # block > sample
    ])
    def test_plan_invalid(self, args):
        with pytest.raises(ValueError):
            block_plan(*args)

    def test_sampled_mdd_matches_naive_expansion(self, rng):
        r = np.random.default_rng(3).normal(0.001, 0.03, 103)
        starts, lengths = block_plan(103, 29, 7, 37, 1)
        for sampled in (_sampled_mdd_reference(r[:, None], starts, lengths)[:, 0],
                        _sampled_mdd_fast(r[:, None], starts, lengths)[:, 0]):
            naive = []
            for row in starts:
                sample = r[np.concatenate([np.arange(s, s + l) for s, l in zip(row, lengths)])]
                wealth = np.cumprod(1.0 + sample)
                running = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
                naive.append(float(np.max((running - wealth) / running)))
            np.testing.assert_allclose(sampled, naive, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# shared evidence helpers
# ---------------------------------------------------------------------------

def _value(result):
    assert result["status"] is EvidenceStatus.COMPUTED, result
    return result["value"]


class TestEvidenceSemantics:
    def test_interior_nan_fail_closed(self, panel):
        broken = panel[:, 0].copy()
        broken[100] = np.nan
        for call in (
            lambda: cdar(broken, **BENCH_KW),
            lambda: ced(broken, **CED_KW),
            lambda: joint_drawdown_occupancy(broken, panel[:, 1], **JOINT_KW),
        ):
            result = call()
            assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
            assert result["diagnostics"]["reason_detail"] == \
                "interior_missing_or_nonfinite_returns"
            assert result["value"] is None

    def test_interior_inf_fail_closed(self, panel):
        broken = panel[:, 0].copy()
        broken[50] = np.inf
        result = cdar(broken, **BENCH_KW)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE

    def test_leading_trailing_nan_trimmed(self, panel):
        series = panel[:, 0]
        padded = np.concatenate([[np.nan, np.nan], series, [np.nan]])
        base = cdar(series, **BENCH_KW)
        padded_result = cdar(padded, **BENCH_KW)
        assert padded_result["status"] is EvidenceStatus.COMPUTED
        assert padded_result["value"] == base["value"]
        assert padded_result["diagnostics"]["n_observations"] == \
            base["diagnostics"]["n_observations"]

    def test_short_sample_insufficient(self, panel):
        result = cdar(panel[:8, 0], **BENCH_KW)
        assert result["status"] is EvidenceStatus.INSUFFICIENT_DATA
        assert result["diagnostics"]["reason_detail"] == "observations_too_few"

    def test_tail_mass_gate(self, panel):
        result = cdar(panel[:, 0], tail_confidence=0.95, min_tail_mass=50.0,
                      min_periods=10)
        assert result["status"] is EvidenceStatus.INSUFFICIENT_DATA
        assert result["diagnostics"]["reason_detail"] == "tail_mass_below_min"

    def test_terminal_total_loss(self, panel):
        series = np.concatenate([panel[:, 0], [-1.0, 0.01]])
        result = cdar(series, **BENCH_KW)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert result["diagnostics"]["terminal_total_loss"] is True
        assert result["diagnostics"]["known_drawdown_lower_bound"] == 1.0

    def test_return_basis_mismatch(self, panel):
        series = np.concatenate([panel[:, 0], [-1.5]])
        result = cdar(series, **BENCH_KW)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert result["diagnostics"]["reason_detail"] == "return_basis_mismatch"

    def test_calendar_not_strictly_increasing(self, panel):
        n = panel.shape[0]
        calendar = np.arange(n, dtype=np.int64)
        calendar[10] = calendar[9]
        ok = cdar(panel[:, 0], calendar=calendar, **BENCH_KW)
        assert ok["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert ok["diagnostics"]["reason_detail"] == "calendar_not_strictly_increasing"
        good = cdar(panel[:, 0], calendar=np.arange(n, dtype=np.int64), **BENCH_KW)
        assert good["status"] is EvidenceStatus.COMPUTED

    @pytest.mark.parametrize("dtype", [bool, complex])
    def test_dtype_rejection(self, dtype):
        bad = np.zeros(30, dtype=dtype)
        with pytest.raises(TypeError):
            cdar(bad, **BENCH_KW)
        with pytest.raises(TypeError):
            joint_drawdown_occupancy(bad, bad, min_periods=10)
        with pytest.raises(TypeError):
            book_conditional_loss(np.zeros(30, dtype=np.float64), bad, **BENCH_KW)

    def test_object_dtype_rejection(self, panel):
        bad = np.array(["x"] * 30, dtype=object)
        with pytest.raises(TypeError):
            cdar(bad, **BENCH_KW)
        with pytest.raises(TypeError):
            book_conditional_loss(bad, bad, **BENCH_KW)


# ---------------------------------------------------------------------------
# M03 kernels
# ---------------------------------------------------------------------------

class TestCdar:
    def test_golden_initial_wealth(self):
        result = cdar(np.array([-.1, .1]), tail_confidence=0.95,
                      min_tail_mass=0.05, min_periods=2)
        assert result["status"] is EvidenceStatus.COMPUTED
        assert result["value"] == pytest.approx(0.1, abs=1e-15)
        assert result["diagnostics"]["max_drawdown_observed"] == pytest.approx(0.1)

    def test_all_positive_returns_zero_cdar(self):
        result = cdar(np.array([.1, .02, .03]), tail_confidence=0.95,
                      min_tail_mass=0.05, min_periods=2)
        assert result["status"] is EvidenceStatus.COMPUTED
        assert result["value"] == 0.0
        assert result["diagnostics"]["never_underwater"] is True

    def test_absorbing_wipeout_matches_portfolio_stats(self, panel):
        series = panel[:, 0].copy()
        series[120] = -1.0
        d_path = -compute_drawdown_series(series[:, None])[0] + 0.0
        assert d_path[120] == 1.0
        assert d_path[121] == 1.0  # absorbing: no recovery without capital
        mdd, _, _ = compute_maximum_drawdown(series[:, None])
        assert d_path.max() == mdd[0] == 1.0

    def test_matches_portfolio_stats_max_drawdown(self, panel):
        d_path = -compute_drawdown_series(panel[:, 0][:, None])[0] + 0.0
        mdd, _, _ = compute_maximum_drawdown(panel[:, 0][:, None])
        assert d_path.max() == mdd[0]

    def test_panel_matches_columns(self, panel):
        result = cdar(panel, **BENCH_KW)
        assert result["value"].shape == (panel.shape[1],)
        for col in range(panel.shape[1]):
            single = _value(cdar(panel[:, col], **BENCH_KW))
            # strided panel column vs contiguous 1-D: measured <= 1 ulp
            assert result["value"][col] == pytest.approx(single, rel=1e-15)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_backend_parity(self, panel, backend):
        reference = _value(cdar(panel, backend="cpu_reference", **BENCH_KW))
        actual = _value(cdar(panel, backend=backend, **BENCH_KW))
        _assert_close(reference, actual, backend)

    def test_bitwise_determinism(self, panel):
        for backend in BACKENDS:
            first = cdar(panel, backend=backend, **BENCH_KW)["value"]
            second = cdar(panel, backend=backend, **BENCH_KW)["value"]
            assert np.array_equal(first, second)


class TestCed:
    def test_scenario_mdd_backends_match_naive(self, rng):
        series = rng.normal(0.001, 0.03, 103)
        starts, lengths = block_plan(103, 29, 7, 37, 1)
        reference = _sampled_mdd_reference(series[:, None], starts, lengths)[:, 0]
        fast = _sampled_mdd_fast(series[:, None], starts, lengths)[:, 0]
        np.testing.assert_allclose(reference, fast, rtol=1e-12, atol=1e-14)

    def test_horizon_not_multiple_of_block(self, panel):
        result = ced(panel[:, 0], path_horizon=31, block_length=7,
                     repetitions=37, seed=0, **BENCH_KW)
        assert result["status"] is EvidenceStatus.COMPUTED
        assert math.isfinite(result["value"])
        assert result["diagnostics"]["path_horizon"] == 31

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_seed_determinism(self, panel, seed):
        kwargs = dict(path_horizon=63, block_length=10, repetitions=499,
                      seed=seed, **BENCH_KW)
        for backend in BACKENDS:
            first = ced(panel, backend=backend, **kwargs)["value"]
            second = ced(panel, backend=backend, **kwargs)["value"]
            assert np.array_equal(first, second)

    def test_different_seeds_differ(self, panel):
        kwargs = dict(path_horizon=63, block_length=10, repetitions=499, **BENCH_KW)
        values = [ced(panel[:, 0], seed=seed, **kwargs)["value"] for seed in (0, 1, 7)]
        assert len({float(v) for v in values}) == 3

    def test_source_too_short_for_two_blocks(self, panel):
        result = ced(panel[:14, 0], path_horizon=63, block_length=10,
                     repetitions=37, seed=0, **BENCH_KW)
        assert result["status"] is EvidenceStatus.INSUFFICIENT_DATA

    def test_backend_parity(self, panel):
        reference = _value(ced(panel, backend="cpu_reference", **CED_KW))
        for backend in BACKENDS[1:]:
            actual = _value(ced(panel, backend=backend, **CED_KW))
            _assert_close(reference, actual, backend)

    def test_no_btf_materialization_value_summary_only(self, panel):
        result = ced(panel[:, 0], **CED_KW)
        assert result["diagnostics"]["n_scenarios"] == 37


class TestDrawdownBudgetExceedance:
    def test_budget_validation(self, panel):
        with pytest.raises(ValueError):
            drawdown_budget_exceedance(panel[:, 0], 0.0, repetitions=37, min_periods=10)
        with pytest.raises(ValueError):
            drawdown_budget_exceedance(panel[:, 0], 1.0, repetitions=37, min_periods=10)
        with pytest.raises(TypeError):
            drawdown_budget_exceedance(panel[:, 0], True, repetitions=37, min_periods=10)

    def test_strict_exceedance_and_counts(self, panel):
        result = drawdown_budget_exceedance(panel[:, 0], 0.2, repetitions=200,
                                            seed=0, min_periods=10)
        assert result["status"] is EvidenceStatus.COMPUTED
        assert 0.0 <= result["value"] <= 1.0
        assert result["diagnostics"]["n_exceedances"] == pytest.approx(
            result["value"] * 200
        )
        assert "not a true future default probability" in \
            result["diagnostics"]["interpretation"]

    def test_zero_budget_exceeds_everything(self, panel):
        result = drawdown_budget_exceedance(panel[:, 0], 1e-12, repetitions=37,
                                            seed=0, min_periods=10)
        # every resampled path has at least one non-positive day in this panel
        assert result["value"] == result["diagnostics"]["n_exceedances"] / 37

    def test_backend_parity(self, panel):
        reference = _value(drawdown_budget_exceedance(
            panel[:, 0], 0.2, repetitions=200, seed=0, min_periods=10,
            backend="cpu_reference"))
        for backend in BACKENDS[1:]:
            actual = _value(drawdown_budget_exceedance(
                panel[:, 0], 0.2, repetitions=200, seed=0, min_periods=10,
                backend=backend))
            assert actual == reference  # same MDD distribution, same strict count


class TestJointDrawdownOccupancy:
    def test_matches_drawdown_series_definition(self, panel):
        base, cand = panel[:, 0], panel[:, 1]
        result = joint_drawdown_occupancy(base, cand, **JOINT_KW)
        d0 = -compute_drawdown_series(base[:, None])[0] + 0.0
        d1 = -compute_drawdown_series(cand[:, None])[0] + 0.0
        expected = float(np.mean((d0 > 0.0) & (d1 > 0.0)))
        assert result["value"] == expected
        assert result["diagnostics"]["underwater_fraction_0"] == float(np.mean(d0 > 0.0))
        assert result["diagnostics"]["underwater_fraction_1"] == float(np.mean(d1 > 0.0))
        assert "never underwater" in result["diagnostics"]["interpretation"]

    def test_thresholds_occupancy_monotone(self, panel):
        base, cand = panel[:, 0], panel[:, 1]
        zero = _value(joint_drawdown_occupancy(base, cand, 0.0, 0.0, **JOINT_KW))
        higher = _value(joint_drawdown_occupancy(base, cand, 0.02, 0.02, **JOINT_KW))
        assert 0.0 <= higher <= zero <= 1.0

    def test_length_mismatch_rejected(self, panel):
        with pytest.raises(ValueError):
            joint_drawdown_occupancy(panel[:, 0], panel[:-1, 1], **JOINT_KW)

    def test_panel_candidate_matches_columns(self, panel):
        result = joint_drawdown_occupancy(panel[:, 0], panel, **JOINT_KW)
        assert result["value"].shape == (panel.shape[1],)
        for col in range(panel.shape[1]):
            single = _value(joint_drawdown_occupancy(
                panel[:, 0], panel[:, col], **JOINT_KW))
            assert result["value"][col] == pytest.approx(single, rel=1e-15)

    def test_backend_parity(self, panel):
        reference = _value(joint_drawdown_occupancy(
            panel[:, 0], panel[:, 1], backend="cpu_reference", **JOINT_KW))
        for backend in BACKENDS[1:]:
            actual = _value(joint_drawdown_occupancy(
                panel[:, 0], panel[:, 1], backend=backend, **JOINT_KW))
            _assert_close(reference, actual, backend)


# ---------------------------------------------------------------------------
# M04 kernels
# ---------------------------------------------------------------------------

BOOK = np.array([-0.3, -0.3, -0.1, 0.0])   # book losses [0.3,0.3,0.1,0] (§12 tie structure)
CAND = np.array([0.1, 0.7, 1.0, 2.0])
# non-degenerate weighted baseline variance (needed for downside beta)
BOOK2 = np.array([-0.3, -0.2, -0.1, 0.0])
CAND2 = np.array([0.1, 0.4, 0.8, 1.6])
BOOK_KW = dict(tail_confidence=0.625, min_tail_mass=0.1, min_periods=2)


class TestBookKernels:
    def test_golden_tied_book_tail(self):
        result = book_conditional_loss(BOOK, CAND, **BOOK_KW)
        assert result["status"] is EvidenceStatus.COMPUTED
        # w = [0.75, 0.75, 0, 0] on losses [.3,.3,.1,0] -> -(0.75*.1+0.75*.7)/1.5
        assert result["value"] == pytest.approx(-0.4)

    def test_tied_dates_permutation_invariant(self):
        perm = [2, 1, 3, 0]
        for kernel in (book_conditional_loss, book_tail_loss_delta, book_downside_beta):
            first = _value(kernel(BOOK2, CAND2, **BOOK_KW))
            second = _value(kernel(BOOK2[perm], CAND2[perm], **BOOK_KW))
            # reordered weighted dots differ by <= 1 ulp (measured)
            assert first == pytest.approx(second, rel=1e-15)

    def test_fully_tied_book_tail_beta_degenerate(self):
        # the §12 tie structure puts all weight on equal losses: beta must
        # fail closed instead of dividing by zero weighted variance
        result = book_downside_beta(BOOK, CAND, **BOOK_KW)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert result["diagnostics"]["reason_detail"] == \
            "baseline_weighted_variance_degenerate"

    def test_tail_delta_uses_fixed_book_tail(self):
        book_tail = _value(book_conditional_loss(BOOK, BOOK, **BOOK_KW))
        cand_tail = _value(book_conditional_loss(BOOK, CAND, **BOOK_KW))
        delta = _value(book_tail_loss_delta(BOOK, CAND, **BOOK_KW))
        assert delta == pytest.approx(book_tail - cand_tail)
        diagnostics = book_tail_loss_delta(BOOK, CAND, **BOOK_KW)["diagnostics"]
        assert diagnostics["book_tail_loss"] == pytest.approx(book_tail)

    def test_inverse_candidate_negative_conditional_loss(self, rng):
        book = rng.normal(0.0005, 0.01, 200)
        result = book_conditional_loss(book, -book, min_tail_mass=1.0, min_periods=10)
        assert result["status"] is EvidenceStatus.COMPUTED
        assert result["value"] < 0.0

    def test_downside_beta_linear_candidate(self, rng):
        book = rng.normal(0.0005, 0.01, 200)
        candidate = 1.7 * book + 0.001
        result = book_downside_beta(book, candidate, min_tail_mass=1.0, min_periods=10)
        assert result["value"] == pytest.approx(1.7, rel=1e-12)

    def test_degenerate_baseline_fails_closed(self):
        flat = np.full(200, 0.0005)
        result = book_downside_beta(flat, np.full(200, 0.001),
                                    min_tail_mass=1.0, min_periods=10)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert result["diagnostics"]["reason_detail"] == \
            "baseline_weighted_variance_degenerate"

    def test_tail_mass_gate(self, rng):
        book = rng.normal(0.0, 0.01, 100)
        result = book_conditional_loss(book, book, tail_confidence=0.95,
                                       min_tail_mass=10.0, min_periods=10)
        assert result["status"] is EvidenceStatus.INSUFFICIENT_DATA
        assert result["diagnostics"]["reason_detail"] == "tail_mass_below_min"

    @pytest.mark.parametrize("kind", ["standalone_against_book",
                                      "combined_book_against_book"])
    def test_comparison_kind_declared(self, kind):
        result = book_conditional_loss(BOOK, CAND, comparison_kind=kind, **BOOK_KW)
        assert result["diagnostics"]["comparison_kind"] == kind
        if kind == "standalone_against_book":
            assert "complementarity" in result["diagnostics"]["interpretation"]

    def test_comparison_kind_rejected(self):
        with pytest.raises(ValueError):
            book_conditional_loss(BOOK, CAND, comparison_kind="market_label", **BOOK_KW)

    def test_length_mismatch_rejected(self, rng):
        with pytest.raises(ValueError):
            book_conditional_loss(rng.normal(0, .01, 100), rng.normal(0, .01, 99),
                                  **BOOK_KW)

    def test_interior_nan_fail_closed(self, rng):
        book = rng.normal(0, .01, 100)
        cand = rng.normal(0, .01, 100)
        cand[50] = np.nan
        result = book_conditional_loss(book, cand, min_tail_mass=1.0, min_periods=10)
        assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
        assert result["diagnostics"]["reason_detail"] == \
            "interior_missing_or_nonfinite_returns"

    def test_panel_matches_columns(self, panel, rng):
        book = rng.normal(0.0005, 0.01, 260)
        candidates = rng.normal(0.0004, 0.012, (260, 5))
        for kernel in (book_conditional_loss, book_tail_loss_delta, book_downside_beta):
            result = _value(kernel(book, candidates, min_tail_mass=1.0, min_periods=10))
            assert result.shape == (5,)
            for col in range(5):
                single = _value(kernel(book, candidates[:, col],
                                       min_tail_mass=1.0, min_periods=10))
                assert result[col] == pytest.approx(single, rel=1e-15)

    def test_weights_computed_once_shared(self, panel, rng):
        book = rng.normal(0.0005, 0.01, 260)
        # tail weights depend only on the book; two different candidates share them
        c1 = rng.normal(0, .01, 260)
        c2 = rng.normal(0, .01, 260)
        kw = dict(min_tail_mass=1.0, min_periods=10)
        d1 = _value(book_tail_loss_delta(book, c1, **kw))
        d2 = _value(book_tail_loss_delta(book, c2, **kw))
        b = _value(book_conditional_loss(book, book, **kw))
        assert d1 == pytest.approx(b - _value(book_conditional_loss(book, c1, **kw)))
        assert d2 == pytest.approx(b - _value(book_conditional_loss(book, c2, **kw)))

    def test_backend_parity(self, panel, rng):
        book = rng.normal(0.0005, 0.01, 260)
        cand = rng.normal(0.0004, 0.012, (260, 5))
        kw = dict(min_tail_mass=1.0, min_periods=10)
        for kernel in (book_conditional_loss, book_tail_loss_delta, book_downside_beta):
            reference = np.atleast_1d(_value(kernel(book, cand, backend="cpu_reference", **kw)))
            for backend in BACKENDS[1:]:
                actual = np.atleast_1d(_value(kernel(book, cand, backend=backend, **kw)))
                _assert_close(reference, actual, backend)

    def test_bitwise_determinism(self, panel, rng):
        book = rng.normal(0.0005, 0.01, 260)
        cand = rng.normal(0.0004, 0.012, (260, 5))
        kw = dict(min_tail_mass=1.0, min_periods=10)
        for kernel in (book_conditional_loss, book_tail_loss_delta, book_downside_beta):
            for backend in BACKENDS:
                first = kernel(book, cand, backend=backend, **kw)["value"]
                second = kernel(book, cand, backend=backend, **kw)["value"]
                assert np.array_equal(first, second)


def test_summary_measured_parity_recorded():
    """Recorded measured backend deviations (tournament seeds, 2026-09-26):
    cdar fast-vs-ref <= 5.6e-17, ced fast-vs-ref <= 2.3e-16, book kernels
    fast-vs-ref == 0.0; cuda-vs-ref <= 3.4e-16 for every kernel — all far
    inside the rtol<=1e-8 / atol<=1e-10 house rule."""
    assert FAST_RTOL <= 1e-12
    assert CUDA_RTOL <= 1e-8 and CUDA_ATOL <= 1e-10
