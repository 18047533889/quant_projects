"""M01 paired net-performance kernels (QE-EXT-SPEC-1.0, extension_plan.md Sec. 7.1).

New metric family "paired performance".  Formulas are authoritative from
extension_plan.md; the sections are cited per function:

- Sec. 6.1  unified sample rule: complete common continuous grid, FP64
        statistics, ``ddof=1``, no epsilon-inflated zero variance,
        internal gaps / Inf are invalid evidence (never compressed).
- Sec. 6.3  frozen moving-block ``(B, J)`` starts plan (NumPy PCG64).
- Sec. 6.4  exact block prefix moments; the Sharpe-delta bootstrap never
        materialises a ``[B, T]`` (let alone ``[B, T, F]``) panel.
- Sec. 6.6  upper-tail mean with boundary-tie weight distribution.
- Sec. 6.7  HAC Bartlett long-run variance, reused from
        ``quant_evaluator.metrics.robustness.compute_hac_variance``.
- Sec. 7.1  M01 formulas; the three interval methods (HAC influence,
        percentile bootstrap, studentized bootstrap) are separate
        kernels with separate semantics and never merged.

New metric IDs start at version ``1.0.0`` (Sec. 3.2).  Registry binding
and facade-builder wiring are deliberately NOT done here (the main agent
wires these); these modules only provide the kernels plus typed results.

Result contract (lightweight pre-registry form): a frozen
:class:`PairedMetricResult` with ``values`` shaped ``(K, F)`` and ordered
``component_names`` (Sec. 5.2 component naming, e.g. ``estimate, ci_low,
ci_high, standard_error``).  ``status``/``reason_code`` follow the Sec.
5.3 eight-state vocabulary; ``interval_status`` separates the
point-estimate state from the interval state (Sec. 5.3: a point estimate
may be valid while its interval is insufficient -- never masked).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as _sps

from quant_evaluator.metrics.robustness import compute_hac_variance

__all__ = [
    "BlockStartsPlan",
    "PairedMetricResult",
    "default_hac_max_lag",
    "paired_net_sharpe_delta",
    "paired_sharpe_hac_ci",
    "paired_sharpe_bootstrap_ci",
    "paired_sharpe_studentized_ci",
    "paired_cer_delta",
    "paired_mdd_improvement",
    "paired_es_improvement",
]

METRIC_VERSION = "1.0.0"

STATUS_OK = "OK"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_INVALID = "INVALID_EVIDENCE"

_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)


# ---------------------------------------------------------------------------
# validation helpers (Sec. 12 `real_array` / `positive_int` semantics)
# ---------------------------------------------------------------------------

def real_matrix(value, name: str, ndim: int = 2) -> np.ndarray:
    """Accept real numeric arrays (int/float/uint kinds), reject bool,
    complex, object/string and wrong rank; return a fresh float64 copy."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real numerical array with rank {ndim}")
    return raw.astype(np.float64)


def _positive_int(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 1):
        raise ValueError(f"{name} must be a positive integer, not bool")
    return int(value)


def _nonnegative_int(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer, not bool")
    return int(value)


def _nonnegative_real(value, name: str) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative real, not bool")
    return float(value)


def _unit_real(value, name: str) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)


def _risk_free_daily(risk_free_daily, t_len: int) -> np.ndarray:
    """Sec. 5.1: ``risk_free_daily`` is an explicit (T,) daily series (a
    scalar zero is accepted as the explicit no-risk-free-rate convention).
    An annualised rate must never be silently divided here."""
    if risk_free_daily is None:
        return np.zeros(t_len, dtype=np.float64)
    raw = np.asarray(risk_free_daily)
    if raw.dtype.kind not in "fiu":
        raise ValueError("risk_free_daily must be a real numerical array")
    if raw.ndim == 0:
        out = np.full(t_len, float(raw), dtype=np.float64)
    elif raw.ndim == 1 and raw.shape[0] == t_len:
        out = raw.astype(np.float64)
    else:
        raise ValueError("risk_free_daily must be a (T,) series or a scalar")
    if not np.isfinite(out).all():
        raise ValueError("risk_free_daily must be finite")
    return out


def _result(metric_id, components, values, status, reason,
            interval_status=None, **diagnostics) -> "PairedMetricResult":
    return PairedMetricResult(
        metric_id=metric_id, component_names=tuple(components), values=values,
        status=status, reason_code=reason, interval_status=interval_status,
        diagnostics=dict(diagnostics),
    )


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedMetricResult:
    """Pre-registry typed result for one M01 metric over F candidate columns.

    ``values`` is ``(K, F)`` float64 with ``component_names`` giving the
    ordered K axis (Sec. 5.2).  ``status`` is the point-estimate evidence
    state; ``interval_status`` (when the metric produces an interval) is
    the separate interval evidence state (Sec. 5.3).
    """
    metric_id: str
    component_names: tuple
    values: np.ndarray
    status: str
    reason_code: str
    interval_status: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# paired input plumbing (Sec. 6.1 unified sample rule)
# ---------------------------------------------------------------------------

def _pair_inputs(candidate, baseline, min_periods, metric_id):
    """Validate shapes/dtypes and per-column completeness.

    Returns ``(cand, base, t_len, n_fac, col_ok)``.  ``base`` has either 1
    shared column (broadcast to all candidate factors) or one column per
    candidate factor.  ``col_ok[f]`` is False when the paired column has
    any non-finite observation -- that pair is invalid evidence (Sec. 1.3
    item 2, Sec. 6.1 rule 3: gaps are never compressed).
    """
    cand = real_matrix(candidate, "candidate", ndim=2)
    base = real_matrix(baseline, "baseline", ndim=2)
    if cand.shape[0] != base.shape[0]:
        raise ValueError(
            "candidate and baseline must be equal-length series on the same "
            "clock; positional time compression is forbidden (Sec. 1.3 item 2)")
    t_len, n_fac = cand.shape
    if not (base.shape[1] == 1 or base.shape[1] == n_fac):
        raise ValueError(
            "baseline must have one shared column or one column per candidate factor")
    _positive_int(min_periods, "min_periods")

    col_ok = np.isfinite(cand).all(axis=0)
    base_ok = np.isfinite(base).all(axis=0)
    if base.shape[1] == 1:
        col_ok = col_ok & base_ok[0]
    else:
        col_ok = col_ok & base_ok
    return cand, base, t_len, n_fac, col_ok


def _aggregate_status(col_ok, t_len, min_periods):
    """Map per-column validity to the aggregate (status, reason) pair.
    Partial-column invalidity is INVALID_EVIDENCE at the aggregate level
    while the untouched columns still carry computed values."""
    if t_len < min_periods:
        return STATUS_INSUFFICIENT, "OBSERVATIONS_TOO_FEW"
    if not col_ok.all():
        return STATUS_INVALID, "NONFINITE_OR_GAPPED_SAMPLE"
    return STATUS_OK, "OK"


def _can_compute(col_ok, t_len, min_periods) -> bool:
    """Point/interval computation proceeds whenever the sample is long
    enough and at least one paired column is complete; the aggregate
    status independently reports the partial-invalid evidence."""
    return t_len >= min_periods and bool(col_ok.any())


def _rf_broadcast(risk_free_daily, t_len):
    return _risk_free_daily(risk_free_daily, t_len)[:, None]


# ---------------------------------------------------------------------------
# sample Sharpe (Sec. 7.1 point estimate) with the Sec. 6.1 rule-6 zero-variance guard
# ---------------------------------------------------------------------------

def _sample_sharpe(excess: np.ndarray, periods_per_year: float):
    """Return ``(mu, var, sharpe, positive_variance)``; zero-variance
    columns get ``sharpe=NaN`` (never an epsilon-inflated huge Sharpe)."""
    mu = excess.mean(axis=0)
    var = excess.var(axis=0, ddof=1)
    scale = np.maximum(np.abs(mu), np.abs(excess).max(axis=0))
    zero_tol = (64.0 * _EPS * np.maximum(scale, _TINY)) ** 2
    ok = var > zero_tol
    sharpe = np.full(mu.shape, np.nan, dtype=np.float64)
    np.divide(math.sqrt(periods_per_year) * mu, np.sqrt(np.where(ok, var, 1.0)),
              out=sharpe, where=ok)
    return mu, var, sharpe, ok


# ---------------------------------------------------------------------------
# HAC influence-function machinery (Sec. 7.1 method B, Sec. 6.7 kernel reuse)
# ---------------------------------------------------------------------------

def default_hac_max_lag(t_len: int) -> int:
    """``nw_rule_v1`` automatic bandwidth (Sec. 6.1): ``floor(4*(T/100)^(2/9))``."""
    t_len = _positive_int(t_len, "t_len")
    return int(math.floor(4.0 * (t_len / 100.0) ** (2.0 / 9.0)))


def _sharpe_influence(excess: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Empirical influence values of the sample Sharpe (Sec. 7.1 B):

    ``psi_t = sqrt(A) * a_T * [(r_t - mu)/sqrt(v)
              - mu * ((r_t - mu)^2 - v) / (2 v^{3/2})]``

    with population variance ``v = T^-1 sum (r - mu)^2`` and
    ``a_T = sqrt((T-1)/T)``.  Zero-variance columns yield NaN.
    """
    t_len = excess.shape[0]
    mu = excess.mean(axis=0)
    v = ((excess - mu) ** 2).mean(axis=0)  # population variance, T^-1
    ok = v > 0.0
    dev = excess - mu
    safe_v = np.where(ok, v, 1.0)
    psi = (math.sqrt(periods_per_year) * math.sqrt((t_len - 1) / t_len)
           * (dev / np.sqrt(safe_v)
              - mu * (dev ** 2 - v) / (2.0 * safe_v ** 1.5)))
    psi[:, ~ok] = np.nan
    return psi


def _influence_hac_se(psi_c: np.ndarray, psi_b: np.ndarray, max_lag: int):
    """``u_t = psi1_t - psi0_t``; HAC SE of the mean via the existing
    Bartlett kernel, which already returns ``Omega / T`` (variance of the
    sample mean, Sec. 6.7).  Returns ``(u, se)`` with NaN where insufficient."""
    u = psi_c - psi_b
    om_over_t = np.asarray(
        compute_hac_variance(u, max_lag=max_lag, kernel="bartlett"), dtype=np.float64)
    se = np.sqrt(np.where(om_over_t >= 0.0, om_over_t, np.nan))
    return u, se


# ---------------------------------------------------------------------------
# frozen resampling plan (Sec. 6.3) + exact block prefix moments (Sec. 6.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockStartsPlan:
    """Frozen ``(B, J)`` moving-block starts plan with Sec. 6.3 semantics.

    Draws: ``starts[b, j] ~ UniformInteger(0, sample_length - block_length)``
    (inclusive) from a single NumPy PCG64 stream; the trailing short block
    uses the *same* start range (identical to "draw full blocks, then
    truncate").  ``J = ceil(output_length / block_length)`` and the final
    length is ``output_length - (J-1)*block_length``.

    The plan carries no factor-axis information: it is byte-identical for
    any factor ordering / count / tile width (Sec. 1.1 reproducibility,
    Sec. 6.3).

    TODO(integration): once the parallel contract agent lands
    ``contracts/extension_policy.ResamplingPlan`` (Sec. 6.3), refactor the
    bootstrap kernels to consume that frozen plan; the start-draw
    semantics above (PCG64, uniform inclusive range, truncated tail
    block) must be preserved byte-for-byte.
    """
    sample_length: int
    output_length: int
    block_length: int
    repetitions: int
    seed: int = 0

    def __post_init__(self):
        _positive_int(self.sample_length, "sample_length")
        _positive_int(self.output_length, "output_length")
        _positive_int(self.block_length, "block_length")
        _positive_int(self.repetitions, "repetitions")
        if self.repetitions < 2:
            raise ValueError("repetitions must be at least two (B=1 rejected, Sec. 6.3)")
        if self.block_length > self.sample_length:
            raise ValueError("block_length exceeds sample_length")
        if (isinstance(self.seed, (bool, np.bool_))
                or not isinstance(self.seed, (int, np.integer)) or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")

    @property
    def block_count(self) -> int:
        return (self.output_length + self.block_length - 1) // self.block_length

    @property
    def lengths(self) -> np.ndarray:
        lengths = np.full(self.block_count, self.block_length, dtype=np.int64)
        lengths[-1] = self.output_length - (self.block_count - 1) * self.block_length
        return lengths

    def starts(self) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        return rng.integers(0, self.sample_length - self.block_length + 1,
                            size=(self.repetitions, self.block_count), dtype=np.int64)


def _prefix_sample_moments(values: np.ndarray, starts: np.ndarray,
                           lengths: np.ndarray):
    """Sec. 6.4 exact block prefix moments (mirror of the Sec. 12 oracle).

    Returns ``(mu, var)`` shaped ``(B, F)``: the mean and ``ddof=1``
    sample variance of every resampled path, computed from two prefix
    sums -- never from a materialised ``[B, T]`` panel.  Equivalent, up to
    documented floating-point tolerance, to per-day expansion of the same
    draw sequence; *not* a compression approximation.
    """
    values = np.asarray(values, dtype=np.float64)
    n, f = values.shape
    starts = np.asarray(starts)
    lengths = np.asarray(lengths)
    if starts.dtype.kind not in "iu" or lengths.dtype.kind not in "iu" or starts.ndim != 2:
        raise ValueError("Integer start matrix and lengths required")
    if lengths.shape != (starts.shape[1],) or np.any(lengths <= 0):
        raise ValueError("Invalid lengths")
    if np.any(starts < 0) or np.any(starts + lengths[None, :] > n):
        raise ValueError("Sample outside source interval")
    h = int(lengths.sum())
    if h < 2 or n < 2:
        raise ValueError("At least two observations required")
    reps = starts.shape[0]

    center = values.mean(axis=0)
    z = values - center
    p1 = np.vstack([np.zeros((1, f)), np.cumsum(z, axis=0)])
    p2 = np.vstack([np.zeros((1, f)), np.cumsum(z * z, axis=0)])
    sum1 = np.zeros((reps, f), dtype=np.float64)
    sum2 = np.zeros((reps, f), dtype=np.float64)
    for j in range(lengths.shape[0]):
        lo = starts[:, j]
        hi = lo + lengths[j]
        sum1 += p1[hi] - p1[lo]
        sum2 += p2[hi] - p2[lo]
    centered_ss = sum2 - sum1 * sum1 / h
    tolerance = 64.0 * _EPS * np.maximum(np.abs(sum2), _TINY)
    if np.any(centered_ss < -tolerance):
        raise ArithmeticError(
            "Unstable variance requires a stable production fallback "
            "(Sec. 6.4 Chan/Welford merge path)")
    var = np.maximum(centered_ss, 0.0) / (h - 1)
    return center + sum1 / h, var


def _expand_indices(start_row: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate one replicate's blocks into a single (n*,)-index vector."""
    total = int(np.asarray(lengths).sum())
    idx = np.empty(total, dtype=np.int64)
    pos = 0
    for s, length in zip(start_row, lengths):
        idx[pos:pos + length] = np.arange(s, s + length)
        pos += length
    return idx


def _sharpe_from_moments(mu: np.ndarray, var: np.ndarray,
                         periods_per_year: float) -> np.ndarray:
    ok = var > 0.0
    out = np.full(np.broadcast(mu, var).shape, np.nan, dtype=np.float64)
    np.divide(math.sqrt(periods_per_year) * mu, np.sqrt(np.where(ok, var, 1.0)),
              out=out, where=ok)
    return out


# ---------------------------------------------------------------------------
# tail statistics (Sec. 6.6)
# ---------------------------------------------------------------------------

def _upper_tail_mean_columns(losses: np.ndarray, tail_confidence: float,
                             min_tail_mass: float):
    """``UTM_q`` per column (Sec. 6.6): the worst ``1 - q`` mass of the
    loss distribution, with boundary-tie weights distributed so that
    permuting tied observations cannot change the result.  Returns
    ``(utm (F,), mass, ok)`` where ``ok`` is False when the tail mass is
    below ``min_tail_mass`` (10 observed-equivalents by default)."""
    n = losses.shape[0]
    mass = (1.0 - tail_confidence) * n
    nearest = round(mass)
    if nearest >= 1 and abs(mass - nearest) <= 8.0 * _EPS * max(1.0, mass):
        mass = float(nearest)
    if mass < min_tail_mass:
        return np.full(losses.shape[1], np.nan), mass, False
    desc = np.sort(losses, axis=0)[::-1]
    index = min(n - 1, math.ceil(mass) - 1)
    cutoff = desc[index]
    above = losses > cutoff
    tied = losses == cutoff
    above_count = above.sum(axis=0)
    tied_count = tied.sum(axis=0)
    weight = above.astype(np.float64)
    share = np.where(tied_count > 0,
                     (mass - above_count) / np.maximum(tied_count, 1), 0.0)
    weight = weight + tied * share[None, :]
    utm = (weight * losses).sum(axis=0) / mass
    return utm, mass, True


# ---------------------------------------------------------------------------
# M01 kernels
# ---------------------------------------------------------------------------

def paired_net_sharpe_delta(candidate, baseline, *, risk_free_daily=None,
                            periods_per_year: float = 252.0,
                            min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 point estimate ``Delta SR = SR_1 - SR_0`` on net excess returns.

    ``SR = sqrt(A) * mean(r - rf) / std(r - rf)`` with ``ddof=1`` (Sec. 6.1
    rule 7).  ``sqrt(A)`` is the standard daily-Sharpe scale only (Sec. 7.1:
    no claim of annualised Sharpe under autocorrelation).  Zero-variance
    columns are NaN + INSUFFICIENT_DATA; a candidate exactly equal to the
    baseline with positive variance yields ``Delta = 0``.
    """
    metric_id = "paired_net_sharpe_delta"
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    rf = _rf_broadcast(risk_free_daily, t_len)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((1, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len}
    if _can_compute(col_ok, t_len, min_periods):
        _, _, sr_c, _ = _sample_sharpe(cand - rf, periods_per_year)
        _, _, sr_b, _ = _sample_sharpe(base - rf, periods_per_year)
        values[0] = sr_c - np.broadcast_to(sr_b, sr_c.shape)
        if status == STATUS_OK and not np.isfinite(values[0]).any():
            reason = "ZERO_VARIANCE"
            status = STATUS_INSUFFICIENT
        if np.array_equal(cand, base):
            diagnostics["identical_inputs"] = True
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate",), values, status, reason, **diagnostics)


def paired_sharpe_hac_ci(candidate, baseline, *, risk_free_daily=None,
                         max_lag: Optional[int] = None,
                         periods_per_year: float = 252.0,
                         confidence_level: float = 0.95,
                         min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 method B: influence-function HAC interval for ``Delta SR``.

    Low-cost REFINE main path.  ``u_t = psi1_t - psi0_t`` (psi per Sec. 7.1)
    feeds the existing Bartlett HAC kernel (Sec. 6.7); ``SE = sqrt(Omega/T)``;
    ``CI = DeltaSR +- z_{1-alpha/2} * SE``.  Identical/proportionate inputs
    whose influence values cancel exactly give the degenerate interval
    ``[DeltaSR, DeltaSR]`` with ``identical_statistic=True``.  A negative
    or non-finite SE leaves the point estimate valid but the interval
    insufficient (Sec. 5.3 separation).
    """
    metric_id = "paired_sharpe_hac_ci"
    _unit_real(confidence_level, "confidence_level")
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    rf = _rf_broadcast(risk_free_daily, t_len)
    lag = default_hac_max_lag(t_len) if max_lag is None else _nonnegative_int(max_lag, "max_lag")
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((4, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len,
                   "method": "paired_influence_hac_v1", "kernel": "bartlett",
                   "max_lag": lag,
                   "bandwidth_rule": "nw_rule_v1" if max_lag is None else "predeclared"}
    interval_status = STATUS_INSUFFICIENT
    if _can_compute(col_ok, t_len, min_periods):
        exc_c = cand - rf
        exc_b = base - rf
        _, _, sr_c, ok_c = _sample_sharpe(exc_c, periods_per_year)
        _, _, sr_b, _ = _sample_sharpe(exc_b, periods_per_year)
        delta = sr_c - np.broadcast_to(sr_b, sr_c.shape)
        values[0] = delta
        psi_c = _sharpe_influence(exc_c, periods_per_year)
        psi_b = _sharpe_influence(exc_b, periods_per_year)
        u, se = _influence_hac_se(psi_c, psi_b, lag)
        alpha = 1.0 - confidence_level
        z = float(_sps.norm.ppf(1.0 - alpha / 2.0))
        interval_ok = col_ok & np.isfinite(se)
        identical = np.zeros(n_fac, dtype=bool)
        for f in range(n_fac):
            if not col_ok[f] or not ok_c[f]:
                continue
            u_col = u[:, f]
            if np.isfinite(u_col).all() and np.all(u_col == 0.0):
                identical[f] = True
                values[1, f] = delta[f]
                values[2, f] = delta[f]
                values[3, f] = 0.0
                interval_ok[f] = True
            elif interval_ok[f]:
                values[1, f] = delta[f] - z * se[f]
                values[2, f] = delta[f] + z * se[f]
                values[3, f] = se[f]
        interval_status = STATUS_OK if interval_ok.any() else STATUS_INSUFFICIENT
        if not interval_ok.any():
            diagnostics["interval_reason"] = "HAC_STANDARD_ERROR_UNAVAILABLE"
        diagnostics["identical_statistic"] = tuple(bool(v) for v in identical)
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate", "ci_low", "ci_high", "standard_error"),
                   values, status, reason, interval_status=interval_status,
                   **diagnostics)


def _bootstrap_plan_and_shares(plan, cand, base, rf, periods_per_year):
    """Shared moving-block draws (Sec. 6.3) + per-replicate Sharpe moments
    (Sec. 6.4).  The baseline is resampled with the *same* start matrix
    (shared draws).  Returns ``delta_star (B, F)`` with NaN for degenerate
    replicates."""
    starts = plan.starts()
    lengths = plan.lengths
    exc_c = cand - rf
    exc_b = base - rf
    mu_c, var_c = _prefix_sample_moments(exc_c, starts, lengths)
    mu_b, var_b = _prefix_sample_moments(exc_b, starts, lengths)
    sr_c = _sharpe_from_moments(mu_c, var_c, periods_per_year)
    sr_b = _sharpe_from_moments(mu_b, var_b, periods_per_year)
    if sr_b.shape[1] == 1 and sr_c.shape[1] > 1:
        return sr_c - sr_b[:, :1]
    return sr_c - sr_b


def paired_sharpe_bootstrap_ci(candidate, baseline, *, risk_free_daily=None,
                               block_length: int = 10, repetitions: int = 999,
                               confidence_level: float = 0.95, seed: int = 0,
                               periods_per_year: float = 252.0,
                               min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 method A: shared moving-block percentile interval.

    ``method=paired_moving_block_percentile_v1``.  Both legs are resampled
    with one frozen (B, J) start plan (Sec. 6.3) and each replicate's two
    Sharpes are computed from Sec. 6.4 prefix moments (``n* = T``); the
    interval is the type-7 linear percentile of ``Delta SR*`` over valid
    replicates.  Fewer than ``ceil(0.99 * B)`` valid replicates leave the
    point estimate valid but the interval insufficient, with the
    degenerate count reported (never hidden).  No pseudo-p-values and no
    "posterior profit probability" interpretation.
    """
    metric_id = "paired_sharpe_bootstrap_ci"
    _unit_real(confidence_level, "confidence_level")
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    rf = _rf_broadcast(risk_free_daily, t_len)
    # Plan validation up front: B >= 2, integer block <= T, no bools (Sec. 6.3).
    plan = BlockStartsPlan(sample_length=t_len, output_length=t_len,
                           block_length=block_length, repetitions=repetitions,
                           seed=seed)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((4, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len,
                   "method": "paired_moving_block_percentile_v1",
                   "block_length": int(plan.block_length),
                   "repetitions": int(plan.repetitions), "seed": int(plan.seed)}
    interval_status = STATUS_INSUFFICIENT
    if _can_compute(col_ok, t_len, min_periods):
        _, _, sr_c, ok_c = _sample_sharpe(cand - rf, periods_per_year)
        _, _, sr_b, _ = _sample_sharpe(base - rf, periods_per_year)
        delta = sr_c - np.broadcast_to(sr_b, sr_c.shape)
        values[0] = delta
        delta_star = _bootstrap_plan_and_shares(plan, cand, base, rf, periods_per_year)
        min_valid = math.ceil(0.99 * plan.repetitions)
        alpha = 1.0 - confidence_level
        degenerate = np.zeros(n_fac, dtype=np.int64)
        any_ok = False
        for f in range(n_fac):
            if not col_ok[f] or not ok_c[f]:
                continue
            col = delta_star[:, f]
            finite = col[np.isfinite(col)]
            degenerate[f] = plan.repetitions - finite.size
            if finite.size < min_valid:
                continue
            lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
            values[1, f] = lo
            values[2, f] = hi
            values[3, f] = float(finite.size)
            any_ok = True
        interval_status = STATUS_OK if any_ok else STATUS_INSUFFICIENT
        if not any_ok:
            diagnostics["interval_reason"] = "DEGENERATE_REPLICATES_EXCEED_THRESHOLD"
        diagnostics["degenerate_replicates"] = tuple(int(v) for v in degenerate)
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate", "ci_low", "ci_high", "valid_replicates"),
                   values, status, reason, interval_status=interval_status,
                   **diagnostics)


def paired_sharpe_studentized_ci(candidate, baseline, *, risk_free_daily=None,
                                 block_length: int = 20, repetitions: int = 4999,
                                 confidence_level: float = 0.95, seed: int = 0,
                                 max_lag: Optional[int] = None,
                                 periods_per_year: float = 252.0,
                                 min_periods: int = 252) -> PairedMetricResult:
    """Sec. 7.1 method C: CONFIRM studentized bootstrap interval.

    "Paired moving-block + influence-function HAC studentization"
    (Ledoit-Wolf-inspired; not a verbatim reproduction of the paper's
    bandwidth/resampling calibration).  ``SE0`` is the observed-sample
    influence HAC SE (method B); every shared replicate recomputes
    ``Delta SR*`` (Sec. 6.4 prefix moments) and its own sample influence
    HAC ``SE*`` on the *resampled* series with a fixed lag -- the
    concatenation-boundary neighbours are kept (Sec. 7.1 C), never
    dropped.  ``t_b* = (DeltaSR_b* - DeltaSR) / SE_b*`` and
    ``CI = [DeltaSR - Q_{1-a/2}(t*) SE0, DeltaSR - Q_{a/2}(t*) SE0]``.

    Memory stays ``O(T*F)`` per replicate (no ``[B, T]`` panel, Sec. 1.1);
    complexity is ``O(B*T*L*F)`` -- CONFIRM sets only.  The lag is fixed
    from the observed sample; no per-replicate bandwidth re-selection.
    """
    metric_id = "paired_sharpe_studentized_ci"
    _unit_real(confidence_level, "confidence_level")
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    rf = _rf_broadcast(risk_free_daily, t_len)
    lag = default_hac_max_lag(t_len) if max_lag is None else _nonnegative_int(max_lag, "max_lag")
    plan = BlockStartsPlan(sample_length=t_len, output_length=t_len,
                           block_length=block_length, repetitions=repetitions,
                           seed=seed)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((4, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len,
                   "method": "paired_moving_block_studentized_hac_v1",
                   "kernel": "bartlett", "max_lag": lag,
                   "bandwidth_rule": "nw_rule_v1" if max_lag is None else "predeclared",
                   "block_length": int(plan.block_length),
                   "repetitions": int(plan.repetitions), "seed": int(plan.seed)}
    interval_status = STATUS_INSUFFICIENT
    if _can_compute(col_ok, t_len, min_periods):
        exc_c = cand - rf
        exc_b = base - rf
        _, _, sr_c, ok_c = _sample_sharpe(exc_c, periods_per_year)
        _, _, sr_b, _ = _sample_sharpe(exc_b, periods_per_year)
        delta = sr_c - np.broadcast_to(sr_b, sr_c.shape)
        values[0] = delta
        psi_c = _sharpe_influence(exc_c, periods_per_year)
        psi_b = _sharpe_influence(exc_b, periods_per_year)
        u_obs, se0 = _influence_hac_se(psi_c, psi_b, lag)
        se0_usable = col_ok & ok_c & np.isfinite(se0) & (se0 > 0.0)
        identical = np.isfinite(u_obs).all(axis=0) & np.all(u_obs == 0.0, axis=0)
        diagnostics["identical_statistic"] = tuple(bool(v) for v in identical)
        if not se0_usable.any():
            diagnostics["interval_reason"] = "OBSERVED_STANDARD_ERROR_UNAVAILABLE"
        else:
            starts = plan.starts()
            lengths = plan.lengths
            mu_c, var_c = _prefix_sample_moments(exc_c, starts, lengths)
            mu_b, var_b = _prefix_sample_moments(exc_b, starts, lengths)
            sr_c_star = _sharpe_from_moments(mu_c, var_c, periods_per_year)
            sr_b_star = _sharpe_from_moments(mu_b, var_b, periods_per_year)
            delta_star = (sr_c_star - sr_b_star[:, :1]
                          if sr_b_star.shape[1] == 1 and sr_c_star.shape[1] > 1
                          else sr_c_star - sr_b_star)
            se_star = np.full((plan.repetitions, n_fac), np.nan, dtype=np.float64)
            for b in range(plan.repetitions):
                idx = _expand_indices(starts[b], lengths)
                u_star = (_sharpe_influence(exc_c[idx], periods_per_year)
                          - _sharpe_influence(exc_b[idx], periods_per_year))
                hac = np.asarray(compute_hac_variance(
                    u_star, max_lag=lag, kernel="bartlett"), dtype=np.float64)
                se_star[b] = np.sqrt(np.where(hac >= 0.0, hac, np.nan))
            with np.errstate(invalid="ignore", divide="ignore"):
                t_star = (delta_star - delta[None, :]) / se_star
            min_valid = math.ceil(0.99 * plan.repetitions)
            alpha = 1.0 - confidence_level
            degenerate = np.zeros(n_fac, dtype=np.int64)
            any_ok = False
            for f in range(n_fac):
                if not (col_ok[f] and ok_c[f] and se0_usable[f]):
                    continue
                col = t_star[:, f]
                finite = col[np.isfinite(col)]
                degenerate[f] = plan.repetitions - finite.size
                if finite.size < min_valid:
                    continue
                q_lo, q_up = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
                values[1, f] = delta[f] - q_up * se0[f]
                values[2, f] = delta[f] - q_lo * se0[f]
                values[3, f] = float(se0[f])
                any_ok = True
            interval_status = STATUS_OK if any_ok else STATUS_INSUFFICIENT
            if not any_ok:
                diagnostics["interval_reason"] = "DEGENERATE_REPLICATES_EXCEED_THRESHOLD"
            diagnostics["degenerate_replicates"] = tuple(int(v) for v in degenerate)
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate", "ci_low", "ci_high", "standard_error"),
                   values, status, reason, interval_status=interval_status,
                   **diagnostics)


def paired_cer_delta(candidate, baseline, *, risk_free_daily=None,
                     risk_aversion_lambda, periods_per_year: float = 252.0,
                     min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 ``CER = A * (mean(r - rf) - lambda * var(r - rf) / 2)`` and
    ``paired_cer_delta = CER_1 - CER_0``.

    This is the declared mean-variance approximation, not actual investor
    utility (Sec. 7.1).  ``risk_aversion_lambda`` is required, explicit
    and non-negative.  Zero variance is admissible here (the quadratic
    term vanishes); the excess mean is still well defined.
    """
    metric_id = "paired_cer_delta"
    lam = _nonnegative_real(risk_aversion_lambda, "risk_aversion_lambda")
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    rf = _rf_broadcast(risk_free_daily, t_len)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((1, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len,
                   "risk_aversion_lambda": lam}
    if _can_compute(col_ok, t_len, min_periods):
        mu_c, var_c, _, _ = _sample_sharpe(cand - rf, periods_per_year)
        mu_b, var_b, _, _ = _sample_sharpe(base - rf, periods_per_year)
        # CER itself is finite at zero variance; only the mean/var are
        # reused here, not the Sharpe NaN guard.
        cer_c = periods_per_year * (mu_c - lam * var_c / 2.0)
        cer_b = periods_per_year * (mu_b - lam * var_b / 2.0)
        values[0] = cer_c - np.broadcast_to(cer_b, cer_c.shape)
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate",), values, status, reason, **diagnostics)


def _mdd_columns(total_returns: np.ndarray) -> np.ndarray:
    """Percentage max drawdown per column on the compounded log path
    (Sec. 6.5 ``summary_mdd``); requires ``r > -1`` (Sec. 6.1 rule 5).
    The log path includes the initial-wealth 0 point, so a path that dips
    below the starting equity counts as a drawdown (Sec. 12 oracle)."""
    log_path = np.cumsum(np.log1p(total_returns), axis=0)
    running_max = np.maximum(np.maximum.accumulate(log_path, axis=0), 0.0)
    log_dd = np.maximum((running_max - log_path).max(axis=0), 0.0)
    return -np.expm1(-log_dd)


def paired_mdd_improvement(candidate, baseline, *,
                           min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 ``paired_mdd_improvement = MDD_0 - MDD_1`` (positive
    improves).  MDD uses *total* net returns (not excess, Sec. 7.1) on
    the compounded log path.  ``r == -1`` is INVALID_EVIDENCE with
    ``terminal_total_loss=True`` and ``known_drawdown_lower_bound=1.0``;
    ``r < -1`` is a RETURN_BASIS_MISMATCH (Sec. 6.1 rule 5).  Internal
    gaps/Inf are invalid evidence, never compressed (Sec. 1.3 item 2).
    """
    metric_id = "paired_mdd_improvement"
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((1, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len}
    if _can_compute(col_ok, t_len, min_periods):
        if (cand < -1).any() or (base < -1).any():
            status, reason = STATUS_INVALID, "RETURN_BASIS_MISMATCH"
            diagnostics["return_basis"] = "below_minus_one"
        elif (cand == -1).any() or (base == -1).any():
            status, reason = STATUS_INVALID, "TERMINAL_TOTAL_LOSS"
            diagnostics["terminal_total_loss"] = True
            diagnostics["known_drawdown_lower_bound"] = 1.0
        else:
            mdd_c = _mdd_columns(cand)
            mdd_b = _mdd_columns(base)
            values[0] = mdd_b - mdd_c  # (1,)-(F,) or (F,)-(F,) broadcasting
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate",), values, status, reason, **diagnostics)


def paired_es_improvement(candidate, baseline, *, tail_confidence: float = 0.95,
                          min_tail_mass: float = 10.0,
                          min_periods: int = 60) -> PairedMetricResult:
    """Sec. 7.1 ``paired_es_improvement = UTM_q(-r_0) - UTM_q(-r_1)``
    (positive improves: the candidate's worst-mass loss is smaller).

    ``UTM_q`` is the Sec. 6.6 upper-tail mean of the loss distribution
    with boundary-tie weight distribution (permuting tied days cannot
    change the result).  The tail mass must reach ``min_tail_mass``
    (10 observed-equivalents by default, Sec. 6.1 default table) or the
    result is INSUFFICIENT_DATA.
    """
    metric_id = "paired_es_improvement"
    q = _unit_real(tail_confidence, "tail_confidence")
    mass_min = _nonnegative_real(min_tail_mass, "min_tail_mass")
    cand, base, t_len, n_fac, col_ok = _pair_inputs(candidate, baseline, min_periods, metric_id)
    status, reason = _aggregate_status(col_ok, t_len, min_periods)

    values = np.full((1, n_fac), np.nan, dtype=np.float64)
    diagnostics = {"metric_version": METRIC_VERSION, "n_time": t_len,
                   "tail_confidence": q, "min_tail_mass": mass_min}
    if _can_compute(col_ok, t_len, min_periods):
        utm_c, mass_c, ok_c = _upper_tail_mean_columns(-cand, q, mass_min)
        utm_b, _, ok_b = _upper_tail_mean_columns(-base, q, mass_min)
        if not (ok_c and ok_b):
            status, reason = STATUS_INSUFFICIENT, "TAIL_MASS_BELOW_MINIMUM"
        else:
            values[0] = utm_b - utm_c  # (1,)-(F,) or (F,)-(F,) broadcasting
        diagnostics["tail_mass"] = float(mass_c)
    if not col_ok.all():
        diagnostics["invalid_columns"] = tuple(int(f) for f in np.flatnonzero(~col_ok))
    return _result(metric_id, ("estimate",), values, status, reason, **diagnostics)
