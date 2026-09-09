"""Robust statistical evidence producers using existing QE kernels."""
from __future__ import annotations
import numpy as np

from quant_evaluator.contracts.statistical_evidence import (
    DSREvidence, HACEvidence, HorizonCurveEvidence, PBOEvidence,
    PairedBootstrapEvidence, RegimeEvidence, RetentionEvidence,
)
from quant_evaluator.metrics.robustness import compute_hac_tstat


def _integer_parameter(name, value, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def build_hac_evidence(series, *, max_lag: int = 5, min_periods: int = 30,
                       kernel: str = "bartlett", bandwidth_rule: str = "predeclared"):
    _integer_parameter("max_lag", max_lag, 0)
    _integer_parameter("min_periods", min_periods, 2)
    if kernel not in {"bartlett", "uniform"}:
        raise ValueError("unsupported HAC kernel")
    if not isinstance(bandwidth_rule, str) or not bandwidth_rule.strip():
        raise ValueError("HAC bandwidth selection rule is required")
    values = np.asarray(series, dtype=float)
    if values.ndim != 1: raise ValueError("HAC evidence requires one time series")
    valid = values[np.isfinite(values)]
    n_time = int(valid.size)
    positions = np.flatnonzero(np.isfinite(values))
    has_internal_gap = bool(n_time and positions[-1] - positions[0] + 1 != n_time)
    # The legacy kernel compresses nonfinite dates. Without an explicit
    # irregular-time estimator this changes the meaning of the lag, so this
    # typed producer refuses to certify a gapped series instead of compressing it.
    if has_internal_gap or n_time < min_periods or max_lag >= n_time:
        return HACEvidence(float(np.mean(valid)) if n_time else np.nan, None, None,
                           n_time, kernel, max_lag, bandwidth_rule, "INSUFFICIENT")
    tstat, se = compute_hac_tstat(valid[:, None], max_lag=max_lag, kernel=kernel)
    if not np.isfinite(tstat[0]) or not np.isfinite(se[0]) or se[0] <= 0:
        return HACEvidence(float(np.mean(valid)), None, None,
                           n_time, kernel, max_lag, bandwidth_rule, "INSUFFICIENT")
    return HACEvidence(float(np.mean(valid)), float(se[0]), float(tstat[0]),
                       n_time, kernel, max_lag, bandwidth_rule, "VALID")


def build_paired_block_bootstrap_difference(candidate, baseline, *, block_length: int = 10,
                                            repetitions: int = 1000, confidence_level: float = 0.95,
                                            seed: int = 0, min_periods: int = 30):
    for name, value, minimum in (("block_length", block_length, 1),
                                  ("repetitions", repetitions, 2),
                                  ("min_periods", min_periods, 2), ("seed", seed, 0)):
        _integer_parameter(name, value, minimum)
    if (isinstance(confidence_level, (bool, np.bool_))
            or not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    a, b = np.asarray(candidate, float), np.asarray(baseline, float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("paired bootstrap inputs must be equal-length time series")
    joint = np.isfinite(a) & np.isfinite(b); diff = a[joint] - b[joint]; n = len(diff)
    positions = np.flatnonzero(joint)
    has_internal_gap = bool(n and positions[-1] - positions[0] + 1 != n)
    # Dropping missing dates before drawing blocks invents new adjacency and
    # can give differently-masked candidate comparisons different time draws.
    # Irregular-time resampling is not certified by this moving-block producer.
    if has_internal_gap or n < min_periods or block_length > n:
        return PairedBootstrapEvidence(None, (None, None), n, block_length, repetitions,
                                       seed, "percentile_shared_moving_blocks", "INSUFFICIENT")
    rng = np.random.default_rng(seed); starts_max = n - block_length + 1
    means = np.empty(repetitions)
    blocks = int(np.ceil(n / block_length))
    for index in range(repetitions):
        starts = rng.integers(0, starts_max, size=blocks)
        sample = np.concatenate([diff[s:s + block_length] for s in starts])[:n]
        means[index] = np.mean(sample)
    alpha = (1 - confidence_level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return PairedBootstrapEvidence(float(np.mean(diff)), (float(lo), float(hi)), n,
                                   block_length, repetitions, seed,
                                   "percentile_shared_moving_blocks", "VALID")


def build_dsr_evidence(returns, *, family_sharpes, effective_trial_count: int,
                       trial_ledger_ref: str, returns_frequency: str,
                       annualization_factor: float,
                       family_sharpe_scale: str = "raw_periodic",
                       min_periods: int = 60):
    """Deflated-Sharpe probability with auditable sample/family inputs."""
    from scipy.stats import kurtosis, norm, skew
    values = np.asarray(returns, float)
    trials = np.asarray(family_sharpes, float)
    if values.ndim != 1 or trials.ndim != 1:
        raise ValueError("DSR requires one return time series and one trial-family vector")
    if not np.isfinite(trials).all():
        raise ValueError("DSR requires complete finite trial-family evidence")
    _integer_parameter("effective_trial_count", effective_trial_count, 1)
    _integer_parameter("min_periods", min_periods, 2)
    values = values[np.isfinite(values)]
    n = len(values)
    if not trial_ledger_ref or effective_trial_count < 1 or len(trials) != effective_trial_count:
        raise ValueError("DSR requires the complete trial family and durable ledger ref")
    if returns_frequency not in {"daily", "weekly", "monthly"} or annualization_factor <= 0:
        raise ValueError("returns frequency and annualization policy are required")
    if family_sharpe_scale not in {"raw_periodic", "annualized"}:
        raise ValueError("family_sharpe_scale must be raw_periodic or annualized")
    if n < min_periods or np.std(values, ddof=1) <= 0 or effective_trial_count < 2:
        return DSREvidence(None, None, None, n, returns_frequency, annualization_factor,
                           family_sharpe_scale, None, None, effective_trial_count,
                           trial_ledger_ref, "INSUFFICIENT")
    raw_sr = float(np.mean(values) / np.std(values, ddof=1))
    observed = raw_sr * np.sqrt(annualization_factor)
    annualizer = np.sqrt(annualization_factor)
    raw_trials = trials / annualizer if family_sharpe_scale == "annualized" else trials
    trial_sigma = float(np.std(raw_trials, ddof=1))
    gamma = 0.5772156649015329
    benchmark_raw = trial_sigma * ((1-gamma)*norm.ppf(1-1/effective_trial_count)
                                   + gamma*norm.ppf(1-1/(effective_trial_count*np.e)))
    benchmark = benchmark_raw * annualizer
    sk = float(skew(values, bias=False)); ku = float(kurtosis(values, fisher=False, bias=False))
    denominator = np.sqrt(max(1e-15, 1 - sk*raw_sr + ((ku-1)/4)*raw_sr**2))
    probability = float(norm.cdf((raw_sr - benchmark_raw)
                                 * np.sqrt(n-1) / denominator))
    return DSREvidence(observed, benchmark, probability, n, returns_frequency,
                       annualization_factor, family_sharpe_scale, sk, ku, effective_trial_count,
                       trial_ledger_ref, "VALID")


def build_pbo_evidence(candidate_returns, *, n_splits: int = 6,
                       candidate_universe_complete: bool,
                       common_cost_spec_ref: str, purpose: str = "diagnostic"):
    """Combinatorial-style contiguous split diagnostic; never production OOS."""
    from itertools import combinations
    matrix = np.asarray(candidate_returns, float)
    if matrix.ndim != 2: raise ValueError("PBO requires a common T×C return matrix")
    t, c = matrix.shape
    if purpose != "diagnostic": raise ValueError("PBO is a diagnostic split only")
    if not candidate_universe_complete or not common_cost_spec_ref:
        raise ValueError("PBO requires the complete candidate universe and common cost spec")
    if c < 2:
        raise ValueError("PBO requires the full candidate matrix, not winner-only returns")
    if not np.isfinite(matrix).all():
        raise ValueError("PBO candidate matrix must be complete on a common time axis")
    if c < 3 or n_splits < 4 or n_splits % 2 or t < n_splits:
        return PBOEvidence(None, (), t, c, n_splits, purpose,
                           candidate_universe_complete, common_cost_spec_ref, "INSUFFICIENT")
    folds = np.array_split(np.arange(t), n_splits); logits = []
    for train_fold_ids in combinations(range(n_splits), n_splits//2):
        train_ids = np.concatenate([folds[i] for i in train_fold_ids])
        test_ids = np.concatenate([folds[i] for i in range(n_splits) if i not in train_fold_ids])
        train_score = np.nanmean(matrix[train_ids], axis=0)
        winner = int(np.nanargmax(train_score)); test_score = np.nanmean(matrix[test_ids], axis=0)
        rank = int(np.sum(test_score <= test_score[winner]))
        relative = min(max(rank / (c + 1), 1e-12), 1-1e-12)
        logits.append(float(np.log(relative/(1-relative))))
    return PBOEvidence(float(np.mean(np.asarray(logits) <= 0)), tuple(logits), t, c,
                       n_splits, purpose, True, common_cost_spec_ref, "VALID")


def build_retention_evidence(train_series, validation_series, *, train_value: float,
                             validation_value: float, near_zero: float = 1e-3,
                             cost_basis: str = "net", **bootstrap_kwargs):
    if cost_basis not in {"gross", "net"}: raise ValueError("cost_basis must be gross or net")
    paired = build_paired_block_bootstrap_difference(validation_series, train_series, **bootstrap_kwargs)
    reversal = train_value * validation_value < 0
    applicable = abs(train_value) >= near_zero and not reversal
    return RetentionEvidence(float(train_value), float(validation_value),
                             float(validation_value-train_value),
                             float(validation_value/train_value) if applicable else None,
                             applicable, reversal, paired.n_time, paired.confidence_interval,
                             cost_basis, paired.status)


def build_horizon_curve_evidence(horizon_ic_series, *, label_refs, min_periods: int = 20,
                                 zero_tolerance: float = 1e-12):
    if (isinstance(zero_tolerance, (bool, np.bool_)) or
            not isinstance(zero_tolerance, (int, float, np.integer, np.floating)) or
            not np.isfinite(zero_tolerance) or zero_tolerance < 0):
        raise ValueError("zero_tolerance must be a finite nonnegative number")
    zero_tolerance = float(zero_tolerance)
    if (len(horizon_ic_series) < 2 or
            any(not isinstance(h, int) or isinstance(h, bool) or h <= 0
                for h in horizon_ic_series)):
        raise ValueError("at least two unique positive integer horizons are required")
    horizons = tuple(sorted(horizon_ic_series)); refs = tuple(label_refs[h] for h in horizons)
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError("every horizon requires a non-empty label ref")
    means, counts = [], []
    for horizon in horizons:
        values = np.asarray(horizon_ic_series[horizon], float)
        if values.ndim != 1:
            raise ValueError(
                "each horizon IC series must be one-dimensional; evaluate "
                "multi-factor inputs per factor instead of flattening them"
            )
        valid = values[np.isfinite(values)]
        means.append(float(np.mean(valid)) if len(valid) else np.nan); counts.append(len(valid))
    if any(count < min_periods for count in counts):
        return HorizonCurveEvidence(horizons, tuple(means), tuple(counts), None,
                                    "INSUFFICIENT", refs, zero_tolerance)
    finite_means = [v for v in means if np.isfinite(v)]
    near_zero = [abs(v) <= zero_tolerance for v in finite_means]
    signs = {np.sign(v) for v in finite_means if abs(v) > zero_tolerance}
    peaks = sum(1 for i in range(1, len(means)-1)
                if abs(means[i]) > abs(means[i-1]) and abs(means[i]) > abs(means[i+1]))
    monotone = all(abs(means[i]) >= abs(means[i+1]) for i in range(len(means)-1))
    if len(signs) > 1: status, half_life = "SIGN_CHANGE", None
    elif near_zero and any(near_zero): status, half_life = "NOT_FIT", None
    elif peaks > 1: status, half_life = "MULTI_PEAK", None
    elif not monotone or any(not np.isfinite(v) for v in means): status, half_life = "NOT_FIT", None
    else:
        x = np.asarray(horizons, float); y = np.log(np.maximum(np.abs(means), 1e-15))
        slope = float(np.polyfit(x, y, 1)[0])
        status, half_life = (("VALID", float(-np.log(2)/slope)) if slope < 0 else ("NOT_FIT", None))
    return HorizonCurveEvidence(horizons, tuple(means), tuple(counts), half_life, status, refs,
                                zero_tolerance)


def build_regime_evidence(values, regime_labels, available_times, *, regime_kind: str,
                          state_ref=None, min_periods: int = 20):
    data=np.asarray(values,float); labels=np.asarray(regime_labels)
    if data.ndim != 1 or labels.shape != data.shape or len(available_times) != len(data):
        raise ValueError("regime evidence inputs must share one time axis")
    if regime_kind not in {"ex_post_diagnostic", "online_causal"}:
        raise ValueError("unknown regime kind")
    if regime_kind == "online_causal" and not state_ref:
        raise ValueError("online causal regime requires frozen state_ref")
    conditional, counts = {}, {}
    for regime in sorted(set(labels.tolist())):
        sample=data[(labels==regime)&np.isfinite(data)]; counts[int(regime)]=len(sample)
        conditional[int(regime)] = float(np.mean(sample)) if len(sample)>=min_periods else None
    status = "VALID" if all(v is not None for v in conditional.values()) else "INSUFFICIENT"
    return RegimeEvidence(regime_kind, tuple(int(v) for v in labels), tuple(available_times),
                          conditional, counts, state_ref, status)
