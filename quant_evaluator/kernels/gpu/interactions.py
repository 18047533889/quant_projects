"""GPU novelty / interactions kernels (spec §30).

Batch-vectorized over the factor axis F on the GPU-friendly (T, F, N)
layout, matching the CPU reference in :mod:`quant_evaluator.metrics.interactions`
exactly:

  - ``batched_pairwise_correlation`` -> ``compute_pairwise_correlation``
  - ``batched_conditional_ic``       -> ``compute_conditional_ic``
  - ``batched_substitution_effect``  -> ``compute_substitution_effect``
  - ``batched_complementarity_score``-> ``compute_complementarity_score``
  - ``batched_interaction_strength`` -> ``compute_interaction_strength``
  - ``batched_incremental_ic``       -> ``compute_incremental_ic`` (IC orthogonalization)

Semantics preserved:
  - pairwise correlation: Pearson or Spearman over the pairwise-finite set,
    min_obs floor, constant rejection (diagonal -> 1.0, off-diagonal -> NaN).
  - conditional IC: IC of each factor within quantiles of a conditioning
    factor (percentile edges, digitize right=False, min_assets floor).
  - substitution / complementarity / interaction: per-period IC of A, B and
    the equal-weighted standardized combination (A+B)/2, with the same
    standardization ``(x - mean) / (std + 1e-8)`` as the CPU reference.
  - incremental IC: residualize the test factor and the label against the
    base factors (OLS with intercept), then IC of the residuals.

CuPy is imported lazily so the module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np


def _import_cp():
    import cupy as cp
    return cp


def _as_tfn(x):
    cp = _import_cp()
    a = cp.asarray(x, dtype=cp.float64)
    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3:
        raise ValueError(f"expected (T, F, N) or (T, N), got ndim={a.ndim}")
    return a


def _pearson_ic(x, y, min_obs):
    """Per-(t,f) Pearson IC between (T,F,N) x and (T,N) y. Returns (T,F)."""
    cp = _import_cp()
    yb = y[:, None, :]  # (T,1,N)
    finite = cp.isfinite(x) & cp.isfinite(yb)
    n = cp.sum(finite, axis=2).astype(cp.float64)  # (T,F)
    x0 = cp.where(finite, x, 0.0)
    y0 = cp.where(finite, yb, 0.0)
    sx = cp.sum(x0, axis=2)
    sy = cp.sum(y0, axis=2)
    sxx = cp.sum(x0 * x0, axis=2)
    syy = cp.sum(y0 * y0, axis=2)
    sxy = cp.sum(x0 * y0, axis=2)
    num = n * sxy - sx * sy
    denom = (n * sxx - sx * sx) * (n * syy - sy * sy)
    denom_safe = cp.maximum(denom, 0.0)
    ic = num / cp.sqrt(denom_safe)
    zero_var = (n * sxx - sx * sx <= 0) | (n * syy - sy * sy <= 0)
    ic = cp.where((n < min_obs) | zero_var | (denom_safe == 0.0), cp.nan, ic)
    return ic


def _spearman_ic(x, y, min_obs):
    """Per-(t,f) Spearman IC between (T,F,N) x and (T,N) y. Returns (T,F)."""
    cp = _import_cp()
    from quant_evaluator.kernels.gpu.rank import batched_rank, batched_distinct_level_count
    yb = y[:, None, :]  # (T,1,N)
    pairwise = cp.isfinite(x) & cp.isfinite(yb)  # (T,F,N)
    x_rank = batched_rank(cp.where(pairwise, x, cp.nan), pct=False)
    y_rank = batched_rank(cp.where(pairwise, yb, cp.nan), pct=False)
    T, F, N = x.shape
    dlx = batched_distinct_level_count(cp.where(pairwise, x, cp.nan)).reshape(T, F)
    dly = batched_distinct_level_count(cp.where(pairwise, yb, cp.nan)).reshape(T, F)
    min_levels = max(min_obs // 2, 2)
    ic = _pearson_ic(x_rank, y, min_obs)
    low = (dlx < min_levels) | (dly < min_levels)
    ic = cp.where(low, cp.nan, ic)
    return ic


def _ic_fn(method):
    return _spearman_ic if method == "spearman" else _pearson_ic


def _standardize(x, eps=1e-8):
    """Standardize (T,F,N) along N: (x - mean) / (std + eps)."""
    cp = _import_cp()
    mean = cp.nanmean(x, axis=2, keepdims=True)
    std = cp.nanstd(x, axis=2, ddof=1, keepdims=True)
    return (x - mean) / (std + eps)


def batched_pairwise_correlation(factor_values, method="pearson", min_obs=30):
    """Pairwise factor correlation matrix, (F, F).

    Matches ``compute_pairwise_correlation`` on a (T, N, F) batch reshaped to
    (T*N, F).  ``factor_values`` may be (T, F, N) or (T, N, F); the latter is
    transposed internally.
    """
    cp = _import_cp()
    a = cp.asarray(factor_values, dtype=cp.float64)
    if a.ndim == 3 and a.shape[1] == a.shape[2]:
        # ambiguous square; assume (T, N, F) with F == N is unlikely; treat as
        # (T, F, N) by default (caller should pass the GPU layout).
        pass
    if a.ndim == 3:
        # decide layout: if the middle dim is the factor count we keep it;
        # the CPU reference uses (T, N, F). We accept (T, F, N) and transpose.
        # Heuristic: the caller passes the GPU-friendly (T, F, N) layout.
        T, F, N = a.shape
        flat = a.reshape(T * F, N)  # (T*F, N) -> columns are assets
        # We need (T*N, F): transpose to (T, N, F) then flatten.
        a_tnf = cp.transpose(a, (0, 2, 1))  # (T, N, F)
        flat = a_tnf.reshape(T * N, F)
    else:
        flat = a
    F = flat.shape[1]
    corr = cp.full((F, F), cp.nan, dtype=cp.float64)
    for i in range(F):
        for j in range(i, F):
            x = flat[:, i]
            y = flat[:, j]
            valid = cp.isfinite(x) & cp.isfinite(y)
            n = cp.sum(valid)
            if n < min_obs:
                continue
            xv = x[valid]
            yv = y[valid]
            if cp.std(xv) == 0 or cp.std(yv) == 0:
                corr[i, j] = 1.0 if i == j else cp.nan
                corr[j, i] = corr[i, j]
                continue
            if method == "pearson":
                c = cp.corrcoef(xv, yv)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                rx = batched_rank(xv[None, :], pct=False)[0]
                ry = batched_rank(yv[None, :], pct=False)[0]
                c = cp.corrcoef(rx, ry)[0, 1]
            corr[i, j] = c
            corr[j, i] = c
    return corr


def batched_conditional_ic(
    factor_values, labels, conditioning_factor_idx, quantiles=5,
    method="pearson", min_assets=10,
):
    """Conditional IC per factor per conditioning quantile, (T, F, Q) + counts.

    Matches ``compute_conditional_ic``.  ``factor_values`` is (T, F, N),
    ``labels`` is (T, N).  Returns (conditional_ic (T,F,Q), sample_counts (T,Q)).
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    cond_ic = cp.full((T, F, quantiles), cp.nan, dtype=cp.float64)
    counts = cp.zeros((T, quantiles), dtype=cp.int32)
    ic_fn = _ic_fn(method)

    for t in range(T):
        cv = x[t, conditioning_factor_idx, :]  # (N,)
        lt = y[t]  # (N,)
        valid_mask = cp.isfinite(cv) & cp.isfinite(lt)
        if cp.sum(valid_mask) < min_assets:
            continue
        cv_fin = cv[valid_mask]
        edges = cp.percentile(cv_fin, cp.linspace(0, 100, quantiles + 1))
        assign = cp.digitize(cv, edges[1:-1], right=False)
        assign = cp.where(valid_mask, assign, -1)
        for q in range(quantiles):
            qmask = assign == q
            nq = cp.sum(qmask)
            if nq < min_assets:
                continue
            counts[t, q] = nq
            # IC of each factor within this quantile
            for f in range(F):
                factor_t = x[t, f, :]
                label_t = lt
                fq = factor_t[qmask]
                lq = label_t[qmask]
                valid_obs = cp.isfinite(fq) & cp.isfinite(lq)
                if cp.sum(valid_obs) < min_assets:
                    continue
                fv = fq[valid_obs]
                lv = lq[valid_obs]
                if cp.std(fv) == 0 or cp.std(lv) == 0:
                    continue
                if method == "pearson":
                    ic = cp.corrcoef(fv, lv)[0, 1]
                else:
                    from quant_evaluator.kernels.gpu.rank import batched_rank
                    rf = batched_rank(fv[None, :], pct=False)[0]
                    rl = batched_rank(lv[None, :], pct=False)[0]
                    ic = cp.corrcoef(rf, rl)[0, 1]
                cond_ic[t, f, q] = ic
    return cond_ic, counts


def batched_substitution_effect(
    factor_values, labels, factor_a_idx, factor_b_idx,
    method="pearson", min_assets=30,
):
    """Per-period IC of A, B and (A+B)/2, each (T,).

    Matches ``compute_substitution_effect``.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    ic_a = cp.full(T, cp.nan, dtype=cp.float64)
    ic_b = cp.full(T, cp.nan, dtype=cp.float64)
    ic_combined = cp.full(T, cp.nan, dtype=cp.float64)
    ic_fn = _ic_fn(method)

    for t in range(T):
        fa = x[t, factor_a_idx, :]
        fb = x[t, factor_b_idx, :]
        lt = y[t]
        valid_a = cp.isfinite(fa) & cp.isfinite(lt)
        valid_b = cp.isfinite(fb) & cp.isfinite(lt)
        valid_both = cp.isfinite(fa) & cp.isfinite(fb) & cp.isfinite(lt)
        if cp.sum(valid_a) >= min_assets:
            fa_v = fa[valid_a]
            la_v = lt[valid_a]
            if cp.std(fa_v) > 0 and cp.std(la_v) > 0:
                if method == "pearson":
                    ic_a[t] = cp.corrcoef(fa_v, la_v)[0, 1]
                else:
                    from quant_evaluator.kernels.gpu.rank import batched_rank
                    rf = batched_rank(fa_v[None, :], pct=False)[0]
                    rl = batched_rank(la_v[None, :], pct=False)[0]
                    ic_a[t] = cp.corrcoef(rf, rl)[0, 1]
        if cp.sum(valid_b) >= min_assets:
            fb_v = fb[valid_b]
            lb_v = lt[valid_b]
            if cp.std(fb_v) > 0 and cp.std(lb_v) > 0:
                if method == "pearson":
                    ic_b[t] = cp.corrcoef(fb_v, lb_v)[0, 1]
                else:
                    from quant_evaluator.kernels.gpu.rank import batched_rank
                    rf = batched_rank(fb_v[None, :], pct=False)[0]
                    rl = batched_rank(lb_v[None, :], pct=False)[0]
                    ic_b[t] = cp.corrcoef(rf, rl)[0, 1]
        if cp.sum(valid_both) >= min_assets:
            fa_b = fa[valid_both]
            fb_b = fb[valid_both]
            l_b = lt[valid_both]
            fa_std = (fa_b - cp.mean(fa_b)) / (cp.std(fa_b) + 1e-8)
            fb_std = (fb_b - cp.mean(fb_b)) / (cp.std(fb_b) + 1e-8)
            combined = (fa_std + fb_std) / 2.0
            if cp.std(combined) > 0 and cp.std(l_b) > 0:
                if method == "pearson":
                    ic_combined[t] = cp.corrcoef(combined, l_b)[0, 1]
                else:
                    from quant_evaluator.kernels.gpu.rank import batched_rank
                    rc = batched_rank(combined[None, :], pct=False)[0]
                    rl = batched_rank(l_b[None, :], pct=False)[0]
                    ic_combined[t] = cp.corrcoef(rc, rl)[0, 1]
    return ic_a, ic_b, ic_combined


def batched_complementarity_score(
    factor_values, labels, factor_a_idx, factor_b_idx,
    method="pearson", min_assets=30, min_periods=20,
):
    """Complementarity score for a factor pair.

    Matches ``compute_complementarity_score``: returns
    (complementarity_score, mean_lift, std_lift).
    """
    cp = _import_cp()
    ic_a, ic_b, ic_combined = batched_substitution_effect(
        factor_values, labels, factor_a_idx, factor_b_idx,
        method=method, min_assets=min_assets,
    )
    valid = cp.isfinite(ic_a) & cp.isfinite(ic_b) & cp.isfinite(ic_combined)
    if cp.sum(valid) < min_periods:
        return cp.nan, cp.nan, cp.nan
    ia = ic_a[valid]
    ib = ic_b[valid]
    icc = ic_combined[valid]
    max_ind = cp.maximum(cp.abs(ia), cp.abs(ib))
    lift = cp.abs(icc) - max_ind
    mean_lift = cp.mean(lift)
    std_lift = cp.std(lift, ddof=1)
    score = mean_lift / std_lift if std_lift > 0 else 0.0
    return score, mean_lift, std_lift


def batched_interaction_strength(
    factor_values, labels, factor_a_idx, factor_b_idx,
    method="pearson", min_assets=30,
):
    """Time-varying interaction strength, (interaction_ic (T,), additive_ic (T,)).

    Matches ``compute_interaction_strength``.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    interaction_ic = cp.full(T, cp.nan, dtype=cp.float64)
    additive_ic = cp.full(T, cp.nan, dtype=cp.float64)

    for t in range(T):
        fa = x[t, factor_a_idx, :]
        fb = x[t, factor_b_idx, :]
        lt = y[t]
        valid = cp.isfinite(fa) & cp.isfinite(fb) & cp.isfinite(lt)
        if cp.sum(valid) < min_assets:
            continue
        fa_v = fa[valid]
        fb_v = fb[valid]
        l_v = lt[valid]
        fa_std = (fa_v - cp.mean(fa_v)) / (cp.std(fa_v) + 1e-8)
        fb_std = (fb_v - cp.mean(fb_v)) / (cp.std(fb_v) + 1e-8)
        interaction = fa_std * fb_std
        additive = (fa_std + fb_std) / 2.0
        if cp.std(interaction) > 0 and cp.std(l_v) > 0:
            if method == "pearson":
                interaction_ic[t] = cp.corrcoef(interaction, l_v)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                ri = batched_rank(interaction[None, :], pct=False)[0]
                rl = batched_rank(l_v[None, :], pct=False)[0]
                interaction_ic[t] = cp.corrcoef(ri, rl)[0, 1]
        if cp.std(additive) > 0 and cp.std(l_v) > 0:
            if method == "pearson":
                additive_ic[t] = cp.corrcoef(additive, l_v)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                ra = batched_rank(additive[None, :], pct=False)[0]
                rl = batched_rank(l_v[None, :], pct=False)[0]
                additive_ic[t] = cp.corrcoef(ra, rl)[0, 1]
    return interaction_ic, additive_ic


def batched_incremental_ic(
    factor_values, labels, base_factor_indices, test_factor_idx,
    method="pearson", min_assets=30,
):
    """Incremental IC of a test factor after residualizing against base factors.

    Matches ``compute_incremental_ic``: returns
    (incremental_ic (T,), base_ic (T,), total_ic (T,)).
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    y = cp.asarray(labels, dtype=cp.float64)
    T, F, N = x.shape
    base_idx = list(base_factor_indices)
    K = len(base_idx)
    incremental_ic = cp.full(T, cp.nan, dtype=cp.float64)
    base_ic = cp.full(T, cp.nan, dtype=cp.float64)
    total_ic = cp.full(T, cp.nan, dtype=cp.float64)

    for t in range(T):
        base_factors = x[t, base_idx, :]  # (K, N)
        test_factor = x[t, test_factor_idx, :]  # (N,)
        label_t = y[t]
        valid_mask = (
            cp.all(cp.isfinite(base_factors), axis=0)
            & cp.isfinite(test_factor)
            & cp.isfinite(label_t)
        )
        if cp.sum(valid_mask) < min_assets:
            continue
        base_valid = base_factors[:, valid_mask].T  # (n, K)
        test_valid = test_factor[valid_mask]
        label_valid = label_t[valid_mask]
        # total IC
        if cp.std(test_valid) > 0 and cp.std(label_valid) > 0:
            if method == "pearson":
                total_ic[t] = cp.corrcoef(test_valid, label_valid)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                rt = batched_rank(test_valid[None, :], pct=False)[0]
                rl = batched_rank(label_valid[None, :], pct=False)[0]
                total_ic[t] = cp.corrcoef(rt, rl)[0, 1]
        # residualize
        X = cp.column_stack([cp.ones(base_valid.shape[0]), base_valid])  # (n, K+1)
        XtX = X.T @ X
        try:
            beta_test = cp.linalg.solve(XtX, X.T @ test_valid)
            beta_label = cp.linalg.solve(XtX, X.T @ label_valid)
        except Exception:
            continue
        test_residual = test_valid - X @ beta_test
        label_residual = label_valid - X @ beta_label
        base_pred = X @ beta_label
        if cp.std(base_pred) > 0 and cp.std(label_valid) > 0:
            if method == "pearson":
                base_ic[t] = cp.corrcoef(base_pred, label_valid)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                rb = batched_rank(base_pred[None, :], pct=False)[0]
                rl = batched_rank(label_valid[None, :], pct=False)[0]
                base_ic[t] = cp.corrcoef(rb, rl)[0, 1]
        if cp.std(test_residual) > 0 and cp.std(label_residual) > 0:
            if method == "pearson":
                incremental_ic[t] = cp.corrcoef(test_residual, label_residual)[0, 1]
            else:
                from quant_evaluator.kernels.gpu.rank import batched_rank
                rt = batched_rank(test_residual[None, :], pct=False)[0]
                rl = batched_rank(label_residual[None, :], pct=False)[0]
                incremental_ic[t] = cp.corrcoef(rt, rl)[0, 1]
    return incremental_ic, base_ic, total_ic
