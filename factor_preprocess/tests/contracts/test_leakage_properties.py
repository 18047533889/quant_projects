"""
Leakage property tests for causal feature transforms (FP closure review).

Three properties are asserted against the package's own transforms:

- CausalPrefixInvariance (test_causal_prefix_invariance)
  A causal transform applied to the prefix [0..t] must agree at time t with
  the same transform computed on the full series. Any disagreement is
  evidence of look-ahead (rows > t influencing the value at t).

- SplitFitIsolation (test_split_fit_isolation_per_row / ..._windowed)
  State/statistics used to transform X_test must be derived only from train.
  For a per-row transform (cross-sectional rank on a time slice) fitting on
  X_train then transforming X_test equals fitting on X_test alone (leak-free).
  For a windowed transform (rolling mean) the first `w` rows of X_test
  computed with train context must equal the last `w` rows of the fit on the
  concatenated train+test series.

- PastOnlyInvariance (test_past_only_invariance)
  The value of a transform at time t depends only on rows <= t: perturbing
  all rows with date > t must leave the output at t unchanged.
"""
import numpy as np
import pandas as pd

from factor_preprocess.transforms.cross_sectional import cs_rank
from factor_preprocess.transforms.rolling import rolling_mean


# ============================================================================
# Shared synthetic panel
# ============================================================================


def _make_panel(n_times: int = 12, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    """Sorted [asset_id, date, value] panel. Values are per-asset random."""
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(n_assets):
        for t in range(n_times):
            rows.append((asset, t, float(rng.normal(asset, 1.0))))
    df = pd.DataFrame(rows, columns=["asset_id", "date", "value"])
    return df.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _aligned(series_values: np.ndarray, ref: pd.DataFrame) -> pd.DataFrame:
    """Attach a transform output to its (asset_id, date) key for alignment."""
    out = ref[["asset_id", "date"]].copy()
    out["v"] = np.asarray(series_values)
    return out.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _assert_series_close(a: np.ndarray, b: np.ndarray) -> None:
    """Assert elementwise equality treating NaN == NaN."""
    assert len(a) == len(b), (len(a), len(b))
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], rtol=1e-10, atol=1e-10), (
        np.flatnonzero(~both_nan & ~np.isclose(a, b, rtol=1e-10, atol=1e-10))[:5]
    )


# ============================================================================
# a. CausalPrefixInvariance — no look-ahead in rolling_mean
# ============================================================================


def test_causal_prefix_invariance():
    window = 3
    df = _make_panel(n_times=14, n_assets=3, seed=7)
    full = rolling_mean(df, window=window, min_periods=window).to_numpy()

    for t in range(window, df["date"].max() + 1):  # t beyond the warmup
        prefix = df[df["date"] <= t].reset_index(drop=True)
        prefix_out = rolling_mean(
            prefix, window=window, min_periods=window
        ).to_numpy()

        # Compare every (asset, t) row of the prefix run against the full run.
        pref_aligned = _aligned(prefix_out, prefix)
        full_aligned = _aligned(full, df)
        merged = pref_aligned.merge(
            full_aligned,
            on=["asset_id", "date"],
            suffixes=("_pref", "_full"),
        )
        _assert_series_close(
            merged["v_pref"].to_numpy(), merged["v_full"].to_numpy()
        )


# ============================================================================
# b. SplitFitIsolation — no test information enters train fit
# ============================================================================


def test_split_fit_isolation_per_row():
    """Per-row transform: rank on train-then-test == rank on test alone."""
    rng = np.random.default_rng(42)
    n_time, n_assets = 12, 5
    x = rng.normal(size=(n_time, n_assets))
    split = 6
    x_train, x_test = x[:split], x[split:]

    rank_test = cs_rank(x_test, axis=-1)
    # Transform computed on the full matrix restricted to test rows (no state
    # leaks across rows: cs_rank acts per time slice independently).
    rank_full_test = cs_rank(x, axis=-1)[split:]

    np.testing.assert_allclose(rank_test, rank_full_test, equal_nan=True)


def test_split_fit_isolation_windowed():
    """
    Windowed transform: first `w` test rows with train context equal the
    last `w` test rows of the concatenated train+test fit.
    """
    window = 3
    n_train = 7
    df = _make_panel(n_times=14, n_assets=3, seed=3)
    train = df[df["date"] < n_train].reset_index(drop=True)
    test = df[df["date"] >= n_train].reset_index(drop=True)

    # Reference: fit on the full concatenated series, keep the test rows.
    concat = pd.concat([train, test], ignore_index=True)
    concat_out = rolling_mean(concat, window=window, min_periods=window)
    concat_test = _aligned(
        concat_out.iloc[concat.index[concat["date"] >= n_train]].to_numpy(),
        test,
    )

    # Candidate: fit on X_train (last w rows per asset as context) and apply
    # to X_test. The first w test dates computed this way must match the
    # last w test rows of the concatenated fit.
    train_tail = train.groupby("asset_id", sort=False).tail(window)
    ctx = pd.concat([train_tail, test], ignore_index=True)
    ctx_out = rolling_mean(ctx, window=window, min_periods=window)
    ctx_test = _aligned(ctx_out.iloc[len(train_tail):].to_numpy(), test)

    first_w_dates = sorted(test["date"].unique())[:window]
    first_w = test["date"].isin(first_w_dates)

    # The "last w rows" of the concatenated fit are exactly the test rows
    # (train rows come first in the concat), so this checks equality on the
    # first w test rows computed both ways.
    _assert_series_close(
        concat_test.loc[first_w, "v"].to_numpy(),
        ctx_test.loc[first_w, "v"].to_numpy(),
    )


# ============================================================================
# c. PastOnlyInvariance — output at t depends only on rows <= t
# ============================================================================


def test_past_only_invariance():
    window = 3
    t0 = 8
    df = _make_panel(n_times=14, n_assets=3, seed=5)
    baseline = rolling_mean(df, window=window, min_periods=window).to_numpy()

    # Perturb every row strictly after t0; the output at t0 must not move.
    perturbed = df.copy()
    future = perturbed["date"] > t0
    perturbed.loc[future, "value"] = perturbed.loc[future, "value"] + 1000.0
    perturbed_out = rolling_mean(
        perturbed, window=window, min_periods=window
    ).to_numpy()

    base_at_t = _aligned(baseline, df)
    pert_at_t = _aligned(perturbed_out, perturbed)
    merged = base_at_t.merge(pert_at_t, on=["asset_id", "date"], suffixes=("_b", "_p"))
    at_t0 = merged[merged["date"] == t0]
    _assert_series_close(at_t0["v_b"].to_numpy(), at_t0["v_p"].to_numpy())
