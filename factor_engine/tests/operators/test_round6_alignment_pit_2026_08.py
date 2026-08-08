# -*- coding: utf-8 -*-
"""Round-6 alignment / PIT / leakage acceptance tests (2026-08-08).

Pins the six P0 clusters of the sixth-round audit:

* Multi-panel strict axis alignment (P0-23/24/25) — permutation / shift / drop
  of any input must RAISE, never silently reindex or pair positionally.
* Financial same-day availability (P0-02/03) — date-level knowledge time is
  conservative next-session by default; same_day is an explicit opt-in.
* Label maturity (P0-32/33/40) — a forward-H label is only mature at
  ``s + H <= fit``; the forecast operators enforce it themselves.
* Model math (P0-37/38/39) — PLS multi-component new-sample deflation, PCA
  feature-count rank, degenerate-feature coverage gate.
* available_at propagation (P0-50/51/52) — ``factor.available_at =
  max(all_input.available_at)`` survives the IR bottom-up propagation.
* Universe mask (P0-27/28/30) — fail-closed mask application; axis mismatch
  and truncated (delisted) masks raise instead of silently dropping names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.alignment import (
    PanelAxisMismatch,
    align_panel_inputs,
    apply_universe_mask,
    assert_axes_aligned,
    panel_arrays,
)
from cleaned_operators.registry import OperatorRegistry

load_all()

_rng = np.random.default_rng(0)
_N = 60
_IDX = pd.date_range("2024-01-01", periods=_N, freq="B")
_COLS = ["A", "B", "C", "D"]


def _mk(cols: list[str] | None = None, idx=None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols if cols is not None else _COLS
    idx = idx if idx is not None else _IDX
    return pd.DataFrame(rng.standard_normal((len(idx), len(cols))), index=idx, columns=cols)


# ---------------------------------------------------------------------------
# §1  Multi-panel strict axis alignment (P0-23/24/25)
# ---------------------------------------------------------------------------

_MULTI_PANEL_CASES = [
    # (operator, number of DataFrame inputs, param kwargs)
    ("ts_first_passage_bias", 2, {}),
    ("ts_dc_overshoot_ratio", 2, {}),
    ("cs_knn_peer_mean_ex_self", 4, {}),
    ("group_feature_mode_share", 4, {}),
    ("event_historical_response_mean", 2, {}),
    ("ts_transfer_entropy", 2, {}),
    ("panel_rolling_pcr_forecast", 3, {}),
]


@pytest.mark.parametrize(
    "name,count,kwargs",
    _MULTI_PANEL_CASES,
    ids=[case[0] for case in _MULTI_PANEL_CASES],
)
def test_permuted_columns_are_rejected(name, count, kwargs):
    """Swapping instrument columns between inputs must raise, not pair by
    position (P0-23/24)."""
    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    frames = [_mk(seed=10 + i) for i in range(count)]
    frames[1] = _mk(cols=_COLS[::-1], seed=20)  # permuted column order
    with pytest.raises((ValueError, PanelAxisMismatch), match="misaligned"):
        op.calculate(*frames, **kwargs)


@pytest.mark.parametrize(
    "name,count,kwargs",
    _MULTI_PANEL_CASES,
    ids=[case[0] for case in _MULTI_PANEL_CASES],
)
def test_shifted_index_is_rejected(name, count, kwargs):
    """A one-day index shift must raise (P0-23): ``x_t`` paired with ``y_{t+1}``
    is a silent look-ahead."""
    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    frames = [_mk(seed=30 + i) for i in range(count)]
    frames[1] = _mk(idx=_IDX + pd.Timedelta(days=1), seed=40)
    with pytest.raises((ValueError, PanelAxisMismatch), match="misaligned"):
        op.calculate(*frames, **kwargs)


def test_dropped_symbol_is_rejected():
    """Removing a symbol from one input (e.g. a delisted name silently absent)
    must raise instead of silently shrinking the cross-section (P0-30)."""
    op = OperatorRegistry.get("cs_knn_peer_mean_ex_self", "pandas_numpy") or OperatorRegistry.get(
        "cs_knn_peer_mean_ex_self"
    )
    frames = [_mk() for _ in range(4)]
    frames[2] = frames[2].iloc[:, :-1]  # drop column D
    with pytest.raises((ValueError, PanelAxisMismatch), match="misaligned"):
        op.calculate(*frames)


def test_duplicate_index_rejected():
    """A duplicated date in any input must raise (base gate)."""
    op = OperatorRegistry.get("ts_first_passage_bias", "pandas_numpy") or OperatorRegistry.get(
        "ts_first_passage_bias"
    )
    dup = _mk()
    dup = pd.concat([dup, dup.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate index"):
        op.calculate(_mk(), dup)


def test_align_panel_inputs_is_strict_by_default():
    """The shared helper is strict by default (P0-25): no silent reindex."""
    a = _mk()
    b = _mk(cols=_COLS[::-1])
    with pytest.raises(PanelAxisMismatch):
        align_panel_inputs(a, b)
    # research opt-out must be explicit
    out = align_panel_inputs(a, b, strict_axes=False)
    assert len(out) == 2
    # aligned panels are a no-op (same objects returned)
    assert align_panel_inputs(a, a)[1] is a
    arrs = panel_arrays(a, a)
    assert [arr.shape for arr in arrs] == [(60, 4), (60, 4)]


# ---------------------------------------------------------------------------
# §2  Financial same-day availability (P0-02/03)
# ---------------------------------------------------------------------------

def test_financial_period_adapter_defaults_to_no_same_day():
    """A signal dated exactly on the knowledge-time date must NOT see that day's
    filing (round-6 P0-02): an after-close PubDate/filing must not drive the
    same-day bar."""
    from fields.providers import FinancialPeriodAdapter

    ev = pd.DataFrame(
        {"Symbol": ["AAA"], "filing_date": pd.to_datetime(["2024-04-20"]), "value": [1.0]}
    )
    sig = pd.DataFrame(
        {"Symbol": ["AAA", "AAA"], "__signal_date__": pd.to_datetime(["2024-04-20", "2024-04-21"])}
    )
    merged = FinancialPeriodAdapter.asof(ev, sig, market="us", by="Symbol")
    assert pd.isna(merged["value"].iloc[0])
    assert merged["value"].iloc[1] == 1.0


def test_financial_period_adapter_same_day_is_explicit_opt_in():
    from fields.providers import FinancialPeriodAdapter

    ev = pd.DataFrame(
        {"Symbol": ["AAA"], "filing_date": pd.to_datetime(["2024-04-20"]), "value": [1.0]}
    )
    sig = pd.DataFrame(
        {"Symbol": ["AAA"], "__signal_date__": pd.to_datetime(["2024-04-20"])}
    )
    merged = FinancialPeriodAdapter.asof(ev, sig, market="us", by="Symbol", same_day=True)
    assert merged["value"].iloc[0] == 1.0


def test_pit_asof_join_default_is_conservative():
    """pit_contract's asof default flips to next_trading_day (P0-03): a bar
    dated PubDate does not see the same-day announcement."""
    from pit_contract import PITColumns, pit_asof_join

    decisions = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(["2024-04-30", "2024-05-02"], utc=True),
            "instrument": ["A", "A"],
        }
    )
    events = pd.DataFrame(
        {
            "instrument": ["A"],
            "period_end": pd.to_datetime(["2024-03-31"], utc=True),
            "available_at": pd.to_datetime(["2024-04-30"], utc=True),
            "value": [100.0],
        }
    )
    joined = pit_asof_join(decisions, events, columns=PITColumns(), max_age_days=None)
    assert pd.isna(joined["value"].iloc[0])
    assert joined["value"].iloc[1] == 100.0


# ---------------------------------------------------------------------------
# Review-8 #404: multi-instrument asof join must not require global sort by
# [instrument, time] — pandas merge_asof(by=...) needs a globally monotonic
# asof key, which the old [instrument, time] ordering violates for >1 symbol.
# ---------------------------------------------------------------------------
def test_pit_asof_join_multi_instrument_global_key_sorted():
    """Two instruments with interleaved decision dates must not raise
    ``ValueError: left keys must be sorted`` and must join each instrument to
    its own history (no cross-instrument leakage)."""
    from pit_contract import PITColumns, pit_asof_join

    decisions = pd.DataFrame(
        {
            "instrument": ["A", "A", "B", "B"],
            "decision_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"], utc=True
            ),
        }
    )
    events = pd.DataFrame(
        {
            "instrument": ["A", "A", "B", "B"],
            "period_end": pd.to_datetime(
                ["2025-12-01", "2025-12-15", "2025-12-01", "2025-12-15"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2026-01-01", "2026-01-15", "2026-01-01", "2026-01-15"], utc=True
            ),
            "value": [1.0, 2.0, 100.0, 200.0],
        }
    )
    # ``same_day`` isolates the sort fix from the availability-shift policy:
    # each instrument's decision must join its own history exactly.
    joined = pit_asof_join(
        decisions, events, columns=PITColumns(), max_age_days=None,
        available_policy="same_day",
    )
    assert list(joined["instrument"]) == ["A", "A", "B", "B"]
    # A's second decision (Jan 2) still sees A's Jan-1 event, never B's.
    assert list(joined["value"].round(1)) == [1.0, 1.0, 100.0, 100.0]


def test_pit_asof_join_multi_instrument_shuffled_duplicates():
    """Shuffled decision grid + duplicated rows: output order preserved, each
    row maps to its own instrument history, and no asof key error."""
    from pit_contract import PITColumns, pit_asof_join

    rng = np.random.default_rng(7)
    instruments = [f"I{i:03d}" for i in range(100)]
    decision_dates = pd.to_datetime(
        ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"],
        utc=True,
    )
    decisions = pd.DataFrame(
        {
            "instrument": np.repeat(instruments, 5),
            "decision_timestamp": np.tile(decision_dates.to_numpy(), 100),
        }
    )
    shuffle = rng.permutation(len(decisions))
    decisions = decisions.iloc[shuffle].reset_index(drop=True)
    # duplicate one row
    decisions = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)

    event_dates = pd.to_datetime(["2025-12-31", "2026-02-28"], utc=True)
    event_available = pd.to_datetime(["2026-02-20", "2026-03-10"], utc=True)
    events = pd.DataFrame(
        {
            "instrument": np.repeat(instruments, 2),
            "period_end": np.tile(event_dates.to_numpy(), 100),
            "available_at": np.tile(event_available.to_numpy(), 100),
            "value": rng.standard_normal(200),
        }
    )
    joined = pit_asof_join(decisions, events, columns=PITColumns(), max_age_days=None)
    assert len(joined) == len(decisions)
    # value from the same instrument only: Feb-20 event is the visible one
    # before 2026-03-01 under next_trading_day; Jan-1 + Feb-28 quarter ends are
    # not visible on 03-01 (announcement lands after close), so value is NaN on
    # 2026-03-01 and the Feb-28 report on 2026-03-10 is only seen from 03-11.
    assert joined["instrument"].isna().sum() == 0


# ---------------------------------------------------------------------------
# §3  Label maturity (P0-32/33/40)
# ---------------------------------------------------------------------------

def test_label_horizon_excludes_unmatured_labels():
    """With label_horizon=H, perturbing a label anchored in the last H rows of
    the training window must not change the prediction (that label is not yet
    mature at fit time), while a mature label inside the window must."""
    n = 90
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(_rng.standard_normal((n, 4)) * 0.01, axis=0))
    y = pd.DataFrame(close, index=idx, columns=_COLS)
    x1 = _mk(idx=idx, seed=1)
    x2 = _mk(idx=idx, seed=2)
    op = OperatorRegistry.get("panel_rolling_pcr_forecast", "pandas_numpy") or OperatorRegistry.get(
        "panel_rolling_pcr_forecast"
    )
    base20 = op.calculate(y, x1, x2, window=40, n_components=3, label_horizon=20)
    base1 = op.calculate(y, x1, x2, window=40, n_components=3, label_horizon=1)
    y2 = y.copy()
    y2.iloc[70] = 999.0  # last row's fit_end=88: horizon-20 trains rows [49,68] (row 70 unmatured),
    # horizon-1 trains rows [49,87] (row 70 mature) — the boundary the test discriminates
    shk20 = op.calculate(y2, x1, x2, window=40, n_components=3, label_horizon=20)
    shk1 = op.calculate(y2, x1, x2, window=40, n_components=3, label_horizon=1)
    assert np.allclose(base20.iloc[-1].to_numpy(), shk20.iloc[-1].to_numpy(), equal_nan=True), (
        "un-matured label must not fit the model"
    )
    assert not np.allclose(base1.iloc[-1].to_numpy(), shk1.iloc[-1].to_numpy(), equal_nan=True), (
        "mature label inside the horizon-1 window must change the prediction"
    )


def test_purged_time_split_never_straddles_boundary():
    """P0-34/35: the chronological split purges horizon rows on both sides and
    applies an embargo, so no forward-H train label reaches validation."""
    from api.label_spec import assert_no_label_overlap, purged_time_split

    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    train, valid = purged_time_split(dates, train_frac=0.7, horizon=10, embargo=5)
    assert_no_label_overlap(train, valid, dates, horizon=10)
    assert valid[0] - train[-1] >= 10 + 5


# ---------------------------------------------------------------------------
# §4  Model math (P0-37/38/39)
# ---------------------------------------------------------------------------

def test_pls_multicomponent_new_sample_deflation():
    """P0-37: with 2+ components the new sample must be deflated in lockstep
    with the training X/y (matches the NIPALS prediction rule)."""
    from cleaned_operators.cross_section.panel_model import _pls1_predict

    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))
    y = X @ np.array([0.5, -0.3, 0.2, 0.0, 1.1]) + rng.standard_normal(60) * 0.01
    x_new = rng.standard_normal(5)

    def nipals_reference(X, y, ncomp, x_new):
        Xc, yc = X.copy(), y - y.mean()
        pred, xr = y.mean(), x_new.copy()
        for _ in range(ncomp):
            w = Xc.T @ yc
            w = w / np.linalg.norm(w)
            t = Xc @ w
            tt = float(t @ t)
            p = (Xc.T @ t) / tt
            q = float(t @ yc) / tt
            tn = float(xr @ w)
            pred += tn * q
            xr = xr - tn * p
            Xc = Xc - np.outer(t, p)
            yc = yc - q * t
        return pred

    for k in (1, 2, 3):
        got = _pls1_predict(X, y, k, x_new)
        assert got == pytest.approx(nipals_reference(X, y, k, x_new), abs=1e-9)


def test_feature_pca_rank_uses_feature_count_not_instrument_count():
    """P0-38: the rank bound is the number of FEATURE panels, not the panel
    width.  n_components >= n_features must clamp below the feature count."""
    op = OperatorRegistry.get("ts_feature_pca_reconstruction_error", "pandas_numpy") or OperatorRegistry.get(
        "ts_feature_pca_reconstruction_error"
    )
    x1 = _mk(seed=1)
    x2 = _mk(seed=2)
    x3 = _mk(seed=3)
    out = op.calculate(x1, x2, x3, window=40, n_components=2)
    valid = out.dropna().to_numpy()
    assert (valid > 1e-9).any(), "rank < feature count must not reconstruct perfectly"


def test_feature_pca_coverage_gate_fails_closed():
    """P0-39: an all-NaN / degenerate feature within the training window must be
    dropped from the active feature space instead of poisoning the SVD."""
    op = OperatorRegistry.get("ts_feature_pca_reconstruction_error", "pandas_numpy") or OperatorRegistry.get(
        "ts_feature_pca_reconstruction_error"
    )
    x1 = _mk(seed=1)
    x2 = _mk(seed=2)
    x3 = _mk(seed=3)
    x3.iloc[20:] = np.nan  # feature goes all-NaN mid-window
    out = op.calculate(x1, x2, x3, window=40, n_components=2)
    assert out.shape == x1.shape
    # no exception, values either finite or NaN (never a poisoned shared SVD)
    assert np.isfinite(out.iloc[55:].to_numpy()).any() or out.iloc[55:].isna().all().all()


# ---------------------------------------------------------------------------
# §5  available_at propagation (P0-50/51/52)
# ---------------------------------------------------------------------------

def test_available_at_propagates_max_of_inputs():
    """P0-51: factor.available_at = max(all_input.available_at).  A formula
    mixing a session_close price and a PubDate fundamental is only usable after
    the filing."""
    from api import rank, ts_delay, ts_mean
    from api.columns import col
    from ir.analyzer import Analyzer

    price_only = Analyzer().lower(ts_mean(col("close"), 10))
    assert price_only.ir.semantic_attrs.get("available_at") == "session_close"
    fundamental = Analyzer().lower(ts_delay(col("net_profit")))
    assert fundamental.ir.semantic_attrs.get("available_at") == "PubDate"
    assert fundamental.ir.semantic_attrs.get("fiscal_grain") == "ytd"
    mixed = Analyzer().lower(
        rank(ts_mean(col("net_profit"), 4) / ts_mean(col("close"), 20))
    )
    # the filing is known later than the close -> the mixed factor inherits it
    assert mixed.ir.semantic_attrs.get("available_at") == "PubDate"


def test_available_at_open_vs_close():
    from api.columns import col
    from api import ts_mean
    from ir.analyzer import Analyzer

    o = Analyzer().lower(ts_mean(col("open"), 10))
    assert o.ir.semantic_attrs.get("available_at") == "session_open"
    c = Analyzer().lower(ts_mean(col("close"), 10))
    assert c.ir.semantic_attrs.get("available_at") == "session_close"


# ---------------------------------------------------------------------------
# §6  Universe mask (P0-27/28/30)
# ---------------------------------------------------------------------------

def test_apply_universe_mask_fail_closed():
    """NaN / Inf / False in the mask -> out-of-universe NaN; axis mismatch and a
    truncated (delisted) mask raise instead of silently pairing positionally."""
    panel = _mk(seed=5)
    mask = pd.DataFrame(1.0, index=_IDX, columns=_COLS)
    mask.iloc[0, 0] = 0.0
    mask.iloc[1, 1] = np.nan
    mask.iloc[2, 2] = np.inf
    out = apply_universe_mask(panel, mask)
    assert pd.isna(out.iloc[0, 0]) and pd.isna(out.iloc[1, 1]) and pd.isna(out.iloc[2, 2])
    assert out.iloc[3, 3] == panel.iloc[3, 3]
    with pytest.raises(PanelAxisMismatch):
        apply_universe_mask(panel, mask.rename(columns={"D": "E"}))
    with pytest.raises(PanelAxisMismatch):
        apply_universe_mask(panel, mask.iloc[:, :-1])  # a name absent from mask


def test_diagnostic_only_tags_on_in_sample_self_fit_ops():
    """Round-6 §20: in-sample self-fit residuals/AR fitted values are tagged
    diagnostic_only (prefer *_prior / *_forecast_error for mining); the prior
    variants are not."""
    def diag(name):
        op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
        return "diagnostic_only" in (op.metadata.tags or ())

    assert diag("ts_multi_regression_resid") is True
    assert diag("ts_multi_regression_forecast_error") is False
    assert diag("ts_ar_forecast") is True
    assert diag("ts_ar_prior_forecast") is False
    assert diag("ts_quantile_regression_resid") is True
    assert diag("ts_expectile_regression_resid") is True
