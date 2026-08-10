# -*- coding: utf-8 -*-
"""Round-11 round-2 technical state-machine / ParamSpec regression tests.

Covers the five P0 items for the parameterized technical indicators:

1. KAMA re-warmup after suspension: a gap INVALIDATES the recursive state and
   KAMA emits NaN until ``er_window`` consecutive finite prices re-accumulate
   (ER re-derived over that contiguous history) — never a frozen flat line.
2. Supertrend gap detection is not close-only: a bar whose high/low/close or
   derived ATR-based bands are non-finite breaks the state (a finite close on a
   high/NaN + low/NaN bar must NOT keep the stale trend alive).
3. Hidden ``missing_policy`` / ``max_gap`` knobs are removed from KAMA /
   Supertrend / SupertrendDirection / PSAR — single ``break + rewarm`` policy.
4. Supertrend ``multiplier=0`` is rejected (collapses upper/lower to mid).
5. FULL ParamSpec sweep for the technical indicators: every scalar parameter
   has a formal ``ParamSpec`` (int windows reject 20.5), and the relational
   constraints (fast<slow, short<medium<long, TSI long>short,
   PSAR acceleration<=maximum) reject degenerate search nodes.

The full ``load_all()`` is currently blocked by an unrelated in-flight surface
edit (``state_since_reduce`` / ``ts_spectral_entropy`` still listed in
``operator_surface.EXTENDED_ONLY_CANONICALS``), so this module bootstraps just
the technical chain directly (mirroring ``load_all``'s import order).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    """Bootstrap the technical operator chain without the layer_governance step.

    Mirrors ``_LOAD_MODULES`` order (signal -> polars_signal -> composite_fastpath
    -> technical.indicators_v2 -> polars_indicators_v2 -> polars_tech_misc) so the
    KAMA override pin (``expected_old_source="composite_fastpath_primitives"``)
    resolves.  No-op when the registry is already frozen or the chain is present.
    """
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if (
        OperatorRegistry.get("KAMA", "pandas_numpy") is not None
        and OperatorRegistry.get("DMI_plus", "pandas_numpy") is not None
    ):
        return
    from cleaned_operators.technical import signal  # noqa: F401
    from cleaned_operators.technical import polars_signal  # noqa: F401
    from cleaned_operators import composite_fastpath  # noqa: F401
    from cleaned_operators.technical import indicators_v2  # noqa: F401
    from cleaned_operators.technical import polars_indicators_v2  # noqa: F401
    from cleaned_operators.technical import polars_tech_misc  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _frame(*values) -> pd.DataFrame:
    return pd.DataFrame({"A": [float(v) for v in values]})


def _ohlc_from_close(close: pd.DataFrame, spread: float = 1.0):
    high = close + spread
    low = close - spread
    return high, low


def _op(name: str):
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


# ---------------------------------------------------------------------------
# TASK 1 — KAMA re-warmup after suspension (P0)
# ---------------------------------------------------------------------------


def test_kama_rewarm_requires_er_window_consecutive_values():
    from cleaned_operators.technical.indicators_v2 import KAMA

    # Gap at row 2; er_window=3 so the re-seed needs er_window+1 = 4 consecutive
    # finite prices AFTER the gap (rows 3,4,5,6 -> first legal seed at row 6).
    x = _frame(1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0)
    out = KAMA(x, 3, 2, 5)
    vals = out["A"].tolist()
    # rows 0,1: initial ER warmup (< er_window+1) -> NaN
    assert np.isnan(vals[0]) and np.isnan(vals[1])
    # row 2: the gap itself -> NaN
    assert np.isnan(vals[2])
    # rows 3,4,5: post-gap contiguous count 1,2,3 (< 4) -> NaN (no re-seed yet;
    # R30 §21 — the ER needs close vs close-shift(er), i.e. er+1 prices)
    assert np.isnan(vals[3]) and np.isnan(vals[4]) and np.isnan(vals[5])
    # row 6: fourth contiguous finite value -> first legal seed (current close)
    assert vals[6] == pytest.approx(6.0)
    # row 7: ER is finite again -> KAMA keeps updating, no frozen flat line
    assert vals[7] != pytest.approx(vals[6])


def test_kama_initial_warmup_emits_nan_not_frozen_seed():
    from cleaned_operators.technical.indicators_v2 import KAMA

    # Even at series start the first er_window bars must NOT be a flat frozen
    # seed line (that is the same "not a valid KAMA" pathology as post-gap).
    # R30 §21: the ER needs er_window+1 prices, so the first legal seed is the
    # (er+1)-th bar (row 3 for er=3), using the current close.
    x = _frame(1.0, 2.0, 3.0, 4.0, 5.0)
    out = KAMA(x, 3, 2, 5)
    vals = out["A"].tolist()
    assert np.isnan(vals[0]) and np.isnan(vals[1]) and np.isnan(vals[2])
    assert vals[3] == pytest.approx(4.0)  # 4th bar = er+1 -> first seed
    assert np.isfinite(vals[4])
    # no flat frozen prefix
    assert not (vals[3] == vals[4])


def test_kama_rewarm_longer_gap_stays_nan_until_er_window():
    from cleaned_operators.technical.indicators_v2 import KAMA

    # Two missing bars, then only two finite bars before the series ends:
    # never enough contiguous history to re-seed -> all NaN after the gap.
    x = _frame(1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0)
    out = KAMA(x, 3, 2, 5)
    vals = out["A"].tolist()
    assert np.isnan(vals[4]) and np.isnan(vals[5]) and np.isnan(vals[6])


# ---------------------------------------------------------------------------
# TASK 2 — Supertrend gap detection is not close-only (P0)
# ---------------------------------------------------------------------------


def test_supertrend_high_low_nan_with_finite_close_breaks_state():
    from cleaned_operators.technical.indicators_v2 import Supertrend

    high = _frame(1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0)
    low = _frame(0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0)
    close = _frame(1.5, 2.5, 3.5, 4.0, 4.5, 5.5, 6.5)  # close is FINITE at row 3

    st = Supertrend(high, low, close, 3, 2.0)
    vals = st["A"].tolist()
    # The high/NaN + low/NaN bar with a finite close must NOT keep the stale
    # trend alive: row 3 is NaN (state broken) rather than a carried level.
    assert np.isnan(vals[3])
    # R30 §22: after a gap the FIRST valid bar is UNKNOWN (no manufactured
    # direction), so row 4 stays NaN; direction re-asserts from the price/band
    # relationship on the FOLLOWING valid bar (row 5).
    assert np.isnan(vals[4])
    assert np.isfinite(vals[5])


def test_supertrend_stale_trend_not_carried_through_bad_bar():
    from cleaned_operators.technical.indicators_v2 import Supertrend

    # Establish a clear up-trend through row 4, then a bar whose high/low are
    # NaN but whose close is finite.  The stale trend must NOT be carried.
    high = _frame(1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0)
    low = _frame(0.0, 1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0)
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 7.5, 8.5)
    st = Supertrend(high, low, close, 3, 2.0)
    vals = st["A"].tolist()
    assert np.isfinite(vals[4])      # trend alive before the bad bar
    assert np.isnan(vals[5])         # high/low NaN + finite close -> state broken
    # R30 §22: row 6 (first valid after gap) is UNKNOWN; row 7 re-asserts.
    assert np.isnan(vals[6])
    assert np.isfinite(vals[7])


def test_supertrend_full_ohlc_gap_breaks_and_rewarms():
    from cleaned_operators.technical.indicators_v2 import Supertrend

    high = _frame(1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    low = _frame(0.0, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    close = _frame(1.5, 2.5, 3.5, np.nan, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)

    st = Supertrend(high, low, close, 3, 2.0)
    vals = st["A"].tolist()
    assert np.isnan(vals[3])  # the gap bar itself
    # R30 §22: after a gap the trend is UNKNOWN — the first valid bars do NOT
    # manufacture a direction.  rows 4 (first valid) and 5 stay NaN while the
    # direction re-asserts from the price/band relationship; a real level
    # resumes once the trend is established (row 6 onward).
    assert np.isnan(vals[4]) and np.isnan(vals[5])
    assert np.isfinite(vals[6])


# ---------------------------------------------------------------------------
# TASK 3 — hidden knobs missing_policy / max_gap removed (P0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["KAMA", "Supertrend", "SupertrendDirection", "PSAR"],
)
def test_hidden_missing_state_knobs_removed(name):
    from cleaned_operators.technical.indicators_v2 import KAMA, Supertrend, SupertrendDirection, PSAR

    fn = {"KAMA": KAMA, "Supertrend": Supertrend, "SupertrendDirection": SupertrendDirection, "PSAR": PSAR}[name]
    sig = inspect.signature(fn)
    assert "missing_policy" not in sig.parameters
    assert "max_gap" not in sig.parameters

    op = _op(name)
    assert "missing_policy" not in op.metadata.param_names
    assert "max_gap" not in op.metadata.param_names


# ---------------------------------------------------------------------------
# TASK 4 — Supertrend(multiplier=0) degenerate (P0)
# ---------------------------------------------------------------------------


def test_supertrend_multiplier_zero_rejected_at_kernel():
    from cleaned_operators.technical.indicators_v2 import Supertrend

    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    high, low = _ohlc_from_close(close)
    with pytest.raises(ValueError, match="multiplier must be > 0"):
        Supertrend(high, low, close, 3, 0.0)


def test_supertrend_multiplier_zero_rejected_at_param_spec():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    high, low = _ohlc_from_close(close)
    with pytest.raises(Exception):
        _op("Supertrend").calculate(high, low, close, atr_window=3, multiplier=0)


# ---------------------------------------------------------------------------
# TASK 5 — FULL ParamSpec sweep (search-space cleanup)
# ---------------------------------------------------------------------------

# Every scalar parameter of these operators must carry a formal ParamSpec.
_SPEC_SWEEP = {
    "DMI_plus": ["window"],
    "DMI_minus": ["window"],
    "DX": ["window"],
    "NATR": ["window"],
    "PPO": ["fast_window", "slow_window"],
    "PPO_signal": ["fast_window", "slow_window", "signal_window"],
    "PPO_hist": ["fast_window", "slow_window", "signal_window"],
    "PVO": ["fast_window", "slow_window"],
    "PVO_signal": ["fast_window", "slow_window", "signal_window"],
    "PVO_hist": ["fast_window", "slow_window", "signal_window"],
    "CMO": ["window"],
    "VortexPlus": ["window"],
    "VortexMinus": ["window"],
    "KeltnerMid": ["ema_window"],
    "KeltnerUpper": ["ema_window", "atr_window", "multiplier"],
    "KeltnerLower": ["ema_window", "atr_window", "multiplier"],
    "KeltnerPosition": ["ema_window", "atr_window", "multiplier"],
    "TSI": ["long_window", "short_window"],
    "TSI_signal": ["long_window", "short_window", "signal_window"],
    "UltimateOscillator": ["short_window", "medium_window", "long_window", "short_weight", "medium_weight", "long_weight"],
    "DEMA": ["window"],
    "TEMA": ["window"],
    "ichimoku_tenkan": ["tenkan_window"],
    "ichimoku_kijun": ["kijun_window"],
    "ichimoku_senkou_a": ["tenkan_window", "kijun_window"],
    "ichimoku_senkou_b": ["senkou_b_window"],
    "ichimoku_cloud_width": ["tenkan_window", "kijun_window", "senkou_b_window"],
    "ichimoku_cloud_position": ["tenkan_window", "kijun_window", "senkou_b_window"],
    "KAMA": ["er_window", "fast_window", "slow_window"],
    "Supertrend": ["atr_window", "multiplier"],
    "SupertrendDirection": ["atr_window", "multiplier"],
    "PSAR": ["acceleration", "maximum"],
}


@pytest.mark.parametrize("name", sorted(_SPEC_SWEEP))
def test_param_specs_cover_every_scalar_parameter(name):
    op = _op(name)
    specs = op.metadata.param_specs
    missing = [p for p in _SPEC_SWEEP[name] if p not in specs]
    assert not missing, f"{name} missing ParamSpec for {missing}"
    for p in _SPEC_SWEEP[name]:
        assert specs[p].dtype is not None, f"{name}.{p} has no dtype declared"


@pytest.mark.parametrize(
    "name,param,panels",
    [
        ("DMI_plus", "window", ("high", "low", "close")),
        ("CMO", "window", ("close",)),
        ("KAMA", "er_window", ("close",)),
    ],
)
def test_int_window_rejects_fractional_value(name, param, panels):
    # 20.5 must be REJECTED, never silently truncated to 20 (fake search node).
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    high, low = _ohlc_from_close(close)
    frames = {"high": high, "low": low, "close": close}
    op = _op(name)
    kwargs = {param: 20.5}
    with pytest.raises(Exception, match="integer"):
        op.calculate(*[frames[p] for p in panels], **kwargs)


def test_tsi_window_ordering_rejected():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    op = _op("TSI")
    # long < short creates a near-duplicate search node -> rejected.
    with pytest.raises(Exception, match="long_window"):
        op.calculate(close, long_window=3, short_window=10)
    # Ordered combination is accepted.
    out = op.calculate(close, long_window=10, short_window=3)
    assert list(out.columns) == ["A"]


def test_ppo_fast_slow_ordering_rejected():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    op = _op("PPO")
    with pytest.raises(Exception, match="fast_window"):
        op.calculate(close, fast_window=10, slow_window=5)


def test_ultimate_oscillator_window_chain_ordering():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    high, low = _ohlc_from_close(close)
    op = _op("UltimateOscillator")
    with pytest.raises(Exception, match="short_window"):
        op.calculate(high, low, close, short_window=10, medium_window=5, long_window=20)
    with pytest.raises(Exception, match="medium_window"):
        op.calculate(high, low, close, short_window=5, medium_window=20, long_window=10)


def test_psar_acceleration_bounded_by_maximum():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    high, low = _ohlc_from_close(close)
    op = _op("PSAR")
    with pytest.raises(Exception, match="acceleration"):
        op.calculate(high, low, acceleration=0.3, maximum=0.2)


def test_kama_fast_slow_ordering_rejected():
    close = _frame(1.5, 2.5, 3.5, 4.5, 5.5)
    op = _op("KAMA")
    with pytest.raises(Exception, match="fast_window"):
        op.calculate(close, er_window=5, fast_window=10, slow_window=3)


def test_multiplier_param_spec_is_strict_positive():
    spec = _op("Supertrend").metadata.param_specs["multiplier"]
    assert spec.dtype is float
    assert spec.min is not None and spec.min > 0
    spec2 = _op("PSAR").metadata.param_specs["acceleration"]
    assert spec2.dtype is float and spec2.min is not None and spec2.min > 0


# ---------------------------------------------------------------------------
# KAMA pandas vs polars parity with the new re-warmup semantics
# ---------------------------------------------------------------------------


def test_kama_polars_parity_rewarm_semantics():
    pl = pytest.importorskip("polars")
    from cleaned_operators.technical.indicators_v2 import KAMA
    from cleaned_operators.technical.polars_signal import _kama_1d

    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rng = np.random.default_rng(5)
    close = pd.DataFrame(np.cumsum(rng.normal(0.0, 1.0, 30)) + 100, index=idx, columns=["A"])
    close.iloc[10, 0] = np.nan  # suspension
    er, fast, slow = 5, 2, 30

    s = close["A"]
    change = (s - s.shift(er)).abs()
    vol = s.diff().abs().rolling(er, min_periods=er).sum()
    efficiency = change / vol.replace(0, np.nan)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2

    pandas_out = KAMA(close, er, fast, slow)["A"].to_numpy()
    polars_out = _kama_1d(close["A"].to_numpy(), sc.to_numpy(), er)
    np.testing.assert_allclose(pandas_out, polars_out, rtol=1e-10, atol=1e-10, equal_nan=True)
    # The gap region is fully NaN until er consecutive prices re-accumulate.
    assert np.isnan(pandas_out[10]) and np.isnan(pandas_out[11]) and np.isnan(pandas_out[12])
