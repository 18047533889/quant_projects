# -*- coding: utf-8 -*-
"""Review #27 (P0) — price-level spectrum vs return spectrum are distinct objects.

A detrended-price spectrum and a differenced-return spectrum are NOT the same
statistical object (they have different DC/low-frequency structure), so they must
not share one canonical with a single ``return_or_continuous_price`` input mode.

The spectral entropy family is split into two directions with per-direction
typed ``input_units`` contracts:

* ``ts_return_spectral_entropy``         — entropy of the spectrum of (log)returns,
  accepts return-typed input only (``{"x": "return_decimal"}``);
* ``ts_detrended_level_spectral_entropy``— entropy of the spectrum of the
  detrended (log)price level, accepts price-level input only
  (``{"x": "continuous_price"}``).

The legacy generic name ``ts_spectral_entropy`` is kept as a LIVE canonical
spelling of the return-direction spectrum (documented mapping: it ==
``ts_return_spectral_entropy``) so the typed-IR spectral gate (which is keyed on
the literal resolved canonical name) keeps rejecting raw split-sensitive price
for the legacy name.  Both directions fail closed (``ValueError``) on the wrong
input type.  Modules are imported directly (their registration is
import-triggered) so this file stays self-contained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.spectral  # noqa: F401  (shared _periodogram kernel)
import cleaned_operators.spectral_ext  # noqa: F401  (registration is import-triggered)

from cleaned_operators.registry import OperatorRegistry

_WINDOW = 60


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame(values, index=idx, columns=["A"])


def _op(canonical: str):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    return op


def _deterministic_price_level(n: int = 120) -> pd.DataFrame:
    """Deterministic (log)price level with a clear nonstationary level path.

    ``log(level) = 0.006*t + 0.1*sin(2*pi*t/16)`` so the level has strong
    low-frequency / trend energy while the log-returns (a centred, both-sign,
    stationary cosine-ish series) do not.
    """
    t = np.arange(float(n))
    level = 100.0 * np.exp(0.006 * t + 0.1 * np.sin(2.0 * np.pi * t / 16.0))
    return _frame(level)


def _log_returns_of(price: pd.DataFrame) -> pd.DataFrame:
    level = price["A"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(level))
    return _frame(log_ret, start=str(price.index[1].date()))


# ---------------------------------------------------------------------------
# (a) the two canonicals measure DIFFERENT statistical objects
# ---------------------------------------------------------------------------
def test_return_and_detrended_level_entropies_differ():
    price = _deterministic_price_level()
    ret = _log_returns_of(price)

    return_ent = _op("ts_return_spectral_entropy").calculate(ret, window=_WINDOW)
    level_ent = _op("ts_detrended_level_spectral_entropy").calculate(price, window=_WINDOW)

    v_ret = return_ent["A"].dropna()
    v_lvl = level_ent["A"].dropna()
    assert v_ret.size > 0, "return-direction entropy produced no finite values"
    assert v_lvl.size > 0, "detrended-level entropy produced no finite values"

    # The two directions are different statistical objects: the detrended-level
    # spectrum retains low-frequency/curvature structure the differenced return
    # spectrum does not, so the two entropies must NOT coincide.
    assert float(v_ret.iloc[-1]) != pytest.approx(float(v_lvl.iloc[-1]), abs=1e-9)
    assert float(np.nanmean(v_ret.values) != pytest.approx(float(np.nanmean(v_lvl.values)), abs=1e-9)


def test_legacy_ts_spectral_entropy_matches_return_direction():
    """Documented mapping: ``ts_spectral_entropy`` == ``ts_return_spectral_entropy``.

    Both are the return spectrum (same kernel, same typed input contract), so on
    the SAME return-typed input they must produce bit-identical outputs and share
    the return-direction ``input_units`` contract.
    """
    ret = _log_returns_of(_deterministic_price_level())

    legacy = _op("ts_spectral_entropy").calculate(ret, window=_WINDOW)
    canonical = _op("ts_return_spectral_entropy").calculate(ret, window=_WINDOW)
    assert np.allclose(
        legacy.to_numpy(dtype=float),
        canonical.to_numpy(dtype=float),
        equal_nan=True,
    )

    leg_meta = _op("ts_spectral_entropy").metadata
    ret_meta = _op("ts_return_spectral_entropy").metadata
    assert leg_meta.input_units == {"x": "return_decimal"}
    assert leg_meta.input_units == ret_meta.input_units


# ---------------------------------------------------------------------------
# (b) fail closed on the wrong input type
# ---------------------------------------------------------------------------
def test_return_canonical_rejects_price_level_input():
    price = _deterministic_price_level()
    with pytest.raises(ValueError, match="return-typed input only"):
        _op("ts_return_spectral_entropy").calculate(price, window=_WINDOW)
    # the legacy return spelling fails closed too
    with pytest.raises(ValueError, match="return-typed input only"):
        _op("ts_spectral_entropy").calculate(price, window=_WINDOW)


def test_detrended_level_canonical_rejects_return_input():
    ret = _log_returns_of(_deterministic_price_level())
    with pytest.raises(ValueError, match="price-level input only"):
        _op("ts_detrended_level_spectral_entropy").calculate(ret, window=_WINDOW)


# ---------------------------------------------------------------------------
# (c) both are finite on normal data
# ---------------------------------------------------------------------------
def test_both_finite_on_normal_data():
    rng = np.random.default_rng(0)
    returns_like = _frame(rng.normal(0.0, 1.0, 120))

    out_ret = _op("ts_return_spectral_entropy").calculate(returns_like, window=_WINDOW)
    v_ret = out_ret["A"].dropna()
    assert v_ret.size > 0
    assert np.all(np.isfinite(v_ret.values))
    assert ((v_ret >= 0.0) & (v_ret <= 1.0)).all()  # normalized Shannon entropy

    # a random-walk price level is a valid level input (and finite output)
    rw = _frame(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 120))))
    out_lvl = _op("ts_detrended_level_spectral_entropy").calculate(rw, window=_WINDOW)
    v_lvl = out_lvl["A"].dropna()
    assert v_lvl.size > 0
    assert np.all(np.isfinite(v_lvl.values))
    assert ((v_lvl >= 0.0) & (v_lvl <= 1.0)).all()

    # a LOG-price level is equally valid (the canonical accepts (log)price-level)
    log_price = _frame(np.log(rw["A"].to_numpy(dtype=float)))
    out_log = _op("ts_detrended_level_spectral_entropy").calculate(log_price, window=_WINDOW)
    v_log = out_log["A"].dropna()
    assert v_log.size > 0
    assert np.all(np.isfinite(v_log.values))


# ---------------------------------------------------------------------------
# typed input_units contract + semantic identity separation
# ---------------------------------------------------------------------------
def test_per_direction_input_units_contract():
    ret_op = _op("ts_return_spectral_entropy")
    lvl_op = _op("ts_detrended_level_spectral_entropy")
    assert ret_op.metadata.input_units == {"x": "return_decimal"}
    assert lvl_op.metadata.input_units == {"x": "continuous_price"}
    # the input mode participates in the operator contract hash -> factor identity
    from cleaned_operators.registry import _contract_hash

    assert _contract_hash(ret_op) != _contract_hash(lvl_op)
