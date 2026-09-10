"""H22 actual registry negative controls (isolated test monkeypatches only)."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators import math_certificate as mc
from factor_engine.cleaned_operators.registry import OperatorRegistry

def test_actual_winner_bad_second_column_is_rejected(monkeypatch):
    class Bad:
        metadata = SimpleNamespace(param_names=["x", "window"], param_specs={})
        calls = 0
        def calculate(self, a, window=10):
            self.calls += 1
            out = a.copy()
            out.iloc[:, 1] = a.iloc[-1, 0]
            return out
    bad = Bad()
    original = OperatorRegistry.get
    def get(cls, name, backend="pandas_numpy", *, mode="production"):
        return bad if name == "ts_mean" else original(name, backend=backend, mode=mode)
    monkeypatch.setattr(OperatorRegistry, "get", classmethod(get))
    ok, detail = mc.check_prefix_invariance_all_causal_ts(canonicals=["ts_mean"])
    assert not ok and bad.calls == 2
    row = detail["per_canonical"]["ts_mean"]
    assert row["test_executed"] and row["implementation_hash"] and row["class"].endswith("Bad")
    assert mc.reference_self_check_causal_examples()[0]

def test_actual_selected_winner_and_zero_coverage():
    ok, detail = mc.check_prefix_invariance_all_causal_ts(canonicals=["ts_mean"])
    assert ok, detail
    assert detail["per_canonical"]["ts_mean"]["test_executed"]
    assert not mc.check_prefix_invariance_all_causal_ts(canonicals=[])[0]
    ok, detail = mc.check_prefix_invariance_all_causal_ts(canonicals=["not_a_real_operator"])
    assert not ok
    assert detail["per_canonical"]["not_a_real_operator"]["status"] == "BLOCKED"

def test_adx_registered_definition_and_independent_clock():
    from factor_engine.recursive_kernel import adx_segment
    from factor_engine.cleaned_operators.overhaul.technical import pd_adx
    rng = np.random.default_rng(17)
    close = 100 + rng.normal(size=100).cumsum()
    high, low = close + 1, close - 1
    high[30] = np.nan
    low[35] = np.nan
    close[40] = np.nan
    expected = pd_adx(*[pd.DataFrame({"x": a}) for a in [high, low, close]], window=5)["x"].to_numpy()
    actual, _ = adx_segment(high, low, close, {}, 5)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isnan(actual[:9]).all()
    first, state = adx_segment(high[:44], low[:44], close[:44], {}, 5)
    second, _ = adx_segment(high[44:], low[44:], close[44:], state, 5)
    np.testing.assert_allclose(np.r_[first, second], expected, equal_nan=True)

def test_registered_alias_is_recorded_without_invented_fallback():
    ok, detail = mc.check_prefix_invariance_all_causal_ts(canonicals=["m_avg"])
    assert ok, detail
    row = detail["per_canonical"]["m_avg"]
    assert row["runtime_resolved"] == "ts_mean"
    assert row["test_executed"] and not row["fallback"]


def test_real_stream_gate_detects_wrong_registered_full_run(monkeypatch):
    original = OperatorRegistry.get
    class BadEMA:
        metadata = SimpleNamespace(param_names=["x", "span"], param_specs={})
        calls = 0
        def calculate(self, frame, span=20):
            self.calls += 1
            return frame.ewm(span=span, adjust=False).mean() + 100
    bad = BadEMA()
    def get(cls, name, backend="pandas_numpy", *, mode="production"):
        return bad if name == "ts_ema" else original(name, backend=backend, mode=mode)
    monkeypatch.setattr(OperatorRegistry, "get", classmethod(get))
    ok, detail = mc.check_chunk_boundary_invariance_all_streamable(n_bars=80, n_random_chunkings=1)
    # MACD composites also resolve EMA; the bad helper can be invoked repeatedly.
    assert not ok and bad.calls >= 1
    assert detail["per_canonical"]["ts_ema"]["status"] == "FAILED"
    assert mc.reference_self_check_stream_examples()[0]
