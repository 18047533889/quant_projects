import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.intraday import sufficient_stats as ss
from factor_engine.cleaned_operators.intraday import sufficient_stats_ops as ops
from factor_engine.cleaned_operators.intraday._core import daily_agg_two
from factor_engine.cleaned_operators.intraday.realized_beta import (
    _aligned_market,
    _beta_daily,
    _market_r2,
    _market_return_ex_self,
    _realized_beta,
)


def _panel(values, *, start="2025-01-02 09:30", columns=("A",)):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return pd.DataFrame(values, index=pd.date_range(start, periods=len(values), freq="min"), columns=columns)


def test_h27_full_identity_and_cached_result_are_protected(monkeypatch):
    from factor_engine.runtime import resource_broker
    class Lease:
        def release(self):
            pass
    class Broker:
        def acquire_memory(self, *_args, **_kwargs):
            return Lease()
    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: Broker())
    frame = _panel([0, 1, 2, 3, 4])
    first = ss.compute_sufficient_statistics(frame)
    frame.iloc[2, 0] += 0.25
    changed = ss.compute_sufficient_statistics(frame)
    assert first is not changed
    assert changed["mean"][0, 0] == pytest.approx(2.05)
    frame.iloc[0, 0] = np.nan
    masked = ss.compute_sufficient_statistics(frame)
    assert masked["mean"][0, 0] == pytest.approx(2.5625)
    shifted = frame.copy()
    shifted.index += pd.Timedelta(days=1)
    assert ss._make_key(shifted) != ss._make_key(frame)
    assert not first["g"].flags.writeable
    with pytest.raises((TypeError, ValueError)):
        first["g"][0, 0, 0] = 99
    with pytest.raises(ValueError):
        first["g"].setflags(write=True)
    with pytest.raises(TypeError):
        first["mean"] = np.array([[99.0]])


def test_h28_inf_and_empty_have_same_finite_policy_and_schema():
    frame = _panel([1, 2, np.inf, 4])
    bundle = ss.compute_sufficient_statistics(frame)
    assert bundle["sum"][0, 0] == 7.0
    empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A"], dtype=float)
    empty_bundle = ss.compute_sufficient_statistics(empty)
    assert empty_bundle["sum"].shape == (0, 1)
    assert empty_bundle["uniq_days"].empty


def test_h29_session_grid_allocation_is_linear_in_days():
    sizes = []
    for days in (4, 8, 16):
        idx = pd.DatetimeIndex([
            pd.Timestamp("2025-01-02") + pd.Timedelta(days=d, minutes=570 + slot)
            for d in range(days) for slot in range(3)
        ])
        frame = pd.DataFrame(np.arange(len(idx) * 2).reshape(len(idx), 2), index=idx, columns=["A", "B"])
        grid = ss._grid3(frame)[0]
        assert grid.shape == (days, 3, 2)
        sizes.append(grid.nbytes)
    assert sizes == [4 * 3 * 2 * 8, 8 * 3 * 2 * 8, 16 * 3 * 2 * 8]


def test_h29_mean_is_on_demand_and_cache_residency_is_broker_leased(monkeypatch):
    from factor_engine.runtime import resource_broker

    class Lease:
        def __init__(self):
            self.released = False
        def release(self):
            self.released = True

    class Broker:
        def __init__(self):
            self.requests = []
        def acquire_memory(self, kind, nbytes, *, lease_id=""):
            lease = Lease()
            self.requests.append((kind, nbytes, lease_id, lease))
            return lease

    broker = Broker()
    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: broker)
    ss._STORE._data.clear()
    ss._STORE._bytes = 0
    frame = _panel([1, 2, 3, 4, 5])
    bundle = ss.compute_sufficient_statistics(frame, required="mean")
    assert set(bundle) == {"g", "fin", "cnt", "sum", "mean", "uniq_days", "columns", "active"}
    assert "quart" not in bundle and "ret" not in bundle and "packed" not in bundle
    assert len(broker.requests) == 1
    assert broker.requests[0][1] == ss._STORE._bytes
    assert ss.compute_sufficient_statistics(frame, required="mean") is bundle
    for lease in ss._STORE._leases.values():
        lease.release()
    ss._STORE._data.clear()
    ss._STORE._leases.clear()
    ss._STORE._sizes.clear()
    ss._STORE._bytes = 0


def test_h29_eviction_keeps_lease_until_held_bundle_is_released(monkeypatch):
    import gc
    from factor_engine.runtime import resource_broker

    class Lease:
        def __init__(self):
            self.released = False
        def release(self):
            self.released = True

    class Broker:
        def __init__(self):
            self.leases = []
        def acquire_memory(self, *_args, **_kwargs):
            lease = Lease()
            self.leases.append(lease)
            return lease

    broker = Broker()
    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: broker)
    ss._STORE._data.clear()
    ss._STORE._sizes.clear()
    ss._STORE._leases.clear()
    ss._STORE._bytes = 0
    monkeypatch.setattr(ss._STORE, "_MAX_BYTES", 100)
    held = ss.compute_sufficient_statistics(_panel([1, 2, 3, 4, 5]), required="mean")
    first_lease = broker.leases[0]
    ss.compute_sufficient_statistics(
        _panel([6, 7, 8, 9, 10], start="2025-01-03 09:30"), required="mean"
    )
    assert all(value is not held for value in ss._STORE._data.values())
    assert not first_lease.released
    del held
    gc.collect()
    assert first_lease.released


def test_h29_array_views_keep_physical_lease_after_bundle_eviction(monkeypatch):
    import gc
    from factor_engine.runtime import resource_broker

    class Lease:
        def __init__(self): self.released = False
        def release(self): self.released = True
    class Broker:
        def __init__(self): self.leases = []
        def acquire_memory(self, *_args, **_kwargs):
            lease = Lease(); self.leases.append(lease); return lease

    broker = Broker()
    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: broker)
    ss._STORE._data.clear(); ss._STORE._sizes.clear(); ss._STORE._leases.clear(); ss._STORE._bytes = 0
    monkeypatch.setattr(ss._STORE, "_MAX_BYTES", 100)
    bundle = ss.compute_sufficient_statistics(_panel([1, 2, 3, 4, 5]), required="mean")
    held_array = np.asarray(bundle["sum"]).view()
    lease = broker.leases[0]
    del bundle
    ss.compute_sufficient_statistics(_panel([6, 7, 8, 9, 10], start="2025-01-03 09:30"), required="mean")
    gc.collect()
    assert not lease.released
    assert held_array[0, 0] == 15.0
    del held_array
    gc.collect()
    assert lease.released


def test_h29_denied_cache_does_not_copy_full_bundle(monkeypatch):
    from factor_engine.runtime import resource_broker

    class Broker:
        def acquire_memory(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: Broker())
    frame = _panel([1, 2, 3, 4, 5])
    grid = ss._grid3(frame)
    returned = ss.compute_sufficient_statistics(frame, grid=grid, required="mean")
    assert returned["g"].base is grid[0]
    assert not returned["g"].flags.writeable


def test_h29_copy_failure_releases_admitted_lease(monkeypatch):
    from factor_engine.runtime import resource_broker

    class Lease:
        def __init__(self): self.released = False
        def release(self): self.released = True
    lease = Lease()
    class Broker:
        def acquire_memory(self, *_args, **_kwargs): return lease

    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: Broker())
    monkeypatch.setattr(np, "ascontiguousarray", lambda _value: (_ for _ in ()).throw(MemoryError()))
    with pytest.raises(MemoryError):
        ss.compute_sufficient_statistics(_panel([1, 2, 3, 4, 5]), required="mean")
    assert lease.released


def test_h29_index_numpy_view_keeps_physical_lease(monkeypatch):
    import gc
    from factor_engine.runtime import resource_broker

    class Lease:
        def __init__(self): self.released = False
        def release(self): self.released = True
    class Broker:
        def __init__(self): self.leases = []
        def acquire_memory(self, *_args, **_kwargs):
            lease = Lease(); self.leases.append(lease); return lease

    broker = Broker()
    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: broker)
    ss._STORE._data.clear(); ss._STORE._sizes.clear(); ss._STORE._leases.clear(); ss._STORE._bytes = 0
    monkeypatch.setattr(ss._STORE, "_MAX_BYTES", 100)
    bundle = ss.compute_sufficient_statistics(_panel([1, 2, 3, 4, 5]), required="mean")
    held_days = bundle["uniq_days"].to_numpy(copy=False)
    lease = broker.leases[0]
    del bundle
    ss.compute_sufficient_statistics(_panel([6, 7, 8, 9, 10], start="2025-01-03 09:30"), required="mean")
    gc.collect()
    assert not lease.released
    del held_days
    gc.collect()
    assert lease.released


def test_h29_cache_bookkeeping_is_consistent_under_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from factor_engine.runtime import resource_broker

    class Lease:
        def release(self):
            pass
    class Broker:
        def acquire_memory(self, *_args, **_kwargs):
            return Lease()

    monkeypatch.setattr(resource_broker, "peek_v2_resource_broker", lambda: Broker())
    ss._STORE._data.clear()
    ss._STORE._sizes.clear()
    ss._STORE._leases.clear()
    ss._STORE._bytes = 0
    monkeypatch.setattr(ss._STORE, "_MAX_BYTES", 512)
    panels = [_panel(np.arange(5.0) + i, start=f"2025-01-{i + 2:02d} 09:30") for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda frame: ss.compute_sufficient_statistics(frame, required="mean"), panels))
    assert set(ss._STORE._data) == set(ss._STORE._sizes) == set(ss._STORE._leases)
    assert ss._STORE._bytes == sum(ss._STORE._sizes.values()) <= ss._STORE._MAX_BYTES


def test_h30_missing_price_breaks_physical_adjacent_returns():
    close = _panel([100, 101, np.nan, 150, 153])
    volume = _panel(np.ones(5))
    expected = (np.log(101 / 100) + np.log(153 / 150)) / 2
    got = daily_agg_two(close, volume, ops._ts_volume_weighted_return, min_finite=2)
    assert got.iloc[0, 0] == pytest.approx(expected)


def test_h32_r2_uses_the_exact_paired_sample():
    r = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    m = np.arange(1.0, 6.0)
    assert _market_r2(r, m) == pytest.approx(1.0)
    r[2] = np.inf
    assert _market_r2(r, m) == pytest.approx(1.0)


def test_h33_finite_positive_prices_ex_self_and_full_day_axis():
    close = _panel([[100, 100], [101, 102], [np.inf, 104], [103, 106]], columns=("A", "B"))
    weights = pd.DataFrame([[np.nan, 10.0]], index=[close.index[0].normalize()], columns=close.columns)
    rets, market, w_bc, w_ret = _aligned_market(close, weights)
    assert rets["A"].iloc[2:].isna().all()
    ex_a = _market_return_ex_self(rets, w_bc, w_ret, "A")
    pd.testing.assert_series_equal(ex_a, rets["B"], check_names=False)

    no_weights = pd.DataFrame([[np.nan, np.nan]], index=weights.index, columns=close.columns)
    out = _beta_daily(close, no_weights, _realized_beta)
    assert out.index.equals(pd.DatetimeIndex([close.index[0].normalize()]))
    assert out.shape == (1, 2)
    assert out.isna().all().all()
