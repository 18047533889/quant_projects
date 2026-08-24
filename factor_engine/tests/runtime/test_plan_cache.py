"""持久化 plan 子树缓存与 data_scope 隔离测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.cache import CacheManager, PersistentPlanCache
from factor_engine.storage.data_scope import compute_data_scope
from tests.helpers import InMemorySeriesSource


def _panel() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series([1.0, 2.0], index=idx)


def test_compute_data_scope_differs_by_date_bounds():
    full = InMemorySeriesSource(data={"close": _panel()})
    scoped = InMemorySeriesSource(data={"close": _panel()})
    scoped.start_date = "2024-01-03"  # type: ignore[attr-defined]
    assert compute_data_scope(full) != compute_data_scope(scoped)


def test_cache_manager_scoped_keys_isolate_entries():
    cache = CacheManager()
    cache_a = cache.with_scope("scope_a")
    cache_b = cache.with_scope("scope_b")
    cache_a.set("plan_x", "value_a")
    cache_b.set("plan_x", "value_b")
    assert cache_a.get("plan_x") == "value_a"
    assert cache_b.get("plan_x") == "value_b"


def test_persistent_plan_cache_roundtrip_disk(tmp_path):
    root = tmp_path / "plan_cache"
    cache = PersistentPlanCache(root, data_scope="s1")
    series = _panel()
    cache.set("subtree_key", series)

    reloaded = PersistentPlanCache(root, data_scope="s1")
    hit = reloaded.get("subtree_key")
    assert hit is not None
    pd.testing.assert_series_equal(hit, series, check_names=False)


def test_persistent_plan_cache_rejects_metadata_for_different_key(tmp_path):
    import json

    root = tmp_path / "plan_cache_key_binding"
    cache = PersistentPlanCache(root, data_scope="s1")
    cache.set("key_a", _panel())

    path = cache._disk_path(cache._scoped_key("key_a"))
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cache_key_digest"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert PersistentPlanCache(root, data_scope="s1").get("key_a") is None


def test_persistent_plan_cache_roundtrip_dataframe(tmp_path):
    root = tmp_path / "plan_cache"
    cache = PersistentPlanCache(root, data_scope="s1")
    panel = _panel().unstack(level="instrument")
    cache.set("panel_key", panel)
    hit = PersistentPlanCache(root, data_scope="s1").get("panel_key")
    assert isinstance(hit, pd.DataFrame)
    pd.testing.assert_frame_equal(hit, panel)


def test_persistent_plan_cache_survives_engine_restart(tmp_path):
    data = {"close": _panel()}
    factor = Factor(name="c", expr=col("close"))
    root = tmp_path / "plan_cache"
    scope = "fixed_scope"

    eng1 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=data),
        cache=PersistentPlanCache(root, data_scope=scope),
    )
    r1 = eng1.run(factor)["result"]

    eng2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=data),
        cache=PersistentPlanCache(root, data_scope=scope),
    )
    r2 = eng2.run(factor)["result"]
    pd.testing.assert_series_equal(r1, r2, check_names=False)
    assert any(root.rglob("*.parquet"))


def test_incremental_persistent_cache_uses_new_scope_not_stale_memory(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=6)
    idx = pd.MultiIndex.from_product([dates, ["X"]], names=["timestamp", "instrument"])
    close_full = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], index=idx)
    close_window = close_full.copy()
    close_window.loc[(dates[-1], "X")] = 999.0

    factor = Factor(name="last_close", expr=col("close"))
    root = tmp_path / "plan_cache"
    cache = PersistentPlanCache(root, data_scope=compute_data_scope(InMemorySeriesSource(data={"close": close_full})))

    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_full}),
        cache=cache,
    )
    assert engine.run(factor)["result"].loc[(dates[-1], "X")] == pytest.approx(100.0)

    engine2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_window}),
        cache=cache,
    )
    inc = engine2.run_incremental(
        factor,
        factor_id="close_v1",
        since=str(dates[-2].date()),
        lookback_extra=0,
        recompute_tail_bars=1,
    )
    assert inc["result"].loc[(dates[-1], "X")] == pytest.approx(999.0)
