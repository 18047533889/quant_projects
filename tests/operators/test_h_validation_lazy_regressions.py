"""H12/H13/H23/H36 negative controls against real production helpers."""
import ast
import inspect
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import math_certificate as mc
from factor_engine.backend import polars_lazy as lazy


@pytest.mark.parametrize("budget", [0, 1])
def test_lazy_hit_only_reapplies_shrinking_budget(monkeypatch, budget):
    b, calls = _bundle(monkeypatch)
    expected = b.materialize_columns(["A"])["A"]
    assert b._cache_bytes() > budget
    b._materialized_budget = budget
    actual = b.materialize_columns(["A"])["A"]
    pd.testing.assert_series_equal(actual, expected)
    assert b._cache_bytes() <= budget
    assert len(calls) == 1


def test_validation_preserves_imaginary_components():
    a = pd.Series([1 + 2j, 3 + 4j])
    b = pd.Series([1 + 8j, 3 + 9j])
    assert not mc._validation_equal(a, b, rtol=1e-8, atol=1e-10)
    assert mc._validation_equal(a, a.copy(), rtol=1e-8, atol=1e-10)


def test_validation_rejects_metadata_drift_and_unbounded_attrs():
    x = pd.Series(np.arange(40.))
    def tagged(a, tag):
        out = a.copy()
        out.attrs["support"] = tag
        return out
    assert not mc.prefix_invariance_check(
        lambda a: tagged(a, "short" if len(a) <= 20 else "long"), x, T=20
    ).passed
    assert not mc.differential_against_reference(
        lambda a: tagged(a, "wrong"), lambda a: tagged(a, "oracle"), [x]
    )[0]
    assert mc.differential_against_reference(
        lambda a: tagged(a, {"count": np.arange(3)}),
        lambda a: tagged(a, {"count": np.arange(3)}), [x]
    )[0]
    for metadata in (object(), "x" * 65537):
        x.attrs["invalid"] = metadata
        with pytest.raises((TypeError, ValueError)):
            mc._validation_copy(x)
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError):
        mc._validation_metadata(cyclic)


def test_prefix_checks_second_column_and_singleton():
    x = np.arange(40.0)
    def bad(a):
        return np.column_stack([a, np.full(len(a), a[-1])])
    assert not mc.prefix_invariance_check(bad, x, T=20).passed
    assert mc.prefix_invariance_check(lambda a: a[:, None], x, T=20).passed
    panel = pd.DataFrame({"a": x, "b": x * 2})
    assert mc.prefix_invariance_check(lambda a: a, panel, T=20).passed
    assert not mc.prefix_invariance_check(lambda a: a.iloc[:, ::-1] if len(a) > 20 else a, panel, T=20).passed


def test_chunk_checks_all_columns_and_coverage():
    x = np.arange(40.0)
    assert mc.chunk_invariance_check(lambda a: a[:, None], x, window=3, boundaries=[20]).passed
    def bad(a):
        return np.column_stack([a, np.full(len(a), a[-1])])
    assert not mc.chunk_invariance_check(bad, x, window=3, boundaries=[20]).passed
    assert not mc.chunk_invariance_check(lambda a: a, x, window=3, boundaries=[]).passed
    panel = pd.DataFrame({"a": x, "b": x * 2})
    assert mc.chunk_invariance_check(lambda a: a, panel, window=3).passed


@pytest.mark.parametrize("view", [False, True])
def test_oracle_isolation_detects_mutation(view):
    fixture = np.arange(20.0).reshape(10, 2)
    original = fixture.copy()
    def bad(a):
        a[:, 1] += 100
        return a
    ref = (lambda a: a[:]) if view else (lambda a: a)
    assert not mc.differential_against_reference(bad, ref, [fixture])[0]
    np.testing.assert_array_equal(fixture, original)
    assert mc.differential_against_reference(lambda a: a.copy(), ref, [fixture])[0]


def test_no_empty_or_degenerate_certification_and_strict_metadata():
    identity = lambda a: a
    assert not mc.differential_against_reference(identity, identity, [])[0]
    assert not mc.differential_against_reference(identity, identity, [np.full(5, np.nan)])[0]
    assert not mc.differential_against_reference(lambda a: a.astype("float32"), identity, [np.arange(5.)])[0]
    panel = pd.DataFrame({"a": np.arange(5.), "b": np.arange(5.)})
    assert not mc.differential_against_reference(lambda a: a.rename(columns={"b": "c"}), identity, [panel])[0]


@pytest.mark.parametrize("canonical,method", [("ts_mean", "mean"), ("ts_sum", "sum"), ("ts_std", "std"), ("ts_var", "var")])
def test_moment_laws(canonical, method):
    x = np.arange(30., dtype=float).reshape(10, 3)
    x[2, 1] = np.nan
    fn = lambda a: getattr(pd.DataFrame(a).rolling(4, min_periods=2), method)().to_numpy()
    support = lambda a: pd.DataFrame(a).rolling(4, min_periods=2).count().to_numpy()
    checks = mc.check_moment_transformations(canonical, fn, x, support_fn=support)
    assert checks and all(c.passed for c in checks), checks
    broken = mc.check_moment_transformations(canonical, lambda a: fn(a) + 7, x, support_fn=support)
    assert not all(c.passed for c in broken)


def test_declarations_have_no_duplicate_keys():
    tree = ast.parse(inspect.getsource(mc))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "METAMORPHIC_PROPERTY_DECLARATIONS":
            keys = [k.value for k in node.value.keys]
            assert len(keys) == len(set(keys))
    assert "rank.column_permutation_equivariance" in mc.METAMORPHIC_PROPERTY_DECLARATIONS["neutralize"]


def _bundle(monkeypatch, budget=100000):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"time": [1, 2], "instrument": ["x", "x"], "A": [1., 2.], "B": [3., 4.]})
    calls = []
    def collect(lf, *, select_cols):
        calls.append(select_cols)
        return lf.select(select_cols).collect().to_arrow()
    monkeypatch.setattr(lazy, "_collect_arrow_table", collect)
    class Lease:
        released = False
        def release(self):
            self.released = True
    class Broker:
        def current_read_budget(self):
            return budget
        def acquire_memory(self, *args, **kwargs):
            return Lease()
    bundle = lazy.LazyColumnBundle(frame.lazy(), "time", "instrument", ("A", "B"), {}, False, None,
                                   snapshot_id="s1", _materialized_budget=budget,
                                   _cache_broker=Broker())
    return bundle, calls


def test_lazy_aliases_use_physical_cache(monkeypatch):
    b, calls = _bundle(monkeypatch)
    assert set(b.materialize_columns(["A"], output_names={"A": "B", "B": "C"})) == {"B"}
    assert set(b.materialize_columns(["B"], output_names={"A": "B", "B": "C"})) == {"C"}
    assert set(b.materialize_columns(["A"], output_names={"A": "price"})) == {"price"}
    assert len(calls) == 2
    assert set(b._materialized) == {"A", "B"}


def test_lazy_eviction_preserves_request_hits(monkeypatch):
    b, calls = _bundle(monkeypatch)
    b.materialize_columns(["A"])
    b._materialized_budget = b._cache_bytes()
    out = b.materialize_columns(["A", "B"])
    assert set(out) == {"A", "B"}
    assert len(b._materialized) <= 1
    np.testing.assert_array_equal(out["A"].to_numpy(), [1., 2.])


def test_lazy_context_and_mapping_validation(monkeypatch):
    b, calls = _bundle(monkeypatch)
    b.materialize_columns(["A"])
    b.snapshot_id = "s2"
    b.materialize_columns(["A"])
    assert len(calls) == 2
    with pytest.raises(ValueError, match="conflicting"):
        b.materialize_columns(["A", "B"], output_names={"A": "price", "B": "price"})


def test_lazy_unknown_and_explicit_zero_disable_residency(monkeypatch):
    b, calls = _bundle(monkeypatch, budget=0)
    b._cache_broker = None
    monkeypatch.setattr(
        "factor_engine.storage.sources.data_access_source._data_cache_authority",
        lambda: None,
    )
    assert set(b.materialize_columns(["A"])) == {"A"}
    assert not b._materialized
    assert len(calls) == 1


def test_lazy_default_discovers_shared_broker_and_explicit_cap_is_clamped(monkeypatch):
    b, _calls = _bundle(monkeypatch, budget=100)
    broker = b._cache_broker
    b._cache_broker = None
    b._materialized_budget = None
    monkeypatch.setattr(
        "factor_engine.storage.sources.data_access_source._data_cache_authority",
        lambda: broker,
    )
    assert b.materialized_budget == 100
    b._materialized_budget = 1000
    assert b.materialized_budget == 100


def test_lazy_cache_lease_survives_eviction_until_physical_owners_die(monkeypatch):
    import gc
    b, _calls = _bundle(monkeypatch)
    b.materialize_columns(["A"])
    token = b._cache_leases["A"]
    lease = b._live_cache_leases[token]["lease"]
    b.release_cached(["A"])
    assert not b._materialized
    assert not b._cache_leases
    gc.collect()
    assert lease.released
    assert not b._live_cache_leases


def test_lazy_cache_lease_survives_external_ndarray_view(monkeypatch):
    import gc
    b, _calls = _bundle(monkeypatch)
    out = b.materialize_columns(["A"])
    token = b._cache_leases["A"]
    lease = b._live_cache_leases[token]["lease"]
    view = out["A"].to_numpy(copy=False)
    b.release_cached(["A"])
    del out
    gc.collect()
    assert not lease.released
    del view
    gc.collect()
    assert lease.released

def test_unknown_and_zero_budget_still_refuse_source_cache(monkeypatch):
    from types import SimpleNamespace
    from factor_engine.storage.sources import data_access_source as source_module
    for authority in (None, SimpleNamespace(current_read_budget=lambda: 0)):
        monkeypatch.setattr(source_module, "_data_cache_authority", lambda: authority)
        source = source_module.DataAccessSource(dataset="synthetic")
        assert not source._put_cache(source._column_cache, "x", pd.Series([1.0]))
        assert not source._column_cache
        assert source._max_cache_bytes == 0
