import numpy as np
import pytest
from factor_preprocess.adapters.data_access import DataAccessAdapter


def bundle():
    return dict(values=np.ones((2, 1)), dates=np.array(["2026-01-01", "2026-01-02"]),
                assets=np.array(["000001.SZ"]), metadata=dict(
                    market="ashare", classification_version="v1", snapshot_ref="snap-1",
                    universe_ref="universe-1", knowledge_time="2026-01-01",
                    effective_time="2026-01-01", lineage={"sources": ["table@1"]}))


class Provider:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        if name.startswith("get_"):
            return lambda *args, **kwargs: self.value
        raise AttributeError(name)


def fetch(value, kind="industry"):
    return DataAccessAdapter(Provider(value)).fetch_multi_exposure(
        "CN", [kind], "2026-01-01", "2026-01-02", ["000001.SZ"])[kind]


@pytest.mark.parametrize("kind", ["industry", "size", "sector", "beta", "custom"])
def test_all_exposure_routes_bind_request(kind):
    value = bundle()
    np.testing.assert_equal(fetch(value, kind)["values"], value["values"])
    value["metadata"]["market"] = "US"
    with pytest.raises(ValueError, match="market"):
        fetch(value, kind)


@pytest.mark.parametrize("key,value", [("market", "us"),
    ("classification_version", None), ("knowledge_time", "2035-01-01"),
    ("effective_time", "2035-01-01"), ("snapshot_ref", "")])
def test_reject_invalid_provenance(key, value):
    original = bundle()
    original["metadata"][key] = value
    with pytest.raises(ValueError):
        fetch(original)


@pytest.mark.parametrize("key,value", [("assets", ["WRONG"]),
    ("dates", ["2020-01-01", "2020-01-02"]),
    ("dates", ["2026-01-01", "2026-01-03"]),
    ("dates", ["2026-01-02", "2026-01-01"])])
def test_reject_mismatched_axes(key, value):
    original = bundle()
    original[key] = np.array(value)
    with pytest.raises(ValueError):
        fetch(original)


def test_nested_metadata_is_detached_and_frozen():
    original = bundle()
    out = fetch(original)
    original["metadata"]["lineage"]["sources"][0] = "changed"
    original["metadata"]["snapshot_ref"] = "changed"
    assert out["metadata"]["snapshot_ref"] == "snap-1"
    assert out["metadata"]["lineage"]["sources"] == ("table@1",)
    with pytest.raises(TypeError):
        out["metadata"]["lineage"]["sources"] = ("changed",)


def test_classification_request_not_silently_substituted():
    adapter = DataAccessAdapter(Provider(bundle()))
    with pytest.raises(ValueError, match="classification"):
        adapter.fetch_industry_exposure("CN", industry_classification="GICS")


def test_reordered_assets_are_not_relabelled_by_position():
    original = bundle()
    original["assets"] = np.array(["B", "A"])
    original["values"] = np.array([[1., 2.], [3., 4.]])
    adapter = DataAccessAdapter(Provider(original))
    with pytest.raises(ValueError, match="assets/order"):
        adapter.fetch_size_exposure("CN", assets=["A", "B"])


def test_arrays_cannot_reenable_writes_and_revalidation_preserves_snapshot():
    from factor_preprocess.adapters.data_access import _validate_exposure_bundle
    out = fetch(bundle())
    for key in ("values", "dates", "assets"):
        with pytest.raises(ValueError):
            out[key].flags.writeable = True
    repeated = _validate_exposure_bundle(out, "industry")
    assert repeated["metadata"]["snapshot_ref"] == "snap-1"


def test_provider_cannot_modify_request_asset_identity():
    assets = ["000001.SZ"]
    class MutatingProvider:
        def get_industry_exposure(self, **kwargs):
            kwargs["assets"][0] = "WRONG"
            return bundle()
    with pytest.raises(TypeError):
        DataAccessAdapter(MutatingProvider()).fetch_industry_exposure("CN", assets=assets)
    assert assets == ["000001.SZ"]
