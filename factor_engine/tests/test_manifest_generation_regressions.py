from types import SimpleNamespace

from factor_engine.cleaned_operators.base import MISSING, ParamSpec
from factor_engine.scripts.generate_layer_manifests import _json_clean
from factor_engine.scripts.generate_operators_catalog import _payload


def test_missing_default_is_not_explicit_none():
    assert _json_clean(ParamSpec(dtype=int))["default"] == {"declared": False}
    assert _json_clean(ParamSpec(dtype=int, default=None))["default"] is None
    assert _json_clean(MISSING) == {"declared": False}


def test_catalog_is_copied_once_per_manifest(monkeypatch):
    from factor_engine.cleaned_operators import operator_spec
    monkeypatch.setattr(operator_spec, "build_operator_spec", lambda name: None)
    calls = []
    def catalog():
        calls.append(1)
        return {name: {"backends": [], "aliases": []} for name in ("a", "b")}
    registry = SimpleNamespace(
        catalog=catalog, list_canonical=lambda: ["a", "b"],
        get=lambda *args, **kwargs: None,
    )
    result = _payload(registry, lambda *args, **kwargs: None,
                      lambda name: "internal", lambda names: {})
    assert result["canonical_count"] == 2
    assert calls == [1]
