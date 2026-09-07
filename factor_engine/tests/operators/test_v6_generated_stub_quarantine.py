import importlib

import pytest


def test_legacy_stub_import_cannot_register_or_replace_real_operators(monkeypatch):
    from factor_engine.cleaned_operators import base

    def forbid_registration(*args, **kwargs):
        raise AssertionError("legacy stubs attempted real registry registration")

    monkeypatch.setattr(base, "register_operator", forbid_registration)
    module = importlib.import_module("factor_engine.cleaned_operators.auto_polars_all")
    module = importlib.reload(module)
    assert len(module.QUARANTINED_DECLARATIONS) == 918
    for cls, declaration in module.QUARANTINED_DECLARATIONS:
        assert declaration["backend"] == "polars"
        assert cls.status == "unsupported"
        with pytest.raises(NotImplementedError, match="no verified implementation"):
            cls()._calculate_series(None)
