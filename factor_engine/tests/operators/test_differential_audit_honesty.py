import numpy as np
import pytest

from factor_engine.scripts import audit_operator_differential_execution as audit
from factor_engine.backend import polars_backend_kind as kinds


class Identity:
    def calculate(self, x, **kwargs):
        return x


class Broken:
    def calculate(self, x, **kwargs):
        raise ValueError("kernel failed")


@pytest.mark.parametrize("kind,polars,pandas,expected", [
    (kinds.PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE, Identity(), Identity(), "supported_via_reference_fallback"),
    (kinds.PolarsImplementationKind.POLARS_NATIVE, Broken(), Identity(), "not_certified"),
    (kinds.PolarsImplementationKind.UNSUPPORTED, None, Broken(), "not_certified"),
    (kinds.PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE, Broken(), Identity(), "not_certified"),
    (kinds.PolarsImplementationKind.UNSUPPORTED, Identity(), Identity(), "not_certified"),
])
def test_certification_requires_successful_execution(monkeypatch, kind, polars, pandas, expected):
    monkeypatch.setattr(kinds, "canonical_polars_kind", lambda canonical: kind)
    registry = type("Registry", (), {"get": staticmethod(lambda c, b: polars if b == "polars" else pandas)})
    result = audit.audit_one("ts_mean", registry, {"nonconstant": {"x": np.array([1., 3., np.nan, 7.])}})
    assert result["certification"] == expected
    if kind != kinds.PolarsImplementationKind.POLARS_NATIVE:
        assert not result["native_used"]


def test_empty_fixtures_cannot_certify(monkeypatch):
    monkeypatch.setattr(kinds, "canonical_polars_kind", lambda c: kinds.PolarsImplementationKind.POLARS_NATIVE)
    registry = type("Registry", (), {"get": staticmethod(lambda c, b: Identity())})
    assert audit.audit_one("ts_mean", registry, {})["certification"] == "not_certified"


def test_reference_mismatch_blocks_native_certification(monkeypatch):
    monkeypatch.setattr(kinds, "canonical_polars_kind", lambda c: kinds.PolarsImplementationKind.POLARS_NATIVE)
    registry = type("Registry", (), {"get": staticmethod(lambda c, b: Identity())})
    result = audit.audit_one("rank", registry, {"nonconstant": {"x": np.array([[3., 1., 2.]])}})
    assert result["native_parity_all_pass"]
    assert not result["reference_all_pass"]
    assert result["certification"] == "not_certified"
