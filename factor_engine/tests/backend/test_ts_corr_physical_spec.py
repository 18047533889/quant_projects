from __future__ import annotations

import polars as pl

from factor_engine.backend.operator_capability import enumerate_physical_inventory
from factor_engine.cleaned_operators.common.polars_ts_rolling import TSCorrNative


def test_ts_corr_native_has_complete_physical_spec_and_exact_id_admission() -> None:
    operator = TSCorrNative()
    result = operator._calculate_series(
        pl.DataFrame({"asset": [1.0, 2.0, 4.0]}),
        pl.DataFrame({"asset": [2.0, 5.0, 9.0]}),
        window=3,
        min_periods=2,
    )
    assert result["asset"].to_list()[-1] is not None

    expected_id = operator._physical_spec.physical_implementation_id
    assert expected_id is not None

    class Registry:
        _catalog = {"ts_corr": {}}

        @staticmethod
        def list_canonical():
            return ["ts_corr"]

        @staticmethod
        def backends_for(canonical):
            return ["polars"]

        @staticmethod
        def get(canonical, backend, mode="production"):
            return operator

    admitted = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: implementation_id
        == expected_id,
    )[0]
    assert admitted.spec is operator._physical_spec
    assert admitted.implementation_id == expected_id
    assert admitted.admission.admitted

    rejected = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: False,
    )[0]
    assert not rejected.admission.admitted
    assert not rejected.admission.evidence_production_safe
    assert "physical ID lacks matching production evidence" in rejected.admission.reasons
