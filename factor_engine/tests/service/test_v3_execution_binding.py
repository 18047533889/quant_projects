from copy import deepcopy

import pytest

from factor_engine.service import app as service
from factor_engine.service.errors import ServiceError
from factor_engine.service.jobstore import JobRecord
from factor_engine.service.security import ANONYMOUS_PRINCIPAL
from factor_engine.runtime.endpoint_policy import EndpointExecutionPolicy


def build(**overrides):
    payload = dict(formula="close + 2", name="bound_factor", market="ashare",
                   calendar="SSE", freq="1d", data_source={"type": "test"})
    payload.update(overrides)
    execution, digest = service._validate_and_build_request(
        payload, endpoint_policy=EndpointExecutionPolicy.RESEARCH,
        principal=ANONYMOUS_PRINCIPAL)
    return execution, JobRecord(run_id="v3-bound-test", request_digest=digest)


def test_default_endpoint_backend_is_executable():
    execution, _ = build()
    assert execution["backend"] == "pandas"


@pytest.mark.parametrize("backend", ["auto", "polars", "duckdb"])
def test_unsupported_route_rejected_before_admission(backend):
    with pytest.raises(ServiceError):
        build(backend=backend)


@pytest.mark.parametrize("market,calendar", [("ashare", "SSE"), ("us", "NYSE")])
def test_actual_inline_factor_scope_and_values(monkeypatch, market, calendar):
    import numpy as np
    import pandas as pd
    from factor_engine.storage.datasource import DataSource
    from factor_engine.storage import factory
    from factor_engine.runtime.engine import FactorEngine

    class Source(DataSource):
        def load_column(self, name):
            assert name == "close"
            return pd.Series(np.arange(6, dtype=float), index=pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["timestamp", "instrument"]))

    source = Source()
    contexts, scopes = [], []
    def source_factory(config, *, build_context):
        contexts.append(build_context)
        return source
    original = FactorEngine.run
    def run(self, factor, **kwargs):
        scopes.append(factor.semantic_identity)
        assert factor.semantic_identity is not None
        assert factor.semantic_identity.market == market
        return original(self, factor, **kwargs)
    monkeypatch.setattr(factory, "build_data_source", source_factory)
    monkeypatch.setattr(FactorEngine, "run", run)
    execution, job = build(market=market, calendar=calendar)
    actual = service._execute_inline(job, execution)["result"]
    pd.testing.assert_series_equal(actual, source.load_column("close") + 2, check_names=False)
    assert contexts[0].market == scopes[0].market == market
    assert contexts[0].calendar_id == scopes[0].calendar_id == calendar


@pytest.mark.parametrize("target", ["backend", "build_context", "data_source", "validated"])
def test_mutated_execution_rejected_before_read(monkeypatch, target):
    from factor_engine.storage import factory
    execution, job = build()
    execution = deepcopy(execution)
    if target == "backend":
        execution[target] = "auto"
    elif target == "data_source":
        execution[target]["path"] = "/foreign"
    else:
        execution[target]["market"] = "us"
    def forbidden(*args, **kwargs):
        pytest.fail("source accessed before binding validation")
    monkeypatch.setattr(factory, "build_data_source", forbidden)
    with pytest.raises(ServiceError):
        service._execute_inline(job, execution)


def test_validation_rejects_source_conflict_and_missing_production_market():
    with pytest.raises(ServiceError, match="conflicts"):
        build(data_source={"type": "test", "market": "us"})
    assert not service.validate_spec({"formula": "close", "run_mode": "production"})["ok"]
    assert service.validate_spec({"formula": "close", "backend": "pandas"})["ok"]
    assert not service.validate_spec({"formula": "close", "backend": "auto"})["ok"]


@pytest.mark.parametrize("market,calendar", [("ashare", "SSE"), ("us", "NYSE")])
def test_http_validate_then_actual_compute(monkeypatch, tmp_path, market, calendar):
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient
    from factor_engine.storage.datasource import DataSource
    from factor_engine.storage import factory
    from factor_engine.service.security import reset_principal_registry

    class Source(DataSource):
        def load_column(self, name):
            return pd.Series(np.arange(6, dtype=float), index=pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["timestamp", "instrument"]))
    seen = []
    def source_factory(config, *, build_context):
        seen.append(build_context)
        return Source()
    monkeypatch.setattr(factory, "build_data_source", source_factory)
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_API_KEY", "v3-test")
    monkeypatch.delenv("FACTOR_ENGINE_SERVICE_API_KEY_MAPPING", raising=False)
    reset_principal_registry()
    monkeypatch.setattr(service, "STORE", service.JobStore(tmp_path))
    client = TestClient(service.create_app())
    headers = {"X-API-Key": "v3-test"}
    payload = dict(formula="close + 2", market=market, calendar=calendar,
                   backend="pandas", data_source={"type": "test"}, sync=True)
    check = client.post("/factor-engine/validate-spec", headers=headers, json=payload)
    assert check.status_code == 200 and check.json()["ok"]
    response = client.post("/factor-engine/research/compute", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    job = service.STORE.get(response.json()["run_id"])
    assert job.status == "succeeded", job.error
    assert job.result_summary["rows"] == 6
    assert seen[0].market == market and seen[0].calendar_id == calendar
    caps = client.get("/factor-engine/capabilities", headers=headers).json()
    assert caps["inline_compute"]["default_backend"] == "pandas"
    assert caps["production_fast"]["supported"] is False
    assert caps["operator_counts"]["canonical"] > 1700
    described = client.get("/factor-engine/operator-descriptions?limit=2", headers=headers)
    assert described.status_code == 200, described.text
    assert len(described.json()["items"]) == 2
    assert described.json()["catalog_digest"] == caps["catalog_digest"]
    assert client.get("/factor-engine/operator-descriptions?limit=101", headers=headers).status_code == 422
    assert client.get("/factor-engine/operator-descriptions?limit=1").status_code == 401
    rejected = client.post("/factor-engine/research/compute", headers=headers,
                           json={**payload, "backend": "auto"})
    assert rejected.status_code == 422
