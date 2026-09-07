import json
import pickle
from types import SimpleNamespace

import pytest

from factor_engine import get_engine
from factor_engine.runtime.default_engine import (
    ApprovedDeploymentProfile, DeploymentConfigurationError, load_deployment_profile,
)


def profile(tmp_path):
    return {"profile_id": "isolated-test", "approval_id": "test-only",
            "data_source": {"type": "data_access", "dataset": "ashare_stock_daily_adj",
                            "start_date": "2024-01-01", "end_date": "2024-01-31",
                            "instrument_filter": ["000001.SZ"]},
            "artifact_root": str(tmp_path), "market": "ashare", "calendar_id": "SSE",
            "timezone": "Asia/Shanghai", "frequency": "1d", "universe_id": "fixture",
            "adjustment": "hfq"}


def test_unconfigured_public_entry_reports_before_importing_or_computing(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_V2_PROFILE", raising=False)
    with pytest.raises(DeploymentConfigurationError) as error:
        get_engine()
    assert error.value.missing_fields == ("FACTOR_ENGINE_V2_PROFILE",)


def test_reports_missing_business_fields_together():
    with pytest.raises(DeploymentConfigurationError) as error:
        ApprovedDeploymentProfile.from_mapping({"data_source": {"type": "data_access"}})
    assert "artifact_root" in error.value.missing_fields
    assert "data_source.start_date" in error.value.missing_fields
    assert "frequency" in error.value.missing_fields


def test_immutable_deployment_snapshot(tmp_path):
    original = profile(tmp_path)
    frozen = ApprovedDeploymentProfile.from_mapping(original)
    original["data_source"]["dataset"] = "changed"
    assert frozen.to_dict()["data_source"]["dataset"] == "ashare_stock_daily_adj"
    assert frozen.digest == ApprovedDeploymentProfile.from_mapping(frozen.to_dict()).digest


@pytest.mark.parametrize("key,value", [
    ("artifact_root", "relative"), ("adjustment", "raw"), ("market", "us"),
    ("frequency", True), ("unexpected", "value"),
])
def test_bad_business_configuration_rejected(tmp_path, key, value):
    data = profile(tmp_path)
    data[key] = value
    with pytest.raises(DeploymentConfigurationError):
        ApprovedDeploymentProfile.from_mapping(data)


@pytest.mark.parametrize("key,value", [
    ("type", "parquet"), ("run_mode", "research"), ("production", False),
    ("pit_enforce", False), ("instrument_filter", ["A", "A"]),
    ("instrument_filter", "all"), ("start_date", "2025-01-01"),
])
def test_raw_source_or_scope_downgrade_rejected(tmp_path, key, value):
    data = profile(tmp_path)
    data["data_source"][key] = value
    with pytest.raises(DeploymentConfigurationError):
        ApprovedDeploymentProfile.from_mapping(data)


def test_file_parser_and_policy_agree(tmp_path):
    file = tmp_path / "profile.json"
    file.write_text(json.dumps(profile(tmp_path)))
    assert load_deployment_profile(file).to_dict() == profile(tmp_path)


def test_duplicate_keys_rejected(tmp_path):
    file = tmp_path / "duplicate.yaml"
    file.write_text("market: ashare\nmarket: us\n")
    with pytest.raises(DeploymentConfigurationError, match="duplicate"):
        load_deployment_profile(file)


def test_parser_error_does_not_echo_profile_contents(tmp_path):
    file = tmp_path / "malformed.yaml"
    file.write_text("secret_token: [secret-do-not-echo\n")
    with pytest.raises(DeploymentConfigurationError) as error:
        load_deployment_profile(file)
    assert "invalid YAML/JSON" in str(error.value)
    assert "secret-do-not-echo" not in str(error.value)


@pytest.mark.parametrize("token", ["", "  ", False, 123, []])
def test_snapshot_identity_must_be_nonempty_string(tmp_path, token):
    data = profile(tmp_path)
    data["expected_snapshot_token"] = token
    with pytest.raises(DeploymentConfigurationError, match="expected_snapshot_token"):
        ApprovedDeploymentProfile.from_mapping(data)


@pytest.mark.parametrize("source_change", [
    {"dataset": "raw_unadjusted_fixture"},
    {"fields": {"close": "Close"}},
    {"read_mode": "event"},
])
def test_hfq_label_cannot_override_actual_source_semantics(tmp_path, source_change):
    data = profile(tmp_path)
    data["data_source"].update(source_change)
    with pytest.raises(DeploymentConfigurationError):
        ApprovedDeploymentProfile.from_mapping(data)


def test_public_factory_constructs_production_core_and_shared_authority(tmp_path):
    file = tmp_path / "profile.json"
    file.write_text(json.dumps(profile(tmp_path)))
    with get_engine(profile_path=file) as first, get_engine(profile_path=file) as second:
        assert first._engine.run_mode == "production"
        assert type(first._engine.backend).__name__ == "HybridBackend"
        assert first._engine.data_source.production is True
        assert first._engine.data_source.pit_enforce is True
        assert first._engine.data_source.dataset == "ashare_stock_daily_adj"
        assert first._engine.resource_broker is second._engine.resource_broker
        clone = first._engine.with_data_source(first._engine.data_source)
        assert clone.default_execution_policy is first.policy
        assert clone.resource_broker is first._engine.resource_broker


def test_explicit_backend_preserved_at_construction(tmp_path):
    file = tmp_path / "profile.json"
    file.write_text(json.dumps(profile(tmp_path)))
    with get_engine(profile_path=file, execution={"backend": "pandas"}) as engine:
        assert engine.policy.backend == "pandas"
        assert type(engine._engine.backend).__name__ == "PandasBackend"


def test_worker_factory_config_roundtrip_contains_no_engine_or_locks(tmp_path):
    from factor_engine.runtime.default_engine import (
        ExecutionCoreWorkerConfig, build_execution_core_from_worker_config,
    )
    from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy

    config = ExecutionCoreWorkerConfig(ApprovedDeploymentProfile.from_mapping(profile(tmp_path)),
                                       DefaultExecutionPolicy())
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    engine = build_execution_core_from_worker_config(restored)
    try:
        assert engine.run_mode == "production"
        assert engine.default_execution_policy == config.policy
        assert engine.data_source.dataset == "ashare_stock_daily_adj"
    finally:
        engine.data_source.close()


def test_source_validation_failure_cannot_claim_policy_singleton(tmp_path, monkeypatch):
    import factor_engine.runtime.default_engine as module
    import factor_engine.runtime.resource_broker as broker_module

    file = tmp_path / "profile.json"
    file.write_text(json.dumps(profile(tmp_path)))
    acquisitions = []

    def acquire(policy):
        acquisitions.append(policy)
        raise AssertionError("invalid construction acquired the policy singleton")

    def bad_contract(source):
        raise DeploymentConfigurationError(detail="incompatible HFQ contract")

    monkeypatch.setattr(broker_module, "get_v2_resource_broker", acquire)
    monkeypatch.setattr(module, "_validate_hfq_source_contract", bad_contract)
    with pytest.raises(DeploymentConfigurationError, match="incompatible HFQ"):
        get_engine(profile_path=file)
    assert acquisitions == []
