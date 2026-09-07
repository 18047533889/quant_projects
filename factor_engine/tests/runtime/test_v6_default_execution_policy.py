import pytest

from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy, resolve_default_policy


def test_zero_flags_single_policy_and_stable_digest():
    policy = resolve_default_policy()
    assert policy.backend == "auto"
    assert policy.memory_fraction == .80
    assert policy.result_mode == "artifact"
    assert policy.automatic_production_publish is False
    assert policy.digest == resolve_default_policy(policy.to_dict()).digest
    assert resolve_default_policy({"connect_seconds": 5}).digest == policy.digest


def test_explicit_precedence_and_identity():
    policy = resolve_default_policy({"backend": "pandas"}, {"backend": "duckdb_sql", "memory_fraction": .5})
    assert policy.backend == "pandas"
    assert policy.memory_fraction == .5
    assert policy.digest != resolve_default_policy().digest


@pytest.mark.parametrize("overrides", [
    {"memory_fraction": True}, {"memory_fraction": float("nan")},
    {"memory_fraction": float("inf")}, {"memory_fraction": 0}, {"memory_fraction": 1.1},
    {"initial_lookahead_factors": True}, {"initial_lookahead_factors": 1.5},
    {"max_manifest_factors": "100000"}, {"optional_admin_cap_bytes": -1},
    {"backend": "autoo"}, {"backend": []}, {"result_mode": "return"},
    {"automatic_production_publish": True}, {"automatic_production_publish": 0},
    {"work_item_max_attempts": 4}, {"root_max_recovery_actions": 3},
    {"max_deterministic_retries": 1}, {"max_compute_timeout_retries": 1},
    {"max_transient_retries": 3}, {"max_oom_replans": 2},
    {"compute_min_seconds": 4000}, {"task_peak_uncertainty_multiplier": .5},
    {"arbitrary_new_flag": True}, {"resource_wait_seconds": None},
])
def test_invalid_policy_never_coerced_or_ignored(overrides):
    with pytest.raises(ValueError):
        resolve_default_policy(overrides)


def test_bad_profile_not_hidden_by_override():
    with pytest.raises(ValueError):
        resolve_default_policy({"backend": "pandas"}, {"backend": "typo"})


def test_policy_frozen():
    with pytest.raises(AttributeError):
        DefaultExecutionPolicy().memory_fraction = .9
