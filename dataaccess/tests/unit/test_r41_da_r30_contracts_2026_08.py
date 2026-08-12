from __future__ import annotations

from dataclasses import replace

import pytest

from data_access.core.exceptions import AccessDeniedError
from data_access.r30.contracts import (
    BackendReadError,
    CapabilityHandshakeError,
    CapabilityManifest,
    EmptyResultError,
    ReadIdentity,
    SnapshotCompatibilityError,
    SnapshotFidelity,
    SnapshotFidelityError,
    import_all_public_modules,
    module_inventory,
)
from data_access.r30.experiment_snapshot import ExperimentDataSnapshot
from data_access.r30.training_read_plan import TrainingReadPlan


class _Store:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def manifest_version(self, dataset):
        return {"has_manifest": True, "manifest_generation_id": f"gen-{dataset}"}

    def read_joined(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _snapshot(store):
    return ExperimentDataSnapshot.build(
        store, "exp-1", "us", {"features": "f", "labels": "l"}
    )


def _plan(store, **kwargs):
    return TrainingReadPlan(
        features=("f1", "f2"),
        label="y",
        universe="sp500",
        date_range=("2020-01-01", "2020-12-31"),
        decision_clock="close_t_minus_1",
        datasets={"features": "f", "labels": "l"},
        experiment_snapshot=_snapshot(store),
        **kwargs,
    )


def test_r41_496_497_all_physical_modules_classified_and_importable():
    inventory = module_inventory()
    assert inventory["unclassified"] == ()
    assert inventory["missing"] == ()
    assert set(import_all_public_modules()) == set(inventory["public"])


def test_r41_499_500_514_515_capability_handshake_and_source_identity():
    manifest = CapabilityManifest(
        package_version="1.2.3",
        build_sha="abc",
        api_version="30.1",
        contract_version="7",
        capabilities=frozenset({"training_read_plan", "arrow"}),
    )
    manifest.handshake(
        required_capabilities=("training_read_plan",),
        min_api_version="30.0",
        max_api_version_exclusive="31",
        snapshot_contract_version="experiment-snapshot-v1",
    )
    with pytest.raises(CapabilityHandshakeError):
        manifest.handshake(required_capabilities=("factor_block",))
    identity = ReadIdentity("snap", "build", "contract-v1", "api-v1")
    changed = replace(identity, contract_version="contract-v2")
    assert identity.cache_key("dataset") != changed.cache_key("dataset")
    assert identity.source_identity()["contract_version"] == "contract-v1"


def test_r41_508_509_516_524_training_plan_fails_closed_on_snapshot_contracts():
    store = _Store(result=[{"f1": 1, "y": 2}])
    with pytest.raises(SnapshotFidelityError):
        _plan(store, snapshot_fidelity=SnapshotFidelity.UNKNOWN)
    plan = _plan(store)
    with pytest.raises(SnapshotCompatibilityError):
        plan.assert_snapshot_compatible("different-snapshot")
    with pytest.raises(ValueError):
        _plan(store, financial_revision_mode="latest_restated")


def test_r41_521_522_training_read_never_turns_failures_into_empty():
    denied = _Store(error=AccessDeniedError("denied"))
    with pytest.raises(AccessDeniedError):
        _plan(denied).execute(denied)
    broken = _Store(error=RuntimeError("backend broke"))
    with pytest.raises(BackendReadError):
        _plan(broken).execute(broken)
    empty = _Store(result=[])
    with pytest.raises(EmptyResultError):
        _plan(empty).execute(empty)


def test_r41_517_519_523_525_training_read_plan_executes_one_pinned_arrow_panel():
    rows = [{"f1": 1.0, "f2": 2.0, "y": 3.0}]
    store = _Store(result=rows)
    plan = _plan(store)
    panel = plan.execute(store)
    assert panel.data == rows
    assert panel.experiment_snapshot.snapshot_id == plan.experiment_snapshot.snapshot_id
    assert store.calls[0]["experiment_snapshot"] is plan.experiment_snapshot
    assert store.calls[0]["decision_clock"] == "close_t_minus_1"
    assert store.calls[0]["financial_revision_mode"] == "point_in_time"
    assert store.calls[0]["representation"] == "arrow"
