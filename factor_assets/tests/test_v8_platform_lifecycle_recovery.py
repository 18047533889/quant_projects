"""Cross-layer recovery proof for FA lifecycle and the platform outbox/inbox."""

from types import SimpleNamespace

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lifecycle import HealthState, LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry import create_platform_lifecycle
from factor_assets.registry.lifecycle import TransitionRequest
from factor_assets.tests.assembly.test_production_and_similarity_artifact import (
    make_admission,
    make_spec,
    make_treatment,
)
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.outbox import Inbox, Outbox


class _TrustedTestAuthority:
    """Explicit test-only authority; production composition still resolves it."""

    def __init__(self, artifact):
        self._artifact = artifact

    def resolve(self, ref):
        if ref != self._artifact.content_hash:
            raise KeyError(ref)
        return self._artifact


def test_lost_callback_retry_redelivery_and_retirement_exclusion(tmp_path):
    db_path = tmp_path / "platform-lifecycle.db"
    authority = _TrustedTestAuthority(
        SimpleNamespace(
            factor_id="F-RECOVERY",
            content_hash="approval:F-RECOVERY:v1",
            decision="APPROVED",
            expires_at="2099-01-01T00:00:00Z",
        )
    )
    lifecycle, repository = create_platform_lifecycle(
        db_path=db_path, authorization_resolver=authority
    )
    repository.register(
        AssetMetadata(
            factor_id="F-RECOVERY",
            canonical_repr="price momentum recovery factor",
            canonical_hash="sha256:f-recovery-v1",
            frequency="daily",
            domains=("price",),
            timing="daily",
        ),
        LineageRef(factor_id="F-RECOVERY", parents=()),
    )
    bundle = EvidenceBundleRef(
        "bundle-recovery", "run-recovery", ("F-RECOVERY",),
        "2026-09-01T00:00:00Z", "qe-v8",
    )
    lifecycle.execute_transition(
        TransitionRequest(
            factor_id="F-RECOVERY",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            evidence_refs=(),
            decision_id="evaluate-recovery-v1",
            evidence_bundle_ref=bundle,
        )
    )
    approval = TransitionRequest(
        factor_id="F-RECOVERY",
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.APPROVED,
        evidence_refs=("gate_results",),
        decision_id="approve-recovery-v1",
        policy_version="admission-v8",
        actor="trusted-test-operator",
        notes="recovery acceptance proof",
        expected_revision=1,
        authorization_ref="approval:F-RECOVERY:v1",
    )
    lifecycle.add_event_listener(lambda _event: (_ for _ in ()).throw(RuntimeError("callback lost")))
    committed = lifecycle.execute_transition(approval)
    assert committed.revision == 2
    assert any("callback lost" in warning for warning in committed.warnings)

    # Model a lost response: discard the result, reopen every durable component,
    # and retry the byte-for-byte-equivalent command.
    lifecycle, repository = create_platform_lifecycle(
        db_path=db_path, authorization_resolver=authority
    )
    replay = lifecycle.execute_transition(approval)
    assert replay.revision == committed.revision
    assert replay.event == committed.event

    platform_db = SqliteDb(str(db_path), create=True)
    approval_key = "factor-lifecycle:F-RECOVERY:2"
    rows = platform_db.query(
        "SELECT idempotency_key, status FROM outbox_events WHERE idempotency_key = ?",
        (approval_key,),
    )
    assert rows == [{"idempotency_key": approval_key, "status": "pending"}]

    handled = []
    inbox = Inbox(platform_db, lambda event: handled.append(event["idempotency_key"]))

    class _AckLostPublisher:
        def __init__(self):
            self.calls = 0

        def publish(self, event):
            self.calls += 1
            inbox.process(event)
            if self.calls == 1:
                raise RuntimeError("broker acknowledgement lost")

    publisher = _AckLostPublisher()
    outbox = Outbox(platform_db, publisher)
    assert outbox.publish_pending(idempotency_key=approval_key) == 0
    platform_db.execute(
        "UPDATE outbox_events SET next_attempt_at = 0 WHERE idempotency_key = ?",
        (approval_key,),
    )
    platform_db._conn.commit()
    assert outbox.publish_pending(idempotency_key=approval_key) == 1
    assert publisher.calls == 2
    assert handled == [approval_key]
    assert inbox.status_of(approval_key) == "DONE"

    repository.update_asset_health(
        "F-RECOVERY", HealthState.RETIRED,
        reason="health policy retirement", actor="health-monitor",
    )
    retired = repository.get("F-RECOVERY")
    assert retired.lifecycle_state is LifecycleState.APPROVED
    assert retired.health_state is HealthState.RETIRED
    with pytest.raises(ValueError, match="no factor assets match"):
        FactorSetAssembler().assemble(
            make_spec("recovery-new-set", "Recovery new set", "manual"),
            [retired],
            admission_artifacts={"F-RECOVERY": make_admission("F-RECOVERY")},
            treatment_selection_artifacts={"F-RECOVERY": make_treatment("F-RECOVERY")},
            production=True,
        )
    platform_db.close()
