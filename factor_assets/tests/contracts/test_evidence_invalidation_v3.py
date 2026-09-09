from factor_assets.contracts.evidence_ref import EvidenceRef, plan_metric_version_invalidation


def _ref(evidence_id, factor_id, metric, version):
    return EvidenceRef(
        evidence_id=evidence_id, evaluation_run_id="run", metric_name=metric,
        metric_version=version, timestamp="2026-09-07T00:00:00Z",
        factor_id=factor_id,
    )


def test_metric_version_invalidation_is_targeted_and_dry_run():
    refs = (
        _ref("old-dd", "F1", "max_drawdown", "1.0.0"),
        _ref("new-dd", "F2", "max_drawdown", "2.0.0"),
        _ref("ic", "F3", "rank_ic", "1.0.0"),
    )
    plan = plan_metric_version_invalidation(refs, {"max_drawdown": "2.0.0"})
    assert plan.stale_evidence_ids == ("old-dd",)
    assert plan.affected_factor_ids == ("F1",)
    assert plan.unaffected_evidence_ids == ("new-dd", "ic")
    assert refs[0].metric_version == "1.0.0"  # no mutation
