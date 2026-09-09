from factor_assets.contracts.evidence_ref import EvidenceRef
from quant_evaluator.runtime.impact_plan import plan_v5_impact


def test_versions_invalidate_only_changed_evidence_not_materialized_factors():
    def ref(eid,metric,version):
        return EvidenceRef(eid,"old-run",metric,version,"2026-09-01T00:00:00Z","factor-a")
    refs=(ref("old-rank","rank_ic","1.0.0"),ref("current-rank","rank_ic","3.0.0"),
          ref("pearson","pearson_ic","1.0.0"),ref("alias","rank_icir_raw","0.1.0"))
    plan=plan_v5_impact(refs)
    assert set(plan["stale_evidence_ids"]) == {"old-rank","alias"}
    assert set(plan["unaffected_evidence_ids"]) == {"current-rank","pearson"}
    assert not plan["rematerialize_factor_values"]
    assert not plan["production_pointer_mutation"]
    assert not plan["delete_old_artifacts"]
    policy=plan_v5_impact(refs,change_kind="GRADING_POLICY")
    assert policy["actions"] == ("RESCORE_STORED_RAW_METRICS",)
    assert not policy["rematerialize_factor_values"]
