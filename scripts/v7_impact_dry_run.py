"""Exercise existing impact authority on explicitly synthetic lineage refs."""
import json
from factor_assets.contracts.evidence_ref import EvidenceRef
from quant_evaluator.runtime.impact_plan import plan_v5_impact

refs=(EvidenceRef('fixture:gpu-sharpe-v1','fixture:run','sharpe_ratio','1.0.0',
                  '2026-09-01T00:00:00Z','fixture:factor'),
      EvidenceRef('fixture:current-sharpe','fixture:run','sharpe_ratio','1.0.1',
                  '2026-09-08T00:00:00Z','fixture:factor'),
      EvidenceRef('fixture:rank-unaffected','fixture:run','rank_ic','3.0.0',
                  '2026-09-01T00:00:00Z','fixture:factor'))
plan=plan_v5_impact(refs,changed_metrics=['sharpe_ratio'])
assert plan['stale_evidence_ids']==('fixture:gpu-sharpe-v1',)
assert not plan['rematerialize_factor_values'] and not plan['production_pointer_mutation']
print(json.dumps({'scope':'SYNTHETIC_DRY_RUN_NOT_PRODUCTION_INVENTORY',
    'invalidated_by_issue':['N01'], 'old_metric_version':'1.0.0',
    'new_metric_version':'1.0.1','plan':plan,
    'actual_historical_refs':'UNRESOLVED: no production evidence inventory supplied/queried',
    'refinement':'Within legacy Sharpe records prioritize CUDA, nonzero RF, invalid observations. Preserve old refs; no deletion.'},indent=2))
