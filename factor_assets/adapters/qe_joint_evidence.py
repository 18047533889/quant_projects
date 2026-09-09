"""Strict adapter from QE distributions to FA raw joint evidence."""
import json, hashlib, math
from factor_assets.selection.decision import RawJointMetricEvidence

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()

def adapt_qe_distributions(*, evidence_id, comparison_context_hash, factor_id,
                           metric_ids, metric_units, window_ids, scenario_ids,
                           artifacts, qualification_scope, source_tree_hash,
                           implementation_hash, route, backend,
                           parameter_domain_hash, metric_instance_hash):
    """Require a complete common QE plan/grid; never synthesize samples."""
    metric_ids,metric_units,window_ids,scenario_ids=map(tuple,(metric_ids,metric_units,window_ids,scenario_ids))
    expected={(w,s,m) for w in window_ids for s in scenario_ids for m in metric_ids}
    if set(artifacts)!=expected: raise ValueError("QE artifact grid is incomplete or contains undeclared axes")
    first=artifacts[next(iter(sorted(expected)))]
    provenance=dict(first.provenance)
    required=("resampling_plan_ref","replicate_ids","clock_ref","time_ids","factor_ids",
              "recipe_hash","fitted_state_hash","data_snapshot_hash","universe_hash",
              "label_hash","value_artifact_hash","resampling_plan_content_hash",
              "sample_identity_hash","time_identity_hash","common_mask_hash")
    if any(not provenance.get(key) for key in required): raise ValueError("QE distribution lacks joint provenance")
    replicate_ids=tuple(provenance["replicate_ids"]); factor_ids=tuple(provenance["factor_ids"])
    if len(replicate_ids)<2 or factor_id not in factor_ids: raise ValueError("QE factor/replicate axis mismatch")
    samples=[]
    for r in range(len(replicate_ids)):
        windows=[]
        for window in window_ids:
            scenarios=[]
            for scenario in scenario_ids:
                values=[]
                for metric in metric_ids:
                    artifact=artifacts[(window,scenario,metric)]; p=dict(artifact.provenance)
                    for key in required:
                        if p.get(key)!=provenance[key]: raise ValueError("QE distributions do not share exact plan/axes")
                    if artifact.metric_id!=metric or tuple(artifact.stat_names)!=factor_ids: raise ValueError("QE metric/factor identity mismatch")
                    value=float(artifact.samples[r,factor_ids.index(factor_id)])
                    if not math.isfinite(value): raise ValueError("QE joint metric sample must be finite")
                    values.append(value)
                scenarios.append(tuple(values))
            windows.append(tuple(scenarios))
        samples.append(tuple(windows))
    candidate_identity=(factor_id,provenance["recipe_hash"],provenance["fitted_state_hash"],
        provenance["data_snapshot_hash"],provenance["universe_hash"],provenance["label_hash"],
        provenance["value_artifact_hash"])
    pairing_identity=(provenance["resampling_plan_content_hash"],provenance["sample_identity_hash"],
        provenance["time_identity_hash"],provenance["common_mask_hash"])
    semantic={"context":comparison_context_hash,"plan":provenance["resampling_plan_ref"],
        "pairing":pairing_identity,"replicates":replicate_ids,"metrics":metric_ids,
        "metric_units":metric_units,"window_ids":window_ids,"scenario_ids":scenario_ids,
        "samples":samples,"qualification":qualification_scope,
        "execution":(source_tree_hash,implementation_hash,route,backend,parameter_domain_hash,metric_instance_hash),
        "candidate":candidate_identity}
    return RawJointMetricEvidence(evidence_id,_digest(semantic),comparison_context_hash,
        provenance["resampling_plan_ref"],replicate_ids,metric_ids,tuple(samples),qualification_scope,
        window_ids=window_ids,scenario_ids=scenario_ids,source_tree_hash=source_tree_hash,
        implementation_hash=implementation_hash,route=route,backend=backend,
        parameter_domain_hash=parameter_domain_hash,metric_instance_hash=metric_instance_hash,
        metric_units=metric_units,candidate_id=factor_id,recipe_hash=provenance["recipe_hash"],
        fitted_state_hash=provenance["fitted_state_hash"],data_snapshot_hash=provenance["data_snapshot_hash"],
        universe_hash=provenance["universe_hash"],label_hash=provenance["label_hash"],
        value_artifact_hash=provenance["value_artifact_hash"],
        resampling_plan_content_hash=provenance["resampling_plan_content_hash"],
        sample_identity_hash=provenance["sample_identity_hash"],
        time_identity_hash=provenance["time_identity_hash"],common_mask_hash=provenance["common_mask_hash"])
