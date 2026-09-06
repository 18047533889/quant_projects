"""Join measured local evidence; never turn unrun operators into certified ones."""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
inventory = json.loads(Path('evidence/r2/20260906_operator_inventory.json').read_text())
audit = json.loads(Path('evidence/r2/20260906_operator_differential_corrected/operator_differential_execution.json').read_text())
audited = {r['canonical']: r for r in audit['results']}
oracle_names = {'sqrt_abs', 'log_abs', 'exp_neg', 'ts_pct', 'signed_log'}
stale_path = Path('evidence/r2/20260906_intraday_stale_binding.json')
stale = json.loads(stale_path.read_text()) if stale_path.exists() else {}
stale_affected = {r['canonical']:r for r in stale.get('affected',[])}

def source_evidence(class_repr):
    if not class_repr or class_repr == "<class 'NoneType'>":
        return None
    dotted = class_repr.removeprefix("<class '").removesuffix("'>").split('.')
    for end in range(len(dotted), 0, -1):
        candidate = Path(*dotted[:end]).with_suffix('.py')
        if candidate.is_file():
            return {'path': str(candidate), 'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return {'status': 'SOURCE_NOT_RESOLVED', 'class': class_repr}

rows=[]
for row in inventory:
    name = row['canonical']
    detail = audited.get(name)
    blockers=[]
    if row['polars_kind'] == 'unsupported':
        blockers.append('POLARS_PHYSICAL_CONTRACT_UNSUPPORTED_OR_MISSING')
    elif row['polars_kind'] == 'polars_udf_pandas_delegate':
        blockers.append('POLARS_PANDAS_DELEGATE_NOT_NATIVE')
    if name not in oracle_names:
        blockers.append('INDEPENDENT_MATH_ORACLE_NOT_RUN_THIS_BATCH')
    blockers.extend(['FULL_PARAMETER_DOMAIN_NOT_RUN', 'FULL_PIT_POISON_NOT_RUN', 'PRODUCTION_GATE_NOT_RUN'])
    if detail and detail['certification'] == 'not_certified':
        blockers.append('BOUNDED_DIFFERENTIAL_NOT_CERTIFIED')
    if name in stale_affected:
        blockers.append('INTRADAY_CERTIFICATE_SOURCE_OR_ORACLE_HASH_STALE')
    rows.append({**row, 'source_evidence': {b: source_evidence(row[b+'_class']) for b in ['pandas','polars']},
        'independent_oracle': {'status':'PASS' if name in oracle_names else 'NOT_RUN',
            'evidence':'factor_engine/tests/operators/test_polars_nonfinite_reference_parity.py' if name in oracle_names else None},
        'bounded_differential':detail or {'status':'NOT_RUN'}, 'blockers':blockers,
        'production_certification':'BLOCKED_STALE_EVIDENCE' if name in stale_affected else 'NOT_RUN',
        'current_production_admission':stale_affected.get(name,{'status':'NOT_RUN'}),
        'source_binding_evidence':str(stale_path) if name in stale_affected else None})

reasons={}
for name, r in audited.items():
    if r['certification'] != 'not_certified': continue
    found=[]
    if not r['native_used'] and not r['fallback_used']: found.append('NO_PRODUCTION_ADMITTED_IMPLEMENTATION_OR_UNKNOWN_PHYSICAL_KIND')
    if r.get('reference_all_pass') is False: found.append('REFERENCE_MISMATCH_OR_EXECUTION_FAILURE')
    if any('err:' in f.get('note','') for f in r['fixtures']): found.append('FIXTURE_EXECUTION_ERROR')
    if r['physical_kind']=='unsupported': found.append('PHYSICAL_KIND_UNSUPPORTED')
    reasons[name]=found or ['PARITY_NOT_CERTIFIED']

payload={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'timestamp':datetime.now(timezone.utc).isoformat(),
    'scope':'Current canonical inventory; evidence from this bounded remediation batch only',
    'counts':{'canonical':len(rows),'physical_kinds':dict(Counter(r['polars_kind'] for r in rows)),
        'independent_oracle_pass':sum(r['independent_oracle']['status']=='PASS' for r in rows)},
    'not_certified_reasons':reasons,'rows':rows}
shared_sources = [
    'factor_engine/backend/cleaned_bridge.py',
    'factor_engine/backend/panel_native.py',
    'factor_engine/fields/registry.py',
    'factor_engine/planner/read_wave_planner.py',
    'factor_engine/cleaned_operators/r23_cert_intraday.py',
    'factor_engine/scripts/audit_operator_differential_execution.py',
]
payload['shared_source_hashes'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in shared_sources}
payload['shared_source_note'] = 'Current shared execution/governance helper bindings for traceability; not a transitive semantic closure or new certification.'
for row in payload['rows']:
    if row['canonical'] == 'zscore':
        row['diagnosis_status'] = 'ORACLE_SEMANTICS_MISMATCH'
        row['diagnosis_evidence'] = 'evidence/r2/20260906_zscore_reference_diagnosis.json'
        row['blockers'].append('ORACLE_SEMANTICS_MISMATCH')
        row['limitations'] = ['Audit oracle excludes Inf but canonical pandas/Polars propagate it. Both backends agree; no mathematical bug was repaired and canonical semantics were not changed.']
payload['limitations'] = ['zscore has ORACLE_SEMANTICS_MISMATCH; see evidence/r2/20260906_zscore_reference_diagnosis.json. This is an unresolved audit-reference policy mismatch, not a repaired operator mathematics defect.']
Path('evidence/r2/20260906_operator_coverage_matrix.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({'counts':payload['counts'],'not_certified_reasons':reasons},indent=2))
