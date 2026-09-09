"""Create narrowly scoped receipts from executed frozen-source JUnit results.

These certify the listed function/test-fixture domains, not all registry routes
or production markets. Nothing here installs a receipt in a trusted resolver.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from quant_evaluator.contracts.qualification import NumericalQualificationReceipt
from v8_final_regression import check_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v8/numerical_receipts.json'
MANIFEST = ROOT / 'evidence/v8/final_source/source_manifest.json'
JUNIT = ROOT / 'evidence/v8/final_regression/qe.xml'

CASES = (
    ('quant_evaluator.metrics.robustness.compute_hac_variance', 'metrics/robustness.py',
     ('test_t31_hac_bartlett_normalization','test_t32_hac_matches_positive_semidefinite_quadratic_form'),
     'CPU_FP64 alternating length20 lag1 offset and independent Bartlett PSD quadratic n/lag grid (20/0,20/1,20/10,60/5)'),
    ('quant_evaluator.metrics.risk.var_cvar.compute_cvar', 'metrics/risk/var_cvar.py',
     ('test_t77_fixed_tail_mass','test_t78_fractional_tail_mass'), 'CPU_FP64 historical cvar .95 length100 and .8 length7/21 empirical mass witnesses'),
    ('quant_evaluator.metrics.multiple_testing.sidak_correction', 'metrics/multiple_testing.py',
     ('test_t87_sidak_tiny_values',), 'CPU_FP64 Sidak p=1e-20 full m=100000 numerical small-p witness'),
    ('quant_evaluator.metrics.temporal.compute_autocorrelation', 'metrics/temporal.py',
     ('test_t38_nonmonotone_lag_pair_counts','test_t38_lag_counts_are_explicit'), 'CPU_FP64 gapped alternating observation clock, nonmonotone lag pair counts'),
    ('quant_evaluator.metrics.stats.cointegration.johansen_test', 'metrics/stats/cointegration.py',
     ('test_t55_johansen_reference',), 'CPU_FP64 K={1,2,3,6,12} det={-1,0,1} VAR-lag={1,2,4} alpha={.01,.05,.10} exact fixture grid; K1 det>=0 only rejection qualified'),
    ('quant_evaluator.kernels.gpu.quantile.batched_quantile_returns','kernels/gpu/quantile.py',
     ('test_quantile_returns_bounded_workspace_matches_full',), 'CUDA_FP64 T3 F2 N200 Q10/20/40 min_assets1 max-tie, 65536-byte scratch, selected Inf/NaN positions'),
)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    check_sources()
    cases = list(ET.parse(JUNIT).getroot().iter('testcase'))
    execution = json.loads((ROOT/'evidence/v8/final_regression/execution.json').read_text())
    if execution['suites']['qe']['returncode'] != 0 or execution['source_changed']:
        raise RuntimeError('cannot qualify failed or changed-source run')
    records = {row['path']:row['sha256'] for row in json.loads(MANIFEST.read_text())['files']}
    versions = {name:importlib.metadata.version(name) for name in ('numpy','scipy','statsmodels')}
    import cupy
    versions['cupy'] = cupy.__version__
    run_ref = 'sha256:' + hashlib.sha256(JUNIT.read_bytes()).hexdigest()
    source_tree = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    output=[]
    for route, source, names, scope in CASES:
        matched=[case for case in cases if any(case.attrib['name'].split('[')[0]==name for name in names)]
        if {c.attrib['name'].split('[')[0] for c in matched} != set(names):
            raise RuntimeError('required executed tests missing: '+route)
        if any(list(c.iter('skipped')) or list(c.iter('failure')) or list(c.iter('error')) for c in matched):
            raise RuntimeError('nonpassing assertion cannot qualify: '+route)
        assertions={c.attrib['classname']+'::'+c.attrib['name']:'PASS' for c in matched}
        parameter_domain=dict(scope=scope, test_source_sha256=records['quant_evaluator/tests/test_v8_numeric_goldens.py'],
                              scope_extent='exact synthetic fixtures only',dependencies=versions)
        implementation=dict(source=records['quant_evaluator/'+source], dependencies=versions)
        if '/gpu/' in source: implementation['rank']=records['quant_evaluator/kernels/gpu/rank.py']
        receipt=NumericalQualificationReceipt(source_tree_hash=source_tree,
            implementation_hash=digest(implementation), route=route,
            backend='CUDA/FP64' if '/gpu/' in source else 'CPU/FP64',
            parameter_domain_hash=digest(parameter_domain),
            metric_instance_hash=digest(dict(route=route,domain=parameter_domain)),
            test_run_ref=run_ref,assertions=assertions)
        output.append(dict(receipt={**receipt.__dict__,'assertions':dict(receipt.assertions)},
                           receipt_hash=receipt.content_hash,parameter_domain=parameter_domain))
    check_sources()
    OUT.write_text(json.dumps(dict(receipts=output,source_manifest_sha256=source_tree,
        installed_in_production_resolver=False,
        excluded='No extension to untested parameters, registry routes, real-market calibration, production certification or deployment.'),indent=2)+'\n')
    print(f'{len(output)} executed-test domain receipts; no production authorization')

if __name__ == '__main__':
    main()
