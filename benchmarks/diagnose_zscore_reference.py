"""Read-only reproduction of the audit's infinity policy mismatch."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.scripts import audit_operator_differential_execution as audit
registry,ok,error=audit.load_registry()
assert ok,error
p=registry.get('zscore','pandas_numpy'); q=registry.get('zscore','polars')
rows=[]
fixtures=audit.build_hostile_fixtures()
for name in ['inf','neg_inf']:
    x=fixtures[name]['x']
    with np.errstate(all='ignore'):
        actual=p.calculate(pd.DataFrame(x)).to_numpy()
        polars=q.calculate(pl.from_pandas(pd.DataFrame(x))).to_numpy()
        reference=audit._zscore_rowwise_np(x)
    different=np.flatnonzero(~np.isclose(actual,reference,equal_nan=True).all(axis=1))
    rows.append({'fixture':name,'different_rows':different.tolist(),
        'input':x[different].tolist(),'audit_finite_only_reference':reference[different].tolist(),
        'pandas_actual':actual[different].tolist(),'polars_actual':polars[different].tolist(),
        'backend_parity':bool(np.allclose(actual,polars,equal_nan=True))})
minimal=np.array([[1.,2.,np.inf],[1.,2.,np.nan],[1.,2.,3.]])
with np.errstate(all='ignore'):
    minimal_actual=p.calculate(pd.DataFrame(minimal)).to_numpy()
    minimal_ref=audit._zscore_rowwise_np(minimal)
report={'changed_production_files':False,'pandas_implementation':str(type(p)),
    'polars_implementation':str(type(q)),'fixtures':rows,
    'minimal_input':minimal.tolist(),'minimal_pandas_actual':minimal_actual.tolist(),
    'minimal_audit_reference':minimal_ref.tolist(),
    'cause':'Both backends use pandas mean/std(ddof=1) without prefiltering infinity; the audit reference first filters isfinite, so finite peers survive only in the audit oracle.',
    'classification':'Reference/canonical nonfinite-policy mismatch, not backend-parity or ddof error. Canonical source implements pandas propagation; existing boundary test is too weak to distinguish the two policies.',
    'recommendation':'Preserve current canonical pandas semantics and align the audit reference, unless an explicit semantic decision changes zscore to finite-only. Any finite-only change must update all backends and semantic evidence together; do not silently alter formula to satisfy this audit.'}
Path('evidence/r2/20260906_zscore_reference_diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
