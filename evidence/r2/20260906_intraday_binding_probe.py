import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.r23_cert_intraday import R23_CERTIFIED_CANONICALS, _minute_shape_from_artifact
from factor_engine.cleaned_operators.operator_spec import build_operator_spec
load_all()
artifact=json.loads(Path('evidence/intraday_minute_parity.json').read_text())
bindings={
    'operator_source_hash':'factor_engine/cleaned_operators/microstructure/intraday_agg.py',
    'test_file_hash':'factor_engine/tests/backend_parity/test_intraday_minute_parity.py',
    'helper_file_hash':'factor_engine/tests/backend_parity/intraday_minute_parity.py'}
comparison={k:{'path':p,'recorded':artifact.get(k),'current':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k,p in bindings.items()}
rows=[{'canonical':c,'allow_in_production':build_operator_spec(c).allow_in_production} for c in sorted(R23_CERTIFIED_CANONICALS)]
payload={'timestamp':datetime.now(timezone.utc).isoformat(),
    'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'bindings':comparison,'minute_shape_admitted':sorted(_minute_shape_from_artifact()),'affected':rows,
    'note':'No old artifact was rewritten or recertified.'}
Path('evidence/r2/20260906_intraday_stale_binding.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
