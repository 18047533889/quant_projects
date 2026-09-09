"""Read-only W0 import/distribution/entry-point inventory for actual checkout."""
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
packages=('quant_evaluator','factor_assets','factor_optimizer','factor_preprocess','quant_platform','factor_engine')
modules={name:importlib.import_module(name) for name in packages}
distribution_map=metadata.packages_distributions()
rows={}
for name,module in modules.items():
    distributions=[]
    for distname in distribution_map.get(name,[]):
        dist=metadata.distribution(distname)
        distributions.append({'name':dist.metadata['Name'],'version':dist.version,
            'metadata_path':str(dist._path),
            'direct_url':json.loads(dist.read_text('direct_url.json') or 'null'),
            'entry_points':[{'name':ep.name,'group':ep.group,'value':ep.value} for ep in dist.entry_points]})
    path=Path(module.__file__)
    rows[name]={'actual_import':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'distributions':distributions}
duplicates=[str(path.relative_to(root)) for package in packages
            for path in (root/package/'build'/'lib').glob('**/__init__.py')]
payload={'python':sys.executable,'sys_path':sys.path,'cwd':str(Path.cwd()),
         'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
         'packages':rows,'build_copy_entrypoints_not_imported':duplicates,
         'scope':'Invocation-level PYTHONPATH pins remediation sources; no installed-wheel replacement or production deployment.'}
print(json.dumps(payload,indent=2,sort_keys=True))
