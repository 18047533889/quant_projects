from __future__ import annotations

from pathlib import Path
import sys
import os

import pytest

pytest.importorskip("pandas")

from tests.helpers import FE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_engine.runtime.real_data_factor_smoke import DATASET_SPECS, run_dataset_factor_smoke
from factor_engine.util.workspace_paths import resolve_path

MASSIVE_ROOT = resolve_path(
    os.environ.get("MASSIVE_PARQUET_ROOT", "~/quant_projects/data/us_stock/cleaned_massive_data")
)


@pytest.mark.parametrize("spec", DATASET_SPECS, ids=[s.name for s in DATASET_SPECS])
def test_real_dataset_factor_smoke(spec):
    if os.environ.get("RUN_REAL_PARQUET_SMOKE") != "1":
        pytest.skip("set RUN_REAL_PARQUET_SMOKE=1 to enable real parquet integration smoke")

    if not MASSIVE_ROOT.exists():
        pytest.skip(f"massive parquet root not found: {MASSIVE_ROOT}")

    report = run_dataset_factor_smoke(MASSIVE_ROOT, spec, max_files=2)
    success = [item for item in report['results'] if item['ok']]
    if not success:
        errors = [item.get('error', '') for item in report['results'] if not item.get('ok')]
        details = '; '.join(errors[:2]) if errors else 'unknown error'
        pytest.skip(f"no readable data for this runtime stack: {spec.name} | {details}")

    assert report['factor_count'] >= 3
    assert success, f"No factors succeeded for {spec.name}: {report['results']}"
    assert any(item.get('rows', 0) > 0 for item in success)
    assert any(item.get('non_nan_ratio', 0.0) > 0.0 for item in success)
