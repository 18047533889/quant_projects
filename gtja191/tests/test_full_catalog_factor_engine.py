from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
for path in (PACKAGE_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog
from lib.dsl_normalize import normalize_operator_names
from validate_factor_engine_coverage import validate_catalog


def test_catalog_count_and_audited_vwap_semantics():
    catalog = deliverable_catalog()
    assert len(catalog) == DELIVERABLE_COUNT == 185
    for name, item in catalog.items():
        source = str(item.get("source_formula", "")).upper()
        formula = str(item["dsl_formula"])
        assert formula == normalize_operator_names(formula), name
        if "VWAP" in source:
            assert "col('vwap')" in formula or 'col("vwap")' in formula, name


def test_all_185_compile_and_execute_on_synthetic_panel():
    report = validate_catalog(execute=True, periods=420, symbols=6)
    assert report["compiled"] == 185, report["errors"][:10]
    assert report["executed"] == 185, report["errors"][:10]
    assert report["ok"], report["errors"][:10]
