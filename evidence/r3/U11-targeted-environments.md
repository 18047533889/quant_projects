# U11 targeted validation environments

- Project environment: `.venv/bin/python`, Python 3.12.3, pandas 2.3.3.
- System pandas 3 environment: `/usr/bin/python3`, Python 3.12.3, pandas 3.0.5.
- `.venv/bin/python_p3_12` resolves to the same Python and pandas versions as
  `.venv/bin/python`; its duplicate run is not counted as a separate environment.

Targeted command in each real environment:

```text
python -m pytest -q tests/operators/test_runtime_operator_snapshot_v3.py factor_engine/tests/operators/test_v9_m33_m36_relational_export.py
```

Both environments completed with 22 passed and 5 warnings. The warnings are
the existing conservative warning for an unknown Polars physical implementation.
