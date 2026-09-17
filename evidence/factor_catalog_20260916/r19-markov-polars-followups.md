# R19 Markov/Polars follow-up evidence

Observed while validating the `KeyError: state` repair on 2026-09-17. These are separate from the repaired pandas selective-output defect and were not modified in this pass.

## Kramers–Moyal local stability

Command:

```text
.venv/bin/pytest -q factor_engine/tests/operators/test_dynamics_pack_2026_08.py::test_dynamics_polars_parity
```

Parameters at failure: `ts_kramers_moyal_local_stability(window=40, bins=5, lag=1, min_count=3, min_state_support=2, min_history=10)`.

Exception: `AttributeError: 'DataFrame' object has no attribute 'diff'` at `polars_native/ts_advanced_batch5.py:1139`, where the Polars implementation calls pandas-style `feature.diff().rolling_mean(window).diff()`.

## Markov entropy-production Polars metadata

Direct reproduction used a 180-row single-column Polars frame and:

```text
ts_markov_entropy_production(window=120, bins=5, lag=1, min_periods=60)
```

Exception: `ValueError: ts_markov_entropy_production: min_history must not exceed the window length ... got min_history={min_history}, window={window}` during relational validation. The pandas backend accepts the same formal parameters and returns shape `(180, 1)`. The unformatted placeholders indicate inconsistent Polars registration metadata/default binding; no fix was attempted here.
