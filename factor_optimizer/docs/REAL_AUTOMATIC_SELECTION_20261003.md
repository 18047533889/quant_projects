# Real automatic TRAIN/VALIDATION selection

The manifest-bound factor `weekly_4cbd7ca6dccc61dc` was read from the existing
authorized COS pool with a 500-date × 256-asset panel. Its object is 5,400,494
bytes; dates span 2024-08-02 through 2026-08-25. Existing DataAccess loaders and
existing optimizer APIs were reused; no factor/data repository was copied.

The expanded default catalogue yielded 120 raw proposals, 116 unique after
deduplication (budget 128). Twenty-nine candidates passed training eligibility
and were compared; 87 were excluded: 68 raw-relative degradation floors,
10 unavailable joint metrics, two insufficient coverage/IC and seven missing
DSL/exposure/control routes. Excluded candidates are not silently substituted
with RAW or counted as evaluated gains.

The TRAIN-frozen winner was `INVERTED_U_REPAIR`, with TRAIN utility gain
0.1974876786 and paired held-out block lower bound 0.1199110457. Validation
coverage was 1.0 and the result reported `improved` / `SUPERIOR`.
Its parameters were `center=0.35`, `power=2.0`, `asymmetry=true`, orientation
+1. This newly enumerated asymmetric branch was not in the former static
auto catalogue; the improvement therefore exercises the added coverage.

| Costed VALIDATION metric | RAW reference | Frozen candidate |
|---|---:|---:|
| Rank IC | -0.02154554 | 0.01404886 |
| Rank ICIR | -0.33568502 | 0.21683216 |
| Sharpe | -0.17736528 | 0.87707279 |
| Max drawdown | 0.09550561 | 0.07717276 |
| Turnover | 0.04116319 | 0.03634264 |
| Worst-block Sharpe | -3.34526847 | -1.16995186 |

The cost-rate parameter was 0.001 with `signal_cash` empty-leg policy.
Only one frozen TRAIN winner was confirmed on VALIDATION. TEST labels were
not scored. The selected plan subsequently transforms the full feature timeline
including TEST-period factor features; that is not TEST-label evaluation.
This is one-factor research evidence, not guaranteed profit or production/PIT
admission. Input data/provenance fingerprints before/after matched; they are
not a complete source/runtime closure. Output factor IDs/order, axes, shape,
finite mask, research context, split and factor-result mapping passed guards.

Evidence: `real_automatic_selection_oct03.json` (142,057 bytes). Reproduce using
configured server-c DataAccess environment and a fresh report path:

`.venv/bin/python factor_optimizer/scripts/audit_real_automatic_oct03.py --report NEW.json`.

## Harness guards

Independent method/automatic harness tests passed 20 cases in 0.95 seconds.
They reject mismatched IDs/dates/masks, TEST-scored results, forced input
replacement, report overwrite and oversized reports. Nonfinite reported
metrics fail closed instead of being silently converted into valid-looking
numbers. This focused test run is not a full optimizer regression.

### Extreme finite-label overflow follow-up

An 84-date × 200-asset synthetic check found that very large but finite labels
could overflow the ordinary float64 reductions when centering the twenty-layer
decay cube. The diagnostic could then report `available` with nonfinite
`mean_excess_returns`. The guard now preserves NumPy's ordinary float64 mean
path and calls QE's existing exact finite mean only for a nonfinite reduction
whose entire source slice is finite; source slices containing NaN/Inf keep
NumPy semantics. The synthetic alternating `1e308`/`9e307` label case now
has a regression requiring an available result with finite curves and proposals,
without warnings. Such magnitudes are a numerical stress test, not representative
financial returns. The initial focused version reported 4 passed but only
checked finiteness conditionally on availability; the strengthened focused
numeric/audit group later passed 14 tests in 7.72s.

The first full FO test run was `1742 passed, 4 failed, 66 warnings` (163.43s).
Two failures were old dynamic-budget expectations (fixture cap 7 versus the
expanded catalogue; focused fix uses 13); two were cache tests comparing the
new QE guarded mean against a stale bincount bit-exact oracle (maximum observed
difference 2.22e-16; checked against the current uncached path and Fraction).

An intermediate full FO attempt exited 2 after 2.94s during test collection, with a
SyntaxError in `test_research_execution_dedup_order_oct03.py` caused by a
missing closing `]`. That syntax has since been fixed, and the ordering test
passed independently (1 passed in 1.20s). The initial four test failures and
this later collection failure are distinct runs, retained here rather than
rewritten as successful attempts.

After the test-file writes completed and QE's three numerical modules were
explicitly frozen, the full command was rerun:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q factor_optimizer/tests`.
Result: **1754 passed, 66 warnings in 159.33s**, exit 0. The API generator's
`--check` also reported `factor_optimizer: reference current`.

QE module SHA256 fingerprints matched before and after that full run:

- `metrics/quantile.py`: `1e3fc5841f38792a81cf9b849f5eb585c4cff1b81076081f9770739a5612770e`
- `metrics/quantile_numba.py`: `8162064ff8f0b7d10de9ae66a903e8eb0a7c03bd27c5751a9861842523ad842a`
- `metrics/quantile_numeric.py`: `a3b138da1c6d0cf89df24aa7b02a07911c7051ba19e416d51da8f53f77951f60`

This is a functional regression result for that source state, not a claim of
global fastest performance, production admission, or exhaustive absence of bugs.
