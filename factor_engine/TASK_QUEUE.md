# FactorEngine Queue (compact)

**Status:** historical 2026-08-14 scan; verify current tree before action. Current campaign authority: `../AUTONOMOUS_MASTER_QUEUE.md`.

| Priority | Item | Current truth / evidence |
|---|---|---|
| P0 | Registry alias/bootstrap blocker | Historical report: `ts_mean_abs_deviation`; re-reproduce before editing shared authorities |
| P0 | Indicator parity: EMA/SMA/RSI/MACD/Bollinger | `BACKEND_PARITY_FINDINGS.md:72-82`; broad parity `NOT_RUN` |
| P1 | Stats parity: corr/cov/skew/kurt/quantile | `BACKEND_PARITY_FINDINGS.md:58-69`; broad parity `NOT_RUN` |
| P1 | `cs_zscore` parity | `BACKEND_PARITY_FINDINGS.md:45`; `NOT_RUN` |
| P1 | Q four-way parity | Q remains fail-closed / `NOT_PRODUCTION_CERTIFIED` |
| P1 | EWM/EMA/`ts_kurt` NaN-hole divergence | Known historical finding; re-reproduce |
| P1 | Fake Polars native (`.to_pandas()` etc.) | `BACKEND_SCAN_FINDINGS.txt`; performance/classification issue |
| P2 | Q compile/runtime/parity evidence | `backend/q_backend/q_capability_evidence.py`; production gates `NOT_RUN` |
| P2 | Base abstract `NotImplementedError` | Expected if truly abstract; triage before change |

Workflow: reproduce → narrow edit → oracle → import/syntax smoke → manifest → independent scoped review → local integration. Never infer PASS from this queue.
