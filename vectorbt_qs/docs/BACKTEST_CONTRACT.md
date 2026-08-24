# vectorbt_qs Backtest Contract Layer

The contract layer (`vectorbt_qs/contracts/`) standardizes how a backtest is
specified, what ledgers it produces, and how those ledgers are verified.  It
adds three things the mvp engine (accurate / fast execution) did not have:

1. A content-addressed **BacktestArtifact** interface.
2. An **Accounting Invariant Gate** that verifies cash / shares / NAV
   consistency.
3. A **ReferenceLedgerSimulator** — a third, deliberately simple reference
   engine for differential testing.

## Design rule: simulation vs. metric authority

**Metric authority belongs to `quant_evaluator` (QE), not to vectorbt_qs.**

The vectorbt_qs layer only *simulates*: it produces positions, orders, trades,
cash, NAV, turnover and cost ledgers.  It does **not** compute canonical
Sharpe / IC / ICIR or any research metric — those are QE's job.  The
`BacktestArtifact` therefore carries raw ledgers and lets QE (or any trusted
consumer) be the single authority for any computed metric.  This resolves the
"two Sharpe authorities" problem by moving metric computation out of the
simulation layer entirely.

## Contents

| Module | Purpose |
|--------|---------|
| `signal.py` | `SignalArtifact` (decision timeline + position targets) |
| `execution.py` | `ExecutionPolicy`, `CostParams`, `CapacityPolicy` |
| `orders.py` | `OrderArtifact`, `OrdersArtifact` |
| `trades.py` | `TradeArtifact`, `TradesArtifact` |
| `positions.py` | `PositionArtifact`, `PositionsArtifact` |
| `ledger.py` | `CashLedgerArtifact` (daily cash/shares/NAV) |
| `backtest.py` | `BacktestRequest`, `BacktestArtifact` |
| `invariants.py` | Accounting Invariant Gate |
| `reference_simulator.py` | ReferenceLedgerSimulator (third reference engine) |

All artifacts are frozen dataclasses, serializable via `to_dict` /
`from_dict`, and content-hash-addressed via `content_hash_of`.

## Accounting Invariant Gate

```python
from vectorbt_qs.contracts.invariants import verify_accounting_invariants

violations = verify_accounting_invariants(backtest, tolerance=1e-6)
# [] == pass; non-empty list == human-readable violations
```

The gate verifies daily:

- `cash_t + sum(shares_t * mark_t) == NAV_t` (mark from the cash ledger,
  including commission / tax / dividend / split / rights / corporate-action
  flows that touched cash or shares);
- no negative shares (unless `allow_short=True`);
- sell <= available shares and buy <= available cash;
- fill <= requested size;
- non-negative NAV.

The verifier accepts **duck-typed** objects — any result exposing
`timestamps` / `cash` / `shares` / `asset_ids` / `nav` (or `mark` + shares) and
an optional `orders` sequence.  It works with a `BacktestArtifact`, a
`ReferenceLedgerResult`, or a plain dict.  Missing required fields raise a
clear `ValueError` rather than silently passing.

## ReferenceLedgerSimulator

```python
from vectorbt_qs.contracts.reference_simulator import ReferenceLedgerSimulator

sim = ReferenceLedgerSimulator(init_cash=1_000_000.0)
res = sim.simulate(signal, prices)  # prices: open/close frames or dict
violations = verify_accounting_invariants(res.to_artifact_dict())
```

This is the **third reference engine**, deliberately simple and slow-but-correct
(supporting at most 20 assets / 100 days), alongside the Python and Numba mvp
planners.  It exists purely for **differential testing**: run the fast engines
and the reference on the same signal + prices and confirm they agree within a
tolerance.  It implements the most basic A-share rules: 100-share board lots,
next-trading-day-open execution, and a simplified buy/sell fee schedule.  It
intentionally does **not** model suspensions, limit-up/down, corporate actions,
or capacity caps — those are mvp-feature territory, not the reference's job.

## Testing

- `vectorbt_qs/tests/test_invariants.py` — 2-asset / 3-day golden scenario plus
  injected violations.
- `vectorbt_qs/tests/test_reference_simulator.py` — golden scenario with
  hand-verified NAV math, scope limits, and a differential test against the mvp
  engine (honestly skipped if the mvp import chain is unavailable).

Run:

```bash
/tmp/fe2/bin/python -m pytest vectorbt_qs/tests -q -p no:cacheprovider
```

## Platform E2E smoke

`scripts/platform_contract_e2e.py` drives a 10-asset / 60-day synthetic fixture
through the full platform chain (FE -> QE -> FA -> FP -> FO) plus a backtest
invariant gate.  Each stage checks importability and SKIPs honestly when a
package or data source is unavailable.
