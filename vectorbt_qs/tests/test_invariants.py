"""Tests for the Accounting Invariant Gate.

Constructs a small 2-asset / 3-day scenario that passes the invariants, then
injects each violation class and asserts the gate catches it.
"""

import unittest

from vectorbt_qs.contracts.invariants import verify_accounting_invariants


def _golden_backtest():
    """A 2-asset, 3-day ledger that satisfies all accounting invariants."""
    timestamps = ["d0", "d1", "d2"]
    asset_ids = ["A", "B"]
    cash = [10000.0, 9700.0, 9700.0]
    shares = [[50.0, 0.0], [50.0, 0.0], [50.0, 0.0]]
    mark = [[30.0, 20.0], [31.0, 20.0], [32.0, 20.0]]
    nav = [
        cash[i] + sum(shares[i][j] * mark[i][j] for j in range(2))
        for i in range(3)
    ]
    return {
        "timestamps": timestamps,
        "asset_ids": asset_ids,
        "cash": cash,
        "shares": shares,
        "mark": mark,
        "nav": nav,
        "orders": [],
    }


class AccountingInvariantTests(unittest.TestCase):
    def test_golden_scenario_passes(self):
        violations = verify_accounting_invariants(_golden_backtest())
        self.assertEqual(violations, [])

    def test_catches_nav_reconciliation_break(self):
        bt = _golden_backtest()
        # Mark price moved without a matching NAV/cash update -> recon break.
        bt["mark"][1][0] += 100.0
        violations = verify_accounting_invariants(bt)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("NAV reconciliation" in v for v in violations))

    def test_catches_negative_shares(self):
        bt = _golden_backtest()
        bt["shares"][0][0] = -10.0
        violations = verify_accounting_invariants(bt)
        self.assertTrue(any("negative shares" in v for v in violations))

    def test_missing_nav_and_mark_raises(self):
        bt = _golden_backtest()
        del bt["nav"]
        del bt["mark"]
        with self.assertRaises(ValueError):
            verify_accounting_invariants(bt)

    def test_short_allowed_turns_off_negative_check(self):
        bt = _golden_backtest()
        bt["shares"][0][0] = -10.0
        violations = verify_accounting_invariants(bt, allow_short=True)
        self.assertFalse(any("negative shares" in v for v in violations))

    def test_accepts_duck_typed_object(self):
        # Any object exposing the fields works, not just a dict.
        class _BT:
            timestamps = ["d0", "d1"]
            asset_ids = ["A"]
            cash = [1000.0, 900.0]
            shares = [[10.0], [10.0]]
            mark = [[10.0], [11.0]]
            nav = [1100.0, 1010.0]

        violations = verify_accounting_invariants(_BT())
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
