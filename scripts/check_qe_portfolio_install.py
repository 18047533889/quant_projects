"""Independent installed/source cohort golden; run with Python -I outside repo."""
import argparse
import json
from pathlib import Path
import numpy as np
import quant_evaluator.metrics.probe_portfolio as portfolio
from quant_evaluator.metrics import probe_portfolio_legacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-only", action="store_true")
    args = parser.parse_args()
    implementation = Path(portfolio.__file__).resolve()
    root = Path(__file__).resolve().parents[1]
    if args.wheel_only and implementation.is_relative_to(root):
        raise AssertionError("wheel-only probe imported the source checkout")
    assert implementation.name == "__init__.py"
    assert portfolio.construct_long_short_portfolio is probe_portfolio_legacy.construct_long_short_portfolio
    factors = np.tile([0., 1.], (5, 1))
    returns = np.tile([-.01, .02], (5, 1))
    out = portfolio.compute_cohort_pnl(factors, returns, np.ones_like(returns),
        n_quantiles=2, holding=1, per_side_cost=.001, require_tradable=False)
    # Signal s enters s+1; its one close-to-close return lands at s+2.
    # Gross .5*.02 - .5*(-.01)=.015, entry/exit each costs .001.
    np.testing.assert_allclose(out["pnl_net"], [0., -.001, .013, .013, .013], atol=1e-12)
    costs = portfolio.compute_transaction_costs([100., 200.], 1000.,
        commission_rate=.001, slippage_rate=.002)
    np.testing.assert_allclose(costs.values, [.0003, .0006], atol=1e-15)
    print(json.dumps({"status": "PASS", "implementation": str(implementation),
        "cohort_pnl": out["pnl_net"].tolist(), "trade_costs": costs.values.tolist(),
        "scope": "two-assets/two-legs/holding-1/known-costs"}))


if __name__ == "__main__":
    main()
