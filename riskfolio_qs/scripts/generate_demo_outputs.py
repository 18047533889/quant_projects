from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from riskfolio_qs.adapters.mock_adapter import MockInputAdapter
from riskfolio_qs.constraints.constraint_builder import ConstraintBuilder
from riskfolio_qs.optimizers.optimizer_router import OptimizerRouter
from riskfolio_qs.optimizers.portfolio_optimizer import PortfolioOptimizer
from riskfolio_qs.runners.pipeline import OptimizationPipeline
from riskfolio_qs.smoothers.signal_smoother import SignalSmoother
from riskfolio_qs.adapters.output_adapter import OutputAdapter


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    frame.to_csv(path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "artifacts" / "demo_run"
    _ensure_dir(output_root)

    scenarios = [
        {
            "name": "rank_topn_long_only",
            "alpha_is_absolute_return": False,
            "benchmark_name": "topn_long_only_equal_weight",
        },
        {
            "name": "absolute_classic_sharpe",
            "alpha_is_absolute_return": True,
            "benchmark_name": "classic_sharpe",
        },
    ]

    manifest = []

    for scenario in scenarios:
        scenario_dir = output_root / scenario["name"]
        _ensure_dir(scenario_dir)

        adapter = MockInputAdapter(alpha_is_absolute_return=scenario["alpha_is_absolute_return"])
        bundle = adapter.build_bundle()

        smoother = SignalSmoother()
        output_adapter = OutputAdapter()
        router = OptimizerRouter(alpha_is_absolute_return=scenario["alpha_is_absolute_return"])
        optimizer = PortfolioOptimizer(
            smoother=smoother,
            output_adapter=output_adapter,
            constraint_builder=ConstraintBuilder(),
        )

        smoothed_alpha, diagnostics = smoother.transform(bundle.alpha)
        spec = router.resolve(scenario["benchmark_name"])
        output = optimizer.optimize(bundle=bundle, benchmark_spec=spec, smoothed_alpha=smoothed_alpha)

        _save_frame(bundle.alpha, scenario_dir / "alpha.csv")
        _save_frame(bundle.market, scenario_dir / "market.csv")
        if bundle.benchmark is not None:
            _save_frame(bundle.benchmark, scenario_dir / "benchmark.csv")
        if bundle.prev_positions is not None:
            _save_frame(bundle.prev_positions, scenario_dir / "prev_positions.csv")
        if bundle.tradable is not None:
            _save_frame(bundle.tradable, scenario_dir / "tradable.csv")

        _save_frame(smoothed_alpha, scenario_dir / "alpha_smoothed.csv")
        _save_frame(diagnostics, scenario_dir / "signal_diagnostics.csv")
        _save_frame(output.target_positions, scenario_dir / "target_positions.csv")
        _save_frame(output.summary, scenario_dir / "summary.csv")
        _save_frame(output.metadata, scenario_dir / "metadata.csv")
        _save_frame(output.trades, scenario_dir / "trades.csv")

        manifest.append(
            {
                "scenario": scenario["name"],
                "benchmark_name": scenario["benchmark_name"],
                "alpha_is_absolute_return": scenario["alpha_is_absolute_return"],
                "output_dir": str(scenario_dir),
                "target_positions_shape": list(output.target_positions.shape),
                "summary_shape": list(output.summary.shape),
                "trade_rows": int(output.trades.shape[0]),
            }
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
