import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vectorbt_qs.mvp.analysis import (
    RiskModelStore,
    analyze_portfolio_exposure,
    export_exposure_dashboard,
    export_industry_exposure_pngs,
    export_style_exposure_pngs,
)


class FakePortfolio:
    def __init__(self, asset_value, value, returns, benchmark_returns):
        self._asset_value = asset_value
        self._value = value
        self._returns = returns
        self._qs_benchmark_returns = benchmark_returns

    def asset_value(self, group_by=False):
        return self._asset_value

    def value(self):
        return self._value

    def returns(self):
        return self._returns


class ExposureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.risk_root = self.root / "risk"
        self.risk_root.mkdir()
        self.data_root = self.root / "lqtp_data"
        (self.data_root / "IndexConstituent").mkdir(parents=True)
        self.dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        self._write_risk_model()
        self._write_benchmark()

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_risk_model(self):
        factors = [
            {"id": "style_size", "type": "continuous"},
            {"id": "industry_x", "type": "dummy"},
            {"id": "industry_y", "type": "dummy"},
        ]
        manifest = {
            "product": "barra_lite",
            "version": "test_v1",
            "factors": factors,
            "files": {
                "exposure": "exposure.parquet",
                "factor_returns": "factor_returns.parquet",
            },
            "quality": {
                "date_min": "2024-01-02",
                "date_max": "2024-01-04",
            },
        }
        (self.risk_root / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        exposure_rows = []
        for date in self.dates.strftime("%Y-%m-%d"):
            exposure_rows.extend(
                [
                    (date, "A", "style_size", 1.0),
                    (date, "A", "industry_x", 1.0),
                    (date, "B", "style_size", -1.0),
                    (date, "B", "industry_y", 1.0),
                ]
            )
        pd.DataFrame(
            exposure_rows,
            columns=["date", "asset", "factor_id", "exposure"],
        ).to_parquet(self.risk_root / "exposure.parquet", index=False)
        return_rows = []
        for date in self.dates.strftime("%Y-%m-%d"):
            return_rows.extend(
                [
                    (date, "style_size", 0.01),
                    (date, "industry_x", 0.0),
                    (date, "industry_y", 0.0),
                ]
            )
        pd.DataFrame(
            return_rows,
            columns=["date", "factor_id", "factor_return"],
        ).to_parquet(self.risk_root / "factor_returns.parquet", index=False)

    def _write_benchmark(self):
        for date in self.dates:
            frame = pd.DataFrame(
                {
                    "TradeDate": [date.date(), date.date()],
                    "IndexSymbol": ["BENCH", "BENCH"],
                    "Symbol": ["A", "B"],
                    "Weight": [50.0, 50.0],
                }
            )
            frame.to_parquet(
                self.data_root / "IndexConstituent" / f"{date.date()}.parquet",
                index=False,
            )

    def test_long_form_aggregation_matches_manual_dot_product(self):
        weights = pd.DataFrame(
            {"A": [0.6, 0.0], "B": [0.4, 1.0]},
            index=self.dates[:2],
        )
        result = RiskModelStore(self.risk_root).aggregate_exposure(weights)

        self.assertAlmostEqual(result.exposure.iloc[0]["style_size"], 0.2)
        self.assertAlmostEqual(result.exposure.iloc[0]["industry_x"], 0.6)
        self.assertAlmostEqual(result.exposure.iloc[0]["industry_y"], 0.4)
        self.assertAlmostEqual(result.exposure.iloc[1]["style_size"], -1.0)
        self.assertTrue((result.coverage["coverage_weight"] == 1.0).all())

    def test_missing_asset_is_reported_instead_of_filled_as_zero(self):
        weights = pd.DataFrame(
            {"A": [0.8], "UNKNOWN": [0.2]},
            index=self.dates[:1],
        )
        result = RiskModelStore(self.risk_root).aggregate_exposure(weights)

        self.assertAlmostEqual(result.coverage.iloc[0]["coverage_weight"], 0.8)
        self.assertAlmostEqual(result.coverage.iloc[0]["unknown_weight"], 0.2)
        self.assertEqual(result.coverage.iloc[0]["missing_asset_count"], 1)
        with self.assertRaisesRegex(ValueError, "缺少风险暴露"):
            RiskModelStore(self.risk_root).aggregate_exposure(
                weights,
                missing_policy="error",
            )

    def test_portfolio_benchmark_active_exposure_and_attribution(self):
        asset_value = pd.DataFrame(
            {"A": [60.0, 60.0, 60.0], "B": [40.0, 40.0, 40.0]},
            index=self.dates,
        )
        value = pd.Series(100.0, index=self.dates)
        returns = pd.Series([0.0, 0.01, 0.02], index=self.dates)
        benchmark_returns = pd.Series([np.nan, 0.005, 0.01], index=self.dates)
        pf = FakePortfolio(asset_value, value, returns, benchmark_returns)

        result = analyze_portfolio_exposure(
            pf,
            risk_model_root=self.risk_root,
            benchmark_index="BENCH",
            benchmark_data_root=self.data_root,
            min_portfolio_coverage=1.0,
            min_benchmark_coverage=1.0,
            include_attribution=True,
        )

        self.assertAlmostEqual(result.portfolio_exposure.iloc[0]["style_size"], 0.2)
        self.assertAlmostEqual(result.benchmark_exposure.iloc[0]["style_size"], 0.0)
        self.assertAlmostEqual(result.active_exposure.iloc[0]["style_size"], 0.2)
        self.assertAlmostEqual(
            result.factor_contribution.iloc[1]["style_size"],
            0.002,
        )
        self.assertAlmostEqual(result.unexplained_return.iloc[1], 0.008)
        self.assertIs(pf._qs_exposure_analysis, result)
        output = result.export(self.root / "output")
        self.assertTrue((output / "portfolio_style_exposure.parquet").is_file())
        self.assertTrue((output / "active_style_exposure.parquet").is_file())
        self.assertTrue((output / "coverage.csv").is_file())
        dashboard = export_exposure_dashboard(result, output)
        self.assertTrue(dashboard.is_file())
        pngs = export_style_exposure_pngs(result, output)
        self.assertEqual(set(pngs), {"portfolio", "benchmark", "active"})
        self.assertTrue(all(path.is_file() for path in pngs.values()))
        industry_pngs = export_industry_exposure_pngs(result, output, top_n=2)
        self.assertEqual(
            set(industry_pngs),
            {
                "active_heatmap",
                "latest_active",
                "portfolio_vs_benchmark",
                "allocation",
            },
        )
        self.assertTrue(all(path.is_file() for path in industry_pngs.values()))


if __name__ == "__main__":
    unittest.main()
