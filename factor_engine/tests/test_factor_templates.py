from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

pd = pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from runtime.real_data_factor_smoke import _build_factors
from storage.datasource import DataSource


@dataclass
class InMemorySeriesSource(DataSource):
    data: dict[str, "pd.Series"]

    def load_column(self, name: str):
        return self.data[name]


DATASET_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fundamentals/balance_sheet", ("total_assets", "total_liabilities", "total_equity")),
    (
        "fundamentals/cash_flow_statement",
        ("net_cash_from_operating_activities", "net_cash_from_investing_activities", "net_income"),
    ),
    ("fundamentals/financials_ratios", ("price_to_earnings", "return_on_equity", "debt_to_equity")),
    (
        "fundamentals/income_statement",
        ("revenue", "operating_income", "net_income_loss_attributable_common_shareholders"),
    ),
    ("fundamentals/short_interest", ("short_interest", "days_to_cover", "avg_daily_volume")),
    ("fundamentals/short_volume", ("short_volume", "total_volume", "short_volume_ratio")),
    ("fundamentals/stocks_floats", ("free_float", "free_float_percent", "outstanding_shares")),
    ("us_stocks_sip/day_aggs_v1", ("close", "volume", "transactions")),
    ("us_stocks_sip/minute_aggs_v1", ("close", "volume", "transactions")),
    ("us_stocks_sip/quotes_v1", ("bid_price", "ask_price", "bid_size")),
    ("us_stocks_sip/trades_v1", ("price", "size", "exchange")),
)


def _build_mock_series(columns: tuple[str, ...]) -> dict[str, "pd.Series"]:
    ts = pd.date_range("2024-01-01", periods=8, freq="D")
    instruments = ["AAA", "BBB", "CCC", "DDD"]
    idx = pd.MultiIndex.from_product([ts, instruments], names=["timestamp", "instrument"])

    data: dict[str, pd.Series] = {}
    for i, name in enumerate(columns):
        # 构造平稳上升序列，确保时序和截面算子都可计算
        values = [(j + 1) * (i + 1) + (k + 1) * 0.1 for j in range(len(ts)) for k in range(len(instruments))]
        data[name] = pd.Series(values, index=idx)
    return data


@pytest.mark.parametrize("dataset_name,columns", DATASET_TEMPLATES, ids=[x[0] for x in DATASET_TEMPLATES])
def test_factor_templates_for_each_dataset_class(dataset_name: str, columns: tuple[str, ...]):
    factors = _build_factors(list(columns), prefix=dataset_name.replace("/", "_"))
    source = InMemorySeriesSource(data=_build_mock_series(columns))
    engine = FactorEngine(backend=PandasBackend(), data_source=source)

    assert len(factors) >= 3
    for factor in factors:
        out = engine.run(factor)
        result = out["result"]

        assert isinstance(result, pd.Series)
        assert result.index.names == ["timestamp", "instrument"]
        assert len(result) > 0
        assert result.notna().any()
