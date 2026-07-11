from dataclasses import dataclass, field

import pytest

pd = pytest.importorskip("pandas")

from storage.composite_source import CompositeDataSource
from storage.datasource import DataSource


def _build_series(rows: list[tuple[str, str, float]]) -> "pd.Series":
    frame = pd.DataFrame(rows, columns=["timestamp", "instrument", "value"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    series = frame.set_index(["timestamp", "instrument"])["value"].sort_index()
    series.index = series.index.set_names(["timestamp", "instrument"])
    return series


@dataclass
class CountingSeriesSource(DataSource):
    data: dict[str, "pd.Series"]
    calls: dict[str, int] = field(default_factory=dict)

    def load_column(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.data[name]


def test_composite_source_aligns_auxiliary_columns_to_anchor_and_caches():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-02", "BBB", 20.0),
                    ("2024-01-03", "AAA", 12.0),
                    ("2024-01-03", "BBB", 18.0),
                ]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "price_to_earnings": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-01", "BBB", 5.0),
                    ("2024-01-03", "AAA", 3.0),
                    ("2024-01-03", "BBB", 6.0),
                ]
            )
        }
    )

    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        aliases={"pe": "fundamental.price_to_earnings"},
        joins={"fundamental": {"method": "asof_backward"}},
    )

    pe = source.load_column("pe")

    assert list(pe.index) == list(price.data["close"].index)
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(2.0)
    assert pe.loc[(pd.Timestamp("2024-01-02"), "BBB")] == pytest.approx(5.0)
    assert pe.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)
    assert pe.loc[(pd.Timestamp("2024-01-03"), "BBB")] == pytest.approx(6.0)
    assert price.calls == {"close": 1}
    assert fundamental.calls == {"price_to_earnings": 1}

    close = source.load_column("close")
    again = source.load_column("fundamental.price_to_earnings")

    assert close.equals(price.data["close"])
    assert again.equals(pe)
    assert price.calls == {"close": 1}
    assert fundamental.calls == {"price_to_earnings": 1}


def test_composite_source_supports_exact_alignment():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "price_to_earnings": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-03", "AAA", 3.0),
                ]
            )
        }
    )

    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "exact"},
    )

    pe = source.load_column("fundamental.price_to_earnings")

    assert pd.isna(pe.loc[(pd.Timestamp("2024-01-02"), "AAA")])
    assert pe.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)