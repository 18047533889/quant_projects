from dataclasses import dataclass, field

import pytest

pd = pytest.importorskip("pandas")

from factor_engine.storage.composite_source import CompositeDataSource
from factor_engine.storage.datasource import DataSource


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

    def snapshot_token(self):
        # R24-079: a stable test token makes production composites verifiable.
        return "test-snap"

    def temporal_contract(self):
        # R24-084..087: a plain panel with no knowledge-time is safe for generic
        # asof — declared, never inferred from the (absent) dataset name.
        from factor_engine.storage.datasource import TemporalContract

        return TemporalContract(
            temporal_sensitivity="none",
            snapshot_capability="token",
            join_capability="generic_asof",
        )


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


@dataclass
class _BatchingSeriesSource(DataSource):
    """带 load_columns 计数的子源（模拟 DataAccessSource 的批量读）。"""

    data: dict[str, "pd.Series"]
    load_columns_calls: list[list[str]] = field(default_factory=list)

    def load_column(self, name: str):
        self.load_columns_calls.append([name])
        return self.data[name]

    def load_columns(self, names: list[str]):
        self.load_columns_calls.append(list(names))
        return {n: self.data[n] for n in names}


def test_composite_batches_same_source_columns_into_one_load():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 10.0), ("2024-01-03", "AAA", 12.0)])}
    )
    fundamental = _BatchingSeriesSource(
        {
            "pe": _build_series([("2024-01-01", "AAA", 2.0), ("2024-01-03", "AAA", 3.0)]),
            "pb": _build_series([("2024-01-01", "AAA", 4.0), ("2024-01-03", "AAA", 5.0)]),
        }
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": {"method": "asof_backward"}},
    )

    batch = source.load_columns(["fundamental.pe", "fundamental.pb"])

    assert set(batch) == {"fundamental.pe", "fundamental.pb"}
    # 同源多列 → 一次 load_columns（而非逐列两次）
    fundamental_calls = [c for c in fundamental.load_columns_calls if c != ["close"]]
    assert fundamental_calls == [["pe", "pb"]]


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


def test_composite_collect_join_reports():
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
        joins={"fundamental": "asof_backward"},
    )
    source.load_column("fundamental.price_to_earnings")
    reports = source.collect_join_reports()
    assert len(reports) == 1
    assert reports[0]["source"] == "fundamental"
    assert reports[0]["matched_rows"] == 2
    assert reports[0]["value_valid_rows"] == 2
    assert reports[0]["key_matched_rows"] == 2
    assert reports[0]["key_unmatched_rows"] == 0
    assert reports[0]["value_null_rows"] == 0
    assert reports[0]["method"] == "asof_backward"


# ---------------------------------------------------------------------------
# Review-8 #474: join report must distinguish key-miss from value-null.
# ---------------------------------------------------------------------------
def test_composite_join_report_distinguishes_key_miss_from_value_null():
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
    # Jan-1 无该 key 事件；Jan-2 有 key 但值为 null。
    fundamental = CountingSeriesSource(
        {
            "pe": _build_series(
                [
                    ("2024-01-02", "AAA", float("nan")),
                    ("2024-01-03", "AAA", 3.0),
                ]
            )
        }
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    source.load_column("fundamental.pe")
    report = source.collect_join_reports()[0]
    # key 全部命中（anchor 2 行，asof 都找到了源行）
    assert report["key_matched_rows"] == 2
    assert report["key_unmatched_rows"] == 0
    # 但 Jan-2 的源值本身是 null → value_valid 只有 1
    assert report["value_valid_rows"] == 1
    assert report["value_null_rows"] == 1
    # 兼容别名：matched == value_valid
    assert report["matched_rows"] == 1


def test_composite_join_report_counts_key_unmatched():
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
    # 源数据晚于锚点 → asof backward 无法命中任何 key。
    fundamental = CountingSeriesSource(
        {"pe": _build_series([("2024-01-05", "AAA", 9.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    source.load_column("fundamental.pe")
    report = source.collect_join_reports()[0]
    assert report["key_matched_rows"] == 0
    assert report["key_unmatched_rows"] == 2
    assert report["value_null_rows"] == 2


# ---------------------------------------------------------------------------
# Review-8 #475: asof tolerance must reject negative durations.
# ---------------------------------------------------------------------------
def test_composite_join_rejects_negative_tolerance():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 10.0)])}
    )
    fundamental = CountingSeriesSource(
        {"pe": _build_series([("2024-01-01", "AAA", 2.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": {"method": "asof_backward", "tolerance": "-1d"}},
    )
    with pytest.raises(ValueError, match=">= 0"):
        source.load_column("fundamental.pe")


# ---------------------------------------------------------------------------
# Review-8 #476: asof source duplicate (timestamp, instrument) must fail closed.
# ---------------------------------------------------------------------------
def test_composite_asof_duplicate_source_key_fails_closed():
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
    dup = pd.DataFrame(
        [
            ("2024-01-01", "AAA", 2.0),
            ("2024-01-01", "AAA", 3.0),
        ],
        columns=["timestamp", "instrument", "value"],
    )
    dup["timestamp"] = pd.to_datetime(dup["timestamp"])
    dup_series = dup.set_index(["timestamp", "instrument"])["value"].sort_index()
    dup_series.index = dup_series.index.set_names(["timestamp", "instrument"])
    fundamental = CountingSeriesSource({"pe": dup_series})
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        source.load_column("fundamental.pe")