# -*- coding: utf-8 -*-
"""R10 #43 / #44 / #55 / #56.

* R10 #43 — composite exact-join report ``key_matched_rows`` counts anchor keys
  that actually exist in the source index, not the full anchor length.
* R10 #44 — CompositeDataSource invalidates ``_column_cache`` and
  ``_anchor_index_cache`` when a child source's snapshot version changes.
* R10 #55 — SourceRef transform params are validated for dtype / range /
  choices (``SourceTransformParamSpec``), not just parameter names.
* R10 #56 — an unknown / undeclared SourceRef transform is rejected in
  production mode; research / compat keeps the constructible behavior.
"""
from __future__ import annotations

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
class _SnapshotAwareSource(DataSource):
    """子源：可变 ``data_snapshot_id`` + 可更新数据（模拟 DataAccessSource）。

    DataAccessSource 暴露只读 ``data_snapshot_id`` 属性；这里用可变的
    ``snapshot_id`` 让测试能模拟「子源快照版本变化」。
    """

    data: dict[str, "pd.Series"]
    calls: dict[str, int] = field(default_factory=dict)
    snapshot_id: str | None = None

    @property
    def data_snapshot_id(self) -> str | None:
        return self.snapshot_id

    def load_column(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.data[name]

    def set_snapshot(
        self,
        snapshot_id: str | None,
        data: dict[str, "pd.Series"] | None = None,
    ) -> None:
        self.snapshot_id = snapshot_id
        if data is not None:
            self.data = data


# ---------------------------------------------------------------------------
# R10 #43 — exact-join key_matched statistics
# ---------------------------------------------------------------------------

def test_exact_join_key_missing_from_source_reports_zero_matched():
    price = _SnapshotAwareSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        }
    )
    # 源只含 anchor 之外的 key：exact 一个都不命中。
    fundamental = _SnapshotAwareSource(
        {"pe": _build_series([("2024-01-01", "AAA", 2.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "exact"},
    )
    source.load_column("fundamental.pe")
    report = source.collect_join_reports()[0]
    # anchor 有 2 行，但源索引中没有任何 anchor key → key_matched 必须是 0，
    # 而不是 anchor 全长 2。
    assert report["anchor_rows"] == 2
    assert report["key_matched_rows"] == 0
    assert report["key_unmatched_rows"] == 2
    assert report["value_valid_rows"] == 0
    assert report["value_null_rows"] == 2


def test_exact_join_key_matched_counts_only_present_source_keys():
    price = _SnapshotAwareSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        }
    )
    # 源含 2024-01-03（anchor key）与 2024-01-01（非 anchor key）。
    fundamental = _SnapshotAwareSource(
        {
            "pe": _build_series(
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
    source.load_column("fundamental.pe")
    report = source.collect_join_reports()[0]
    # anchor 2 行，只有 2024-01-03 在源中真实存在 → key_matched=1。
    assert report["key_matched_rows"] == 1
    assert report["key_unmatched_rows"] == 1
    assert report["value_valid_rows"] == 1
    assert report["value_null_rows"] == 1
    # 兼容别名：matched == value-valid。
    assert report["matched_rows"] == 1


# ---------------------------------------------------------------------------
# R10 #44 — child snapshot change invalidates composite caches
# ---------------------------------------------------------------------------

def test_composite_cache_invalidated_when_child_snapshot_changes():
    price = _SnapshotAwareSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        },
        snapshot_id="s1",
    )
    aux = _SnapshotAwareSource(
        {
            "pe": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-03", "AAA", 3.0),
                ]
            )
        },
        snapshot_id="s1",
    )
    composite = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "aux": aux},
        joins={"aux": "asof_backward"},
    )

    first = composite.load_column("aux.pe")
    assert first.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)
    assert aux.calls["pe"] == 1

    # 子源快照变化 + 数据更新：组合缓存必须重建，返回新值。
    aux.set_snapshot(
        "s2",
        {"pe": _build_series(
            [
                ("2024-01-01", "AAA", 2.0),
                ("2024-01-03", "AAA", 9.0),
            ]
        )},
    )
    second = composite.load_column("aux.pe")
    assert second.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(9.0)
    assert aux.calls["pe"] == 2  # 缓存被清除，重新读了子源


def test_composite_anchor_index_cache_invalidated_when_child_snapshot_changes():
    price = _SnapshotAwareSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        },
        snapshot_id="s1",
    )
    aux = _SnapshotAwareSource(
        {"pe": _build_series([("2024-01-01", "AAA", 2.0)])},
        snapshot_id="s1",
    )
    composite = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "aux": aux},
        joins={"aux": "asof_backward"},
    )
    composite.load_column("aux.pe")
    assert price.calls["close"] == 1

    # anchor 源快照变化：锚点索引缓存也要失效重建。
    price.set_snapshot(
        "s2",
        {"close": _build_series(
            [
                ("2024-01-02", "AAA", 10.0),
                ("2024-01-03", "AAA", 12.0),
                ("2024-01-04", "AAA", 14.0),
            ]
        )},
    )
    composite.load_column("aux.pe")
    assert price.calls["close"] == 2  # 锚点索引重建


def test_composite_cache_hit_without_snapshot_change_does_not_reread():
    price = _SnapshotAwareSource(
        {"close": _build_series([("2024-01-02", "AAA", 10.0)])},
        snapshot_id="s1",
    )
    aux = _SnapshotAwareSource(
        {"pe": _build_series([("2024-01-01", "AAA", 2.0)])},
        snapshot_id="s1",
    )
    composite = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "aux": aux},
        joins={"aux": "asof_backward"},
    )
    composite.load_column("aux.pe")
    assert aux.calls["pe"] == 1
    # 快照未变：第二次 load 命中缓存，不再读子源。
    again = composite.load_column("aux.pe")
    assert aux.calls["pe"] == 1
    assert again.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# R10 #55 — transform param dtype / range / choices validation
# ---------------------------------------------------------------------------

def test_source_ref_transform_param_out_of_range_raises():
    from factor_engine.api.source_ref import make_source_ref

    with pytest.raises(ValueError, match=">= 1"):
        make_source_ref(
            "DailyBar", "Close",
            transform="financial_lag", transform_params={"quarters": 0},
        )
    with pytest.raises(ValueError, match=">= 1"):
        make_source_ref(
            "StockMinuteBar", "Close",
            transform="minute_bar", transform_params={"period": 0, "index": 0},
        )
    with pytest.raises(ValueError, match=">= 0"):
        make_source_ref(
            "StockMinuteBar", "Close",
            transform="minute_bar", transform_params={"period": 5, "index": -1},
        )


def test_source_ref_transform_param_wrong_dtype_raises():
    from factor_engine.api.source_ref import make_source_ref

    with pytest.raises(ValueError, match="integer"):
        make_source_ref(
            "DailyBar", "Close",
            transform="financial_lag", transform_params={"quarters": "2"},
        )
    with pytest.raises(ValueError, match="integer"):
        make_source_ref(
            "StockMinuteBar", "Close",
            transform="minute_bar", transform_params={"period": 5.5, "index": 0},
        )
    # bool 不是真实整数（Review-8 #469 精神）。
    with pytest.raises(ValueError, match="integer"):
        make_source_ref(
            "StockMinuteBar", "Close",
            transform="minute_bar", transform_params={"period": True, "index": 0},
        )
    with pytest.raises(ValueError, match="integer"):
        make_source_ref(
            "StockMinuteBar", "Close",
            transform="minute_resample", transform_params={"period": 1.9},
        )


def test_source_ref_transform_valid_params_accepted():
    from factor_engine.api.source_ref import make_source_ref

    spec = make_source_ref(
        "StockMinuteBar", "Close",
        transform="minute_bar", transform_params={"period": 5, "index": 0},
    )
    assert spec.transform_params_dict() == {"index": 0, "period": 5}
    # 整数浮点（2.0）与真整数等价，接受。
    spec2 = make_source_ref(
        "DailyBar", "Close",
        transform="financial_lag", transform_params={"quarters": 2.0},
    )
    assert spec2.transform_params_dict() == {"quarters": 2.0}


def test_source_transform_param_spec_choices_validation():
    from factor_engine.api.source_ref import SourceTransformParamSpec

    spec = SourceTransformParamSpec(dtype=str, choices=("open", "close"))
    spec.validate("side", "open")  # 合法白名单值不抛
    with pytest.raises(ValueError, match="one of"):
        spec.validate("side", "high")


# ---------------------------------------------------------------------------
# R10 #56 — unknown transform rejected in production
# ---------------------------------------------------------------------------

def test_unknown_transform_constructible_in_research():
    from factor_engine.api.source_ref import make_source_ref

    # research / compat 保持现状：未知变换可构造（opt-in strictness）。
    spec = make_source_ref(
        "DailyBar", "Close",
        transform="bogus_transform", transform_params={"x": 1},
    )
    assert spec.transform == "bogus_transform"
    assert spec.transform_params_dict() == {"x": 1}


def test_unknown_transform_rejected_in_production():
    from factor_engine.api.source_ref import make_source_ref

    with pytest.raises(ValueError, match="not a declared production transform"):
        make_source_ref(
            "DailyBar", "Close",
            transform="bogus_transform", transform_params={"x": 1},
            production=True,
        )
    # with_transform 路径同样在 production 拒绝未知变换。
    base = make_source_ref("DailyBar", "Close", production=True)
    with pytest.raises(ValueError, match="not a declared production transform"):
        base.with_transform("bogus_transform", x=1)


def test_known_transform_bad_param_rejected_in_production():
    from factor_engine.api.source_ref import make_source_ref

    with pytest.raises(ValueError, match=">= 1"):
        make_source_ref(
            "DailyBar", "Close",
            transform="financial_lag", transform_params={"quarters": 0},
            production=True,
        )


def test_transform_source_col_production_rejects_unknown_transform():
    from factor_engine.api.columns import col
    from factor_engine.api.source_ref import encode_source_ref, make_source_ref, transform_source_col

    ref = col(encode_source_ref(make_source_ref("StockMinuteBar", "Close")))
    with pytest.raises(ValueError, match="not a declared production transform"):
        transform_source_col(ref, "bogus_transform", production=True, x=1)
    # research 默认：构造成功。
    out = transform_source_col(ref, "bogus_transform", x=1)
    assert "__fe_source_ref_v1__" in str(out.name)
