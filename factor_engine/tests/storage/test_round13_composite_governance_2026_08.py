"""Round-13 composite governance: P1-05 .. P1-09.

* P1-05: production 下 non-anchor source 必须显式声明 join policy；
  research 才允许默认 asof_backward。
* P1-06: snapshot refresh 失败 → production 抛 SnapshotVerificationError；
  research clear cache + warning（旧 cache 不再被信任）。
* P1-07: CompositeSnapshotBarrier 原子一致性 —— token 漂移触发整体 retry 一次，
  仍漂移则抛 CompositeSnapshotVerificationError。
* P1-08: composite production authority 来自父级 build context，
  child.production=False 不能把 parent production 降成 warning。
* P1-09: COS contract 查询失败 → production fail-closed
  （CompositeContractUnavailableError）；research 才允许 fallback。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from factor_engine.storage.composite_source import (
    CompositeContractUnavailableError,
    CompositeDataSource,
    CompositeJoinPolicyError,
    CompositeSnapshotBarrier,
    CompositeSnapshotVerificationError,
    SnapshotVerificationError,
)
from factor_engine.storage.datasource import DataSource
from factor_engine.storage.factory import DataSourceBuildContext, build_data_source
from tests.storage.test_composite_source import CountingSeriesSource, _build_series


# ---------------------------------------------------------------------------
# helper sources
# ---------------------------------------------------------------------------


@dataclass
class _FailingRefreshSource(DataSource):
    """子源：refresh_snapshot 抛错，token 稳定为 'snap-A'。"""

    data: dict[str, "pd.Series"]
    load_calls: int = 0

    def refresh_snapshot(self, *, force: bool = False):
        raise RuntimeError("refresh boom")

    def snapshot_token(self):
        return "snap-A"

    def temporal_contract(self):
        # R24-084..087: plain panel — generic asof is allowed.
        from factor_engine.storage.datasource import TemporalContract

        return TemporalContract(
            temporal_sensitivity="none",
            snapshot_capability="none",
            join_capability="generic_asof",
        )

    def load_column(self, name: str):
        self.load_calls += 1
        return self.data[name]


@dataclass
class _EraSource(DataSource):
    """子源：带快照 era 的 token；可在指定第几次 load 后推进 era。"""

    data: dict[str, "pd.Series"]
    era: str = "snap-1"
    load_calls: int = 0
    #: 第几次 load_column 之后推进 era（1-based）；None = 永不推进。
    advance_after_load: int | None = None

    def refresh_snapshot(self, *, force: bool = False):
        return self.era

    def snapshot_token(self):
        return self.era

    def temporal_contract(self):
        # R24-084..087: plain panel — generic asof is allowed.
        from factor_engine.storage.datasource import TemporalContract

        return TemporalContract(
            temporal_sensitivity="none",
            snapshot_capability="none",
            join_capability="generic_asof",
        )

    def load_column(self, name: str):
        self.load_calls += 1
        if (
            self.advance_after_load is not None
            and self.load_calls == self.advance_after_load
        ):
            n = int(self.era.split("-")[1]) + 1
            self.era = f"snap-{n}"
        return self.data[name]


@dataclass
class _AlwaysAdvanceTokenSource(DataSource):
    """子源：每次 load_column 都推进 token，模拟持续漂移。"""

    data: dict[str, "pd.Series"]
    load_calls: int = 0

    def refresh_snapshot(self, *, force: bool = False):
        return None

    def snapshot_token(self):
        return f"snap-{self.load_calls}"

    def load_column(self, name: str):
        self.load_calls += 1
        return self.data[name]


def _make_sources():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [("2024-01-02", "AAA", 100.0), ("2024-01-03", "AAA", 101.0)]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "pe": _build_series(
                [("2024-01-01", "AAA", 10.0), ("2024-01-03", "AAA", 11.0)]
            )
        }
    )
    return price, fundamental


# ---------------------------------------------------------------------------
# P1-05: explicit join policy
# ---------------------------------------------------------------------------


def test_production_composite_requires_explicit_join_policy():
    price, fundamental = _make_sources()
    with pytest.raises(CompositeJoinPolicyError):
        CompositeDataSource(
            anchor_source="price",
            anchor_column="close",
            sources={"price": price, "fundamental": fundamental},
            production=True,
        )


def test_research_composite_defaults_to_asof_backward():
    price, fundamental = _make_sources()
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
    )
    assert source._effective_production is False
    assert source.joins["fundamental"].method == "asof_backward"
    assert source.joins["fundamental"].explicit_join is False
    pe = source.load_column("fundamental.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)


def test_production_composite_explicit_join_policy_accepted():
    price, fundamental = _make_sources()
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": {"method": "asof_backward"}},
        production=True,
    )
    assert source.joins["fundamental"].explicit_join is True
    pe = source.load_column("fundamental.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# P1-06: refresh failure
# ---------------------------------------------------------------------------


def test_production_refresh_failure_raises_snapshot_verification_error():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = _FailingRefreshSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
        production=True,
    )
    with pytest.raises(SnapshotVerificationError):
        source.load_column("fundamental.pe")


def test_research_refresh_failure_clears_cache_and_reloads():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = _FailingRefreshSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    pe1 = source.load_column("fundamental.pe")
    assert pe1.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)
    # refresh 每次失败都清缓存 → 第二次 load 必须重读子源，不能命中旧缓存。
    pe2 = source.load_column("fundamental.pe")
    assert pe2.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)
    assert fundamental.load_calls == 2
    assert source._column_cache == {}
    assert source._anchor_index_cache is None


# ---------------------------------------------------------------------------
# P1-07: CompositeSnapshotBarrier atomic coherence
# ---------------------------------------------------------------------------


def test_snapshot_barrier_retries_once_when_token_moves_then_stabilizes():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = _EraSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])},
        advance_after_load=1,
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    pe = source.load_column("fundamental.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)
    # 第一次读期间 token 漂移 → 整体 retry 一次；第二次 load 后稳定。
    assert fundamental.era == "snap-2"
    assert fundamental.load_calls == 2


def test_snapshot_barrier_raises_when_token_keeps_moving():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = _AlwaysAdvanceTokenSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    with pytest.raises(CompositeSnapshotVerificationError):
        source.load_column("fundamental.pe")


def test_snapshot_barrier_ok_when_tokens_stable():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = _EraSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])},
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    barrier = CompositeSnapshotBarrier(source)
    barrier.begin()
    barrier.revalidate()  # token 稳定 → 不抛
    pe = source.load_column("fundamental.pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# P1-08: production authority from parent build context
# ---------------------------------------------------------------------------


def test_parent_production_not_downgraded_by_child_production_false():
    price, fundamental = _make_sources()
    # child 显式 research，parent production=True → 组合仍是 production。
    fundamental.production = False
    price.production = False
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
        production=True,
    )
    assert source._effective_production is True
    # production 权威来自 parent：即使子源 research，缺省 join 仍被拒绝。
    with pytest.raises(CompositeJoinPolicyError):
        CompositeDataSource(
            anchor_source="price",
            anchor_column="close",
            sources={"price": price, "fundamental": fundamental},
            production=True,
        )


def test_factory_injects_production_authority_from_build_context():
    composite = build_data_source(
        {
            "type": "composite",
            "anchor": "price",
            "anchor_column": "close",
            "sources": {
                "price": {"type": "data_access", "dataset": "us_stocks_sip_day_aggs"},
                "ratios": {
                    "type": "data_access",
                    "dataset": "financials_ratios",
                    "production": False,
                },
            },
            "joins": {"ratios": "asof_backward"},
        },
        build_context=DataSourceBuildContext(run_mode="production"),
    )
    # parent production authority 注入 → 组合 production，即使子源 production=False。
    assert composite._effective_production is True
    assert composite.sources["ratios"].production is False


def test_factory_production_requires_explicit_join_policy():
    with pytest.raises(CompositeJoinPolicyError):
        build_data_source(
            {
                "type": "composite",
                "anchor": "price",
                "anchor_column": "close",
                "sources": {
                    "price": {
                        "type": "data_access",
                        "dataset": "us_stocks_sip_day_aggs",
                    },
                    "ratios": {
                        "type": "data_access",
                        "dataset": "financials_ratios",
                    },
                },
            },
            build_context=DataSourceBuildContext(run_mode="production"),
        )


# ---------------------------------------------------------------------------
# P1-09: COS contract lookup failure is fail-closed in production
# ---------------------------------------------------------------------------


def test_cos_contract_lookup_failure_fail_closed_in_production(monkeypatch):
    import data_access.cos_contract as cc

    def boom(dataset: str):
        raise RuntimeError("contract service down")

    monkeypatch.setattr(cc, "get_cos_contract", boom)

    child = SimpleNamespace(dataset="some_dataset", read_mode="panel")
    with pytest.raises(CompositeContractUnavailableError):
        CompositeDataSource._is_pit_sensitive_source(child, production=True)


def test_cos_contract_lookup_failure_research_falls_back(monkeypatch):
    import data_access.cos_contract as cc

    def boom(dataset: str):
        raise RuntimeError("contract service down")

    monkeypatch.setattr(cc, "get_cos_contract", boom)

    child = SimpleNamespace(dataset="some_dataset", read_mode="panel")
    # research 允许 fallback：无法证明 PIT-sensitivity → 按非敏感处理（False）。
    assert CompositeDataSource._is_pit_sensitive_source(child, production=False) is False
