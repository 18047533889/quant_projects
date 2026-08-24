# -*- coding: utf-8 -*-
"""R30-P1-001/011/025/017/002 —— adapter / join_cost / data_quality / policy / formats_contract。

纯 Python + 临时文件：fake store + 临时 parquet/csv。
覆盖：
    - SourceCapabilities / choose_backend 选择正确；
    - FormatSourceAdapter 包装既有 adapter（假 format adapter + 真 formats.py）；
    - JoinCostPlanner 的 small-dimension-first / minute-preaggregate 决策；
    - DataQualityService 用 fake store + 临时 parquet/csv：OHLC 不一致 → BLOCK；
      正常 → PASS；negative volume → BLOCK；qualify_source；
    - PolicyManifest digest 稳定 + 字段变化 digest 变；from_config / as_json /
      into_context_dict；
    - CSVProductionContract validate 显式 schema 通过/不通过（改列 → 报问题）。
"""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from data_access.r30.adapter import (
    BackendAdapter,
    FormatSourceAdapter,
    SourceCapabilities,
    choose_backend,
)
from data_access.r30.join_cost import JoinCostPlanner, JoinInput
from data_access.r30.data_quality import (
    BLOCK,
    PASS,
    WARN,
    DataQualityService,
    qualify_source,
)
from data_access.r30.policy import PolicyManifest
from data_access.r30.formats_contract import (
    CSVProductionContract,
    build_csv_contract_from_table,
)


# ---------------------------------------------------------------------------
# fake store / handle（DataQualityService 测试用）
# ---------------------------------------------------------------------------

class _FakeHandle:
    def __init__(self, table, snapshot=None):
        self._table = table
        self.snapshot = snapshot

    def to_arrow(self):
        return self._table


class _FakeStore:
    def __init__(self, table, snapshot=None):
        self._table = table
        self._snapshot = snapshot

    def read(self, dataset, **params):
        return _FakeHandle(self._table, self._snapshot)


# ---------------------------------------------------------------------------
# adapter —— SourceCapabilities / choose_backend
# ---------------------------------------------------------------------------

def test_source_capabilities_subset_and_to_dict():
    a = SourceCapabilities(projection_pushdown=True, filter_pushdown=True)
    b = SourceCapabilities(projection_pushdown=True, filter_pushdown=True, streaming=True)
    assert a.subset(b)
    assert not b.subset(a)          # b 的 streaming 不是 a 的
    assert a.subset(a)
    d = a.to_dict()
    assert d["projection_pushdown"] is True
    assert d["asof"] is False
    assert set(d) >= {"projection_pushdown", "filter_pushdown", "asof", "remote"}


def test_choose_backend_min_cost_satisfying():
    # asof + projection + filter：duckdb 与 polars 都满足；cost 选最小
    required = SourceCapabilities(projection_pushdown=True, filter_pushdown=True, asof=True)
    costs = {"duckdb": 3.0, "polars": 1.0, "pyarrow": 2.0}
    assert choose_backend(required, costs) == "polars"

    # 只留 duckdb 满足 transaction+snapshot → 选 duckdb
    required2 = SourceCapabilities(transaction=True, snapshot=True)
    costs2 = {"duckdb": 2.0, "polars": 1.0, "pyarrow": 1.0}
    assert choose_backend(required2, costs2) == "duckdb"


def test_choose_backend_returns_none_when_unsatisfied():
    # 需要 asof+snapshot+transaction，而 costs 里没有 duckdb → 无满足 → None
    required = SourceCapabilities(asof=True, snapshot=True, transaction=True)
    costs = {"polars": 1.0, "pyarrow": 1.0, "local": 1.0}
    assert choose_backend(required, costs) is None


def test_backend_adapter_caps():
    duck = BackendAdapter("duckdb").capabilities()
    assert duck.asof is True and duck.projection_pushdown and duck.streaming
    assert duck.write is True
    local = BackendAdapter("local").capabilities()
    assert local.remote is False
    cos = BackendAdapter("cos").capabilities()
    assert cos.remote is True


# ---------------------------------------------------------------------------
# adapter —— FormatSourceAdapter 包装
# ---------------------------------------------------------------------------

class _FakeFormatAdapter:
    """最小假 FormatAdapter：只有 data_format / uses_duckdb。"""

    def __init__(self, fmt, uses_duckdb=True):
        self.data_format = fmt
        self.uses_duckdb = uses_duckdb
        self.spec = None


def test_format_source_adapter_caps_by_format():
    fsa_csv = FormatSourceAdapter(_FakeFormatAdapter("csv"))
    caps = fsa_csv.capabilities()
    assert caps.streaming is True
    assert caps.asof is False
    assert caps.partition_pruning is False

    fsa_pq = FormatSourceAdapter(_FakeFormatAdapter("parquet"))
    caps_pq = fsa_pq.capabilities()
    assert caps_pq.partition_pruning is True
    assert caps_pq.streaming is False
    assert caps_pq.projection_pushdown is True

    fsa_arrow = FormatSourceAdapter(_FakeFormatAdapter("arrow", uses_duckdb=False))
    caps_arrow = fsa_arrow.capabilities()
    assert caps_arrow.streaming is True
    assert caps_arrow.asof is False


def test_format_source_adapter_wraps_real_format_adapter():
    from data_access.read.formats import get_format_adapter

    adapter = get_format_adapter("csv")
    fsa = FormatSourceAdapter(adapter)
    assert fsa.capabilities().streaming is True
    resolved = fsa.resolve({"row_count": 100, "byte_size": 1024})
    assert resolved["format"] == "csv"
    assert resolved["uses_duckdb"] is True
    est = fsa.estimate(resolved)
    assert est["estimated_rows"] == 100
    assert fsa.healthcheck() is True


# ---------------------------------------------------------------------------
# join_cost —— JoinCostPlanner
# ---------------------------------------------------------------------------

def test_join_planner_small_dimension_first():
    planner = JoinCostPlanner()
    left = JoinInput(relation="bars", row_count=1_000_000, byte_size=500_000_000)
    right = JoinInput(relation="universe", row_count=5_000, byte_size=5_000_000)
    d = planner.plan(left, right)
    assert d.join_order == ["universe", "bars"]
    assert d.build_side == "universe"


def test_join_planner_minute_preaggregate():
    planner = JoinCostPlanner()
    left = JoinInput(
        relation="minute_bars", row_count=10_000_000,
        byte_size=8_000_000_000, frequency="minute",
    )
    right = JoinInput(
        relation="fundamentals", row_count=10_000,
        byte_size=10_000_000, frequency="quarterly",
    )
    d = planner.plan(left, right)
    assert d.pre_aggregate is True

    # 右表不是季度/事件 → 不预聚合
    right2 = JoinInput(
        relation="daily_facts", row_count=10_000,
        byte_size=10_000_000, frequency="daily",
    )
    d2 = planner.plan(left, right2)
    assert d2.pre_aggregate is False


def test_join_planner_filter_before_join_and_backend():
    planner = JoinCostPlanner()
    left = JoinInput(
        relation="big", row_count=1_000_000, byte_size=100_000_000, selectivity=0.2
    )
    right = JoinInput(
        relation="small", row_count=10_000, byte_size=1_000_000, selectivity=1.0
    )
    d = planner.plan(left, right, asof=True)
    assert "big" in d.filter_before_join
    assert "small" not in d.filter_before_join
    assert d.backend == "duckdb"
    assert d.materialize is True        # min(100MB, 1MB)=1MB < 256MB
    assert d.build_side == "small"

    d_noduck = planner.plan(left, right)
    assert d_noduck.backend == "polars"


def test_join_cost_estimate():
    planner = JoinCostPlanner()
    left = JoinInput(
        relation="a", row_count=100_000, byte_size=8_000_000,
        selectivity=1.0, unique_keys=1_000,
    )
    right = JoinInput(
        relation="b", row_count=50_000, byte_size=4_000_000,
        selectivity=0.5, unique_keys=500,
    )
    d = planner.plan(left, right)
    est = planner.estimate(left, right, d)
    assert est["estimated_rows"] > 0
    assert est["estimated_bytes"] > 0
    assert est["build_side"] == "b"
    assert est["backend"] == "polars"
    assert est["join_order"] == ["b", "a"]
    assert d.to_dict()["join_order"] == ["b", "a"]


# ---------------------------------------------------------------------------
# data_quality —— DataQualityService（fake store + 临时 parquet/csv）
# ---------------------------------------------------------------------------

def test_dq_ohlc_inconsistency_block(tmp_path):
    table = pa.table({
        "date": ["2024-01-02", "2024-01-03"],
        "high": [10.0, 8.0],
        "low": [11.0, 7.0],          # row0: high < low
        "open": [9.5, 7.2],
        "close": [9.0, 7.5],
        "volume": [100, 200],
    })
    p = tmp_path / "ohlc_bad.parquet"
    pq.write_table(table, p)
    stored = pq.read_table(p)

    result = DataQualityService().run(_FakeStore(stored), "ohlc_bad")
    assert result.severity == BLOCK
    assert result.ok() is False
    assert result.checks.get("ohlc") == BLOCK
    assert any("high < low" in m for m in result.problems)


def test_dq_normal_pass(tmp_path):
    table = pa.table({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "high": [10.0, 11.0, 12.0],
        "low": [9.0, 10.0, 11.0],
        "open": [9.2, 10.2, 11.2],
        "close": [9.5, 10.5, 11.5],
        "volume": [100, 200, 300],
        "currency": ["USD", "USD", "USD"],
    })
    p = tmp_path / "ok.csv"
    pa_csv.write_csv(table, p)
    stored = pa_csv.read_csv(p)

    result = DataQualityService().run(_FakeStore(stored), "ok")
    assert result.severity == PASS
    assert result.ok() is True


def test_dq_negative_volume_block(tmp_path):
    table = pa.table({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "volume": [100, -5, 200],
    })
    p = tmp_path / "negvol.csv"
    pa_csv.write_csv(table, p)
    stored = pa_csv.read_csv(p)

    result = DataQualityService().run(_FakeStore(stored), "negvol")
    assert result.severity == BLOCK
    assert result.checks.get("negative_volume") == BLOCK


def test_dq_unknown_currency_warn(tmp_path):
    table = pa.table({
        "date": ["2024-01-02", "2024-01-03"],
        "close": [1.5, 2.5],
        "currency": ["USD", "NOTACURRENCY"],
    })
    p = tmp_path / "ccy.csv"
    pa_csv.write_csv(table, p)
    stored = pa_csv.read_csv(p)

    result = DataQualityService().run(_FakeStore(stored), "ccy")
    assert result.severity == WARN
    assert result.checks.get("currency_enum") == WARN


def test_dq_qualify_source(tmp_path):
    table = pa.table({
        "date": ["2024-01-02", "2024-01-03"],
        "high": [10.0, 11.0],
        "low": [9.0, 10.0],
        "close": [9.5, 10.5],
    })
    p = tmp_path / "q.csv"
    pa_csv.write_csv(table, p)
    stored = pa_csv.read_csv(p)

    assert qualify_source(_FakeStore(stored), "q") == PASS
    assert DataQualityService().qualify_source(_FakeStore(stored), "q") == PASS


# ---------------------------------------------------------------------------
# policy —— PolicyManifest
# ---------------------------------------------------------------------------

def test_policy_manifest_digest_stable_and_changes():
    m1 = PolicyManifest(
        "v1", {"a": {"role": "trader"}}, {"d1": "restricted"}, {"f1": "read"}
    )
    m2 = PolicyManifest(
        "v1", {"a": {"role": "trader"}}, {"d1": "restricted"}, {"f1": "read"}
    )
    assert m1.digest() == m2.digest()          # 同字段 → 同 digest
    assert isinstance(m1.digest(), str) and len(m1.digest()) == 64

    m3 = PolicyManifest(
        "v1", {"a": {"role": "admin"}}, {"d1": "restricted"}, {"f1": "read"}
    )
    assert m3.digest() != m1.digest()          # 字段变化 → digest 变


def test_policy_manifest_from_config_json_context():
    m = PolicyManifest.from_config(
        "1.0",
        [
            {"principal_id": "p1", "role": "trader"},
            {"principal_id": "p2", "role": "pm"},
        ],
        {"daily_bars": "restricted"},
        {"momentum": "read_only"},
    )
    assert m.principal_mappings == {
        "p1": {"principal_id": "p1", "role": "trader"},
        "p2": {"principal_id": "p2", "role": "pm"},
    }
    d = m.to_dict()
    assert d["policy_version"] == "1.0"
    assert d["digest"] == m.digest()
    assert json.loads(m.as_json())["policy_version"] == "1.0"
    ctx = m.into_context_dict()
    assert ctx["policy_digest"] == m.digest()
    assert ctx["dataset_classifications"] == {"daily_bars": "restricted"}


# ---------------------------------------------------------------------------
# formats_contract —— CSVProductionContract
# ---------------------------------------------------------------------------

def test_csv_contract_build_and_validate_pass(tmp_path):
    table = pa.table({
        "date": pa.array(["2024-01-02", "2024-01-03"]).cast(pa.date32()),
        "close": pa.array([1.5, 2.5], type=pa.float64()),
        "volume": pa.array([100, 200], type=pa.int64()),
    })
    p = tmp_path / "bars.csv"
    pa_csv.write_csv(table, p)

    contract = build_csv_contract_from_table(table, delimiter=",")
    assert contract.delimiter == ","
    problems = contract.validate(str(p))
    assert problems == []                    # 显式 schema 一致 → 通过
    assert contract.validate_no_pandas(str(p)) == []


def test_csv_contract_validate_fails_on_column_change(tmp_path):
    table = pa.table({
        "date": pa.array(["2024-01-02", "2024-01-03"]).cast(pa.date32()),
        "close": pa.array([1.5, 2.5], type=pa.float64()),
        "volume": pa.array([100, 200], type=pa.int64()),
    })
    p = tmp_path / "bars.csv"
    pa_csv.write_csv(table, p)

    contract = build_csv_contract_from_table(table)
    # 改列名：契约里 close → close_price，文件仍是 close → validate 报问题
    contract.schema = {"date": contract.schema["date"], "close_price": pa.float64(),
                       "volume": contract.schema["volume"]}
    problems = contract.validate(str(p))
    assert problems, "改列名后必须报问题"
    assert any("close_price" in m or "mismatch" in m for m in problems)


def test_csv_contract_validate_fails_on_type_change(tmp_path):
    table = pa.table({
        "date": pa.array(["2024-01-02", "2024-01-03"]).cast(pa.date32()),
        "close": pa.array([1.5, 2.5], type=pa.float64()),
    })
    p = tmp_path / "bars.csv"
    pa_csv.write_csv(table, p)

    contract = build_csv_contract_from_table(table)
    # 改类型：close 声明成 int64，文件是 1.5 → cast 失败 → 报问题
    contract.schema = {"date": contract.schema["date"], "close": pa.int64()}
    problems = contract.validate(str(p))
    assert problems, "类型冲突必须报问题"


def test_csv_contract_no_pandas(tmp_path):
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "simple.csv"
    pa_csv.write_csv(table, p)
    contract = build_csv_contract_from_table(table)
    assert contract.validate_no_pandas(str(p)) == []
