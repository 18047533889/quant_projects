"""R39 分钟→日聚合性能整改：PERF-033/034/035/037 回归。

覆盖：
    - PERF-033：Counter 去重（O(K)），重复输出列名错误消息逐字不变；
    - PERF-034：整数分钟过滤与旧 ``strftime('%H:%M')`` 字符串过滤在边界行上
      结果逐行一致（09:30:59 排除、09:31:00 包含）；
    - PERF-035：同一 minute-window 的多个输出共享一个过滤条件列（SQL 里条件
      只出现一次）；
    - PERF-037：``result_mode="relation"`` 惰性句柄物化后与默认 arrow 值一致；
      ``"arrow_stream"`` 流式消费与 arrow 值一致。
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.read.aggregation import (
    AggregationItem,
    AggregationSpec,
    ParsedAggregationItem,
    _build_bundle_sql,
    aggregate_minute_bundle,
    aggregate_minute_to_daily,
    build_minute_aggregation_sql,
)
from data_access.read.minute_filter import FilterSignature, hhmm_to_minutes
from data_access.registry import load_registry
from data_access.store import DataAccessStore

# 边界行（UTC 存储，北京时间 = UTC+8）：
#   CST 09:30:58/09:30:59 → 应被 09:31 起点的窗口排除
#   CST 09:31:00/09:31:01/09:31:59 → 应包含
#   CST 09:59:59 / 10:00:00 → 10:00 闭区间包含
#   CST 14:30:00 / 14:30:59 / 15:00:00 → 午后窗口
_BOUNDARY_ROWS = [
    ("2024-01-02 01:30:58", "A", 1),  # CST 09:30:58
    ("2024-01-02 01:30:59", "A", 2),  # CST 09:30:59
    ("2024-01-02 01:31:00", "A", 3),  # CST 09:31:00
    ("2024-01-02 01:31:01", "A", 4),  # CST 09:31:01
    ("2024-01-02 01:31:59", "A", 5),  # CST 09:31:59
    ("2024-01-02 01:59:59", "A", 7),  # CST 09:59:59
    ("2024-01-02 02:00:00", "A", 6),  # CST 10:00:00
    ("2024-01-02 06:30:00", "A", 10),  # CST 14:30:00
    ("2024-01-02 06:30:59", "A", 11),  # CST 14:30:59
    ("2024-01-02 07:00:00", "A", 12),  # CST 15:00:00
]


def _write_minute_parquet(tmp_path, rows=_BOUNDARY_ROWS) -> str:
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "volume"])
    # 字符串时间 → datetime，保证 parquet 列是 TIMESTAMP（而不是 VARCHAR），
    # 否则 ``AT TIME ZONE`` 无法把字符串解释成 UTC 时间。
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    pq.write_table(pa.table(df), str(tmp_path / "m.parquet"))
    return str(tmp_path / "m.parquet")


def _minute_store(tmp_path, rows=_BOUNDARY_ROWS, *, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    path = _write_minute_parquet(tmp_path, rows)
    (tmp_path / "datasets.yaml").write_text(
        f"""
minute_ds:
  kind: static
  access_mode: published
  layout: plain
  format: parquet
  root: {tmp_path}
  glob: "*.parquet"
  time_column: timestamp
  instrument_column: symbol
  schema:
    timestamp: timestamp
    symbol: string
    volume: int
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    ), path


# ---------------------------------------------------------------------------
# PERF-033：Counter 去重 + 错误消息不变
# ---------------------------------------------------------------------------

def test_counter_dup_output_name_raises_same_error(tmp_path, monkeypatch):
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    items = [
        AggregationItem("volume", AggregationSpec(aggregation="minute_range", start="09:31", end="10:00"), "signal"),
        AggregationItem("volume", AggregationSpec(aggregation="minute_range", start="10:00", end="11:00"), "signal"),
    ]
    with pytest.raises(Exception, match="输出列名重复") as exc_info:
        aggregate_minute_bundle(store, "minute_ds", items)
    msg = str(exc_info.value)
    assert "aggregate_minute_bundle 输出列名重复: ['signal']。" in msg
    # 错误消息与旧实现逐字一致（"每个 AggregationItem 必须用 output_name 区分…"）
    assert "每个 AggregationItem 必须用 output_name 区分" in msg


# ---------------------------------------------------------------------------
# PERF-034：整数分钟过滤 == 旧 strftime 字符串过滤（边界行）
# ---------------------------------------------------------------------------

def test_hhmm_to_minutes_basics():
    assert hhmm_to_minutes("09:31") == 571
    assert hhmm_to_minutes("14:30") == 870
    assert hhmm_to_minutes("00:00") == 0
    assert hhmm_to_minutes("23:59") == 1439


def test_integer_minute_filter_matches_strftime_boundary(tmp_path):
    """minute_range 09:31~10:00：整数分钟过滤与旧 strftime 过滤结果逐行一致。

    边界语义：09:30:59（570）排除、09:31:00（571）包含、10:00:00（600）包含。
    """
    path = _write_minute_parquet(tmp_path)
    import duckdb

    con = duckdb.connect(":memory:")
    spec = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    new_sql, new_params = build_minute_aggregation_sql(
        time_column="timestamp",
        instrument_column="symbol",
        value_column="volume",
        field="volume",
        spec=spec,
    )
    new_rows = con.execute(new_sql, [path, *new_params]).fetchall()

    # 旧实现的 SQL 形态：strftime 字符串比较 + 绑定参数
    loc = "timezone('Asia/Shanghai', timestamp AT TIME ZONE 'UTC')"
    old_sql = (
        f"SELECT CAST({loc} AS DATE) AS ts, symbol AS inst, sum(volume) AS value "
        f"FROM (SELECT * FROM read_parquet(?)) "
        f"WHERE strftime({loc}, '%H:%M') >= ? AND strftime({loc}, '%H:%M') <= ? "
        f"GROUP BY CAST({loc} AS DATE), symbol"
    )
    old_rows = con.execute(old_sql, [path, "09:31", "10:00"]).fetchall()

    assert new_rows == old_rows
    # 09:30:58/59（1+2）排除；09:31 三笔（3+4+5）+09:59:59(7)+10:00(6) = 25
    assert new_rows == [(__import__("datetime").date(2024, 1, 2), "A", 25)]


def test_minute_at_integer_filter_boundary(tmp_path):
    """minute_at('09:31') 只命中 09:31:00~09:31:59（3+4+5=12），09:30:59 排除。"""
    path = _write_minute_parquet(tmp_path)
    import duckdb

    con = duckdb.connect(":memory:")
    spec = AggregationSpec(aggregation="minute_at", hhmm="09:31", market="ashare")
    new_sql, new_params = build_minute_aggregation_sql(
        time_column="timestamp",
        instrument_column="symbol",
        value_column="volume",
        field="volume",
        spec=spec,
    )
    new_rows = con.execute(new_sql, [path, *new_params]).fetchall()

    loc = "timezone('Asia/Shanghai', timestamp AT TIME ZONE 'UTC')"
    old_sql = (
        f"SELECT CAST({loc} AS DATE) AS ts, symbol AS inst, sum(volume) AS value "
        f"FROM (SELECT * FROM read_parquet(?)) "
        f"WHERE strftime({loc}, '%H:%M') = ? "
        f"GROUP BY CAST({loc} AS DATE), symbol"
    )
    old_rows = con.execute(old_sql, [path, "09:31"]).fetchall()
    assert new_rows == old_rows
    assert new_rows == [(__import__("datetime").date(2024, 1, 2), "A", 12)]


# ---------------------------------------------------------------------------
# PERF-035：FilterSignature 规范化 + 同一窗口只生成一次过滤条件
# ---------------------------------------------------------------------------

def test_filter_signature_canonicalize_dedup_equivalence():
    spec_a = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    spec_b = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    sig_a = FilterSignature.canonicalize(spec_a, session_total_bars=240)
    sig_b = FilterSignature.canonicalize(spec_b, session_total_bars=240)
    assert sig_a == sig_b
    assert sig_a.kind == "minute_range"
    assert sig_a.start_minute == 571 and sig_a.end_minute == 600
    # 与旧字符串条件的等价 SQL（整数分钟内联）
    assert (
        sig_a.to_sql(minute_expr="_minute", elapsed_expr="_el")
        == "_minute >= 571 AND _minute <= 600"
    )


def test_bundle_dedup_emits_filter_once():
    """3 个 item 共享同一窗口：过滤条件在 SQL 里只定义一次（_f0），多列复用。"""
    spec = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    parsed_items = [
        ParsedAggregationItem(
            item=AggregationItem("volume", spec, f"out_{i}"), spec=spec
        )
        for i in range(3)
    ]
    sql, params = _build_bundle_sql(
        from_clause="read_parquet('/tmp/x.parquet')",
        predicate_where="",
        time_column="timestamp",
        instrument_column="symbol",
        parsed_items=parsed_items,
        market="ashare",
        timezone=None,
    )
    # 条件 `_minute >= 571 AND _minute <= 600` 只在 `_scan` 定义一次（571 还会出现在
    # A 股 session elapsed 的 CASE 常量里，所以不能用裸 "571" 计数）
    assert sql.count("_minute >= 571") == 1
    assert sql.count("_minute <= 600") == 1
    assert "AS _f0" in sql
    # 3 个输出列都复用 _f0：定义 1 次 + FILTER 引用 3 次
    assert sql.count("_f0") == 4
    assert params == []


def test_bundle_dedup_distinct_windows_emit_separate_cols():
    """两个不同窗口 → 两个 _f 列；共享窗口的多个输出仍复用同一列。"""
    spec_morning = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    spec_afternoon = AggregationSpec(
        aggregation="minute_range", start="14:30", end="15:00", market="ashare"
    )
    parsed_items = [
        ParsedAggregationItem(AggregationItem("volume", spec_morning, "m1"), spec_morning),
        ParsedAggregationItem(AggregationItem("volume", spec_morning, "m2"), spec_morning),
        ParsedAggregationItem(AggregationItem("volume", spec_afternoon, "a1"), spec_afternoon),
    ]
    sql, _ = _build_bundle_sql(
        from_clause="read_parquet('/tmp/x.parquet')",
        predicate_where="",
        time_column="timestamp",
        instrument_column="symbol",
        parsed_items=parsed_items,
        market="ashare",
        timezone=None,
    )
    assert "AS _f0" in sql and "AS _f1" in sql
    # morning（571）与 afternoon（870）的过滤条件各只生成一次
    assert sql.count("_minute >= 571") == 1
    assert sql.count("_minute >= 870") == 1
    # m1/m2 共享 _f0
    assert sql.count("FILTER (WHERE _f0)") == 2
    assert sql.count("FILTER (WHERE _f1)") == 1


# ---------------------------------------------------------------------------
# PERF-037：result_mode = arrow / relation / arrow_stream 值一致
# ---------------------------------------------------------------------------

def _rel_to_arrow(rel):
    if hasattr(rel, "to_arrow_table"):
        return rel.to_arrow_table()
    if hasattr(rel, "fetch_arrow_table"):
        return rel.fetch_arrow_table()
    if hasattr(rel, "fetchall"):
        rows = rel.fetchall()
        cols = list(rel.columns)
        return pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    raise AssertionError(f"无法物化 relation: {type(rel)}")


def _reader_to_arrow(reader):
    batches = list(reader)
    return pa.Table.from_batches(batches) if batches else pa.table({})


def test_result_mode_relation_to_daily_same_values(tmp_path, monkeypatch):
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    spec = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    arrow_t = aggregate_minute_to_daily(
        store, "minute_ds", "volume", spec, result_mode="arrow"
    ).to_arrow()
    rel = aggregate_minute_to_daily(
        store, "minute_ds", "volume", spec, result_mode="relation"
    )
    assert rel is not None
    rel_t = _rel_to_arrow(rel)
    assert rel_t.to_pydict() == arrow_t.to_pydict()
    assert arrow_t.to_pydict()["value"] == [25]


def test_result_mode_arrow_stream_to_daily_same_rows(tmp_path, monkeypatch):
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    spec = AggregationSpec(
        aggregation="minute_range", start="09:31", end="10:00", market="ashare"
    )
    arrow_t = aggregate_minute_to_daily(
        store, "minute_ds", "volume", spec, result_mode="arrow"
    ).to_arrow()
    reader = aggregate_minute_to_daily(
        store, "minute_ds", "volume", spec, result_mode="arrow_stream"
    )
    stream_t = _reader_to_arrow(reader)
    assert stream_t.to_pydict() == arrow_t.to_pydict()


def test_result_mode_bundle_relation_and_stream_same_values(tmp_path, monkeypatch):
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    items = [
        AggregationItem(
            "volume",
            AggregationSpec(aggregation="minute_range", start="09:31", end="10:00"),
            "morning_vol",
        ),
        AggregationItem(
            "volume",
            AggregationSpec(aggregation="minute_at", hhmm="10:00"),
            "at_1000",
        ),
    ]
    arrow_t = aggregate_minute_bundle(
        store, "minute_ds", items, market="ashare", result_mode="arrow"
    ).to_arrow()
    rel = aggregate_minute_bundle(
        store, "minute_ds", items, market="ashare", result_mode="relation"
    )
    rel_t = _rel_to_arrow(rel)
    assert rel_t.to_pydict() == arrow_t.to_pydict()

    reader = aggregate_minute_bundle(
        store, "minute_ds", items, market="ashare", result_mode="arrow_stream"
    )
    stream_t = _reader_to_arrow(reader)
    assert stream_t.to_pydict() == arrow_t.to_pydict()


def test_result_mode_invalid_rejected(tmp_path, monkeypatch):
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    with pytest.raises(Exception, match="result_mode"):
        aggregate_minute_to_daily(
            store, "minute_ds", "volume", AggregationSpec(), result_mode="bogus"
        )


def test_result_mode_default_arrow_is_readhandle(tmp_path, monkeypatch):
    """默认 result_mode='arrow' 返回 ReadHandle（零行为变化）。"""
    store, _ = _minute_store(tmp_path, monkeypatch=monkeypatch)
    h = aggregate_minute_bundle(
        store,
        "minute_ds",
        [
            AggregationItem(
                "volume",
                AggregationSpec(
                    aggregation="minute_range", start="09:31", end="10:00", market="ashare"
                ),
                "morning_vol",
            )
        ],
    )
    from data_access.read.read_handle import ReadHandle

    assert isinstance(h, ReadHandle)
    assert h.to_arrow().to_pydict()["morning_vol"][0] == 25
