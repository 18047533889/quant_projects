# -*- coding: utf-8
"""#收官轮（3+1 排除式复查）DataAccess 侧回归测试。

覆盖（closure ledger，对应外部 AI 复查 3+1 项）：
    A. ReadPlan / CompiledDataRequest **真正 immutable**：compiled 的
       source_params/filters 嵌套结构、ReadPlan 的 datasets/per_dataset_columns/
       snapshot_policy/plan_snapshot_tokens 在 plan() 后全部不可变——调用方
       「改 plan」直接 TypeError，explain 显示的 == execute 执行的。
    B. engine/result strict enum + DataRequest typed：``engine="polarr"`` /
       ``result="lazzy"`` 不再静默落到 duckdb；``pit="false"`` 不再 bool()→True；
       limit 非负 int|None；snapshot_policy/engine/result 严格 enum。
    C. factor_lake_wide contract 修正：宽表 pivot（标的在列轴）标记
       specialized_only，generic read/scan 一律拒绝，不再在 asset 假列上过滤。
    D. ReadLineage 保留 None（全市场）与 ()（空股票池）区别；params canonicalize
       成不可变 tuple（stream/aggregation 不再塞 mutable dict）。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.read.data_request import DataRequest, compile_data_request
from data_access.registry import load_registry
from data_access.store import DataAccessStore

_PKG = "/home/shw/quant_projects/dataaccess"


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _static_store(tmp_path):
    """单数据集静态 store：d(date) × s(string) × v(double)。"""
    root = tmp_path / "d"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.0},
            {"d": _dt.date(2024, 1, 2), "s": "BBB", "v": 2.0},
            {"d": _dt.date(2024, 1, 3), "s": "AAA", "v": 3.0},
        ]),
        str(root / "part.parquet"),
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
    v: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))


# ---------------------------------------------------------------------------
# A. ReadPlan / CompiledDataRequest 真正 immutable
# ---------------------------------------------------------------------------


def test_plan_compiled_source_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    req = DataRequest(
        fields=["v"], source_params={"ds": {"probe": "p1"}}, filters={"s": "AAA"}
    )
    plan = store.plan(req)
    compiled = plan.compiled
    # source_params 是深冻结的不可变映射（MappingProxyType），写即 TypeError
    assert isinstance(compiled.source_params, MappingProxyType)
    with pytest.raises(TypeError):
        compiled.source_params["ds"]["probe"] = "hacked"  # type: ignore[index]
    with pytest.raises(TypeError):
        compiled.source_params["ds"] = {"probe": "x"}  # type: ignore[index]
    # filters 也冻结成 Mapping（proxy）
    assert isinstance(compiled.filters, MappingProxyType)
    with pytest.raises(TypeError):
        compiled.filters["s"] = "BBB"  # type: ignore[index]
    # plan 后改活 req 不影响执行：仍是 plan 时刻 filters={"s": "AAA"}（2 行），
    # 而不是改后 filters={"s": "BBB"}（0 行）
    req.filters = {"s": "BBB"}
    out = plan.execute().to_arrow()
    assert out.num_rows == 2
    assert list(out.column("s").to_pylist()) == ["AAA", "AAA"]


def test_plan_state_immutable(tmp_path):
    store = _static_store(tmp_path)
    plan = store.plan(DataRequest(fields=["v"], snapshot_policy="fail_if_changed"))
    assert plan.datasets == ("ds",)
    assert plan.per_dataset_columns["ds"] == ("v",)
    # 顶层赋值 → frozen dataclass
    with pytest.raises(AttributeError):
        plan.snapshot_policy = "latest"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        plan.engine = "duckdb"  # type: ignore[misc]
    # 容器变异 → tuple / MappingProxyType
    with pytest.raises(AttributeError):
        plan.datasets.clear()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        plan.per_dataset_columns["ds"] = ["other"]  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.plan_snapshot_tokens["ds"]["has_manifest"] = True  # type: ignore[index]
    # 冻结后 execute 照常工作（读只受 plan 时刻语义约束）
    out = plan.execute().to_arrow()
    assert out.num_rows == 3


def test_compiled_request_immutable_direct():
    """不经 store，直接 compile_data_request 也应冻结。"""
    req = DataRequest(
        fields=["a"],
        joins={"ds2": "asof"},
        transforms={"a": "minute_at"},
        field_params={"a": {"lag": 2}},
    )
    c = compile_data_request(req)
    for field in ("joins", "transforms", "field_params"):
        obj = getattr(c, field)
        assert isinstance(obj, MappingProxyType), field
    with pytest.raises(TypeError):
        c.transforms["a"] = "pct"  # type: ignore[index]


class _UncopyableMutable:
    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        raise RuntimeError("deepcopy disabled")

    def __copy__(self):
        raise RuntimeError("copy disabled")


def test_compiled_request_rejects_uncopyable_mutable_value():
    value = _UncopyableMutable("live")
    with pytest.raises(
        ValidationError,
        match="cannot be canonically encoded or detached: dict",
    ):
        compile_data_request(DataRequest(fields=["a"], filters={"value": value}))


def test_compiled_request_detaches_mutable_value_before_plan():
    value = {"nested": ["before"]}
    compiled = compile_data_request(DataRequest(fields=["a"], filters=value))
    value["nested"].append("after")
    assert compiled.filters["nested"] == ("before",)  # type: ignore[index]


def test_compiled_request_accepts_immutable_canonical_values():
    compiled = compile_data_request(
        DataRequest(
            fields=["a"],
            start="2024-01-01",
            end=3,
            filters={"flag": True, "name": "stable", "coords": (1, 2)},
        )
    )
    assert compiled.start == "2024-01-01"
    assert compiled.end == 3
    assert compiled.filters["flag"] is True  # type: ignore[index]


# ---------------------------------------------------------------------------
# B. engine/result strict enum + DataRequest typed
# ---------------------------------------------------------------------------


def test_read_rejects_unknown_engine(tmp_path):
    store = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="engine"):
        store.read("ds", columns=["v"], engine="polarr")
    with pytest.raises(ValidationError, match="engine"):
        store.read("ds", columns=["v"], engine="duck")


def test_read_rejects_unknown_result(tmp_path):
    store = _static_store(tmp_path)
    with pytest.raises(ValidationError, match="result"):
        store.read("ds", columns=["v"], result="lazzy")
    with pytest.raises(ValidationError, match="result"):
        store.read("ds", columns=["v"], result="table")


def test_read_accepts_valid_engine_result(tmp_path):
    store = _static_store(tmp_path)
    for eng, res in [("duckdb", "arrow"), ("polars", "pandas"), ("pyarrow", "lazy")]:
        out = store.read("ds", columns=["v"], engine=eng, result=res).to_arrow()
        assert out.num_rows == 3


def test_data_request_typed_bool_rejects_str():
    for kw in ({"pit": "false"}, {"normalize_units": "yes"}, {"time_varying_universe": 1}):
        with pytest.raises(ValidationError, match="bool"):
            DataRequest(fields=["a"], **kw)  # type: ignore[arg-type]


def test_data_request_typed_bool_accepts_real_bool():
    r = DataRequest(fields=["a"], pit=False, normalize_units=True)
    c = compile_data_request(r)
    assert c.pit is False and c.normalize_units is True


def test_data_request_limit_nonnegative_int():
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValidationError, match="limit"):
            DataRequest(fields=["a"], limit=bad)  # type: ignore[arg-type]
    r = DataRequest(fields=["a"], limit=0)
    assert r.limit == 0


def test_data_request_enum_rejects_and_normalizes():
    with pytest.raises(ValidationError, match="snapshot_policy"):
        DataRequest(fields=["a"], snapshot_policy="bogus")
    with pytest.raises(ValidationError, match="engine"):
        DataRequest(fields=["a"], engine="DuckDbX")
    # 大小写不敏感：normalize 成小写规范值
    r = DataRequest(fields=["a"], engine="DuckDB", result="Arrow")
    assert r.engine == "duckdb" and r.result == "arrow"


# ---------------------------------------------------------------------------
# C. factor_lake_wide contract：specialized_only，generic read 拒绝
# ---------------------------------------------------------------------------


def test_factor_lake_wide_specialized_only_registry():
    reg = load_registry()
    ds = reg.get("factor_lake_wide")
    assert ds.specialized_only is True
    assert ds.specialized_only_reason
    # asset 是列轴不是物理列 → instrument_column 不该声明
    assert ds.instrument_column is None


def test_factor_lake_wide_generic_read_rejected(tmp_path):
    reg = load_registry()
    store = DataAccessStore(registry=reg, engine=DuckDBEngine(threads=1))
    for call in (
        lambda: store.read("factor_lake_wide", columns=["datetime"], params={"factor_id": "f"}),
        lambda: store.read_arrow("factor_lake_wide", params={"factor_id": "f"}),
        lambda: store.scan_polars("factor_lake_wide", params={"factor_id": "f"}),
        lambda: store.read_result("factor_lake_wide", params={"factor_id": "f"}),
    ):
        with pytest.raises(ValidationError, match="specialized_only"):
            call()


def test_specialized_only_requires_reason(tmp_path):
    """specialized_only=true 必须给 reason（防止配置裸标记无人能读）。"""
    root = tmp_path / "w"
    root.mkdir(parents=True, exist_ok=True)
    from data_access.registry.loader import _parse_dataset

    with pytest.raises(ValidationError, match="specialized_only_reason"):
        _parse_dataset("w", {"kind": "static", "access_mode": "published", "layout": "plain",
                             "root": str(root), "glob": "*.parquet",
                             "time_column": "d", "instrument_column": "s",
                             "specialized_only": True})


# ---------------------------------------------------------------------------
# D. ReadLineage None（全市场）vs ()（空股票池）+ params 不可变
# ---------------------------------------------------------------------------


def test_read_lineage_preserves_none_vs_empty(tmp_path):
    store = _static_store(tmp_path)
    # None → lineage.instrument_filter is None（全市场，不限制）
    h_none = store.read("ds", columns=["v"])
    assert h_none.lineage.instrument_filter is None
    # [] → 空股票池 0 行，lineage 记为 ()
    h_empty = store.read("ds", columns=["v"], instrument_filter=[])
    assert h_empty.lineage.instrument_filter == ()
    assert h_empty.lineage.instrument_filter is not None
    assert h_empty.to_arrow().num_rows == 0
    # 具体标的 → tuple
    h_aaa = store.read("ds", columns=["v"], instrument_filter=["AAA"])
    assert h_aaa.lineage.instrument_filter == ("AAA",)
    assert h_aaa.to_arrow().num_rows == 2


def test_read_lineage_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    h = store.read("ds", columns=["v"])
    params = h.lineage.params
    assert isinstance(params, tuple)
    assert all(isinstance(pair, tuple) for pair in params)
    # 值全部冻结：没有可变 dict/list 泄漏
    for _k, v in params:
        assert not isinstance(v, (dict, list))


def test_stream_lineage_params_immutable(tmp_path):
    store = _static_store(tmp_path)
    handle = store.read("ds", columns=["v"], result="stream")
    params = handle.lineage.params
    assert isinstance(params, tuple)
    assert all(isinstance(pair, tuple) for pair in params)
    for _k, v in params:
        assert not isinstance(v, (dict, list))
