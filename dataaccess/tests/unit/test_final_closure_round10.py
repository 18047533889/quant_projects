"""#10 收官轮：运行时 / 破坏性修复的回归测试。

覆盖（closure ledger ID）：
    25  publish 双 rename 崩溃 → journal 确定性 recovery（target 不永久缺失）
    26  audit JSONL 多进程行原子性（flock 兜底，非 PIPE_BUF 假设）
    27  PyArrow/DuckDB/Polars 三后端 parity：date-only end 覆盖完整一天；
        instruments=[] → 0 行（不是全市场）
    28  _check_factor_versions 三 gate 正交（显式版本不再豁免 snapshot/universe）
    29  require_same_universe 缺 universe 元数据列 → fail-closed
    30  refresh_factor_catalog(factor_ids=[...]) partial refresh = merge（不删 B/C）
    31  factor_matrix 列探测失败 → MatrixCoverageMiss（不读全矩阵）
    33  mirror 期望 partition degraded 标记随结果返回（无 module-global 互相覆盖）
    34  RelationHandle SQL 只能 FROM _sub（information_schema / 裸表 / 嵌套子查询拒绝）
    35  remote-only coverage 不再永久 unavailable（declared_remote / partial）
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    DataError,
    MatrixCoverageMiss,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore

# ``data_access`` is a repo-root shim forwarding to ``dataaccess/`` source root;
# subprocess workers must have the repo root on sys.path to resolve it.
_PKG = str(Path(__file__).resolve().parents[3])  # quant_projects/ repo root


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("QUANT_AUDIT_LOG", raising=False)


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


def _publish_pair(tmp_path):
    stag = tmp_path / "staging"
    tgt = tmp_path / "target"
    stag.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.0}]
        ),
        str(stag / "part.parquet"),
    )
    cfg = {
        "st": {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(stag), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
        "tg": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(tgt), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
    }
    store = _store(tmp_path, cfg)
    return store, stag, tgt


# ---------------------------------------------------------------------------
# 25) publish 崩溃 → journal recovery
# ---------------------------------------------------------------------------


def test_publish_crash_midswitch_recovered(tmp_path):
    store, stag, tgt = _publish_pair(tmp_path)
    env = dict(os.environ, QUANT_AUDIT_LOG=str(tmp_path / "audit.log"))
    # 首次发布建 old gen
    store.publish_from_staging("st", "tg")
    assert tgt.exists() and list(tgt.glob("*.parquet"))

    base = (
        "import os, sys\n"
        "sys.path.insert(0,%r)\n"
        "from data_access.registry.loader import load_registry\n"
        "from data_access.store import DataAccessStore\n"
        "from data_access.core.engine import DuckDBEngine\n"
        "real=os.rename\n"
        "state={'hit':0}\n"
        "def rename(src,dst):\n"
        "    real(src,dst)\n"
        "    if 'archive' in str(dst) and state['hit']==0 and 'candidate' not in str(src):\n"
        "        state['hit']=1\n"
        "        os._exit(99)   # crash 在 old→archive 之后、candidate→target 之前\n"
        "os.rename=rename\n"
        "s=DataAccessStore(load_registry(%r), DuckDBEngine(threads=2))\n"
        "s.publish_from_staging('st','tg')\n"
    ) % (_PKG, str(tmp_path / "datasets.yaml"))
    # staging 更新成新数据
    pq.write_table(
        pa.Table.from_pylist(
            [{"d": _dt.date(2024, 1, 3), "s": "AAA", "v": 2.0}]
        ),
        str(stag / "part.parquet"),
    )
    r = subprocess.run([sys.executable, "-c", base], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 99  # crashed
    assert not tgt.exists(), "崩溃后 target 应缺失（recovery 前）"
    # 读路径触发 recovery → target 必须恢复（new generation 或 old，绝不缺失）
    store2 = _store(tmp_path, {
        "tg": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(tgt), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
    })
    t = store2.read("tg", columns=["d", "s", "v"]).to_arrow()
    assert t.num_rows >= 1
    assert not list(tgt.parent.glob(".publish.*.journal")), "recovery 后 journal 应清"
    assert not list(tgt.parent.glob(".publish_candidate.*")), "recovery 后 candidate 应清"


# ---------------------------------------------------------------------------
# 26) audit 多进程行原子性
# ---------------------------------------------------------------------------


def test_audit_multiprocess_lines_atomic(tmp_path):
    log = tmp_path / "audit.log"
    env = dict(os.environ, QUANT_AUDIT_LOG=str(log))
    worker = (
        "import os, sys\n"
        "sys.path.insert(0,%r)\n"
        "from data_access.core import audit\n"
        "pid=os.getpid()\n"
        "for i in range(25):\n"
        "    audit.record(op='write', dataset='test', ok=True, rows=i,\n"
        "        params={'pid': pid, 'i': i},\n"
        "        extra={'pad': 'x' * (2000 if i %% 2 else 8000)},\n"
        "        durable=True)\n"
    ) % _PKG
    procs = [
        subprocess.Popen([sys.executable, "-c", worker], env=env) for _ in range(12)
    ]
    for p in procs:
        assert p.wait() == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12 * 25, f"丢失 {12*25 - len(lines)} 条"
    for ln in lines:
        obj = json.loads(ln)  # 任何一行不合法 JSON 都会抛 → 测试失败
        assert obj["op"] == "write"


# ---------------------------------------------------------------------------
# 27) 三后端 parity：date-only end 完整一天 + instruments=[] 空结果
# ---------------------------------------------------------------------------


def _ts_store(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    rows = [
        {"ts": _dt.datetime(2026, 7, 9, 15, 0), "s": "AAA", "v": 1.0},
        {"ts": _dt.datetime(2026, 7, 10, 0, 0), "s": "AAA", "v": 2.0},
        {"ts": _dt.datetime(2026, 7, 10, 10, 0), "s": "AAA", "v": 3.0},
        {"ts": _dt.datetime(2026, 7, 11, 0, 0), "s": "AAA", "v": 4.0},
    ]
    pq.write_table(pa.Table.from_pylist(rows), str(root / "part.parquet"))
    return _store(tmp_path, {
        "d": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(root), "glob": "*.parquet",
            "time_column": "ts", "instrument_column": "s",
            "schema": {"ts": "timestamp", "s": "string", "v": "double"},
        },
    })


def test_backend_parity_date_only_end(tmp_path):
    store = _ts_store(tmp_path)
    rng = ("2026-07-09", "2026-07-10")
    for mode in ("arrow", "polars"):
        t = store.read_auto("d", mode=mode, time_range=rng, columns=["ts", "s", "v"])
        assert t.num_rows == 3, f"{mode} date-only end 丢最后一天白天行"
        assert sorted(t.column("v").to_pylist()) == [1.0, 2.0, 3.0]
    t_duck = store.read("d", time_range=rng, columns=["ts", "s", "v"]).to_arrow()
    assert t_duck.num_rows == 3


def test_backend_parity_empty_instruments(tmp_path):
    store = _ts_store(tmp_path)
    rng = ("2026-07-09", "2026-07-10")
    for mode in ("arrow", "polars"):
        t = store.read_auto("d", mode=mode, time_range=rng,
                            columns=["ts", "s", "v"], instrument_filter=[])
        assert t.num_rows == 0, f"{mode}: instruments=[] 必须是空结果（不是全市场）"
    t_duck = store.read("d", time_range=rng, columns=["ts", "s", "v"],
                        instrument_filter=[]).to_arrow()
    assert t_duck.num_rows == 0


# ---------------------------------------------------------------------------
# 28/29) _check_factor_versions 正交 + fail-closed
# ---------------------------------------------------------------------------


def _factor_lake_cfg(root: Path) -> dict[str, dict]:
    return {
        "factor_lake": {
            "kind": "parametric", "access_mode": "published", "layout": "hive",
            "root_template": str(root) + "/factors/{factor_id}",
            "glob_template": "year=*/panel.parquet",
            "partition_columns": ["year"],
            "time_column": "datetime", "instrument_column": "asset",
            "hive_partitioning": True,
            "params_schema": {"factor_id": "str"},
            "schema": {
                "datetime": "timestamp", "asset": "string", "value": "double",
                "factor_version": "string", "data_snapshot_id": "string",
            },
        },
    }


def _write_factor(root: Path, fid: str, *, version: str, snapshot: str):
    d = root / "factors" / fid / "year=2024"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"datetime": _dt.datetime(2024, 1, 2, 9, 30), "asset": "AAA",
             "value": 1.0, "factor_version": version, "data_snapshot_id": snapshot},
        ]),
        str(d / "panel.parquet"),
    )


def test_factor_versions_gates_orthogonal(tmp_path):
    # 28：显式 version 匹配 + 不同 data_snapshot → 必须拒绝（版本 gate 不豁免 snapshot gate）
    root = tmp_path / "lake"
    _write_factor(root, "f1", version="v1", snapshot="snapA")
    _write_factor(root, "f2", version="v1", snapshot="snapB")
    store = _store(tmp_path, _factor_lake_cfg(root))
    with pytest.raises(ValidationError):
        store._check_factor_versions(
            ["f1", "f2"], versions={"f1": "v1", "f2": "v1"},
            require_same_data_snapshot=True, time_range=("2024-01-01", "2024-01-31"),
        )
    # 版本不同 → 仍然拒绝（原逻辑）
    with pytest.raises(ValidationError):
        store._check_factor_versions(
            ["f1"], versions={"f1": "v9"}, time_range=("2024-01-01", "2024-01-31"),
        )


def test_require_same_universe_fail_closed_missing_column(tmp_path):
    # 29：factor_lake schema 无 universe 列 → require_same_universe=True 必须 fail-closed
    root = tmp_path / "lake"
    _write_factor(root, "f1", version="v1", snapshot="s")
    store = _store(tmp_path, _factor_lake_cfg(root))
    with pytest.raises(DataError):
        store._check_factor_versions(
            ["f1"], require_same_universe=True,
            time_range=("2024-01-01", "2024-01-31"),
        )


# ---------------------------------------------------------------------------
# 30) refresh_factor_catalog partial = merge
# ---------------------------------------------------------------------------


def _lake_with_meta(root: Path, fids: list[str]):
    for fid in fids:
        d = root / "factors" / fid
        d.mkdir(parents=True, exist_ok=True)
        (d / "_factor_meta.json").write_text(
            json.dumps({"version": "v1", "display_name": fid}), encoding="utf-8"
        )


def test_refresh_factor_catalog_partial_merge(tmp_path):
    from data_access.read.factors import FactorCatalog, factor_catalog_root

    root = tmp_path / "lake"
    _lake_with_meta(root, ["A", "B", "C"])
    cfg = {
        "factor_lake": {
            "kind": "parametric", "access_mode": "published", "layout": "hive",
            "root_template": str(root) + "/factors/{factor_id}",
            "glob_template": "year=*/panel.parquet",
            "partition_columns": ["year"],
            "time_column": "datetime", "instrument_column": "asset",
            "params_schema": {"factor_id": "str"},
            "schema": {"datetime": "timestamp", "asset": "string", "value": "double"},
        },
    }
    store = _store(tmp_path, cfg)
    # 全量 refresh → A,B,C
    store.refresh_factor_catalog()
    cats = FactorCatalog.load(factor_catalog_root(store, "factor_lake"))
    assert set(cats.ids()) == {"A", "B", "C"}
    # 只 refresh A → 目录仍保留 B,C（merge，不覆盖删除）
    store.refresh_factor_catalog(factor_ids=["A"])
    cats2 = FactorCatalog.load(factor_catalog_root(store, "factor_lake"))
    assert set(cats2.ids()) == {"A", "B", "C"}, "partial refresh 删掉了 B/C"


# ---------------------------------------------------------------------------
# 31) factor_matrix 列探测失败 → MatrixCoverageMiss
# ---------------------------------------------------------------------------


def test_factor_matrix_unprobeable_fail_closed(tmp_path):
    mroot = tmp_path / "matrix"
    mroot.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"datetime": _dt.datetime(2024, 1, 2), "asset": "AAA"}]),
        str(mroot / "part.parquet"),
    )
    store = _store(tmp_path, {
        "factor_matrix": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(mroot), "glob": "*.parquet",
            "time_column": "datetime", "instrument_column": "asset",
            "schema": {},  # 无 schema → 只能靠 DESCRIBE/probe
        },
        "factor_lake": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(tmp_path / "fl"), "glob": "*.parquet",
            "time_column": "datetime", "instrument_column": "asset",
        },
    })
    # probe 失败（矩阵需要 universe/frequency 参数但 registry 无 params_schema，
    # 读会因缺参数失败）→ matrix_cols=None → 必须 MatrixCoverageMiss，不能读全矩阵
    with pytest.raises(MatrixCoverageMiss):
        store._read_factor_matrix(
            ["f1"], time_range=("2024-01-01", "2024-01-02"),
            universe="u", frequency="daily", columns=None, limit=None,
            engine="duckdb", result="auto", prefer_polars=False, batch_size=1,
            query_budget=None,
        )


# ---------------------------------------------------------------------------
# 33) mirror 期望 partition degraded 标记随结果返回（无 module-global）
# ---------------------------------------------------------------------------


def test_expected_dates_degraded_per_call(monkeypatch, tmp_path):
    from data_access.cos import mirror as mirror_mod

    class _NoCal:
        has_data = False

    monkeypatch.setattr(
        "data_access.read.session_calendar.get_market_calendar",
        lambda market, store=None: _NoCal(),
    )
    # 无日历 → degraded=True（随结果返回）
    dates, degraded = mirror_mod._expected_dates_with_degraded(
        "us_stock_daily", _dt.date(2024, 1, 1), _dt.date(2024, 1, 10)
    )
    assert degraded is True
    # calendar_day 域永不 degraded（contract 声明 calendar_day）
    class _CalDomainContract:
        calendar_domain = "calendar_day"

    monkeypatch.setattr(
        "data_access.cos_contract.get_cos_contract",
        lambda d: _CalDomainContract(),
    )
    dates2, degraded2 = mirror_mod._expected_dates_with_degraded(
        "ashare_calendar", _dt.date(2024, 1, 1), _dt.date(2024, 1, 10)
    )
    assert degraded2 is False
    # 无 module-global 互相覆盖：连续两次不同 dataset，各自 degraded 独立返回
    _, d3 = mirror_mod._expected_dates_with_degraded(
        "ashare_calendar", _dt.date(2024, 1, 1), _dt.date(2024, 1, 2)
    )
    assert d3 is False
    assert not hasattr(mirror_mod, "_expected_dates_degraded"), "module-global 应已移除"


# ---------------------------------------------------------------------------
# 34) RelationHandle SQL 只能 FROM _sub
# ---------------------------------------------------------------------------


def test_relation_handle_sql_from_scope(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.0}]),
        str(root / "part.parquet"),
    )
    store = _store(tmp_path, {
        "d": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(root), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
    })
    rh = store.sql_relation("SELECT * FROM read_parquet(?)",
                            params=[str(root / "*.parquet")])
    # 合法：FROM _sub
    ok = rh.sql("SELECT s, v FROM _sub WHERE v > ?", params=[0.0]).collect()
    assert ok.num_rows == 1
    # bypass 全部拒绝（information_schema / 裸表 / 嵌套子查询）
    for bad in [
        "SELECT * FROM _sub JOIN information_schema.tables t ON TRUE",
        "SELECT * FROM _sub JOIN some_other_table t ON TRUE",
        "SELECT * FROM _sub WHERE s IN (SELECT x FROM hidden_catalog)",
    ]:
        with pytest.raises(ValidationError):
            rh.sql(bad)


# ---------------------------------------------------------------------------
# 35) remote-only coverage 不再永久 unavailable
# ---------------------------------------------------------------------------


def test_coverage_remote_only_declared(tmp_path, monkeypatch):
    class _FakeContract:
        coverage_start = "2024-01-01"
        coverage_end = "2024-12-31"
        expected_cadence = "daily"
        max_staleness = None
        missing_partition_semantics = "error"

    monkeypatch.setattr(
        "data_access.cos_contract.get_cos_contract", lambda d: _FakeContract()
    )

    class _FakeDS:
        name = "remote_ds"
        root = "s3://bucket/remote"
        glob = "*.parquet"
        time_column = "d"
        instrument_column = "s"
        access_mode = "published"

    class _FakeReg:
        def get(self, name):
            return _FakeDS()

    class _FakeStore:
        _registry = _FakeReg()
        registry = _FakeReg()

        def _resolve_raw_paths(self, ds, **kw):
            return ["s3://bucket/remote/*.parquet"]

    from data_access.read.coverage import compute_coverage

    report = compute_coverage(_FakeStore(), "remote_ds")
    assert report.status == "partial"
    assert report.authority == "declared_remote"
    assert any("remote-only" in p for p in report.problems)
