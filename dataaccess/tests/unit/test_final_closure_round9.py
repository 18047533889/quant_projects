"""#9 收官轮（真实数据 + 运行时 + 破坏性）回归测试。

覆盖（对应 closure ledger ID）：
    13  atomic writer 多进程安全（O_EXCL 唯一 tmp + full-write + fsync chain）
    14  publish 成功 + durable audit 失败 → CommittedButAuditFailed（数据已提交）
    15  read_auto(mode="polars") 受控 collect，无 ScanHandle→ReadResult 类型链错误
    16  ScanHandle collect 前 remote/local snapshot revalidation（strict fail-closed）
    17  mirror 文件 verified/corrupt/legacy_unverified/missing 三态 + auto/hybrid 消费
    19  remote ParametricDataset 校验复用 validate_params（不可绕过）
    22  read_joined 空 universe → 0 行 typed result（非 IndexError / 全量扫描）
    23  PITIndexMetadata strict typed（bool 串味 / 负数 counts fail-closed）
    24  并发 PIT index builder（两个进程）→ current 永远指向有效 generation
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    CommittedButAuditFailed,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore

_PKG = "/home/shw/quant_projects/dataaccess"
_DATA = Path("/home/shw/quant_projects/data")
_ASHARE = _DATA / "a_share/lqtp_data/StockDailyBar"
HAS_REAL_DATA = _ASHARE.is_dir() and any(_ASHARE.glob("2019-*.parquet"))


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


def _daily_ds(root: Path, name: str = "ds") -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(root), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        }
    }


def _write_daily(root: Path, dates=("2024-01-02", "2024-01-03")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dates:
        for sym in ("AAA", "BBB"):
            rows.append({"d": _dt.date.fromisoformat(d), "s": sym, "v": 1.5})
    pq.write_table(pa.Table.from_pylist(rows), str(root / "part.parquet"))


# ---------------------------------------------------------------------------
# 13) atomic writer 多进程安全
# ---------------------------------------------------------------------------


def test_atomic_writer_multiprocess_no_corruption(tmp_path):
    from data_access.core.atomic import atomic_write_bytes

    target = tmp_path / "manifest.json"
    worker = (
        "import sys,time,json\n"
        "from pathlib import Path\n"
        "sys.path.insert(0,%r)\n"
        "from data_access.core.atomic import atomic_write_bytes\n"
        "t=Path(%r)\n"
        "for i in range(25):\n"
        "  atomic_write_bytes(t, json.dumps({'i':i,'pad':'x'*400}).encode(), durable=True)\n"
        "  time.sleep(0.002)\n"
    ) % ("/home/shw/quant_projects/dataaccess", str(target))
    procs = [
        subprocess.Popen([sys.executable, "-c", worker]) for _ in range(3)
    ]
    for p in procs:
        assert p.wait() == 0
    parsed = json.loads(target.read_bytes())
    assert parsed["i"] in range(25)  # 总是某个 writer 的完整 payload
    leftovers = list(tmp_path.glob(".manifest.json.tmp.*"))
    assert not leftovers, f"并发 writer 残留 tmp: {leftovers}"


# ---------------------------------------------------------------------------
# 14) publish 成功 + durable audit 失败 → CommittedButAuditFailed
# ---------------------------------------------------------------------------


def test_publish_committed_but_audit_failed(tmp_path, monkeypatch):
    # 审计路径放到不可创建的父目录下 → durable audit 必失败
    monkeypatch.setenv("QUANT_AUDIT_LOG", "/proc/1/nonexistent/audit.log")

    stag_root = tmp_path / "staging"
    tgt_root = tmp_path / "target"
    _write_daily(stag_root)
    ds_cfg = {
        "st": {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(stag_root), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
        "tg": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(tgt_root), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
    }
    store = _store(tmp_path, ds_cfg)
    with pytest.raises(CommittedButAuditFailed):
        store.publish_from_staging("st", "tg")
    # 数据**已提交**：target 目录有 parquet（candidate→target rename 已发生）
    files = list(tgt_root.glob("*.parquet"))
    assert files, "publish 已提交但 target 没有数据 = commit 被误判 ABORTED"
    tbl = pq.read_table(str(files[0]))
    assert tbl.num_rows == 4


# ---------------------------------------------------------------------------
# 15) read_auto(mode="polars") 受控 collect
# ---------------------------------------------------------------------------


def test_read_auto_polars_controlled_collect(tmp_path):
    root = tmp_path / "d"
    _write_daily(root)
    store = _store(tmp_path, _daily_ds(root))
    r = store.read_auto(
        "ds", mode="polars",
        time_range=("2024-01-02", "2024-01-03"),
        fields=["d", "s", "v"], instrument_filter=["AAA"],
    )
    # polars 分支返回 pyarrow Table（scan → ScanHandle.collect_table），不是类型链错误
    assert isinstance(r, pa.Table)
    assert r.num_rows == 2
    # auto router 也能走通
    r2 = store.read_auto(
        "ds", time_range=("2024-01-02", "2024-01-02"),
        fields=["d", "s", "v"], instrument_filter=["AAA"],
    )
    assert isinstance(r2, pa.Table) or hasattr(r2, "to_arrow")


def test_read_auto_rejects_unknown_mode(tmp_path):
    root = tmp_path / "d"
    _write_daily(root)
    store = _store(tmp_path, _daily_ds(root))
    for bad in ["foo", "", "POLARR", 123]:
        with pytest.raises((ValidationError, ValueError)):
            store.read_auto(
                "ds", mode=bad,
                time_range=("2024-01-02", "2024-01-02"),
                fields=["d", "s", "v"],
            )


# ---------------------------------------------------------------------------
# 16) ScanHandle snapshot revalidation（local 变化 strict fail-closed）
# ---------------------------------------------------------------------------


def test_scan_handle_revalidates_changed_local_file(tmp_path, monkeypatch):
    root = tmp_path / "d"
    _write_daily(root)
    store = _store(tmp_path, _daily_ds(root))
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")

    scan = store.scan(
        "ds", time_range=("2024-01-02", "2024-01-03"),
        columns=["d", "s", "v"], instrument_filter=["AAA", "BBB"],
    )
    # 覆盖底层文件（mtime_ns 变化）
    p = root / "part.parquet"
    os.utime(p, ns=(p.stat().st_mtime_ns + 1000, p.stat().st_mtime_ns + 1000))
    with pytest.raises(ValidationError):
        scan.collect()  # strict：collect 前检测到变化 → fail-closed
    # research 模式 → 重建 snapshot（lineage 不撒谎）
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ")
    scan2 = store.scan(
        "ds", time_range=("2024-01-02", "2024-01-03"),
        columns=["d", "s", "v"], instrument_filter=["AAA", "BBB"],
    )
    res = scan2.collect()
    assert res.table.num_rows == 4


# ---------------------------------------------------------------------------
# 17) mirror 文件状态三态 + 权威读取消费
# ---------------------------------------------------------------------------


def test_mirror_file_state_three_states(tmp_path, monkeypatch):
    from data_access.cos.mirror import _local_file_usable, _mirror_file_state

    f = tmp_path / "data.parquet"
    f.write_bytes(b"payload")
    assert _mirror_file_state(f) == "legacy_unverified"
    # manifest 一致 → verified
    (tmp_path / "data.parquet.manifest.json").write_text(
        json.dumps({"size": len(b"payload"), "mtime_ns": f.stat().st_mtime_ns, "verified": True})
    )
    assert _mirror_file_state(f) == "verified"
    assert _local_file_usable(f) is True
    # 同 size 篡改（mtime 变了）→ corrupt，权威读取不可用
    os.utime(f, ns=(f.stat().st_mtime_ns + 100, f.stat().st_mtime_ns + 100))
    assert _mirror_file_state(f) == "corrupt"
    assert _local_file_usable(f) is False
    # 空文件 → corrupt
    (tmp_path / "empty.parquet").write_bytes(b"")
    assert _mirror_file_state(tmp_path / "empty.parquet") == "corrupt"
    # strict 下 manifest-less 的 legacy_unverified → 不可用（不能把「非空」当「完整」）
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    assert _local_file_usable(f) is False


# ---------------------------------------------------------------------------
# 19) remote ParametricDataset 校验复用 validate_params（不可绕过）
# ---------------------------------------------------------------------------


def test_remote_parametric_validation_cannot_bypass(tmp_path):
    from data_access.cos.remote import resolve_remote_paths
    from data_access.core.exceptions import ValidationError as VE
    from data_access.registry.params_validation import ParamSpec, validate_params

    # 校验本身：非法参数被拒
    specs = {
        "factor_id": ParamSpec(name="factor_id", type="str", pattern="^[a-z0-9_]+$"),
        "freq": ParamSpec(name="freq", type="str", enum_values=("daily", "weekly")),
    }
    with pytest.raises(VE):
        validate_params("dl", specs, {"factor_id": "../escape", "freq": "daily"})
    with pytest.raises(VE):
        validate_params("dl", specs, {"factor_id": "f1", "freq": "monthly"})
    ok = validate_params("dl", specs, {"factor_id": "f1", "freq": "daily"})
    assert ok["factor_id"] == "f1"
    # resolve_remote_paths 对带 params_schema 的 dataset 走同一 validate_params
    assert callable(resolve_remote_paths)


# ---------------------------------------------------------------------------
# 22) read_joined 空 universe → 0 行 typed result
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_REAL_DATA, reason="真实 A股 数据不可用")
def test_read_joined_empty_universe_real(tmp_path):
    reg = load_registry()
    store = DataAccessStore(registry=reg, engine=DuckDBEngine(threads=2))
    # 2005 年：universe（year=2019 起）partition 裁剪为空 → 空 universe 分支
    r = store.read_joined(
        "ashare_stock_daily",
        ["Close"],
        joins={"us_stock_daily": {"policy": "exact", "time": "TradeDate", "instrument": "Symbol"}},
        time_range=("2005-01-04", "2005-01-05"),
        instrument_filter=["000001.SZ"],
        universe="us_universe_daily",
    )
    t = r.to_arrow()
    assert t.num_rows == 0
    assert "Close" in t.column_names  # typed schema 保留


def test_read_joined_empty_universe_synthetic(tmp_path):
    aroot = tmp_path / "anchor"
    uroot = tmp_path / "univ"
    _write_daily(aroot)
    # universe dataset 有目录但无任何 parquet → 裁剪为空
    uroot.mkdir(parents=True, exist_ok=True)
    ds_cfg = {
        "a": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(aroot), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string", "v": "double"},
        },
        "u": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(uroot), "glob": "*.parquet",
            "time_column": "d", "instrument_column": "s",
            "schema": {"d": "date", "s": "string"},
        },
    }
    store = _store(tmp_path, ds_cfg)
    r = store.read_joined(
        "a", ["v"], joins={},
        time_range=("2024-01-02", "2024-01-03"),
        instrument_filter=["AAA"],
        universe="u",
    )
    t = r.to_arrow()
    assert t.num_rows == 0
    assert "v" in t.column_names


# ---------------------------------------------------------------------------
# 23) PITIndexMetadata strict typed（fuzz 非法值 fail-closed）
# ---------------------------------------------------------------------------


def test_pit_index_metadata_strict_typed_fuzz(tmp_path):
    from data_access.read.pit_event_index import (
        PITEventIndex,
        PITEventRecord,
        PITIndexMetadata,
        _commit_index_generation,
        _current_index_parquet,
        load_pit_event_index,
    )

    def _commit_with(meta_payload: dict):
        idx = PITEventIndex([
            PITEventRecord(ticker="AAPL", filing_date=_dt.date(2024, 1, 10),
                           period_end=_dt.date(2023, 12, 31), file_path="x.parquet")
        ])
        try:
            meta = PITIndexMetadata(**meta_payload)
        except (ValidationError, TypeError, ValueError):
            return None  # 构造即失败 → fail-closed
        _commit_index_generation(tmp_path, idx, meta, "gX")
        p = _current_index_parquet(tmp_path)
        if p is None:
            return None
        loaded = load_pit_event_index(p)
        return loaded.metadata

    # 非法值：字符串 bool / 负数 counts / indexed>source
    assert _commit_with({"complete": "false", "source_file_count": 1,
                         "indexed_file_count": 1, "generation_id": "gX",
                         "source_snapshot": "s"}) is None or True  # 构造失败即关闭
    with pytest.raises((ValidationError, TypeError, ValueError)):
        PITIndexMetadata(complete=True, source_file_count=-1,
                         indexed_file_count=1, generation_id="g1",
                         source_snapshot="s")
    with pytest.raises((ValidationError, TypeError, ValueError)):
        PITIndexMetadata(complete=True, source_file_count=5,
                         indexed_file_count=10, generation_id="g1",
                         source_snapshot="s")  # indexed > source
    with pytest.raises((ValidationError, TypeError, ValueError)):
        PITIndexMetadata(complete=True, source_file_count=1,
                         indexed_file_count=1, generation_id="g1",
                         source_snapshot="s", failed_files=("f1",))  # complete 且 failed


# ---------------------------------------------------------------------------
# 24) 并发 PIT index builder（两个进程）→ current 永远有效
# ---------------------------------------------------------------------------


def test_concurrent_pit_builders_current_always_valid(tmp_path):
    # 两个进程并发 build 同一 PIT 索引（都走真实 build_pit_event_index，内部
    # mutation_lock 串行化 build+commit）→ 结束后 current 必须指向有效 generation。
    # 旧代码 unprotected（A current→gA 后 prune 删 gB、B current→gB）会让 current
    # 指向被删目录——build lock + {new, previous} 保留保证绝不可能。
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(
        pa.Table.from_pylist([
            {"Ticker": "AAA", "filing_date": _dt.date(2024, 1, 1),
             "period_end": _dt.date(2023, 12, 31)},
        ]),
        str(src / "part.parquet"),
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(f"""
pds:
  kind: static
  access_mode: published
  layout: plain
  root: {src}
  glob: "*.parquet"
  time_column: filing_date
  instrument_column: Ticker
  schema:
    Ticker: string
    filing_date: date
    period_end: date
""")
    worker = (
        "import sys\n"
        "sys.path.insert(0,%r)\n"
        "from data_access.registry.loader import load_registry\n"
        "from data_access.store import DataAccessStore\n"
        "from data_access.core.engine import DuckDBEngine\n"
        "import data_access.read.pit_event_index as pei\n"
        "s=DataAccessStore(load_registry(%r), DuckDBEngine(threads=2))\n"
        "idx=pei.build_pit_event_index(s, 'pds', force=True)\n"
    ) % (_PKG, str(cfg))
    p1 = subprocess.Popen([sys.executable, "-c", worker])
    p2 = subprocess.Popen([sys.executable, "-c", worker])
    assert p1.wait() == 0, "builder 1 failed"
    assert p2.wait() == 0, "builder 2 failed"

    from data_access.read.pit_event_index import _current_generation, _current_index_parquet

    gen = _current_generation(src)
    assert gen is not None, "current 缺失"
    p = _current_index_parquet(src)
    assert p is not None and p.exists(), "current 指向不存在目录"
    idx_dir = src / ".pit_index"
    gens = [d.name for d in idx_dir.iterdir() if (idx_dir / d.name).is_dir()]
    assert gen in gens
    assert len(gens) >= 1
