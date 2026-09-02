"""Phase D —— cold-start seed ingest 测试。

覆盖：
① 小样本（30 条）ingest 幂等：跑两遍新增=第二遍 0
② parse 失败条目进 audit 不中断
③ 六类计数守恒：raw = fresh_new + family_merge + exact + sign + parse_fail
④ benchmark 快速模式可调用、返回结构齐全（小 N）

DB 全部走 tmp_path 隔离。FE identity 需 quant_projects 引导；CI 环境 FE 不可达
则 pytest.skip（不让整套测试因此变红）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from alphaprobe.authority import FactorIdentityAuthorityError, ensure_authority_available

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_cold_start", _SRC_DIR.parent / "scripts" / "ingest_cold_start.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ingest = _load_script_module()

_SEED_PATH = Path(
    "/home/sunhaiwei/quant_projects/cold_start_library/library/factorengine_cold_start_backend_audited_v9.json"
)

try:
    ensure_authority_available()
    _FE_OK = True
except FactorIdentityAuthorityError:
    _FE_OK = False

try:
    from alphaprobe.seen import build_seen_index
    from alphaprobe.seen.dedup_service import DedupService
except Exception:  # noqa: BLE001
    DedupService = None
    build_seen_index = None

_FE_SKIP = pytest.mark.skipif(
    not (_FE_OK and DedupService is not None),
    reason="factor_engine.identity or seen unavailable in CI",
)


def _load_seed(limit: int = 30) -> list[dict]:
    with open(_SEED_PATH, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return d[:limit] if limit else d


def _build_svc(tmp_path, name: str):
    return build_seen_index(str(tmp_path / name))


@_FE_SKIP
def test_idempotent_ingest_small(tmp_path):
    svc = _build_svc(tmp_path, "idem.sqlite3")
    seed = _load_seed(30)
    a1 = ingest.run_ingest(svc, seed, limit=None)
    a2 = ingest.run_ingest(svc, seed, limit=None)
    assert a2["fresh_new"] == 0
    assert a2["family_merge"] == 0
    assert a1["conservation"]["holds"] is True
    svc._store.close()


@_FE_SKIP
def test_parse_failure_isolated_not_broken(tmp_path):
    svc = _build_svc(tmp_path, "parsefail.sqlite3")
    seed = [
        {"factor_id": "bad_1", "status": "engine_ready", "formula_canonical": "rank("},
        {"factor_id": "bad_2", "status": "engine_ready",
         "formula_canonical": "ts_mean(close,20) ~ definitely_bad("},
        *_load_seed(10),
    ]
    a = ingest.run_ingest(svc, seed, limit=None)
    assert a["fresh_new"] + a["family_merge"] + a["exact"] + a["sign"] > 0
    assert a["parse_fail"] >= 1
    assert len(a["parse_failure_samples"]) >= 1
    assert a["conservation"]["holds"] is True
    svc._store.close()


@_FE_SKIP
def test_conservation_six_buckets(tmp_path):
    svc = _build_svc(tmp_path, "cons.sqlite3")
    seed = _load_seed(30)
    a = ingest.run_ingest(svc, seed, limit=None)
    assert a["conservation"]["holds"] is True
    buckets_sum = a["fresh_new"] + a["family_merge"] + a["exact"] + a["sign"] + a["parse_fail"]
    assert buckets_sum == a["raw"] == len(seed)
    svc._store.close()


def test_benchmark_quick():
    # benchmark 快速模式（小 N）：依赖 seen 但不用 FE（合成串）。seen 不可达则 skip。
    import pytest as _pt

    if DedupService is None:
        _pt.skip("seen unavailable")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        svc = build_seen_index(str(Path(td) / "bench.sqlite3"))
        forms = ["rank(ts_sum(close,3))", "ts_mean(vol,20)", "safe_div(close,pre_close)"]
        res = ingest.run_benchmark(svc, forms, bench_n=20)
        assert "ingest" in res and "lookup" in res
        assert set(res["lookup"].keys()) == {"exact_ms", "family_ms", "nearest_ms"}
        assert res["lookup"]["exact_ms"]["n"] >= 0
        assert res["ingest"]["throughput_items_per_sec"] >= 0
        svc._store.close()
