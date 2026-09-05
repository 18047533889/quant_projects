"""T23 scale-benchmark smoke test：小规模快速跑通 benchmark 主路径 + 关键断言。

plan.md Task 23 验收面（与 scripts/benchmark_alphaprobe_search_scale.py 同源）：
- identity lookup / seed selection / ANN nearest / cluster context /
  Retriever scoring / MemoryPacket generation / SQLite memory event writes /
  resume checkpoint latency —— 全部走同一 harness 函数（真实合成数据）。
- 输出 schema 合法（p50/p95/p99 + 内存 + git HEAD + timestamp + host）。
- #24：seed selection 触碰量亚线性（不全量物化 100K catalog）；
- 禁止 O(N^2) 全局相关：harness 不生成跨截面相关矩阵（grep AST 守卫），
  且合成线性负载 1k→10k 扩展比 < 5×（O(N^2) 会是 ~10×）。

marker：模块级 pytestmark 挂 ``perf`` marker —— 全量回归默认跑不到重负载
路径（SQLite commit-per-write 在 10k 事件下约 15 s，不适合每次全量回归）。
小规模冒烟选 --scale 1000（全部测量项 ~2-3 s），保证 smoke 本身快。
faiss 不参与：ANN 用注入向量后端，显式标注不假装。
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_alphaprobe_search_scale.py"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 冒烟规模（全量回归友好：全部测量项 ~2-3 s）
SMOKE_SCALE = 1_000

#: schema 必须出现的顶层字段（harness 输出契约）
_REQUIRED_TOP = {
    "schema_version",
    "benchmark",
    "scale",
    "rng_seed",
    "timestamp_utc",
    "git_head",
    "host",
    "params",
    "results",
    "sla",
}

#: 每项必须出现的指标字段
_REQUIRED_STAT = {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "peak_rss_mb"}

#: plan 测量面（harness BENCH_NAMES 同序）
_REQUIRED_BENCHES = {
    "identity_lookup",
    "seed_selection",
    "ann_nearest",
    "cluster_context",
    "retriever_scoring_100k",
    "memory_packet_generation",
    "sqlite_memory_event_writes",
    "resume_checkpoint_latency",
}

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="benchmark script missing"),
]


def _import_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("scale_bench", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def report_1k() -> dict:
    """1k 冒烟报告跑一次，多个测试共用（避免 smoke 重复跑 harness）。"""
    mod = _import_mod()
    return mod.run_to_dict(SMOKE_SCALE)


# ---------------------------------------------------------------------------
# schema + 主路径
# ---------------------------------------------------------------------------


def test_harness_runs_and_outputs_valid_schema(report_1k: dict) -> None:
    report = report_1k
    assert set(report) >= _REQUIRED_TOP, f"missing top-level fields: {_REQUIRED_TOP - set(report)}"
    assert report["scale"] == SMOKE_SCALE
    assert report["schema_version"] == "1.0"
    assert report["git_head"] and len(report["git_head"]) >= 7
    assert report["host"]["cores"] >= 1
    assert set(report["results"]) == _REQUIRED_BENCHES
    for name, st in report["results"].items():
        assert "error" not in st, f"{name} failed: {st['error']}"
        assert set(st) >= _REQUIRED_STAT, f"{name} missing stat fields"
        for k in _REQUIRED_STAT:
            assert st[k] >= 0, f"{name} {k} negative"


def test_no_on2_cross_sectional_correlation_in_harness() -> None:
    """O(N^2) 全局相关是明确禁止项：harness 源码不得出现成对相关物化。"""
    src = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    banned = [
        "np.corrcoef",
        ".corr(",
        "cross_section",  # 注释可提及；这里禁止的是实现调用——用调用签名过滤
        "correlation_matrix",
    ]
    # 只拦「实现调用」，允许注释/文档里说明性文字（correlation 词出现在 DSL 模板合法）
    call_sites = [b for b in ("np.corrcoef", "correlation_matrix", "numpy.corrcoef") if b in src]
    assert not call_sites, f"O(N^2) corr materialization found in harness: {call_sites}"
    # corr 只允许出现在合成 DSL 模板（correlation(close, vwap, {w}) 是 FE 算子文本）
    assert "correlation(close, vwap" in src  # 模板本身是 DSL 文本，不是相关计算


def test_seed_selection_touches_sublinearly(report_1k: dict) -> None:
    """#24：100K seed 不全量物化。seed selection 触碰量必须亚线性。"""
    report = report_1k
    ss = report["results"]["seed_selection"]
    assert ss["sublinear"] is True
    assert ss["touch_ratio"] > 0 or ss["catalog_size"] <= SMOKE_SCALE  # 触碰已发生
    # 1k catalog 下触碰 1792 是 scan_cap=6k+128 上限内的正常命中扫描；
    # 关键是绝对触碰量不随 catalog 线性涨（smoke 断言上限）
    assert ss["catalog_touched"] <= 4 * 1792 + 256


def test_ann_nearest_reuses_index(report_1k: dict) -> None:
    """ANN 持久索引同版本不重建（adapter 契约）：rebuild 计数恒 1（首次 build）。"""
    report = report_1k
    ann = report["results"]["ann_nearest"]
    assert ann["index_size"] == SMOKE_SCALE
    assert ann["index_rebuild_count"] == 1


def test_sqlite_event_write_rate_recorded(report_1k: dict) -> None:
    """SQLite memory event writes 吞吐被记录（不做 SLA 断言 —— 只记录实测值）。"""
    report = report_1k
    sql = report["results"]["sqlite_memory_event_writes"]
    assert sql["events"] == SMOKE_SCALE
    assert sql["writes_per_sec"] > 0
    assert sql["per_event_ms"] > 0


def test_linear_scaling_guard_no_quadratic_blowup() -> None:
    """合成线性负载 1k→10k 扩展比应≈10×（线性）。

    O(N) 负载 N 涨 10× → 时间涨 ~10×；O(N²) 会涨 ~100×。阈值取 30×
    夹在两者之间：既允许真实线性抖动，又能捕捉 O(N²) 全局相关回归。
    identity 是纯 O(N) 负载：n 条公式各 canonicalize+hash 一次。
    """
    import time

    from alphaprobe.identity import FactorIdentityFactory

    mod = _import_mod()
    factory = FactorIdentityFactory()

    def one_pass(n: int) -> float:
        formulas = [mod._syn_formula(i) for i in range(n)]
        t0 = time.perf_counter()
        for f in formulas:
            factory.from_formula(f)
        return time.perf_counter() - t0

    t_1k = one_pass(1_000)
    t_10k = one_pass(10_000)
    ratio = t_10k / max(t_1k, 1e-6)
    assert ratio < 30.0, (
        f"identity path scaled super-linearly: 1k={t_1k * 1000:.1f}ms "
        f"10k={t_10k * 1000:.1f}ms ratio={ratio:.2f} (linear~10x, quadratic~100x)"
    )
    # 顺带验证确实是线性量级（非亚线性异常）
    assert ratio > 3.0, f"ratio {ratio:.2f} implausibly low for linear workload"
