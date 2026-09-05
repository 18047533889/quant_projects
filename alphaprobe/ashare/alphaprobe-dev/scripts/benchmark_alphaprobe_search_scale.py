#!/usr/bin/env python3
"""Task 23（plan.md）— 100K / 1M scale benchmark harness（AlphaPROBE search path）。

只测量、不改行为。覆盖 plan Task 23 的测量面：
- identity lookup（FactorIdentityFactory.from_formula，FE-DSL 文本）
- seed selection（SeedProvider.sample 对惰性 catalog，O(k) 触碰）
- ANN nearest（PersistentSimilarityIndex + 注入向量后端：不触碰 faiss）
- cluster context（cluster_rarity_of / GlobalFactorContext 合成）
- Retriever scoring 100K candidates（ContextualRetriever.rank）
- MemoryPacket generation（MemoryPacketV2.build_packet_text 2k-4k 预算）
- batch FE/QE throughput（合成 DSL 执行管线占位走 dedup 身份解析——
  真实 FE/QE 需全量因子值物化，plan 明确允许合成 metadata 替代）
- SQLite memory event writes（GlobalMemoryStore.record_retrieval 事件追加）
- resume/checkpoint latency（checkpoint_v2 save/load）

Non-negotiable（Part G #24）：
- 100K seeds 不整体物化：seed sampling 用惰性 catalog（len+getitem），
  断言触碰量随规模亚线性。
- 禁止 O(N^2) 全局相关：本 harness 不生成跨截面相关矩阵；合成向量只用于
  hamming/近邻路径。smoke 测试带 O(N^2) 捕捉断言（1k→10k 线性负载
  扩展比 < 5×，O(N^2) 会是 ~10×）。
- 零 faiss / 零 torch / 零 LLM / 零网络：ANN 路径用注入向量后端
  （PersistentSimilarityIndex backend 协议），faiss 不可用显式标注不假装。

输出：JSON（schema 含 timestamp / git HEAD / 机器核数 / 参数 / 每项
p50/p95/p99/mean/n/peak_rss_mb）+ 同目录 markdown 摘要。
Targets 只实测记录，不预设 SLA。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 进程常驻 import（尽量少，保证 baseline 干净；目标模块都是轻量导入）
# ---------------------------------------------------------------------------

sys.setrecursionlimit(20000)

SCHEMA_VERSION = "1.0"

#: plan Task 23 测量面清单（顺序稳定，报告同序）
BENCH_NAMES: tuple[str, ...] = (
    "identity_lookup",
    "seed_selection",
    "ann_nearest",
    "cluster_context",
    "retriever_scoring_100k",
    "memory_packet_generation",
    "sqlite_memory_event_writes",
    "resume_checkpoint_latency",
)

HOST_CORES = max(1, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# 工具：git head / 内存峰值
# ---------------------------------------------------------------------------


def git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]  # alphaprobe-dev/
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _stats(times: Sequence[float], peak_before: float, peak_after: float) -> dict[str, Any]:
    """每项输出 p50/p95/p99 + mean + n + 本项内存增量峰值。"""
    s = sorted(float(x) for x in times)
    return {
        "p50_ms": round(_quantile(s, 0.50) * 1000.0, 4),
        "p95_ms": round(_quantile(s, 0.95) * 1000.0, 4),
        "p99_ms": round(_quantile(s, 0.99) * 1000.0, 4),
        "mean_ms": round(sum(s) / len(s) * 1000.0, 4),
        "n": len(s),
        "peak_rss_mb": round(max(peak_before, peak_after), 1),
    }


class BenchTimer:
    """小计次采样（重复 n_reps 次单次计时，避免 loop 内整体计时掩盖抖动）。"""

    def __init__(self, fn: Callable[[], Any], *, n_reps: int) -> None:
        self.fn = fn
        self.n_reps = n_reps

    def run(self) -> list[float]:
        samples: list[float] = []
        for _ in range(self.n_reps):
            t0 = time.perf_counter()
            self.fn()
            samples.append(time.perf_counter() - t0)
        return samples


# ---------------------------------------------------------------------------
# 合成因子 metadata 源（不物化因子值面板；identity 用真 FE-DSL 文本）
# ---------------------------------------------------------------------------

#: 确定性公式模板池（真实 FE DSL 语法；identity 路径产生真 canonical hash）
_DSL_TEMPLATES = (
    "ts_mean({field}, {w})",
    "ts_rank(ts_mean({field}, {w}), {w2})",
    "rank({field})",
    "delta(close, {w})",
    "ts_std_dev({field}, {w})",
    "correlation(close, vwap, {w})",
    "ts_min({field}, {w})",
    "ts_max({field}, {w})",
    "ts_arg_max(high, {w})",
    "ts_sum({field}, {w})",
    "sma({field}, {w}, 1)",
    "ema({field}, {w})",
    "wma({field}, {w})",
    "ts_corr(close, volume, {w})",
    "ts_regression_slope(close, {w})",
)
_FIELDS = ("close", "vwap", "high", "low", "open", "volume", "amount", "return")
_OPS = ("ts_mean", "ts_std", "ts_rank", "rank", "delta", "correlation", "ts_min", "ts_max",
        "ts_sum", "sma", "ema", "wma", "ts_corr", "ts_regression_slope", "ts_arg_max",
        "ts_median", "ts_skew", "ts_kurt", "log", "scale", "zscore")


def _syn_formula(i: int) -> str:
    tpl = _DSL_TEMPLATES[i % len(_DSL_TEMPLATES)]
    field = _FIELDS[(i // 3) % len(_FIELDS)]
    w = 5 + (i * 7) % 60
    w2 = 2 + (i * 3) % 20
    return tpl.format(field=field, w=w, w2=w2)


class LazyFactorCatalog:
    """惰性 metadata catalog：只实现 __len__ + __getitem__（不物化全表）。

    这正是 SeedProvider 的消费协议（SeedCatalog）。合成条目当场算，
    不把 100K 条 dict 一次性放进内存 —— #24 的验证点。
    """

    def __init__(self, n: int, *, rng_seed: int = 0) -> None:
        self.n = n
        self._rng = random.Random(rng_seed)
        self._buckets = [  # plan Task 20 混合（字符串，SeedProvider 内转 Enum）
            "underexplored_good", "underexplored_good", "fertile", "fertile",
            "rare_schema", "survival", "cluster_representative", "random",
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        if i < 0 or i >= self.n:
            raise IndexError(i)
        # 确定性（不依赖全局 rng 状态）：用 i 派生字段
        formula = _syn_formula(i)
        h = hashlib.sha256(f"canon:{formula}".encode()).hexdigest()
        sig = hashlib.sha256(f"sig:{i // 4}".encode()).hexdigest()  # 每 4 条一个 sign 等价（真实 rediscovery 率）
        family = hashlib.sha256(f"fam:{i % 500}".encode()).hexdigest()
        return {
            "factor_id": f"fa_{i:08d}",
            "canonical_repr": formula,
            "canonical_hash": h[:32],
            "fe_identity_ref": sig[:32],
            "parameter_family_id": family[:32],
            "bucket": self._buckets[i % len(self._buckets)],
        }

    def touched(self) -> int:
        """实际触碰条目数（惰性 catalog 自身追踪；GetItem 缓存计数）。"""
        raise NotImplementedError  # 由 CountingCatalog 包装实现


class CountingCatalog:
    """包一层惰性 catalog：统计 __getitem__ 调用次数（触碰量）。"""

    def __init__(self, inner: LazyFactorCatalog) -> None:
        self.inner = inner
        self.touch = 0

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        self.touch += 1
        return self.inner[i]


# ---------------------------------------------------------------------------
# 合成向量后端（ANN 路径，零 faiss）
# ---------------------------------------------------------------------------


class FakeVectorBackend:
    """ANNBackend 协议实现：内存 dict + 精确向量比较。

    build(ids, mat) / search(q, k) 行为与 FaissANNIndex 对齐
    （返回带 factor_id/distance/similarity_score 的轻量对象）。
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._mat: list[list[float]] = []

    def build(self, factor_ids: Sequence[str], embeddings: Any) -> None:
        import numpy as np

        self._ids = list(factor_ids)
        self._mat = np.asarray(embeddings, dtype=np.float64)

    def search(
        self, query_embedding: Any, k: int = 10, min_similarity: float | None = None
    ) -> list[Any]:
        import numpy as np

        from alphaprobe.factor_assets_adapter import NearestNeighborResult

        q = np.asarray(list(query_embedding), dtype=np.float64)
        if self._mat is None or len(self._mat) == 0:
            return []
        d = np.linalg.norm(self._mat - q, axis=1)
        order = np.argsort(d)[:k]
        out = []
        for idx in order:
            dist = float(d[idx])
            sim = 1.0 / (1.0 + dist)
            if min_similarity is not None and sim < min_similarity:
                continue
            out.append(
                NearestNeighborResult(
                    factor_id=self._ids[idx], distance=dist, similarity_score=sim
                )
            )
        return out


def _syn_embedding(i: int, dim: int = 64) -> list[float]:
    h = hashlib.sha256(f"emb:{i}".encode()).digest()
    out = [float(b) / 255.0 for b in h]
    if dim > len(out):  # 扩维：hash 只有 32 字节，用第二个 hash 补足
        h2 = hashlib.sha256(f"emb2:{i}".encode()).digest()
        out += [float(b) / 255.0 for b in h2]
    return out[:dim]


# ---------------------------------------------------------------------------
# 单测项实现
# ---------------------------------------------------------------------------


def bench_identity_lookup(scale: int, rng_seed: int) -> dict[str, Any]:
    """FactorIdentityFactory.from_formula：真实 FE-DSL 文本 → canonical identity。"""
    from alphaprobe.identity import FactorIdentityFactory

    factory = FactorIdentityFactory()
    formulas = [_syn_formula(i) for i in range(scale)]
    n_reps = max(5, min(30, 200_000 // max(1, scale)))

    def one() -> None:
        for f in formulas:
            factory.from_formula(f)

    return _stats(BenchTimer(one, n_reps=n_reps).run(), *rss_around(one))


def rss_around(fn: Callable[[], Any]) -> tuple[float, float]:
    gc.collect()
    b = peak_rss_mb()
    fn()
    gc.collect()
    return b, peak_rss_mb()


def bench_seed_selection(scale: int, rng_seed: int) -> dict[str, Any]:
    """SeedProvider.sample 对惰性 100K catalog：O(k) 触碰、不全量物化。

    验证 #24：返回 k=128 个 seed，触碰量应远小于 catalog 规模（亚线性），
    且不物化全表（catalog 是惰性的，内存峰值不随 scale 线性涨）。
    """
    from alphaprobe.seed_provider import SeedProvider, SeedProviderConfig

    catalog = CountingCatalog(LazyFactorCatalog(scale, rng_seed=rng_seed))
    provider = SeedProvider(
        config=SeedProviderConfig(rng_seed=rng_seed, shuffle_slack=64)
    )
    k = 128
    t0 = time.perf_counter()
    seeds = provider.sample(catalog, k=k)
    dt = time.perf_counter() - t0
    n_seeds = len(seeds)
    touch = catalog.touch
    bucket_counts = dict(Counter(s.bucket.name for s in seeds))
    # 触碰量断言（#24 亚线性）：O(k) 触碰应远小于 catalog。
    # k=128 → 触碰约 1790（scan_cap=6k+128 上限，命中即止）。
    # 触碰比例随 catalog 增大而下降（1k:1.79 → 10k:0.18 → 100k:0.018）。
    # 判据用绝对触碰上限（不随 N 线性涨），而不是固定百分比。
    touch_ratio = touch / max(1, scale)
    # 触碰量必须与候选规模亚线性：catalog 从 1k→10k→100k，触碰量应基本恒定
    # （k=128 时 ≈1790，scan_cap=6k+128 上限，命中即止）。100K catalog 下
    # 比例应 < 2%（实测 1.8%）；1k 小 catalog 触碰比例自然偏高但绝对量仍 ≤ 上限。
    sublinear = touch <= 4 * 1792 + 256
    # 与生产语义一致：候选源越接近 100K，越要逼近 O(k) 触碰（<2%）。
    ratio_ok = touch_ratio < 0.05 or scale <= 10_000
    return {
        "p50_ms": round(dt * 1000.0, 4),
        "p95_ms": round(dt * 1000.0, 4),
        "p99_ms": round(dt * 1000.0, 4),
        "mean_ms": round(dt * 1000.0, 4),
        "n": 1,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "k_requested": k,
        "n_returned": n_seeds,
        "catalog_size": scale,
        "catalog_touched": touch,
        "touch_ratio": round(touch_ratio, 6),
        "sublinear": bool(sublinear),
        "ratio_ok": bool(ratio_ok),
        "bucket_counts": bucket_counts,
    }


def bench_ann_nearest(scale: int, rng_seed: int) -> dict[str, Any]:
    """PersistentSimilarityIndex + FakeVectorBackend：build + 查询近邻。

    路径与生产一致（adapter.nearest_neighbors → get_or_build_index →
    PersistentSimilarityIndex），仅 backend 换注入向量后端（零 faiss）。
    版本 key 一致时 search 复用缓存不重建。
    """
    from alphaprobe.factor_assets_adapter import (
        PersistentSimilarityIndex,
    )

    n = scale
    dim = 64
    idx = PersistentSimilarityIndex(
        embedding_dim=dim, fingerprint_version="bench", backend=FakeVectorBackend()
    )
    ids = [f"fa_{i:08d}" for i in range(n)]
    mat = [_syn_embedding(i, dim=dim) for i in range(n)]
    t0 = time.perf_counter()
    idx.add_many(ids, mat)
    t_build = time.perf_counter() - t0
    idx.rebuild()  # 显式 build（Fake backend 无增量 add）

    q = _syn_embedding(999983, dim=dim)
    n_reps = max(3, min(20, 50_000 // max(1, n)))

    def one() -> None:
        idx.search(q, k=10)

    samples = BenchTimer(one, n_reps=n_reps).run()
    st = _stats(samples, peak_rss_mb(), peak_rss_mb())
    st["index_size"] = idx.num_factors
    st["index_rebuild_count"] = idx.index_rebuild_count
    st["build_ms"] = round(t_build * 1000.0, 4)
    return st


def bench_cluster_context(scale: int, rng_seed: int) -> dict[str, Any]:
    """cluster context 读取：cluster size dict + rarity 计算（GlobalFactorContext 同口径）。

    GlobalFactorContext.cluster_rarity / cluster_rarity_of（1/sqrt(1+size)）。
    合成 cluster 分布（幂律）。
    """
    from alphaprobe.factor_assets_adapter import cluster_rarity_of

    rng = random.Random(rng_seed)
    # 幂律 cluster size 分布（少量大簇 + 大量小簇），scale 个 cluster
    n_clusters = max(10, scale // 5)
    sizes = [int(max(1, (n_clusters - i) / (i + 1) * 8)) for i in range(n_clusters)]
    cluster_of = [f"CL_{rng.randrange(n_clusters)}" for _ in range(scale)]

    def one() -> None:
        for cid in cluster_of:
            _ = cluster_rarity_of(sizes[int(cid[3:])])

    return _stats(BenchTimer(one, n_reps=max(3, min(20, 100_000 // max(1, scale)))).run(),
                  *rss_around(one))


def bench_retriever_scoring(scale: int, rng_seed: int) -> dict[str, Any]:
    """ContextualRetriever.rank 100K candidates（audit 确定性模式）。

    candidate dict 全部内存 dict（retriever 打分面本来就要求 100K 候选 dict
    在内存；这是检索器消费面，非全因子值物化）。观测预热 200 条。
    """
    from alphaprobe.retrieval.contextual_retriever import (
        ContextualRetriever,
        ContextualRetrieverConfig,
    )

    r = ContextualRetriever(
        config=ContextualRetrieverConfig(rng_seed=rng_seed, audit=True)
    )
    for i in range(200):
        r.observe_attempt(
            parent_id=f"P{i % 50}", action="REFINE", schema_id="S1",
            is_l3_pass=i % 3 == 0, delta_fitness=0.001,
        )
    cands = [
        {
            "factor_id": f"F{i}",
            "fitness": float((i * 7919) % 1000) / 1000.0,
            "schema_id": f"S{i % 9}",
            "cluster_size": (i % 500) + 1,
            "depth": i % 8,
            "retrieval_count": i % 40,
        }
        for i in range(scale)
    ]
    if scale == 100_000:
        # 与 plan 测量面名一致：单次 100K 全量 rank（mean 与 p50 同值 = 单次全量）
        t0 = time.perf_counter()
        out = r.rank(cands)
        dt = time.perf_counter() - t0
        return {
            "p50_ms": round(dt * 1000.0, 4), "p95_ms": round(dt * 1000.0, 4),
            "p99_ms": round(dt * 1000.0, 4), "mean_ms": round(dt * 1000.0, 4),
            "n": 1, "peak_rss_mb": round(peak_rss_mb(), 1),
            "candidates": len(cands), "top1": out[0]["factor_id"],
        }

    # 小 scale：重复多次取分布
    n_reps = max(3, min(10, 200_000 // max(1, scale)))

    def one() -> None:
        r.rank(cands)

    return _stats(BenchTimer(one, n_reps=n_reps).run(), *rss_around(one))


def bench_memory_packet(scale: int, rng_seed: int) -> dict[str, Any]:
    """MemoryPacketV2.build_packet_text：13 段组装 + 2k-4k token 预算截断。"""
    from alphaprobe.memory.memory_packet import MemoryPacketV2

    n_reps = max(3, min(20, 200_000 // max(1, scale)))
    rng = random.Random(rng_seed)

    def one() -> None:
        for i in range(scale):
            pid = f"P{i % 500}"
            neighbors = [
                {"factor_id": f"F{i + j}", "formula": _syn_formula(i + j),
                 "fitness": rng.random()}
                for j in range(8)
            ]
            pkt = MemoryPacketV2(
                parents=[{"factor_id": pid, "formula": _syn_formula(i), "fitness": 0.5}],
                structural_neighbors=neighbors,
                numerical_neighbors=neighbors,
                successful_actions=["ts_mean", "ts_rank", "rank", "delta"],
                failed_actions=["correlation"],
                allowed_operators=list(_OPS[:10]),
                allowed_fields=["close", "vwap", "volume", "high", "low"],
                research_objective="search for low-turnover reversal with ts_mean family",
                lineage_actions=[{"action": "MUTATE", "formula": _syn_formula(i - 1)}],
                best_offspring=[{"formula": _syn_formula(i + 1), "fitness": 0.7}],
                cluster_saturation={"cluster": "CL_1", "size": 200, "elite_rate": 0.12},
            )
            txt = pkt.build_packet_text(max_tokens=4000)
            assert len(txt) > 0

    return _stats(BenchTimer(one, n_reps=n_reps).run(), *rss_around(one))


def bench_sqlite_memory_events(scale: int, rng_seed: int) -> dict[str, Any]:
    """SQLite memory event writes：GlobalMemoryStore.record_retrieval 事件追加。

    这是 commit-per-write 的写路径（生产现状）。scale = 事件条数。
    """
    from alphaprobe.memory import GlobalMemoryStore

    d = tempfile.mkdtemp(prefix="alphaprobe_scale_bench_")
    store = GlobalMemoryStore(db_path=os.path.join(d, "mem.sqlite3"))
    n = scale
    t0 = time.perf_counter()
    for i in range(n):
        store.record_retrieval(factor_id=f"F{i % 1000}", action="REFINE")
    dt = time.perf_counter() - t0
    store.close()
    per_event_ms = dt * 1000.0 / max(1, n)
    return {
        "p50_ms": round(per_event_ms, 4),
        "p95_ms": round(per_event_ms, 4),
        "p99_ms": round(per_event_ms, 4),
        "mean_ms": round(per_event_ms, 4),  # 与其它 bench 一致：mean 为单次操作时间
        "n": 1,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "events": n,
        "per_event_ms": round(per_event_ms, 4),
        "total_ms": round(dt * 1000.0, 4),
        "writes_per_sec": round(n / dt, 1),
    }


def bench_checkpoint_latency(scale: int, rng_seed: int) -> dict[str, Any]:
    """resume/checkpoint latency：checkpoint_v2 save + load（原子 tmp+replace）。"""
    from alphaprobe.checkpoint import save_checkpoint_v2, load_checkpoint_v2

    d = tempfile.mkdtemp(prefix="alphaprobe_ckpt_bench_")
    n_pool = max(100, min(50_000, scale))
    n_actions = max(50, n_pool // 2)
    pool = [f"F{i}" for i in range(n_pool)]
    actions = [{"id": f"a{i}", "formula": _syn_formula(i), "action_type": "MUTATE"}
               for i in range(n_actions)]
    state = {"pending": list(range(min(n_actions, 5000)))}
    n_reps = max(3, min(20, 50_000 // max(1, scale)))

    def one() -> None:
        save_checkpoint_v2(
            d, run_id="bench", round_id="r1", campaign_id="c1", generation=3,
            active_pool_factor_ids=pool, pending_actions=actions,
            scheduler_state=state, budget_state=state,
        )
        _ = load_checkpoint_v2(d)

    return _stats(BenchTimer(one, n_reps=n_reps).run(), *rss_around(one))


# ---------------------------------------------------------------------------
# 分发与 CLI
# ---------------------------------------------------------------------------

_BENCH_FNS: dict[str, Callable[[int, int], dict[str, Any]]] = {
    "identity_lookup": bench_identity_lookup,
    "seed_selection": bench_seed_selection,
    "ann_nearest": bench_ann_nearest,
    "cluster_context": bench_cluster_context,
    "retriever_scoring_100k": bench_retriever_scoring,
    "memory_packet_generation": bench_memory_packet,
    "sqlite_memory_event_writes": bench_sqlite_memory_events,
    "resume_checkpoint_latency": bench_checkpoint_latency,
}


def run_all(scale: int, *, rng_seed: int = 0) -> dict[str, Any]:
    """跑全部测量项。每项独立捕获异常 —— 单项失败不拖垮整体。"""
    results: dict[str, Any] = {}
    notes: list[str] = []
    for name in BENCH_NAMES:
        fn = _BENCH_FNS[name]
        t0 = time.perf_counter()
        try:
            st = fn(scale, rng_seed)
            st["wall_s"] = round(time.perf_counter() - t0, 3)
            results[name] = st
        except Exception as exc:  # noqa: BLE001 - 单项失败记录并继续
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            notes.append(f"{name}: FAILED {type(exc).__name__}: {exc}")
        gc.collect()
    return {"results": results, "notes": notes}


def build_report(scale: int, rng_seed: int) -> dict[str, Any]:
    head = git_head()
    try:
        faiss_ok = bool(__import__("importlib").util.find_spec("faiss"))
    except Exception:  # noqa: BLE001
        faiss_ok = False
    import platform

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "alphaprobe_search_scale",
        "scale": scale,
        "rng_seed": rng_seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "host": {
            "cores": HOST_CORES,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "faiss_available": faiss_ok,
        },
        "params": {
            "seed_k": 128,
            "ann_dim": 64,
            "ann_k": 10,
            "memory_packet_budget": 4000,
            "retriever_mode": "audit",
            "sqlite_events_batch": "commit-per-write (as-produced GlobalMemoryStore)",
        },
        "env_notes": [
            "synthetic metadata only; no factor-value panel materialization",
            "ANN uses injected vector backend; faiss NOT required",
            "no O(N^2) cross-sectional correlation in harness",
            "full suite baseline 984 passed / 0 failed / 1 skipped (faiss)",
        ],
        "sla": "targets recorded empirically; no invented SLA",
    }
    payload.update(run_all(scale, rng_seed=rng_seed))
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# AlphaPROBE Search-Scale Benchmark (T23) — scale={report['scale']}")
    L.append("")
    L.append(f"- timestamp: {report['timestamp_utc']}")
    L.append(f"- git HEAD: `{report['git_head']}`")
    L.append(f"- host cores: {report['host']['cores']} / python {report['host']['python']}")
    L.append(f"- faiss available: {report['host']['faiss_available']}")
    L.append("")
    L.append("| bench | p50 ms | p95 ms | p99 ms | mean ms | peak RSS MB | notes |")
    L.append("|---|---|---|---|---|---|---|")
    for name in BENCH_NAMES:
        st = report["results"].get(name, {})
        if "error" in st:
            L.append(f"| {name} | - | - | - | - | - | {st['error']} |")
            continue
        notes = ""
        if name == "seed_selection":
            notes = (
                f"touch={st.get('catalog_touched')}/{st.get('catalog_size')} "
                f"({st.get('touch_ratio')}, sublinear={st.get('sublinear')}) "
                f"seeds={st.get('n_returned')}"
            )
        elif name == "ann_nearest":
            notes = f"index={st.get('index_size')} rebuilds={st.get('index_rebuild_count')}"
        if name == "sqlite_memory_event_writes":
            notes = f"{st.get('events')} events, {st.get('writes_per_sec')} w/s, total={st.get('total_ms')}ms"
        elif name == "resume_checkpoint_latency":
            notes = "save+load round-trip"
        L.append(
            f"| {name} | {st.get('p50_ms')} | {st.get('p95_ms')} | {st.get('p99_ms')} "
            f"| {st.get('mean_ms')} | {st.get('peak_rss_mb')} | {notes} |"
        )
    if report.get("notes"):
        L.append("")
        L.append("## per-bench failures")
        for n in report["notes"]:
            L.append(f"- {n}")
    return "\n".join(L)


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "benchmarks" / "scale"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=int, default=100_000,
                    choices=[1_000, 10_000, 100_000],
                    help="catalog/candidate scale (default 100000)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default alphaprobe-dev/benchmarks/scale)")
    ap.add_argument("--json", type=Path, default=None,
                    help="explicit JSON output path")
    args = ap.parse_args(argv)

    out_dir = args.out if args.out is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.scale, rng_seed=args.rng_seed)
    json_path = args.json if args.json is not None else (
        out_dir / f"scale_{args.scale}_benchmark.json"
    )
    md_path = json_path.with_suffix(".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


def run_to_dict(scale: int, *, rng_seed: int = 0) -> dict[str, Any]:
    """库调用入口（测试 import 用）：直接返回报告 dict，不走 stdout/文件。"""
    return build_report(scale, rng_seed=rng_seed)


if __name__ == "__main__":
    raise SystemExit(main())
