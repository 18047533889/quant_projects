"""P1-pool 过渡测试：AlphaKnowledgePool fitness_mode=search_fitness 驱动进化方向。

覆盖任务书验收点：
- search_fitness 模式 parent 选择分与 legacy 不同且单调合理（高 rankic + 低复杂度排前）
- leaf novelty top-K 聚合：构造 max≠mean 场景断言两模式排序不同
- embedding 缓存：池版本不变时第二次 search() 不重算（monkeypatch 计数 encode）
- legacy 模式回归不变

全部用 object.__new__ 构造最小 AlphaKnowledgePool（绕过 SentenceTransformer
初始化 / StockData 加载），search() 全流程离线可跑。
"""

from __future__ import annotations

import numpy as np
import torch

import pytest

from alphaprobe.trainer.pool import (
    AlphaKnowledgePool,
    _fitness_proxy,
    reset_shared_calibrator,
)


# ---------------------------------------------------------------------------
# 最小 fake 组件
# ---------------------------------------------------------------------------


class _DummyData:
    device = torch.device("cpu")


class _DummyExpr:
    def __init__(self, s: str) -> None:
        self.s = s

    def evaluate(self, data):
        rng = np.random.RandomState(hash(self.s) % (2 ** 31))
        return torch.tensor(rng.randn(8, 5), dtype=torch.float32)

    def __str__(self) -> str:
        return self.s


class _DummyTarget:
    def evaluate(self, data):
        return torch.tensor(np.random.RandomState(1).randn(8, 5), dtype=torch.float32)


class _Payload:
    def __init__(self, source: str, length: int, expression) -> None:
        self.source = source
        self.length = length
        self.expression = expression


class _Node:
    def __init__(self, expr, ic: float = 0.05, icir: float = 0.2, depth: int = 0) -> None:
        self.payload = _Payload(str(expr), len(str(expr)), expr)
        self.children = set()
        self.depth = depth
        self.times = 0
        self.ic = ic
        self.icir = icir
        self.topic = str(expr)
        self.description = str(expr)

    def as_dict(self) -> dict:
        return {}


class _DummyEncoder:
    """encode 调用计数（语义缓存验证用）。embedding 相似度=文本长度相似度。"""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True):
        self.encode_calls += 1
        n = len(texts)
        lengths = np.array([len(t) for t in texts], dtype=float)
        E = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                E[i][j] = 1.0 - abs(lengths[i] - lengths[j]) / max(1.0, float(max(lengths)))
        return torch.tensor(E, dtype=torch.float32)


def _make_pool(
    *,
    corrs: list[list[float]],
    ics: list[float],
    icirs: list[float],
    fitness_mode: str = "search_fitness",
    semantic: bool = False,
    encoder=None,
    top_k: int = 4,
    novelty_top_k: int = 5,
) -> AlphaKnowledgePool:
    pool = object.__new__(AlphaKnowledgePool)
    pool.capacity = len(corrs)
    pool.top_k = top_k
    pool.depth_decay = 0.05
    pool.times_decay = 0.1
    pool.start_times = 2
    pool.use_res_correlation = True
    pool.use_semantic_similarity = semantic
    pool.embedding_model = encoder
    pool.use_edit_distance = False
    pool.separate_leaf_non_leaf = False
    pool.fitness_mode = fitness_mode
    pool.novelty_top_k = novelty_top_k
    pool.knowledge_graph = None
    pool._pool_version = 0
    pool._semantic_cache = None
    pool._semantic_cache_valid = None
    pool._semantic_cache_pool_version = None
    n = len(corrs)
    pool.size = n
    pool.exprs = [None] * (n + 1)
    pool.values = [None] * (n + 1)
    pool.single_ics = np.zeros(n + 1)
    pool.mutual_ics = np.identity(n + 1)
    pool.weights = np.zeros(n + 1)
    pool.icir = [None] * (n + 1)
    pool.topics = [None] * (n + 1)
    pool.descriptions = [None] * (n + 1)
    pool.expr2node = [None] * (n + 1)
    for i in range(n):
        e = _DummyExpr(f"f{i}")
        pool.exprs[i] = e
        pool.expr2node[i] = _Node(e, ic=ics[i], icir=icirs[i])
        pool.single_ics[i] = ics[i]
        pool.icir[i] = icirs[i]
        pool.descriptions[i] = f"desc {i}"
        for j in range(n):
            pool.mutual_ics[i][j] = pool.mutual_ics[j][i] = corrs[i][j]
    pool.data = _DummyData()
    pool.target = _DummyTarget()
    return pool


@pytest.fixture(autouse=True)
def _clean_calibrator():
    reset_shared_calibrator()
    yield
    reset_shared_calibrator()


# ---------------------------------------------------------------------------
# 1. parent 选择分：search_fitness vs legacy 不同且单调合理
# ---------------------------------------------------------------------------


class TestParentSelection:
    def test_search_fitness_prefers_high_rankic_low_complexity(self):
        """高 rankic + 低复杂度 parent 排前（fitness 单调性）。"""
        pool = _make_pool(
            corrs=[[0, 0.5], [0.5, 0]],
            ics=[0.08, 0.02],
            icirs=[0.4, 0.05],
            fitness_mode="search_fitness",
        )
        # 高 rankic 节点（f0）payload 长=低复杂度；低 rankic 节点 f1 payload 长=高复杂度
        nodes, _ = pool.search()
        assert nodes[0].payload.source == "f0"
        # 直接验证 proxy 分数单调
        f_hi = _fitness_proxy(pool.expr2node[0], 0.4, single_ic=0.08, ast_nodes=2)
        f_lo = _fitness_proxy(pool.expr2node[1], 0.05, single_ic=0.02, ast_nodes=50)
        assert f_hi > f_lo

    def test_search_fitness_differs_from_legacy(self):
        """同一池：search_fitness 排序与 legacy 不同（打分内核确实切换了）。"""
        corrs = np.zeros((6, 6))
        # f0 近邻 [0.7,0.9,0.9,0.9,0.001]：mean novel 0.32, top3 novel 0.1
        # f1 近邻 [0.7,0.7,0.7,0.7,0.7]：mean novel 0.3, top3 novel 0.3
        # legacy(mean)：f0 novel 更高 → f0 在 f1 前；search_fitness(top3)：f1 前
        corrs[0][1] = 0.7
        corrs[0][2] = 0.9
        corrs[0][3] = 0.9
        corrs[0][4] = 0.9
        corrs[0][5] = 0.001
        corrs[1][2] = 0.7
        corrs[1][3] = 0.7
        corrs[1][4] = 0.7
        corrs[1][5] = 0.7
        for i in range(2, 6):
            for j in range(2, 6):
                if i != j:
                    corrs[i][j] = 0.5
        for i in range(6):
            for j in range(6):
                corrs[j][i] = corrs[i][j]
        np.fill_diagonal(corrs, 0.0)
        ics = [0.05] * 6
        icirs = [0.2] * 6
        pool_sf = _make_pool(
            corrs=corrs.tolist(), ics=ics, icirs=icirs,
            fitness_mode="search_fitness", novelty_top_k=3, top_k=6,
        )
        nodes_sf, _ = pool_sf.search()
        pool_leg = _make_pool(
            corrs=corrs.tolist(), ics=ics, icirs=icirs,
            fitness_mode="legacy", novelty_top_k=3, top_k=6,
        )
        nodes_leg, _ = pool_leg.search()
        order_sf = [n.payload.source for n in nodes_sf]
        order_leg = [n.payload.source for n in nodes_leg]
        assert order_sf != order_leg


# ---------------------------------------------------------------------------
# 2. leaf novelty top-K 聚合：max≠mean 场景两模式排序不同
# ---------------------------------------------------------------------------


class TestLeafNoveltyTopK:
    def test_novelty_topk_changes_ordering(self):
        """5 节点，1 个高相关簇 + 2 个低相关：search_fitness 的 top-K 聚合
        （1-max(topK corr)）会让「近邻都极高相关」的节点被压得更低。"""
        corrs = [
            [0, 0.9, 0.95, 0.95, 0.1],
            [0.9, 0, 0.95, 0.95, 0.1],
            [0.95, 0.95, 0, 0.9, 0.1],
            [0.95, 0.95, 0.9, 0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0],
        ]
        ics = [0.05, 0.05, 0.05, 0.05, 0.05]
        icirs = [0.2, 0.2, 0.2, 0.2, 0.2]
        pool_sf = _make_pool(
            corrs=corrs, ics=ics, icirs=icirs,
            fitness_mode="search_fitness", novelty_top_k=3,
        )
        nodes_sf, _ = pool_sf.search()
        # f4 与所有节点低相关（novelty 高）→ top1
        assert nodes_sf[0].payload.source == "f4"
        # 簇内（f0..f3）top-3 corr 均值都 >0.9 → novelty 低 → 排在 f4 之后
        assert all(n.payload.source != "f4" for n in nodes_sf[1:])

    def test_legacy_mean_vs_sf_topk_order_differs(self):
        """legacy（mean 聚合）与 search_fitness（top-K 聚合）对同一池给出不同
        novelty 排名：构造 mean 与 top-K 出现逆序的场景。"""
        corrs = np.zeros((6, 6))
        # f0 近邻 [0.7,0.9,0.9,0.9,0.001]：mean novel 0.32 > f1 的 0.3（legacy f0 前）
        # 但 top3 novel：f0 0.1 < f1 0.3（search_fitness f1 前）
        corrs[0][1] = 0.7
        corrs[0][2] = 0.9
        corrs[0][3] = 0.9
        corrs[0][4] = 0.9
        corrs[0][5] = 0.001
        corrs[1][2] = 0.7
        corrs[1][3] = 0.7
        corrs[1][4] = 0.7
        corrs[1][5] = 0.7
        for i in range(2, 6):
            for j in range(2, 6):
                if i != j:
                    corrs[i][j] = 0.5
        for i in range(6):
            for j in range(6):
                corrs[j][i] = corrs[i][j]
        np.fill_diagonal(corrs, 0.0)
        ics = [0.05] * 6
        icirs = [0.2] * 6
        pool_sf = _make_pool(
            corrs=corrs.tolist(), ics=ics, icirs=icirs,
            fitness_mode="search_fitness", novelty_top_k=3, top_k=6,
        )
        nodes_sf, _ = pool_sf.search()
        pool_leg = _make_pool(
            corrs=corrs.tolist(), ics=ics, icirs=icirs,
            fitness_mode="legacy", novelty_top_k=3, top_k=6,
        )
        nodes_leg, _ = pool_leg.search()
        order_sf = [n.payload.source for n in nodes_sf]
        order_leg = [n.payload.source for n in nodes_leg]
        assert order_sf != order_leg
        # 明确断言逆转：legacy 里 f0 在 f1 前，search_fitness 里 f1 在 f0 前
        assert order_leg.index("f0") < order_leg.index("f1")
        assert order_sf.index("f1") < order_sf.index("f0")


# ---------------------------------------------------------------------------
# 3. embedding 缓存：池版本不变第二次 search() 不重算
# ---------------------------------------------------------------------------


class TestSemanticCache:
    def test_cache_avoids_reencode(self):
        enc = _DummyEncoder()
        pool = _make_pool(
            corrs=[[0, 0.9], [0.9, 0]],
            ics=[0.06, 0.04],
            icirs=[0.3, 0.2],
            semantic=True,
            encoder=enc,
        )
        pool.search()
        assert enc.encode_calls == 1
        pool.search()
        assert enc.encode_calls == 1  # 池未变 → 复用缓存
        pool._pool_version += 1  # 模拟 try_new_expr 递增
        pool.search()
        assert enc.encode_calls == 2  # 池版本变化 → 重算

    def test_cache_respects_pool_version_after_add(self):
        enc = _DummyEncoder()
        pool = _make_pool(
            corrs=[[0, 0.9], [0.9, 0]],
            ics=[0.06, 0.04],
            icirs=[0.3, 0.2],
            semantic=True,
            encoder=enc,
        )
        pool.search()
        pool._pool_version += 1
        pool.search()
        pool._pool_version += 1
        pool.search()
        assert enc.encode_calls == 3  # 每次版本变化都重算


# ---------------------------------------------------------------------------
# 4. legacy 模式回归不变
# ---------------------------------------------------------------------------


class TestLegacyRegression:
    def test_legacy_uses_icir_depth_times(self):
        """legacy 模式：quality_prob 仍由 ICIR logit × depth/times decay 驱动。"""
        corrs = [[0, 0.1], [0.1, 0]]
        ics = [0.05, 0.05]
        icirs = [0.4, 0.05]
        pool = _make_pool(
            corrs=corrs, ics=ics, icirs=icirs,
            fitness_mode="legacy",
        )
        nodes, _ = pool.search()
        # 高 ICIR 节点排前（depth/times 相同）
        assert nodes[0].payload.source == "f0"

    def test_legacy_mode_does_not_need_fitness_module(self):
        """legacy 模式不触发 fitness 组件（calibrator 仍为 None / 不建单例）。"""
        corrs = [[0, 0.1], [0.1, 0]]
        pool = _make_pool(
            corrs=corrs, ics=[0.05, 0.04], icirs=[0.2, 0.1],
            fitness_mode="legacy",
        )
        pool.search()
        import alphaprobe.trainer.pool as pool_mod

        assert pool_mod._FITNESS_CALIBRATOR is None


# ---------------------------------------------------------------------------
# 5. fitness proxy 单调性
# ---------------------------------------------------------------------------


class TestFitnessProxy:
    def test_proxy_monotone_rankic(self):
        reset_shared_calibrator()
        node = _Node(_DummyExpr("x"), ic=0.05, icir=0.2)
        # warmup 样本集 = 池内全体节点（比 0.01 / 0.08 跨档）
        _fitness_proxy._nodes = [
            _Node(_DummyExpr("a"), ic=0.01, icir=0.1),
            _Node(_DummyExpr("b"), ic=0.04, icir=0.2),
            _Node(_DummyExpr("c"), ic=0.08, icir=0.4),
        ]
        try:
            lo = _fitness_proxy(node, 0.1, single_ic=0.01, ast_nodes=10)
            hi = _fitness_proxy(node, 0.4, single_ic=0.08, ast_nodes=10)
        finally:
            _fitness_proxy._nodes = None
        assert hi > lo

    def test_proxy_complexity_penalty(self):
        reset_shared_calibrator()
        node = _Node(_DummyExpr("x"), ic=0.05, icir=0.2)
        simple = _fitness_proxy(node, 0.3, single_ic=0.06, ast_nodes=10)
        complex_ = _fitness_proxy(node, 0.3, single_ic=0.06, ast_nodes=50)
        assert simple > complex_
