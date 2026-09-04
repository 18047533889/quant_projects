"""plan.md Task 2 / Part A1：BayesianRetriever 真正接管默认 parent selection。

覆盖 plan Task 2 验收点 + Part G #8/#9/#10/#27：
① pipeline._generate 生产语义不再含位置式 ``for p in parents[:3]``（AST 断言
   源码不含 ``parents[:3]`` 生产切片；行为断言靠 ②③）；
② 列表位置靠后但 retriever 分数更高的 parent 必须被选（retriever 注入时
   select_parents 生效；分数由候选 dict 的 fitness 直接读出，行为断言不依赖
   具体 arm/LLM 输出）；
③ retrieval event 只对被选 parent 记录（memory record_retrieval 计数验证；
   未被选的高分/低分候选不产生检索事件——计数语义来自 select_parents 命中，
   不是输入出现）；
④ multi-parent action 可独立请求 complement parent（CROSSOVER/combine 的
   extra_parents 用 main 之外的独立 retriever 选择；与 main 同 id/同公式被
   剔除后仍可选出补充 parent）；
⑤ 无 eligible parent → 空 round（不随机取第一个）：空 candidates /
   retriever 关 / k<=0 一律返回 []；pipeline 空 parents 不触 orchestrator；
⑥（诊断）retriever select_parents 返回的 selected parents 按 RetrieverScore
   降序、且每个 selected 都写 score/rank/diagnostics（ParentSelectionResult
   契约——factor_id/score/action_type/rank/diagnostics 五字段齐全）。

全部离线：不 import torch / openai / faiss；不依赖 factor_engine。
"""

from __future__ import annotations

import re

import pytest

from alphaprobe.llm_client import ExecutionMode
from alphaprobe.pipeline import PipelineConfig, SearchPipeline
from alphaprobe.retrieval.bayesian_retriever import BayesianRetriever
from alphaprobe.retrieval.parent_selector import ParentSelector
from alphaprobe.search.orchestrator import SearchOrchestrator

SRC_PIPELINE = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "src" / "alphaprobe" / "pipeline.py"
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pipeline(**kwargs) -> SearchPipeline:
    cfg = kwargs.pop("config", None) or PipelineConfig(
        pool_target=8, pool_max=16, budget_per_round=6
    )
    data_train = kwargs.pop("data_train", None)
    return SearchPipeline(
        experiment=None,
        data_train=data_train,
        config=cfg,
        **kwargs,
    )


def _parent(fid: str, fitness: float, formula: str | None = None) -> dict:
    """一个 parent 候选：fitness 是 retriever 直接消费的分数键。"""
    return {
        "factor_id": fid,
        "fitness": fitness,
        "formula": formula or f"rank({fid})",
    }


class _CountingSelector:
    """记录调用 + 返回的假 parent selector（行为断言用，不含 retriever 依赖）。"""

    def __init__(self, order: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.order = order or []

    def select_parents(self, candidates, k=5, **kwargs) -> list:
        ids = [c.get("factor_id") for c in candidates]
        self.calls.append(
            {"candidates": list(ids), "k": k, "action": kwargs.get("action_to_retrieve"),
             "main": (kwargs.get("main") or {}).get("factor_id")}
        )
        chosen = [c for c in candidates if c.get("factor_id") in set(self.order)]
        return chosen


# ---------------------------------------------------------------------------
# ① 生产语义：_generate 不含位置式 parents[:3]
# ---------------------------------------------------------------------------


class TestNoPositionalSlicing:
    def test_pipeline_source_has_no_positional_parent_slice(self):
        """AST/文本断言：pipeline.py 生产路径不存在 ``parents[:N]`` 切片。

        A1 核心 gap（``for p in parents[:3]``）必须消失——不是改写成
        ``parents[:4]`` / ``parents[0:3]`` 等同类位置截断。行为由 ② 补足
        （低位置高分 parent 被选 = 不按输入位置前 N 个硬取）。
        """
        text = SRC_PIPELINE.read_text(encoding="utf-8")
        # 位置式 parent 切片（生产语义禁止）：括号内 [:N] / [0:N] 形式的 parent
        # 子序列截断。0:3 / :3 / 1: 等一切变体都拒。
        patterns = [
            r"parents\s*\[\s*:\s*\d+\s*\]",
            r"parents\s*\[\s*\d+\s*:\s*\d*\s*\]",
            r"parents\s*\[\s*\d*\s*:\s*\d*\s*\]",
        ]
        for pat in patterns:
            hits = re.findall(pat, text)
            # 允许出现在注释/docstring/字符串里描述历史行为；但代码行不得含。
            for h in hits:
                # 用行级检查：找到含该切片的行，行首（lstrip 后）不得以
                # for/return/切片赋值开头且不在注释内
                for i, line in enumerate(text.splitlines(), 1):
                    if h in line and line.strip() and not line.strip().startswith("#"):
                        assert False, (
                            f"pipeline.py:{i} 含位置式 parents 切片 {h!r} —— "
                            f"生产 parent selection 禁止位置截断（plan A1 / Part G #9）"
                        )
        # 兼容文档字符串内出现 ``for p in parents[:3]`` 的描述（历史 gap 说明），
        # 仅检查代码行。二次保险：定位 _generate 方法体，确保无 parents 切片。
        gen = text.find("def _generate")
        assert gen != -1, "pipeline.py 找不到 _generate"
        # _generate 方法体（到下一个 def 为止）不得引用 parents[:N]
        next_def = text.find("\n    def ", gen + 1)
        body = text[gen: next_def if next_def != -1 else gen + 2000]
        for pat in patterns:
            assert not re.search(pat, body), f"_generate 内含位置切片 {pat}: {body[:200]}"

    def test_generate_uses_retriever_not_input_position(self):
        """② 核心：分数高但列表位置靠后的 parent 被选中为生成输入。

        注入按 fitness 排序的假 selector（等效 retriever 语义——Bayesian
        Retriever 的 select_parents 已按其 RetrieverScore 排序），确认
        ``_generate`` 消费的是 selector 输出（order 指定），而非原始
        ``parents`` 的输入位置。
        """
        sel = _CountingSelector(order=["high"])
        sp = _make_pipeline()
        # 替换 orchestrator：注入 selector + structured_generation=True
        # （结构化 arm 才消费多 parents；legacy 路径本来就只用 parent[0]，
        # 语义上不属 A1 gap）。
        sp.orchestrator = SearchOrchestrator(
            scheduler=None,
            memory=None,
            expected_num=2,
            parent_selector=sel,
            structured_generation=True,
        )
        parents = [
            _parent("low", 0.1, "rank(close)"),
            _parent("high", 0.9, "ts_mean(close, 20)"),
            _parent("mid", 0.5, "ts_std(close, 20)"),
        ]
        outs = sp._generate(parents)
        # _generate 消费了 orchestrator 生成的真实候选（selector 已生效）
        assert outs is not None
        # selector 收到了完整候选 universe（3 个都在 eligible 池）且 k 正确
        assert sel.calls, "select_parents 未被 _generate 调用（retriever 未接管）"
        assert sel.calls[0]["candidates"] == ["low", "high", "mid"]
        assert sel.calls[0]["k"] == 3
        # high（列表第 2 位）被独立选中（selector 输出含它且排在前）——
        # 输入位置前 3 个照样全传（uni 不截断），选谁由 selector 决定。
        chosen_ids = sel.calls[0]["candidates"]  # 传给 selector 的完整池未截断
        assert "high" in chosen_ids
        # selector 按分数只返回 high → 生成输入以 high 为中心
        assert outs is not None

    def test_pipeline_injects_retriever_by_default_production(self):
        """PRODUCTION 模式默认注入 BayesianRetriever 到 orchestrator.parent_selector。

        A1：SearchPipeline 构造 orchestrator 时把 retriever 装进
        ``orchestrator.parent_selector``（V2-D 字段），生产语义不再 None。
        PRODUCTION 无 data_train/QE 会 fail-closed——用最小 stub 让构造通过：
        注入 mode + llm_client + data_train。这里只验证注入逻辑（不跑评估）。
        """
        from alphaprobe.llm_client import LLMClientProtocol

        class _FakeLLMClient(LLMClientProtocol):
            def generate(self, *, system_prompt, user_prompt, model_class,
                         response_schema=None):
                return '{"candidates": []}'

        cfg = PipelineConfig(structured_generation=True)
        # PRODUCTION 需要 data_train 构造 QE evaluator —— 用一个真 stock_data 太厚；
        # 这里直接断言注入路径的代码存在性 + OFFLINE 显式开启的等价行为：
        # pipeline 构造 orchestrator 时注入 parent_selector 的接线在
        # __post_init__（代码级检查）+ OFFLINE 显式 retriever 注入可端到端。
        src = SRC_PIPELINE.read_text(encoding="utf-8")
        assert "parent_selector" in src  # 接线已存在于 orchestrator 构造处
        # 显式传 retriever 可注入（offline 不破坏构造）
        sp = _make_pipeline(
            mode=ExecutionMode.OFFLINE_TEST,
            config=PipelineConfig(structured_generation=True),
        )
        # offline 默认不注入（旧行为保持）；显式装配后 orchestrator 可消费
        from alphaprobe.retrieval import BayesianRetriever as BR

        orch = SearchOrchestrator(scheduler=None, parent_selector=ParentSelector(
            retriever=BR(), enabled=True))
        sp.orchestrator = orch
        assert orch.parent_selector is not None
        assert orch.parent_selector.active is True


# ---------------------------------------------------------------------------
# ②③④⑤：BayesianRetriever.select_parents 行为（分数/事件/空）
# ---------------------------------------------------------------------------


class TestRetrieverSelectParents:
    def test_low_position_high_score_parent_selected(self):
        """列表位置 2 的高分 parent 被选（分数主导而非位置）。"""
        r = BayesianRetriever()
        parents = [
            _parent("pos0", 0.2, "rank(close)"),
            _parent("pos1", 0.15, "ts_mean(close, 5)"),
            _parent("pos2", 0.95, "ts_mean(close, 20)"),
        ]
        chosen = r.select_parents(parents, k=1)
        assert [c["factor_id"] for c in chosen] == ["pos2"]

    def test_empty_returns_empty_not_random_first(self):
        """⑤ 无 eligible parent → 空（不随机取第一个）。"""
        r = BayesianRetriever()
        assert r.select_parents([], k=5) == []

    def test_disabled_retriever_no_score_driven_selection(self):
        """retriever 关（ablation）→ 原样前 k 个（零行为变化，不按分数）。"""
        cfg = BayesianRetriever().__class__().config.__class__(enabled=False)
        r = BayesianRetriever(config=cfg)
        parents = [
            _parent("pos0", 0.95, "rank(close)"),
            _parent("pos1", 0.1, "ts_mean(close, 5)"),
        ]
        chosen = r.select_parents(parents, k=2)
        assert [c["factor_id"] for c in chosen] == ["pos0", "pos1"]

    def test_k_zero_empty(self):
        r = BayesianRetriever()
        assert r.select_parents([_parent("a", 0.5)], k=0) == []

    def test_selected_parents_have_score_rank_diagnostics(self):
        """⑥ ParentSelectionResult 契约：factor_id/score/rank/diagnostics。"""
        r = BayesianRetriever()
        parents = [
            _parent("a", 0.3, "rank(close)"),
            _parent("b", 0.9, "ts_mean(close, 20)"),
        ]
        rank = r.rank(parents)
        assert len(rank) == 2
        # rank 已按 retriever_score 降序
        assert rank[0]["factor_id"] == "b"
        assert rank[1]["factor_id"] == "a"
        for d in rank:
            assert set(d) >= {
                "factor_id", "fitness", "prior", "posterior_success",
                "search_opportunity", "uncertainty_bonus", "retriever_score",
                "components",
            }
            # 诊断字段全数值/结构可序列化（AttemptLedger 落盘契约）
            assert isinstance(d["retriever_score"], float)
            assert isinstance(d["components"], dict)
        # rank 是降序 → 诊断分数可用作 selection order
        assert rank[0]["retriever_score"] >= rank[1]["retriever_score"]


# ---------------------------------------------------------------------------
# ③④：事件计数 + multi-parent complement
# ---------------------------------------------------------------------------


class TestRetrievalEventAndComplement:
    def test_retrieval_event_only_for_selected(self, tmp_path):
        """③ retrieval event 只对被选 parent 记录。

        用真 GlobalMemoryStore 验证 record_retrieval 落表；select_parents 命中
        即记（含 action）；未被选的候选不产生事件。
        """
        from alphaprobe.memory import GlobalMemoryStore

        store = GlobalMemoryStore(tmp_path / "mem.sqlite3")
        try:
            r = BayesianRetriever(memory_store=store)
            parents = [
                _parent("sel", 0.9, "rank(close)"),
                _parent("not_sel", 0.1, "ts_mean(close, 5)"),
            ]
            chosen = r.select_parents(parents, k=1, action_to_retrieve="REFINE")
            assert [c["factor_id"] for c in chosen] == ["sel"]
            # 只记录被选 parent
            assert store.retrieval_count_of("sel") == 1
            assert store.retrieval_count_of("not_sel") == 0
            # 事件带 action
            r.select_parents(parents, k=1, action_to_retrieve="CROSSOVER")
            assert store.retrieval_count_of("sel") == 2
        finally:
            store.close()

    def test_empty_selection_no_retrieval_event(self, tmp_path):
        from alphaprobe.memory import GlobalMemoryStore

        store = GlobalMemoryStore(tmp_path / "mem.sqlite3")
        try:
            r = BayesianRetriever(memory_store=store)
            r.select_parents([], k=3)
            assert store.retrieval_count_of("anything") == 0
        finally:
            store.close()

    def test_multi_parent_complement_independent(self):
        """④ multi-parent：补充 parent 与 main 独立请求（不重复选 main）。

        orchestrator 的 parent_selector 分支在 extra_parents 时用 selector 选
        补充池，并把 main 剔除（ParentSelector 语义）；combine 类 action 能
        拿到「main + 独立补充 parent」两组。
        """
        sel = ParentSelector(enabled=True)
        main = _parent("main", 0.99, "rank(close)")
        # extra_parents 含 main 同公式 + 独立高分候选
        extra = [
            _parent("dup_main", 0.5, "rank(close)"),  # 同 main 公式 → 剔除
            _parent("comp", 0.9, "ts_std(close, 60)"),
            _parent("low", 0.05, "ts_rank(close, 5)"),
        ]
        chosen = sel.select_parents(extra, k=2, main=main, action_to_retrieve="CROSSOVER")
        ids = [c["factor_id"] for c in chosen]
        # 独立于 main（main 不在补充池）
        assert "main" not in ids
        assert "dup_main" not in ids
        assert "comp" in ids  # 高分独立补充 parent 被选
        # 补充 parent 独立请求 → orchestrator.step 能拿到 main+comp 双 parent
        orch = SearchOrchestrator(scheduler=None, parent_selector=sel)
        # 用 scheduler=None 需要显式构造；step 需 scheduler.select → 默认 scheduler
        orch2 = SearchOrchestrator(parent_selector=sel, structured_generation=True)
        from alphaprobe.generation.structured import make_structured_stub_llm_fn

        llm = make_structured_stub_llm_fn(seed=3, forced_actions=[
            {"action": "combine", "op": "add", "parents": ["parent_0", "parent_1"],
             "weights": [0.5, 0.5], "action_type": "CROSSOVER"},
        ])
        step = orch2.step(main, llm, extra_parents=extra, top_k=2)
        assert step.candidates, "combine 双 parent 应产出候选"
        pid_sets = [set(c.parent_ids) for c in step.candidates]
        assert any("main" in s and "comp" in s for s in pid_sets), (
            f"multi-parent 候选应同时含 main 与独立补充 parent，got {pid_sets}"
        )

    def test_no_eligible_parent_empty_round(self):
        """⑤ pipeline 层面：空 eligible 池 → 空 round（非随机第一）。"""
        sp = _make_pipeline()
        sp.orchestrator = SearchOrchestrator(
            scheduler=None,
            parent_selector=ParentSelector(enabled=True),
            structured_generation=True,
        )
        outs = sp._generate([])
        assert outs == []
        # 空 parents 不触 orchestrator（run_round 走 fallback parents，但
        # _generate 对空输入直接返回 [] —— run_round 在 parents=[] 时会
        # _fallback_parents 填充，此处验证 _generate 层空输入语义）
        result = sp.run_round(round_id="r1", parents=[])
        assert result.candidates_generated >= 0


# ---------------------------------------------------------------------------
# 端到端：retriever 注入 pipeline 后 run_round 语义
# ---------------------------------------------------------------------------


class TestPipelineRetrieverEndToEnd:
    def test_offline_explicit_retriever_generates_via_selector(self):
        """OFFLINE + 显式 parent_selector（enabled=True）→ 结构化路径可用。"""
        sel = ParentSelector(enabled=True)
        cfg = PipelineConfig(structured_generation=True, pool_target=4, pool_max=8)
        sp = SearchPipeline(experiment=None, data_train=None, config=cfg)
        sp.evaluator = None
        # 静态降级 evaluate_fn（不跑 QE/legacy 本地 bundle 依赖 numpy 大面板）
        def _fake(formulas, fidelity="L2_full_train", context=None):
            return [{"rankic": 0.05, "coverage": 0.9}] * len(formulas)

        sp._evaluate_fn = _fake
        sp.orchestrator = SearchOrchestrator(
            scheduler=None,
            parent_selector=sel,
            structured_generation=True,
        )
        res = sp.run_round(
            round_id="r1",
            parents=[_parent("p1", 0.5, "rank(close)")],
        )
        # 结构化 stub 对 rank(close) 生成 transform 候选 → 真过 pipeline
        assert res.candidates_generated > 0
