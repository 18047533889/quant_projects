"""plan Task 24（plan.md）：Algorithm Ablation Harness 验收测试。

覆盖 plan Task 24 Tests + Non-negotiable Part G #30 / plan primary-metrics 口径：
1. 10 配置（legacy reference → +next-version Survival Memory）可装配；每个
   配置的开关位与 plan 对应（累积 feature tower：配置 N = 配置 N-1 全部特性
   + 第 N 项）；序列化 roundtrip 可复现。
2. 指标聚合：从注入的合成 attempt/evaluation 数据算 EliteYield /
   CostAdjustedEliteYield / duplicate rate / coverage / cost-per-elite，
   口径正确：
   - EliteYield 分母是 **attempts**（不是 generated factor count；plan 原文
     「Never use generated factor count as primary success metric」）；
   - CostAdjustedEliteYield = elites×1000 ÷ 归一总成本；
   - quality 指标只对 L3+ 的已评估事实取中位（无数据 → None，不编造）。
3. 配置 diff：两个配置的开关差异可枚举（ablation 可比性基础）。
4. 无数据/空 run → 指标中性输出不炸（空 metrics 全 0/None）。
5. 装配自检：contextual（T10）是 bayesian（T2）超集 → contextual 开时
   bayesian.enabled 必须仍 True（装配矛盾 fail-fast）；关闭某特性 → 对应
   probe 关。

全部合成数据、确定性、零 LLM / 零模型 / 零网络；不 import torch / faiss /
sklearn / factor_engine / data_access / quant_evaluator。
"""

from __future__ import annotations

import json
import math

import pytest

from alphaprobe.experiments.ablation import (
    ABLATION_CONFIG_NAMES,
    ELITE_YIELD_DENOMINATOR,
    FEATURE_ORDER,
    AblationAttempt,
    AblationConfig,
    AblationReport,
    AblationRunner,
    ablation_config_matrix,
    all_ablation_configs,
    build_ablation_config,
    compute_metrics,
    configs_diff,
    make_feature_switch,
)


# ---------------------------------------------------------------------------
# 1. 10 配置矩阵可装配 + 开关位与 plan 对应 + 序列化可复现
# ---------------------------------------------------------------------------


class TestTenConfigMatrix:
    def test_ten_named_configs_in_plan_order(self):
        assert list(ABLATION_CONFIG_NAMES) == [
            "legacy_reference",
            "plus_fitness_v21",
            "plus_bayesian_retrieval",
            "plus_contextual_retriever",
            "plus_schema_space",
            "plus_logic_memory",
            "plus_paradigm_scheduler",
            "plus_trajectory_credit",
            "plus_evi_promotion",
            "plus_next_version_survival",
        ]

    def test_build_all_ten_assembles_without_error(self):
        cfgs = all_ablation_configs()
        assert len(cfgs) == 10
        for i, cfg in enumerate(cfgs, start=1):
            assert cfg.name == ABLATION_CONFIG_NAMES[i - 1]
            asm = cfg.assembly()  # 装配即校验；失败 raise
            assert asm.config_name == cfg.name
            assert asm.validate() == []

    def test_cumulative_feature_tower_matches_plan(self):
        """配置 N 的开关 = 配置 N-1 全开关 + plan 第 N 项。"""
        prev: set[str] = set()
        for i in range(1, 11):
            cfg = build_ablation_config(i)
            on = {f for f, v in cfg.features.items() if v}
            # legacy_reference 是标记位（无行为开关），不参与 tower 增量
            behavioral = on - {"legacy_reference"}
            assert behavioral == prev | {FEATURE_ORDER[i - 1]} - {"legacy_reference"}
            prev = behavioral
        # 最后一个配置打开除 legacy 外全部 9 个行为特性
        last = build_ablation_config(10)
        assert {f for f, v in last.features.items() if v} == set(FEATURE_ORDER)

    def test_feature_switch_positions_legacy_is_marker_only(self):
        """legacy_reference 开关无任何 settings/probe（纯标记）。"""
        s = make_feature_switch("legacy_reference", True)
        assert s.settings == {}
        assert s.probe == {}

    def test_each_config_serializes_and_roundtrips(self):
        for i in range(1, 11):
            cfg = build_ablation_config(i)
            d = cfg.to_dict()
            cfg2 = AblationConfig.from_dict(d)
            assert cfg2.to_dict() == d
            # 装配产物序列化亦可复现
            asm = cfg.assembly()
            ser = json.loads(json.dumps(asm.to_dict()))
            assert ser["config_name"] == cfg.name
            assert len(ser["switches"]) == len(asm.switches)

    def test_matrix_summary_has_ten_rows(self):
        matrix = ablation_config_matrix()
        assert len(matrix) == 10
        assert matrix[0]["features_on"] == ["legacy_reference"]
        assert "next_version_survival" in matrix[-1]["features_on"]

    def test_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            AblationConfig.for_index(0)
        with pytest.raises(ValueError):
            AblationConfig.for_index(11)

    def test_unknown_feature_key_rejected(self):
        with pytest.raises(ValueError):
            AblationConfig(name="custom", features={"not_a_feature": True})

    def test_unknown_switch_raises(self):
        with pytest.raises(KeyError):
            make_feature_switch("nope", True)


# ---------------------------------------------------------------------------
# 装配语义：contextual 是 bayesian 超集；关闭特性 probe 正确
# ---------------------------------------------------------------------------


class TestAssemblySemantics:
    def test_contextual_config_superset_keeps_bayesian_enabled(self):
        """配置 #4（contextual 在 #3 之上）bayesian.enabled 保持 True。"""
        c3 = build_ablation_config(3).assembly()
        c4 = build_ablation_config(4).assembly()
        assert c3.bayesian.enabled is True
        assert c4.bayesian.enabled is True  # T10 是 T2 超集，不降级
        assert c3.contextual.enabled is False
        assert c4.contextual.enabled is True

    def test_legacy_all_disabled(self):
        asm = build_ablation_config(1).assembly()
        assert asm.bayesian.enabled is False
        assert asm.contextual.enabled is False
        assert asm.paradigm.enabled is False
        assert asm.promotion.enabled is False
        assert asm.fitness["use_v21"] is False
        assert asm.orchestrator["structured_generation"] is False
        assert asm.orchestrator["multistage_generation"] is False

    def test_full_config_all_on(self):
        asm = build_ablation_config(10).assembly()
        assert asm.bayesian.enabled is True
        assert asm.bayesian.stagnation_enabled is True
        assert asm.contextual.enabled is True
        assert asm.paradigm.enabled is True
        assert asm.paradigm.repair_enabled is True
        assert asm.promotion.enabled is True
        assert asm.promotion.use_model is True
        assert asm.fitness["use_v21"] is True
        assert asm.fitness["apply_confidence_shrinkage"] is True
        assert asm.orchestrator["structured_generation"] is True
        assert asm.orchestrator["multistage_generation"] is True
        assert asm.survival["with_interactions"] is True
        assert asm.opportunity.w_survival == 0.15
        # 每个开关位都出现在 switches（diff/序列化可枚举）
        names = asm.switch_names()
        for f in ("fitness_v21", "bayesian_retrieval", "contextual_retriever",
                  "schema_space", "logic_memory", "paradigm_scheduler",
                  "trajectory_credit", "evi_promotion", "next_version_survival"):
            assert f in names

    def test_trajectory_maps_to_paradigm_repair(self):
        c7 = build_ablation_config(7).assembly()
        c8 = build_ablation_config(8).assembly()
        assert c7.paradigm.enabled is True
        # #7 开范式但 trajectory 未开时 repair_enabled 仍默认 True（Paradigm 本体
        # 默认带 repair 语义；#8 的增量由 diff 枚举——装配上保持无自相矛盾）。
        assert c8.paradigm.repair_enabled is True
        # 关闭某特性 → probe 关
        off = AblationConfig(name="custom", features={
            "bayesian_retrieval": False, "fitness_v21": False}).assembly()
        assert off.probes["bayesian_retrieval"] is False
        assert off.probes["fitness_v21"] is False

    def test_assembly_validate_detects_missing_bayesian(self):
        """校验能发现 bayesian 缺失/未开（fail-fast 不静默）。"""
        from alphaprobe.experiments.ablation import AblationAssembly

        asm = AblationAssembly(
            config_name="broken",
            switches=(make_feature_switch("bayesian_retrieval", True),),
            bayesian=None,
        )
        problems = asm.validate()
        assert any("bayesian" in p for p in problems)


# ---------------------------------------------------------------------------
# 2. 指标聚合：口径正确性（分母是 attempts，不是 generated 数）
# ---------------------------------------------------------------------------


def _attempt(
    i: int,
    *,
    elite: bool = False,
    novel: bool = True,
    fidelity: str = "L3_search_valid",
    dup: bool = False,
    schema: str | None = "S1",
    logic: str | None = "LOG_1",
    cluster: str | None = None,
    fitness: float | None = None,
    sharpe: float | None = None,
    stability: float | None = None,
    llm_cost: float = 0.0,
    fe_seconds: float = 0.0,
    qe_seconds: float = 0.0,
    survival_rate: float | None = None,
) -> dict:
    return {
        "attempt_id": f"a{i}",
        "factor_id": f"f{i}",
        "fidelity": fidelity,
        "elite": elite,
        "globally_novel": novel,
        "duplicate": dup,
        "schema_id": schema,
        "logic_id": logic,
        "cluster_id": cluster,
        "fitness": fitness,
        "long_short_sharpe": sharpe,
        "stability": stability,
        "llm_cost": llm_cost,
        "fe_seconds": fe_seconds,
        "qe_seconds": qe_seconds,
        "survival_rate": survival_rate,
    }


class TestEliteYieldCaliber:
    def test_elite_yield_denominator_is_attempts_not_generated(self):
        """EliteYield = novel L3/L4 elite ÷ 1000 attempts 当量。

        注入 500 attempts、10 elites → yield = 20.0（=10*1000/500），
        generated factor count 同样 500 —— 该口径验证「分母 attempts」。
        """
        rows = [
            _attempt(i, elite=(i < 10), fitness=0.7 if i < 10 else 0.4)
            for i in range(500)
        ]
        m = compute_metrics(rows)
        assert m["attempts"] == 500
        assert m["elite_count"] == 10
        assert m["elite_yield_per_1000_attempts"] == pytest.approx(10.0 * 1000.0 / 500.0)
        assert m["elite_yield_per_1000_attempts"] == pytest.approx(20.0)
        # generated_factors 只是信息列，不是 success metric
        assert m["generated_factors"] == 500

    def test_non_novel_or_below_l3_elite_excluded(self):
        """非 globally novel / L2 的 elite 不计入 EliteYield。"""
        rows = [
            _attempt(0, elite=True, novel=True, fidelity="L4_pool_audit"),
            _attempt(1, elite=True, novel=False, fidelity="L3_search_valid"),
            _attempt(2, elite=True, novel=True, fidelity="L2_full_train"),  # 低于 L3
            _attempt(3, elite=False, novel=True, fidelity="L3_search_valid"),
        ]
        m = compute_metrics(rows)
        assert m["elite_count"] == 1
        # 分母 = attempts（4 条）
        assert m["elite_yield_per_1000_attempts"] == pytest.approx(1.0 * 1000.0 / 4.0)
        assert m["elite_yield_per_1000_attempts"] == pytest.approx(250.0)

    def test_elite_yield_zero_when_no_attempts(self):
        m = compute_metrics([])
        assert m["attempts"] == 0
        assert m["elite_yield_per_1000_attempts"] == 0.0
        assert m["duplicate_rate"] == 0.0

    def test_cost_adjusted_elite_yield_math(self):
        """CostAdjustedEliteYield = elites×1000 ÷ (llm+fe+qe 归一成本)。

        1000 attempts、10 elites；llm_cost=0.1×1000=100、compute=22×1000=22000
        → 10000/(100+22000)=0.4524886...
        """
        rows = [
            _attempt(i, elite=(i < 10), llm_cost=0.1, fe_seconds=2.0, qe_seconds=20.0)
            for i in range(1000)
        ]
        m = compute_metrics(rows)
        assert m["cost_adjusted_elite_yield"] == pytest.approx(10000.0 / 22100.0, rel=1e-6)

    def test_cost_per_elite(self):
        rows = [
            _attempt(i, elite=(i < 5), llm_cost=1.0, fe_seconds=10.0, qe_seconds=40.0)
            for i in range(100)
        ]
        m = compute_metrics(rows)
        # elite=5；llm total=100×1=100 → per elite 20；compute=100×50=5000 → per elite 1000
        assert m["llm_cost_per_elite"] == pytest.approx(20.0)
        assert m["compute_seconds_per_elite"] == pytest.approx(1000.0)

    def test_duplicate_rate_and_coverage(self):
        rows = [
            _attempt(i, dup=(i % 10 == 0), schema=f"S{i % 3}", logic=f"LOG_{i % 2}",
                     cluster=f"C{i % 4}")
            for i in range(100)
        ]
        m = compute_metrics(rows)
        assert m["duplicate_rate"] == pytest.approx(0.1)
        assert m["unique_schema_coverage"] == 3
        assert m["unique_logic_coverage"] == 2
        assert m["unique_cluster_coverage"] == 4

    def test_quality_metrics_only_from_l3_plus(self):
        """L2 的已评估事实不进入 median L3 fitness。"""
        rows = [
            _attempt(0, fidelity="L3_search_valid", fitness=0.3, sharpe=0.5, stability=0.4),
            _attempt(1, fidelity="L3_search_valid", fitness=0.9, sharpe=1.5, stability=0.8),
            _attempt(2, fidelity="L2_full_train", fitness=0.99, sharpe=5.0, stability=0.99),
        ]
        m = compute_metrics(rows)
        assert m["median_l3_fitness"] == pytest.approx(0.6)
        assert m["best_l3_fitness"] == pytest.approx(0.9)
        assert m["median_long_short_sharpe"] == pytest.approx(1.0)
        assert m["median_stability"] == pytest.approx(0.6)

    def test_no_evaluated_evidence_returns_none_not_fabricated(self):
        rows = [_attempt(i, fidelity="L0_static", fitness=None) for i in range(10)]
        m = compute_metrics(rows)
        assert m["median_l3_fitness"] is None
        assert m["best_l3_fitness"] is None
        assert m["next_version_survival_rate"] is None
        assert m["unique_schema_coverage"] == 0

    def test_survival_rate_median_over_l3_elites(self):
        rows = [
            _attempt(i, elite=(i < 4), fidelity="L3_search_valid",
                     survival_rate=0.8 if i < 4 else None)
            for i in range(10)
        ]
        m = compute_metrics(rows)
        assert m["next_version_survival_rate"] == pytest.approx(0.8)

    def test_empty_run_neutral_does_not_raise(self):
        """无数据/空 run → 指标中性输出不炸（全 0/None，无异常）。"""
        m = compute_metrics([])
        assert m["attempts"] == 0
        assert m["duplicate_rate"] == 0.0
        assert m["elite_yield_per_1000_attempts"] == 0.0
        assert m["cost_adjusted_elite_yield"] == 0.0
        assert m["median_l3_fitness"] is None
        assert m["unique_schema_coverage"] == 0

    def test_ablation_attempt_from_dict_tolerates_extra_keys(self):
        a = AblationAttempt.from_dict({"attempt_id": "x", "status": "elite", "bogus": 1})
        assert a.attempt_id == "x"
        assert a.elite is True  # status=elite 兼容


# ---------------------------------------------------------------------------
# 3. 配置 diff 可枚举（ablation 可比性）
# ---------------------------------------------------------------------------


class TestConfigDiff:
    def test_adjacent_diff_adds_exactly_next_feature(self):
        for i in range(1, 10):
            d = configs_diff(build_ablation_config(i), build_ablation_config(i + 1))
            assert d["removed"] == []
            # 相邻配置只新增第 i+1 个特性（legacy 标记位 shared，不进 added）
            assert FEATURE_ORDER[i] in d["added"]
            assert set(d["added"]) <= {FEATURE_ORDER[i]}
            assert d["shared"]  # 非空（前序特性共享）

    def test_full_to_legacy_diff_removes_all_behavioral(self):
        d = configs_diff(build_ablation_config(10), build_ablation_config(1))
        assert set(d["removed"]) == set(FEATURE_ORDER[1:])  # 除 legacy 外全部移除
        assert d["added"] == []

    def test_diff_is_symmetric_under_reverse(self):
        da = configs_diff(build_ablation_config(3), build_ablation_config(7))
        db = configs_diff(build_ablation_config(7), build_ablation_config(3))
        assert set(da["added"]) == set(db["removed"])
        assert set(da["removed"]) == set(db["added"])

    def test_overrides_captured_in_diff(self):
        ca = AblationConfig(name="custom", features={"fitness_v21": True},
                            overrides={"fitness.k": 128.0})
        cb = AblationConfig(name="custom2", features={"fitness_v21": True},
                            overrides={"fitness.k": 64.0})
        d = configs_diff(ca, cb)
        assert d["a_only_overrides"] == {"fitness.k": 128.0}
        assert d["b_only_overrides"] == {"fitness.k": 64.0}


# ---------------------------------------------------------------------------
# 4. AblationRunner + AblationReport（含 run 回调注入与空 run 中性）
# ---------------------------------------------------------------------------


class _SyntheticCampaign:
    """合成 run_fn：注入 attempts 给指定配置名（不真跑 LLM/FE/QE）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, config, assembly) -> list[dict]:
        self.calls.append(config.name)
        elite = config.name == ABLATION_CONFIG_NAMES[-1]
        return [
            _attempt(i, elite=elite, fitness=0.75 if elite else 0.5,
                     survival_rate=0.9 if elite else None)
            for i in range(200)
        ]


class TestRunnerReport:
    def test_runner_dry_run_no_campaign_called(self):
        runner = AblationRunner()
        report = runner.run_all(build_ablation_config(1) if False else
                                [build_ablation_config(1), build_ablation_config(2)])
        assert len(report.config_names) == 2
        # dry-run：无 run_fn → 空 attempts → 中性指标
        assert report.rows[0]["attempts"] == 0

    def test_runner_injects_callback_and_aggregates(self):
        camp = _SyntheticCampaign()
        runner = AblationRunner(run_fn=camp)
        report = runner.run_all([build_ablation_config(1), build_ablation_config(10)])
        assert camp.calls == [ABLATION_CONFIG_NAMES[0], ABLATION_CONFIG_NAMES[-1]]
        # 配置 1 无 elite；配置 10 全 elite
        assert report.rows[0]["elite_count"] == 0
        assert report.rows[1]["elite_count"] == 200
        assert report.rows[1]["elite_yield_per_1000_attempts"] == pytest.approx(200.0 * 1000.0 / 200.0)

    def test_report_diffs_adjacent_metric_delta(self):
        camp = _SyntheticCampaign()
        runner = AblationRunner(run_fn=camp)
        report = runner.run_all([build_ablation_config(1), build_ablation_config(10)])
        assert len(report.diffs) == 1
        d = report.diffs[0]
        assert d["from"] == ABLATION_CONFIG_NAMES[0]
        assert d["to"] == ABLATION_CONFIG_NAMES[-1]
        assert d["metric_delta"]["elite_yield_per_1000_attempts"] == pytest.approx(1000.0)

    def test_empty_report(self):
        rep = AblationReport.from_results([])
        assert rep.rows == []
        assert rep.diffs == []
        assert rep.config_names == []

    def test_markdown_renders(self):
        runner = AblationRunner()
        rep = runner.dry_run_all()
        md = rep.to_markdown()
        assert md.startswith("| config |")
        assert "plus_next_version_survival" in md

    def test_ablation_attempt_is_dataclass_roundtrip(self):
        a = AblationAttempt(attempt_id="a1", factor_id="f1", elite=True, fidelity="L3_search_valid")
        d = a.to_dict()
        b = AblationAttempt.from_dict(d)
        assert b == a
