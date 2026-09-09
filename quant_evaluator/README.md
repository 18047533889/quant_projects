# quant_evaluator — 因子评估器（60 项证据指标）

批量因子评估器：对显式 `LabelBundle`（严格时序 + 因果链）评估因子批次，返回类型化
`EvaluationBundle` 证据包。评估因子 / 入库证据 / 模型 OOS 校验的唯一权威。

**定位:** 企业级 A 股横截面日频多因子量化项目的**评估层** —— 因子值进来，60 项指标
证据出去；所有"这因子行不行"的判断都从这里拿。

**版本:** 0.0.1a1 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator (私有)

---

## 它是什么 / 不是什么

**做什么：** 对 (FactorBatch, LabelBundle) 算 51 个已注册指标（27 个可立即计算）、
支持分片/分层/切分绑定评估、流式与并行批量评估、tear-sheet 报告、缓存与预算控制。

**不做什么：** 不算因子值（factor_engine 做）、不自己造标签时序（所有时序由调用方显式提供，
**QE 从不推断 / shift / fill**）、不做入库决策（factor_assets 做）。

## 安装与第一次评估

```bash
# Python >= 3.10
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator.git
cd quant_evaluator
python -m pip install -e .               # core runtime
python -m pip install -e ".[full]"      # 可选：或 [polars] / [stats] / [reporting]
```

```python
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

fb = FactorBatch(
    factor_ids=("f1",),
    time_axis=AxisRef(name="time", dtype="int64", size=T, values=...),
    asset_axis=AxisRef(name="asset", dtype="int64", size=N, values=...),
    values=values,            # (T, N, F) float64，3D 必须
)
lb = LabelBundle(
    target_id="fwd_vwap_5d",
    values=forward_returns,          # VWAP_{t+H}/VWAP_t - 1
    horizon=5,
    decision_time=tuple(...),         # 必填，严格递增
    label_start_time=tuple(...),
    label_end_time=tuple(...),        # 因果链：decision <= signal <= execution <= label_start < label_end
    price_convention="vwap_to_vwap",  # 平台硬性口径，不要改
)
bundle: EvaluationBundle = evaluate(
    fb, lb,
    metrics=("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage"),
)
```

核心契约：
- **LabelBundle** — 显式前向标签；时序全部由调用方提供；默认 `price_convention="vwap_to_vwap"`；
  `__post_init__` 强制校验严格递增 + 因果链。
- **FactorBatch** — `(时间 × 资产 × 因子)` 3D 因子值 + validity。
- **EvaluationBundle** — `metric_values` / `diagnostics` / `grouped_metrics`（多因子时逐因子值在
  `grouped_metrics[factor_id]`）/ `series_refs` / `split_ref` / `config_hash` / `warnings`。
- **SealedSplitRef** — 可选 sealed train/valid/test 切分绑定。

## 指标全景（51 个已注册，27 个可算）

> 权威清单：`docs/METRIC_REGISTRY_COVERAGE.csv`（60 行 = 33 已注册 + 27 NOT_IMPLEMENTED 占位）。
> **澄清：** "60+" = 51 个已注册 + 占位（如 sector_exposure / sharpe_ratio / long_short_returns /
> pairwise_correlation / fdr / capacity / liquidity 等**未实现**，不可直接请求）。

### IC 族（19）
| metric | 说明 |
|---|---|
| `rank_ic` ★core | 日均 Spearman IC 的时间均值，**全库唯一定义**（alias ic.rank.mean） |
| `pearson_ic` ★core | 日均 Pearson IC 时间均值（alias ic.pearson.mean） |
| `mean_ic` ★core | Pearson IC 均值；obs 口径特殊（报告联合有效面板格数） |
| `ic_std` ★core / `ic_ir` ★core | Spearman IC 标准差 / 信息比率 mean/std |
| `rank_ic_series` | 每日 Spearman IC 序列 (T,F)（alias ic.rank.daily） |
| `pearson_ic_series` | 每日 Pearson IC 序列（alias ic.pearson.daily） |
| `pearson_ic_std` / `pearson_ic_ir` | Pearson 版 std / IR |
| `ic_median` | IC 时间中位数 |
| `spearman_ic` / `ic_summary` / `rank_ic_time_series` / `rank_ic_cross_section` / `rolling_ic` / `ic_decay` / `autocorrelation_ic` | metadata-only（每日秩相关 / 汇总统计 / 时间片 / 截面 / 滚动窗 / 衰减 / 自相关） |
| `hac_tstat` / `hac_pvalue` | HAC 稳健 t 统计 / 双侧 p 值（min_periods=30） |

### 分层族（5）
`quantile_spread` ★core（顶层减底层分位收益差，**需 N≥50**）、`quantile_returns`、
`quantile_returns_full`（每分位平均收益向量，N≥50）、`quantile_stability`。

### 回撤/风险族（9）
`max_drawdown` / `drawdown_duration` / `calmar_ratio` / `var_95` / `var_99` /
`cvar_95` / `cvar_99` / `skewness` / `kurtosis`（均 metadata-only）。

### 换手族（6）
`turnover` ★core（canonical：`0.5*Σ|w_t−w_{t−1}|`，秩作代理权重）、
`factor_turnover_rate`（顶/底分位成员换手，min_periods=30）、`turnover_rate`、
`turnover_cost`（估计交易成本 bps）、`turnover_adjusted_ic`、`turnover_stability`。

### 自相关/半衰/置信区间族（4）
`ic_autocorr_lag1`（min_periods=30）、`half_life`（AR(1) 半衰期，min_periods=60）、
`block_bootstrap_ci`（block bootstrap 95% CI 半宽，min_periods=60）、
`subsample_stability`（min_periods=40）。

### 覆盖族（4）
`coverage` ★core（每因子联合有效格占比，**绝不跨因子列平均**）、`factor_coverage`、
`return_coverage`、`joint_coverage`。

### 集中度族（2） / 稳定性族（3）
`hhi_concentration`、`hhi_effective_n`；`ic_stability`、`coverage_stability`、`rank_stability`。

### 未注册但存在的 kernel（别直接请求，会抛 InvalidContractError）
`metrics/interactions/`（pairwise / conditional / substitution / complementarity）、
`metrics/stats/`（协整 / Granger / regime / 结构突变）、
`sector_exposure` / `sharpe_ratio` / `long_short_returns` / `fdr` / `capacity` / `liquidity` 等。

- **注意 min_periods 门限**：quantile_spread=20、hac_*=30、half_life=60、block_bootstrap=60、
  factor_turnover_rate=30 —— 不足抛 `InvalidContractError`。
- Canonical dotted 别名可解析：`ic.rank.mean`→rank_ic、`ic.pearson.daily`→pearson_ic_series、
  `ic.rank.hac_t`→hac_tstat 等 11 条。

## 价格口径

`LabelBundle.price_convention` 默认 `"vwap_to_vwap"`，**纯 provenance 字段**——QE 不做任何换算/校验。
文档约定调用方提供 `VWAP_{t+H}/VWAP_t - 1`（内存中即前向收益，不 shift 不推断）。
时序一致性由 `decision_time == factor time_axis` 对齐校验 + 因果链校验保证。

## 架构

```
api/           EvaluationRequest / EvaluationBundle / MetricValue / FactorDiagnosis
contracts/     LabelBundle, FactorBatch, SealedSplitRef, EvidenceStatus(8 态),
               MetricArtifact(5 类), TreatmentEvaluationArtifact, QuantileTiePolicy
registry/      MetricSpec + MetricRegistry（BUILDING→SEALED 可 seal），51 metric，presets
metrics/       ic / quantile / turnover / risk / portfolio_stats / quality / exposure /
               temporal / robustness(HAC·bootstrap) / distribution / stats /
               interactions / slicing / delta / robustness_cube
planner/       MetricDependencyGraph（DAG 拓扑序）、BatchPlan（chunk 切分）
runtime/       Evaluator（批量+缓存+预算）、StreamingEvaluator、ParallelBatchExecutor、
               cache_v2（Memory/Disk/Redis 多层）、budgets、intermediates
backends/      后端自动选择（numpy / numba / cupy / polars）、BackendRegistry
kernels/       fast.py（fast_ic_batch / fast_quantile_binning / fast_turnover）+ numba + reference_bridge
adapters/      factor_engine / data_access（stub，见下）、pandas.py（reference-only）、recipe_refs
reporting/     tear_sheet（22 institutional panels）、library_reports、chart_spec、artifacts
diagnosis/     diagnose_all_factors、WarningSystem（severity）
docs/          METRIC_REGISTRY_COVERAGE.csv、METRIC_COVERAGE_COMPILER.csv
tests/         29 个测试文件（396 collected / 393 passed / 3 skipped）
```

### adapters 现状（重要）
- `adapters/factor_engine.py` 与 `adapters/data_access.py` **均为 stub**（方法抛
  `NotImplementedError`，仅定义 `FactorBatchProvider` / `FactorIdentityProvider` /
  `ContextProvider` / `UniverseProvider` Protocol）——等待 FE MaterializedResult 合同冻结。
- `adapters/pandas.py` **唯一可用**（`dataframe_to_factor_batch` 等），**reference/debug only，
  禁止生产路径**。
- `adapters/recipe_refs.py`：recipe → `FactorValueRef` / LabelBundle ref / `EvaluationRequest`
  （factor_preprocess 集成路径）。

### 流式 / 并行
- `StreamingEvaluator`：生成器逐块喂入 `(factor_batch, label_bundle)`，sufficient statistics
  （Welford）增量累积 IC/coverage/summary，**常数内存支撑 100k+ 因子**；
  `evaluate_stream()` 校验跨 chunk 连续性；`evaluate_large_batch()` 自动切块。
- `ParallelBatchExecutor`：multiprocessing Pool 多 batch 并行。

### reporting
`generate_tear_sheet(EvaluationResult) → Dict[str, ChartSpec]`：**22 个 institutional panels**
（year×month IC heatmap、rolling rank IC、IC HAC 置信带、IC horizon surface、quantile
monotonicity/curvature、cumulative top-bottom spread、top-bottom drawdown、cost/delay
sensitivity、industry/style exposure、spec robustness cube）+ 约 22 个 legacy panels。
**只消费预计算 `MetricArtifact`，绝不重算**；缺 artifact 渲染 `NOT_COMPUTED` 占位，从不伪造 0.0。
`library_reports.py`：`LibraryReport` / `ComparisonReport` / `AdversarialReport`。

## 硬性规则

- **禁止**自行推断标签 / 位移 / 填充 —— QE 只消费 `LabelBundle` 给定的时序。
- **vwap-to-vwap** 是全局收益口径（`LabelBundle.price_convention`）。
- 指标注册表 seal 后不可变 —— 新增指标需新注册路径，不许静默改。

## 测试

```bash
cd quant_evaluator && pytest tests/ -q     # 396 collected / 393 passed / 3 skipped，约 14s
```
extras：`fast`(numba) / `polars` / `arrow` / `stats`(statsmodels) / `reporting`(matplotlib+jinja2) / `full`。

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`factor_engine`、`data_access`（lazy import，core 不硬依赖）。
- **被谁调用**：
  - `factor_optimizer` → 评估证据（`adapters/quant_evaluator.py`）
  - `factor_assets` → `QEEvidenceProvider`（bundle → EvidenceRef）
  - `quant_platform` → EvidenceStatus / 候选流水线
  - `modeling` → 模型 OOS 独立 IC 校验

## 相关仓库

- **factor_engine** — 上游因子值
- **data_access** — universe / 日历 provider
- **factor_optimizer / factor_assets / modeling** — 证据消费者（见上）
