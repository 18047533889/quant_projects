# modeling — 模型层（ML 训练 / walk-forward / 泄漏防护）

模型层的单一真相源：训练、walk-forward **日期权威**切分、标签区间 purge、embargo、
泄漏防护（含负向对照电池）、artifact 生命周期、预测与评估。
**所有时序泄漏防护都在本库执行。**

**定位:** 企业级 A 股横截面日频多因子量化项目的**建模层** —— 在因子预处理（factor_preprocess）
之后、组合优化/回测（riskfolio_qs / vectorbt_qs）之前。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/modeling (私有)
**权威声明:** `AUTHORITY.md`（MODEL2-P0-006）

---

## 它是什么 / 不是什么

**做什么：** 训练严格防泄漏的模型并产出带完整血统（manifest）的 `ModelArtifact`：
walk-forward 日期权威切分 → label 区间 purge + embargo → train-only 冻结预处理 →
超参网格搜索（只准 TRAIN fit）→ 验证集择优 → refit → 冻结 artifact → OOS 预测 → 评估。

**不做什么：** 不算因子值、不评估因子 IC 作为入库证据（quant_evaluator 做）、不做组合优化。
它专注"把 panel 数据变成不泄漏的、可审计的模型"。

## 核心概念

- **PanelDataset** — 长格式 pooled panel（每行一个 (stock,date) 观测），
  `date_col/stock_col/feature_cols/label_col`；生产模式要求显式 `FeatureSchema`；
  (stock,date) 唯一（`duplicate_policy="error"`）；`as_matrix()`、`panel_telemetry()`。
- **DecisionClock** — 声明每类信息何时可得（decision_at / feature_available_at /
  label_available_at / score_written_at / execution_at）。A 股日频两个场景常量：
  `AFTER_CLOSE_TO_NEXT_VWAP`（t 收盘后出因子，t+1 VWAP 执行）与
  `BEFORE_SAME_DAY_VWAP`（t 日 VWAP 执行，close/vwap/volume 属未来信息禁用）。
- **LabelContract** — `return_basis="vwap_to_vwap"`（A 股权威口径，§8）：
  `label_t(H) = VWAP_{t+H}/VWAP_t - 1`；`horizon_bars` / `overlapping` / `purge_by_interval` /
  `embargo_bars`；`semantic_hash` / `label_interval(anchor)→[t, t+H]`。
- 其他契约：`SampleAdequacyContract`、`EmbargoSpec`（days ≥ label horizon）、
  `ApplicationWindow`（strict OOS 窗口）、`PredictionOutputContract`/`PredictionBatch`、
  `ParamRole`（7 类）+ `ParameterSearchPolicy`（NUMERICAL / DATA_POLICY 禁 searchable）、
  `ModelOperatorSpec`、`FitFingerprint`。

## 拥有什么（功能全清单）

### 日期权威切分 `split.py`
`date_bounded_split`（train/val/test 三切分 + 日期权威断言）、`split_by_date_cutoff`、
`assert_date_authoritative`。**每个日期只属于一个切分。**

### Walk-forward `walk_forward.py`
- `make_walk_forward_splits(ds, WalkForwardSpec, date_col=)` → `list[WalkForwardFold]`
  （`WalkForwardSpec`：train_lookback_bars / train_expanding / validation_bars / test_bars /
  step_bars / retrain_every_bars / purge_policy / embargo_bars / gap_days / min_train_dates /
  stocks / obs 守卫 + decay_half_life_bars）
- `purge_overlap(train, validation, label_contract, calendar=None)` — 标签区间 `[t,t+H]`
  与验证窗口重叠的训练行剔除
- `apply_embargo(train, embargo_bars, calendar=None)` — 丢弃训练尾部 embargo_bars 个日期
- `purge_before_boundary(train, boundary, label_contract)` — 无验证窗口时对 test boundary 补洞
- `purge_and_embargo` 组合；`nested_splits`（双层 walk-forward 选超参）；`decay_weights`
  （`0.5**((max_pos-pos)/half_life)`）；`stitch_oos_windows`（OOSStitchPolicy：
  ACTIVE_UNTIL_NEXT_RETRAIN / LATEST_LEGAL_ARTIFACT）；`check_fold_order`

### 泄漏防护 `leakage_guard.py`
- **TrainOnlyFitGuard** — context manager：评分期间任何 `fit` 调用 → `RuntimeError`（§13.1/§22）
- **assert_frozen_preprocessing** — 冻结变换追加未来行后共享行输出必须逐行相同（§13.2/13.3）
- **run_all_negative_controls**（§58 负向对照电池，合成小面板毫秒级）：
  `future_poison`（毒化 t 之后行，≤t 预测不变）、`label_poison`（毒化最后 horizon 行标签）、
  `scaler_poison`、`hyperparam_poison`、`universe_poison`、`revision_poison`、
  `execution_clock_poison`（BEFORE_SAME_DAY_VWAP 下 close/vwap/volume 必须被拒）。
  每项返回 `exercised / mutation_effect_verified / status(PASS|NOT_RUN|FAIL|INVALID_FIXTURE)`。

### 训练 `trainer.py`
`train_model(learner_cls, train_ds, validation_ds, *, preprocessing_spec, hyperparam_grid,
label_contract, decision_clock, feature_schema_hash, universe_hash, data_source_hash,
retrain_policy, sample_contract, governance_contract, evaluation_boundary, decay_half_life_bars,
sample_weight_policy, cancel_token)` → `TrainResult(artifact, selected_hyperparams,
validation_scores, preprocessing)`。

单折流程：purge_and_embargo → TRAIN-ONLY 预处理（imputer/winsor/standardize）→
validate_search_grid（§69 批准网格）→ 逐候选 fit + validation rank_ic/icir/mse →
select_best_validation（§28 rank_ic 目标）→ refit → freeze ModelArtifact。
**Fail-closed**：无候选达标抛 ValueError，绝不返回降级 artifact。

### Artifact 生命周期 `artifact.py` + `model_catalog.py` + `registry.py`
- **ModelArtifact**：manifest（§21 血统）+ learner + frozen + FrozenPreprocessing + fit_info。
  Manifest 含 MF-P0-005 细粒度时间字段（final_fit_anchor_end / label_maturity_cutoff /
  fit_completed_at / artifact_available_at / activation_at），乱序 fail-closed；`lineage_hash()`。
- `predict` / `predict_oos` 强制 PredictionContext（application_window + dates + asof +
  feature_schema_hash），schema hash 不匹配 / 不可用 asof / 窗口越界均抛错；`is_legal_asof`；
  `cache_key`（§52 含 temporal 身份）；`save`（拒绝覆盖，原子 link+fsync）/ `load`。
- `model_catalog.py`（sqlite 持久化）、`model_semantic_registry.py`（MODEL_SEMANTIC_REGISTRY /
  PRODUCTION_LANES）、`ModelRegistry`（§48）。

### 预测 `predictor.py`
`Predictor.predict(artifact, X, context)`（内部 guard 断言绝不 fit）、`predict_panel`、`batch_predict`。

### 评估 `evaluation.py`
- `ICMetricConvention`（每日横截面 Spearman、ddof=1、min_pairs≥3，违反抛 EvaluationContractError）
- `per_date_rank_ic` / `ic_series` → (dates, daily IC)；`cross_sectional_ic` → {rank_ic, icir, n_dates}
- `block_aware_ic`（按 overlap_horizon 分块）
- `evaluate_predictions(pred, y, dates, security_ids, ...)` → **EvaluationReport**：
  mse/mae/rank_ic/pearson_ic/icir/ic_positive_ratio/long_short_spread/turnover/coverage/
  subperiod_stability/year_by_year/rolling_oos_ic/bull_bear/large_small_cap/liquidity_bucket/
  coverage_layers/portfolio_support/metric_values

### 内置学习器 `learners/`（5 个，线性族）
`PCRLearner`（主成分回归）、`PLSLearner`（偏最小二乘）、`ElasticNetLearner`、
`RegimeLearner`（regime 门控）、`MixtureOfExpertsLearner`（MoE）。
`LEARNER_REGISTRY` + `register_learner/get_learner`；`effective_parameter_count`（真实自由度）；
`default_sample_contracts()` 三档（linear / regime / moe）；非收敛抛 `ModelConvergenceError`。
（注：内置学习器不含树模型；LightGBM 训练在 `lightgbm_qs` 仓库。）

### 辅助
`selection.py`、`hyperparams.py`（validate_search_grid）、`timing.py`、`presets.py`、
`sample_policy.py`、`diagnostics.py`（参数稳定性/特征消融/label-shuffle 对照/复杂度）、
`evidence.py`、`monitoring.py`（漂移：PSI、production drift metrics）、`ledger.py`、
`benchmark.py`、`trainer_governance.py`、`dsl_bridge.py`、`legacy.py`。

## 安装

```bash
cd /home/sunhaiwei/quant_projects/modeling
pip install -r requirements.txt
```

依赖版本以 `pyproject.toml` 为唯一来源；`requirements.txt` 只提供源码目录安装入口。

## 示例

```python
from modeling.dataset import PanelDataset, FeatureSchema
from modeling.trainer import train_model
from modeling.walk_forward import make_walk_forward_splits, purge_and_embargo
from modeling.contracts import LabelContract, AFTER_CLOSE_TO_NEXT_VWAP

label = LabelContract(label_name="fwd_vwap_5d", horizon_bars=5,
                      return_basis="vwap_to_vwap")
folds = make_walk_forward_splits(ds, spec=...)
train_ds = purge_and_embargo(folds[0].train, label)
artifact = train_model(train_ds, spec=..., learner=...)
pred = artifact.predict_oos(folds[0].valid)   # 泄漏安全
```

## 硬性规则

- **vwap-to-vwap** 标签唯一口径（`LabelContract.return_basis`）。
- **禁止** shuffle / 非时序切分 —— 一律日期权威。
- 预处理在 scoring 前必须冻结为 train-only 状态。
- 测试从 monorepo **根目录**跑：`pytest tests/modeling -q`（33 个测试文件）。

## 依赖与接口（谁 import 谁）

- **依赖**：`numpy`/`pandas`/`scipy`（无外部平台库硬依赖，纯独立）。
- **被谁调用**：`lightgbm_qs`（训练管线）、`quant_platform`（model DTO 映射）。

## 相关仓库

- **factor_preprocess** — 上游特征管线（`FeatureBundle`）；modeling 强制 frozen-preprocessing
- **quant_evaluator** — 模型 OOS 独立指标校验
- **data_access** — 面板数据读
