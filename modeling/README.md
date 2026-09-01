# modeling — 模型层（ML 训练 / walk-forward / 泄漏防护）

模型层的单一真相源：训练、walk-forward 日期权威切分、标签区间 purge、embargo、
泄漏防护、artifact 生命周期、预测评估。**所有时序泄漏防护都在本库执行。**

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/modeling (私有)
**权威声明:** `AUTHORITY.md`（MODEL2-P0-006）

## 安装

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/modeling.git
cd modeling
pip install -e .
```

## 核心概念

- **日期权威切分（date-authoritative）** — 每个日期只属于一个切分。`date_bounded_split`、
  `split_by_date_cutoff`、`assert_date_authoritative`。
- **Walk-forward** — `make_walk_forward_splits`、`WalkForwardSpec`（含 `gap_days`）、
  `purge_overlap`（标签区间 purge）、`apply_embargo`、`purge_before_boundary`、
  `check_fold_order`、`nested_splits`、`stitch_oos_windows`。
- **泄漏防护** — `TrainOnlyFitGuard`（证明 fit 不在 scoring 中被调用）、
  `assert_frozen_preprocessing`；外加一组**负向对照**（`future_poison`、`label_poison`、
  `scaler_poison`、`hyperparam_poison`、`universe_poison`、`revision_poison`、
  `execution_clock_poison`、`run_all_negative_controls`）。
- **时序契约** — `DecisionClock`（decision at `t_close`，execution at `t+1 VWAP`，
  label matured at `t+H close`）、`LabelContract`（`return_basis="vwap_to_vwap"`、
  `purge_by_interval`、`embargo_bars`）、`EmbargoSpec`、`ApplicationWindow`、
  `AFTER_CLOSE_TO_NEXT_VWAP` / `BEFORE_SAME_DAY_VWAP`。
- **训练** — `train_model(train_ds, spec, learner, ...)` → `ModelArtifact`
  （frozen preprocessing + model + manifest）；`TrainResult`；治理
  （`GovernanceContract`、`CandidateExposureLedger`、`select_best_validation`、
  `neighborhood_stability`）；`hyperparams.validate_search_grid`。
- **预测** — `Predictor`、`predict_oos`（强制 OOS 安全）、`PredictionOutputContract`
  （finite / aligned / status-gated）、`PredictionBatch`。
- **评估** — `per_date_rank_ic`、`ic_series`、`cross_sectional_ic`、`block_aware_ic`、
  `evaluate_predictions`、`EvaluationReport`、`run_walk_forward_evidence`。
- **学习器（`learners/`）** — base + elastic_net、pls、pcr、regime、mixture_of_experts
  （五类执行模型：local-rolling / recursive-state / same-time cross-sectional /
  predictive-supervised / research-structural，见 `ModelExecutionClass`）。
- **artifact 生命周期** — `artifact.py`（ModelArtifact、FrozenPreprocessing、
  ModelArtifactManifest）、`model_catalog.py`、`registry.py`、`monitoring.py`
  （漂移：PSI、production drift metrics）、`diagnostics.py`（参数稳定性、特征消融、
  label-shuffle 对照、复杂度）、`evidence.py`、`ledger.py`、`timing.py`、`dsl_bridge.py`、`sample_policy.py`。

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
