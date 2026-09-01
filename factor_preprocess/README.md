# factor_preprocess — 因子预处理层

把选出的因子变成模型可用的特征：**中性化（行业/市值）、标准化、去极值、滚动/平滑、regime 检测**，
输出带 deep-freeze / 因果契约的 `FeatureBundle`。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess (私有)

## 安装与第一步

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess.git
cd factor_preprocess
pip install -e .            # + [fast] 装 numba/polars/bottleneck
```

```python
from factor_preprocess.registry.policies import get_default_policy_registry
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.contracts.feature_bundle import FeatureBundle

policies = get_default_policy_registry()     # 预设：production_full / research_full / cs_only / ...
policy = policies.get("production_full")
# 构建 transform 链 → FittedState → FeatureBundle
```

## 拥有什么

- **策略预设（`registry/policies.py`）**：`production_full`、`research_full`、`cs_only`、
  `minimal`、`returns_preprocessing`、`causal_basic`、... 每个是一串 `TransformStep`
  （cs_winsor → cs_rank/cs_zscore → neutralization → smoothing...）。
- **变换（`transforms/`）**：截面（rank/zscore/demean/winsor/scale）、滚动（mean/std/zscore）、
  平滑（ewma/kama/kalman/低通/robust_ewma）、波动率（scale / realized vol / garch-inspired）、
  event_decay、freshness、missingness、treatment_variants。
- **中性化（`neutralization/`）**：OLS / ridge / lasso / elastic-net，`NeutralizationSpec`
  （method、exposures、standardization、PIT identity），诊断 artifact（秩亏处理）。
- **Regime（`regime/`）**：方差/相关 regime 检测、自适应权重、因果检测。
- **表示层（`representation/`）**：`linear_ready` / `tree_ready` / `neural_ready` / `multichannel`。
- **资格（`eligibility/`）**：treatment 搜索空间 + 资格规则（factor family × treatment family）。
- **契约（`contracts/`）**：`FeatureBundle`、`FittedState`、`TreatmentRecipe`、
  `TreatmentLineage`、`FactorProfileArtifact`、`PreprocessingPolicy`（deprecated view）。
- **后端/内核**：polars backend + selector；numba 快内核 + reference bridge（parity 测试）。

## 因果/泄漏纪律

- `TreatmentRecipe` 的 `FitBoundary` + `FittedState` deep-freeze；变换逐行确定性，
  只依赖 train-only 冻结状态（配合 `modeling.leakage_guard.assert_frozen_preprocessing`）。
- KAMA 等平滑默认严格因果 `shift(1)`；`use_current` 仅当外部时钟（`DecisionClock`）
  已决定可用性时才启用。

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`data_access`（`adapters/data_access.py`）、
  `factor_assets`（`adapters/factor_assets.py`）。
- **被谁调用**：`modeling`（模型输入 FeatureBundle）、`quant_platform`（feature_set DTO 映射）、
  `lightgbm_qs`（`cs_winsor/cs_rank/cs_zscore` 直接复用）。

## 相关仓库

- **data_access** — universe/日历/行情
- **factor_assets** — 适配器
- **modeling** — 下游模型消费
- **quant_platform** — feature_set DTO
