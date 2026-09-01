# factor_preprocess — 因子预处理层（模型输入工程）

把选出的因子变成模型可用的特征：**中性化（行业/市值/beta）、标准化、去极值、滚动/平滑、
缺失/新鲜度处理、regime 检测**，输出带 deep-freeze / 因果契约的 `FeatureBundle`。

**定位:** 企业级 A 股横截面日频多因子量化项目的**预处理层** —— 因子评估/入库之后、
模型训练之前的一环。

**版本:** 0.1.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess (私有)
**文档:** [`docs/README.md`](docs/README.md) · `docs/DLIB-FP-PHASE0.md` · schemas/feature_manifest.schema.json

---

## 它是什么 / 不是什么

**做什么：** 把 1 个（或多个）因子值的时序 × 截面网格，逐变换转换为模型输入：
去极值 → 缺失填充 → 平滑 → 截面归一 → 行业/市值中性化 → 输出带内容哈希与因果契约的
`FeatureBundle`。每一步都可以由 `TransformRegistry` 查函数、由 `PolicyPreset` 编排。

**不做什么：** 不算因子值（factor_engine 做）、不评估因子（quant_evaluator 做）、
不训练模型（modeling 做）。它只负责"预处理"这一件事，且每一步都有**内容哈希**与
**因果安全**契约。

## 安装与第一步

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess.git
cd factor_preprocess
pip install -e .                # 依赖 numpy/pandas/scipy/statsmodels/PyWavelets
pip install -e ".[fast]"        # + numba / bottleneck / polars 加速内核
```

```python
from factor_preprocess.registry.policies import get_default_policy_registry
from factor_preprocess.registry.transforms import get_default_registry

policy = get_default_policy_registry().get("cs_only")   # PolicyPreset
registry = get_default_registry()                        # TransformRegistry（36 个变换）

# 1) 校验策略在当前 registry 下可执行
ok, errors = policy.validate(registry)

# 2) 逐步骤执行变换链
for step in policy.steps:
    fn = registry.get_function(step.name)
    # out = fn(factor_values, **step.parameters)   → FeatureBundle
```

## 拥有什么（功能全清单）

### 策略预设 `registry/policies.py`
内置 6 个 `PolicyPreset`，每个是 `PolicyLevel`（RESEARCH / STAGING / PRODUCTION）标注的
一串变换步骤；**生产级策略强制 causal_safe**；`policy_identity` 是步骤+参数+实现的内容哈希。

| Preset | 用途 |
|---|---|
| `cs_only` | 纯横截面归一（rank 归一化） |
| `causal_basic` | 因果安全 + EWMA |
| `production_full` | 生产级完整链：forward_fill → missing_indicator → winsor → ewma → volatility_scale → cs_rank → cs_zscore → ols_neutralize（要求 universe） |
| `returns_preprocessing` | 收益序列专用 |
| `minimal` | 最小链 |
| `research_full` | 研究级完整链 |

### 变换全清单 `transforms/`（36 个注册项，`TransformRegistry`）

- **截面族 CROSS_SECTIONAL**：`cs_rank`（横截面排名）、`cs_zscore`、`cs_demean`、
  `cs_winsor`（去极值/截尾）、`cs_scale`
- **时序族 TEMPORAL**：`rolling_mean`、`rolling_std`、`rolling_zscore`；
  平滑族 `ewma`、`trailing_sma`（滞后 SMA）、`trailing_median`（滞后滚动中位数）、
  `robust_ewma`（winsor 后 EWMA）、`kama`（Kaufman 自适应 MA，仅前向）、
  `one_sided_iir_lowpass`（单侧 IIR 低通）、`kalman_local_level`（单侧卡尔曼）、
  `event_decay`（短半衰期事件衰减）；`detect_correlation_regime`（RESEARCH_ONLY）
- **波动率族 VOLATILITY**：`volatility_scale`、`volatility_scale_returns`、`realized_volatility`
- **缺失族 MISSINGNESS**：`forward_fill`、`missing_indicator`、`missing_run_length`、
  `missing_rate`、`impute_with_fallback`（RESEARCH_ONLY）
- **新鲜度族 FRESHNESS**：`days_since_update`、`observation_age`、`freshness_score`、
  `stale_data_indicator`、`freshness_aware_fill`（新鲜度感知指数衰减填充）
- **中性化族 NEUTRALIZATION**：`ols_neutralize`（内核）、`compute_exposures`；
  语义别名 `industry_neutral` / `size_neutral` / `dual_neutral`（全部绑定到 ols_neutralize 内核）
- **分解族（OFFLINE_ONLY，全序列非因果，禁生产）**：`bandpass_filter`、`extract_cycle`、
  `christiano_fitzgerald_filter`、`wavelet_decompose`/`wavelet_smooth`/`wavelet_denoise`、
  `hp_filter`/`hp_decompose`（HP 滤波非前缀不变，禁生产）

每个变换带 `TransformMetadata`（深度不可变：`signature_hash` / `implementation_hash` /
`numeric_policy_hash` 三哈希 + 语义元数据 semantic_id/stage/family_tags）。
`TransformRegistry.seal()` 可冻结成快照身份；`diagnostic_events()` 记录审计轨迹。

### 中性化 `neutralization/`
- `NeutralizationSpec`（frozen dataclass）：
  - `NeutralizationMethod`：**OLS / RIDGE / HUBER**（生产可入）；PCA / KERNEL / QUANTILE / LAD（研究向）
  - `ExposureSet`：NONE / INDUSTRY / SIZE / INDUSTRY_SIZE / INDUSTRY_SIZE_BETA / CUSTOM_STYLE_SET
  - `Standardization`：NONE / ZSCORE / RANK
  - `ConditionNumberPolicy`：FAIL / WARN / REGULARIZE
  - `PitIdentity`：PIT / AS_OF / RESTATED
- 内核：`ols.py`（ols_neutralize + compute_exposures）、`regularized.py`（ridge/huber）、
  `diagnostics.py` + `diagnostics_artifact.py`（`NeutralizationDiagnostics` / 秩亏处理）、
  `advanced/`（pca / kernel / quantile / robust 回归，研究向）

### Regime 检测 `regime/`
- `detect_variance_regime`（方差 regime）、`detect_correlation_regime`（全样本，RESEARCH_ONLY）、`RegimeState`
- `regime_adaptive_weights` / `fit_regime_weights`（自适应权重）、`RegimeWeightState`
- `regime_switching_transform` / `fit_regime_switching`（切换）、`CausalRegimeDetector`（因果安全检测器）

### 表示层 `representation/`
`linear_ready` / `tree_ready` / `neural_ready` / `multichannel` —— 按下游模型类型产出
就绪格式的表示。

### 资格引擎 `eligibility/`
按因子语义元数据查询可用变换，构成"自动处理寻优"的合规空间（配合 factor_optimizer 使用）。

### 契约 `contracts/`（全部深度不可变 + 内容哈希）
- **FeatureBundle**：`bundle_id` + 时间/资产轴（AxisRef）+ channels（feature/missing/freshness/exposure）
  + 值数组；布局 NF=[asset,feature] / TNF=[time,asset,feature]；`FeatureManifest` 精确平铺
  特征轴（无洞无重叠）；`is_immutable()`；数值自动校验。
- **FittedState**：transform 名/版本 + fit 窗口 + `state_kind`（STATELESS/FITTED）；
  `production=True` 时 state_id 内容派生且必需全部 provenance 字段（fail-closed）。
- **TreatmentRecipe**（内容寻址）：`ordered_steps` + neutralization_spec + `FitBoundary`
  （EXPANDING/ROLLING/FULL_SAMPLE_RESEARCH）；`content_hash` 派生并校验。
- **FactorProfileArtifact**（内容哈希）：7 个语义族（PRICE_VOLUME / HIGH_TURNOVER /
  FUNDAMENTAL / SPARSE_UPDATE / EVENT / BINARY / DISCRETE）+ 分布/时序行为/数据行为/暴露指标；
  `split_ref` 限定只能从 TRAIN 计算（治理契约）。
- **treatment_lineage.py**：TransformStage / TransformSemanticID / TransformLineage /
  `map_fe_dsl_to_semantic`（把 factor_engine DSL 算子映射到语义变换）。

### 后端/内核
- `backends/`：numpy / numba / cupy / polars 自动后端选择
- `kernels/fast.py`：滚动均值 / CS rank / zscore 等快速内核 + numba_transforms + reference_bridge（parity 测试）
- `errors.py`：25+ 错误类型（ContractError / TimingContractError / FittedStateMismatchError 等）

## 因果/泄漏纪律

- `TreatmentRecipe.FitBoundary` + `FittedState` deep-freeze：变换逐行确定性，只依赖 train-only
  冻结状态（配合 `modeling.leakage_guard.assert_frozen_preprocessing` 证明）。
- KAMA 等平滑默认严格因果 `shift(1)`；`use_current` 仅当外部时钟（`DecisionClock`）已决定可用性时才启用。
- 生产策略强制 `causal_safe`；分解族（小波/HP/滤波）标 OFFLINE_ONLY，禁止进生产。

## 测试

```bash
cd factor_preprocess && pytest tests/ -q     # 12 个测试文件
```
markers：future_poison / fold_local / parity / benchmark。

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`data_access`（`adapters/data_access.py`：行业/市值/beta 暴露）、
  `factor_assets`（`adapters/factor_assets.py`）。
- **被谁调用**：`modeling`（模型输入 FeatureBundle）、`quant_platform`（feature_set DTO 映射）、
  `lightgbm_qs`（`cs_winsor/cs_rank/cs_zscore` 直接复用）。

## 相关仓库

- **data_access** — universe/日历/行情/暴露数据
- **factor_assets** — 因子资产适配器
- **modeling** — 下游模型消费（frozen preprocessing）
- **quant_platform** — feature_set DTO
- **factor_optimizer** — eligibility 搜索空间配合
