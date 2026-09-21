# factor_preprocess 完整模块与接口索引

先读 [功能与算法手册](FUNCTIONAL_GUIDE.md)，再查本页的具体入口、参数和实现位置。
扫描实际包目录：**73 个 Python 模块、426 个公开函数/类/方法定义**。
收录非下划线开头的顶层定义及类的公开方法，不把所有内部模块都承诺为稳定API；私有辅助算法见功能手册。
参数、类型、默认值直接取自源码语法树，不导入或启动可选后端。类型注解不代表生产可用性。
未写独立说明的入口会明确标记，不凭名称编造功能；算法讲解、约束、完整流程与例子见功能手册。

## 模块目录

| 模块 | 定义数 | 模块说明 |
|---|---:|---|
| [factor_preprocess/__init__.py](../factor_preprocess/__init__.py) | 1 | Factor Preprocess: Model-input preparation layer. |
| [factor_preprocess/adapters/__init__.py](../factor_preprocess/adapters/__init__.py) | 0 | Adapters package - optional integration with other packages. |
| [factor_preprocess/adapters/_data_access_impl.py](../factor_preprocess/adapters/_data_access_impl.py) | 7 | Default exposure provider backed by the optional ``data_access`` package. |
| [factor_preprocess/adapters/data_access.py](../factor_preprocess/adapters/data_access.py) | 15 | Data Access adapter - optional integration with the data_access package. |
| [factor_preprocess/adapters/ewma_full_replay.py](../factor_preprocess/adapters/ewma_full_replay.py) | 3 | Exact, bounded restart replay for the public lagged FP EWMA. |
| [factor_preprocess/adapters/factor_assets.py](../factor_preprocess/adapters/factor_assets.py) | 12 | Factor Assets adapter - optional integration with factor_assets package. |
| [factor_preprocess/adapters/fe_operator.py](../factor_preprocess/adapters/fe_operator.py) | 7 | FE operator adapter (R61-FI-041, plan §26 F2/F3). |
| [factor_preprocess/adapters/fitted_recipe.py](../factor_preprocess/adapters/fitted_recipe.py) | 2 | Apply-only execution of narrowly supported frozen FP fitted state. |
| [factor_preprocess/backends/__init__.py](../factor_preprocess/backends/__init__.py) | 0 | Automatic backend selection for factor_preprocess. |
| [factor_preprocess/backends/polars_backend.py](../factor_preprocess/backends/polars_backend.py) | 6 | Polars backend for high-performance cross-sectional transforms. |
| [factor_preprocess/backends/registry.py](../factor_preprocess/backends/registry.py) | 10 | Backend registry for managing computational kernels across backends. |
| [factor_preprocess/backends/selector.py](../factor_preprocess/backends/selector.py) | 11 | Backend selector for automatic backend selection based on data size and capabilities. |
| [factor_preprocess/contracts/__init__.py](../factor_preprocess/contracts/__init__.py) | 0 | Contracts package. |
| [factor_preprocess/contracts/_deep_freeze.py](../factor_preprocess/contracts/_deep_freeze.py) | 2 | Deep immutability helper (package-local; not a shared common/utils). |
| [factor_preprocess/contracts/factor_profile.py](../factor_preprocess/contracts/factor_profile.py) | 2 | Factor profile artifact contract for the auto-treatment optimizer. |
| [factor_preprocess/contracts/feature_bundle.py](../factor_preprocess/contracts/feature_bundle.py) | 23 | Feature bundle contract for model input. |
| [factor_preprocess/contracts/fit_apply.py](../factor_preprocess/contracts/fit_apply.py) | 12 | Fit-apply governance contract for treatments (R61-FI-029, plan §29). |
| [factor_preprocess/contracts/lineage_policy.py](../factor_preprocess/contracts/lineage_policy.py) | 10 | Lineage deduplication & canonicalization policy (R61-FI-043, plan §26 F4). |
| [factor_preprocess/contracts/policy.py](../factor_preprocess/contracts/policy.py) | 5 | Preprocessing policy and transform specification contracts. |
| [factor_preprocess/contracts/state.py](../factor_preprocess/contracts/state.py) | 3 | Fitted transform state contract. |
| [factor_preprocess/contracts/treatment_lineage.py](../factor_preprocess/contracts/treatment_lineage.py) | 15 | Treatment lineage contracts for the auto-treatment optimizer. |
| [factor_preprocess/contracts/treatment_recipe.py](../factor_preprocess/contracts/treatment_recipe.py) | 12 | TreatmentRecipe — canonical prescription for applying a treatment to a factor. |
| [factor_preprocess/contracts/treatment_spec.py](../factor_preprocess/contracts/treatment_spec.py) | 12 | Treatment identity split — SPEC vs MATERIALIZATION (P0-FP #103). |
| [factor_preprocess/eligibility/__init__.py](../factor_preprocess/eligibility/__init__.py) | 0 | Eligibility package for the auto-treatment optimizer. |
| [factor_preprocess/eligibility/engine.py](../factor_preprocess/eligibility/engine.py) | 6 | Treatment eligibility engine for the auto-treatment optimizer. |
| [factor_preprocess/eligibility/rules.py](../factor_preprocess/eligibility/rules.py) | 13 | EligibilityRuleRegistry — the *single* authority for which semantic treatment families a factor type may search (DLIB-FP-014). |
| [factor_preprocess/errors.py](../factor_preprocess/errors.py) | 29 | Core error taxonomy for factor_preprocess. |
| [factor_preprocess/grammar/__init__.py](../factor_preprocess/grammar/__init__.py) | 0 | Search grammar package for the auto-treatment optimizer. |
| [factor_preprocess/grammar/search_grammar.py](../factor_preprocess/grammar/search_grammar.py) | 10 | Search grammar for the auto-treatment optimizer. |
| [factor_preprocess/kernels/__init__.py](../factor_preprocess/kernels/__init__.py) | 0 | Kernels package - reference bridge for fast implementations. |
| [factor_preprocess/kernels/fast.py](../factor_preprocess/kernels/fast.py) | 8 | Fast kernel implementations for factor preprocessing. |
| [factor_preprocess/kernels/numba_transforms.py](../factor_preprocess/kernels/numba_transforms.py) | 9 | Numba-accelerated transforms for factor preprocessing. |
| [factor_preprocess/kernels/reference_bridge.py](../factor_preprocess/kernels/reference_bridge.py) | 5 | Reference implementation bridge for parity checking. |
| [factor_preprocess/neutralization/__init__.py](../factor_preprocess/neutralization/__init__.py) | 0 | Neutralization package. |
| [factor_preprocess/neutralization/advanced/__init__.py](../factor_preprocess/neutralization/advanced/__init__.py) | 0 | Advanced neutralization methods. |
| [factor_preprocess/neutralization/advanced/kernel_regression.py](../factor_preprocess/neutralization/advanced/kernel_regression.py) | 1 | Kernel-based non-parametric neutralization for cross-sectional residuals. |
| [factor_preprocess/neutralization/advanced/pca_neutralization.py](../factor_preprocess/neutralization/advanced/pca_neutralization.py) | 1 | PCA-based neutralization for cross-sectional residuals. |
| [factor_preprocess/neutralization/advanced/quantile_regression.py](../factor_preprocess/neutralization/advanced/quantile_regression.py) | 1 | Quantile regression neutralization for cross-sectional residuals. |
| [factor_preprocess/neutralization/advanced/robust_regression.py](../factor_preprocess/neutralization/advanced/robust_regression.py) | 2 | Robust regression neutralization for cross-sectional residuals. |
| [factor_preprocess/neutralization/diagnostics.py](../factor_preprocess/neutralization/diagnostics.py) | 7 | Diagnostics for neutralization quality and numerical stability. |
| [factor_preprocess/neutralization/diagnostics_artifact.py](../factor_preprocess/neutralization/diagnostics_artifact.py) | 2 | Neutralization diagnostics artifact (DLIB-FP-020 / DLIB-FP-021). |
| [factor_preprocess/neutralization/ols.py](../factor_preprocess/neutralization/ols.py) | 2 | OLS neutralization for cross-sectional residuals. |
| [factor_preprocess/neutralization/regularized.py](../factor_preprocess/neutralization/regularized.py) | 3 | Regularized cross-sectional neutralization with explicit solver evidence. |
| [factor_preprocess/neutralization/spec.py](../factor_preprocess/neutralization/spec.py) | 9 | Neutralization specification contract for the auto-treatment optimizer. |
| [factor_preprocess/regime/__init__.py](../factor_preprocess/regime/__init__.py) | 0 | Regime-adaptive transforms for factor preprocessing. |
| [factor_preprocess/regime/adaptive_weights.py](../factor_preprocess/regime/adaptive_weights.py) | 6 | Regime-dependent factor weighting. |
| [factor_preprocess/regime/causal_detector.py](../factor_preprocess/regime/causal_detector.py) | 5 | Stateful, checkpointable past-only correlation regime detector. |
| [factor_preprocess/regime/detector.py](../factor_preprocess/regime/detector.py) | 3 | Online regime detection for volatility and correlation regimes. |
| [factor_preprocess/regime/switching.py](../factor_preprocess/regime/switching.py) | 3 | Regime-based transform switching. |
| [factor_preprocess/registry/__init__.py](../factor_preprocess/registry/__init__.py) | 0 | Registry package - transform and policy catalogs. |
| [factor_preprocess/registry/policies.py](../factor_preprocess/registry/policies.py) | 14 | Policy presets for transform pipelines. |
| [factor_preprocess/registry/transforms.py](../factor_preprocess/registry/transforms.py) | 27 | Transform registry with versioning and discovery. |
| [factor_preprocess/representation/__init__.py](../factor_preprocess/representation/__init__.py) | 0 | Representation package - multi-channel representations for model input. |
| [factor_preprocess/representation/linear_ready.py](../factor_preprocess/representation/linear_ready.py) | 5 | Linear model ready features: dense feature matrix builder. |
| [factor_preprocess/representation/multichannel.py](../factor_preprocess/representation/multichannel.py) | 6 | Multichannel representation: raw, rank, zscore, residual channels. |
| [factor_preprocess/representation/neural_ready.py](../factor_preprocess/representation/neural_ready.py) | 5 | Neural network ready features: robust scaling and embedding preparation. |
| [factor_preprocess/representation/policy.py](../factor_preprocess/representation/policy.py) | 17 | Model-specific representation policy (R61-FI-044, plan §28). |
| [factor_preprocess/representation/tree_ready.py](../factor_preprocess/representation/tree_ready.py) | 5 | Tree model ready features: categorical encoding stubs. |
| [factor_preprocess/transforms/__init__.py](../factor_preprocess/transforms/__init__.py) | 0 | Transforms package. |
| [factor_preprocess/transforms/cross_sectional.py](../factor_preprocess/transforms/cross_sectional.py) | 5 | Cross-sectional transforms: rank, zscore, demean, winsor. |
| [factor_preprocess/transforms/decomposition/__init__.py](../factor_preprocess/transforms/decomposition/__init__.py) | 0 | Time Series Decomposition Transforms |
| [factor_preprocess/transforms/decomposition/cycle.py](../factor_preprocess/transforms/decomposition/cycle.py) | 3 | Cycle extraction using full-series bandpass filters. |
| [factor_preprocess/transforms/decomposition/seasonal.py](../factor_preprocess/transforms/decomposition/seasonal.py) | 5 | Seasonal decomposition using STL (Seasonal and Trend decomposition using Loess). |
| [factor_preprocess/transforms/decomposition/trend.py](../factor_preprocess/transforms/decomposition/trend.py) | 2 | Trend extraction using Hodrick-Prescott filter. |
| [factor_preprocess/transforms/decomposition/wavelet.py](../factor_preprocess/transforms/decomposition/wavelet.py) | 3 | Wavelet decomposition for offline multi-scale time series analysis. |
| [factor_preprocess/transforms/event_decay.py](../factor_preprocess/transforms/event_decay.py) | 1 | Short-halflife event-decay persistence (causal, one-sided). |
| [factor_preprocess/transforms/freshness.py](../factor_preprocess/transforms/freshness.py) | 4 | Data freshness transforms for tracking observation age and staleness. |
| [factor_preprocess/transforms/missingness.py](../factor_preprocess/transforms/missingness.py) | 7 | Missingness transforms for handling missing data. |
| [factor_preprocess/transforms/repair_shapes.py](../factor_preprocess/transforms/repair_shapes.py) | 6 | Stateless value-repair primitives used by research repair plans. |
| [factor_preprocess/transforms/rolling.py](../factor_preprocess/transforms/rolling.py) | 4 | Rolling (time-series) transforms with explicit causality. |
| [factor_preprocess/transforms/smoothing.py](../factor_preprocess/transforms/smoothing.py) | 6 | Causal one-sided signal smoothers. |
| [factor_preprocess/transforms/treatment_variants.py](../factor_preprocess/transforms/treatment_variants.py) | 1 | Treatment variant transforms that the eligibility engine may propose. |
| [factor_preprocess/transforms/volatility.py](../factor_preprocess/transforms/volatility.py) | 5 | Volatility scaling transforms with explicit causality. |

## factor_preprocess/__init__.py

Factor Preprocess: Model-input preparation layer.

显式导出（含重导出）：`PolicyPreset`、`PolicyRegistry`、`PolicyLevel`、`TransformStep`、`create_default_policies`、`get_default_policy_registry`、`TransformRegistry`、`TransformMetadata`、`TransformCategory`、`create_default_registry`、`get_default_registry`、`TransformStage`、`TransformSemanticID`、`TransformLineage`、`ExistingTreatmentSignature`、`ExistingTreatmentStatus`、`build_signature_from_lineage`、`map_fe_dsl_to_semantic`、`OutputProperties`、`derive_output_properties`、`LINEAGE_POLICY_VERSION`、`IDEMPOTENT_SEMANTIC_CLASSES`、`NEUTRALIZATION_TOPOLOGY`、`RedundancyClass`、`LineagePolicyDecision`、`RedundantTransformError`、`UnsupportedPolicyVersionError`、`LineagePolicyError`、`canonicalize_lineage`、`is_losslessly_collapsible`、`TreatmentRecipe`、`RecipeStep`、`FitBoundary`、`RecipeSchemaVersion`、`FactorProfileArtifact`、`NeutralizationDiagnostics`、`RankDeficientResolution`、`NeutralizationSpec`、`PriceBasis`、`ValueUnit`、`MaterializationSplit`、`TreatmentSpecIdentity`、`TreatmentMaterializationIdentity`、`neutralization_spec_identity`、`neutralization_binding`、`ensure_materialization_split_valid`、`materialize_identity`、`spec_to_materialization_identity`、`ASHARE_INDUSTRY_SCHEMA`、`ASHARE_SIZE_DEFINITION`、`CausalityClass`、`FitScope`、`ApplicationSplit`、`TreatmentFitApplyDeclaration`、`assert_prefix_invariant`、`assert_fit_state_context_match`、`assert_label_access_legal`、`RepresentationProfileId`、`REPRESENTATION_POLICY_VERSION`、`RepresentationPolicy`、`UnknownRepresentationProfileError`、`get_representation_policy`、`list_representation_policies`、`ArtifactKind`、`CANONICAL_FACTOR_NAMESPACE_PREFIX`、`REPRESENTATION_NAMESPACE_PREFIX`、`CanonicalAssetOverwriteError`、`FeatureRepresentationArtifact`、`register_feature_representation`、`write_feature_representation`、`NonInferiorityTolerance`、`NON_INFERIORITY_POLICY_VERSION`、`DEFAULT_NON_INFERIORITY_TOLERANCE`、`non_inferior`、`SignalDestructionConflict`、`FittedState`、`PreprocessingPolicy`、`TransformSpec`、`TransformKind`、`TransformMode`、`FeatureBundle`、`AxisRef`、`ChannelRef`、`FeatureManifest`、`FactorPreprocessError`、`ContractError`、`SchemaVersionError`、`MissingInputError`、`InvalidContractError`、`TimingContractError`、`SnapshotMismatchError`、`TreatmentMaterializationError`、`CapabilityError`、`UnsupportedTransformError`、`OptionalDependencyMissing`、`DataError`、`InsufficientObservations`、`InvalidValidityMask`、`MissingFittedStateError`、`StaleFittedStateError`、`ExecutionError`、`NumericalFailure`、`OverflowOrNonFiniteError`、`BudgetExceededError`、`CancellationError`、`GovernanceError`、`FullSampleFitError`、`FittedStateMismatchError`、`ContractChangeRequired`。

### package_info

[实际实现](../factor_preprocess/__init__.py#L279)。

Return package metadata and capability information.

参数：`()`。

## factor_preprocess/adapters/__init__.py

Adapters package - optional integration with other packages.

显式导出（含重导出）：`FactorAssetsError`、`DataAccessError`、`EWMA_REPLAY_EXECUTION_MODE`、`FrozenEwmaReplaySpec`、`apply_frozen_ewma_full_replay`。

## factor_preprocess/adapters/_data_access_impl.py

Default exposure provider backed by the optional ``data_access`` package.

显式导出（含重导出）：`DefaultExposureProvider`。

### DefaultExposureProvider

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L50)。

Real exposure provider sourcing from the ``data_access`` package.

### DefaultExposureProvider.__init__

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L60)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### DefaultExposureProvider.get_industry_exposure

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L161)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, industry_classification: str='default')`。

返回类型：`Dict[str, Any]`。

### DefaultExposureProvider.get_size_exposure

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L198)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, size_metric: str='market_cap')`。

返回类型：`Dict[str, Any]`。

### DefaultExposureProvider.get_sector_exposure

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L240)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, sector_classification: str='default')`。

返回类型：`Dict[str, Any]`。

### DefaultExposureProvider.get_beta_exposure

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L277)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, window_days: int=252)`。

返回类型：`Dict[str, Any]`。

### DefaultExposureProvider.get_custom_exposure

[实际实现](../factor_preprocess/adapters/_data_access_impl.py#L314)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, exposure_name: str, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, **kwargs: Any)`。

返回类型：`Dict[str, Any]`。

## factor_preprocess/adapters/data_access.py

Data Access adapter - optional integration with the data_access package.

显式导出（含重导出）：`ExposureProvider`、`DataAccessAdapter`、`OptionalDependencyMissing`、`check_data_access_available`、`create_adapter`、`SUPPORTED_EXPOSURE_TYPES`、`REQUIRED_PROVENANCE_KEYS`、`_validate_exposure_bundle`。

### OptionalDependencyMissing

[实际实现](../factor_preprocess/adapters/data_access.py#L16)。

Raised when an optional dependency is required but not available.

基类：`Exception`。

### OptionalDependencyMissing.__init__

[实际实现](../factor_preprocess/adapters/data_access.py#L19)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, package_name: str, feature_name: str)`。

### ExposureProvider

[实际实现](../factor_preprocess/adapters/data_access.py#L169)。

Protocol for exposure data providers.

基类：`Protocol`。

### ExposureProvider.get_industry_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L187)。

Retrieve industry classification exposures.

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, industry_classification: str='default')`。

返回类型：`Dict[str, Any]`。

### ExposureProvider.get_size_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L216)。

Retrieve size exposures (market cap, log market cap, etc.).

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, size_metric: str='market_cap')`。

返回类型：`Dict[str, Any]`。

### ExposureProvider.get_sector_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L244)。

Retrieve sector classification exposures (broader than industry).

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, sector_classification: str='default')`。

返回类型：`Dict[str, Any]`。

### ExposureProvider.get_beta_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L273)。

Retrieve market beta exposures.

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, window_days: int=252)`。

返回类型：`Dict[str, Any]`。

### ExposureProvider.get_custom_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L301)。

Retrieve custom exposure by name.

参数：`(self, exposure_name: str, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, **kwargs: Any)`。

返回类型：`Dict[str, Any]`。

### DataAccessAdapter

[实际实现](../factor_preprocess/adapters/data_access.py#L332)。

Adapter for data_access package integration.

### DataAccessAdapter.__init__

[实际实现](../factor_preprocess/adapters/data_access.py#L340)。

Initialize adapter with a provider.

参数：`(self, provider: ExposureProvider)`。

### DataAccessAdapter.fetch_industry_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L349)。

Fetch industry exposure for neutralization.

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, industry_classification: str='default')`。

返回类型：`Dict[str, Any]`。

### DataAccessAdapter.fetch_size_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L387)。

Fetch size exposure for neutralization.

参数：`(self, market: str, start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, size_metric: str='market_cap')`。

返回类型：`Dict[str, Any]`。

### DataAccessAdapter.fetch_multi_exposure

[实际实现](../factor_preprocess/adapters/data_access.py#L424)。

Fetch multiple exposures in one call for efficiency.

参数：`(self, market: str, exposure_types: List[str], start_date: Optional[str]=None, end_date: Optional[str]=None, assets: Optional[List[str]]=None, **kwargs: Any)`。

返回类型：`Dict[str, Dict[str, Any]]`。

### check_data_access_available

[实际实现](../factor_preprocess/adapters/data_access.py#L504)。

Check if the data_access package is available.

参数：`()`。

返回类型：`bool`。

### create_adapter

[实际实现](../factor_preprocess/adapters/data_access.py#L513)。

Create a DataAccessAdapter instance.

参数：`(provider: Optional[ExposureProvider]=None)`。

返回类型：`DataAccessAdapter`。

## factor_preprocess/adapters/ewma_full_replay.py

Exact, bounded restart replay for the public lagged FP EWMA.

### FrozenEwmaReplaySpec

[实际实现](../factor_preprocess/adapters/ewma_full_replay.py#L39)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `state_key` | `str` | `必填/未声明默认` |
| `source_identity` | `str` | `必填/未声明默认` |
| `feature_identity` | `str` | `必填/未声明默认` |
| `security_identity` | `str` | `必填/未声明默认` |
| `time_identity` | `str` | `必填/未声明默认` |
| `halflife` | `float` | `必填/未声明默认` |
| `min_periods` | `int` | `1` |
| `max_history_rows` | `int` | `100000` |

### FrozenEwmaReplaySpec.identity

[实际实现](../factor_preprocess/adapters/ewma_full_replay.py#L64)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### apply_frozen_ewma_full_replay

[实际实现](../factor_preprocess/adapters/ewma_full_replay.py#L151)。

Append one immutable segment and return its exact lagged FP EWMA values.

参数：`(segment: pd.DataFrame, *, spec: FrozenEwmaReplaySpec, checkpoint_path: str \| Path, bootstrap: bool=False)`。

返回类型：`tuple[pd.Series, dict[str, Any]]`。

## factor_preprocess/adapters/factor_assets.py

Factor Assets adapter - optional integration with factor_assets package.

显式导出（含重导出）：`FactorSetProvider`、`FactorAssetsAdapter`、`OptionalDependencyMissing`、`check_factor_assets_available`、`create_adapter`。

### OptionalDependencyMissing

[实际实现](../factor_preprocess/adapters/factor_assets.py#L13)。

Raised when an optional dependency is required but not available.

基类：`Exception`。

### OptionalDependencyMissing.__init__

[实际实现](../factor_preprocess/adapters/factor_assets.py#L16)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, package_name: str, feature_name: str)`。

### FactorSetProvider

[实际实现](../factor_preprocess/adapters/factor_assets.py#L25)。

Protocol for factor set providers.

基类：`Protocol`。

### FactorSetProvider.get_factor_values

[实际实现](../factor_preprocess/adapters/factor_assets.py#L33)。

Retrieve factor values for a single factor.

参数：`(self, factor_id: str, start_date: Optional[str]=None, end_date: Optional[str]=None, universe: Optional[str]=None)`。

返回类型：`Dict[str, Any]`。

### FactorSetProvider.get_factor_batch

[实际实现](../factor_preprocess/adapters/factor_assets.py#L52)。

Retrieve factor values for multiple factors efficiently.

参数：`(self, factor_ids: list[str], start_date: Optional[str]=None, end_date: Optional[str]=None, universe: Optional[str]=None)`。

返回类型：`Dict[str, Any]`。

### FactorSetProvider.validate_factor_set

[实际实现](../factor_preprocess/adapters/factor_assets.py#L72)。

Validate that a FactorSet is well-formed and factors exist.

参数：`(self, factor_set: Any)`。

返回类型：`bool`。

### FactorAssetsAdapter

[实际实现](../factor_preprocess/adapters/factor_assets.py#L85)。

Adapter for factor_assets package integration.

### FactorAssetsAdapter.__init__

[实际实现](../factor_preprocess/adapters/factor_assets.py#L93)。

Initialize adapter with a provider.

参数：`(self, provider: FactorSetProvider)`。

### FactorAssetsAdapter.load_factor_set

[实际实现](../factor_preprocess/adapters/factor_assets.py#L102)。

Load factor values from a FactorSet.

参数：`(self, factor_set: Any, start_date: Optional[str]=None, end_date: Optional[str]=None)`。

返回类型：`Dict[str, Any]`。

### FactorAssetsAdapter.load_single_factor

[实际实现](../factor_preprocess/adapters/factor_assets.py#L159)。

Load a single factor by ID.

参数：`(self, factor_id: str, start_date: Optional[str]=None, end_date: Optional[str]=None, universe: Optional[str]=None)`。

返回类型：`Dict[str, Any]`。

### check_factor_assets_available

[实际实现](../factor_preprocess/adapters/factor_assets.py#L189)。

Check whether the factor_assets integration can construct its default provider.

参数：`()`。

返回类型：`bool`。

### create_adapter

[实际实现](../factor_preprocess/adapters/factor_assets.py#L199)。

Create a FactorAssetsAdapter instance.

参数：`(provider: Optional[FactorSetProvider]=None)`。

返回类型：`FactorAssetsAdapter`。

## factor_preprocess/adapters/fe_operator.py

FE operator adapter (R61-FI-041, plan §26 F2/F3).

显式导出（含重导出）：`FeOperatorExecutor`、`FeRecipeExecutor`、`get_fe_executor`。

### FeOperatorExecutor

[实际实现](../factor_preprocess/adapters/fe_operator.py#L142)。

Callable executor for one FE canonical operator.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `_PARAM_ALIASES` | `类常量/枚举` | `{'max_lag': 'max_periods', 'min_observations': 'min_obs'}` |

### FeOperatorExecutor.__init__

[实际实现](../factor_preprocess/adapters/fe_operator.py#L151)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, canonical: str, fallback: Optional[Callable]=None, *, allow_research_fallback: bool=False)`。

### FeOperatorExecutor.__call__

[实际实现](../factor_preprocess/adapters/fe_operator.py#L169)。

Route a long-format FP call through the FE operator.

参数：`(self, *args, **kwargs)`。

### FeRecipeExecutor

[实际实现](../factor_preprocess/adapters/fe_operator.py#L262)。

Execute an all-FE stateless TreatmentRecipe with one panel boundary.

### FeRecipeExecutor.__init__

[实际实现](../factor_preprocess/adapters/fe_operator.py#L265)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, recipe, registry, *, execution_context=None, backend=None, allow_research=False)`。

### FeRecipeExecutor.__call__

[实际实现](../factor_preprocess/adapters/fe_operator.py#L273)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, value_col='value', time_col='date', asset_col='asset_id')`。

### get_fe_executor

[实际实现](../factor_preprocess/adapters/fe_operator.py#L301)。

Return a lazily-FE-backed executor for ``canonical``.

参数：`(canonical: str, fallback: Optional[Callable]=None, *, allow_research_fallback: bool=False)`。

返回类型：`Optional[FeOperatorExecutor]`。

## factor_preprocess/adapters/fitted_recipe.py

Apply-only execution of narrowly supported frozen FP fitted state.

显式导出（含重导出）：`FITTED_STANDARDIZE_NAME`、`FITTED_STANDARDIZE_VERSION`、`apply_frozen_fitted_recipe`、`fitted_standardize_implementation_hash`。

### fitted_standardize_implementation_hash

[实际实现](../factor_preprocess/adapters/fitted_recipe.py#L24)。

Content identity of the exact apply-only numeric implementation.

参数：`()`。

返回类型：`str`。

### apply_frozen_fitted_recipe

[实际实现](../factor_preprocess/adapters/fitted_recipe.py#L29)。

Apply one frozen train-standardization state; fitting is impossible here.

参数：`(panel: pd.DataFrame, *, recipe: Any, fitted_states: Mapping[str, FittedState], factor_definition_hash: str, decision_start: Any)`。

返回类型：`tuple[pd.DataFrame, tuple[str, ...]]`。

## factor_preprocess/backends/__init__.py

Automatic backend selection for factor_preprocess.

显式导出（含重导出）：`BackendSelector`、`BackendType`、`BackendCapabilities`、`get_backend_capabilities`、`benchmark_backends`、`BackendRegistry`、`register_backend`、`get_backend`、`polars_backend`。

## factor_preprocess/backends/polars_backend.py

Polars backend for high-performance cross-sectional transforms.

显式导出（含重导出）：`cs_rank_polars`、`cs_zscore_polars`、`cs_demean_polars`、`cs_winsor_polars`、`cs_scale_polars`、`ols_neutralize_polars`、`POLARS_AVAILABLE`。

### cs_rank_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L30)。

Cross-sectional rank using polars for high performance.

参数：`(df: Union[pd.DataFrame, pl.DataFrame], value_col: str, group_col: str='date', method: Literal['average', 'min', 'max', 'dense', 'ordinal']='average', pct: bool=False)`。

返回类型：`Union[pd.Series, pl.Series]`。

### cs_zscore_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L112)。

Cross-sectional z-score using polars for high performance.

参数：`(df: Union[pd.DataFrame, pl.DataFrame], value_col: str, group_col: str='date', ddof: int=1, constant_value: float=0.0)`。

返回类型：`Union[pd.Series, pl.Series]`。

### cs_demean_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L170)。

Cross-sectional demean using polars for high performance.

参数：`(df: Union[pd.DataFrame, pl.DataFrame], value_col: str, group_col: str='date')`。

返回类型：`Union[pd.Series, pl.Series]`。

### cs_winsor_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L215)。

Cross-sectional winsorization using polars for high performance.

参数：`(df: Union[pd.DataFrame, pl.DataFrame], value_col: str, group_col: str='date', lower: float=0.01, upper: float=0.99)`。

返回类型：`Union[pd.Series, pl.Series]`。

### cs_scale_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L272)。

Cross-sectional scaling to target std using polars for high performance.

参数：`(df: Union[pd.DataFrame, pl.DataFrame], value_col: str, group_col: str='date', target_std: float=1.0, ddof: int=1)`。

返回类型：`Union[pd.Series, pl.Series]`。

### ols_neutralize_polars

[实际实现](../factor_preprocess/backends/polars_backend.py#L328)。

Cross-sectional OLS neutralization using polars for high performance.

参数：`(values: Union[pd.DataFrame, pl.DataFrame], exposures: Union[pd.DataFrame, pl.DataFrame], date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True)`。

返回类型：`Union[pd.Series, pl.Series]`。

## factor_preprocess/backends/registry.py

Backend registry for managing computational kernels across backends.

### BackendRegistry

[实际实现](../factor_preprocess/backends/registry.py#L14)。

Registry for backend-specific kernel implementations.

### BackendRegistry.__init__

[实际实现](../factor_preprocess/backends/registry.py#L31)。

Initialize empty registry.

参数：`(self)`。

### BackendRegistry.register

[实际实现](../factor_preprocess/backends/registry.py#L36)。

Register a backend implementation for an operation.

参数：`(self, operation: str, backend: BackendType, implementation: Callable, metadata: Optional[Dict[str, Any]]=None)`。

### BackendRegistry.get

[实际实现](../factor_preprocess/backends/registry.py#L67)。

Get implementation for an operation.

参数：`(self, operation: str, preferred: Optional[BackendType]=None, fallback_order: Optional[list[BackendType]]=None)`。

返回类型：`Optional[Callable]`。

### BackendRegistry.list_operations

[实际实现](../factor_preprocess/backends/registry.py#L119)。

List all registered operations.

参数：`(self)`。

返回类型：`list[str]`。

### BackendRegistry.list_backends

[实际实现](../factor_preprocess/backends/registry.py#L123)。

List available backends for an operation.

参数：`(self, operation: str)`。

返回类型：`list[BackendType]`。

### BackendRegistry.get_metadata

[实际实现](../factor_preprocess/backends/registry.py#L141)。

Get metadata for a specific implementation.

参数：`(self, operation: str, backend: BackendType)`。

返回类型：`Optional[Dict[str, Any]]`。

### register_backend

[实际实现](../factor_preprocess/backends/registry.py#L166)。

Decorator to register a backend implementation.

参数：`(operation: str, backend: BackendType, metadata: Optional[Dict[str, Any]]=None)`。

### get_backend

[实际实现](../factor_preprocess/backends/registry.py#L196)。

Get implementation from global registry.

参数：`(operation: str, preferred: Optional[BackendType]=None)`。

返回类型：`Optional[Callable]`。

### auto_backend

[实际实现](../factor_preprocess/backends/registry.py#L224)。

Decorator that automatically selects backend based on input size.

参数：`(operation: str)`。

## factor_preprocess/backends/selector.py

Backend selector for automatic backend selection based on data size and capabilities.

### BackendType

[实际实现](../factor_preprocess/backends/selector.py#L18)。

Available backend types.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `NUMPY` | `类常量/枚举` | `'numpy'` |
| `NUMBA` | `类常量/枚举` | `'numba'` |
| `CUPY` | `类常量/枚举` | `'cupy'` |
| `POLARS` | `类常量/枚举` | `'polars'` |

### BackendCapabilities

[实际实现](../factor_preprocess/backends/selector.py#L27)。

Backend capability information.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `backend` | `BackendType` | `必填/未声明默认` |
| `available` | `bool` | `必填/未声明默认` |
| `supports_gpu` | `bool` | `False` |
| `supports_jit` | `bool` | `False` |
| `supports_dataframe` | `bool` | `False` |
| `version` | `Optional[str]` | `None` |
| `device_info` | `Optional[str]` | `None` |

### BackendSelector

[实际实现](../factor_preprocess/backends/selector.py#L38)。

Automatic backend selector with data-size-aware heuristics.

### BackendSelector.__init__

[实际实现](../factor_preprocess/backends/selector.py#L61)。

Initialize backend selector.

参数：`(self, numba_threshold: int=50000, cupy_threshold: int=500000, polars_threshold: int=100000, auto_benchmark: bool=False)`。

### BackendSelector.get_capabilities

[实际实现](../factor_preprocess/backends/selector.py#L161)。

Return detected backend capabilities.

参数：`(self)`。

返回类型：`Dict[BackendType, BackendCapabilities]`。

### BackendSelector.select_for_cs_rank

[实际实现](../factor_preprocess/backends/selector.py#L165)。

Select optimal backend for cross-sectional ranking.

参数：`(self, T: int, N: int)`。

返回类型：`BackendType`。

### BackendSelector.select_for_cs_zscore

[实际实现](../factor_preprocess/backends/selector.py#L204)。

Select optimal backend for cross-sectional z-score.

参数：`(self, T: int, N: int)`。

返回类型：`BackendType`。

### BackendSelector.select_for_rolling

[实际实现](../factor_preprocess/backends/selector.py#L243)。

Select optimal backend for rolling operations.

参数：`(self, T: int, N: int, window: int)`。

返回类型：`BackendType`。

### BackendSelector.select_for_neutralization

[实际实现](../factor_preprocess/backends/selector.py#L284)。

Select optimal backend for neutralization.

参数：`(self, T: int, N: int, n_features: int)`。

返回类型：`BackendType`。

### get_backend_capabilities

[实际实现](../factor_preprocess/backends/selector.py#L356)。

Get capabilities of all available backends.

参数：`()`。

返回类型：`Dict[BackendType, BackendCapabilities]`。

### benchmark_backends

[实际实现](../factor_preprocess/backends/selector.py#L374)。

Benchmark available backends for a specific operation.

参数：`(operation: str='rolling', T: int=500, N: int=1000, window: int=20, n_runs: int=3)`。

返回类型：`Dict[str, Dict[str, Any]]`。

## factor_preprocess/contracts/__init__.py

Contracts package.

显式导出（含重导出）：`PreprocessingPolicy`、`TransformSpec`、`TransformKind`、`TransformMode`、`FittedState`、`FeatureBundle`、`AxisRef`、`ChannelRef`、`FeatureManifest`、`OutputProperties`、`derive_output_properties`、`PriceBasis`、`ValueUnit`、`MaterializationSplit`、`TreatmentSpecIdentity`、`TreatmentMaterializationIdentity`、`neutralization_spec_identity`、`neutralization_binding`、`ensure_materialization_split_valid`、`materialize_identity`、`spec_to_materialization_identity`、`ASHARE_INDUSTRY_SCHEMA`、`ASHARE_SIZE_DEFINITION`。

## factor_preprocess/contracts/_deep_freeze.py

Deep immutability helper (package-local; not a shared common/utils).

### as_plain

[实际实现](../factor_preprocess/contracts/_deep_freeze.py#L65)。

Convert a deeply-frozen value back to plain dict/list/set (debugging).

参数：`(value)`。

### deep_freeze

[实际实现](../factor_preprocess/contracts/_deep_freeze.py#L78)。

Public entry point that deep-freezes a container.

参数：`(value)`。

## factor_preprocess/contracts/factor_profile.py

Factor profile artifact contract for the auto-treatment optimizer.

显式导出（含重导出）：`FactorProfileArtifact`。

### FactorProfileArtifact

[实际实现](../factor_preprocess/contracts/factor_profile.py#L53)。

Immutable profile of a factor for treatment eligibility.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_id` | `str` | `必填/未声明默认` |
| `factor_version` | `str` | `必填/未声明默认` |
| `semantic_family` | `str` | `必填/未声明默认` |
| `source_type` | `str` | `必填/未声明默认` |
| `update_frequency` | `str` | `必填/未声明默认` |
| `natural_horizon` | `int` | `必填/未声明默认` |
| `distribution` | `Dict[str, float]` | `field(default_factory=dict)` |
| `time_behavior` | `Dict[str, float]` | `field(default_factory=dict)` |
| `data_behavior` | `Dict[str, float]` | `field(default_factory=dict)` |
| `exposures` | `Dict[str, float]` | `field(default_factory=dict)` |
| `existing_transform_lineage` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `snapshot_ref` | `Optional[str]` | `None` |
| `universe_ref` | `Optional[str]` | `None` |
| `split_ref` | `Optional[str]` | `None` |
| `raw_turnover` | `Optional[float]` | `None` |
| `autocorrelation` | `Optional[float]` | `None` |
| `half_life` | `Optional[float]` | `None` |
| `sparsity` | `Optional[float]` | `None` |
| `missingness` | `Optional[float]` | `None` |
| `freshness` | `Optional[float]` | `None` |
| `outlier_rate` | `Optional[float]` | `None` |
| `cross_section_cardinality` | `Optional[int]` | `None` |
| `sign_stability` | `Optional[float]` | `None` |
| `scale_drift` | `Optional[float]` | `None` |
| `industry_exposure` | `Optional[float]` | `None` |
| `size_exposure` | `Optional[float]` | `None` |
| `tail_concentration` | `Optional[float]` | `None` |
| `event_semantics` | `Optional[str]` | `None` |
| `preexisting_treatments` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `content_hash` | `str` | `''` |

### FactorProfileArtifact.with_lineage

[实际实现](../factor_preprocess/contracts/factor_profile.py#L213)。

Return a copy with a replaced transform lineage (content hash updated).

参数：`(self, lineage: Tuple[str, ...])`。

返回类型：`'FactorProfileArtifact'`。

## factor_preprocess/contracts/feature_bundle.py

Feature bundle contract for model input.

### AxisRef

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L38)。

Reference to a time or asset axis.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `axis_name` | `str` | `必填/未声明默认` |
| `axis_values` | `Any` | `必填/未声明默认` |
| `axis_dtype` | `str` | `必填/未声明默认` |

### ChannelRef

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L66)。

Reference to a feature channel.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `channel_name` | `str` | `必填/未声明默认` |
| `channel_type` | `str` | `必填/未声明默认` |
| `feature_ids` | `Tuple[str, ...]` | `必填/未声明默认` |

### FeatureManifest

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L82)。

Unified column index mapping for multi-channel layouts (FP-P0-02).

### FeatureManifest.__init__

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L96)。

Initialize feature manifest.

参数：`(self, feature_ids: List[str], channel_offsets: Dict[str, int], channel_sizes: Dict[str, int], _allow_aux_channels: bool=False, feature_channels: Optional[List[str]]=None)`。

### FeatureManifest.from_channels

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L260)。

Build a unified manifest from ALL channels (feature + auxiliary).

参数：`(cls, channels: Dict[str, ChannelRef])`。

返回类型：`'FeatureManifest'`。

### FeatureManifest.feature_ids

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L290)。

All feature IDs in order (feature-type channels only).

参数：`(self)`。

返回类型：`Tuple[str, ...]`。

### FeatureManifest.total_features

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L295)。

Number of feature-type features.

参数：`(self)`。

返回类型：`int`。

### FeatureManifest.total_width

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L300)。

Physical feature-axis width (sum of ALL channel sizes).

参数：`(self)`。

返回类型：`int`。

### FeatureManifest.channel_offsets

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L305)。

Starting column index for each channel.

参数：`(self)`。

返回类型：`MappingProxyType[str, int]`。

### FeatureManifest.channel_sizes

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L310)。

Number of columns per channel.

参数：`(self)`。

返回类型：`MappingProxyType[str, int]`。

### FeatureManifest.get_column_index

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L314)。

Get the physical column index for a feature ID.

参数：`(self, feature_id: str)`。

返回类型：`int`。

### FeatureManifest.get_channel_slice

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L337)。

Get slice for extracting a channel's columns.

参数：`(self, channel_name: str)`。

返回类型：`slice`。

### FeatureManifest.get_feature_indices

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L357)。

Get column indices for multiple feature IDs.

参数：`(self, feature_ids: List[str])`。

返回类型：`List[int]`。

### FeatureBundle

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L390)。

Transformed features ready for modeling.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `bundle_id` | `str` | `必填/未声明默认` |
| `time_axis` | `AxisRef` | `必填/未声明默认` |
| `asset_axis` | `AxisRef` | `必填/未声明默认` |
| `channels` | `MappingProxyType` | `必填/未声明默认` |
| `values` | `Any` | `必填/未声明默认` |
| `layout` | `str` | `'NF'` |
| `dtype` | `str` | `'float64'` |
| `source_factor_ids` | `Tuple[str, ...]` | `()` |
| `fitted_state_refs` | `Tuple[str, ...]` | `()` |
| `manifest` | `Optional[FeatureManifest]` | `None` |
| `transform_start_time` | `Optional[datetime]` | `None` |
| `transform_end_time` | `Optional[datetime]` | `None` |
| `has_missing_channel` | `bool` | `False` |
| `has_freshness_channel` | `bool` | `False` |
| `has_exposure_channel` | `bool` | `False` |
| `missing_reason_plane` | `Any` | `None` |
| `allow_duplicate_axis_labels` | `bool` | `False` |
| `policy_id` | `Optional[str]` | `None` |
| `schema_version` | `str` | `'0.1.0'` |
| `producer` | `str` | `'factor_preprocess'` |
| `producer_version` | `str` | `'0.1.0'` |
| `created_at` | `Optional[datetime]` | `None` |
| `config_hash` | `Optional[str]` | `None` |

### FeatureBundle.from_primary_with_auxiliary

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L576)。

Build a bundle without allowing auxiliary outputs to replace signal.

参数：`(cls, *, bundle_id: str, primary_values: np.ndarray, feature_ids: Tuple[str, ...], time_axis: AxisRef, asset_axis: AxisRef, original_validity_mask: np.ndarray, missing_reason_plane: Any=None, freshness_values: Optional[np.ndarray]=None, layout: str='TNF', **metadata)`。

返回类型：`'FeatureBundle'`。

### FeatureBundle.get_missing_reason_plane

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L651)。

Return the immutable DA reason plane; never infer one from NaN.

参数：`(self)`。

返回类型：`Any`。

### FeatureBundle.get_primary_values

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L657)。

Return the canonical signal channel; auxiliary channels never qualify.

参数：`(self)`。

返回类型：`np.ndarray`。

### FeatureBundle.get_original_validity_mask

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L669)。

Recover the original, pre-imputation validity mask.

参数：`(self)`。

返回类型：`np.ndarray`。

### FeatureBundle.get_channel

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L753)。

Retrieve a specific channel by name.

参数：`(self, channel_name: str)`。

返回类型：`Optional[ChannelRef]`。

### FeatureBundle.has_channel

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L757)。

Check if bundle has a specific channel.

参数：`(self, channel_name: str)`。

返回类型：`bool`。

### FeatureBundle.get_channel_values

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L761)。

Extract values for a specific channel using manifest.

参数：`(self, channel_name: str)`。

返回类型：`np.ndarray`。

### FeatureBundle.get_feature_values

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L791)。

Extract values for a specific feature using manifest.

参数：`(self, feature_id: str)`。

返回类型：`np.ndarray`。

### FeatureBundle.is_immutable

[实际实现](../factor_preprocess/contracts/feature_bundle.py#L817)。

Verify bundle immutability.

参数：`(self)`。

返回类型：`bool`。

## factor_preprocess/contracts/fit_apply.py

Fit-apply governance contract for treatments (R61-FI-029, plan §29).

显式导出（含重导出）：`CausalityClass`、`FitScope`、`ApplicationSplit`、`TreatmentFitApplyDeclaration`、`assert_prefix_invariant`、`assert_fit_state_context_match`、`assert_label_access_legal`。

### CausalityClass

[实际实现](../factor_preprocess/contracts/fit_apply.py#L60)。

Information-consumption class of a treatment step (plan §29).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CROSS_SECTIONAL_CAUSAL` | `类常量/枚举` | `'cross_sectional_causal'` |
| `ONE_SIDED_CAUSAL` | `类常量/枚举` | `'one_sided_causal'` |
| `TRAIN_FITTED` | `类常量/枚举` | `'train_fitted'` |
| `OFFLINE_ONLY` | `类常量/枚举` | `'offline_only'` |
| `NO_OP` | `类常量/枚举` | `'no_op'` |

### FitScope

[实际实现](../factor_preprocess/contracts/fit_apply.py#L70)。

The data scope a step is allowed to fit on.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TRAIN_ONLY` | `类常量/枚举` | `'train_only'` |
| `EXPANDING_CAUSAL` | `类常量/枚举` | `'expanding_causal'` |
| `CROSS_SECTIONAL_DATE` | `类常量/枚举` | `'cross_sectional_date'` |
| `FULL_SAMPLE_RESEARCH` | `类常量/枚举` | `'full_sample_research'` |

### FitScope.is_research_only

[实际实现](../factor_preprocess/contracts/fit_apply.py#L79)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`bool`。

### ApplicationSplit

[实际实现](../factor_preprocess/contracts/fit_apply.py#L83)。

The split a step may be applied to.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TRAIN` | `类常量/枚举` | `'train'` |
| `VALIDATION` | `类常量/枚举` | `'validation'` |
| `TEST` | `类常量/枚举` | `'test'` |
| `PRODUCTION` | `类常量/枚举` | `'production'` |
| `FULL_SAMPLE_RESEARCH` | `类常量/枚举` | `'full_sample_research'` |

### ApplicationSplit.is_evaluation_valid

[实际实现](../factor_preprocess/contracts/fit_apply.py#L93)。

Splits that feed evaluation/production decisions.

参数：`(self)`。

返回类型：`bool`。

### TreatmentFitApplyDeclaration

[实际实现](../factor_preprocess/contracts/fit_apply.py#L103)。

Declared fit-apply contract of one treatment step (plan §29).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `treatment_id` | `str` | `必填/未声明默认` |
| `causality_class` | `CausalityClass` | `必填/未声明默认` |
| `fit_scope` | `FitScope` | `必填/未声明默认` |
| `fit_split` | `ApplicationSplit` | `必填/未声明默认` |
| `state_identity` | `str` | `必填/未声明默认` |
| `application_scope` | `ApplicationSplit` | `必填/未声明默认` |

### TreatmentFitApplyDeclaration.validate_self_consistent

[实际实现](../factor_preprocess/contracts/fit_apply.py#L147)。

Fail closed on internally contradictory declarations.

参数：`(self)`。

返回类型：`None`。

### TreatmentFitApplyDeclaration.assert_no_split_leakage

[实际实现](../factor_preprocess/contracts/fit_apply.py#L172)。

Reject ANY split leakage: only TRAIN-fitted steps are evaluable.

参数：`(self, *, application_split=None)`。

返回类型：`None`。

### TreatmentFitApplyDeclaration.assert_asof_legal

[实际实现](../factor_preprocess/contracts/fit_apply.py#L225)。

Reject an apply whose as-of date precedes the fit window end.

参数：`(self, asof_timestamp, fit_end_timestamp)`。

返回类型：`None`。

### assert_prefix_invariant

[实际实现](../factor_preprocess/contracts/fit_apply.py#L240)。

Reject FUTURE APPEND leakage: recompute and compare on a prefix.

参数：`(compute_fn, values, *, split_index, atol: float=1e-12)`。

返回类型：`None`。

### assert_fit_state_context_match

[实际实现](../factor_preprocess/contracts/fit_apply.py#L281)。

Reject FIT-STATE REUSE across a wrong snapshot or universe.

参数：`(*, state_identity: str, state_data_snapshot_ref: str, apply_data_snapshot_ref: str, state_universe_ref: str, apply_universe_ref: str, state_feature_order, apply_feature_order)`。

返回类型：`None`。

### assert_label_access_legal

[实际实现](../factor_preprocess/contracts/fit_apply.py#L318)。

Reject WRONG LABEL ACCESS: a label must be known at the as-of time.

参数：`(*, label_knowledge_time, asof_timestamp, label_name: str='label')`。

返回类型：`None`。

## factor_preprocess/contracts/lineage_policy.py

Lineage deduplication & canonicalization policy (R61-FI-043, plan §26 F4).

显式导出（含重导出）：`LINEAGE_POLICY_VERSION`、`IDEMPOTENT_SEMANTIC_CLASSES`、`NEUTRALIZATION_TOPOLOGY`、`CANONICAL_DUAL_SID`、`RedundancyClass`、`LineagePolicyError`、`RedundantTransformError`、`UnsupportedPolicyVersionError`、`LineagePolicyDecision`、`canonicalize_lineage`、`is_losslessly_collapsible`。

### RedundancyClass

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L120)。

Outcome classification of the guard (audit / tests).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `OK` | `类常量/枚举` | `'ok'` |
| `COLLAPSED` | `类常量/枚举` | `'collapsed'` |
| `REJECTED` | `类常量/枚举` | `'rejected'` |

### LineagePolicyError

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L128)。

A lineage violates the dedup / canonicalization policy.

基类：`ContractError`。

### RedundantTransformError

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L132)。

A redundant operation has no lossless fold and is rejected.

基类：`LineagePolicyError`。

### UnsupportedPolicyVersionError

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L136)。

The lineage was declared under an unsupported policy table version.

基类：`LineagePolicyError`。

### LineagePolicyDecision

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L141)。

Result of canonicalizing / validating a lineage under the policy.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `lineage` | `TransformLineage` | `必填/未声明默认` |
| `redundancy_class` | `RedundancyClass` | `必填/未声明默认` |
| `rejections` | `Tuple[Tuple[str, str, str], ...]` | `()` |
| `policy_version` | `str` | `LINEAGE_POLICY_VERSION` |

### LineagePolicyDecision.is_valid

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L150)。

True when the lineage passed the guard (collapsed counts as valid).

参数：`(self)`。

返回类型：`bool`。

### LineagePolicyDecision.was_collapsed

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L155)。

True when a redundant / superset chain was folded to fewer steps.

参数：`(self)`。

返回类型：`bool`。

### LineagePolicyDecision.canonical_steps

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L160)。

Ordered canonical steps after folds/rejections (audit convenience).

参数：`(self)`。

### canonicalize_lineage

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L205)。

Validate / canonicalize a lineage under the duplicate-guard policy.

参数：`(lineage, *, policy_version: str=LINEAGE_POLICY_VERSION)`。

返回类型：`LineagePolicyDecision`。

### is_losslessly_collapsible

[实际实现](../factor_preprocess/contracts/lineage_policy.py#L331)。

Pure predicate: can ``step_a`` then ``step_b`` fold to ONE step?

参数：`(step_a, step_b)`。

返回类型：`Tuple[bool, Optional[Dict[str, object]]]`。

## factor_preprocess/contracts/policy.py

Preprocessing policy and transform specification contracts.

### TransformKind

[实际实现](../factor_preprocess/contracts/policy.py#L15)。

Type of transform operation.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CROSS_SECTIONAL` | `类常量/枚举` | `'cross_sectional'` |
| `ROLLING` | `类常量/枚举` | `'rolling'` |
| `NEUTRALIZATION` | `类常量/枚举` | `'neutralization'` |
| `REPRESENTATION` | `类常量/枚举` | `'representation'` |
| `SUPERVISED_FITTED` | `类常量/枚举` | `'supervised_fitted'` |

### TransformMode

[实际实现](../factor_preprocess/contracts/policy.py#L24)。

Whether transform requires fitting.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `STATELESS` | `类常量/枚举` | `'stateless'` |
| `FITTED` | `类常量/枚举` | `'fitted'` |

### TransformSpec

[实际实现](../factor_preprocess/contracts/policy.py#L31)。

Specification for a single transform operation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `kind` | `TransformKind` | `必填/未声明默认` |
| `mode` | `TransformMode` | `必填/未声明默认` |
| `version` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `requires_exposure` | `bool` | `False` |
| `produces_channels` | `List[str]` | `field(default_factory=lambda: ['transformed'])` |
| `config_hash` | `Optional[str]` | `None` |

### PreprocessingPolicy

[实际实现](../factor_preprocess/contracts/policy.py#L61)。

Ordered sequence of transforms to apply.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `policy_id` | `str` | `必填/未声明默认` |
| `transforms` | `List[TransformSpec]` | `必填/未声明默认` |
| `version` | `str` | `'0.1.0'` |
| `requires_universe` | `bool` | `False` |
| `requires_industry` | `bool` | `False` |
| `requires_size` | `bool` | `False` |
| `output_channels` | `List[str]` | `field(default_factory=lambda: ['features'])` |
| `missing_indicator` | `bool` | `True` |
| `freshness_channel` | `bool` | `False` |

### PreprocessingPolicy.has_fitted_transforms

[实际实现](../factor_preprocess/contracts/policy.py#L92)。

Check if any transform requires fitting.

参数：`(self)`。

返回类型：`bool`。

## factor_preprocess/contracts/state.py

Fitted transform state contract.

### StateKind

[实际实现](../factor_preprocess/contracts/state.py#L20)。

Whether a transform requires fitted state.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `STATELESS` | `类常量/枚举` | `'stateless'` |
| `FITTED` | `类常量/枚举` | `'fitted'` |

### FittedState

[实际实现](../factor_preprocess/contracts/state.py#L114)。

Immutable state from fitting a transform.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `transform_name` | `str` | `必填/未声明默认` |
| `transform_version` | `str` | `必填/未声明默认` |
| `fit_start_time` | `datetime` | `必填/未声明默认` |
| `fit_end_time` | `datetime` | `必填/未声明默认` |
| `state_id` | `str` | `''` |
| `state_kind` | `StateKind` | `StateKind.STATELESS` |
| `fit_universe_ref` | `Optional[str]` | `None` |
| `feature_ids` | `List[str]` | `field(default_factory=list)` |
| `feature_order` | `List[str]` | `field(default_factory=list)` |
| `learned_params` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `learned_params_hash` | `Optional[str]` | `None` |
| `implementation_hash` | `Optional[str]` | `None` |
| `data_snapshot_ref` | `Optional[str]` | `None` |
| `split_ref` | `Optional[str]` | `None` |
| `universe_ref` | `Optional[str]` | `None` |
| `calendar_ref` | `Optional[str]` | `None` |
| `fit_coordinate_hash` | `Optional[str]` | `None` |
| `policy_hash` | `Optional[str]` | `None` |
| `config_hash` | `Optional[str]` | `None` |
| `created_at` | `Optional[datetime]` | `None` |
| `producer` | `str` | `'factor_preprocess'` |
| `producer_version` | `str` | `'0.1.0'` |
| `production` | `bool` | `False` |

### FittedState.is_compatible_with

[实际实现](../factor_preprocess/contracts/state.py#L281)。

Check if factors match the positional fitted feature contract.

参数：`(self, factor_ids: List[str])`。

返回类型：`bool`。

## factor_preprocess/contracts/treatment_lineage.py

Treatment lineage contracts for the auto-treatment optimizer.

显式导出（含重导出）：`TransformStage`、`TransformSemanticID`、`TransformStep`、`TransformLineage`、`ExistingTreatmentSignature`、`ExistingTreatmentStatus`、`OutputProperties`、`derive_output_properties`、`build_signature_from_lineage`、`map_fe_dsl_to_semantic`。

### TransformStage

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L19)。

Semantic stage of a transform within a treatment pipeline.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PRE_NEUTRALIZATION` | `类常量/枚举` | `'pre_neutralization'` |
| `POST_NEUTRALIZATION` | `类常量/枚举` | `'post_neutralization'` |
| `TEMPORAL` | `类常量/枚举` | `'temporal'` |
| `MISSINGNESS` | `类常量/枚举` | `'missingness'` |
| `OUTLIER` | `类常量/枚举` | `'outlier'` |
| `REPRESENTATION` | `类常量/枚举` | `'representation'` |
| `NEUTRALIZATION` | `类常量/枚举` | `'neutralization'` |
| `SCALING` | `类常量/枚举` | `'scaling'` |

### TransformSemanticID

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L37)。

Canonical semantic identifier of a transform treatment.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `value` | `str` | `必填/未声明默认` |

### TransformStep

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L67)。

A single transform step in a treatment lineage.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `semantic_id` | `TransformSemanticID` | `必填/未声明默认` |
| `stage` | `TransformStage` | `必填/未声明默认` |
| `name` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `field(default_factory=dict)` |

### TransformLineage

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L93)。

Ordered list of transform steps applied to a factor.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `steps` | `Tuple[TransformStep, ...]` | `field(default_factory=tuple)` |

### TransformLineage.semantic_ids

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L107)。

Return the ordered semantic IDs of the lineage.

参数：`(self)`。

返回类型：`List[str]`。

### TransformLineage.has_semantic

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L111)。

True if any step carries the given semantic ID.

参数：`(self, semantic_id: str)`。

返回类型：`bool`。

### TransformLineage.has_stage

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L115)。

True if any step is applied at the given semantic stage.

参数：`(self, stage: TransformStage)`。

返回类型：`bool`。

### TransformLineage.dedupe

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L119)。

Return a lineage with duplicate (semantic_id, stage) pairs removed.

参数：`(self)`。

返回类型：`'TransformLineage'`。

### ExistingTreatmentStatus

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L138)。

Completeness of the detected existing-treatment signature.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `KNOWN` | `类常量/枚举` | `'known'` |
| `UNKNOWN` | `类常量/枚举` | `'unknown'` |
| `INCOMPLETE` | `类常量/枚举` | `'incomplete'` |

### OutputProperties

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L154)。

Properties guaranteed at the current lineage root, not historical tags.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `status` | `ExistingTreatmentStatus` | `ExistingTreatmentStatus.KNOWN` |
| `ranked` | `bool` | `False` |
| `scaling_axis` | `Optional[str]` | `None` |
| `orthogonal_to` | `Tuple[str, ...]` | `()` |
| `mask_ref` | `Optional[str]` | `None` |
| `weight_ref` | `Optional[str]` | `None` |
| `last_semantic_id` | `Optional[str]` | `None` |

### derive_output_properties

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L171)。

Derive conservative root postconditions from the ordered actual steps.

参数：`(lineage)`。

返回类型：`OutputProperties`。

### ExistingTreatmentSignature

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L219)。

Summary of treatments already applied to a factor.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `winsor` | `bool` | `False` |
| `winsor_params` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `industry_neutral` | `bool` | `False` |
| `industry_schema` | `Optional[str]` | `None` |
| `size_neutral` | `bool` | `False` |
| `cs_rank` | `bool` | `False` |
| `temporal_smoothing` | `bool` | `False` |
| `temporal_smoothing_params` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `status` | `ExistingTreatmentStatus` | `ExistingTreatmentStatus.KNOWN` |

### ExistingTreatmentSignature.is_unknown_or_incomplete

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L252)。

True when the signature must not be treated as fully known.

参数：`(self)`。

返回类型：`bool`。

### build_signature_from_lineage

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L264)。

Build an :class:`ExistingTreatmentSignature` from a TransformLineage.

参数：`(lineage, semantic_ids_known=None)`。

返回类型：`ExistingTreatmentSignature`。

### map_fe_dsl_to_semantic

[实际实现](../factor_preprocess/contracts/treatment_lineage.py#L353)。

Map a known FE DSL transform name to a semantic ID.

参数：`(name: str)`。

返回类型：`Optional[str]`。

## factor_preprocess/contracts/treatment_recipe.py

TreatmentRecipe — canonical prescription for applying a treatment to a factor.

显式导出（含重导出）：`TreatmentRecipe`、`RecipeStep`、`FitBoundary`、`RecipeSchemaVersion`。

### FitBoundary

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L50)。

Where the fitted statistics may be drawn from.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `EXPANDING` | `类常量/枚举` | `'expanding'` |
| `ROLLING` | `类常量/枚举` | `'rolling'` |
| `FULL_SAMPLE_RESEARCH` | `类常量/枚举` | `'full_sample_research'` |

### RecipeSchemaVersion

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L58)。

Schema version of the recipe artifact.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `V1` | `类常量/枚举` | `'1.0.0'` |

### RecipeStep

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L98)。

A single ordered transform step within a :class:`TreatmentRecipe`.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `step_id` | `str` | `必填/未声明默认` |
| `semantic_transform_id` | `str` | `必填/未声明默认` |
| `implementation_ref` | `str` | `必填/未声明默认` |
| `stage` | `str` | `必填/未声明默认` |
| `requires_fit` | `bool` | `False` |
| `state_ref` | `Optional[str]` | `None` |
| `parameters` | `Dict[str, Any]` | `field(default_factory=dict)` |

### TreatmentRecipe

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L122)。

Deep-immutable, content-addressed prescription for a treatment.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `recipe_id` | `str` | `必填/未声明默认` |
| `source_factor_definition_ref` | `str` | `必填/未声明默认` |
| `source_factor_value_ref` | `str` | `必填/未声明默认` |
| `ordered_steps` | `tuple` | `必填/未声明默认` |
| `existing_treatment_signature` | `Optional[Dict[str, Any]]` | `None` |
| `neutralization_spec` | `Optional[Dict[str, Any]]` | `None` |
| `fit_boundary` | `FitBoundary` | `FitBoundary.EXPANDING` |
| `registry_snapshot_identity` | `Optional[str]` | `None` |
| `policy_identity` | `Optional[str]` | `None` |
| `causality_certificate_ref` | `Optional[str]` | `None` |
| `schema_version` | `RecipeSchemaVersion` | `RecipeSchemaVersion.V1` |
| `content_hash` | `str` | `''` |

### TreatmentRecipe.treatment_identity

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L238)。

Content-derived identity over the ordered steps + context.

参数：`(self)`。

返回类型：`str`。

### TreatmentRecipe.spec_identity

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L247)。

SPEC identity of this recipe (definition-only, data-independent).

参数：`(self)`。

返回类型：`'TreatmentSpecIdentity'`。

### TreatmentRecipe.is_full_sample_research

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L277)。

True if this recipe fits on the full sample (research-only).

参数：`(self)`。

返回类型：`bool`。

### TreatmentRecipe.materialize

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L287)。

Materialize this recipe against a concrete context.

参数：`(self, *, materialization_ref: str, data_snapshot_ref: str, universe_ref: str, split_ref: str, split, neutralization_spec=None, asof_timestamp=None, price_basis=None, industry_schema=None, size_definition=None, pit_identity=None, unit=None, require_split_valid: bool=True)`。

返回类型：`'TreatmentMaterializationIdentity'`。

### TreatmentRecipe.step_semantic_ids

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L331)。

Ordered semantic transform ids of the recipe steps.

参数：`(self)`。

返回类型：`tuple`。

### TreatmentRecipe.to_canonical_dict

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L335)。

Render a lossless, JSON-compatible recipe representation.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TreatmentRecipe.from_canonical_dict

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L374)。

Parse the canonical representation and revalidate its content hash.

参数：`(cls, payload: Dict[str, Any])`。

返回类型：`'TreatmentRecipe'`。

### TreatmentRecipe.compile

[实际实现](../factor_preprocess/contracts/treatment_recipe.py#L380)。

Resolve every actual step against the registry, failing closed.

参数：`(self, registry, *, fitted_state_refs=())`。

返回类型：`tuple`。

## factor_preprocess/contracts/treatment_spec.py

Treatment identity split — SPEC vs MATERIALIZATION (P0-FP #103).

显式导出（含重导出）：`PriceBasis`、`ValueUnit`、`MaterializationSplit`、`TreatmentSpecIdentity`、`TreatmentMaterializationIdentity`、`neutralization_spec_identity`、`neutralization_binding`、`ensure_materialization_split_valid`、`materialize_identity`、`spec_to_materialization_identity`、`ASHARE_INDUSTRY_SCHEMA`、`ASHARE_SIZE_DEFINITION`。

### PriceBasis

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L60)。

Price basis the treated values are defined on (A股 convention).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `ADJ` | `类常量/枚举` | `'adj'` |
| `UNADJ` | `类常量/枚举` | `'unadj'` |

### ValueUnit

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L67)。

Unit/basis of the materialized values (A股 convention).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `DECIMAL` | `类常量/枚举` | `'decimal'` |
| `BASIS_POINT` | `类常量/枚举` | `'basis_point'` |

### MaterializationSplit

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L74)。

Which split a materialization is valid for (PIT contract).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TRAIN` | `类常量/枚举` | `'train'` |
| `VALIDATION` | `类常量/枚举` | `'validation'` |
| `TEST` | `类常量/枚举` | `'test'` |
| `PRODUCTION` | `类常量/枚举` | `'production'` |
| `FULL_SAMPLE_RESEARCH` | `类常量/枚举` | `'full_sample_research'` |

### MaterializationSplit.is_evaluation_valid

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L89)。

True for splits that feed evaluation/production decisions.

参数：`(self)`。

返回类型：`bool`。

### neutralization_spec_identity

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L160)。

Content identity of a :class:`NeutralizationSpec`.

参数：`(spec: Optional[NeutralizationSpec])`。

返回类型：`Optional[str]`。

### TreatmentSpecIdentity

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L187)。

Content identity of a treatment *definition* (SPEC).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `spec_id` | `str` | `必填/未声明默认` |
| `recipe_ref` | `str` | `必填/未声明默认` |
| `semantic_transform_ids` | `tuple` | `()` |
| `neutralization_spec_identity` | `Optional[str]` | `None` |
| `fit_boundary` | `FitBoundary` | `FitBoundary.EXPANDING` |
| `identity` | `str` | `''` |

### TreatmentSpecIdentity.from_recipe

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L240)。

Derive a spec identity from a :class:`TreatmentRecipe`.

参数：`(cls, recipe: TreatmentRecipe, neutralization_spec: Optional[NeutralizationSpec]=None)`。

返回类型：`'TreatmentSpecIdentity'`。

### TreatmentMaterializationIdentity

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L273)。

Content identity of a treatment *materialization*.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `spec_identity` | `TreatmentSpecIdentity` | `必填/未声明默认` |
| `materialization_ref` | `str` | `必填/未声明默认` |
| `data_snapshot_ref` | `str` | `必填/未声明默认` |
| `universe_ref` | `str` | `必填/未声明默认` |
| `split_ref` | `str` | `必填/未声明默认` |
| `split` | `MaterializationSplit` | `MaterializationSplit.FULL_SAMPLE_RESEARCH` |
| `asof_timestamp` | `Optional[datetime]` | `None` |
| `price_basis` | `PriceBasis` | `PriceBasis.ADJ` |
| `industry_schema` | `Optional[str]` | `None` |
| `size_definition` | `Optional[str]` | `None` |
| `pit_identity` | `PitIdentity` | `PitIdentity.PIT` |
| `unit` | `ValueUnit` | `ValueUnit.DECIMAL` |
| `neutralization_spec_identity` | `Optional[str]` | `None` |
| `identity` | `str` | `''` |

### neutralization_binding

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L380)。

A股 (A-share) neutralization binding surface.

参数：`(spec: Optional[NeutralizationSpec], *, industry_schema: Optional[str]=None, size_definition: Optional[str]=None)`。

返回类型：`Dict[str, Any]`。

### ensure_materialization_split_valid

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L412)。

Runtime fail-closed check applied where treatments are materialized.

参数：`(recipe: TreatmentRecipe, split: MaterializationSplit, *, neutralization_spec: Optional[NeutralizationSpec]=None)`。

返回类型：`None`。

### materialize_identity

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L452)。

Materialize a recipe into a :class:`TreatmentMaterializationIdentity`.

参数：`(recipe: TreatmentRecipe, *, materialization_ref: str, data_snapshot_ref: str, universe_ref: str, split_ref: str, split: MaterializationSplit, neutralization_spec: Optional[NeutralizationSpec]=None, asof_timestamp: Optional[datetime]=None, price_basis: PriceBasis=PriceBasis.ADJ, industry_schema: Optional[str]=None, size_definition: Optional[str]=None, pit_identity: Optional[PitIdentity]=None, unit: ValueUnit=ValueUnit.DECIMAL, require_split_valid: bool=True)`。

返回类型：`TreatmentMaterializationIdentity`。

### spec_to_materialization_identity

[实际实现](../factor_preprocess/contracts/treatment_spec.py#L531)。

Deprecated alias kept for callers that used the previous naming.

参数：`(recipe: TreatmentRecipe, neutralization_spec: Optional[NeutralizationSpec], *, materialization_ref: str, data_snapshot_ref: str, universe_ref: str, split_ref: str, split: MaterializationSplit, asof_timestamp: Optional[datetime]=None, price_basis: PriceBasis=PriceBasis.ADJ, industry_schema: Optional[str]=None, size_definition: Optional[str]=None, unit: ValueUnit=ValueUnit.DECIMAL)`。

返回类型：`TreatmentMaterializationIdentity`。

## factor_preprocess/eligibility/__init__.py

Eligibility package for the auto-treatment optimizer.

显式导出（含重导出）：`TreatmentEligibilityEngine`、`TreatmentSearchSpace`、`PRICE_VOLUME`、`HIGH_TURNOVER`、`FUNDAMENTAL`、`SPARSE_UPDATE`、`EVENT`、`BINARY`、`DISCRETE`、`RAW_SEMANTIC_ID`、`TreatmentFamily`、`FactorFamily`、`FamilyRule`、`EligibilityRuleRegistry`、`create_default_eligibility_rules`、`get_default_eligibility_rules`。

## factor_preprocess/eligibility/engine.py

Treatment eligibility engine for the auto-treatment optimizer.

显式导出（含重导出）：`TreatmentEligibilityEngine`、`TreatmentSearchSpace`、`PRICE_VOLUME`、`HIGH_TURNOVER`、`FUNDAMENTAL`、`SPARSE_UPDATE`、`EVENT`、`BINARY`、`DISCRETE`、`RAW_SEMANTIC_ID`。

### TreatmentSearchSpace

[实际实现](../factor_preprocess/eligibility/engine.py#L100)。

The set of legal transforms (with parameter ranges) for a factor.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_id` | `str` | `必填/未声明默认` |
| `allowed_transform_ids` | `Dict[str, Dict[str, Tuple[float, float]]]` | `field(default_factory=dict)` |
| `notes` | `Tuple[str, ...]` | `field(default_factory=tuple)` |

### TreatmentSearchSpace.allows

[实际实现](../factor_preprocess/eligibility/engine.py#L122)。

True if the given transform name is in the search space.

参数：`(self, transform_name: str)`。

返回类型：`bool`。

### TreatmentSearchSpace.transform_names

[实际实现](../factor_preprocess/eligibility/engine.py#L126)。

Return the allowed transform names (RAW always included).

参数：`(self)`。

返回类型：`List[str]`。

### TreatmentEligibilityEngine

[实际实现](../factor_preprocess/eligibility/engine.py#L131)。

Rule-based phase-1 eligibility engine (single-transform-authority).

### TreatmentEligibilityEngine.__init__

[实际实现](../factor_preprocess/eligibility/engine.py#L134)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, rules=None, registry=None)`。

### TreatmentEligibilityEngine.build_search_space

[实际实现](../factor_preprocess/eligibility/engine.py#L139)。

Build the legal search space for a factor profile.

参数：`(self, profile: FactorProfileArtifact, existing: Optional[ExistingTreatmentSignature]=None)`。

返回类型：`TreatmentSearchSpace`。

## factor_preprocess/eligibility/rules.py

EligibilityRuleRegistry — the *single* authority for which semantic treatment families a factor type may search (DLIB-FP-014).

显式导出（含重导出）：`TreatmentFamily`、`FactorFamily`、`FamilyRule`、`EligibilityRuleRegistry`、`create_default_eligibility_rules`、`get_default_eligibility_rules`、`RAW_SEMANTIC_ID`。

### TreatmentFamily

[实际实现](../factor_preprocess/eligibility/rules.py#L21)。

Semantic treatment families a factor type may search.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `SMOOTHING` | `类常量/枚举` | `'smoothing'` |
| `EVENT_DECAY` | `类常量/枚举` | `'event_decay'` |
| `FRESHNESS_FILL` | `类常量/枚举` | `'freshness_fill'` |
| `WINSOR` | `类常量/枚举` | `'winsor'` |
| `RANK` | `类常量/枚举` | `'rank'` |
| `ZSCORE` | `类常量/枚举` | `'zscore'` |
| `NEUTRALIZATION` | `类常量/枚举` | `'neutralization'` |

### FactorFamily

[实际实现](../factor_preprocess/eligibility/rules.py#L33)。

Known factor semantic families.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PRICE_VOLUME` | `类常量/枚举` | `'PRICE_VOLUME'` |
| `HIGH_TURNOVER` | `类常量/枚举` | `'HIGH_TURNOVER'` |
| `FUNDAMENTAL` | `类常量/枚举` | `'FUNDAMENTAL'` |
| `SPARSE_UPDATE` | `类常量/枚举` | `'SPARSE_UPDATE'` |
| `EVENT` | `类常量/枚举` | `'EVENT'` |
| `BINARY` | `类常量/枚举` | `'BINARY'` |
| `DISCRETE` | `类常量/枚举` | `'DISCRETE'` |

### FamilyRule

[实际实现](../factor_preprocess/eligibility/rules.py#L50)。

A single family -> legal treatment families rule (versioned).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_family` | `str` | `必填/未声明默认` |
| `allowed_families` | `Tuple[TreatmentFamily, ...]` | `必填/未声明默认` |
| `policy_version` | `str` | `'1.0.0'` |
| `notes` | `Tuple[str, ...]` | `field(default_factory=tuple)` |

### FamilyRule.allows

[实际实现](../factor_preprocess/eligibility/rules.py#L63)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: TreatmentFamily)`。

返回类型：`bool`。

### EligibilityRuleRegistry

[实际实现](../factor_preprocess/eligibility/rules.py#L67)。

Versioned catalog of factor-family -> allowed treatment families.

### EligibilityRuleRegistry.__init__

[实际实现](../factor_preprocess/eligibility/rules.py#L70)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### EligibilityRuleRegistry.register

[实际实现](../factor_preprocess/eligibility/rules.py#L74)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, rule: FamilyRule)`。

返回类型：`None`。

### EligibilityRuleRegistry.set_default

[实际实现](../factor_preprocess/eligibility/rules.py#L78)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: str, allowed: Tuple[TreatmentFamily, ...])`。

返回类型：`None`。

### EligibilityRuleRegistry.rule_for

[实际实现](../factor_preprocess/eligibility/rules.py#L81)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: str)`。

返回类型：`Optional[FamilyRule]`。

### EligibilityRuleRegistry.allowed_families

[实际实现](../factor_preprocess/eligibility/rules.py#L84)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: str)`。

返回类型：`Tuple[TreatmentFamily, ...]`。

### EligibilityRuleRegistry.policy_version_for

[实际实现](../factor_preprocess/eligibility/rules.py#L90)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: str)`。

返回类型：`Optional[str]`。

### create_default_eligibility_rules

[实际实现](../factor_preprocess/eligibility/rules.py#L95)。

Create the canonical family-rule registry (DLIB-FP-023).

参数：`()`。

返回类型：`EligibilityRuleRegistry`。

### get_default_eligibility_rules

[实际实现](../factor_preprocess/eligibility/rules.py#L196)。

Get or create the default family-rule registry.

参数：`()`。

返回类型：`EligibilityRuleRegistry`。

## factor_preprocess/errors.py

Core error taxonomy for factor_preprocess.

显式导出（含重导出）：`FactorPreprocessError`、`ContractError`、`SchemaVersionError`、`MissingInputError`、`InvalidContractError`、`TimingContractError`、`SnapshotMismatchError`、`TreatmentMaterializationError`、`CapabilityError`、`UnsupportedTransformError`、`OptionalDependencyMissing`、`DataError`、`InsufficientObservations`、`InvalidValidityMask`、`MissingFittedStateError`、`StaleFittedStateError`、`UnknownRegimeError`、`SupervisedTargetRequiredError`、`UnsupportedTargetError`、`ExecutionError`、`NumericalFailure`、`OverflowOrNonFiniteError`、`BudgetExceededError`、`CancellationError`、`GovernanceError`、`FullSampleFitError`、`FittedStateMismatchError`、`ContractChangeRequired`。

### FactorPreprocessError

[实际实现](../factor_preprocess/errors.py#L11)。

Base exception for factor_preprocess.

基类：`Exception`。

### ContractError

[实际实现](../factor_preprocess/errors.py#L21)。

Data contract or schema violation.

基类：`FactorPreprocessError`。

### SchemaVersionError

[实际实现](../factor_preprocess/errors.py#L26)。

Schema version mismatch or unsupported version.

基类：`ContractError`。

### MissingInputError

[实际实现](../factor_preprocess/errors.py#L31)。

Required input field is missing.

基类：`ContractError`。

### InvalidContractError

[实际实现](../factor_preprocess/errors.py#L36)。

Input violates a contract constraint.

基类：`ContractError`。

### TimingContractError

[实际实现](../factor_preprocess/errors.py#L41)。

Timing/ordering constraint violated.

基类：`ContractError`。

### SnapshotMismatchError

[实际实现](../factor_preprocess/errors.py#L46)。

Snapshot or context reference mismatch.

基类：`ContractError`。

### TreatmentMaterializationError

[实际实现](../factor_preprocess/errors.py#L51)。

A treatment materialization violates the fail-closed contract.

基类：`ContractError`。

### CapabilityError

[实际实现](../factor_preprocess/errors.py#L74)。

Requested capability is not available.

基类：`FactorPreprocessError`。

### UnsupportedTransformError

[实际实现](../factor_preprocess/errors.py#L79)。

Requested transform is not implemented.

基类：`CapabilityError`。

### OptionalDependencyMissing

[实际实现](../factor_preprocess/errors.py#L84)。

Optional dependency (FA, DA, etc.) is not installed.

基类：`CapabilityError`。

### OptionalDependencyMissing.__init__

[实际实现](../factor_preprocess/errors.py#L87)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, package_name: str, feature_name: str)`。

### DataError

[实际实现](../factor_preprocess/errors.py#L101)。

Data or evidence quality issue.

基类：`FactorPreprocessError`。

### InsufficientObservations

[实际实现](../factor_preprocess/errors.py#L106)。

Not enough valid observations for fitting or transform.

基类：`DataError`。

### InvalidValidityMask

[实际实现](../factor_preprocess/errors.py#L111)。

Validity mask is malformed or contradictory.

基类：`DataError`。

### MissingFittedStateError

[实际实现](../factor_preprocess/errors.py#L116)。

Required fitted state is missing or unavailable.

基类：`DataError`。

### StaleFittedStateError

[实际实现](../factor_preprocess/errors.py#L121)。

Fitted state is outdated or bound to wrong fit window.

基类：`DataError`。

### UnknownRegimeError

[实际实现](../factor_preprocess/errors.py#L126)。

Encountered a regime label that was not seen at fit time.

基类：`DataError`。

### SupervisedTargetRequiredError

[实际实现](../factor_preprocess/errors.py#L131)。

A supervised transform requires a target/label series.

基类：`DataError`。

### UnsupportedTargetError

[实际实现](../factor_preprocess/errors.py#L136)。

A label-free transform was given a target it does not accept.

基类：`DataError`。

### ExecutionError

[实际实现](../factor_preprocess/errors.py#L146)。

Execution or computation failed.

基类：`FactorPreprocessError`。

### NumericalFailure

[实际实现](../factor_preprocess/errors.py#L151)。

Numerical computation failed (overflow, NaN, Inf, etc.).

基类：`ExecutionError`。

### OverflowOrNonFiniteError

[实际实现](../factor_preprocess/errors.py#L156)。

Result is infinite, NaN, or overflowed.

基类：`NumericalFailure`。

### BudgetExceededError

[实际实现](../factor_preprocess/errors.py#L161)。

Computational budget or resource limit exceeded.

基类：`ExecutionError`。

### CancellationError

[实际实现](../factor_preprocess/errors.py#L166)。

Operation was cancelled.

基类：`ExecutionError`。

### GovernanceError

[实际实现](../factor_preprocess/errors.py#L176)。

Governance policy or constraint violation.

基类：`FactorPreprocessError`。

### FullSampleFitError

[实际实现](../factor_preprocess/errors.py#L181)。

Full-sample fitting before split is forbidden.

基类：`GovernanceError`。

### FittedStateMismatchError

[实际实现](../factor_preprocess/errors.py#L186)。

Fitted state does not match expected universe or features.

基类：`GovernanceError`。

### ContractChangeRequired

[实际实现](../factor_preprocess/errors.py#L191)。

Operation requires contract schema change.

基类：`GovernanceError`。

## factor_preprocess/grammar/__init__.py

Search grammar package for the auto-treatment optimizer.

显式导出（含重导出）：`Stage`、`STAGE_ORDER`、`CausalityClass`、`OrderTemplate`、`CERTIFIED_TEMPLATES`、`is_certified_template`、`validate_stage_order`。

## factor_preprocess/grammar/search_grammar.py

Search grammar for the auto-treatment optimizer.

显式导出（含重导出）：`Stage`、`STAGE_ORDER`、`STAGE_META_TO_CANONICAL`、`CausalityClass`、`OrderTemplate`、`CERTIFIED_TEMPLATES`、`is_certified_template`、`is_certified_exception`、`validate_stage_order`、`stage_of_transform_name`、`certify_preset_stages`、`certify_production_presets`、`CERTIFIED_PRODUCTION_RECIPES`。

### Stage

[实际实现](../factor_preprocess/grammar/search_grammar.py#L22)。

Canonical treatment stages in order.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `MISSINGNESS` | `类常量/枚举` | `'A_Missingness'` |
| `OUTLIER` | `类常量/枚举` | `'B_Outlier'` |
| `TEMPORAL` | `类常量/枚举` | `'C_TemporalStabilization'` |
| `NEUTRALIZATION` | `类常量/枚举` | `'D_Neutralization'` |
| `REPRESENTATION` | `类常量/枚举` | `'E_RepresentationScaling'` |

### stage_of_transform_name

[实际实现](../factor_preprocess/grammar/search_grammar.py#L55)。

Map a transform name to its canonical Stage via registry metadata.

参数：`(name: str, *, registry=None)`。

返回类型：`'Stage'`。

### CausalityClass

[实际实现](../factor_preprocess/grammar/search_grammar.py#L93)。

Causality class of a template.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CAUSAL` | `类常量/枚举` | `'causal'` |
| `CROSS_SECTIONAL` | `类常量/枚举` | `'cross_sectional'` |
| `NO_OP` | `类常量/枚举` | `'no_op'` |

### OrderTemplate

[实际实现](../factor_preprocess/grammar/search_grammar.py#L102)。

A certified legal stage ordering.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `stages` | `Tuple[Stage, ...]` | `必填/未声明默认` |
| `preconditions` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `causality_class` | `CausalityClass` | `CausalityClass.CAUSAL` |

### OrderTemplate.stage_names

[实际实现](../factor_preprocess/grammar/search_grammar.py#L119)。

Return the ordered stage names.

参数：`(self)`。

返回类型：`List[str]`。

### is_certified_template

[实际实现](../factor_preprocess/grammar/search_grammar.py#L174)。

True if the given stage sequence matches a certified template.

参数：`(stages: Tuple[Stage, ...])`。

返回类型：`bool`。

### is_certified_exception

[实际实现](../factor_preprocess/grammar/search_grammar.py#L182)。

True when a preset carries an explicit certified-exception name.

参数：`(name: str)`。

返回类型：`bool`。

### validate_stage_order

[实际实现](../factor_preprocess/grammar/search_grammar.py#L193)。

Validate that a stage sequence respects the canonical stage order.

参数：`(stages: Tuple[Stage, ...])`。

返回类型：`Tuple[bool, List[str]]`。

### certify_preset_stages

[实际实现](../factor_preprocess/grammar/search_grammar.py#L225)。

Certify an ordered preset transform list against the stage grammar.

参数：`(preset_name: str, transform_names: Tuple[str, ...], *, registry=None, certified_exceptions=frozenset({'SMOOTH_NEUTRALIZE_RANK', 'NEUTRALIZE_SMOOTH_RANK', 'WINSOR_NEUTRALIZE_ZSCORE', 'RANK_THEN_NEUTRALIZE', 'RAW'}))`。

返回类型：`Tuple[bool, List[str]]`。

### certify_production_presets

[实际实现](../factor_preprocess/grammar/search_grammar.py#L288)。

Certify every production preset of a policy registry.

参数：`(policy_registry=None, transform_registry=None)`。

## factor_preprocess/kernels/__init__.py

Kernels package - reference bridge for fast implementations.

显式导出（含重导出）：`reference_cs_rank`、`reference_cs_zscore`、`reference_cs_demean`、`reference_rolling_mean`、`reference_rolling_std`、`fast_cs_rank`、`fast_cs_zscore`、`fast_cs_demean`、`fast_rolling_mean`、`fast_rolling_std`、`numba_rolling_mean`、`numba_rolling_std`、`get_capabilities`。

## factor_preprocess/kernels/fast.py

Fast kernel implementations for factor preprocessing.

### fast_cs_rank

[实际实现](../factor_preprocess/kernels/fast.py#L41)。

Fast cross-sectional rank using bottleneck.

参数：`(values: np.ndarray, axis: int=-1, method: Literal['average', 'min', 'max', 'dense', 'ordinal']='average', pct: bool=False)`。

返回类型：`np.ndarray`。

### fast_cs_zscore

[实际实现](../factor_preprocess/kernels/fast.py#L107)。

Fast cross-sectional z-score normalization.

参数：`(values: np.ndarray, axis: int=-1, ddof: int=1, constant_value: float=0.0)`。

返回类型：`np.ndarray`。

### fast_cs_demean

[实际实现](../factor_preprocess/kernels/fast.py#L154)。

Fast cross-sectional demean.

参数：`(values: np.ndarray, axis: int=-1)`。

返回类型：`np.ndarray`。

### fast_rolling_mean

[实际实现](../factor_preprocess/kernels/fast.py#L234)。

Fast rolling mean using stride tricks.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### fast_rolling_std

[实际实现](../factor_preprocess/kernels/fast.py#L284)。

Fast rolling standard deviation using stride tricks.

参数：`(values: np.ndarray, window: int, axis: int=0, ddof: int=1)`。

返回类型：`np.ndarray`。

### numba_rolling_mean

[实际实现](../factor_preprocess/kernels/fast.py#L429)。

Numba-accelerated rolling mean.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### numba_rolling_std

[实际实现](../factor_preprocess/kernels/fast.py#L476)。

Numba-accelerated rolling std.

参数：`(values: np.ndarray, window: int, axis: int=0, ddof: int=1)`。

返回类型：`np.ndarray`。

### get_capabilities

[实际实现](../factor_preprocess/kernels/fast.py#L529)。

Return available fast kernel capabilities.

参数：`()`。

返回类型：`dict`。

## factor_preprocess/kernels/numba_transforms.py

Numba-accelerated transforms for factor preprocessing.

### numba_rolling_mean

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L474)。

Numba-accelerated rolling mean with dimension handling.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### numba_rolling_std

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L500)。

Numba-accelerated rolling std with automatic dimension handling.

参数：`(values: np.ndarray, window: int, axis: int=0, ddof: int=1)`。

返回类型：`np.ndarray`。

### numba_rolling_sum

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L550)。

Numba-accelerated rolling sum.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### numba_rolling_min

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L563)。

Numba-accelerated rolling minimum.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### numba_rolling_max

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L576)。

Numba-accelerated rolling maximum.

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### numba_cs_rank

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L589)。

Numba-accelerated cross-sectional rank.

参数：`(values: np.ndarray, axis: int=-1, pct: bool=False)`。

返回类型：`np.ndarray`。

### numba_cs_zscore

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L620)。

Numba-accelerated cross-sectional z-score.

参数：`(values: np.ndarray, axis: int=-1, ddof: int=1, constant_value: float=0.0)`。

返回类型：`np.ndarray`。

### numba_cs_winsorize

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L658)。

Numba-accelerated cross-sectional winsorization.

参数：`(values: np.ndarray, lower: float=0.05, upper: float=0.95, axis: int=-1)`。

返回类型：`np.ndarray`。

### has_numba

[实际实现](../factor_preprocess/kernels/numba_transforms.py#L696)。

Check if Numba is available.

参数：`()`。

返回类型：`bool`。

## factor_preprocess/kernels/reference_bridge.py

Reference implementation bridge for parity checking.

### reference_cs_rank

[实际实现](../factor_preprocess/kernels/reference_bridge.py#L12)。

Reference cross-sectional rank implementation.

参数：`(values: np.ndarray, axis: int=-1, method: Literal['average', 'min', 'max', 'dense', 'ordinal']='average', pct: bool=False)`。

返回类型：`np.ndarray`。

### reference_cs_zscore

[实际实现](../factor_preprocess/kernels/reference_bridge.py#L65)。

Reference cross-sectional z-score implementation.

参数：`(values: np.ndarray, axis: int=-1, ddof: int=1, constant_value: float=0.0)`。

返回类型：`np.ndarray`。

### reference_cs_demean

[实际实现](../factor_preprocess/kernels/reference_bridge.py#L107)。

Reference cross-sectional demean implementation.

参数：`(values: np.ndarray, axis: int=-1)`。

返回类型：`np.ndarray`。

### reference_rolling_mean

[实际实现](../factor_preprocess/kernels/reference_bridge.py#L133)。

Reference rolling mean (simple stride-based implementation).

参数：`(values: np.ndarray, window: int, axis: int=0)`。

返回类型：`np.ndarray`。

### reference_rolling_std

[实际实现](../factor_preprocess/kernels/reference_bridge.py#L176)。

Reference rolling standard deviation.

参数：`(values: np.ndarray, window: int, axis: int=0, ddof: int=1)`。

返回类型：`np.ndarray`。

## factor_preprocess/neutralization/__init__.py

Neutralization package.

显式导出（含重导出）：`ols_neutralize`、`compute_exposures`、`ridge_neutralize`、`lasso_neutralize`、`elastic_net_neutralize`、`compute_condition_number`、`compute_exposure_correlation`、`check_residual_exposures`、`diagnose_neutralization`、`compute_variance_reduction`、`flag_ill_conditioned_dates`、`summarize_diagnostics`、`pca_neutralize`、`huber_neutralize`、`lad_neutralize`、`quantile_neutralize`、`kernel_neutralize`。

## factor_preprocess/neutralization/advanced/__init__.py

Advanced neutralization methods.

显式导出（含重导出）：`pca_neutralize`、`huber_neutralize`、`lad_neutralize`、`quantile_neutralize`、`kernel_neutralize`。

## factor_preprocess/neutralization/advanced/kernel_regression.py

Kernel-based non-parametric neutralization for cross-sectional residuals.

### kernel_neutralize

[实际实现](../factor_preprocess/neutralization/advanced/kernel_regression.py#L32)。

Cross-sectional kernel regression neutralization.

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, bandwidth: Optional[float]=None, kernel: Literal['gaussian', 'epanechnikov', 'tricube']='gaussian', date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, local_constant: bool=True, normalize: bool=True)`。

返回类型：`pd.Series`。

## factor_preprocess/neutralization/advanced/pca_neutralization.py

PCA-based neutralization for cross-sectional residuals.

### pca_neutralize

[实际实现](../factor_preprocess/neutralization/advanced/pca_neutralization.py#L13)。

Cross-sectional PCA neutralization.

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, n_components: Optional[int]=None, variance_threshold: Optional[float]=None, date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True, center: bool=True, scale: bool=True)`。

返回类型：`pd.Series`。

## factor_preprocess/neutralization/advanced/quantile_regression.py

Quantile regression neutralization for cross-sectional residuals.

### quantile_neutralize

[实际实现](../factor_preprocess/neutralization/advanced/quantile_regression.py#L13)。

Cross-sectional quantile regression neutralization.

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, quantile: float=0.5, date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True, normalize: bool=True, max_iter: int=100, tol: float=0.0001)`。

返回类型：`pd.Series`。

## factor_preprocess/neutralization/advanced/robust_regression.py

Robust regression neutralization for cross-sectional residuals.

### huber_neutralize

[实际实现](../factor_preprocess/neutralization/advanced/robust_regression.py#L12)。

Cross-sectional Huber regression neutralization.

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, delta: float=1.35, date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True, normalize: bool=True, max_iter: int=100, tol: float=0.0001)`。

返回类型：`pd.Series`。

### lad_neutralize

[实际实现](../factor_preprocess/neutralization/advanced/robust_regression.py#L175)。

Cross-sectional LAD (Least Absolute Deviations) neutralization.

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True, normalize: bool=True, max_iter: int=100, tol: float=0.0001)`。

返回类型：`pd.Series`。

## factor_preprocess/neutralization/diagnostics.py

Diagnostics for neutralization quality and numerical stability.

### compute_condition_number

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L12)。

Compute condition number of exposure matrix per date.

参数：`(exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', add_intercept: bool=False, sample_weights: Optional[pd.Series]=None)`。

返回类型：`pd.DataFrame`。

### compute_exposure_correlation

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L113)。

Compute correlation matrix of exposures per date.

参数：`(exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id')`。

返回类型：`pd.DataFrame`。

### check_residual_exposures

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L170)。

Check residual exposures after neutralization.

参数：`(residuals: pd.DataFrame, exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='residual')`。

返回类型：`pd.DataFrame`。

### diagnose_neutralization

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L254)。

Comprehensive neutralization diagnostics.

参数：`(values: pd.DataFrame, residuals: pd.DataFrame, exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='value', residual_col: str='residual')`。

返回类型：`Dict[str, pd.DataFrame]`。

### compute_variance_reduction

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L324)。

Compute variance reduction from neutralization.

参数：`(values: pd.DataFrame, residuals: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='value', residual_col: str='residual')`。

返回类型：`pd.DataFrame`。

### flag_ill_conditioned_dates

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L399)。

Flag dates with ill-conditioned exposure matrices.

参数：`(exposures: pd.DataFrame, threshold: float=100.0, date_col: str='date', asset_col: str='asset_id')`。

返回类型：`pd.DataFrame`。

### summarize_diagnostics

[实际实现](../factor_preprocess/neutralization/diagnostics.py#L443)。

Summarize diagnostics into key metrics.

参数：`(diagnostics: Dict[str, pd.DataFrame])`。

返回类型：`Dict[str, any]`。

## factor_preprocess/neutralization/diagnostics_artifact.py

Neutralization diagnostics artifact (DLIB-FP-020 / DLIB-FP-021).

显式导出（含重导出）：`NeutralizationDiagnostics`、`RankDeficientResolution`。

### RankDeficientResolution

[实际实现](../factor_preprocess/neutralization/diagnostics_artifact.py#L39)。

How a rank-deficient neutralization was resolved.

基类：`str`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `FAIL` | `类常量/枚举` | `'FAIL'` |
| `FALLBACK_REGULARIZED` | `类常量/枚举` | `'FALLBACK_REGULARIZED'` |
| `WARN_CONTINUED` | `类常量/枚举` | `'WARN_CONTINUED'` |

### NeutralizationDiagnostics

[实际实现](../factor_preprocess/neutralization/diagnostics_artifact.py#L48)。

Deep-immutable audit record of one neutralization run.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `neutralization_ref` | `str` | `必填/未声明默认` |
| `effective_n` | `int` | `0` |
| `rank` | `Optional[int]` | `None` |
| `condition_number` | `Optional[float]` | `None` |
| `r_squared` | `Optional[float]` | `None` |
| `residual_variance` | `Optional[float]` | `None` |
| `missing_exposure_count` | `int` | `0` |
| `industry_coverage` | `Optional[float]` | `None` |
| `solver` | `Optional[str]` | `None` |
| `regularization` | `Optional[str]` | `None` |
| `warnings` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `rank_deficient_resolution` | `Optional[str]` | `None` |
| `artifact_evidence_ref` | `Optional[str]` | `None` |
| `content_hash` | `str` | `''` |

## factor_preprocess/neutralization/ols.py

OLS neutralization for cross-sectional residuals.

### ols_neutralize

[实际实现](../factor_preprocess/neutralization/ols.py#L12)。

Cross-sectional OLS neutralization (per-date residuals).

参数：`(values: pd.DataFrame, exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='value', min_observations: int=10, add_intercept: bool=True)`。

返回类型：`pd.Series`。

### compute_exposures

[实际实现](../factor_preprocess/neutralization/ols.py#L134)。

Compute exposure coefficients from factor-exposure regression.

参数：`(residuals: pd.DataFrame, exposures: pd.DataFrame, date_col: str='date', asset_col: str='asset_id', value_col: str='value')`。

返回类型：`pd.DataFrame`。

## factor_preprocess/neutralization/regularized.py

Regularized cross-sectional neutralization with explicit solver evidence.

### ridge_neutralize

[实际实现](../factor_preprocess/neutralization/regularized.py#L64)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values, exposures, alpha=1.0, date_col='date', asset_col='asset_id', value_col='value', min_observations=10, add_intercept=True, normalize=True)`。

### lasso_neutralize

[实际实现](../factor_preprocess/neutralization/regularized.py#L66)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values, exposures, alpha=1.0, date_col='date', asset_col='asset_id', value_col='value', min_observations=10, add_intercept=True, normalize=True, max_iter=1000, tol=0.0001)`。

### elastic_net_neutralize

[实际实现](../factor_preprocess/neutralization/regularized.py#L68)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values, exposures, alpha=1.0, l1_ratio=0.5, date_col='date', asset_col='asset_id', value_col='value', min_observations=10, add_intercept=True, normalize=True, max_iter=1000, tol=0.0001)`。

## factor_preprocess/neutralization/spec.py

Neutralization specification contract for the auto-treatment optimizer.

显式导出（含重导出）：`NeutralizationMethod`、`ExposureSet`、`Standardization`、`ConditionNumberPolicy`、`PitIdentity`、`NeutralizationSpec`。

### NeutralizationMethod

[实际实现](../factor_preprocess/neutralization/spec.py#L19)。

Neutralization regression method.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `OLS` | `类常量/枚举` | `'ols'` |
| `RIDGE` | `类常量/枚举` | `'ridge'` |
| `HUBER` | `类常量/枚举` | `'huber'` |
| `PCA` | `类常量/枚举` | `'pca'` |
| `KERNEL` | `类常量/枚举` | `'kernel'` |
| `QUANTILE` | `类常量/枚举` | `'quantile'` |
| `LAD` | `类常量/枚举` | `'lad'` |

### ExposureSet

[实际实现](../factor_preprocess/neutralization/spec.py#L32)。

The set of exposures to neutralize against.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `NONE` | `类常量/枚举` | `'none'` |
| `INDUSTRY` | `类常量/枚举` | `'industry'` |
| `SIZE` | `类常量/枚举` | `'size'` |
| `INDUSTRY_SIZE` | `类常量/枚举` | `'industry_size'` |
| `INDUSTRY_SIZE_BETA` | `类常量/枚举` | `'industry_size_beta'` |
| `CUSTOM_STYLE_SET` | `类常量/枚举` | `'custom_style_set'` |

### Standardization

[实际实现](../factor_preprocess/neutralization/spec.py#L43)。

How exposures are standardized before fitting.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `NONE` | `类常量/枚举` | `'none'` |
| `ZSCORE` | `类常量/枚举` | `'zscore'` |
| `RANK` | `类常量/枚举` | `'rank'` |

### ConditionNumberPolicy

[实际实现](../factor_preprocess/neutralization/spec.py#L51)。

How ill-conditioned exposure matrices are handled.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `FAIL` | `类常量/枚举` | `'fail'` |
| `WARN` | `类常量/枚举` | `'warn'` |
| `REGULARIZE` | `类常量/枚举` | `'regularize'` |

### PitIdentity

[实际实现](../factor_preprocess/neutralization/spec.py#L59)。

PIT (point-in-time) identity of the exposures used.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PIT` | `类常量/枚举` | `'pit'` |
| `AS_OF` | `类常量/枚举` | `'as_of'` |
| `RESTATED` | `类常量/枚举` | `'restated'` |

### NeutralizationSpec

[实际实现](../factor_preprocess/neutralization/spec.py#L84)。

Semantic recipe for a neutralization treatment.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `method` | `NeutralizationMethod` | `NeutralizationMethod.OLS` |
| `exposure_set` | `ExposureSet` | `ExposureSet.INDUSTRY` |
| `industry_schema` | `Optional[str]` | `None` |
| `size_definition` | `Optional[str]` | `None` |
| `standardization` | `Standardization` | `Standardization.ZSCORE` |
| `weights` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `min_obs` | `int` | `10` |
| `condition_number_policy` | `ConditionNumberPolicy` | `ConditionNumberPolicy.WARN` |
| `pit_identity` | `PitIdentity` | `PitIdentity.PIT` |

### NeutralizationSpec.is_production_admissible

[实际实现](../factor_preprocess/neutralization/spec.py#L160)。

True if the method can auto-admit to production.

参数：`(self)`。

返回类型：`bool`。

### NeutralizationSpec.is_research_only

[实际实现](../factor_preprocess/neutralization/spec.py#L165)。

True if the method is research/staging only.

参数：`(self)`。

返回类型：`bool`。

### NeutralizationSpec.kernel_name

[实际实现](../factor_preprocess/neutralization/spec.py#L169)。

Return the underlying computational kernel name.

参数：`(self)`。

返回类型：`str`。

## factor_preprocess/regime/__init__.py

Regime-adaptive transforms for factor preprocessing.

显式导出（含重导出）：`detect_variance_regime`、`detect_correlation_regime`、`RegimeState`、`regime_adaptive_weights`、`fit_regime_weights`、`RegimeWeightState`、`UnknownRegimePolicy`、`serialize_regime_weights`、`deserialize_regime_weights`、`regime_switching_transform`、`fit_regime_switching`、`RegimeSwitchingState`、`CausalRegimeDetector`。

## factor_preprocess/regime/adaptive_weights.py

Regime-dependent factor weighting.

### UnknownRegimePolicy

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L40)。

How an unknown regime label is handled at apply time (FP-P0-03).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `FAIL` | `类常量/枚举` | `'FAIL'` |
| `FAIL_NAN` | `类常量/枚举` | `'FAIL_NAN'` |
| `FALLBACK_GLOBAL` | `类常量/枚举` | `'FALLBACK_GLOBAL'` |
| `IDENTITY_RESEARCH_ONLY` | `类常量/枚举` | `'IDENTITY_RESEARCH_ONLY'` |

### RegimeWeightState

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L70)。

Fitted regime-specific weights (FP-P0-01).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `regime_weights` | `Dict[int, np.ndarray]` | `必填/未声明默认` |
| `factor_names` | `list[str]` | `必填/未声明默认` |
| `n_regimes` | `int` | `必填/未声明默认` |
| `fit_window_start` | `pd.Timestamp` | `必填/未声明默认` |
| `fit_window_end` | `pd.Timestamp` | `必填/未声明默认` |
| `fit_method` | `str` | `'equal'` |
| `target_required` | `bool` | `False` |
| `transform_kind` | `TransformKind` | `field(default_factory=lambda: TransformKind.ROLLING)` |
| `admission` | `str` | `'PRODUCTION'` |
| `global_unqualified` | `Optional[np.ndarray]` | `None` |
| `fallback_reasons` | `Mapping[int, str]` | `field(default_factory=dict)` |
| `effective_counts` | `Mapping[int, tuple[int, ...]]` | `field(default_factory=dict)` |
| `factor_reasons` | `Mapping[int, tuple[str, ...]]` | `field(default_factory=dict)` |

### serialize_regime_weights

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L140)。

Serialize a RegimeWeightState into the canonical FittedState typed payload.

参数：`(state: RegimeWeightState)`。

返回类型：`dict`。

### deserialize_regime_weights

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L166)。

Rebuild a RegimeWeightState from a FittedState payload (FP-P0-01).

参数：`(payload: dict)`。

返回类型：`RegimeWeightState`。

### fit_regime_weights

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L192)。

Fit regime-specific factor weights on a training window.

参数：`(factors: pd.DataFrame, regime_labels: pd.Series, target: Optional[pd.Series]=None, method: Literal['equal', 'volatility_inverse', 'sharpe', 'target_corr_over_vol']='equal', time_col: str='date', factor_cols: Optional[list[str]]=None, min_obs_per_regime: int=30)`。

返回类型：`RegimeWeightState`。

### regime_adaptive_weights

[实际实现](../factor_preprocess/regime/adaptive_weights.py#L403)。

Apply regime-specific weights to factors.

参数：`(factors: pd.DataFrame, regime_labels: pd.Series, fitted_state: RegimeWeightState, time_col: str='date', factor_cols: Optional[list[str]]=None, check_staleness: bool=True, unknown_regime_policy: Union[str, UnknownRegimePolicy]=UnknownRegimePolicy.FAIL, fallback_weights: Optional[np.ndarray]=None)`。

返回类型：`pd.DataFrame`。

## factor_preprocess/regime/causal_detector.py

Stateful, checkpointable past-only correlation regime detector.

### CausalRegimeDetector

[实际实现](../factor_preprocess/regime/causal_detector.py#L9)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `window` | `int` | `必填/未声明默认` |
| `n_regimes` | `int` | `2` |
| `percentiles` | `Optional[List[float]]` | `None` |
| `min_periods` | `Optional[int]` | `None` |
| `time_col` | `str` | `'date'` |
| `value_cols` | `Optional[List[str]]` | `None` |

### CausalRegimeDetector.fit

[实际实现](../factor_preprocess/regime/causal_detector.py#L23)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, train)`。

### CausalRegimeDetector.detect

[实际实现](../factor_preprocess/regime/causal_detector.py#L30)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, data, *, allow_historical_replay=False)`。

### CausalRegimeDetector.checkpoint

[实际实现](../factor_preprocess/regime/causal_detector.py#L46)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### CausalRegimeDetector.from_checkpoint

[实际实现](../factor_preprocess/regime/causal_detector.py#L50)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, p)`。

## factor_preprocess/regime/detector.py

Online regime detection for volatility and correlation regimes.

### RegimeState

[实际实现](../factor_preprocess/regime/detector.py#L13)。

Regime classification result.

基类：`NamedTuple`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `regime` | `pd.Series` | `必填/未声明默认` |
| `regime_strength` | `pd.Series` | `必填/未声明默认` |
| `transition_flag` | `pd.Series` | `必填/未声明默认` |

### detect_variance_regime

[实际实现](../factor_preprocess/regime/detector.py#L33)。

Detect variance regimes using rolling realized volatility.

参数：`(values: pd.DataFrame, window: int, n_regimes: int=2, percentiles: Optional[list[float]]=None, min_periods: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`RegimeState`。

### detect_correlation_regime

[实际实现](../factor_preprocess/regime/detector.py#L196)。

Detect correlation regimes using rolling average pairwise correlation.

参数：`(values: pd.DataFrame, window: int, n_regimes: int=2, percentiles: Optional[list[float]]=None, min_periods: Optional[int]=None, time_col: str='date', value_cols: Optional[list[str]]=None)`。

返回类型：`RegimeState`。

## factor_preprocess/regime/switching.py

Regime-based transform switching.

### RegimeSwitchingState

[实际实现](../factor_preprocess/regime/switching.py#L20)。

Fitted regime-specific transform parameters.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `regime_params` | `Dict[int, Dict[str, Any]]` | `必填/未声明默认` |
| `transform_type` | `str` | `必填/未声明默认` |
| `factor_name` | `str` | `必填/未声明默认` |
| `n_regimes` | `int` | `必填/未声明默认` |
| `fit_window_start` | `pd.Timestamp` | `必填/未声明默认` |
| `fit_window_end` | `pd.Timestamp` | `必填/未声明默认` |

### fit_regime_switching

[实际实现](../factor_preprocess/regime/switching.py#L48)。

Fit regime-specific transform parameters on a training window.

参数：`(values: pd.DataFrame, regime_labels: pd.Series, transform_type: Literal['zscore', 'rank', 'winsor', 'scale', 'none']='zscore', time_col: str='date', value_col: str='value', min_obs_per_regime: int=30)`。

返回类型：`RegimeSwitchingState`。

### regime_switching_transform

[实际实现](../factor_preprocess/regime/switching.py#L176)。

Apply regime-specific transform to values.

参数：`(values: pd.DataFrame, regime_labels: pd.Series, fitted_state: RegimeSwitchingState, time_col: str='date', value_col: str='value', check_staleness: bool=True, unknown_regime_policy=UnknownRegimePolicy.FAIL, fallback_regime: Optional[int]=None)`。

返回类型：`pd.Series`。

## factor_preprocess/registry/__init__.py

Registry package - transform and policy catalogs.

显式导出（含重导出）：`PolicyRegistry`、`PolicyPreset`、`PolicyLevel`、`TransformStep`、`create_default_policies`、`get_default_policy_registry`、`TransformRegistry`、`TransformMetadata`、`TransformCategory`、`create_default_registry`、`get_default_registry`、`PreprocessingPolicy`、`TransformSpec`、`TransformKind`、`TransformMode`。

## factor_preprocess/registry/policies.py

Policy presets for transform pipelines.

### PolicyLevel

[实际实现](../factor_preprocess/registry/policies.py#L26)。

Policy strictness level.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `RESEARCH` | `类常量/枚举` | `'research'` |
| `STAGING` | `类常量/枚举` | `'staging'` |
| `PRODUCTION` | `类常量/枚举` | `'production'` |

### TransformStep

[实际实现](../factor_preprocess/registry/policies.py#L34)。

Single transform step in a pipeline.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `step_id` | `Optional[str]` | `None` |
| `skip_if_missing` | `bool` | `False` |

### PolicyPreset

[实际实现](../factor_preprocess/registry/policies.py#L66)。

Named policy configuration for preprocessing pipelines.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `description` | `str` | `必填/未声明默认` |
| `level` | `PolicyLevel` | `必填/未声明默认` |
| `steps` | `List[TransformStep]` | `必填/未声明默认` |
| `causal_safe` | `bool` | `True` |
| `requires_universe` | `bool` | `False` |
| `requires_returns` | `bool` | `False` |
| `tags` | `List[str]` | `field(default_factory=list)` |

### PolicyPreset.policy_identity

[实际实现](../factor_preprocess/registry/policies.py#L94)。

Content-derived identity of the policy (FP-P1-06).

参数：`(self)`。

返回类型：`str`。

### PolicyPreset.validate

[实际实现](../factor_preprocess/registry/policies.py#L122)。

Validate policy configuration.

参数：`(self, transform_registry=None)`。

返回类型：`Tuple[bool, List[str]]`。

### PolicyRegistry

[实际实现](../factor_preprocess/registry/policies.py#L170)。

Registry of named policy presets.

### PolicyRegistry.__init__

[实际实现](../factor_preprocess/registry/policies.py#L178)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### PolicyRegistry.register

[实际实现](../factor_preprocess/registry/policies.py#L181)。

Register a policy preset.

参数：`(self, policy: PolicyPreset)`。

返回类型：`None`。

### PolicyRegistry.get

[实际实现](../factor_preprocess/registry/policies.py#L204)。

Get an isolated policy snapshot by name.

参数：`(self, name: str)`。

返回类型：`Optional[PolicyPreset]`。

### PolicyRegistry.list_by_level

[实际实现](../factor_preprocess/registry/policies.py#L209)。

List isolated policy snapshots at a given strictness level.

参数：`(self, level: PolicyLevel)`。

返回类型：`List[PolicyPreset]`。

### PolicyRegistry.list_causal_safe

[实际实现](../factor_preprocess/registry/policies.py#L216)。

List isolated snapshots of all causal-safe policies.

参数：`(self)`。

返回类型：`List[PolicyPreset]`。

### PolicyRegistry.all_policies

[实际实现](../factor_preprocess/registry/policies.py#L223)。

Get isolated snapshots of all registered policies.

参数：`(self)`。

返回类型：`List[PolicyPreset]`。

### create_default_policies

[实际实现](../factor_preprocess/registry/policies.py#L228)。

Create registry with standard policy presets.

参数：`()`。

返回类型：`PolicyRegistry`。

### get_default_policy_registry

[实际实现](../factor_preprocess/registry/policies.py#L449)。

Get or create the default global policy registry.

参数：`()`。

返回类型：`PolicyRegistry`。

## factor_preprocess/registry/transforms.py

Transform registry with versioning and discovery.

### TransformCategory

[实际实现](../factor_preprocess/registry/transforms.py#L34)。

Transform category classification.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CROSS_SECTIONAL` | `类常量/枚举` | `'cross_sectional'` |
| `TEMPORAL` | `类常量/枚举` | `'temporal'` |
| `VOLATILITY` | `类常量/枚举` | `'volatility'` |
| `MISSINGNESS` | `类常量/枚举` | `'missingness'` |
| `FRESHNESS` | `类常量/枚举` | `'freshness'` |
| `NEUTRALIZATION` | `类常量/枚举` | `'neutralization'` |
| `REPRESENTATION` | `类常量/枚举` | `'representation'` |

### TransformMetadata

[实际实现](../factor_preprocess/registry/transforms.py#L129)。

Metadata for a registered transform.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `func` | `Callable` | `必填/未声明默认` |
| `category` | `TransformCategory` | `必填/未声明默认` |
| `version` | `str` | `必填/未声明默认` |
| `description` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `tags` | `Set[str]` | `field(default_factory=set)` |
| `causal_safe` | `bool` | `True` |
| `admission` | `str` | `'PRODUCTION'` |
| `signature_hash` | `Optional[str]` | `None` |
| `implementation_hash` | `Optional[str]` | `None` |
| `numeric_policy_hash` | `Optional[str]` | `None` |
| `semantic_id` | `Optional[str]` | `None` |
| `stage` | `Optional[str]` | `None` |
| `family_tags` | `Set[str]` | `field(default_factory=set)` |
| `causality_class` | `Optional[str]` | `None` |
| `requires_fit` | `bool` | `False` |
| `requires_exposure` | `bool` | `False` |
| `requires_universe` | `bool` | `False` |
| `allowed_factor_families` | `Set[str]` | `field(default_factory=set)` |
| `allowed_asset_types` | `Set[str]` | `field(default_factory=set)` |
| `allowed_frequencies` | `Set[str]` | `field(default_factory=set)` |
| `parameter_domain` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `numeric_policy` | `Optional[str]` | `None` |
| `production_admission` | `Optional[str]` | `None` |
| `fe_equivalent_semantics` | `Optional[str]` | `None` |
| `cost_class` | `Optional[str]` | `None` |
| `output_channels` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `implementation_origin` | `Optional[str]` | `None` |
| `fe_operator_id` | `Optional[str]` | `None` |
| `fit_kind` | `Optional[str]` | `None` |

### TransformMetadata.__init__

[实际实现](../factor_preprocess/registry/transforms.py#L212)。

Manually-defined initializer (``init=False``).

参数：`(self, name: str, func: Callable, category: TransformCategory, version: str, description: str, parameters: Optional[Dict[str, Any]]=None, tags: Optional[Set[str]]=None, causal_safe: bool=True, admission: str='PRODUCTION', signature_hash: Optional[str]=None, implementation_hash: Optional[str]=None, numeric_policy_hash: Optional[str]=None, semantic_id: Optional[str]=None, stage: Optional[str]=None, family_tags: Optional[Set[str]]=None, causality_class: Optional[str]=None, requires_fit: bool=False, requires_exposure: bool=False, requires_universe: bool=False, allowed_factor_families: Optional[Set[str]]=None, allowed_asset_types: Optional[Set[str]]=None, allowed_frequencies: Optional[Set[str]]=None, parameter_domain: Optional[Dict[str, Any]]=None, numeric_policy: Optional[str]=None, production_admission: Optional[str]=None, fe_equivalent_semantics: Optional[str]=None, cost_class: Optional[str]=None, output_channels: Optional[Tuple[str, ...]]=None, implementation_origin: Optional[str]=None, fe_operator_id: Optional[str]=None, fit_kind: Optional[str]=None)`。

### TransformMetadata.bind_parameters

[实际实现](../factor_preprocess/registry/transforms.py#L440)。

Validate a partial configured keyword mapping.

参数：`(self, parameters: Dict[str, Any])`。

返回类型：`None`。

### TransformMetadata.bind_call

[实际实现](../factor_preprocess/registry/transforms.py#L450)。

Bind and validate one complete runtime call before execution.

参数：`(self, *args, **kwargs)`。

返回类型：`inspect.BoundArguments`。

### TransformRegistry

[实际实现](../factor_preprocess/registry/transforms.py#L481)。

Central registry for preprocessing transforms.

### TransformRegistry.__init__

[实际实现](../factor_preprocess/registry/transforms.py#L493)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### TransformRegistry.register

[实际实现](../factor_preprocess/registry/transforms.py#L512)。

Register a transform.

参数：`(self, name: str, func: Callable, category: TransformCategory, version: str='1.0.0', description: str='', parameters: Optional[Dict[str, Any]]=None, tags: Optional[Set[str]]=None, causal_safe: bool=True, admission: Optional[str]=None, semantic_id: Optional[str]=None, stage: Optional[str]=None, family_tags: Optional[Set[str]]=None, causality_class: Optional[str]=None, requires_fit: bool=False, requires_exposure: bool=False, requires_universe: bool=False, allowed_factor_families: Optional[Set[str]]=None, allowed_asset_types: Optional[Set[str]]=None, allowed_frequencies: Optional[Set[str]]=None, parameter_domain: Optional[Dict[str, Any]]=None, numeric_policy: Optional[str]=None, fe_equivalent_semantics: Optional[str]=None, cost_class: Optional[str]=None, output_channels: Optional[Tuple[str, ...]]=None, implementation_origin: Optional[str]=None, fe_operator_id: Optional[str]=None, fit_kind: Optional[str]=None)`。

返回类型：`None`。

### TransformRegistry.get

[实际实现](../factor_preprocess/registry/transforms.py#L706)。

Get an isolated deep-frozen transform metadata snapshot by name.

参数：`(self, name: str)`。

返回类型：`Optional[TransformMetadata]`。

### TransformRegistry.enrich

[实际实现](../factor_preprocess/registry/transforms.py#L718)。

Attach additional semantic metadata to an existing transform.

参数：`(self, name: str, **fields)`。

返回类型：`None`。

### TransformRegistry.validate_production

[实际实现](../factor_preprocess/registry/transforms.py#L784)。

Resolve registry metadata and fail closed for production admission.

参数：`(self, name: str)`。

返回类型：`TransformMetadata`。

### TransformRegistry.resolve_origin

[实际实现](../factor_preprocess/registry/transforms.py#L795)。

Resolve the effective implementation origin for a transform.

参数：`(self, name: str)`。

返回类型：`str`。

### TransformRegistry.get_execution

[实际实现](../factor_preprocess/registry/transforms.py#L812)。

Return the callable to execute for a transform.

参数：`(self, name: str)`。

### TransformRegistry.get_function

[实际实现](../factor_preprocess/registry/transforms.py#L838)。

Reject the retired unvalidated native-kernel execution bypass.

参数：`(self, name: str)`。

返回类型：`Optional[Callable]`。

### TransformRegistry.get_research_reference_function

[实际实现](../factor_preprocess/registry/transforms.py#L854)。

Return a retained native parity kernel only with explicit consent.

参数：`(self, name: str, *, allow_research: bool=False)`。

返回类型：`Optional[Callable]`。

### TransformRegistry.get_recipe_execution

[实际实现](../factor_preprocess/registry/transforms.py#L868)。

Compile an FE recipe with explicit execution authority.

参数：`(self, recipe, *, execution_context=None, backend=None, fitted_state_refs=(), allow_research=False)`。

### TransformRegistry.list_by_category

[实际实现](../factor_preprocess/registry/transforms.py#L890)。

List isolated deep-frozen transform metadata snapshots in a category.

参数：`(self, category: TransformCategory)`。

返回类型：`List[TransformMetadata]`。

### TransformRegistry.list_by_tag

[实际实现](../factor_preprocess/registry/transforms.py#L895)。

List isolated deep-frozen snapshots with a given tag.

参数：`(self, tag: str)`。

返回类型：`List[TransformMetadata]`。

### TransformRegistry.list_causal_safe

[实际实现](../factor_preprocess/registry/transforms.py#L900)。

List isolated deep-frozen snapshots of all causal-safe transforms.

参数：`(self)`。

返回类型：`List[TransformMetadata]`。

### TransformRegistry.all_transforms

[实际实现](../factor_preprocess/registry/transforms.py#L907)。

Get isolated deep-frozen snapshots of all registered transforms.

参数：`(self)`。

返回类型：`List[TransformMetadata]`。

### TransformRegistry.get_signature_hash

[实际实现](../factor_preprocess/registry/transforms.py#L911)。

Get signature hash for reproducibility tracking.

参数：`(self, name: str)`。

返回类型：`Optional[str]`。

### TransformRegistry.seal

[实际实现](../factor_preprocess/registry/transforms.py#L916)。

Freeze the registry into an immutable snapshot identity (FP-P1-05).

参数：`(self)`。

返回类型：`str`。

### TransformRegistry.diagnostic_events

[实际实现](../factor_preprocess/registry/transforms.py#L938)。

Stable ordered audit trail of registry lifecycle events.

参数：`(self)`。

返回类型：`List[str]`。

### TransformRegistry.snapshot_identity

[实际实现](../factor_preprocess/registry/transforms.py#L950)。

Content-derived identity of the sealed snapshot (None if not sealed).

参数：`(self)`。

返回类型：`Optional[str]`。

### TransformRegistry.is_sealed

[实际实现](../factor_preprocess/registry/transforms.py#L955)。

Whether the registry has been frozen.

参数：`(self)`。

返回类型：`bool`。

### create_default_registry

[实际实现](../factor_preprocess/registry/transforms.py#L960)。

Create registry with all built-in transforms.

参数：`()`。

返回类型：`TransformRegistry`。

### get_default_registry

[实际实现](../factor_preprocess/registry/transforms.py#L1525)。

Get or create the default global registry.

参数：`()`。

返回类型：`TransformRegistry`。

## factor_preprocess/representation/__init__.py

Representation package - multi-channel representations for model input.

显式导出（含重导出）：`build_multichannel`、`MultichannelConfig`、`MultichannelResult`、`build_linear_ready`、`assess_collinearity`、`LinearReadyConfig`、`LinearReadyResult`、`build_tree_ready`、`suggest_tree_params`、`TreeReadyConfig`、`TreeReadyResult`、`build_neural_ready`、`prepare_embeddings`、`NeuralReadyConfig`、`NeuralReadyResult`、`RepresentationProfileId`、`REPRESENTATION_POLICY_VERSION`、`RepresentationPolicy`、`UnknownRepresentationProfileError`、`get_representation_policy`、`list_representation_policies`、`ArtifactKind`、`CANONICAL_FACTOR_NAMESPACE_PREFIX`、`REPRESENTATION_NAMESPACE_PREFIX`、`CanonicalAssetOverwriteError`、`FeatureRepresentationArtifact`、`register_feature_representation`、`write_feature_representation`、`NonInferiorityTolerance`、`NON_INFERIORITY_POLICY_VERSION`、`DEFAULT_NON_INFERIORITY_TOLERANCE`、`non_inferior`、`SignalDestructionConflict`。

## factor_preprocess/representation/linear_ready.py

Linear model ready features: dense feature matrix builder.

### LinearReadyConfig

[实际实现](../factor_preprocess/representation/linear_ready.py#L12)。

Configuration for linear-ready feature preparation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `fill_method` | `Literal['zero', 'mean', 'drop']` | `'zero'` |
| `add_intercept` | `bool` | `False` |
| `standardize` | `bool` | `True` |
| `ddof` | `int` | `1` |

### LinearReadyResult

[实际实现](../factor_preprocess/representation/linear_ready.py#L28)。

Result of linear-ready transformation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `X` | `np.ndarray` | `必填/未声明默认` |
| `feature_names` | `list` | `必填/未声明默认` |
| `n_samples` | `int` | `必填/未声明默认` |
| `n_features` | `int` | `必填/未声明默认` |
| `has_intercept` | `bool` | `必填/未声明默认` |
| `fill_stats` | `dict` | `必填/未声明默认` |

### LinearReadyResult.validate

[实际实现](../factor_preprocess/representation/linear_ready.py#L38)。

Validate that features are ready for linear models.

参数：`(self)`。

返回类型：`bool`。

### build_linear_ready

[实际实现](../factor_preprocess/representation/linear_ready.py#L59)。

Build dense feature matrix ready for linear models.

参数：`(values: np.ndarray, config: LinearReadyConfig, feature_names: Optional[list]=None)`。

返回类型：`LinearReadyResult`。

### assess_collinearity

[实际实现](../factor_preprocess/representation/linear_ready.py#L174)。

Assess collinearity in feature matrix.

参数：`(X: np.ndarray, threshold: float=0.99)`。

返回类型：`dict`。

## factor_preprocess/representation/multichannel.py

Multichannel representation: raw, rank, zscore, residual channels.

### MultichannelConfig

[实际实现](../factor_preprocess/representation/multichannel.py#L17)。

Configuration for multichannel representation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `channels` | `List[ChannelType]` | `必填/未声明默认` |
| `rank_pct` | `bool` | `True` |
| `zscore_ddof` | `int` | `1` |
| `zscore_constant_value` | `float` | `0.0` |
| `residual_basis` | `Optional[str]` | `None` |

### MultichannelResult

[实际实现](../factor_preprocess/representation/multichannel.py#L38)。

Result of multichannel transformation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `channels` | `Dict[str, np.ndarray]` | `必填/未声明默认` |
| `channel_order` | `List[str]` | `必填/未声明默认` |
| `shape` | `tuple` | `必填/未声明默认` |
| `n_factors` | `int` | `必填/未声明默认` |
| `n_channels` | `int` | `必填/未声明默认` |

### MultichannelResult.get_channel

[实际实现](../factor_preprocess/representation/multichannel.py#L47)。

Retrieve a specific channel by name.

参数：`(self, channel_name: str)`。

返回类型：`np.ndarray`。

### MultichannelResult.as_stacked

[实际实现](../factor_preprocess/representation/multichannel.py#L53)。

Stack all channels into a single array.

参数：`(self)`。

返回类型：`np.ndarray`。

### MultichannelResult.as_interleaved

[实际实现](../factor_preprocess/representation/multichannel.py#L65)。

Interleave channels by factor.

参数：`(self)`。

返回类型：`np.ndarray`。

### build_multichannel

[实际实现](../factor_preprocess/representation/multichannel.py#L98)。

Build multichannel representation from factor values.

参数：`(values: np.ndarray, config: MultichannelConfig, axis: int=-2)`。

返回类型：`MultichannelResult`。

## factor_preprocess/representation/neural_ready.py

Neural network ready features: robust scaling and embedding preparation.

### NeuralReadyConfig

[实际实现](../factor_preprocess/representation/neural_ready.py#L12)。

Configuration for neural-ready feature preparation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `scaling` | `Literal['robust', 'minmax', 'standard', 'none']` | `'robust'` |
| `robust_quantile_range` | `tuple` | `(25.0, 75.0)` |
| `minmax_range` | `tuple` | `(0.0, 1.0)` |
| `clip_outliers` | `bool` | `True` |
| `outlier_quantile_range` | `tuple` | `(0.5, 99.5)` |
| `handle_missing` | `Literal['zero', 'mean', 'forward_fill']` | `'zero'` |
| `add_time_features` | `bool` | `False` |
| `normalize_per_sample` | `bool` | `False` |

### NeuralReadyResult

[实际实现](../factor_preprocess/representation/neural_ready.py#L49)。

Result of neural-ready transformation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `X` | `np.ndarray` | `必填/未声明默认` |
| `feature_names` | `List[str]` | `必填/未声明默认` |
| `n_samples` | `int` | `必填/未声明默认` |
| `n_features` | `int` | `必填/未声明默认` |
| `scaling_params` | `Dict` | `必填/未声明默认` |
| `preprocessing_stats` | `Dict` | `必填/未声明默认` |

### NeuralReadyResult.validate

[实际实现](../factor_preprocess/representation/neural_ready.py#L59)。

Validate that features are ready for neural networks.

参数：`(self)`。

返回类型：`bool`。

### build_neural_ready

[实际实现](../factor_preprocess/representation/neural_ready.py#L80)。

Build feature matrix ready for neural networks.

参数：`(values: np.ndarray, config: NeuralReadyConfig, feature_names: Optional[List[str]]=None)`。

返回类型：`NeuralReadyResult`。

### prepare_embeddings

[实际实现](../factor_preprocess/representation/neural_ready.py#L264)。

Prepare categorical features for embedding layers.

参数：`(categorical_features: np.ndarray, embedding_dim: Optional[int]=None)`。

返回类型：`Dict`。

## factor_preprocess/representation/policy.py

Model-specific representation policy (R61-FI-044, plan §28).

显式导出（含重导出）：`RepresentationProfileId`、`REPRESENTATION_POLICY_VERSION`、`RepresentationPolicy`、`UnknownRepresentationProfileError`、`get_representation_policy`、`list_representation_policies`、`ArtifactKind`、`CANONICAL_FACTOR_NAMESPACE_PREFIX`、`REPRESENTATION_NAMESPACE_PREFIX`、`CanonicalAssetOverwriteError`、`FeatureRepresentationArtifact`、`register_feature_representation`、`write_feature_representation`、`NonInferiorityTolerance`、`NON_INFERIORITY_POLICY_VERSION`、`DEFAULT_NON_INFERIORITY_TOLERANCE`、`non_inferior`、`SignalDestructionConflict`、`decide_representation`。

### RepresentationProfileId

[实际实现](../factor_preprocess/representation/policy.py#L49)。

The four frozen model representation policy profiles (plan §28).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TREE_TABULAR` | `类常量/枚举` | `'TREE_TABULAR'` |
| `LINEAR` | `类常量/枚举` | `'LINEAR'` |
| `NEURAL_TABULAR` | `类常量/枚举` | `'NEURAL_TABULAR'` |
| `SEQUENCE_MODEL` | `类常量/枚举` | `'SEQUENCE_MODEL'` |

### RepresentationPolicy

[实际实现](../factor_preprocess/representation/policy.py#L63)。

Frozen, versioned representation policy for one model family.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `profile_id` | `RepresentationProfileId` | `必填/未声明默认` |
| `version` | `str` | `必填/未声明默认` |
| `display_name` | `str` | `必填/未声明默认` |
| `description` | `str` | `必填/未声明默认` |
| `allowed_representations` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `required_normalizations` | `Tuple[str, ...]` | `field(default_factory=tuple)` |
| `required_normalization_any_of` | `Tuple[Tuple[str, ...], ...]` | `field(default_factory=tuple)` |
| `zscore_optional` | `bool` | `False` |
| `robust_outlier_handling_required` | `bool` | `False` |
| `train_fitted_scaling_required` | `bool` | `False` |
| `missing_channel_required` | `bool` | `False` |
| `stable_clipping_required` | `bool` | `False` |
| `rank_gaussian_allowed` | `bool` | `False` |
| `temporal_causal_normalization_required` | `bool` | `False` |

### RepresentationPolicy.require_normalizations

[实际实现](../factor_preprocess/representation/policy.py#L109)。

Fail closed unless all required normalization ALTERNATIVES are met.

参数：`(self, proposed_normalizations: Tuple[str, ...])`。

返回类型：`None`。

### RepresentationPolicy.allows_representation

[实际实现](../factor_preprocess/representation/policy.py#L134)。

True when the profile permits this alpha-side representation.

参数：`(self, representation: str)`。

返回类型：`bool`。

### UnknownRepresentationProfileError

[实际实现](../factor_preprocess/representation/policy.py#L239)。

A representation profile id is not one of the four frozen profiles.

基类：`InvalidContractError`。

### get_representation_policy

[实际实现](../factor_preprocess/representation/policy.py#L243)。

Resolve a frozen representation policy by id (fail closed).

参数：`(profile_id)`。

返回类型：`RepresentationPolicy`。

### list_representation_policies

[实际实现](../factor_preprocess/representation/policy.py#L261)。

All four frozen representation policies in stable profile order.

参数：`()`。

返回类型：`Tuple[RepresentationPolicy, ...]`。

### ArtifactKind

[实际实现](../factor_preprocess/representation/policy.py#L271)。

What a model-input artifact represents (never a canonical factor asset).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `MODEL_SPECIFIC_REPRESENTATION` | `类常量/枚举` | `'model_specific_representation'` |

### CanonicalAssetOverwriteError

[实际实现](../factor_preprocess/representation/policy.py#L283)。

A model-specific representation tried to overwrite a canonical asset.

基类：`InvalidContractError`。

### FeatureRepresentationArtifact

[实际实现](../factor_preprocess/representation/policy.py#L288)。

Model-specific representation of a factor, recorded as its OWN artifact.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `artifact_id` | `str` | `必填/未声明默认` |
| `factor_id` | `str` | `必填/未声明默认` |
| `canonical_factor_ref` | `str` | `必填/未声明默认` |
| `model_id` | `str` | `必填/未声明默认` |
| `profile_id` | `RepresentationProfileId` | `必填/未声明默认` |
| `profile_version` | `str` | `必填/未声明默认` |
| `factor_version` | `str` | `必填/未声明默认` |
| `canonical_storage_ref` | `str` | `必填/未声明默认` |
| `destination_storage_ref` | `str` | `必填/未声明默认` |
| `transform_chain` | `Tuple[Dict[str, Any], ...]` | `field(default_factory=tuple)` |
| `representation_name` | `str` | `''` |
| `artifact_kind` | `ArtifactKind` | `ArtifactKind.MODEL_SPECIFIC_REPRESENTATION` |
| `content_hash` | `str` | `''` |

### FeatureRepresentationArtifact.to_dict

[实际实现](../factor_preprocess/representation/policy.py#L380)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### register_feature_representation

[实际实现](../factor_preprocess/representation/policy.py#L398)。

Create + register a model-specific representation artifact.

参数：`(*, artifact_id: str, factor_id: str, canonical_factor_ref: str, model_id: str, factor_version: str, canonical_storage_ref: str, destination_storage_ref: str, profile_id, transform_chain, representation_name: str='')`。

返回类型：`FeatureRepresentationArtifact`。

### write_feature_representation

[实际实现](../factor_preprocess/representation/policy.py#L440)。

Create a representation file without overwriting any resolved object.

参数：`(artifact: FeatureRepresentationArtifact, payload: bytes, *, canonical_storage_path, representation_storage_root)`。

返回类型：`Path`。

### SignalDestructionConflict

[实际实现](../factor_preprocess/representation/policy.py#L493)。

A model-mandated normalization materially destroys the alpha signal.

基类：`InvalidContractError`。

### decide_representation

[实际实现](../factor_preprocess/representation/policy.py#L503)。

Delegate representation admission to the shared DecisionProvider.

参数：`(provider, request, treated_candidate_id: str)`。

### NonInferiorityTolerance

[实际实现](../factor_preprocess/representation/policy.py#L528)。

Versioned tolerance for the model-safety non-inferiority test.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `alpha_metric` | `str` | `必填/未声明默认` |
| `max_relative_destroy` | `float` | `必填/未声明默认` |
| `policy_version` | `str` | `NON_INFERIORITY_POLICY_VERSION` |
| `allow_exact_equal` | `bool` | `True` |

### non_inferior

[实际实现](../factor_preprocess/representation/policy.py#L563)。

True when the model-treated representation is non-inferior.

参数：`(evidence_alpha: float, evidence_model_treated: float, tolerance: NonInferiorityTolerance=DEFAULT_NON_INFERIORITY_TOLERANCE)`。

返回类型：`bool`。

## factor_preprocess/representation/tree_ready.py

Tree model ready features: categorical encoding stubs.

### TreeReadyConfig

[实际实现](../factor_preprocess/representation/tree_ready.py#L12)。

Configuration for tree-ready feature preparation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `handle_missing` | `Literal['keep', 'flag', 'fill_median']` | `'keep'` |
| `add_missing_indicator` | `bool` | `False` |
| `categorical_encoding` | `Literal['ordinal', 'onehot', 'none']` | `'none'` |
| `clip_outliers` | `bool` | `False` |
| `outlier_std_threshold` | `float` | `5.0` |

### TreeReadyResult

[实际实现](../factor_preprocess/representation/tree_ready.py#L36)。

Result of tree-ready transformation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `X` | `np.ndarray` | `必填/未声明默认` |
| `feature_names` | `List[str]` | `必填/未声明默认` |
| `n_samples` | `int` | `必填/未声明默认` |
| `n_features` | `int` | `必填/未声明默认` |
| `missing_indicators` | `Optional[np.ndarray]` | `必填/未声明默认` |
| `preprocessing_stats` | `Dict` | `必填/未声明默认` |

### TreeReadyResult.validate

[实际实现](../factor_preprocess/representation/tree_ready.py#L46)。

Validate that features are ready for tree models.

参数：`(self)`。

返回类型：`bool`。

### build_tree_ready

[实际实现](../factor_preprocess/representation/tree_ready.py#L64)。

Build feature matrix ready for tree-based models.

参数：`(values: np.ndarray, config: TreeReadyConfig, feature_names: Optional[List[str]]=None, categorical_mask: Optional[np.ndarray]=None)`。

返回类型：`TreeReadyResult`。

### suggest_tree_params

[实际实现](../factor_preprocess/representation/tree_ready.py#L189)。

Suggest initial tree model hyperparameters based on data characteristics.

参数：`(X: np.ndarray)`。

返回类型：`Dict`。

## factor_preprocess/transforms/__init__.py

Transforms package.

显式导出（含重导出）：`cs_rank`、`cs_zscore`、`cs_demean`、`cs_winsor`、`cs_scale`、`rolling_mean`、`rolling_std`、`rolling_zscore`、`ewma`、`trailing_sma`、`trailing_median`、`robust_ewma`、`kama`、`one_sided_iir_lowpass`、`kalman_local_level`、`volatility_scale`、`volatility_scale_returns`、`realized_volatility`、`ewma_volatility`、`garch_inspired_volatility`、`forward_fill`、`missing_indicator`、`missing_run_length`、`missing_rate`、`linear_interpolate`、`time_weighted_interpolate`、`impute_with_fallback`、`days_since_update`、`observation_age`、`freshness_score`、`stale_data_indicator`、`event_decay`、`freshness_aware_fill`。

## factor_preprocess/transforms/cross_sectional.py

Cross-sectional transforms: rank, zscore, demean, winsor.

### cs_rank

[实际实现](../factor_preprocess/transforms/cross_sectional.py#L11)。

Cross-sectional rank with explicit tie handling.

参数：`(values: np.ndarray, axis: int=-1, method: Literal['average', 'min', 'max', 'dense', 'ordinal']='average', pct: bool=False)`。

返回类型：`np.ndarray`。

### cs_zscore

[实际实现](../factor_preprocess/transforms/cross_sectional.py#L79)。

Cross-sectional z-score normalization.

参数：`(values: np.ndarray, axis: int=-1, ddof: int=1, constant_value: float=0.0)`。

返回类型：`np.ndarray`。

### cs_demean

[实际实现](../factor_preprocess/transforms/cross_sectional.py#L128)。

Cross-sectional demean.

参数：`(values: np.ndarray, axis: int=-1)`。

返回类型：`np.ndarray`。

### cs_winsor

[实际实现](../factor_preprocess/transforms/cross_sectional.py#L154)。

Cross-sectional winsorization by quantiles.

参数：`(values: np.ndarray, lower: float=0.01, upper: float=0.99, axis: int=-1)`。

返回类型：`np.ndarray`。

### cs_scale

[实际实现](../factor_preprocess/transforms/cross_sectional.py#L203)。

Cross-sectional scaling to target standard deviation.

参数：`(values: np.ndarray, axis: int=-1, target_std: float=1.0, ddof: int=1)`。

返回类型：`np.ndarray`。

## factor_preprocess/transforms/decomposition/__init__.py

Time Series Decomposition Transforms

显式导出（含重导出）：`hp_filter`、`hp_decompose`、`stl_decompose`、`seasonal_component`、`trend_component`、`residual_component`、`bandpass_filter`、`extract_cycle`、`christiano_fitzgerald_filter`、`wavelet_decompose`、`wavelet_smooth`、`wavelet_denoise`。

## factor_preprocess/transforms/decomposition/cycle.py

Cycle extraction using full-series bandpass filters.

### bandpass_filter

[实际实现](../factor_preprocess/transforms/decomposition/cycle.py#L14)。

Offline-only bandpass filter for cycle extraction.

参数：`(values: pd.DataFrame, low_freq: float, high_freq: float, order: int=4, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### extract_cycle

[实际实现](../factor_preprocess/transforms/decomposition/cycle.py#L118)。

Extract cycle component with specified period range.

参数：`(values: pd.DataFrame, low_period: int, high_period: int, order: int=4, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### christiano_fitzgerald_filter

[实际实现](../factor_preprocess/transforms/decomposition/cycle.py#L191)。

Offline-only Christiano-Fitzgerald bandpass filter for cycle extraction.

参数：`(values: pd.DataFrame, low_period: int, high_period: int, drift: bool=True, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/decomposition/seasonal.py

Seasonal decomposition using STL (Seasonal and Trend decomposition using Loess).

### STLResult

[实际实现](../factor_preprocess/transforms/decomposition/seasonal.py#L17)。

Container for STL decomposition results.

基类：`NamedTuple`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trend` | `pd.Series` | `必填/未声明默认` |
| `seasonal` | `pd.Series` | `必填/未声明默认` |
| `residual` | `pd.Series` | `必填/未声明默认` |

### stl_decompose

[实际实现](../factor_preprocess/transforms/decomposition/seasonal.py#L24)。

Causal STL (Seasonal-Trend decomposition using Loess) decomposition.

参数：`(values: pd.DataFrame, period: int, seasonal: int=7, trend: Optional[int]=None, low_pass: Optional[int]=None, seasonal_deg: int=1, trend_deg: int=1, low_pass_deg: int=1, robust: bool=False, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`Tuple[pd.Series, pd.Series, pd.Series]`。

### seasonal_component

[实际实现](../factor_preprocess/transforms/decomposition/seasonal.py#L172)。

Extract seasonal component using STL.

参数：`(values: pd.DataFrame, period: int, seasonal: int=7, trend: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### trend_component

[实际实现](../factor_preprocess/transforms/decomposition/seasonal.py#L219)。

Extract trend component using STL.

参数：`(values: pd.DataFrame, period: int, seasonal: int=7, trend: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### residual_component

[实际实现](../factor_preprocess/transforms/decomposition/seasonal.py#L266)。

Extract residual component using STL.

参数：`(values: pd.DataFrame, period: int, seasonal: int=7, trend: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/decomposition/trend.py

Trend extraction using Hodrick-Prescott filter.

### hp_filter

[实际实现](../factor_preprocess/transforms/decomposition/trend.py#L21)。

Hodrick-Prescott filter for trend extraction. OFFLINE_ONLY.

参数：`(values: pd.DataFrame, lambda_param: float=1600.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### hp_decompose

[实际实现](../factor_preprocess/transforms/decomposition/trend.py#L118)。

Hodrick-Prescott decomposition into trend and cycle. OFFLINE_ONLY.

参数：`(values: pd.DataFrame, lambda_param: float=1600.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`Tuple[pd.Series, pd.Series]`。

## factor_preprocess/transforms/decomposition/wavelet.py

Wavelet decomposition for offline multi-scale time series analysis.

### wavelet_decompose

[实际实现](../factor_preprocess/transforms/decomposition/wavelet.py#L14)。

Offline-only wavelet decomposition into approximation and detail coefficients.

参数：`(values: pd.DataFrame, wavelet: str='db4', level: Optional[int]=None, mode: str='symmetric', asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`Dict[str, pd.Series]`。

### wavelet_smooth

[实际实现](../factor_preprocess/transforms/decomposition/wavelet.py#L167)。

Offline-only wavelet smoothing by reconstructing approximation coefficients.

参数：`(values: pd.DataFrame, wavelet: str='db4', level: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### wavelet_denoise

[实际实现](../factor_preprocess/transforms/decomposition/wavelet.py#L227)。

Offline-only wavelet denoising using soft/hard thresholding.

参数：`(values: pd.DataFrame, wavelet: str='db4', level: Optional[int]=None, threshold_mode: str='soft', threshold_scale: float=1.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/event_decay.py

Short-halflife event-decay persistence (causal, one-sided).

### event_decay

[实际实现](../factor_preprocess/transforms/event_decay.py#L24)。

Causal short-halflife event decay.

参数：`(values: pd.DataFrame, halflife: float, min_periods: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/freshness.py

Data freshness transforms for tracking observation age and staleness.

### days_since_update

[实际实现](../factor_preprocess/transforms/freshness.py#L12)。

Count days since last valid observation for each asset.

参数：`(values: pd.DataFrame, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### observation_age

[实际实现](../factor_preprocess/transforms/freshness.py#L60)。

Compute age of observation relative to decision time.

参数：`(values: pd.DataFrame, observation_date_col: str, current_date_col: str, asset_col: str='asset_id', time_col: str='date')`。

返回类型：`pd.Series`。

### freshness_score

[实际实现](../factor_preprocess/transforms/freshness.py#L110)。

Compute exponential freshness score based on data age.

参数：`(values: pd.DataFrame, halflife_days: float, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### stale_data_indicator

[实际实现](../factor_preprocess/transforms/freshness.py#L167)。

Binary indicator for stale data.

参数：`(values: pd.DataFrame, max_days: int, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/missingness.py

Missingness transforms for handling missing data.

### forward_fill

[实际实现](../factor_preprocess/transforms/missingness.py#L14)。

Forward fill missing values with bounded lag.

参数：`(values: pd.DataFrame, max_lag: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### missing_indicator

[实际实现](../factor_preprocess/transforms/missingness.py#L68)。

Create binary indicator for missing values.

参数：`(values: pd.DataFrame, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### missing_run_length

[实际实现](../factor_preprocess/transforms/missingness.py#L106)。

Count consecutive missing observations up to current time.

参数：`(values: pd.DataFrame, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### missing_rate

[实际实现](../factor_preprocess/transforms/missingness.py#L160)。

Compute rolling missing rate from lagged observations.

参数：`(values: pd.DataFrame, window: int, min_periods: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### linear_interpolate

[实际实现](../factor_preprocess/transforms/missingness.py#L218)。

Linear interpolation with bounded gap size.

参数：`(values: pd.DataFrame, max_gap: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### time_weighted_interpolate

[实际实现](../factor_preprocess/transforms/missingness.py#L275)。

Time-weighted interpolation respecting actual time distances.

参数：`(values: pd.DataFrame, max_gap: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### impute_with_fallback

[实际实现](../factor_preprocess/transforms/missingness.py#L343)。

Impute missing values with fallback strategy hierarchy.

参数：`(values: pd.DataFrame, fallback_strategy: str='forward_fill', max_lag: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/repair_shapes.py

Stateless value-repair primitives used by research repair plans.

### cross_sectional_rank

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L31)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, method: str='average')`。

返回类型：`pd.Series`。

### rank_shape

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L37)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, center: float, power: float, inverted: bool=False, asymmetric: bool=False)`。

返回类型：`pd.Series`。

### capped_zscore

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L59)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, cap: float)`。

返回类型：`pd.Series`。

### tail_hinge

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L66)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, hinge: str, hinge_value: float)`。

返回类型：`pd.Series`。

### tail_saturation

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L77)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, quantile: float, side: str)`。

返回类型：`pd.Series`。

### robust_scale

[实际实现](../factor_preprocess/transforms/repair_shapes.py#L92)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: pd.DataFrame, *, scale: str, center: str)`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/rolling.py

Rolling (time-series) transforms with explicit causality.

### rolling_mean

[实际实现](../factor_preprocess/transforms/rolling.py#L25)。

Causal rolling mean per asset.

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### rolling_std

[实际实现](../factor_preprocess/transforms/rolling.py#L81)。

Causal rolling standard deviation per asset.

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, ddof: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### rolling_zscore

[实际实现](../factor_preprocess/transforms/rolling.py#L136)。

Causal rolling z-score normalization per asset.

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, ddof: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### ewma

[实际实现](../factor_preprocess/transforms/rolling.py#L207)。

Causal exponentially-weighted moving average per asset.

参数：`(values: pd.DataFrame, halflife: float, min_periods: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/smoothing.py

Causal one-sided signal smoothers.

### trailing_sma

[实际实现](../factor_preprocess/transforms/smoothing.py#L30)。

Lagged trailing simple moving average (SMA).

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### trailing_median

[实际实现](../factor_preprocess/transforms/smoothing.py#L85)。

Trailing rolling median (robust to spikes/outliers).

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### robust_ewma

[实际实现](../factor_preprocess/transforms/smoothing.py#L140)。

Lagged EWMA computed on winsorized values (robust to outliers).

参数：`(values: pd.DataFrame, halflife: float, winsor_std: float=4.0, min_periods: int=1, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### kama

[实际实现](../factor_preprocess/transforms/smoothing.py#L207)。

Kaufman Adaptive Moving Average (KAMA), built recursively forward only.

参数：`(values: pd.DataFrame, period_fast: int=2, period_slow: int=30, period_er: int=10, asset_col: str='asset_id', time_col: str='date', value_col: str='value', use_current: bool=False)`。

返回类型：`pd.Series`。

### one_sided_iir_lowpass

[实际实现](../factor_preprocess/transforms/smoothing.py#L342)。

One-pole IIR low-pass filter applied forward only.

参数：`(values: pd.DataFrame, alpha: float, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### kalman_local_level

[实际实现](../factor_preprocess/transforms/smoothing.py#L404)。

One-sided Kalman local-level smoother.

参数：`(values: pd.DataFrame, process_noise: float, measurement_noise: float, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/treatment_variants.py

Treatment variant transforms that the eligibility engine may propose.

### freshness_aware_fill

[实际实现](../factor_preprocess/transforms/treatment_variants.py#L25)。

Freshness-aware forward fill.

参数：`(values: pd.DataFrame, max_lag: Optional[int]=None, decay_halflife: float=10.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## factor_preprocess/transforms/volatility.py

Volatility scaling transforms with explicit causality.

### volatility_scale

[实际实现](../factor_preprocess/transforms/volatility.py#L12)。

Scale values by lagged volatility to target volatility.

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, ddof: int=1, target_vol: float=1.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### volatility_scale_returns

[实际实现](../factor_preprocess/transforms/volatility.py#L88)。

Scale returns by lagged return volatility.

参数：`(returns: pd.DataFrame, window: int, min_periods: Optional[int]=None, ddof: int=1, target_vol: float=0.01, asset_col: str='asset_id', time_col: str='date', return_col: str='return')`。

返回类型：`pd.Series`。

### realized_volatility

[实际实现](../factor_preprocess/transforms/volatility.py#L144)。

Compute realized volatility from lagged observations.

参数：`(values: pd.DataFrame, window: int, min_periods: Optional[int]=None, ddof: int=1, annualization_factor: float=1.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### ewma_volatility

[实际实现](../factor_preprocess/transforms/volatility.py#L212)。

Compute EWMA volatility from lagged observations.

参数：`(values: pd.DataFrame, halflife: float, min_periods: int=1, annualization_factor: float=1.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

### garch_inspired_volatility

[实际实现](../factor_preprocess/transforms/volatility.py#L271)。

Compute GARCH-inspired rolling volatility with short-term and long-term components.

参数：`(values: pd.DataFrame, short_window: int=5, long_window: int=20, min_periods: Optional[int]=None, ddof: int=1, annualization_factor: float=1.0, asset_col: str='asset_id', time_col: str='date', value_col: str='value')`。

返回类型：`pd.Series`。

## 源码一致性

<details>
<summary>展开模块指纹</summary>

| 模块 | SHA-256 |
|---|---|
| `factor_preprocess/__init__.py` | `f5a80d33d15bec2ea64c588986c0351f5270612e9cc6c7e8061a03825c224197` |
| `factor_preprocess/adapters/__init__.py` | `ba37f7ae44f9dc1984ca159fb788ed0159503ea7e324a4325ff31f5f97db6d6a` |
| `factor_preprocess/adapters/_data_access_impl.py` | `81d88b87b3a14b93afae6a975ee2d852d28f81b115d099d248387b681270e1ea` |
| `factor_preprocess/adapters/data_access.py` | `4bb432d31734d8b16062d1e36010082d829388ca60916c4e0dd0db6b8992e71f` |
| `factor_preprocess/adapters/ewma_full_replay.py` | `f5d653976ab97b8581d299dddbd49494e65639cba164ee4cea722f30a74d5fb5` |
| `factor_preprocess/adapters/factor_assets.py` | `490a6d44a56e71ac9b039f871837c410af9b3f78ab2dd4fd02df95b0eed4aa9b` |
| `factor_preprocess/adapters/fe_operator.py` | `218862f6ac98098b14f611aab9a9644fc6d1dfa6646cac902b5de28b443f8a73` |
| `factor_preprocess/adapters/fitted_recipe.py` | `f6e770d3b0ad1090b52ce527169005b4ca7f06b58b06462c35db06e9c2c42e2e` |
| `factor_preprocess/backends/__init__.py` | `764cb2de6fa732845a8040a4910996e1c0c19c210e659b9d9688100589ced7cc` |
| `factor_preprocess/backends/polars_backend.py` | `e96429361fbae7b5f0d69b23e7e2bfeb7b77e77078a0f8c84f80cd993d8de22a` |
| `factor_preprocess/backends/registry.py` | `08418aca69b9d69bac44ca9397c105baf6e07f5c113236186cdb6f7d7d1cdf32` |
| `factor_preprocess/backends/selector.py` | `4ceaaf71031a1974730629aed885e87fadade578e6334a677d30fda0adb12fc5` |
| `factor_preprocess/contracts/__init__.py` | `f52df967671f0845fc2bda361f01cb2cfdb242d7d8f2453a717014b2722b22ac` |
| `factor_preprocess/contracts/_deep_freeze.py` | `9c4aaca5894bbca82e1a6904f447cd24367ca779e131970043a0d33777179b40` |
| `factor_preprocess/contracts/factor_profile.py` | `13fdd015e2aa2dfbc0985711e67fc8f360713902af3bd14c13ef32028591bb3a` |
| `factor_preprocess/contracts/feature_bundle.py` | `f999cb87e5bd80f472e4dde1da88335c307d5ec94a0e979ba31ef191e15bc96d` |
| `factor_preprocess/contracts/fit_apply.py` | `082144c1b6d2db6a6e6fcce5608c610299effbfb82b145de0b5d0cc87fad5711` |
| `factor_preprocess/contracts/lineage_policy.py` | `4753ef544ad22c808ee6fb971d38b94fb625352bb0e32d053808845e85848cc0` |
| `factor_preprocess/contracts/policy.py` | `cf43779e26d6a6cfc40ddd7e1aa1f9a4b37f0bb53071263611a572f9da3d0f5f` |
| `factor_preprocess/contracts/state.py` | `06e1a99e68b64ff3156f7c3269a145c6a120fad5e76e29233ebed004f651e181` |
| `factor_preprocess/contracts/treatment_lineage.py` | `ff2959d7c94a85d074be84fc0597261dcca21903f3459d65fc905f3895fda713` |
| `factor_preprocess/contracts/treatment_recipe.py` | `3e0e68d11eeb519dc53a9dff373c737721e59222ec26b65df9a4d9bbe08c32c6` |
| `factor_preprocess/contracts/treatment_spec.py` | `52decec3275815743bb29f12f84c8bdb2e52f5d78e577777f8d4b8358eafac6f` |
| `factor_preprocess/eligibility/__init__.py` | `8518c047fb259bff9903eef05e374b97bde20a705e0350609ae96c5be09ca3dd` |
| `factor_preprocess/eligibility/engine.py` | `56b2717e2114719f56cbdd4d0ec95b84e0b20d8509d2134be832337d2e7655f1` |
| `factor_preprocess/eligibility/rules.py` | `cfacc842a38d6bb577902a578342564c4202736c46ea4b2d305a279b2543891a` |
| `factor_preprocess/errors.py` | `cd0bdb8b6fa31096b849484d19df8eb3ce1d532f3b2dfa32e6e2a9cf6d0939fd` |
| `factor_preprocess/grammar/__init__.py` | `47bff6d9a866a17cc553c615db7e085997e3c7344cf10d3246b4afaa3a15e52f` |
| `factor_preprocess/grammar/search_grammar.py` | `0b94c7426eae3690f40bbf86019d1c658433e53b23f08e2a2545dff3c72528be` |
| `factor_preprocess/kernels/__init__.py` | `4d8fed12660a5de3d88fad34b29adc2b35f95699eac6a372ae508fd394c9c274` |
| `factor_preprocess/kernels/fast.py` | `9fa4d829ef61d238df56fd650aa5c78d1e647a83594bd851bca2a9f2730602c0` |
| `factor_preprocess/kernels/numba_transforms.py` | `476cd86d29d41dfc05164d4ed433a34a06be5877c4d0cf85fb50a5d870d58700` |
| `factor_preprocess/kernels/reference_bridge.py` | `c30a9e53039216c5c94063fb5b31283611999ed2374bf7c93fcf31061f63b44a` |
| `factor_preprocess/neutralization/__init__.py` | `3fab35cba90c4caa4a4bfac7f1488e0079f2676220f0fd67b0a03b7ec7fe98c6` |
| `factor_preprocess/neutralization/advanced/__init__.py` | `17838b24dca4048c6b2677a1e36e0b48d3ae309d776cf00fb12949f81a45901e` |
| `factor_preprocess/neutralization/advanced/kernel_regression.py` | `fb1530e59e3167d9dd20a8de28c29c93fd462de100b4fe72bd9360d685a40bb8` |
| `factor_preprocess/neutralization/advanced/pca_neutralization.py` | `bf05f007753f101b350a9958f5f53615b15658a31d996623ecfcb532d7dc4fc5` |
| `factor_preprocess/neutralization/advanced/quantile_regression.py` | `2df9ec7d56a31dfa51c90e0c77e5b8fd860b17469ab57cd6f12355bad4ac81a6` |
| `factor_preprocess/neutralization/advanced/robust_regression.py` | `4bf6e81f1e4ab5d3fbc4187749cdbeb4a31c926f30b184c255ad1405ab001316` |
| `factor_preprocess/neutralization/diagnostics.py` | `03bd20c0d96540ec980b867bbf9312f4163d0ad3295bb89546f3c55450495487` |
| `factor_preprocess/neutralization/diagnostics_artifact.py` | `f2bfb6b5a11f513162e1feadf23b6c80f1249759d404227bbeea73b76d64e2e0` |
| `factor_preprocess/neutralization/ols.py` | `05595b174c9fe6eddd4ddaae6f6cf98ad5d702d85224bd0903f3f13e2cae67e2` |
| `factor_preprocess/neutralization/regularized.py` | `addbde6651b22f7fa5c29342a6d33728221323fd21b3c988930e4f84028b9998` |
| `factor_preprocess/neutralization/spec.py` | `75f210fa4ef2e1c0a2145cd9a8fd341915e9a08b570a47f7dd35e69a2d8e16e1` |
| `factor_preprocess/regime/__init__.py` | `d6ab83b45ce1968507568b0f12c44ec555e62a7daa58429715ba315a0a2b107d` |
| `factor_preprocess/regime/adaptive_weights.py` | `d9afad769e0ecd98650c72e1d57d86c8258e713ec01252791847c3c6c5883519` |
| `factor_preprocess/regime/causal_detector.py` | `09bf81ebe35ea4c0c1e63b3b0bbcd94682d468a33c845a8ebdecb0820329091e` |
| `factor_preprocess/regime/detector.py` | `f1d803b5f87c942b83e72c699d10429324c3feeda5544fc0169d0586b62553f1` |
| `factor_preprocess/regime/switching.py` | `a6114ab07de853206d2d705d79946a162efb5d8cb04a8c7352e87ca5e5c8c43a` |
| `factor_preprocess/registry/__init__.py` | `f7d6d9c9d84116aa51d6c7d38695d2b439f63c69cb01d54327a998d970936329` |
| `factor_preprocess/registry/policies.py` | `6f0c776875e25205ce2e8f072dfdb70ae15caacdb6f9dc516a6151cace7fd202` |
| `factor_preprocess/registry/transforms.py` | `539102cf0ffc6280b8f7ce450c07b2f4cf955524cd97102567702bb9d0ed753a` |
| `factor_preprocess/representation/__init__.py` | `5b62691fd8b68400db3bcc025ec528ed68240d3dd127d5e036fe3e08fb58942d` |
| `factor_preprocess/representation/linear_ready.py` | `c0f394448c6be34cc02bf83000e182da18d4f022fe83c82cd524b5582f20c81e` |
| `factor_preprocess/representation/multichannel.py` | `f9edaba509c7bf1c12a1534eb9ca758e523f7efc62f955e09b8609bc6c346dfe` |
| `factor_preprocess/representation/neural_ready.py` | `945eecd9546ce434a17b42d540d1129889d1fbdfcd125d459127fd65b6bedaba` |
| `factor_preprocess/representation/policy.py` | `fdb0967671f949a93a8a7e59853530b35d2b5917213c53c256dba6dbc23de386` |
| `factor_preprocess/representation/tree_ready.py` | `6362ac24a6a3916733652199bfcb3b47096e2b80aa53ffd2b1633ae4ec80c659` |
| `factor_preprocess/transforms/__init__.py` | `6b7f63f73848b60b67fff56a3b414a62c7c8019def1c110cf9029dd4d34dddec` |
| `factor_preprocess/transforms/cross_sectional.py` | `6c145d736f221f2eb92055768537350b99197d55fad86d6ef0be32559f33899e` |
| `factor_preprocess/transforms/decomposition/__init__.py` | `089d5af717071ab530734b69b74a98ccfec25ad6557b58cd1b4b7e5abdcb1f11` |
| `factor_preprocess/transforms/decomposition/cycle.py` | `0a0c3c5493af64c178ab556776dfd1d0ae36625cfc445c0fba012ee618a0482e` |
| `factor_preprocess/transforms/decomposition/seasonal.py` | `a7ed56c6b957f742d000e195f58ac031053562a3cdbfaa568a8aa598d8744874` |
| `factor_preprocess/transforms/decomposition/trend.py` | `9638c320aa03f3c02247d1c221891e0154ce76336c5ecc1b726f7a753e31dec0` |
| `factor_preprocess/transforms/decomposition/wavelet.py` | `b746ea2d02d1589735f3afe1ee6b1f1a1bcdc3f81d8f907a91941560edd29da1` |
| `factor_preprocess/transforms/event_decay.py` | `f384f17802b5ed0c0b6de29bc21c2addc0b64d4d9f1f75de6773d738842ba13d` |
| `factor_preprocess/transforms/freshness.py` | `cb62293004c74338f715e75a91f849a4b2b3b4d6f11d2d258ddfcbcf3b97e69a` |
| `factor_preprocess/transforms/missingness.py` | `42dcbf19613e38f7be1e18283ca4a67e208332b88a6a0fe625b6a86c0724ea4d` |
| `factor_preprocess/transforms/repair_shapes.py` | `dccea81e1c00e68e144f530eb103e7c217677cc1ae82c5bd5b3ee12695cacea5` |
| `factor_preprocess/transforms/rolling.py` | `e7b9cc19a0e32c62b2f2d1d216e998ed49c72a3579404efd735f25f1c04e288b` |
| `factor_preprocess/transforms/smoothing.py` | `5a8acbd7530a05945c00a21ab306e3ff1c1bb3c5e10babf051a9ed36f374697f` |
| `factor_preprocess/transforms/treatment_variants.py` | `5d3a3dd9c7c0e014be750edcf9e37c628113acd17d8ec3dc09076e13aa3b2b4b` |
| `factor_preprocess/transforms/volatility.py` | `5361592b9fc419a82fb6201b9bf0ce68ff81c48adc4f34762af411bd6e4b5d12` |

</details>

重新生成：`python scripts/build_api_reference.py`；检查：`python scripts/build_api_reference.py --check`。
