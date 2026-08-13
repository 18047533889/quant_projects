# 08 — Integration Contracts & Public APIs

## 8.1 原则

Contract 是跨库边界，不是内部实现模型。

- schema versioned
- backward compatibility 明确
- private class 不跨包
- 大矩阵走 Arrow/NumPy/Polars/value ref，不走巨大 JSON

## 8.2 核心 Contract

### FactorDefinitionRef

- factor_id
- canonical_repr/hash
- source definition ref
- frequency
- lookback metadata
- required semantic fields/domains
- timing/availability metadata（由 FE/DA 提供）

### FactorBatch

- factor IDs
- date/asset axis
- values
- validity
- layout/dtype
- value_ref/hash optional

### EvaluationRequest

- metrics/preset
- slice/groupby
- horizon/label spec
- context refs
- tier/cost budget

### EvaluationBundle

- metric results
- grouped/sliced results
- FactorDiagnosis
- warnings
- metric versions/config
- context/data hashes
- optional large-series refs

### CandidateMutation

- mutation ID/spec/version
- parent IDs
- parameters
- mechanism hypothesis
- expected signatures
- complexity estimate

### FactorAsset / FactorSet

- identity/lineage/evidence refs/memberships/lifecycle
- selected features / aggregation specs

### PreprocessingPolicy

- ordered transforms
- stateless/fitted flags
- train-fit semantics
- exposure context requirements
- output channels

### FeatureBundle

- feature matrix/ref
- feature IDs/source factor IDs
- channel types
- fitted state refs
- timing/missingness metadata

## 8.3 Adapter Pattern

每个 package optional adapter：

- `quant_evaluator.adapters.dataaccess`
- `quant_evaluator.adapters.factor_engine`
- `factor_optimizer.adapters.factor_engine`
- `factor_optimizer.adapters.quant_evaluator`
- `factor_assets.adapters.quant_evaluator`
- `factor_preprocess.adapters.factor_assets`（若需要）

核心 package import 不应因 optional dependency 缺失而失败。

## 8.4 Error Taxonomy

跨包错误至少标准化：

- SchemaVersionError
- MissingInputError
- TimingContractError
- UnsupportedMetric/Transform/Mutation
- InsufficientObservations
- NumericalFailure
- OptionalDependencyMissing

不要用裸 `ValueError("bad")` 作为跨包接口。

## 8.5 Versioning

- public API semver
- schema_version 独立
- metric/transform/mutation 各自有 version
- migration path 由 tests 验证

修改数学语义必须 bump metric version，即使 Python 函数名不变。
