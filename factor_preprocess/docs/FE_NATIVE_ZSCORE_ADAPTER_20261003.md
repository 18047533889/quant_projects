# FE 原生分组 z-score 的可选适配器

这是显式可选入口，不改变 `cs_zscore`、registry、recipe 或 optimizer 的默认执行器，
也不代表宽表 FE 算子已取得生产准入。完整功能域和大规模性能仍须分别验证。

```python
import numpy as np
from factor_preprocess.adapters.fe_zscore import (
    cs_zscore_finite_anchor_fe_native, execution_identity,
)

# 日期 × 股票；每个日期独立标准化。三维数组也可以指定 axis=(0, 2)。
values = np.array([[1.0, 3.0, 5.0], [2.0, np.nan, 6.0]])
result = cs_zscore_finite_anchor_fe_native(values, axis=-1, ddof=0.5)
identity = execution_identity()
assert result.shape == values.shape
assert identity["production_admitted"] is False
```

## 实际执行链

适配器把非归约轴排列在前、归约轴排列在后，为每个归约组生成整数 group ID。
随后直接调用 `factor_engine.backend.long_stable_zscore.finite_anchor_centered_zscore_long`，
最后恢复原轴和原顺序。NumPy 只负责转换、排列与还原数组；标准化的统计计算
由 FE 的 Polars 分组表达式完成。没有 Pandas 宽表 pivot，也没有 FP 数值 fallback。
FE 或 Polars 缺失时不会假装已调用 FE。

## 已支持的候选契约

- 输入为可转换为 Float64 的实数数值数组；复数、object、字符串拒绝。
- `axis` 支持整数、无重复整数的 tuple、`None`；空 tuple 表示不归约轴。
  布尔轴拒绝。输出保持输入形状；不修改只读输入或共享 view。
- 本适配器限定 `ddof` 为有限实数且在 `[0, 1]`，包含小数。
  FE grouped-long helper 本身允许更宽的非负实数域；宽表候选仍只允许整数。
- 只支持 `numeric_policy="finite_anchor_centered_v2"`。
  legacy policy 必须显式使用原 FP 执行器，不能以 FE 身份执行 FP fallback。
- `max_chunk_cells=1_000_000` 限制每次 FE 长表调用的行数。
  一个完整归约组超过上限时明确拒绝，不拆组改变统计口径。
  分块直接从原数组的轴坐标取值，不为排列后的整个因子立方体创建密集副本。
- `max_result_bytes=256 * 1024**2` 限制最终 Float64 输出大小。
  预算检查在输出分配前执行；过大批次应由调用方按因子切分。
  显式传 `None` 可取消结果预算，但调用方须自行保障内存。
  分块上限是逻辑行数限制，不是硬 RSS 上限，Polars 中间列和输入转换也占内存。
- NaN 保持缺失。含 Inf 的组对非缺失位置使用有限常数回退。
  常数组、有限样本数不大于 `ddof` 的组也使用常数回退。

## 有限输入的计算公式

令一组有限值为 $x_i$，数量为 $n$。取
$a=\min(x)/2+\max(x)/2$，$s=\max_i|x_i-a|$，
$u_i=(x_i-a)/s$，$c_i=u_i-\frac1n\sum_j u_j$。
非退化情形的输出为

$$z_i=\frac{c_i}{\sqrt{\sum_j c_j^2/(n-\mathrm{ddof})}}.$$

这与通常的中心化 z-score 公式等价，但避免直接对极大数求和或平方造成溢出。
退化、缺失和 Inf 情形按上面的显式规则处理，不在公式中隐式忽略。

## 身份与验证边界

`execution_identity()` 包含适配器和 FE helper 两个完整模块的有界源码摘要，
以及 NumPy、Polars 运行版本。摘要记录磁盘源码，不证明实际已加载代码的传递闭包。
不可把此候选身份改写成已准入的 FE operator 身份。

已有定向测试覆盖独立 Decimal 参考、轴恢复、极端值、小数自由度、缺失/Inf、
实际 FE 调用检查及禁止 FP fallback。定向通过不等于多年全量股票性能最优；
默认切换前仍需测量长表搬运、峰值内存和完整请求 A/B，而非只计核心公式时间。

## 显式通过 TreatmentRecipe 编译执行

可选的 `compile_fe_native_zscore_recipe` 接口把已有 recipe 验证与 FE 执行连接起来。
它先调用原来的 `recipe.compile(registry)`，保留 semantic、stage、参数域和 fit 校验；
随后为支持的步骤生成单独的候选执行计划。它不会修改 registry 的默认路由或
把 `cs_zscore` 的 `FP_NATIVE` 标记改写为已生产准入的 `FE_OPERATOR`。

```python
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from factor_preprocess.adapters.fe_zscore_recipe import compile_fe_native_zscore_recipe

registry = get_default_registry()
metadata = registry.get("cs_zscore")
recipe = TreatmentRecipe(
    recipe_id="example-fe-zscore",
    source_factor_definition_ref="your-factor-definition-ref",
    source_factor_value_ref="your-factor-value-ref",
    ordered_steps=(RecipeStep(
        step_id="normalize", semantic_transform_id=metadata.semantic_id,
        implementation_ref="cs_zscore", stage=metadata.stage,
        parameters={"axis": -1, "ddof": 0.5,
                    "numeric_policy": "finite_anchor_centered_v2"},
    ),),
)
plan = compile_fe_native_zscore_recipe(recipe, registry,
    max_chunk_cells=1_000_000, max_result_bytes=256 * 1024**2)
result = plan.run(values)  # values 是运行时输入，不放进 recipe.parameters。
assert plan.spec_identity == recipe.spec_identity
print(plan.execution_identity)
```

当前候选只支持一个无 fitted state 的 `cs_zscore` 步骤，EXPANDING fit boundary，
且不接受额外 neutralization spec 或 existing treatment signature；不支持的配置
明确拒绝，不会忽略。这里只是无拟合统计的显式执行候选，不是完整多步骤 recipe
的 FE 准入方案，也不自动发布结果或替代 materialization identity。

recipe 的 `spec_identity` 描述处理规范；候选 `execution_identity` 则另行记录
完整绑定的默认/显式参数、资源上限、适配器/FE helper/recipe 绑定模块的磁盘源码摘要
及运行版本。身份摘要不是已加载代码的证明，也不覆盖传递依赖闭包。
在更新实现或运行环境后应重新编译计划，不应把旧编译身份当成当前运行证明。
已有 recipe 回归使用独立 Decimal 计算对照，多轴、小数自由度、NaN/Inf、常数组、
只读输入、预算拒绝及禁止 FP 数值 fallback 均在验证范围内。

此 ndarray 候选只验证数值轴，不知道哪一维是时间；“无需拟合”不等于任何轴选择
都因果安全。日期 × 股票输入应按股票轴归约；不能把跨未来日期或跨 TRAIN/TEST 的
全样本归约作为线上预处理。`FitBoundary.EXPANDING` 在这里是 recipe 的配置约束，
不是对没有时间坐标的 ndarray 自动施加的分割器。生产准入前仍需要带时间/样本边界
的完整 pipeline 验证。

## 通用配方参数与运行时输入隔离

`TreatmentRecipe.compile` 会在原有参数域校验之前检查函数签名，拒绝把第一个
固定位置参数（运行时因子输入）写入 `RecipeStep.parameters`。例如
`func(values, data=None)` 只禁止配置 `values`，允许合法的 `data` 配置；
`func(x, axis=-1)` 禁止配置 `x`，仍允许 `axis`。此保护也覆盖位置专用输入参数。
没有固定位置参数的可变关键字函数保留 `values`、`x`、`data` 为输入别名；
无法确定输入边界的签名明确拒绝。保护不改变原来的 semantic、stage、fit、
参数域或执行协议，也不替代处理步骤的因果性检查。
