# FactorEngine 字段、算子、配方与研究工具分层

## 1. 字段（Field）

字段由数据层直接提供，不接受计算参数。`vwap` 是字段，不是算子。字段权威清单由 `canonical_data_fields.json` 生成到 `field_manifest.json`。

## 2. 生产基础算子（Primitive Operator）

基础算子是参数明确、可复用、输出保持 panel 形状的计算节点。生产 registry 只包含元素级、滚动时序、横截面、分组、PIT 报告期和少数必要内部节点。

编译器内部可以使用 `protected_div` 等 lowering 节点，但这些节点属于 `internal` surface，不提供给因子作者或 LLM。

## 3. 复合指标（Fused Composite）

只保留需要递归平滑、复杂边界或共享中间量的少数指标：

- `MACD_line` / `MACD_signal` / `MACD_hist`
- `RSI_WILDER`
- `ATR_WILDER`
- `ADX`

简单技术指标不占 operator canonical，迁入 `FactorRecipeRegistry`。

## 4. 因子配方（Factor Recipe）

配方是由 production primitive 展开的经济含义表达式，包括布林带、动量、随机指标、隔夜缺口、波动率、财务比率、盈利能力、成长、应计等。权威清单生成到 `factor_recipe_manifest.json`。

`factor_recipes/compiler.py` 提供安全 AST 展开、递归 recipe 展开、循环依赖检测和确定性 DAG 编译。批量编译多个因子时，相同的 `ts_mean`、`ts_std`、`ts_ema`、回归等子表达式只保留一个节点；执行器按节点缓存结果，避免重复 rolling 和 materialize。

配方不会与字段重名，也不会以另一份重复实现进入 `OperatorRegistry`。

## 5. 研究工具（Research Tool）

统计检验、概率分布、矩阵、频域、信号处理、泛化模型、无锚点 expanding/cumulative、模糊聚合和宽松数据修复迁入 `ResearchToolRegistry`。它们可用于研究分析，但不进入生产 DSL、operator catalog 或 LLM authoring surface。

## 6. 基本面 PIT 契约

财报事件必须至少保留：

- `instrument`
- `period_end`
- `available_at`
- `revision_id`（存在修订时）

日频拼接必须满足 `available_at <= decision_timestamp`，不得先按报告期去重成最终修订值。`pit_contract.py` 提供验证、backward as-of join 和最大陈旧期限控制。

旧的 `ttm`、`quarter`、`yoy`、`avg2` 已删除。生产公式必须使用 `period_lag`、`period_change`、`period_average`、`period_cagr`、`quarter_from_cumulative`、`ttm_from_quarterly`、`ttm_from_cumulative` 和 `yoy_by_period`。

## 7. 状态型算子 checkpoint

`ts_ema`、Wilder RSI/ATR、ADX 和 MACD 家族具有递归状态。分段或增量执行不得把切片首行当作新的全历史起点。

`stateful_contract.py` 为每个状态型算子声明：

- 状态 schema 和语义版本；
- 最小历史长度；
- 必须持久化的状态字段；
- 缺失值策略和依赖；
- 分段执行是否必须提供 checkpoint。

checkpoint 同时绑定 instrument、`as_of`、输入数据指纹和语义版本。版本、标的或输入指纹不一致时拒绝恢复。权威清单生成到 `stateful_operator_manifest.json`。

## 8. 使用频率与下线审查

`operator_usage.py` 静态扫描生产因子 manifest，解析 DSL 调用并解析 alias，生成 `operator_usage_report.json`。

只对少数 fused composite 执行90天审查：

- 90天内有使用：`retain_active`；
- 有历史使用但超过90天：`review_dormant`；
- 从未使用：`review_unused`；
- 缺少时间戳：`review_missing_timestamp`。

报告使用 manifest 中最新时间戳作为确定性参考时间，不依赖 CI 运行当天，因此重复生成不会漂移。

## 9. CI 强制规则

- `field_manifest ∩ operator_manifest = ∅`
- `vwap` 只能出现在字段清单
- recipe replacement 必须存在声明
- production recipe 必须可安全展开和编译
- 公共子表达式必须由 DAG 节点复用
- research 工具不得进入 OperatorRegistry
- 状态型分段执行必须具有匹配 checkpoint
- 四层 manifest、状态 manifest 和使用报告必须可重复生成且无 diff
- 重复 `canonical + backend` 注册在 bootstrap 完成后默认报错
- 显式覆盖必须提供 `replace=True` 和 `replacement_reason`
- 所有注册的 Polars backend 不得通过 Pandas bridge 实现
