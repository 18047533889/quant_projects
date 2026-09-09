# riskfolio_qs v0.2 冻结参数规范

## 1. 文档目的与适用范围

本文定义 v0.2 一键组合优化的参数冻结规则，目标是保证：

1. 同一优化器名称在同一参数版本下行为稳定。
2. 用户可覆盖的参数边界清晰。
3. 回测、仿真、生产可复现。

本文适用于基础默认优化器集合：

1. minvar_enhance_index
2. meanvar_enhance_index
3. meanvar_absolute_return
4. topn_long_only_equal_weight（兜底）

## 2. 术语与版本规则

术语：

1. 冻结参数：默认固定值，未显式覆盖时必须使用。
2. 严格参数：不允许业务侧覆盖。
3. 可覆盖参数：允许在白名单范围内调整。
4. 参数版本：参数快照版本号，格式建议 vMAJOR.MINOR.PATCH。

版本规则：

1. 修改严格参数默认值：MAJOR+1。
2. 新增可覆盖参数：MINOR+1。
3. 文档纠错不改语义：PATCH+1。

## 3. 参数分层定义

参数分三层：

1. Global 层：所有优化器共享。
2. Optimizer 层：某优化器独有默认值。
3. Runtime 层：运行期覆盖值（仅限白名单参数）。

优先级：

1. Runtime 覆盖（白名单内）
2. Optimizer 默认
3. Global 默认

## 4. 全局默认参数

| 参数名 | 默认值 | 单位 | 类型 | 可覆盖 |
|---|---:|---|---|---|
| budget | 1.0 | weight | float | 否 |
| long_only | true | - | bool | 否（基础版） |
| gross_exposure_cap | 1.0 | weight | float | 否 |
| rebalance_frequency | D | enum | string | 否 |
| risk_mode_default_index | barra_factor | enum | string | 否 |
| risk_mode_default_abs | historical_cov | enum | string | 否 |
| fallback_policy_default | fail_fast | enum | string | 否 |

## 5. 合约模式参数冻结表

### 5.1 minvar_enhance_index

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---:|---|---|
| objective_id | OBJ_MINVAR_ACTIVE | string | 否 |
| single_name_max | 0.03 | float | 是 |
| active_weight_abs_max | 0.02 | float | 是 |
| turnover_cap | 0.20 | float | 是 |
| linear_cost_penalty | 1.0 | float | 否 |
| impact_cost_penalty | 0.0 | float | 是 |
| industry_neutral_mode | strict | string | 否 |
| style_neutral_mode | band | string | 否 |
| style_band | 0.10 | float | 是 |
| mcap_neutral_mode | band | string | 否 |
| mcap_band | 0.10 | float | 是 |

### 5.2 meanvar_enhance_index

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---:|---|---|
| objective_id | OBJ_MEANVAR_ACTIVE | string | 否 |
| alpha_weight | 1.0 | float | 是 |
| risk_weight | 1.0 | float | 是 |
| single_name_max | 0.03 | float | 是 |
| active_weight_abs_max | 0.02 | float | 是 |
| turnover_cap | 0.20 | float | 是 |
| linear_cost_penalty | 1.0 | float | 否 |
| impact_cost_penalty | 0.0 | float | 是 |
| industry_neutral_mode | strict | string | 否 |
| style_neutral_mode | band | string | 否 |
| style_band | 0.10 | float | 是 |
| mcap_neutral_mode | band | string | 否 |
| mcap_band | 0.10 | float | 是 |

### 5.3 meanvar_absolute_return

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---:|---|---|
| objective_id | OBJ_MEANVAR_ABS | string | 否 |
| alpha_weight | 1.0 | float | 是 |
| risk_weight | 1.0 | float | 是 |
| single_name_max | 0.05 | float | 是 |
| turnover_cap | 0.25 | float | 是 |
| linear_cost_penalty | 1.0 | float | 否 |
| impact_cost_penalty | 0.0 | float | 是 |

### 5.4 topn_long_only_equal_weight

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---:|---|---|
| objective_id | OBJ_TOPN_LONG_ONLY | string | 否 |
| topn_n | 50 | int | 是 |
| topn_weight_mode | equal | string | 否 |
| single_name_max | 0.05 | float | 否 |

## 6. 目标函数模式默认参数

当 mode=objective 时，默认参数：

| objective.type | 默认 alpha_weight | 默认 risk_weight | 默认 risk_mode |
|---|---:|---:|---|
| mean_variance | 1.0 | 1.0 | historical_cov |
| min_variance | 0.0 | 1.0 | historical_cov |
| index_enhancement | 1.0 | 1.0 | barra_factor |

说明：

1. objective 模式默认不继承 preset 专属约束模板。
2. 仅继承 Global 层与 objective.type 对应默认值。

## 7. 风险相关参数

| 参数名 | 默认值 | 说明 | 可覆盖 |
|---|---|---|---|
| risk_mode | 跟随优化器 | barra_factor/historical_cov/none | 否 |
| barra_cov_lookback_days | 252 | 从 F_ret 派生 Sigma_f 的窗口 | 是 |
| specific_risk_lookback_days | 252 | 从 F_spec 派生 D 的窗口 | 是 |
| cov_shrinkage | 0.05 | 协方差收缩强度 | 是 |

限制：

1. 使用 barra_factor 时，F/F_ret/F_spec 缺失即 fail_fast（基础版）。

## 8. 中性约束参数

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---|---|---|
| industry_neutral_mode | strict | enum(strict/band/off) | 否（基础版） |
| style_neutral_mode | band | enum(strict/band/off) | 否 |
| style_band | 0.10 | float | 是 |
| mcap_neutral_mode | band | enum(strict/band/off) | 否 |
| mcap_band | 0.10 | float | 是 |

说明：

1. style_band 与 mcap_band 单位为“主动暴露绝对值上限”。

## 9. 交易成本与换手参数

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---:|---|---|
| turnover_cap | 0.20 | float | 是 |
| turnover_penalty | 1.0 | float | 是 |
| linear_cost_penalty | 1.0 | float | 否（基础版） |
| impact_cost_penalty | 0.0 | float | 是 |
| enable_impact_cost | false | bool | 是 |

说明：

1. impact_cost_penalty > 0 时建议同时 enable_impact_cost=true。

## 10. 可交易与资产池参数

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---|---|---|
| enforce_tradable_mask | true | bool | 否 |
| allow_new_asset_with_zero_prev | true | bool | 是 |
| min_alpha_coverage | 0.80 | float | 是 |
| min_benchmark_coverage | 0.95 | float | 是 |

说明：

1. 覆盖率低于阈值时按 fallback_policy 执行。

## 11. 求解器参数

| 参数名 | 默认值 | 类型 | 可覆盖 |
|---|---|---|---|
| backend | riskfolio_backend | string | 否 |
| solver_name | CLARABEL | string | 是 |
| max_iters | 5000 | int | 是 |
| solver_tol | 1e-6 | float | 是 |

规则型优化器（topn_long_only_equal_weight）固定：

1. backend=rule_backend
2. solver_name=none

## 12. 参数覆盖规则

覆盖白名单（基础版）：

1. single_name_max
2. active_weight_abs_max
3. turnover_cap
4. turnover_penalty
5. style_band
6. mcap_band
7. alpha_weight
8. risk_weight
9. impact_cost_penalty
10. enable_impact_cost
11. solver_name
12. max_iters
13. solver_tol

覆盖规则：

1. 非白名单参数出现覆盖请求时直接失败。
2. 覆盖值先过参数校验，再写入 resolved_config。

## 13. 参数校验规则

通用校验：

1. single_name_max in (0, 1]
2. active_weight_abs_max in [0, 1]
3. turnover_cap in [0, 1]
4. style_band, mcap_band >= 0
5. alpha_weight, risk_weight >= 0
6. max_iters > 0
7. solver_tol > 0

一致性校验：

1. long_only=true 时不得启用净空头约束模板。
2. rule_backend 时不得传 solver_name != none。

## 14. 参数变更治理

1. 任何默认值变更必须更新参数版本号。
2. 变更需记录：变更项、原因、影响范围、回归结果。
3. 生产环境仅允许使用 active 状态参数版本。

## 15. 审计与回放要求

每次运行必须落盘：

1. resolved_config.yaml
2. resolved_params.yaml
3. parameter_version
4. 覆盖参数差异 diff

metadata 必填：

1. optimizer_name
2. parameter_version
3. fallback_used
4. data_version_hash

## 16. 附录 A：参数字段字典

最小字段模板：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| param_name | string | 是 | 参数名 |
| param_value | scalar | 是 | 参数值 |
| source_level | enum | 是 | global/optimizer/runtime |
| is_overridden | bool | 是 | 是否运行期覆盖 |
| parameter_version | string | 是 | 参数版本 |

## 17. 附录 B：参数模板示例

```yaml
parameter_version: "v0.2.0"
global:
  budget: 1.0
  long_only: true
  gross_exposure_cap: 1.0

optimizers:
  minvar_enhance_index:
    objective_id: OBJ_MINVAR_ACTIVE
    single_name_max: 0.03
    active_weight_abs_max: 0.02
    turnover_cap: 0.20
    linear_cost_penalty: 1.0
    impact_cost_penalty: 0.0
    industry_neutral_mode: strict
    style_neutral_mode: band
    style_band: 0.10
    mcap_neutral_mode: band
    mcap_band: 0.10

  meanvar_enhance_index:
    objective_id: OBJ_MEANVAR_ACTIVE
    alpha_weight: 1.0
    risk_weight: 1.0
    single_name_max: 0.03
    active_weight_abs_max: 0.02
    turnover_cap: 0.20
    linear_cost_penalty: 1.0
    impact_cost_penalty: 0.0

overrides_whitelist:
  - single_name_max
  - active_weight_abs_max
  - turnover_cap
  - style_band
  - mcap_band
  - alpha_weight
  - risk_weight
  - impact_cost_penalty
  - solver_name
  - max_iters
  - solver_tol
```
