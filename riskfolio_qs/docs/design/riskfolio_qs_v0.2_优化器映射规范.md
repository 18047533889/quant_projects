# riskfolio_qs v0.2 优化器映射规范

## 1. 文档目的与边界

本文定义“优化器名称 -> 数学表达 -> 约束集合 -> 数据依赖 -> 输出字段”的统一映射规则。

本文边界：

1. 定义映射契约，不实现优化器代码。
2. 定义配置语义，不定义具体参数取值（参数值由冻结参数规范管理）。
3. 所有数学对象引用以纯净数学定义为准。

关联文档：

1. riskfolio_qs_v0.2_纯净数学定义.md
2. riskfolio_qs_v0.2_数据链路初版.md
3. riskfolio_qs_v0.2_冻结参数规范.md

## 2. 映射总览

优化器映射统一对象：

1. optimizer_name：对外暴露名称。
2. objective_id：目标函数模板标识。
3. hard_constraint_set：硬约束集合标识。
4. soft_constraint_set：软约束集合标识。
5. risk_mode：风险项口径（barra_factor 或 historical_cov）。
6. required_inputs：最小输入清单。
7. fallback_policy：缺失输入时行为。

推荐首批优化器名称：

1. minvar_enhance_index
2. meanvar_enhance_index
3. topn_long_only_equal_weight
4. meanvar_absolute_return

说明：topn_long_short_equal_weight 作为可选扩展优化器，不纳入基础默认启用集。

统一映射主表（示例）：

| optimizer_name | objective_id | hard_constraint_set | soft_constraint_set | risk_mode | fallback_policy |
|---|---|---|---|---|---|
| minvar_enhance_index | OBJ_MINVAR_ACTIVE | HC_INDEX_ENHANCE_CORE | SC_TC_TURNOVER | barra_factor | fail_fast |
| meanvar_enhance_index | OBJ_MEANVAR_ACTIVE | HC_INDEX_ENHANCE_CORE | SC_TC_TURNOVER | barra_factor | fail_fast |
| topn_long_only_equal_weight | OBJ_TOPN_LONG_ONLY | HC_TOPN_LONG_ONLY | SC_NONE | none | degrade_to_rule |
| meanvar_absolute_return | OBJ_MEANVAR_ABS | HC_LONG_ONLY_CORE | SC_TC_TURNOVER | historical_cov | fail_fast |

## 3. 标准数学表达引用关系

目标函数模板引用规则：

1. OBJ_MINVAR_ACTIVE 对应最小主动风险。
2. OBJ_MEANVAR_ACTIVE 对应主动收益-风险权衡。
3. OBJ_MEANVAR_ABS 对应绝对收益 mean-variance。
4. OBJ_TOPN_LONG_ONLY/OBJ_TOPN_LS_EQUAL 对应规则型组合。

统一写法：

1. 相对收益模板使用主动权重 $a = w - w^b$。
2. Barra 风险口径使用外采 $F, F_{\mathrm{ret}}, F_{\mathrm{spec}}$。
3. 市值约束口径使用 $F_{\mathrm{mcap}}$。

引用规范：

1. 数学公式不在本文件重复推导，统一引用纯净数学定义章节号。
2. 本文件只定义“哪个 optimizer_name 对应哪个公式模板”。

## 4. 优化器名称命名规范

命名格式：

1. {objective}_{scenario}_{constraint_profile}

命名约束：

1. 小写下划线。
2. 不包含具体参数值。
3. 名称语义稳定，参数变化通过版本与冻结表管理。

示例：

1. minvar_enhance_index
2. meanvar_enhance_index
3. meanvar_absolute_return

## 5. 优化器名称到目标函数映射

| optimizer_name | objective_id | 数学口径 | 说明 |
|---|---|---|---|
| minvar_enhance_index | OBJ_MINVAR_ACTIVE | 最小化主动风险项 | 指增基线模型 |
| meanvar_enhance_index | OBJ_MEANVAR_ACTIVE | 主动收益 - 风险 + 成本惩罚 | 指增主模型 |
| meanvar_absolute_return | OBJ_MEANVAR_ABS | 绝对收益 mean-variance | 非指增场景 |
| topn_long_only_equal_weight | OBJ_TOPN_LONG_ONLY | 规则型 topN 等权 | 非凸优化兜底 |

## 6. 优化器名称到硬约束映射

| hard_constraint_set | 约束清单 |
|---|---|
| HC_INDEX_ENHANCE_CORE | 预算约束、个股上下限、主动权重带宽、行业中性、风格中性、市值中性、换手上限 |
| HC_LONG_ONLY_CORE | 预算约束、个股上下限、换手上限 |
| HC_TOPN_LONG_ONLY | topN 成分约束、等权约束 |
| HC_TOPN_LONG_SHORT | 多空选股约束、净敞口与总敞口约束 |

映射表：

| optimizer_name | hard_constraint_set |
|---|---|
| minvar_enhance_index | HC_INDEX_ENHANCE_CORE |
| meanvar_enhance_index | HC_INDEX_ENHANCE_CORE |
| meanvar_absolute_return | HC_LONG_ONLY_CORE |
| topn_long_only_equal_weight | HC_TOPN_LONG_ONLY |

## 7. 优化器名称到软约束映射

| soft_constraint_set | 软约束清单 |
|---|---|
| SC_TC_TURNOVER | 线性交易成本惩罚、冲击成本惩罚（可选）、换手惩罚 |
| SC_NONE | 无软约束 |

映射表：

| optimizer_name | soft_constraint_set |
|---|---|
| minvar_enhance_index | SC_TC_TURNOVER |
| meanvar_enhance_index | SC_TC_TURNOVER |
| meanvar_absolute_return | SC_TC_TURNOVER |
| topn_long_only_equal_weight | SC_NONE |

## 8. 优化器名称到数据依赖映射

统一依赖字段：

1. 强依赖：缺失即失败。
2. 弱依赖：缺失可降级。

| optimizer_name | 强依赖 | 弱依赖 |
|---|---|---|
| minvar_enhance_index | alpha, benchmark, prev_positions, F, G, F_mcap, F_ret, F_spec | cost_impact |
| meanvar_enhance_index | alpha, benchmark, prev_positions, F, G, F_mcap, F_ret, F_spec | cost_impact |
| meanvar_absolute_return | alpha, market, prev_positions | cost_impact |
| topn_long_only_equal_weight | alpha | benchmark, prev_positions |

说明：

1. 使用 barra_factor 风险口径时，F_ret 与 F_spec 为强依赖。
2. 使用 historical_cov 口径时，market.ret_1d 为强依赖。

## 9. 缺失输入时的降级与失败策略

策略枚举：

1. fail_fast：立即失败并输出缺失清单。
2. degrade_to_rule：降级到规则型优化器。
3. disable_optional_constraints：禁用可选约束继续运行。

默认策略：

1. 指增类（minvar_enhance_index/meanvar_enhance_index）：fail_fast。
2. 规则类（topn_*）：degrade_to_rule（本身即规则）。
3. 成本增强项缺失：disable_optional_constraints。

失败输出要求：

1. 缺失字段清单。
2. 缺失日期范围。
3. 受影响 optimizer_name。

## 10. 风险项口径映射

| risk_mode | 风险定义 | 输入要求 |
|---|---|---|
| barra_factor | 以 F, F_ret, F_spec 构建因子风险口径 | F, F_ret, F_spec |
| historical_cov | 历史收益协方差口径 | market.ret_1d |
| none | 不构建显式风险项（规则模型） | 无 |

映射：

1. minvar_enhance_index -> barra_factor
2. meanvar_enhance_index -> barra_factor
3. meanvar_absolute_return -> historical_cov
4. topn_long_only_equal_weight -> none

## 11. 中性约束口径映射

中性口径：

1. 行业中性：基于 G 的主动暴露中性。
2. 风格中性：基于 F 的主动暴露中性。
3. 市值中性：基于 F_mcap 的主动暴露中性。

控制方式：

1. strict：等式中性。
2. band：带宽中性。

默认映射：

1. minvar_enhance_index：行业 strict，beta/size strict，其余风格 band。
2. meanvar_enhance_index：行业 strict，风格 band，市值 band。
3. meanvar_absolute_return：默认不启用相对基准中性。

## 12. 交易成本约束口径映射

口径：

1. 线性成本：必选软约束。
2. 冲击成本：可选软约束。
3. 换手上限：可选硬约束。

默认映射：

1. minvar_enhance_index：线性成本 + 换手上限，冲击成本可选。
2. meanvar_enhance_index：线性成本 + 换手上限，冲击成本可选。
3. meanvar_absolute_return：线性成本默认开启。
4. topn_*：默认不启用成本约束（由外层策略控制）。

## 13. 求解器后端与参数映射

后端枚举：

1. riskfolio_backend（主后端）
2. rule_backend（规则型后端）

映射：

| optimizer_name | backend | 默认求解器 |
|---|---|---|
| minvar_enhance_index | riskfolio_backend | CLARABEL |
| meanvar_enhance_index | riskfolio_backend | CLARABEL |
| meanvar_absolute_return | riskfolio_backend | CLARABEL |
| topn_long_only_equal_weight | rule_backend | none |

说明：

1. solver name 与 option 由冻结参数文档给出默认值。
2. 映射文档只定义“可用后端集合”。

## 14. 产物字段映射

每个 optimizer_name 统一输出：

1. target_positions
2. trades
3. summary
4. metadata

metadata 必填字段：

1. optimizer_name
2. objective_id
3. hard_constraint_set
4. soft_constraint_set
5. risk_mode
6. fallback_used
7. data_version_hash

## 15. 一致性与可复现要求

1. 同一 optimizer_name 在同一参数版本下不得改变数学语义。
2. 映射变更必须提升 mapping_version。
3. 运行必须落盘 resolved_mapping.yaml。
4. 任何降级路径必须在 metadata 明确记录。

## 16. 测试用例矩阵

建议最小测试矩阵：

| 用例ID | optimizer_name | 场景 | 预期 |
|---|---|---|---|
| T01 | minvar_enhance_index | 输入完整 | 正常产出 + no fallback |
| T02 | minvar_enhance_index | 缺失 F_ret | fail_fast |
| T03 | meanvar_enhance_index | 缺失 cost_impact | disable_optional_constraints |
| T04 | meanvar_absolute_return | 无 benchmark | 正常运行 |
| T05 | topn_long_only_equal_weight | 仅 alpha | 正常运行 |

测试要求：

1. 每次映射变更至少回归 T01-T05。
2. 失败用例要校验错误信息可读性。

## 17. 附录 A：优化器清单模板

模板字段：

1. optimizer_name
2. version
3. objective_id
4. hard_constraint_set
5. soft_constraint_set
6. risk_mode
7. backend
8. fallback_policy
9. status（draft/active/deprecated）

## 18. 附录 B：映射 YAML 示例

```yaml
mapping_version: "v0.2.0"
optimizers:
  - optimizer_name: minvar_enhance_index
    objective_id: OBJ_MINVAR_ACTIVE
    hard_constraint_set: HC_INDEX_ENHANCE_CORE
    soft_constraint_set: SC_TC_TURNOVER
    risk_mode: barra_factor
    backend: riskfolio_backend
    fallback_policy: fail_fast
    required_inputs:
      strong: [alpha, benchmark, prev_positions, F, G, F_mcap, F_ret, F_spec]
      weak: [cost_impact]

  - optimizer_name: topn_long_only_equal_weight
    objective_id: OBJ_TOPN_LONG_ONLY
    hard_constraint_set: HC_TOPN_LONG_ONLY
    soft_constraint_set: SC_NONE
    risk_mode: none
    backend: rule_backend
    fallback_policy: degrade_to_rule
    required_inputs:
      strong: [alpha]
      weak: [benchmark, prev_positions]
```
