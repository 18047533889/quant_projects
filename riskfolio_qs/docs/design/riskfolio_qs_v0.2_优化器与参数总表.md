# riskfolio_qs v0.2 优化器与参数总表（单页）

## 1. 适用范围

本总表用于快速查看“基础版 3+1”一键组合优化配置。

基础默认优化器：

1. minvar_enhance_index
2. meanvar_enhance_index
3. meanvar_absolute_return
4. topn_long_only_equal_weight（兜底）

扩展优化器（不默认启用）：

1. topn_long_short_equal_weight

---

## 2. 优化器主映射总表

| optimizer_name | objective_id | hard_constraint_set | soft_constraint_set | risk_mode | backend | fallback_policy |
|---|---|---|---|---|---|---|
| minvar_enhance_index | OBJ_MINVAR_ACTIVE | HC_INDEX_ENHANCE_CORE | SC_TC_TURNOVER | barra_factor | riskfolio_backend | fail_fast |
| meanvar_enhance_index | OBJ_MEANVAR_ACTIVE | HC_INDEX_ENHANCE_CORE | SC_TC_TURNOVER | barra_factor | riskfolio_backend | fail_fast |
| meanvar_absolute_return | OBJ_MEANVAR_ABS | HC_LONG_ONLY_CORE | SC_TC_TURNOVER | historical_cov | riskfolio_backend | fail_fast |
| topn_long_only_equal_weight | OBJ_TOPN_LONG_ONLY | HC_TOPN_LONG_ONLY | SC_NONE | none | rule_backend | degrade_to_rule |

---

## 3. 强依赖输入总表

| optimizer_name | 强依赖输入 |
|---|---|
| minvar_enhance_index | alpha, benchmark, prev_positions, F, G, F_mcap, F_ret, F_spec |
| meanvar_enhance_index | alpha, benchmark, prev_positions, F, G, F_mcap, F_ret, F_spec |
| meanvar_absolute_return | alpha, market, prev_positions |
| topn_long_only_equal_weight | alpha |

弱依赖（可降级）：

1. cost_impact
2. benchmark, prev_positions（对 rule 兜底优化器）

---

## 4. 默认冻结参数（按优化器）

### 4.1 minvar_enhance_index

| 参数 | 默认值 |
|---|---:|
| single_name_max | 0.03 |
| active_weight_abs_max | 0.02 |
| turnover_cap | 0.20 |
| linear_cost_penalty | 1.0 |
| impact_cost_penalty | 0.0 |
| industry_neutral_mode | strict |
| style_neutral_mode | band |
| style_band | 0.10 |
| mcap_neutral_mode | band |
| mcap_band | 0.10 |

### 4.2 meanvar_enhance_index

| 参数 | 默认值 |
|---|---:|
| alpha_weight | 1.0 |
| risk_weight | 1.0 |
| single_name_max | 0.03 |
| active_weight_abs_max | 0.02 |
| turnover_cap | 0.20 |
| linear_cost_penalty | 1.0 |
| impact_cost_penalty | 0.0 |
| industry_neutral_mode | strict |
| style_neutral_mode | band |
| style_band | 0.10 |
| mcap_neutral_mode | band |
| mcap_band | 0.10 |

### 4.3 meanvar_absolute_return

| 参数 | 默认值 |
|---|---:|
| alpha_weight | 1.0 |
| risk_weight | 1.0 |
| single_name_max | 0.05 |
| turnover_cap | 0.25 |
| linear_cost_penalty | 1.0 |
| impact_cost_penalty | 0.0 |

### 4.4 topn_long_only_equal_weight

| 参数 | 默认值 |
|---|---:|
| topn_n | 50 |
| topn_weight_mode | equal |
| single_name_max | 0.05 |

---

## 5. 可覆盖参数白名单

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

非白名单参数覆盖请求：直接失败。

---

## 6. 默认运行路由（基础版）

1. index_enhancement 场景：优先 meanvar_enhance_index。
2. 若目标是更保守风险控制：使用 minvar_enhance_index。
3. absolute_return 场景：使用 meanvar_absolute_return。
4. 若关键输入缺失且允许兜底：降级 topn_long_only_equal_weight。

---

## 7. 版本与审计字段

每次运行建议至少落盘：

1. optimizer_name
2. objective_id
3. risk_mode
4. parameter_version
5. mapping_version
6. fallback_used
7. data_version_hash

---

## 8. 关联文档

1. riskfolio_qs_v0.2_纯净数学定义.md
2. riskfolio_qs_v0.2_数据链路初版.md
3. riskfolio_qs_v0.2_优化器映射规范.md
4. riskfolio_qs_v0.2_冻结参数规范.md
