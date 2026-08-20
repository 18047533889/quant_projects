# 算子新增实施计划 (2026-08-12)

## 执行起点

**Current HEAD:** 854bdc2278678e3db03c894c059cd5fdbb7dac1b  
**Baseline:** 519 EXTENDED + 1097 DAILY 算子已存在  
**待处理清单:** 245 个算子名称 (223 候选 + 22 待认证)

## Phase 1: Preflight 分类 (中央执行)

生成 `operator_master_gap_preflight_854bdc22.csv`，对每个候选判定：

```
EXISTING_EXACT          — 已存在完全相同算子
EXISTING_ALIAS          — 已存在别名映射
EXISTING_EQUIVALENT     — 已存在语义等价算子
COMPOSABLE_NO_NEW       — 可用 2-3 个 primitive 组成
TRUE_GAP_IMPLEMENT      — 真实功能缺口，需实现
PROD_RECERTIFY          — 已存在但需后端/证据认证
RESEARCH_ONLY           — 仅 research surface
BLOCKED_DATA_CONTRACT   — 数据字段缺失阻塞
REJECT_LOOKAHEAD        — 违反 PIT 原则
ARCHITECTURE_SUPERSEDED — 架构层已替代
```

**关键规则:**
- 清单中 KAMA/HMA/WMA 等名称 → 当前已有 ts_kama 等，需映射而非重复实现
- filter 层 22 个算子已存在 → PROD_RECERTIFY 不重复造
- fiscal_* 大量候选 → 当前已有 fin_* 族，需 semantic 比对
- intra_* 分钟算子 → 当前已有 intra_* 族，逐个比对

## Phase 2: 并行实施 (按文件簇分配 subagents)

### Cluster A: Filter/Smoother 新增 (P0 priority 28 个)
**Agent 责任**: `cleaned_operators/filter_*.py` 新增模块
- `state_change_point_adaptive_ema`
- `state_filter_reset_on_break`
- `state_gain_scheduler`
- `state_elastic_turnover_prox`
- `ts_l1_trend_filter_trailing`
- `ts_total_variation_filter_trailing`
- `ts_robust_kalman_level`
- `ts_one_euro_filter` (P1)
- `ts_alpha_beta_filter` (P1)
- ... (完整列表见清单 §4)

**交付**:
- Pandas reference + Polars bridge
- DuckDB 不强求 (复杂 filter 通常不适合 SQL)
- 测试: 数值语义 + causality + NaN policy
- 注册到 EXTENDED_ONLY_CANONICALS

### Cluster B: Cross-sectional Shrinkage (P0 priority 3 个)
**Agent 责任**: `cleaned_operators/cs_shrinkage.py` 新模块
- `cs_shrink_to_market_mean`
- `cs_shrink_to_group_mean`
- `cs_empirical_bayes_shrinkage`

**后端策略**: Polars 强制，DuckDB 可选

### Cluster C: Fiscal/Fundamental 补充
**Agent 责任**: `cleaned_operators/fundamental/*.py` 扩展
- preflight 后确认的 TRUE_GAP fiscal_* 算子
- 严格 PIT: 基于 PubDate + fiscal period identity
- 不重复实现已有 fin_* 算子

### Cluster D: Technical Indicators 映射
**Agent 责任**: `cleaned_operators/technical_indicators.py` 或别名注册
- HMA/QQE/RSX/ALMA/CoppockCurve/ElderRay/FisherTransform
- 优先检查已有实现，建立别名映射
- 若缺失才实现 reference

### Cluster E: 后端认证 (22 个已存在算子)
**Agent 责任**: 验证 + 补全，不新增算子
- Polars backend 真实性检查
- Numeric parity 测试
- Parameter domain evidence
- Checkpoint/resume 验证

## Phase 3: 证据重生

所有新算子完成后：
- 重新生成 `factor_operator_verified.json`
- 重新生成 `primitive_verified.json`
- 更新 `CURRENT_HEAD_COLDSTART_OPERATOR_CONTRACT.csv`

## Phase 4: 全量回归

串行运行：
- `pytest tests/operators/` (新增算子测试)
- `pytest tests/backend_parity/` (三后端一致性)
- 完整 FE 回归 (排除 DA 依赖失败)

## 不实现项

- 需要 Level2/Tick 的分钟算子 (数据合约缺失)
- Research-only 高级算法 (topology/neural/particle filter) — 标记 RESEARCH_ONLY 不进 production
- 组合因子 (保留为 DSL expression)
- 架构已替代项 (report_asof → DA PIT join)

## 时间预估

- Preflight 分类: ~30min (中央)
- Cluster A (28 filter): ~3h (1 agent)
- Cluster B (3 shrinkage): ~1h (1 agent)
- Cluster C (fiscal gaps): ~2h (1 agent, 视 preflight 结果)
- Cluster D (technical mapping): ~1h (1 agent)
- Cluster E (22 recertify): ~2h (1 agent)
- 证据重生: ~30min (中央)
- 全量回归: ~1.5h (中央串行)

**总计: ~11h 并行执行 → 实际约 4-5h wall-clock**
