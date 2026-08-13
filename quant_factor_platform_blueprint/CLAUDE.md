# Quant Factor Platform — Project Rules

## Mission

本项目当前只解决 **模型输入之前** 的因子研究基础设施：

`DataAccess -> FactorEngine -> QuantEvaluator -> (optional FactorOptimizer loop) -> FactorAssets -> FactorPreprocess -> ModelInput`

模型训练、组合优化、执行撮合不在本阶段范围内。

## Source of Truth

1. **服务器当前 working tree 是代码唯一事实源。** GitHub 仅用于公开/历史对照。
2. 数据语义、PIT 数据访问、calendar、universe、snapshot 等以现有 `dataaccess` 为事实源。
3. 因子 DSL、AST/IR、算子语义、因子计算与现有物化能力以 `factor_engine` 为事实源。
4. 新指标数学语义归 `quant_evaluator`；新因子优化策略归 `factor_optimizer`；资产状态与关系归 `factor_assets`；模型输入前表示归 `factor_preprocess`。

## Mandatory First Step

任何写代码任务开始前：

- 阅读 `AI_GUIDE/00_START_HERE.md`。
- 加载 Skill `quant-factor-platform-development`。
- 运行本地仓库审计：`git status`、`git diff --stat`、最近 commit、目录树、imports。
- 搜索现有 DataAccess / FactorEngine / legacy 是否已有能力。
- 未证明“现有能力不存在或不适用”前，禁止新建重复模块。

## Do Not Rebuild

禁止新建第二套：

- Data lake / factor lake / parquet reader / COS/S3 wrapper / DuckDB wrapper
- PIT join engine / calendar / universe / snapshot manager
- Factor DSL / operator engine / factor compiler / factor materialization engine
- 通用 cache platform / 通用 distributed DAG / 通用 orchestration framework
- 大型 backtest / execution engine / portfolio optimizer（本阶段）

## Package Boundaries

- `quant_evaluator`：纯 Evidence/Metric/Diagnosis；不得承担数据平台、因子管理、模型训练。
- `factor_optimizer`：产生合法 mutation/search decision；不得复制 FE operator kernels 或 QE metric kernels。
- `factor_assets`：identity/registry/seen/novelty/graph/cluster/selection/aggregation/lifecycle；不得成为大规模因子值仓库。
- `factor_preprocess`：stateless/fitted transforms、neutralization、representation；不得做最终模型训练或因子资产准入。

## Engineering Rules

- Batch-first；单因子 API 只是 batch API 的薄包装。
- Reference/Fast 双实现；fast 必须 parity。
- Production 路径不得偷偷 Pandas fallback，除非明确标记 reference/debug。
- 时间相关 fitted object 必须记录 fit window / fit_end_time；禁止 full-sample fit。
- 所有 lifecycle / decision / trial 都可追溯，不硬删除历史因子。
- 不使用固定单分数代替所有决策；优先 hard gates + evidence vector + conditional novelty + Pareto/budgeted decision。
- 不使用 `corr > threshold -> delete` 作为唯一去重。
- 不允许 `sys.path.append('../')`、monorepo parent import、legacy runtime import。

## Required Acceptance

每个新 package：

- 独立 venv 安装通过
- 独立 pytest 通过
- legacy import audit 通过
- package boundary audit 通过
- reference/fast parity（适用模块）通过
- benchmark 有基线记录
- real corpus regression 通过
- docs / public API / schemas 同步

详细规范见 `AI_GUIDE/`。
