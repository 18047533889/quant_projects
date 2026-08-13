# Quant Factor Pre-Model Platform — Development Blueprint

本包用于直接交给服务器上的 Claude Code / AI 开发工具执行。目标是把现有 `quant_projects` 中有价值的代码迁移出来，并在 **不重复建设 DataAccess / FactorEngine 已有能力** 的前提下，完成模型输入之前的四个新库：

1. `quant_evaluator` — 因子证据与诊断
2. `factor_optimizer` — 证据驱动的因子优化与搜索
3. `factor_assets` — 因子资产治理、去重、组织、聚合
4. `factor_preprocess` — 模型输入前的因子表示与预处理

已有底座：

- `dataaccess`：数据语义、读取、PIT 数据访问、snapshot、批量读取等
- `factor_engine`：DSL/AST/IR、算子语义、因子计算、批量执行/CSE、后端与现有物化能力

## 使用顺序

1. 把本包解压到服务器项目根目录旁边，**不要直接覆盖现有代码**。
2. 先阅读 `CLAUDE.md` 与 `AI_GUIDE/00_START_HERE.md`。
3. 在 Claude Code 中优先执行 `START_CLAUDE_CODE.md` 中的启动提示。
4. 第一阶段只做服务器本地仓库审计、迁移矩阵和 Contract Freeze，不允许立即大规模写代码。
5. 第二阶段再按 `AI_GUIDE/10_CLAUDE_PARALLEL_ORCHESTRATION.md` 多 Agent 并行开发。

## 文件定位

- `PRE_MODEL_FACTOR_PLATFORM_OVERVIEW.html`：给人看的整体架构说明。
- `AI_GUIDE/*.md`：给 AI 直接执行的详细开发规范。
- `CLAUDE.md`：项目级最高优先级开发规则模板。
- `.claude/skills/quant-factor-platform-development/`：Claude Code Skill 模板。
- `.claude/agents/`：可直接参考/复制到项目的 Subagent 模板。

## 最重要的原则

> Server working tree is the source of truth. GitHub is only a baseline/reference.

> Migrate proven semantics; do not inherit legacy architecture.

> Before writing a new capability, prove that DataAccess, FactorEngine, or reusable legacy code does not already provide it.
