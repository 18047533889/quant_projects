# 10 — Claude Code Parallel Development / Subagents Plan

## 10.1 目标

最大化并行，但避免共享文件冲突与架构漂移。

Claude Code 当前可采用：

- project subagents：`.claude/agents/*.md`
- worktree isolation：并行编辑互不覆盖
- Agent Teams（若当前环境启用）：用于需要互相讨论的架构/审计阶段

Agent Teams 属实验能力；不要把 `.claude/teams/*.json` 当项目配置。若未启用，则完全用主会话 + subagents/worktrees 执行。

## 10.2 Wave 0 — Read-only Audit

并行：

- qe-legacy-miner
- fp-legacy-miner
- fo-legacy-miner
- fa-legacy-miner
- platform-boundary-auditor
- corpus-regression-auditor
- license-auditor
- architecture-redteam（可复用 simplification/boundary agent）

输出：

- local repo snapshot
- function-level migration matrix
- duplicated capability list
- proposed package skeleton
- contract conflicts

Wave 0 禁止 production edit。

## 10.3 Wave 1 — Contract Freeze

Lead Architect + Chief Integrator：

- schemas
- public APIs
- package dependencies
- file ownership
- error taxonomy
- migration order

共享文件只有这两个角色修改。

## 10.4 Wave 2 — Reference Migration

QE：旧 evaluator/reference。
FP：AlphaPurifier/cross-sectional reference。
FO：registry/complexity contracts。
FA：catalog/registry/identity metadata。

先完成 reference + tests，不追求快。

## 10.5 Wave 3 — Parallel Core Build

建议同时 16–24 个 worker，按服务器资源调节。

### QE

- qe-runtime-agent
- qe-core-metrics-agent
- qe-temporal-robustness-agent
- qe-performance-agent

### FP

- fp-transform-agent
- fp-neutralization-agent
- fp-representation-agent
- fp-performance-agent

### FO

- fo-grammar-agent
- fo-search-agent
- fo-llm-agent

### FA

- fa-registry-identity-agent
- fa-novelty-agent
- fa-graph-cluster-agent
- fa-aggregation-agent

每个 agent 有独立 file ownership；尽量 `isolation: worktree`。

## 10.6 Shared Files

只有 chief-integrator 可修改：

- root schemas
- root pyproject/workspace config（若有）
- package `__init__.py` 最终 public exports
- package versions
- integration test top-level fixtures

Worker 不允许随意改别人的 `registry.py`/`pyproject.toml`。

## 10.7 Wave 4 — Integration

- FE adapter + QE
- QE + FO loop
- QE + FA novelty shortlist
- FA -> FP FeatureBundle
- Research Control refs

用真实 corpus 小样本先联调，再大 benchmark。

## 10.8 Wave 5 — Independent Audit

并行：

- math-auditor
- leakage-auditor
- performance-auditor
- boundary-auditor
- simplification-auditor
- extraction-auditor
- license-auditor
- corpus-regression-auditor

任何 FAIL 回到对应 owner 修复；auditor 不自己顺手改大量 production 代码，避免审计独立性丢失。

## 10.9 Agent Team vs Subagent

如果 Agent Teams 已启用：

- 架构审计/跨库设计用 Agent Team，因需要互相发消息。
- 文件实现仍尽量切成 worktree-isolated worker。

如果未启用：

- 主 Claude 负责 shared task list。
- 所有 worker 用 project subagents + worktrees。

不要依赖嵌套 Subagent 自行继续生成 Subagent；顶层 coordinator 负责调度。

## 10.10 Stop Conditions

暂停并回收并行度，如果：

- 多 agent 同时改同一 public contract
- package import graph 出现循环
- test semantics 未冻结却开始 fast optimization
- legacy migration 来源不明
- Agent 试图复制第三方 GPL runtime code
- 出现第二套 DA/FE 能力
