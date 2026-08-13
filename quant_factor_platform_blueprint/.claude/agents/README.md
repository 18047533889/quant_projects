# Project Subagents

这些 Markdown 可复制到实际项目 `.claude/agents/`。项目级 custom subagents 由 Claude Code 自动发现。

建议：

- Wave 0 先启用四个 legacy miner + boundary/license/corpus auditor。
- 架构冻结后再启用 developer agents。
- Developer agents 默认 `isolation: worktree`，降低并行修改冲突。
- Auditor agents 默认禁用 Write/Edit，保持独立审计。
- 若 `model: sonnet` 与你的当前 Claude Code 配置不兼容，可由主会话统一调整；不要为了修改模型字段改变 agent 职责。
