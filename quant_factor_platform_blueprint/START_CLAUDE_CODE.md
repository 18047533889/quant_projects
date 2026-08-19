# Claude Code 启动提示

把下面短段贴进主会话。**不要**要求完整加载 AI_GUIDE / LOOP archive / memory archives。

```text
你负责 quant_projects 模型前因子基础设施。先读仓库根 CLAUDE.md 与 /home/shw/CLAUDE.md；状态只读 compact：LOOP_ENGINEERING_STATUS.md / RESUME_STATE.md / AGENT_STATUS.md。禁止加载 docs/R2_HISTORY_ARCHIVE.md、archives/**/*_full_*、~/.claude/.../memory/archives/*。

硬约束：working tree 是事实源；禁止 git checkout/restore/stash/clean/reset 与批量 AST 重写；≤15GiB、≤2 低内存 Writer、串行 pytest（BLAS/OMP/MKL/Polars=1）；未跑门控记 NOT_RUN；不重建第二套 DA/PIT/calendar/DSL/materialization/cache/DAG。

改代码时：只加载当前包所需 AI_GUIDE 章节或 FE skill；subagent brief ≤30 行 + 文件列表 + evidence 路径。用户未明确要求时不做 GitHub/push。直接执行，不要 plan/确认。
```

Blueprint 新包工作额外 skim：`quant_factor_platform_blueprint/CLAUDE.md`；需要时再读对应 `AI_GUIDE/*.md`。
