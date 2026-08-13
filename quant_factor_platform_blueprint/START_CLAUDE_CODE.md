# Claude Code 启动提示

把下面整段发送给服务器项目根目录中的 Claude Code 主会话。不要删减硬约束。

```text
你现在负责重构 quant_projects 的模型前因子研究基础设施。请先阅读项目根目录 CLAUDE.md，并加载项目 Skill：quant-factor-platform-development。然后完整阅读 AI_GUIDE/00_START_HERE.md。

硬约束：
1. 服务器当前 working tree 是唯一代码事实源。先运行 git status、git diff --stat、git log -1、目录树和 import 搜索；不要假定 GitHub main 与服务器一致。
2. 第一阶段禁止大规模写代码。先并行审计现有 dataaccess、factor_engine 和所有 legacy/旧模块，生成函数级 Migration Matrix，分类为 REUSE_AFTER_TEST / REWRITE / REFERENCE_ONLY / CORPUS_ONLY / DISCARD。
3. 不允许重建第二套 DataAccess、PIT engine、calendar、universe、storage、COS/S3 wrapper、DuckDB wrapper、Factor DSL、Factor materialization、factor cache 或通用 DAG 平台。
4. 新开发目标只有 quant_evaluator、factor_optimizer、factor_assets、factor_preprocess 四个可独立安装的 package；Research Control 是中央轻量 ledger，不是第五个重型运行库。
5. 所有第三方 GPL 代码默认只做参考/测试 corpus，禁止直接复制进专有 runtime，除非 license audit 明确批准。
6. 所有新 package 必须可单独复制到 /tmp、新建 venv、pip install、pytest，不得依赖 monorepo sys.path hack 或 legacy import。
7. QuantEvaluator 必须 Batch/Matrix-First，高性能且 Reference/Fast 双实现；FactorOptimizer 不重复实现 FE 算子或 QE 指标；FactorAssets 不保存大规模 factor values；FactorPreprocess 必须严格区分 stateless transform 与 fitted transform，任何 fitted state 只能用训练窗口拟合。
8. 所有跨包依赖只能通过 public API、schema/versioned contract 或 optional adapter，不允许 private import。

执行方式：
- 先建立并行审计任务，优先使用项目 .claude/agents 中的 legacy miner / auditor 角色；若当前 Claude Code 支持并启用了 Agent Teams，可用 Agent Team 做审计和架构协作。文件修改任务使用 worktree 隔离。
- 审计完成后，由 lead-architect 输出并冻结：目录、public API、contracts、dependency DAG、migration matrix、file ownership。
- Contract Freeze 后才启动开发 agents，并按照 AI_GUIDE/10_CLAUDE_PARALLEL_ORCHESTRATION.md 并行推进四个 package。
- 每个 Wave 结束必须由独立 audit agents 验证，不允许原开发 agent 自己宣告完成。

开始时只做 Wave 0：本地仓库状态、全量代码盘点、旧代码函数级迁移清单和边界冲突分析。不要直接改生产代码。
```
