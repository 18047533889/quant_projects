# server-c 工作树约束（用户明确要求，2026-09-09）

- 直接修改 `/home/sunhaiwei/quant_projects` 正式工作树；FactorEngine 路径是其中的 `factor_engine/`。用户说“本地”就是这里，不是 Mac 或 GitHub。
- 保留用户及其他 AI 已有改动。不创建分支、worktree、整库副本或独立 staging 代码树。子代理按文件划分责任，且仅使用 GPT-5.6-sol。
- 不自动 commit、push、部署、发布生产因子；任何旧指引中的自动推送要求不适用。
- 不为测试复制数据、`.git`、虚拟环境或整个仓库。先检查磁盘空间，临时文件必须小且有界，用完及时清理；独有代码、必要证据和正式数据不得误删。
- 大型持久产物使用项目 DataAccess 写入现有授权 COS 目标，不在本地堆积。不猜 bucket 或凭据，不将清理解释为删除正式数据的授权。
