# server-c 工作树约束（用户明确要求，2026-09-09）

- 直接修改 `/home/sunhaiwei/quant_projects` 正式工作树；FactorEngine 路径是其中的 `factor_engine/`。用户说“本地”就是这里，不是 Mac 或 GitHub。
- 保留用户及其他 AI 已有改动。不创建分支、worktree、整库副本或独立 staging 代码树。子代理按文件划分责任，且仅使用 GPT-5.6-sol。
- 用户于 2026-09-18 明确授权本项工作可以直接 commit 和 push。提交前检查并保留其他 AI/用户改动，仅提交已验证的相关修复；推送前核对目标 remote/main，不强推。不部署、不发布生产因子（仍须另行授权）。
- 不为测试复制数据、`.git`、虚拟环境或整个仓库。先检查磁盘空间，临时文件必须小且有界，用完及时清理；独有代码、必要证据和正式数据不得误删。
- 大型持久产物使用项目 DataAccess 写入现有授权 COS 目标，不在本地堆积。不猜 bucket 或凭据，不将清理解释为删除正式数据的授权。

## 上传边界（2026-09-22，必须遵守）

- 个人 `18047533889/quant_projects/main` 才能接收整个总仓库。
- HKUST-QUANT-SOCIETY 的 13 个独立库只能接收总仓库同名目录的 Git tree；例如 `factor_engine` 必须等于 `main:factor_engine`，不能推总仓库 HEAD，也不能用提交消息声称“sync”代替内容核验。
- 统一入口：`./push_both.sh --dry-run` 预检，授权后 `./push_both.sh` 上传个人仓库及全部独立库；`--only factor_engine` 只同步对应 HKUST 库。
- 有其他 AI 未提交工作时，不得 git add 全仓库、删除改动或替其他人提交。明确仅发布已提交快照时可加 `--allow-dirty-committed`；必须告知未提交改动未上传。
- 不创建分支或 PR、不强推；非快进冲突必须停止核查。不得绕过 pre-push（包括 --no-verify、临时 core.hooksPath、删除 hook）。
- 本机 pre-push 必须指向 `scripts/hooks`，只做同步校验，禁止后台启动上传。所有上传前后都核对目标地址和 Git tree。
- 操作及恢复说明：`docs/PUSH_PUBLICATION.md`。这不是新增的 commit/push 授权，仍以当前用户授权为准。
