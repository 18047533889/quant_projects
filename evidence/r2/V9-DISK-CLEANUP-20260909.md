# server-c 临时副本清理记录

用户于 2026-09-09 明确授权清理误建临时副本，要求今后直接修改正式工作树。

- 正式目录：`/home/sunhaiwei/quant_projects`。未删除正式数据；未创建分支、提交、推送或部署。
- 操作时观察到现有分支 main，HEAD `91b6539c0df3bbfdd96262828375fd71860fd567`；保留外部更新，不回退到旧检查点。
- 已删除 `/tmp/v9-m37-stage`。先逐字节验证并删除部分一致大文件，随后依据用户授权删除已识别的临时数据副本及剩余整库副本；并非对全部 333 GB 做了逐字节一致性认证。
- 已删除停用副本 `/tmp/v8-pending-overlay`、`/tmp/v9-m11c-runtime-overlay`、`/tmp/v9-m36-resource.BrBjqZ`。
- 最终 df 显示根盘约 338 GB 可用，使用率 83%；清理前本轮约 654 MB 可用。数值会随其他任务变化。
- 正式工作树代码、数据、已落地测试证据保留。临时副本直接删除，不在回收站；仍在正式目录的文件可从原件重建。

删除前保留与正式目录不同的源码/说明/补丁小备份，存于 Mac 的
`/Users/shw/Documents/Codex/2026-09-06/qin/work/v9-root/`，不是继续执行的代码树：

- `v9-m37-differing-source-before-cleanup.tar.gz`：286 项，约 1.1 MB；SHA256 `104c8407f32e9bb67aba0cf7aceb2e2e4b74467736488c997f6bfc8372207809`。
- `v9-small-drafts-before-cleanup.tar.gz`：29 项，约 144 KB；SHA256 `668ca1fdc198bc9346842a39f6d224a42859ce22e3b321dcfaf0fe5a67043dca`。
- 两个压缩包均完成 gzip 完整性及 tar 目录检查。差异可能是旧版，不能覆盖正式源码；未合入 pending-resume 草稿仍需审查/负测，不算已完成。

规则已记录于 Mac 全局 `/Users/shw/.codex/AGENTS.md` 和正式工作树 `AGENTS.md`：直接修改、保留并发改动、禁止分支/worktree/整库 staging、及时清理小临时文件，大型持久产物通过现有 DataAccess/COS 接口存储。不为清理而上传整库、凭据或无用重复数据。本轮没有执行 COS 上传，也没有配置新 bucket。
