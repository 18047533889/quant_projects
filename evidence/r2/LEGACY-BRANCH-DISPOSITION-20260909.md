# 旧远端分支内容核对与保留边界

审计基准：主干 `9ea3a7ad556dd723e35c58866f9697b31f4c4442`；旧历史锚点
`f56de261192ccf41086dbd8c0535d5e3d2740de7`（`r2-q-empty-authority-gates`）。
原始逐路径证据：`LEGACY-BRANCH-CONTENT-20260909.json`；只读复核工具：
`scripts/audit_legacy_branch_content.py`。

## 结论

34 个分支的提交日期在 2026-07-09 至 2026-08-20 之间。旧历史与目前重建主干
通常没有共同祖先，不能将“日期早”或“没有合并关系”直接解释为无用。
此次没有把旧目录整体覆盖回正式代码，也没有声称全部旧改动已经恢复。

以下 10 个标签的完整提交历史都可从保留的旧锚点到达，允许删除重复标签，
不需要复制代码；删除前必须再次确认远端锚点与各目标 tip，使用预期 tip 保护。

- `agent/factor-cold-start-library-20260720`
- `agent/factor-cold-start-library-merge-staging`
- `agent/factorengine-consolidated-hardening-20260801`
- `agent/factorengine-consolidated-hardening-20260801-temp`
- `agent/factorengine-production-completion-20260801`
- `agent/factorengine-production-hardening`
- `agent/harden-causal-operators`
- `agent/rename-data-access-to-dataaccess`
- `fix/dataaccess-factorengine-hardening`
- `sync/data-access-0.5.0`

另 `temp/factorengine-wheelhouse-20260802` 有 4 个独有中间提交，但其最终树与
可从旧锚点到达的共同基准树相同，没有独有最终文件内容，可删除临时标签。
这不等于 4 个中间提交已进入目前主干。

其余 23 个标签（包含旧锚点）保留。AutoFactorEvaluation 重建、冷启动库、
旧算子及数据契约等分支存在尚未逐项语义裁决的差异，不能为清理界面而丢弃。
后续需按功能与现主干测试对照，仅迁移确认缺失且适用的实现。

## 证据局限

原始 JSON 的 `BLOB_PRESENT_IN_MAIN_HISTORY` 仅说明相同 blob 曾出现于主干历史，
不保证当前版本仍有该功能；`REVIEW_DELTA` 也不意味着应该照搬。
删除路径尚未按旧 `dataaccess/` 与新 `data_access/` 等迁移映射裁决。
`commits_beyond_old_anchor=0` 仅证明旧锚点覆盖，不能证明现主干覆盖。

## 已执行（2026-09-09）

上述 11 个冗余远端分支已经在重新核对 live refs、保留锚点及预期 tip 后原子删除。
操作前后主干保持 `e0c8fbfcd439028623a4b9a2bc50d71c545a8e18`，其他保留引用的 SHA 未变；
远端剩余 23 个非 main 分支。10 个重复分支的历史仍可从保留锚点到达；临时分支的
最终文件树与保留基准一致，不将其独有中间提交误称为已合入 main。

实际回执：`evidence/team_sync_20260909/legacy_branch_cleanup_result.json`。
