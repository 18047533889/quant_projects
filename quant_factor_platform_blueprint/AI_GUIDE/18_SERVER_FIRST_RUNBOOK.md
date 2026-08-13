# 18 — Server-first Execution Runbook

## Step 1 — Snapshot

```bash
pwd
python --version
git status --short || true
git log -1 --oneline || true
find . -maxdepth 2 -type d | sort > /tmp/quant_dirs.txt
```

禁止先 `git pull` 覆盖服务器未提交代码。

## Step 2 — Detect Packages

检查：

- pyproject/setup.cfg/setup.py
- package names/version
- editable installs
- virtualenv/conda
- import paths
- local path hacks

## Step 3 — DA/FE Capability Inventory

让 boundary agent 输出“已有能力清单”，然后将新需求逐一映射，标记：USE_EXISTING / NEED_ADAPTER / TRUE_GAP。

## Step 4 — Legacy Miners

并行生成四份函数级 inventory。

## Step 5 — Contract Freeze

主会话合并审计；在任何大规模 edit 前请求/记录 architecture decision。

## Step 6 — Branch/Worktree Strategy

即使用户最终在服务器直接改，不走 GitHub PR，也建议本地 Git worktree/branch 隔离并行写入。不得假设必须 push remote。

## Step 7 — Incremental Merge

每个 agent 交付：

- changed files
- tests
- migration source
- known limitations
- benchmark

Chief Integrator 才合并共享 API。

## Step 8 — Full Audit

完成后运行 extraction/no-legacy/boundary/leakage/performance/corpus。

## Step 9 — Archive

只有在新功能通过真实 regression 后才 archive legacy；不要边迁边删除唯一 reference。
