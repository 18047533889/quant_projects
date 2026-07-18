# Git 同步与仓库结构（企业级必读）

## 问题背景

若 GitHub 仓库（如 `hsunbj_quant_projects`）里看不到 `dataaccess/`、`factor_engine/`，通常**不是代码没写**，而是同步链路断了：

| 原因 | 现象 |
|------|------|
| 仅 `git add -u` | 新建文件永远不会进仓库 |
| `workspaces/` 被 `.gitignore` | 代码若在 workspace 下则永久不可见 |
| 空 commit 定时同步 | GitHub 有 commit 但 `files=[]` |
| 错误仓库结构 | README 写了 `factor_engine/` 但目录未上传 |

## 推荐仓库布局（方案 A）

公共基础设施必须在**仓库顶层**跟踪：

```text
quant_projects/          # 或 hsunbj_quant_projects 根
├── dataaccess/         # ✅ 必须 git track
├── factor_engine/       # ✅ 必须 git track
├── factor_layer/
├── scripts/
├── docs/
├── .github/workflows/   # CI 门禁
├── data/                # ❌ 仅本地数据，gitignore
├── logs/                # ❌ gitignore
└── workspaces/          # ❌ 个人实验，gitignore（若存在）
```

**禁止**把 `data_access` / `factor_engine` 只放在被忽略的 `workspaces/` 下。

## 服务器上一键同步（替代 git add -u）

```bash
cd ~/quant_projects

# 1. 校验关键路径是否已被 Git 跟踪
bash scripts/verify_repo_tracking.sh

# 2. 预览将加入暂存区的文件（白名单 add，含新文件）
bash scripts/git_sync_public_code.sh --dry-run

# 3. 加入暂存区
bash scripts/git_sync_public_code.sh

# 4. 提交并推送
bash scripts/git_sync_public_code.sh \
  --commit "feat(data_access): P0/P1 read snapshot, publish manifest, CI gate" \
  --push
```

### 白名单路径

`scripts/git_sync_public_code.sh` 使用：

```bash
git add -A -- data_access factor_engine scripts docs .github ...
```

**不会** add：`data/`、`logs/`、`workspace_data/`、`workspaces/`（若在 gitignore 中）。

### 多远程

```bash
export GIT_SYNC_REMOTE=origin   # 或 hsunbj 远程名
export GIT_SYNC_BRANCH=main
bash scripts/git_sync_public_code.sh --commit "..." --push
```

## 诊断命令

```bash
cd ~/quant_projects

# 公共代码在哪
find . -path './dataaccess/store.py' -o -path './factor_engine/runtime/engine.py'

# 是否被 ignore
git check-ignore -v dataaccess/store.py

# Git 实际跟踪了什么
git ls-files dataaccess/ | wc -l
git ls-files factor_engine/ | wc -l

# 未跟踪新文件
git status --short --untracked-files=all dataaccess/ factor_engine/
```

## CI 门禁

推送后 GitHub Actions 应出现：

- `data-access-unit`
- `data-access-contract`
- `data-access-no-direct-parquet`
- `data-access-pit`
- `data-access-path-security`
- `data-access-concurrency`
- `data-access-publish-atomicity`
- `data-access-repo-tracking`

定义于：`.github/workflows/data-access-gate.yml`

## 与 Codex / 代码审计的关系

审计必须基于**固定 commit SHA** 上可见的文件。流程：

```text
本地改代码 → verify_repo_tracking → git_sync_public_code → push
→ CI 全绿 → 记录 SHA → 代码复审 / production gate
```

在未 push 之前，GitHub 上的审计结论**不能**代表服务器最新代码状态。
