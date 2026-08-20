# R32 / MF / REM Execution Baseline

> 本文件由整改执行 AI 在开始前生成，记录真实执行起点。

## Repository state at execution start

```text
current_head    = 854bdc2278678e3db03c894c059cd5fdbb7dac1b
current_branch  = main
dirty_state     = DIRTY (195 paths reported by `git status --short`)
is_git_repo     = true (root: /home/shw/quant_projects)
```

注意：三份任务书记录的审计基线分别为
`4b1577fce14c097fba48619885b38d6c596242eb`（DataAccess R32）与
`ea02121edf2a9dd19b067dfe88f20d483cc9c0fb`（FE Model/Filter、最终集成清单）。
执行起点 HEAD 已经领先上述两者，因此每一项必须重新判定状态，
不得机械按任务书文字重复造模块。

## Runtime profile

```text
python_version   = 3.10.12
duckdb_version   = 1.5.4
polars_version   = 1.42.1
pyarrow_version  = 25.0.0
pandas_version   = 2.3.3
numpy_version    = 2.2.6
scipy_version    = 1.15.3
sklearn_version  = 1.7.2
platform         = Linux-5.15.0-171-generic-x86_64-with-glibc2.35
```

## Directly verified facts at baseline (spot-checked before dispatch)

| Fact | Evidence | Taskbook item |
|---|---|---|
| `MODEL_CURRENT_HEAD.json` 绑定 `6fd715953558da50d976951e9e7cc3c19f3dca36`，与 repo HEAD 不一致 | `factor_engine/docs/evidence/model_operators/MODEL_CURRENT_HEAD.json` | MF-P0-001 / REM-025 CONFIRMED_OPEN |
| `final_direct_use_ready_count = 0` | 同上 | MF-P0-002 / REM-024 CONFIRMED_OPEN |
| `build_optimizer_pass_manager()` 无任何生产调用者（仅测试引用） | `grep -rn build_optimizer_pass_manager` → 仅 `planner/optimizer_passes.py` 定义 + `tests/r42/test_compiler_pass_manager.py` | REM-011 / REM-197 ORPHANED (确认) |
| `Optimizer.optimize()` 直接串联 fold/lower/rewrite/canonicalize，不经过 PassManager | `factor_engine/planner/optimizer.py:16-35` | REM-011 ORPHANED (确认) |
| `factor_engine/signal_filter/` 不存在；filter 实现位于 `cleaned_operators/filter_*.py`（2346 行） | `ls` | MF §92 — 允许保留当前布局，但契约必须收口 |
| `dataaccess/r30/` 存在且含 `resolution_lease.py` | `ls dataaccess/r30/` | R32 §3 事实基础一致 |

## Status vocabulary used in this round

任务书要求的判定集合：

```text
CONFIRMED_OPEN | ALREADY_FIXED | PARTIALLY_FIXED | REGRESSION
SUPERSEDED_BY_CANONICAL | NOT_APPLICABLE | DEFERRED_WITH_REASON
ORPHANED | NOT_RUN_EVIDENCE | LOCAL_FIX_WAITING_INTEGRATION
DEEP_ARCH_NOT_CLOSED | VERIFY_REQUIRED | P1_BACKLOG
```

最终 P0 只允许收敛到：

```text
CLOSED_WITH_CURRENT_HEAD_PROOF | NOT_APPLICABLE_WITH_PROOF | BLOCKED_BY_EXTERNAL_DEPENDENCY
```

## Concurrency note

本轮无并行外部 agent。所有 subagent 由本次会话集中调度，
按文件簇划分以避免写冲突；全量回归测试由中央串行执行（单进程，小内存主机）。
