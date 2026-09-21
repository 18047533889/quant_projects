# R60 — 全量算子落值优化（第二批）+ 路由 bug 修复（2026-09-21）

## 一、emitter 態路径清零（polars_expr_emitter.py）

R59 后审计发现剩余 `rolling_map` 逐行 UDF 派发点，本轮全部转为块帧/向量化 kernel：

| 算子 | 旧路径 | 新 kernel | 生产探针* |
|---|---|---|---|
| ts_decay_linear / decay_linear | rolling_map/行 | `_np_linear_decay`（age-slot 权重精确对齐） | 5.01s（≈数据地板） |
| ofi_imbalance_persistence | rolling_map/行 | `_np_ofi_persistence`（活跃配对前缀和，O(n)） | **76.8s → 2.80s** |
| ofi_reversal_rate | rolling_map/行 | `_np_ofi_reversal`（同前缀和技巧） | **11.0s → 2.79s** |
| m1_momentum_strength | rolling_map/行 ×2 | `_np_m1_momentum_strength`（顺序乘保 total==0 精确零契约） | 3.69s → ≈2.7s |
| m1_momentum_speed_change | rolling_map/行 ×2 | `_np_m1_speed_change`（快窗=慢窗尾部列，一次扫描） | **坏→2.69s**（见下） |

*生产探针 = ~5000 标的 × 2 年日线端到端（含 warmup）；基准面板(25×500)数字见下。

### 关键修复：m1_momentum_speed_change 双 emitter off-by-one

`_int_attr(input_index)` 语义是「第 index 个字面量（相对 series 之后）」，实际读 `inputs[index+1]`。
历史代码传 1/2 → 读到 (slow, 越界) → 一切位置参数调用必然 raise `fast_window must be < slow_window`：
- polars_expr_emitter.py：input_index 1/2 → **0/1**（修复后 `m1_momentum_speed_change(close,5,20)` 端到端 4.8s）；
- sql_pushdown/emitter.py：attr-only 读取 → 位置参数被静默忽略（永远用默认窗口）→ 补 input_index 0/1。

## 二、min_periods 语义澄清（测试抓出）

`test_flowmom_wave3_parity` 抓出 reversal gate 语义差：管道中非有限值以 **null** 传递，
`rolling_map min_samples` 只数非 null → 有效 gate 是**有限值计数 ≥ mp**（非物理行长）。
已修正 `_np_ofi_reversal`；该测试恢复基线（49 passed，仅剩 1 个预存的 SQL 注册清单失败）。

## 三、全量扫描（1236 个无 emitter dispatch 的 daily 算子）

按「每标的一次 registry kernel 调用」粒度计时（T=150，20s 超时，377 个成功计时）：

- 最慢：ts_l1_trend_filter_trailing 2.13s/标的、ts_total_variation_filter_trailing 1.33s（ADMM 迭代求解器，批量化为后续 batch-3）；
- ordinal 熵家族 0.4–0.9s/标的（batch-3 候选：共享 ordinal-pattern 向量化 helper）；
- 其余多数 <0.15s/标的。
- 859 个因自动参数填充不足未计时（多面板/复杂参数），非后端缺陷。

## 四、回归

- `test_flowmom_wave3_parity`：**49 passed / 1 failed（预存）**，与旧 emitter 基线一致；
- volval/ashare/csgrp wave3：10 failed / 117 passed，与基线完全一致（预存 duckdb 组操作问题）；
- R60 kernel A/B（敌对面含边缘行/NaN 洞/const/zeros/prices）：**0 fails**；
- R59 kernel A/B：18 OK（3 个已知 harness 伪影不变）；
- `load_all()` 治理检查通过。

## 五、证据

- `/tmp/r60_emit.patch`（emitter diff）、sweep/sweep2 JSON、before/after 基准 JSON（`~/_r60_bench_*.json`）
- 本轮未 commit 的历史：R57–R59 改动随本 commit 一并入库（同属 FactorEngine 优化工作流）。
