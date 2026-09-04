# MULTIWORKER_SCALING — worker 1/2/4/8 throughput（100k GO §52）

## 结论（拐点）
四档均保持增长（末档增速 >10%），尚未出现明显拐点；峰值 37.6 f/s 在 w8。建议延伸 16 worker 复核上限。（若机器 32 核，8 档可能已是物理上限附近）

## 硬件环境
- CPU: 32 vCPU（AMD EPYC 9K84，32 核 92G 实例）
- OOM 守卫：所有子进程合计最多 31 核（`multiworker_governance.DEFAULT_TOTAL_CORES`）
- 合成面板：1200 股 × 300 日；每 worker 批跑 30 个 ts_mean/ts_std/ts_sum/ts_max 因子
- seed 固定可复跑；后端 PandasBackend（CPU-bound、GIL 相关）

## 1/2/4/8 worker 实测表格

| worker | OMP/worker | 线程上限(workers×OMP) | 因子数 | wall(s) | throughput(f/s) | scaling vs 1 | CPU% | RSS(MB) |
| ------ | ---------- | -------------------- | ------ | ------- | --------------- | ------------ | ---- | ------- |
| 1 | 31 | 31 | 30 | 5.266 | 5.7 | 1.0 | 26 | 403 |
| 2 | 15 | 30 | 60 | 5.646 | 10.6 | 1.865 | 32 | 797 |
| 4 | 7 | 28 | 120 | 5.576 | 21.5 | 3.775 | 34 | 1596 |
| 8 | 3 | 24 | 240 | 6.381 | 37.6 | 6.598 | 50 | 3188 |

## 超卖场景（验证 governor 必要性）

| 配置 | OMP/worker | 线程上限 | 因子数 | wall(s) | throughput(f/s) | scaling vs 1 | CPU% |
| ---- | ---------- | -------- | ------ | ------- | --------------- | ------------ | ---- |
| 8×8 | 8 | 64（>31 违反治理） | 240 | 6.554 | 36.62 | 6.424 | 51 |

## 与 multiworker_governance 配额规则一致性

`workers × OMP <= floor(31 / coexist_count)` 在 1/2/4/8 各档均满足：
- w1: 1×31=31 ✅
- w2: 2×15=30 ✅（coexist=2 → floor(31/2)=15）
- w4: 4×7=28 ✅（coexist=4 → floor(31/4)=7）
- w8: 8×3=24 ✅（coexist=8 → floor(31/8)=3）
- 超卖 8×8=64 > 31 ❌：violates governor，用于量化不加治理的劣化。

## 治理决策
默认推荐：单主进程内部并发（solo 31 核，`FACTOR_ENGINE_RESOURCE_PROFILE=solo`）。
多进程多 worker 必须显式 `FACTOR_ENGINE_COEXIST=N` 让系统按 `floor(31/N)` 分桶。

原始 JSON：`/tmp/bench_worker_scaling.json`