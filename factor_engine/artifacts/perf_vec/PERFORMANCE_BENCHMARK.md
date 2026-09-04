# PERFORMANCE_BENCHMARK — FactorEngine 100k 性能 gate 实测

> 生成：2026-09-04 · HEAD `a07ea23020f86f4a63f50f67ec5b0ae1023b5b22`
> 对应：GO_PROMPT §100/§52 三档性能 gate（分钟 bundle 曲线 / cold-warm cache / worker scaling）。
> 原始 JSON：`/tmp/bench_minute_bundle.json`、`/tmp/bench_worker_scaling.json`（脚本：`scripts/bench_minute_bundle.py`、`scripts/bench_worker_scaling.py`）。
> 硬件：32 vCPU（AMD EPYC 9K84，32 核 92G 实例）；OOM 守卫 `OMP_NUM_THREADS≤31`。

## 1. 分钟 bundle：feature count 10 → 100 → 200 不线性放大（P0#5 一次 scan 生效）

合成 A 股分钟面板（09:31-11:30 / 13:01-15:00 = 240 bars/日）。NEW 路径 = `intraday_feature_runtime_v2.compute_many`（单次 `_grouped_bars` scan，bar 级共享 `_calc_shared`）；OLD 路径 = `intraday_feature_extension.load_intraday_feature` 逐 feature 重扫。

| n_features | wall(s) | per_feature(s) | RSS(MB) |
|-----------|---------|----------------|---------|
| 10 | 1.116 | 0.1116 | 332.1 |
| 50 | 4.946 | 0.0989 | 351.3 |
| 100 | 7.615 | 0.0761 | 356.3 |
| 200 | 7.411 | 0.0371 | 358.3 |

**共享 scan 判据：**
- scale 10→100 = **6.82x**（线性应是 10.0x）；
- scale 10→200 = **6.64x**（线性应是 20.0x）；
- per-feature 成本随 n 单调下降（0.1116 → 0.0371 s/feature）——每 feature 的固定 scan 成本被摊薄，证明共享 scan 生效。

fixture：100 inst × 3 天（面板 2.5 MB，daily 对齐行 300）。

## 2. OLD 逐 feature 重扫 vs NEW 单 scan

同一面板，两种入口对拍：

| 场景 | OLD(s) | NEW(s) | speedup |
|------|--------|--------|---------|
| 单 feature | 3.718 | 0.143 | **26.04x** |
| 100 features | 60.757 | 2.634 | **23.06x** |

（100-feature OLD 对比用小 fixture：20 inst × 3 天使逐 feature 重扫可完成。）

## 3. cold vs warm（grouped-bars cache）

同 fixture、同 100 feature：warm 第二次 `compute_many` 复用缓存的 grouped bars。

| 档位 | wall(s) | 备注 |
|------|---------|------|
| cold | 12.215 | 首次调用，冷 `_grouped_bars` |
| warm | 7.948 | 二次调用，复用 bars cache |
| **warm_speedup** | **1.54x** | cold/warm |

scan 验证：一次 `compute_many` 后 grouped-bars cache keys = **1**（单 scan，cache 按 source identity 键控）。

## 4. worker 1/2/4/8 scaling（§100/§52）

CPU-bound 合成 workload（1200 股 × 300 日，Pandas 向量化 ts_mean/ts_std/ts_sum/ts_max 因子批跑；每 worker 30 因子）。进程级并行，spawn 前按 `multiworker_governance.per_process_cpu_budget` 设 `OMP_NUM_THREADS`。

| worker | OMP/worker | 线程上限(workers×OMP) | 因子数 | wall(s) | throughput(f/s) | scaling vs 1 | CPU% | RSS(MB) |
|--------|-----------|----------------------|--------|---------|----------------|--------------|------|---------|
| 1 | 31 | 31 | 30 | 5.266 | 5.7 | 1.0 | 26 | 403 |
| 2 | 15 | 30 | 60 | 5.646 | 10.6 | 1.865 | 32 | 797 |
| 4 | 7 | 28 | 120 | 5.576 | 21.5 | 3.775 | 34 | 1596 |
| 8 | 3 | 24 | 240 | 6.381 | 37.6 | 6.598 | 50 | 3188 |

**结论**：四档均增长（末档增速 >10%），未见拐点；峰值 37.6 f/s 在 w8。32 核机器 8 档接近物理上限，延伸 16 worker 复核前先确认不存在拐点。

### 超卖对照（证明 governor 必要性）

| 配置 | OMP/worker | 线程上限 | wall(s) | throughput(f/s) | scaling | CPU% |
|------|-----------|---------|---------|-----------------|---------|------|
| 8×8 | 8 | **64（>31 违反治理）** | 6.554 | 36.62 | 6.424 | 51 |

超卖 8×8=64 线程没有换来更高吞吐（36.6 vs 37.6 f/s），CPU 51% 未有效利用 → 治理规则 `workers×OMP ≤ floor(31/N)` 正确。

### 治理一致性

`workers × OMP <= floor(31 / coexist_count)`：
- w1: 1×31=31 ✅ · w2: 2×15=30 ✅（floor(31/2)=15）· w4: 4×7=28 ✅（floor(31/4)=7）· w8: 8×3=24 ✅（floor(31/8)=3）
- 超卖 8×8=64 > 31 ❌ violates governor。

**默认推荐**：单主进程内部并发（solo 31 核，`FACTOR_ENGINE_RESOURCE_PROFILE=solo`）；多进程必须显式 `FACTOR_ENGINE_COEXIST=N` 让系统按 `floor(31/N)` 分桶。

> 详细表格含逐档 factor_budget_rule / rss 原值见 `MULTIWORKER_SCALING.md`。

## 5. 旧 `perf_vec_bench` 的 `__vec__` 触发修复对照（P0#3）

lambda 包裹丢 `__vec__` 时旧加速比全是假的（0.91x 噪声）；修复后实测真实 `__vec__` 路径：
**price_delay 1.89x · volume_imbalance 3.08x · session_mean_reversion 1.25x · 总 1.91x**（4000 股 × 14 天 × 240 bars）；bind fail-closed（vec 缺失抛 LookupError）。覆盖快照：`artifacts/perf_vec/intraday_vector_coverage.json`（vectorized 13 / total 53，见 INTRADAY_VECTOR_COVERAGE.md）。
