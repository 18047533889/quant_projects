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

---

# Real DA/COS IO scaling（R61-P1 #56）

用户质疑「worker benchmark 是 synthetic Pandas」——它从不触达真实 DA/COS IO、
远程扫描字节、缓存命中率、重复下载。本段用**真实 data_access 读链路 + 真实 COS**
（`data_access.cos.mirror` → `research-cos`/quant-admin CLI 网关）重测 1/2/4/8。

## 环境与真实性声明

- **模式：真实 COS（非本地模拟）**。`DATA_ACCESS_COS_READ_MODE=remote`，
  每档全新 cache root（`/home/sunhaiwei/cos_data/cos_bench/cache_w<N>`，落在
  research-cos 批准 local root `/home/{user}` 下），`ASHARE_PARQUET_ROOT=<空目录>`
  强制本地镜像视图为空 → 每个对象都是**冷启动真实远程下载**。
- CLI：`/usr/local/bin/research-cos`（linux-sudo-policy-coscli 网关，
  quant-admin 成员，user=sunhaiwei，`whoami` 已验证）。COS 源：
  `cos://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/{date}.parquet`。
- 工作负载：`store.prepare_read + execute_prepared_read(ashare_stock_daily,
  cols=[TradeDate,Symbol,Close,Vwap,Volume,Factor], time_range=8 天窗口 × 轮)`
  —— 真实读链路（registry → mirror/remote → snapshot → governor → DuckDB）。
  窗口 `2026-06-15..2026-06-30`（10 个实际交易日对象；2026-06-19 真实缺失 → empty_ok）。
- 单飞：`_sync_cos_file` 对 per-key 持 flock 覆盖**整个下载事务**（fresh 复检 +
  cp + os.replace + manifest 原子写）；fetch 计数 = 真正执行 research-cos cp 的次数
  （`DATA_ACCESS_COS_FETCH_LOG` JSONL 每行一次 cp）。
- 治理：`multiworker_governance.resolve_worker_budget` → 每 worker
  `OMP_NUM_THREADS=floor(31/workers)`；`workers×OMP<=31` 全档满足。
- CPU%/RSS：本 benchmark 对 worker 进程用 communicate() 后采样已太迟 → 记 0，
  不伪造。CPU-bound 结论见上表 synthetic 段。
- 复跑：`python factor_engine/scripts/bench_worker_scaling_real.py --levels 1,2,4,8`

## 1/2/4/8 worker 实测（真实 DA/COS IO）

| worker | OMP/worker | 线程上限 | reads | rows | wall(s) | throughput(f/s) | scaling vs 1 | remote fetch | duplicate-download | cache hit % |
| ------ | ---------- | -------- | ----- | ---- | ------- | --------------- | ------------ | ------------ | ------------------ | ----------- |
| 1 | 31 | 31 | 1 | 31 240 | 3.662 | 0.27 | 1.0 | 10 | 0 | 37.5 |
| 2 | 15 | 30 | 2 | 62 480 | 3.011 | 0.66 | 2.44 | 10 | 0 | 68.75 |
| 4 | 7 | 28 | 4 | 124 960 | 2.658 | 1.50 | 5.56 | 10 | 0 | 84.38 |
| 8 | 3 | 24 | 8 | 249 920 | 3.013 | 2.65 | 9.81 | 10 | 0 | 92.19 |

（f/s = reads/s；单 worker 每 read 拉取 10 个对象，即 10 次远程 fetch = 10 个
~420KB parquet ≈ 4.2MB 远程扫描/档。cache hit % = 对象级 `hit/(hit+fetch)`，
1 worker 因冷启动后无跨进程共享故 37.5%，随 worker 数升到 92.2%。）

关键结果：
1. **跨进程单飞成立：每档 remote fetch 恒 = 10（唯一对象数），duplicate-download
   = 0** —— w8 下 8 个 worker 并发请求同一批日对象也只发生 10 次真实下载
   （此前锁只在 `_run_cos_cli` 内时 dup=28，根因=锁未覆盖 os.replace+manifest
   窗口；已修：flock 覆盖整个下载事务）。
2. **worker 伸缩在真实 IO 下成立**：f/s 0.27 → 2.65（w8 vs w1 = 9.8×），
   与 synthetic CPU-bound 段同向（37.6/5.7=6.6×）但增幅更大——真实远程 IO
   场景多进程并发主要赢在**并行下载 + 单飞去重**，8 worker 让同一批对象的
   缓存预热 + 查询并发执行。
3. **cache hit % 单调上升**（37.5→92.2）：进程越多，共享缓存复用越充分；
   这是「加 worker 换吞吐」在 DA/COS 场景的真实来源。

## 真实 COS vs synthetic 结论差异

- synthetic 段是纯 CPU-bound（Pandas ts_mean/ts_std）：w8 峰值 37.6 f/s，
  瓶颈是 GIL/核数。
- 真实 IO 段是 remote-read-bound：每 read 要过 remote LIST/HEAD/下载/governor，
  单进程只能串行下载（w1 f/s 0.27 受限于 ~420KB×10 ≈ 4.2MB 下载耗时 ~3.5s）。
  w8 把下载并行化后 f/s 提升到 2.65，且**不增加远程字节**（单飞去重）。
- 结论不变：多 worker 资源治理（`workers×OMP<=floor(31/coexist)`）仍是硬约束，
  但真实 IO 场景下 worker 数的收益曲线更陡——因为 IO 等待天然让出 CPU。

## 复现 & 证据

- benchmark：`factor_engine/scripts/bench_worker_scaling_real.py`
- 原始 JSON：`factor_engine/artifacts/perf_vec/MULTIWORKER_REAL.json`
- 单飞证明（单元测试）：`data_access/tests/unit/test_cache_single_flight.py`
  （2 passed：真实 COS 4 进程并发 → fetch=1/hit=3；本地对象存储模拟同语义）
- 复跑（约 1 分钟）：
  `python factor_engine/scripts/bench_worker_scaling_real.py --levels 1,2,4,8`