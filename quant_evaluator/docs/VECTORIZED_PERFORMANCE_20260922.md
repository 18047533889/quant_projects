# QE 向量化性能优化与等价性测试（2026-09-22）

范围：quant_evaluator 正式树（HEAD `fb0c6766c` 之上，未提交）。本文档是本轮
"指标级性能优化 + 逐项 AB 等价验证"的正式记录，遵循
`BOOTSTRAP_PERFORMANCE_20260922.md` 的报告纪律：同时给出输入规模、设备、
线程、缓存与端到端口径，等价性比较声明容差。

## 测量环境与口径

- 服务器 qs-compute-gpu-hk-01，CPU 单核（`OPENBLAS_NUM_THREADS=1`），
  冷热混合缓存（每组计时前有 1 次预热调用），`time.perf_counter` 取重复计时的最优值。
- 规模档位：S = T=250/N=50，M = T=750/N=100，L = T=1250/N=300，F=1。
- 「内核级」= 直接调用指标 compute 函数；「端到端」= 经过 `evaluate()` 公共入口。
- 等价性标准：**位级精确优先**（`np.array_equal(..., equal_nan=True)`）；
  确实达不到的项，按库内 GPU parity 家规 `rtol<=1e-8, atol<=1e-10` 封顶，
  并在测试 docstring 记录实测最大偏差（本轮全部 ≤1e-12 量级）。

## 优化明细

### 1. `metrics/ic.py` — compute_daily_ic 向量化（45 个指标共同上游）

- 方法：validity/pairwise mask/valid_counts 全面向量化；Spearman 秩用
  "+inf 哨兵 + `scipy.stats.rankdata(axis=1)`" 批量计算——哨兵排在最后，
  不参与并列，有效值的平均秩与压缩后逐行 rankdata **逐位一致**（300 组随机
  含并列含 NaN 样本实测 0 差）。逐行保留原 `np.corrcoef` 压缩路径，保证
  系数位级不变（ QE 证据用内容哈希做身份，位级不变避免下游 FO/FA 身份漂移）。
- 原实现保留为 `_compute_daily_ic_reference`（等价性 oracle）。
- 加速：Spearman 内核 S 2.3× / M 2.1× / L 1.6×（位级一致）；
  端到端 IC 族指标 L 规模 155–175ms → 102–104ms（~1.55×）。
  剩余瓶颈是逐行 corrcoef 的 Python 开销，位级约束下的合理下界。
- 新测试：`tests/metrics/test_daily_ic_vectorized_equivalence.py`，18 个
  （11 等价 + oracle 已知值 + 置换/正仿射不变性 + valid_counts 一致性）。

### 2. `metrics/quantile.py` / `shape_evidence.py` — quantile 族

- `_percentile_boundaries` 向量化（`_percentile_boundaries_from_sorted`），
  严格复现 1e-9 整数 snap 与溢出回退分支；
- `assign_quantiles_batch`：单次 `np.sort(axis=1)` + 整块 (T,F,nq-1) 向量化
  边界（逐 (t,f) 调用反而因 ~15µs 调用开销变慢 0.60×，已修正为整块）；
- `compute_adaptive_quantile_count`：候选 Q=20/10/5 构成整除链时由最细分箱
  直接推导粗分箱，全 panel 只 sort 一次；
- `compute_shape_bootstrap_confidence`：moving-block 索引按 factor 一次性
  fancy-index，nanmean 面板化。
- 加速（位级一致，oracle 同文件保留）：adaptive 1.53×（L）/1.37×（S）；
  assign_quantiles_batch 1.21×（L）/1.55×（S）；shape_bootstrap 1.20×。
- 新测试：`tests/metrics/test_quantile_perf_equivalence.py`，116 用例。

### 3. `metrics/portfolio_stats.py` / `underwater.py` / `turnover.py` /
    `probe_portfolio/sharpe.py` / `risk/drawdown_analysis.py` — 回测/组合族

| 项 | 方法 | 等价 | 加速 |
|---|---|---|---|
| aligned wealth curve | cumprod + maximum.accumulate 吸收归零 | 位级 | L 44.6× |
| worst period return (月/季/年) | sliding_window_view + prod(axis=1) | 位级 | 41.8 / 21.4 / 7.7× |
| rolling Sharpe（tail & quantile） | cumsum 闭式滚动均值/方差 | ulp≤2.6e-15 | L 444× |
| 秩权重矩阵 | rankdata(axis=1) 批量 + 哨兵 | 位级 | 18.2× |
| 秩换手估计 | 面板化 \|Δw\| 求和 | 位级 | 21.1× |
| 回撤事件状态机 | cumprod + RLE 切事件段 | 位级 | 18.6× |
| sharpe/sortino/maxdd/calmar | 全 (T,F) 面板化 | ulp≤5.3e-15 | 1.2–2.8×（F 越大越受益） |
| **long_short_returns cutoffs** | 单次 sort + 虚拟索引 + numpy `_lerp` 双分支精确复现 | **位级**（vs `np.nanquantile` 逐行） | 见下 |

- `compute_long_short_returns`：agent 先做面板化（桶均值为掩码求和，与
  压缩 `np.mean` 存在已记录 ulp 差，实测 ≤1e-15 量级）；随后把每日
  `np.nanquantile`（内部逐行 Python 循环，2500 次 np.quantile 调用占 154ms）
  替换为 `_batched_linear_quantile_cutoffs`：一次 sort + 向量化虚拟索引 +
  逐位复现 numpy `_lerp`（含 t≥0.5 的反向插值分支）。200 组 × 4 分位
  随机对照 0 差。
- 内核加速：**L 17.6×（198→11.3ms）、S 75.2×**；evaluate 端到端
  211ms → 41.5ms。
- 新测试：`tests/metrics/test_backtest_perf_equivalence.py`（170 用例）+
  `tests/metrics/test_long_short_cutoff_equivalence.py`（52 用例）。

### 4. 测试套件本身的提速

- `tests/metrics/test_coverage_compiler.py`：`compile_coverage_rows()`
  原被 3 个测试各自调用（每次 ~26s 全包扫描）。改为 session 级 fixture，
  断言一字未改；CSV 写入测试仍端到端真实执行。该文件 105s → 52s。

## 端到端效果（evaluate 公共入口，L 规模）

| 指标 | 优化前 | 优化后 | 加速 |
|---|---|---|---|
| rank_ic（及 IC 族 30+ 指标） | ~156ms | ~102ms | 1.55× |
| adaptive_quantile_count | 278ms | ~203ms* | 1.37× |
| long_short_returns | 218ms | ~42ms | ~5.2× |
| quantile_spread | 84ms | ~84ms* | —（JIT 内核已在用） |
| bonferroni_correction | 173ms | ~104ms | 1.66× |

\* 画像在 cutoff 优化后未重跑 adaptive 项，表值为 cutoff 优化前测量。

批内复用说明：同一 `evaluate()` 调用内，IC 序列已按 `(method, min_assets)`
去重（`ic_series_cache`），典型批量（全 spearman + 全 pearson）只算 2 次；
跨调用缓存（V2IntermediateCache）对 facade 指标暂不生效（`_cache_identity`
对 runtime.register_metric 注册的自定义函数返回 None），这是后续工作——
启用它会影响证据内容哈希身份，需与 FA/FO 的身份链一起设计，本轮不動。

## 真实场景观察（只记录，不改语义）

1. `missing_return_policy="zero_fill"`（缺收益记 0 = 平盘假设）在稀疏
   覆盖宇宙会把组合收益拉向 0，偏乐观；`drop` 才是缺测剔除。默认值是
   向后兼容选择，已写入 docstring；准入/研究侧应审计 coverage 而非依赖
   zero_fill 的平滑性。
2. 多空路径是 100% gross 等权、信号即执行、单日全额建仓成本，无滑点/
   部分成交/冲击成本模型——fast 路径收益不能冒充可执行策略证据，
   可执行口径在 vectorbt_qs accurate 路径。
3. Sharpe/CAGR 年化按等间隔观测 ×252，缺测日的 `drop` 口径会改变实际
   年化时钟，需与 coverage 指标联合解读。
4. cohort 持有期回测（`probe_portfolio/_core.py` `compute_cohort_pnl`）
   仍是 start×t 双重 Python 循环，为本轮**唯一未动的回测热点**（语义最
   敏感：per_side_cost、terminal_position_policy、trade_eligibility）。
   建议方案已写入其文件头注释与交接：差分数组构造日度权重矩阵 +
   面板化聚合，oracle 对拍 rtol=1e-8 守卫，预计随 T×H 线性放大收益。

## 新增测试清单（全部进默认 pytest 套件）

| 文件 | 用例数 | 等价标准 |
|---|---|---|
| tests/metrics/test_daily_ic_vectorized_equivalence.py | 18 | 位级 |
| tests/metrics/test_quantile_perf_equivalence.py | 116 | 位级 |
| tests/metrics/test_backtest_perf_equivalence.py | 170 | 位级为主，rolling-Sharpe 族 ulp≤2.6e-15 |
| tests/metrics/test_long_short_cutoff_equivalence.py | 52 | cutoff 位级；端到端 NaN 对齐 + 1e-12 |
| 合计 | **356** | — |

全量回归：优化前基线 1348 passed / 105.6s；最终全量（含新增 356 用例）
以 `/tmp/qe_final.txt` 回执为准（本文档撰写时正在运行，结果见交接汇报）。
