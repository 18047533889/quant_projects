# 后端覆盖率报告 — 伪 Polars 实测 / 频次加权改造清单 / SQL 下推验证

日期：2026-09-19（Asia/Hong_Kong）
工作树：`/home/sunhaiwei/quant_projects`（`qs-server-c`）
基线提交：`ac33a820`（step3/batch1）

---

## 0. 一句话结论

1. **伪 polars（delegate）确实没有任何加速**：它的 polars 槽位就是 `pl→pd → 跑同一份 pandas 内核 → pd→pl`，CPU/墙钟 = **1.00**（严格单线程），而原生 polars 表达式在同一面板上是 **1.80**。它永远不可能更快，因为它是 pandas 的超集。
2. 但**它也没有明显更慢**：16 个高频 delegate 算子实测倍率（委托/ pandas）**中位数 1.036x、最小 0.915x**，最坏 10.544x。相对开销来自一块固定的 marshaling 税（本面板 **约 2.1 ms/面板**），在计算量大的内核上被完全淹没（`ts_cusum_pressure` 0.995x，计算占委托耗时 100.5%）。唯一一个 10x 级异常（`ts_beta`）**不是 marshaling 造成的**，是该算子用了一个比 pandas 参考慢 9.9 倍的辅助内核（见 §1.6）。
3. **"全部删掉"是错的**：按清单统计，**113893 行因子中有 53870 行（47.3%）至少含一个 delegate 算子**。删掉 delegate 槽位后这些行在 `polars_long` 上会变 UNSUPPORTED，只能回退 pandas 参考路径，实测直接丢掉 `polars_long` 的 **1.94x–2.90x** 端到端加速。**"慢几个百分点"远好于"47.3% 的因子跑不了 polars_long"。**
4. 正确的做法是**修复而不是删除**，且要按证据区分两类工作：**(A) 531 个算子已有真 polars 内核、只是没声明 spec**（补齐即可接通通道）；**(B) 639 个算子内核确实走 pandas 往返**（要改写内核）。本报告给出频次加权的工作清单与覆盖率曲线，明确"剩下的 900 个值不值得做"。
5. 覆盖率关键数字：**当前全清单只有 1361/113893 = 1.19% 的因子行能完整跑**（要求一行里所有算子都是真实现）。改造 **前 50 个算子 → 28.61%**，**前 300 个 → 58.23%**，**前 750 个 → 90.18%**，**前 1000 个 → 98.81%**。**尾部 900 个非常值得做**——因为一行需要其全部算子可用，长尾直接决定"能不能跑"。
6. 顺带发现 3 个阻塞级问题（只报告，未改）：**polars/duckdb 生产准入对全部 1823 个算子都是关闭的**（证据工件失效，514 条哈希校验错误）；**`_data_source_kind()` 对真实 `DataAccessSource` 仍返回 `"memory"`**（`cleaned_bridge` 未修，`plan_cost_router` 的修复因缺 `capabilities` 属性而未生效）；**549 个有 SQL 槽位的算子中 0 个声明了 SQL 执行类型**。

---

## 1. 第 1 步：伪 polars（delegate）到底值不值得留 —— 实测

### 1.1 方法与环境

| 项 | 值 |
| --- | --- |
| 数据 | `~/cos_data/StockDailyBarAdj`（真实后复权日线，一文件一日） |
| 区间 | 2019-01-02 ～ 2022-12-30 |
| 面板 | **30 个标的 × 972 个交易日 = 29160 格**（脚本亦支持 25 标的 × 972 天复核） |
| 线程 | `POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4` |
| 环境 | `ASHARE_PARQUET_ROOT=$HOME/cos_data`、`DATA_ACCESS_SKIP_COS_MIRROR=1` |
| 路径 (a) | `pandas_numpy` 参考实现：`OperatorRegistry.get(op,"pandas_numpy").calculate(pd_panel,…)` |
| 路径 (b) | polars 委托槽位：`OperatorRegistry.get(op,"polars",mode="any")._calculate_series(pl_panel,…)` = `pl→pd → 同一份 pandas 内核 → pd→pl` |
| 计时 | 预热 1 次后按目标 0.6s 自适应迭代（≥3 次），报告 **中位数** 与 **最小值** |
| 面板形态 | polars 侧 **带 `date` 列**（与生产一致，`frame_time_index` 可解析），py/pd 两侧同一份真实数据 |

脚本：`evidence/_step1_delegate_bench_real.py`；原始结果：`evidence/_step1_delegate_bench_real.json`
算子选取：`_operator_classification.json` 与 `r57c_operator_frequency.json` 取交集，按频次降序取前 16 个 delegate 算子。

### 1.2 逐算子实测（委托 / pandas 倍率）

| 算子 | 频次(行) | 面板数 | pandas (s) | delegate (s) | **x(中位)** | x(最小) | NaN 掩码一致 | 最大绝对偏差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `ts_cusum_pressure` | 2863 | 1 | 0.74419 | 0.68592 | **0.922** | 0.995 | ✅ | 0.0 |
| `ts_trend_tstat` | 606 | 1 | 0.53723 | 0.53684 | **0.999** | 0.987 | ✅ | 0.0 |
| `ts_vol_of_vol` | 533 | 1 | 0.51258 | 0.50992 | **0.995** | 0.996 | ✅ | 0.0 |
| `cs_spline_resid` | 560 | 2 | 0.11435 | 0.10459 | **0.915** | 1.000 | ✅ | 0.0 |
| `intraday_volume_clock_path_efficiency` | 521 | 2 | 3.02582 | 2.97284 | **0.982** | 0.951 | ✅ | ⚠ 见下 |
| `cs_quantile_resid` | 805 | 2 | 1.51211 | 1.53284 | **1.014** | 1.014 | ✅ | 0.0 |
| `cs_isotonic_residual` | 772 | 2 | 0.08926 | 0.09295 | **1.041** | 1.041 | ✅ | 0.0 |
| `ts_edge_effective_spread` | 723 | 4 | 7.16624 | 7.43871 | **1.038** | 1.001 | ✅ | 0.0 |
| `ts_wasserstein_shift` | 613 | 1 | 2.65584 | 2.75738 | **1.038** | 1.013 | ✅ | 0.0 |
| `ts_hysteresis_state` | 512 | 1 | 0.05306 | 0.05564 | **1.049** | 1.047 | ✅ | 0.0 |
| `directional_change_state` | 552 | 1 | 0.03024 | 0.03349 | **1.107** | 1.097 | ✅ | 0.0 |
| `ts_spectral_flatness` | 618 | 1 | 1.36458 | 1.63700 | **1.200** | 1.005 | ✅ | 0.0 |
| `composition_normalized_entropy` | 1304 | **8** | 0.00602 | 0.01349 | **2.242** | 2.142 | ✅ | 0.0 |
| `ts_beta` | 525 | 2 | 0.00746 | 0.07861 | **10.544** | 10.664 | ✅ | 0.0 |
| `ashare_fiscal_quarter_from_period_end` ⚠ | 4096 | 1 | 0.06039 | 0.06242 | **1.034** | 1.033 | ✅ | 未测 |
| `fiscal_direction_consistency` ⚠ | 668 | 2 | 2.87337 | 2.97352 | **1.035** | 1.001 | ✅ | 未测 |

**汇总（14 个有效测量）**：中位数 **1.036x**，最小 **0.915x**，最大 **10.544x**，均值 1.697x。
> ⚠ 三个算子（`ashare_fiscal_quarter_from_period_end`、`fiscal_direction_consistency`、`intraday_volume_clock_path_efficiency`）的输入面板是日期/日内语义，我用价格面板绑定，两路输出**均为全 NaN**（`max_abs_dev` 记为"未测"）。其倍率只说明"同一段退化计算的两条路径耗时相当"，**不构成有效工作负载的吞吐对比**，请勿引用为性能证据。

**读数**：倍率 ≤ 1.05 的算子共 12/16；唯一的量级异常是 `ts_beta`（10.5x），而它的**绝对差只有 +71 ms**（0.0075 s → 0.0786 s）。

### 1.3 不可约的 marshaling 底线（同一面板实测）

| 项 | 耗时 |
| --- | ---: |
| `to_pandas_panel(pl)` | 0.000786 s |
| `from_pandas_panel(pl, pdf)` | 0.000710 s |
| **单面板往返合计** | **0.002146 s** |

即委托路径有一块 **≈2.1 ms / 面板** 的固定税，与内核计算量无关。这就是为什么"重内核几乎无损（0.99x）、轻内核明显吃亏"：

- `ts_cusum_pressure`：内核 0.701 s ≫ 税 2.1 ms → **0.995x**
- `composition_normalized_entropy`：8 个面板 × ≈0.97 ms = 7.7 ms 税，而内核只有 2.1 ms → **3.89x**

### 1.4 在 polars_long 管道里插入一个 delegate 的代价

| 场景 | 耗时 | 说明 |
| --- | ---: | --- |
| 6 元素原生 polars 懒链 `.collect()` | 0.000280 s | 基线 |
| ＋强制 materialize + pl↔pd 往返（不跑算子） | 0.003252 s | **打断原生链的纯代价 = +2.97 ms（11.6x）** |
| ＋插入真实 delegate 算子（`ts_cusum_pressure`） | 0.724487 s | 相比纯原生 collect **+0.7242 s** |

**解读（不要混用）**：+2.97 ms 是"打断原生链、必须 materialize"的代价——在一条**近乎零成本**的链上它显得是 11.6x，绝对量只有 3 ms。+0.7242 s 那一条**主要不是"打断"的代价**，而是 `ts_cusum_pressure` 本身在这块面板上就要 0.69 s 的单线程计算。两者必须分开引用。

### 1.5 单线程证据（这是"永远不可能更快"的根因）

| 路径 | CPU时间 / 墙钟时间 |
| --- | ---: |
| delegate `ts_cusum_pressure` | **1.001** |
| delegate `composition_normalized_entropy` | 1.285 |
| 原生 polars 30 列懒链 | **1.795** |

delegate 的 CPU/墙钟 = **1.00**，即**严格单线程**；同面板原生 polars 表达式能用到 **1.80 个核**。这与 `evidence/R57_LANDING_DIAGNOSIS_20260919.md` 记录的 CPU 占比（pandas 路径 ~100%、`polars_long` 358%）方向一致。**结论：delegate 在结构上不可能加速——它是同一份 pandas 内核 + 额外 marshaling，并放弃多核。**

### 1.6 10x 异常归因：`ts_beta` 的问题不在 delegate 包装，在它自己的辅助内核

| 分解项 | 耗时 |
| --- | ---: |
| pandas 参考 `MovingBeta._calculate_series`（向量化 30 列一次算完） | **0.007273 s** |
| marshaling 进口（2 面板） | 0.002041 s |
| 回写 pd→pl | 0.001181 s |
| **delegate 端到端** | **0.078875 s（10.85x）** |
| 残差（既非 marshal 也非帧校验） | **0.068380 s** |
| ↑ 其中：`compute_rolling_beta(…)` | **0.072260 s** |
| `panel_pandas_bridge` 纯桥接（恒等函数） | 0.001898 s |
| 逐列 `pl.Series(...)` 重建（30 列） | 0.000734 s |
| 一次性 `from_pandas` 重建（对照） | 0.001170 s |

**根因（已定位到源码）**：
`factor_engine/cleaned_operators/price_volume/polars_price_volume.py:78` 的 `_rolling_beta_polars` 调用
`factor_engine/cleaned_operators/price_volume/beta_helpers.py:22` 的 `compute_rolling_beta`，后者是
**逐列 Python 循环**：

```python
for col in ret.columns:                       # 30 次
    out[col] = rolling_beta(ret[[col]], bench[[col]], window=w, min_periods=mp)[col]
```

而 pandas 参考 `MovingBeta` 直接 `rolling_beta(y, x, window=w, min_periods=mp)` **一次向量化算完整面板**。
同一份数据、同一数学结果（NaN 掩码一致、maxdev 0.0），**一个 0.0723 s，一个 0.0073 s，差 9.9 倍**。
`panel_pandas_bridge` 本身不是瓶颈（恒等往返只要 1.9 ms）——我先前的怀疑被实测否定。

### 1.7 反事实：删掉 delegate 槽位会怎样

| 指标 | 值 |
| --- | ---: |
| 清单中 delegate 类算子 | **621 个** |
| delegate 算子出现次数 | **69705 次** |
| **含 ≥1 个 delegate 算子的因子行** | **53870 / 113893 = 47.3%** |
| 受影响因子在 polars_long 上的后果 | **UNSUPPORTED → 回退 pandas 参考路径** |
| 实测回退的机会成本（top400 票池） | 丢掉 **1.94x–2.90x** 端到端加速 |

polars_long 相对 pandas 的实测端到端加速（真实数据，top400 票池 × 2 年，`SPECS` 无关）：

| 因子 | pandas (s) | polars_long (s) | 加速比 |
| --- | ---: | ---: | ---: |
| `ts_mean(close,20)` | 3.817 | 1.347 | **2.83x** |
| `rank(close)` | 3.010 | 1.300 | **2.31x** |
| `cs_pct_rank(close)` | 3.474 | 1.306 | **2.66x** |
| `ts_sum(close,20)` | 3.194 | 1.322 | **2.42x** |
| `gt(close,10)` | 3.325 | 1.690 | **1.97x** |
| `tanh(close)` | 3.288 | 1.386 | **2.37x** |
| `multiply(rank(close), ts_std(close,20))` | 7.084 | 2.524 | **2.81x** |

### 1.8 第 1 步明确结论

> **Q：委托比 pandas 更快 / 持平 / 更慢？**
> **A：持平到略慢，从不更快。** 14 个有效测量中位数 **1.036x**（+3.6%），12/16 ≤1.05x；多面板轻内核最坏 3.9x；`ts_beta` 10.5x 另有他因（§1.6）。绝对开销是 ≈2.1 ms/面板的固定 marshaling 税。**CPU/墙钟 = 1.00 证明它不可能带来加速。**

> **Q：那就全删掉？**
> **A：不要。** 删除会让 **47.3%（53870 行）的因子在 polars_long 上直接跑不了**，被迫回退 pandas，实测丢掉 1.94x–2.90x 的端到端加速。**"慢 3.6%"换"47.3% 的因子失去 2–3 倍加速"是明显的亏本交易。**
> 正确的处置顺序是：
> 1. **保留 delegate 槽位**（保住可执行性）；
> 2. 把 delegate **如实标注**为 CPU delegate（现状已如此），不当作加速；
> 3. 按 §2 的频次加权清单，**优先把高频 delegate 改写成真 polars/numpy 内核**（§3 已交付第一批）；
> 4. 顺手修 §1.6 这类"委托路径用了更慢辅助内核"的 bug——这是零风险纯收益。

---

## 2. 第 2 步：频次加权改造工作清单 + 覆盖率曲线

产物：`evidence/factor_catalog_20260916/backend_coverage_worklist.json`（脚本 `evidence/_build_worklist_v2.py`）

### 2.1 口径校验（先证明"rows"是什么）

从 `r57c_20260919_final/factor_catalog_review_r57c_final.csv.gz` 的 `r57_formula` 正则解析算子集合，与 `r57c_operator_frequency.json` 逐算子对比：

| 校验项 | 结果 |
| --- | --- |
| 频次表中算子数 / 我解析到的算子数 | 1314 / 1316 |
| 共同算子中**计数完全一致**的个数 | **1314 / 1314** |
| 计数不一致的算子 | **0** |

**结论**：频次文件的 `rows` = **"包含该算子的因子行数"**，语义确认无误。
- 清单因子行：**113893**
- 公式里出现的不同算子：**1316**（其中 55 个名字未注册）
- 算子出现总次数：**740063**（频次表内 719717，平均 ≈6.3 个算子/行）

### 2.2 当前后端类别（按 1316 个算子，非 1823 canonical）

| 类别 | 算子树 | 出现次数 | 说明 |
| --- | ---: | ---: | --- |
| `no_explicit_physical_spec` | 549 | — | 生产模式 fail-closed 判 UNSUPPORTED；**多数其实有真内核** |
| `polars_pandas_delegate` | 610 | 69705 | pl→pd→pl 委托 |
| `other:delegate_python` | 11 | 904 | Python 委托 |
| `polars_numpy_kernel` | 60 | — | 真 numpy 列内核 |
| `polars_native_expr` | 31 | — | 真 polars 表达式 |
| `not_in_catalog` | 55 | — | 名字未注册（`field`、`safe_div`、`ts_return` 等 DSL 关键字/别名） |

**可改造性判定**（依据 = 注册表 `backend_meta[backend]["source"]` + **运行时 marshal 探针**：打桩 `pl.DataFrame.to_pandas` 计数，实跑真实数据）

| 判定 | 算子树 | 含义 |
| --- | ---: | --- |
| `already_real` | 91 | 已是真 polars/numpy/SQL 执行类型 |
| **`declare_spec_only`** | **531** | 静态检查无 `to_pandas()`/`_pl_to_pd`，**运行时探针 0 次 marshal → 真内核，只缺 `_physical_spec` 声明** |
| `rewrite_needed` | 639 | 内核确实经 pandas 往返 → 需改写为 polars expr / numpy 列内核 |
| `needs_inspection` | 55 | 名字未注册，转换前需人工核 |

### 2.3 覆盖率曲线（这是"值不值得做"的答案）

**指标定义**——两种，必须分开看：
- **出现次数加权**：Σ(已覆盖算子的出现次数) / 740063。宽松。
- **行门控（严格，真正决定"能不能跑"）**：一行因子**当且仅当它的全部算子都可用**才算能跑。当前基线 **1361/113893 = 1.19%**。

| 改造前 N 个算子 | 出现次数加权（增量） | 累计绝对 | **行门控：能跑的因子行** |
| ---: | ---: | ---: | ---: |
| 0（现状） | — | **43.04%** | **1361 行 = 1.19%** |
| 1 | +12.24% | 55.28% | 6793 行 = **5.96%** |
| 5 | +20.87% | 63.91% | 9262 行 = **8.13%** |
| 10 | +26.16% | 69.21% | 12100 行 = **10.62%** |
| 20 | +31.65% | 74.70% | 20413 行 = **17.92%** |
| 30 | +34.49% | 77.53% | 26266 行 = **23.06%** |
| 50 | +37.71% | 80.76% | 32584 行 = **28.61%** |
| 75 | +40.04% | 83.09% | 37092 行 = **32.57%** |
| 100 | +41.80% | 84.85% | 40690 行 = **35.73%** |
| 150 | +44.53% | 87.57% | 47208 行 = **41.45%** |
| 200 | +46.53% | 89.57% | 53677 行 = **47.13%** |
| 300 | +49.30% | 92.35% | 66317 行 = **58.23%** |
| 500 | +52.72% | 95.77% | 85006 行 = **74.64%** |
| 750 | +55.41% | 98.46% | 102706 行 = **90.18%** |
| 1000 | +56.78% | 99.82% | 112542 行 = **98.81%** |
| 1225 | +56.95% | 99.996% | 113863 行 = **99.97%** |

**怎么读这张表（决定剩下 900 个值不值得做）**：
- **值得做，而且必须做。** 行门控口径下：
  - 只做前 **50** 个 → **28.6%**
  - 只做前 **300** 个 → **58.2%**
  - 做满前 **750** 个 → **90.2%**
  - 做满前 **1000** 个 → **98.8%**
- 边际收益在 **50 → 750 区间是"缓降"而不是"急降"**（每 +100 个算子仍稳定带来 ≈+10~16 个百分点），**没有出现"做到 100 个就该停"的拐点**。
- 出现次数加权曲线看起来更"划算"（前 100 个就 84.85%），但它**高估了收益**：那些高频算子只是到处出现，缺一个低频算子照样让整行跑不了。**请以行门控列为准。**
- 前 30 个改造项（按频次）：`rank`(90565)、`cs_pct_rank`(17401)、`gt`(14929)、`ts_sum`(12429)、`price_impact`(9888)、`ts_zscore`(9110)、`and_`(6762)、`tanh`(6460)、`efficiency_ratio`(6070)、`lt`(5270)、`turnover_zscore`(4347)、`fin_quarter_from_cumulative`(3743)、`ts_delta`(3465)、`relative_volume`(2893)、`ts_spectral_entropy`(2812)、`log_positive_or_nan`(2779)、`intra_realized_skewness`(2567)、`ts_sharpe`(2283)、`ts_delay`(2164)、`cs_mad_zscore`(2059)、`amihud_illiquidity`(1903)、`intra_realized_variance`(1871)、`fiscal_pct_change`(1808)、`cs_mean`(1800)、`ts_max`(1766)…

每个算子的 `category` / `convertible_to` / `planned_way` / `basis`（含运行时探针原始计数）都在 JSON 里逐条可查。

---

## 3. 第 3 步：改造成真实现

### 3.1 本批（batch 2）：15 个最高频算子声明真 `PhysicalImplementationSpec`

**性质说明（务必注意）**：这 15 个算子**内核本来就是真 polars 表达式**（`pl.col(...)` + `with_columns`/`pl.concat_list`/`.over()`），只是**缺少 `_physical_spec` 声明**，导致 `canonical_polars_kind(production_mode=True)` 返回 **UNSUPPORTED**、`_polars_status` 返回 **unsupported**。本批**只补声明，不改内核**——与 batch1（commit `ac33a820`）同一模式，属于"把走不到的生产通道接通"。

| 算子 | 频次(行) | 类 | 声明 `execution_kind` | 翻转前 → 后 |
| --- | ---: | --- | --- | --- |
| `rank` | 90565 | `common/cross_sectional.py:RankPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `cs_pct_rank` | 17401 | `common/cross_sectional.py:CsPctRankPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `gt` | 14929 | `common/polars_auto.py:_ComparePolars`(工厂) | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_sum` | 12429 | `common/time_series.py:TSSumPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_zscore` | 9110 | `common/time_series.py:TSZScorePolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `and_` | 6762 | `common/polars_auto.py:AndPolarsAuto` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `tanh` | 6460 | `common/polars_extended.py:TanhPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `lt` | 5270 | `common/polars_auto.py:_ComparePolars`(工厂) | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_delta` | 3465 | `common/time_series.py:TSDeltaPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_sharpe` | 2283 | `common/polars_daily_native.py:TSSharpeNative` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_delay` | 2164 | `common/time_series.py:TSDelayPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `cs_mad_zscore` | 2059 | `common/polars_daily_native.py:CSMadZscoreNative` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `cs_mean` | 1800 | `common/cross_sectional.py:CrossSectionalMeanPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_max` | 1766 | `common/time_series.py:TSMaxPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |
| `ts_min` | 1381 | `common/time_series.py:TSMinPolars` | `POLARS_NATIVE_EXPR` | unsupported → polars_native |

合计覆盖 **177844 次算子出现**。补丁脚本：`evidence/_patch_batch2.py`（AST 定位类体插入，`--apply` 才落盘）。

**声明口径的诚实性**：
- `execution_kind=POLARS_NATIVE_EXPR` 有依据：内核体是 `pl.col`/`pl.concat_list`/`pl.mean_horizontal`/`.over()`，**运行时探针 0 次 `pl.DataFrame.to_pandas`**。
- **`supports_lazy` / `supports_streaming` 保持默认 `False`**——这些内核接收**已 materialize 的 `pl.DataFrame`** 并返回 eager frame，声明 lazy 会是超报。`materializes_full_panel=True`。
  > 附带发现：batch1 的 9 个逐元素算子声明了 `supports_lazy=True, supports_streaming=True`，但它们的 `_binary_colwise`/`_unary` 同样作用于 eager `pl.DataFrame`。**该声明疑似超报，建议复核（未改）。**
- `implementation_source_hash` 沿用 batch1 的**稳定明文身份标签**约定（如 `common.time_series:TSSumPolars:v1`），而非内容摘要——与 batch1 保持一致。**若生产准入实际要求 64 位内容摘要，这批与 batch1 都需要换成真实闭包哈希（未验证该要求）。**

### 3.2 数值等价验证（真实数据，独立验证）

命令：
```bash
ASHARE_PARQUET_ROOT=$HOME/cos_data DATA_ACCESS_SKIP_COS_MIRROR=1 \
PYTHONPATH=/home/sunhaiwei/quant_projects POLARS_MAX_THREADS=4 VER_NSYM=25 \
python3 evidence/_verify_batch_specs.py rank cs_pct_rank gt ts_sum ts_zscore and_ tanh lt \
  ts_delta ts_sharpe ts_delay cs_mad_zscore cs_mean ts_max ts_min
```
面板：**25 标的 × 972 交易日（2019-01-02 ～ 2022-12-30）**；产物 `evidence/_verify_batch_specs.json`。

| 算子 | NaN 掩码一致 | 最大绝对偏差 | 幂等（跑两次逐位一致） | marshal 调用 | polars 分类 |
| --- | :---: | ---: | :---: | ---: | :---: |
| `rank` | ✅ | 0 | ✅ | 0 | polars_native |
| `cs_pct_rank` | ✅ | 0 | ✅ | 0 | polars_native |
| `gt` | ✅ | 0 | ✅ | 0 | polars_native |
| `ts_sum` | ✅ | 0 | ✅ | 0 | polars_native |
| `ts_zscore` | ✅ | 4.43e-11 | ✅ | 0 | polars_native |
| `and_` | ✅ | 0 | ✅ | 0 | polars_native |
| `tanh` | ✅ | 0 | ✅ | 0 | polars_native |
| `lt` | ✅ | 0 | ✅ | 0 | polars_native |
| `ts_delta` | ✅ | 0 | ✅ | 0 | polars_native |
| `ts_sharpe` | ✅ | 1.90e-09 | ✅ | 0 | polars_native |
| `ts_delay` | ✅ | 0 | ✅ | 0 | polars_native |
| `cs_mad_zscore` | ✅ | 0 | ✅ | 0 | polars_native |
| `cs_mean` | ✅ | 2.73e-12 | **❌** | 0 | polars_native |
| `ts_max` | ✅ | 0 | ✅ | 0 | polars_native |
| `ts_min` | ✅ | 0 | ✅ | 0 | polars_native |

- **NaN 掩码：15/15 完全一致**（最重要项）。
- **最大绝对偏差**：10 个为 0，其余 3 个在 ≥1.9e-09 及以下（浮点求和顺序差异），远超任何研究显著性阈值。
- **幂等**：14/15 逐位一致。**`cs_mean` 两次运行结果不是逐位相同**——`pl.mean_horizontal` 的并行归约顺序不稳定。该算子行级差异量级 2.7e-12，**不影响研究结论，但违反"同输入同输出"的确定性要求**（清单里有 `determinism_verified` 契约位）。**建议按需修复，本批未改。**
- **marshal 调用 15/15 = 0**，佐证不是 delegate。
- 分类翻转 15/15 ✅。

### 3.3 引擎级 A/B：声明 spec 有没有带来加速？（诚实答案：**没有可测量的变化**）

同进程内对比（`evidence/_ab_engine_specs.py`，`SPECS=on` = 当前工作树，`SPECS=off` = 运行时摘掉本批 15 个类的 `_physical_spec`）：

固定票池（20 只 × 4 年）与 top400 票池（× 2 年）各跑 7 个因子 × 2 后端：

| 因子 | polars_long `SPECS=on` (s) | polars_long `SPECS=off` (s) | 差异 |
| --- | ---: | ---: | ---: |
| `ts_mean(close,20)` | 1.347 | 1.256 | +7.2% |
| `rank(close)` | 1.300 | 1.290 | +0.8% |
| `cs_pct_rank(close)` | 1.306 | 1.289 | +1.3% |
| `ts_sum(close,20)` | 1.322 | 1.392 | −5.0% |
| `gt(close,10)` | 1.690 | 1.684 | +0.3% |
| `tanh(close)` | 1.386 | 1.416 | −2.1% |
| `multiply(rank, ts_std)` | 2.524 | 2.432 | +3.8% |

（`AB_UNIVERSE=top400 AB_START=2021-01-04 AB_END=2022-12-30 AB_REPS=2`）

**结论：差异全部在 ±7% 噪声内、方向不一致 → 声明 spec 对引擎运行时间无可测量影响。**
- 原因之一：这些因子的运行时间被**数据访问主导**（`polars_long` 侧基本恒定在 1.26–1.42 s，与表达式复杂度无关）。
- 原因之二（已直接验证）：**即使摘掉 spec，`polars_long` 路径照常被使用**（`used_polars_long_path=True, used_polars_long_native=True`），说明 research 模式下这条路径不依赖本批声明。
- 所以本批的**真实价值是"能力元数据与事实对齐"**（分类器不再把真内核误判为 UNSUPPORTED），**而不是性能**。它的收益要等 §6.1 的证据工件门修好、生产准入打开之后才兑现。

**同时必须记录一个风险**：宽表面板内核级对比显示，`rank` 的 polars 实现比自己 pandas 参考**慢 77 倍**（234 ms vs 3.02 ms），`cs_pct_rank` 慢 **187 倍**（161 ms vs 0.86 ms），`cs_mad_zscore` 2.75x，`ts_zscore` 2.6x，`cs_mean` 4.9x。
但 top400 引擎级实测中 `rank` 的 polars_long 仍比 pandas **快 2.31x**，说明 **`polars_long` 走的是自己的长表表达式路径，没有调用这些宽表内核**。**因此上述慢内核当前不构成回归**；但若将来有路由把这些宽表实现放到热路径上，`rank`/`cs_pct_rank` 需要先重写内核（`pl.concat_list`+`list.eval(rank())` 是瓶颈）。

---

### 3.4 回归测试：本批**零新增失败**（严格前后对照）

只把本批改动的 5 个文件临时还原成 `HEAD` 版本跑一遍，再换回本批版本跑一遍，按**失败节点 ID** 逐条 diff（脚本 `/tmp/b2_test_ab.sh`，`git show HEAD:<path>` 还原 + `trap` 保证回滚，改后已校验 5 个文件 sha256 与备份一致）：

| | baseline（HEAD 版本） | after（batch 2 版本） |
| --- | ---: | ---: |
| 结果 | 307 failed, 1384 passed, 269 skipped (453.18s) | 307 failed, 1384 passed, 269 skipped (458.35s) |
| 唯一失败节点 | 307 | 307 |
| **本批新引入的失败** | — | **0** |
| 本批修好的失败 | — | 0 |

```
--- NEW failures introduced by batch-2 (in after, not in baseline) ---
（空）
--- failures FIXED by batch-2 (in baseline, not in after) ---
（空）
```

**读法**：那 307 个失败**全部是既有失败**（`backend_parity` + `operator_contracts` 两个目录），与本批无关；且**逐个节点集合完全相同**，不存在"数量相同但内容不同"的掩盖。请勿把 307 当作本批引入的回归。

---

## 4. 第 4 步：DuckDB / SQL 下推验证（只验证，未改源码）

脚本 `evidence/_step4_sql_verify.py`，产物 `evidence/_step4_sql_verify.json`。数据源：真实 `DataAccessSource`（`ashare_stock_daily_adj`，5 标的 × 2022H1）。

### 4.1 数据源类型判定（bug 现状）

| 检查 | 结果 |
| --- | --- |
| 数据源类 | `DataAccessSource`（MRO：DataAccessSource → DataSource → ABC） |
| **`hasattr(ds,"capabilities")`** | **False** |
| `plan_cost_router._data_source_kind(ctx)` | **`"memory"`** ❌（期望 `duckdb`） |
| `cleaned_bridge._data_source_kind(ctx)` | **`"memory"`** ❌（期望 `duckdb`） |

- `plan_cost_router.py:371-396` 的审计 #355 修复**确实已落地**（优先读 `capabilities.engine_kind`/`dialect`），但 `DataAccessSource` **没有 `capabilities` 属性**，于是掉进回退分支 `plan_cost_router.py:399-404` 的 `"data_access" in name` 子串判断——**类名 `DataAccessSource` 小写是 `dataaccesssource`，不含下划线，匹配失败 → 返回 `"memory"`**。**修复因缺属性而未生效。**
- `cleaned_bridge.py:399-408` 的 `_data_source_kind` **仍是旧子串逻辑，完全未修**（同样返回 `"memory"`）。
- → 你描述的"因 `"data_access"` 子串匹配失败把数据源误判成 `memory`"**在真实 `DataAccessSource` 上依然复现，两处都有**。

### 4.2 SQL 槽位与实现普查

| 指标 | 值 |
| --- | ---: |
| 带 SQL 槽位的 canonical 算子 | **549**（与你给的数字一致） |
| 其中**声明了 SQL 执行类型**（`DUCKDB_NATIVE_SQL`/`SQL_PYTHON_UDF`）的 | **0** |
| `_sql_status(duckdb_sql)` = `implemented`（research 口径） | 548 |
| `_sql_status(duckdb_sql)` = `production_safe` | **0** |
| `supports_sql(production)` 为 True 的算子数 | **0 / 1823** |

→ **549 个算子有能力标签但没有一条 SQL 实现声明**，SQL 能力的"可证明"面是空的。

### 4.3 SQL 下推是否生效（端到端）

| 后端 / 因子 | 结果 | SQL 运行时字段 |
| --- | --- | --- |
| `pandas` × 4 因子 | 全部 ok | — |
| `polars_long` × 4 因子 | 全部 ok | `used_polars_long_path=True, used_polars_long_native=True` |
| **`duckdb_sql` × 4 因子** | **全部 ok** | **`used_sql_pushdown=True`、`sql_fully_pushed=True`、`sql_query_count=1`、`sql_dialect='duckdb'`、`sql_full_execution_failed=False`、`sql_fallback_subtree_count=0`** |
| `auto` × 4 因子 | **全部失败** | `PhysicalPlanRequiredError: HybridBackend execution requires a planner-admitted PhysicalRegionPlan; FactorEngine has no physical-plan consumer wired` |

因子（真实数据，585 行结果）：`ts_mean(AdjClose,20)`、`add(ts_mean(…5), multiply(ts_std(…20),2))`、`rank(ts_mean(…10))`、`where(lt(AdjClose,10), ts_sum(…20), abs(AdjClose))`。

**结论**：
- ✅ **显式指定 `duckdb_sql` 时，SQL 全量下推确实生效**（4/4 因子的整棵计划编译为 1 条 SQL，`sql_fully_pushed=True`），且**未出现** `sql_full_execution_failed`。
- ⚠ 你提到的"auto 路由到 duckdb 会出现 `sql_full_execution_failed` 并回退"**在本次条件下没有复现**：`auto` 在更早的阶段就被**硬拒绝**（planner 未接通物理计划消费者），根本走不到下推。
- ⚠ 尽管 `_data_source_kind` 误判为 `"memory"`，显式 `duckdb_sql` 仍能下推——说明这条路径没有依赖该函数；该函数的误判主要会伤害**方言自动选择**（而 `auto` 目前被硬拒绝，所以暂时表现为潜伏问题）。

### 4.4 无权改的算子

本步**只做验证，未改动任何 SQL 源码**（且 `plan_cost_router.py` / `cleaned_bridge.py` 属你正在修的范围之外，但证据链需要你确认后统一处理）。

---

## 5. 改不动的算子 / 保留 kernel 路径

| 类别 | 数量 | 处置 | 原因 |
| --- | ---: | --- | --- |
| `polars_pandas_delegate`（含 `other:delegate_python`） | 621 | **保留 kernel 路径（不删）** | 删除会让 47.3% 的因子行在 polars_long 上 UNSUPPORTED（§1.7）。已如实标注为 CPU delegate。**只能逐个改写内核，不能整批删除。** |
| `not_in_catalog`（`field`、`safe_div`、`ts_return` 等 55 个） | 55 | **保留，需人工定级** | 是 DSL 字段/别名而非注册算子，不能当算子改造；需在 DSL 层归一化后重新统计。 |
| 需改写但本批未动的 `no_explicit_physical_spec` | 549−15=534 | **保留，列入后续批次** | 本批只做"内核已真、只缺声明"的 15 个。其余需逐个查内核：若内核实为 pandas 往返，则属 `rewrite_needed`，**不得靠补声明伪装成 native**。 |
| 宽表内核慢于参考实现的算子（`rank` 77x、`cs_pct_rank` 187x、`cs_mean` 4.9x、`cs_mad_zscore` 2.75x、`ts_zscore` 2.6x） | 5 | **保留声明，标注待重写内核** | 当前不在 polars_long 热路径上（§3.3 实测无回归），但重写优先级应高于普通算子。 |
| `compute_rolling_beta` 的逐列循环（`ts_beta` 等） | 1 处 | **未改，建议修** | 一行改动即可 9.9x（§1.6），但属共享辅助函数、影响面需评估，故只报告。 |

**未硬塞任何假实现**：本批 15 个算子全部是"内核本来就是真 polars 表达式、只是缺声明"，有运行时零 marshal 证据；任何内核实为 pandas 的算子**没有**被声明成 native。

---

## 6. 发现的其它问题（只报告，未改）

### 6.1 【阻塞级】polars / duckdb 生产准入对全部算子关闭 —— 证据工件失效

| 检查 | 实测 |
| --- | ---: |
| `_polars_status == "production_safe"` 的 canonical 数 | **0 / 1823** |
| `supports_polars(mode="production")` 为 True 的算子数 | **0 / 1823** |
| `supports_polars(mode="research")` 为 True 的算子数 | 1075 / 1823 |
| `evidence_artifact_validation_errors()` 条数 | **514** |
| 工件 `provenance.commit_sha` / `passed_at` | `46a50b5f…` / `2026-09-16T03:11:36Z` |

根因链（已读源码 + 实跑复现）：
`_polars_status`（`operator_capability.py:771`）只有当算子同时属于
`POLARS_REFERENCE_PARITY_VERIFIED` ∧ `POLARS_EDGE_VERIFIED` ∧ `POLARS_NO_FALLBACK_VERIFIED`
才返回 `production_safe`；而这三个集合由 `primitive_evidence.py:97-105` 的 `_load_verified_set()` 填充，该函数在 `evidence_artifact_valid()` 为假时**直接返回空 frozenset**。当前 `evidence_artifact_valid()` 为假（514 条错误），于是**三个集合全为空**（实测各 0 个元素）→ **任何算子都不可能 production_safe**。

错误样例（节选）：
```
emitter_hashes: expected={'implementation_hash_polars_emitter': 'b3e96647…', ...}
operator[add].implementation_hash_polars: expected='5feb89c5…' actual='2fbabffc…'
operator[add].implementation_hash_duckdb:  expected='1e8e955d…' actual='2732f830…'
operator[add].semantic_contract_hash:     expected='0d5fb740…' actual='7b61385c…'
operator[add].bridge_parameter_hash:      expected='468513d6…' actual='c469ecba…'
operator[add].test_source_hash:           expected='76610d9db445004' actual='c91a48d6479d3d3e'
```
> **注意**：其中 `implementation_hash_polars_emitter` 不匹配，而 `factor_engine/backend/polars_expr_emitter.py` **正是你当前正在修改的文件之一**。换言之，**只要继续改 emitter，这个工件就会持续失效，生产 polars 通道一直关着**。
> **处置建议（未执行）**：改完代码后按设计重跑证据认证流程（`certify_primitive_evidence.py` 对应的 pytest 流程）重新生成工件。我**没有**擅自重跑——它会改写一份已提交的认证工件，且当前工作树有多处未提交修改。
> **影响**：这一步不修，§3 的所有 spec 声明、以及"真 polars 真加速"在**生产口径**下都拿不到准入；research 口径下 `polars_long` 仍可用（§3.3 实测 1.94x–2.90x）。

### 6.2 【bug】`_data_source_kind()` 两处仍误判 `DataAccessSource` → `"memory"`

见 §4.1。`plan_cost_router` 的修复依赖一个**不存在的** `capabilities` 属性；`cleaned_bridge` 未修。
建议：给 `DataAccessSource` 补 `capabilities`（含 `engine_kind="duckdb"` / `dialect` / `supports_sql_pushdown`），并让 `cleaned_bridge._data_source_kind` 复用同一套 `capabilities` 逻辑，删掉子串启发。

### 6.3 【bug】`compute_rolling_beta` 逐列循环，比 pandas 参考慢 9.9x

见 §1.6。位置 `factor_engine/cleaned_operators/price_volume/beta_helpers.py:22-41`。
修法（一行）：直接用 `rolling_beta(ret, bench, window=w, min_periods=mp)` 向量化，替掉 `for col in ret.columns`。**需评估其它调用方是否依赖 `mp = max(2, w//3)` 的默认分支。**

### 6.4 【问题】`cs_mean` 非逐位确定性

见 §3.2。`CrossSectionalMeanPolars` 用 `pl.mean_horizontal`，两次运行结果不是逐位一致（量级 2.7e-12）。清单中存在 `determinism_verified` 契约位，建议按该契约要求处置。

### 6.5 【问题】batch1 声明 `supports_lazy/supports_streaming=True` 疑似超报

见 §3.1。batch1 的 9 个逐元素算子的内核作用于 eager `pl.DataFrame`。**未改**，建议复核声明口径。

### 6.6 【测试基建】`factor_engine/tests/` 存在与我改动无关的收集失败

| 测试模块 | 失败原因 | 是否我引入 |
| --- | --- | --- |
| `tests/operators/test_v9_bds_common_center.py` | `ModuleNotFoundError: No module named 'statsmodels'`（环境缺包，`python3 -c "import statsmodels"` 同样失败） | 否 |
| `tests/test_ts_batch1_rank_if_direct.py` | `ImportError: cleaned_operators.base is an alias … was installed in sys.modules`（模块别名导入顺序问题，单独运行触发） | 否 |
| `tests/runtime/test_r49_membership_churn.py` | 单独收集**无错误**；仅在与其他模块同批运行时因 `sys.modules` 别名污染而报错（测试顺序伪影） | 否 |
| `tests/tools/` 13 个收集失败 | `evidence.*` import 路径问题（你已确认为既有问题） | 否 |

### 6.7 【口径】`auto` 后端在公共 `FactorEngine.run` 上被硬拒绝

`PhysicalPlanRequiredError: HybridBackend execution requires a planner-admitted PhysicalRegionPlan; FactorEngine has no physical-plan consumer wired and refuses`。
与 `factor_engine/backend/factory.py:25-27` 的注释一致（"公共 FactorEngine run/run_many 系列尚未接通，显式请求会类型化拒绝"）。**这是设计现状，不是回归**，但意味着 `auto`（以及依赖它的"自动选最快正确路径"）目前不可用。

---

## 7. 复现命令

```bash
# 0) 公共环境
export ASHARE_PARQUET_ROOT=$HOME/cos_data DATA_ACCESS_SKIP_COS_MIRROR=1
export PYTHONPATH=/home/sunhaiwei/quant_projects POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4
cd /home/sunhaiwei/quant_projects

# 1) 第1步：逐算子 delegate vs pandas（真实数据 30 标的 × 972 天）
python3 evidence/_step1_delegate_bench_real.py
#    → evidence/_step1_delegate_bench_real.json / _step1_real.out

# 1b/1c) marshaling 底线、单线程证据、ts_beta 归因
python3 evidence/_step1b_overhead_decomp.py
python3 evidence/_step1c_ts_beta_decomp.py

# 2) 第2步：频次加权工作清单 + 覆盖率曲线
python3 evidence/_build_worklist_v2.py
#    → evidence/factor_catalog_20260916/backend_coverage_worklist.json / _worklist_v2.out

# 3) 第3步：批次2 补丁（干跑 / 落盘）
python3 evidence/_patch_batch2.py            # dry-run
python3 evidence/_patch_batch2.py --apply    # 落盘

# 3b) 数值等价独立验证（真实数据 25 标的 × 972 天）
VER_NSYM=25 python3 evidence/_verify_batch_specs.py \
  rank cs_pct_rank gt ts_sum ts_zscore and_ tanh lt ts_delta ts_sharpe \
  ts_delay cs_mad_zscore cs_mean ts_max ts_min
#    → evidence/_verify_batch_specs.json

# 3c) 引擎级 A/B：声明 spec 是否带来加速
SPECS=on  AB_UNIVERSE=top400 AB_START=2021-01-04 AB_END=2022-12-30 AB_REPS=2 \
  python3 evidence/_ab_engine_specs.py
SPECS=off AB_UNIVERSE=top400 AB_START=2021-01-04 AB_END=2022-12-30 AB_REPS=2 \
  python3 evidence/_ab_engine_specs.py

# 4) 第4步：SQL 下推验证（只读）
python3 evidence/_step4_sql_verify.py
#    → evidence/_step4_sql_verify.json
```

---

## 8. 局限 / 未测

**未测（不要当成已完成的结论）**：
1. **全量 113893 因子的编译/落值**未跑（按要求不动全量）。
2. **`auto` 后端的 SQL 下推/回退行为**未测到——`auto` 在 planner 阶段被硬拒绝（§6.7），因此"auto 路由到 duckdb 出现 `sql_full_execution_failed`"这一现象**本次未复现，也未否定**。
3. **生产准入要求 `implementation_source_hash` 是否为 64 位内容摘要**：未验证。本批与 batch1 都用明文身份标签。
4. **`supports_lazy=True` 是否真的能走 lazy/streaming**：未做流式执行验证；本批保守声明 `False`。
5. **§3.3 引擎级 A/B 的算子级敏感度不足**：`polars_long` 侧耗时被数据访问主导（恒定 1.26–1.42 s），因此"声明 spec 无加速"这个结论对**算子级成本**不敏感；算子级结论请以 §1.2 / §3.2 的内核级数据为准。
6. **全 universe（5461 标的）下的内核成本**：未测。`rank` 宽表内核 77x 的差距在大面板上会更显著；本次只证明它**当前不在 polars_long 热路径**。
7. **delegate 算子中 `other:delegate_python` 的 11 个**未单独计时。
8. **§4.2 的 549 个 SQL 槽位算子未按频次补齐实现**（本步为只读验证；补齐属后续批次）。

**已知测量口径提醒**：
- §1.2 的 `ashare_fiscal_quarter_from_period_end` / `fiscal_direction_consistency` / `intraday_volume_clock_path_efficiency` 三个算子的绑定是**退化**的（两路输出全 NaN），其倍率不可作为吞吐证据。
- 早期一次 Step-1 运行（修正前）使用了**不带 `date` 列**的 polars 面板，委托路径少做了索引恢复与索引一致性校验；§1.2 的全部数字均为**修正后**（带 `date` 列、与生产一致）的结果。

---

## 9. batch 3（2026-09-20）：再补 13 个高频算子的真 polars 声明 + harness 两处缺陷修复

### 9.1 本批范围（只做一批，按频次降序）

优先级口径与 batch 2 完全相同：`r57c_operator_frequency.json` 的**行频次降序** →
`backend_coverage_worklist.json` 的 `workqueue_by_frequency`。
本批取「`convertible_to == declare_spec_only` 且频次最高的 13 个」，
跳过 `not_in_catalog` 的 DSL 关键字（`field` / `safe_div` / `ts_return` / `source_col`），
也跳过 `rewrite_needed`（需重写内核，属下一批的"其次"类别）。

| # | 算子 | 频次(行) | polars 内核 | 声明位置 |
| ---: | --- | ---: | --- | --- |
| 1 | `price_impact` | 9888 | `PolarsLiquidityV2_price_impact` | `price_volume/polars_liquidity_v2.py` |
| 2 | `efficiency_ratio` | 6070 | `PolarsTechMisc_efficiency_ratio` | `technical/polars_tech_misc.py` |
| 3 | `turnover_zscore` | 4347 | `PolarsLiquidityV2_turnover_zscore` | `price_volume/polars_liquidity_v2.py` |
| 4 | `fin_quarter_from_cumulative` | 3743 | `PolarsFundamental_fin_quarter_from_cumulative` | `fundamental/polars_fundamental.py` |
| 5 | `relative_volume` | 2893 | `PolarsLiquidityV2_relative_volume` | `price_volume/polars_liquidity_v2.py` |
| 6 | `intra_realized_skewness` | 2567 | `IntradayPolars_intra_realized_skewness` | `intraday/polars_next_stage.py` |
| 7 | `amihud_illiquidity` | 1903 | `PolarsLiquidityV2_amihud_illiquidity` | `price_volume/polars_liquidity_v2.py` |
| 8 | `intra_realized_variance` | 1871 | `IntradayPolarsFull_intra_realized_variance` | `intraday/polars_intraday_full.py` |
| 9 | `ts_pct` | 1738 | `TSPctPolars` | `common/polars_ops.py` |
| 10 | `signed_sqrt` | 1427 | `SignedSqrtNative` | `common/polars_daily_native.py` |
| 11 | `sigmoid` | 1400 | `SigmoidPolarsAuto` | `common/polars_auto.py` |
| 12 | `intra_realized_kurtosis` | 1308 | `IntradayPolars_intra_realized_kurtosis` | `intraday/polars_next_stage.py` |
| 13 | `log_abs` | 1206 | `LogAbsPolarsAuto` | `common/polars_auto.py` |

合计 **40361 次算子出现**。补丁脚本 `evidence/_patch_batch3.py`（AST 定位 + 输出 unified diff，
经 `git apply --check` 后才 `git apply`）。**内核一行未改，只补声明。**

注入方式按各模块既有风格分三类：

* 工厂动态建类（`_register` / `_mk`）→ 在 `type(...)` 的类 dict 里按 canonical 注入 `_physical_spec`；
* 普通类体 → 类体内加 `_physical_spec = _batch3_native_spec(...)`；
* `polars_auto._register_unary`（一个工厂服务多个 canonical）→ 类体内按
  `canonical in _NATIVE_UNARY_SPEC_CANONICALS` 条件声明。

`polars_intraday_full.py` 另有一处**必要的顺序调整**：该模块 `_mk` 的**首次调用**（行 205）
早于它原有的中途 `contracts` import（行 273），因此把 import 提到文件顶部、
把 spec 表放到首次调用之前。第一版补丁没注意这点，`load_all()` 直接
`NameError: _NATIVE_SPECS`——已修。

### 9.2 分类翻转（13/13）

脚本 `evidence/_batch3_flip_check.py` → `evidence/_batch3_classification_flip.json`。

13/13 `canonical_polars_kind(production_mode=True)` 由 **unsupported → polars_native**；
每条的 `execution_kind=polars_native_expr`、`supports_lazy=False`、`supports_streaming=False`、
`materializes_full_panel=True`。

### 9.3 数值等价验证（真实数据，独立验证，**三个面板**）

本批含三类语义完全不同的算子，单一日线面板会让其中两类退化成**空验证**，故用三个面板：

| 面板 | 数据 | 规模 | 算子数 |
| --- | --- | --- | ---: |
| 日线 | `~/cos_data/StockDailyBarAdj` | 25 标的 × 972 交易日（2019-01-02 ～ 2022-12-30） | 9 |
| 分钟 | `~/cos_data/StockMinuteBarAdj` | 20 标的 × 5 个交易日 × 240 根分钟 bar（1200 bar） | 3 |
| fiscal | 真实日期轴 + 真实标的 + 真实季度 `SUM(AdjAmount)` 的 YTD 累计 | 25 标的 × 972 交易日 | 1 |

**先如实记录第一版的空验证**（这是本批返工的起点）：

* 3 个日内算子在 25×972 的**日线**面板上输出**两侧全 NaN**（`n_finite=0`，
  NaN 掩码比较平凡通过）→ 未验证；
* `fin_quarter_from_cumulative` 被喂价格面板，`nan_mask_identical=false` 且两侧无共同有限格 → 未验证。

因此另写两个专用 harness（`_verify_batch3_intraday.py` / `_verify_batch3_fiscal.py`），
**这 4 个算子在第一版里没有算作"已验证"**。

逐算子实测（NaN 掩码一致 / 最大绝对偏差 / 幂等 / marshal 调用 / 参与比较的有限格数 / polars ms / pandas ms / 倍率）：

| 算子 | 面板 | NaN 掩码 | 最大绝对偏差 | 幂等 | marshal | 有限格 | polars(ms) | pandas(ms) | 倍率 |
| --- | --- | :---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `price_impact` | daily | ✅ | 0 | ✅ | 0 | 23825 | 3.945 | 1.390 | 0.35x |
| `efficiency_ratio` | daily | ✅ | 0 | ✅ | 0 | 23800 | 5.838 | 1.582 | 0.27x |
| `turnover_zscore` | daily | ✅ | 1.00e-10 | ✅ | 0 | 23800 | 4.827 | 2.420 | 0.50x |
| `relative_volume` | daily | ✅ | 0 | ✅ | 0 | 23800 | 5.242 | 1.584 | 0.30x |
| `amihud_illiquidity` | daily | ✅ | 0 | ✅ | 0 | 23576 | 4.809 | 1.821 | 0.38x |
| `ts_pct` | daily | ✅ | 0 | ✅ | 0 | 24275 | 1.558 | 1.012 | 0.65x |
| `signed_sqrt` | daily | ✅ | 0 | ✅ | 0 | 24300 | 0.470 | 0.872 | 1.85x |
| `sigmoid` | daily | ✅ | 0 | ✅ | 0 | 24300 | 0.752 | 1.046 | 1.39x |
| `log_abs` | daily | ✅ | 8.88e-16 | ✅ | 0 | 24300 | 0.956 | 0.397 | 0.42x |
| `fin_quarter_from_cumulative` | fiscal | ✅ | 0 | ✅ | 0 | 24300 | 280.134 | 224.354 | 0.80x |
| `intra_realized_variance` | minute | ✅ | 7.29e-17 | ✅ | 0 | 100 | 5.021 | 266.275 | **53.03x** |
| `intra_realized_skewness` | minute | ✅ | 4.44e-15 | ✅ | 0 | 100 | 2.628 | 2.489 | 0.95x |
| `intra_realized_kurtosis` | minute | ✅ | 6.39e-14 | ✅ | 0 | 100 | 2.571 | 3.032 | 1.18x |

* **NaN 掩码 13/13 完全一致**（最重要项）。
* **最大绝对偏差**：8 个为 0，其余 5 个 ≤ 6.4e-14（浮点重排），远低于任何研究阈值。
* **幂等 13/13**。
* **marshal 13/13 = 0**：佐证不是 delegate。
* 有限格数全为真实非零量级（23576 / 100 / 24300），**不存在空比较**。

### 9.4 关于"加速倍率"：本表**不足以**支撑性能结论（诚实说明）

上表 ms 是**单次测量**（polars 侧 = 第 1 次调用，pandas 侧 = 第 3 次调用），
面板仅 24300 格、耗时落在 0.4～6 ms → **噪声主导**。
9 个日线算子里 7 个"更慢"（0.27x–0.65x）、2 个更快（`signed_sqrt` 1.85x、`sigmoid` 1.39x），
方向不一致，**不构成本批的性能结论**。

唯一远超噪声的是 **`intra_realized_variance`：polars 5.0 ms vs pandas 266 ms ≈ 53x**
（分钟面板）。原因是 pandas 参考实现逐 bar Python 循环，polars 走 `group_by + sum`。
此一条可作结论，其余不可。

**本批的真实价值仍是"能力元数据与事实对齐"**（与 §3.3 结论一致）：
把 13 个真 polars 内核从"生产模式 fail-closed 判 UNSUPPORTED"变成 `polars_native`。
性能收益要等 §6.1 的证据工件门修好、生产准入打开之后才兑现。

### 9.5 本批修掉的两个 harness 缺陷（都是 batch 2 harness 引入的）

| 缺陷 | 现象 | 修法 |
| --- | --- | --- |
| `_verify_batch_specs.fill_kwargs` 缺省值哨兵判定失效 | 判据写的是 `type(d).__name__ != "MISSING"`，而线上哨兵类型是 `_MissingDefaultType` → 守卫**从不触发**，`window` 等参数原样传哨兵，5 个算子直接 `OperatorParameterError` 中断验证 | 改为 `"missing" not in type(d).__name__.lower()` |
| `_verify_batch_specs.as_array` 按**位置**比较数组 | pivot 类内核的输出列顺序是 pivot 发现顺序，未必等于输入面板列顺序 → 会拿**不同标的互相比较**。本批在分钟面板上实测到伪结果：`max_abs_dev=26.07`、`nan_mask_identical=false`（内核其实逐位一致） | 两侧都按列名排序后再比较 |

第二处是**会得出错误结论**的缺陷。已复核 batch 2 的 15 个算子：它们全部是
`x.with_columns(...)`，输出列顺序 = 输入列顺序，因此 **batch 2 已发布的结论不受影响**；
但该缺陷必须修，否则任何 pivot 类算子的验证都不可信。

另修掉 batch 2 注入器留在 `common/polars_auto.py` 里的**重复 import + 重复常量**
（`DYN_NOTE__register_compare` 与 contracts import 各出现两次）。

### 9.6 回归测试：本批**零新增失败**（严格前后对照）

脚本 `evidence/_batch3_test_ab.sh`：只把本批改动的 9 个文件临时还原成 `HEAD` 版本跑一遍，
再用本批版本跑一遍（`trap` 保证回滚，跑完已校验 9 个文件 sha256 与快照一致），
按**失败节点 ID** 逐条 diff。范围与 batch 2 的 `/tmp/b2_test_ab.sh` **完全一致**
（`factor_engine/tests/backend_parity` + `factor_engine/tests/operator_contracts`），
因此两者可直接对比。

| | baseline（HEAD 版本） | after（batch 3 版本） |
| --- | ---: | ---: |
| 结果 | 307 failed, 1384 passed, 269 skipped (470.22s) | 307 failed, 1384 passed, 269 skipped (456.48s) |
| 唯一失败节点 | 307 | 307 |
| **本批新引入的失败** | — | **0** |
| 本批修好的失败 | — | 0 |

```
--- NEW failures introduced by batch-3 (in after, not in baseline) ---
（空）
--- failures FIXED by batch-3 (in baseline, not in after) ---
（空）
```

baseline 的 307/1384/269 与 §3.4 batch 2 已发布的 307/1384/269 **完全一致**，
说明两次用的口径相同、可比。那 307 个失败全部是既有失败（`backend_parity` +
`operator_contracts`），本批**逐个节点集合完全相同**，不存在"数量相同但内容不同"的掩盖。
跑完已校验 9 个改动文件 sha256 与快照一致（见 `_batch3_test_ab.sh` 输出尾部）。

补充的目标化单测（本批触及的算子对应的测试模块，一次跑完）：

```bash
python3 -m pytest \
  factor_engine/tests/operators/test_intraday_next_stage.py \
  factor_engine/tests/operators/test_polars_next_stage_parity.py \
  factor_engine/tests/backend/test_polars_fundamental.py \
  factor_engine/tests/operators/test_fundamental_flow_semantics_v2.py \
  factor_engine/tests/operators/test_catalog_field_recipes.py \
  -q -p no:cacheprovider --tb=short
```

> **踩坑记录**：第一版 A/B 把范围放到整个 `factor_engine/tests/` 并带上
> `ASHARE_PARQUET_ROOT` / `DATA_ACCESS_SKIP_COS_MIRROR`，结果会话在 75% 处
> **停滞 10 分钟以上**（输出文件字节数完全不增长）。原因是有若干测试模块在真实数据根
> 存在时会去扫整棵 parquet 树。**单测不要带真实数据环境变量**——只有算子级等价验证才需要。
> 另外整个 `factor_engine/tests/` 目录在无 `--continue-on-collection-errors` 时会因
> `tests/tools/` 等既有收集失败直接 `Interrupted` 中止，比较结果会变成空集。

### 9.7 覆盖率：**实测**，不再是推断

`evidence/_batch3_coverage.py` 用与 `_build_worklist_v2.py` **完全相同**的行门控
（一行因子当且仅当其 `r57_formula` 里**全部**算子都被覆盖才算能跑），
在 113893 行真实清单上重算，覆盖集 = `already_real` ∪ 本流已声明算子：
→ 产物 `evidence/_batch3_coverage.json`。

| 阶段 | 本批新增可覆盖 | 累计声明算子 | 行门控能跑的因子行 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| 现状 | — | 0 | 1361 | **1.19%** |
| batch 1 | +0 | 9 | 1361 | 1.19% |
| batch 2 | +15 | 24 | 11373 | **9.99%** |
| batch 3 | +13 | 37 | 19285 | **16.93%** |

**基线复现校验**：重算出的现状 = 1361 行 / **1.19%**，与 §2.3 已发布数字**完全一致**
→ 说明这套 gate 实现与当初一致，后面的数字可比。

**为什么 batch 1 的 9 个"没让覆盖率动"**：worklist 的 `already_real`（91 个）
**本来就把** `add / subtract / multiply / divide / abs / neg / log / exp / sign`
记为"已有真 polars 槽"，所以行门控口径早就把它们算进去了。
batch 1 改的是**生产准入**（`canonical_polars_kind(production_mode=True)`）——
**这个口径的行门控曲线测不出来**。这也是本批同时报"分类翻转"和"行门控"两个口径的原因。

对照 §2.3 已发布曲线（那是"假设把前 N 个全部改造"的**上界**，且未扣除 55 个永不注册的
DSL 关键字）：本批 37 个算子做到 **16.93%**，落在 landmark N=10（10.62%）与
N=20（17.92%）之间、略低于 N=20——差额正好由无法改造的非算子名字解释：
`field`(19102 行)、`safe_div`(6960)、`ts_return`(5094)、`source_col`(1244)。

### 9.8 本批未完成 / 卡点（需用户决策）

1. **§6.1 的生产准入闸门仍然关着，本批一样打不开**：`_polars_status == "production_safe"`
   仍是 **0 / 1823**。根因是证据工件失效（514 条错误），而其中
   `implementation_hash_polars_emitter` 不匹配，正是因为
   `factor_engine/backend/polars_expr_emitter.py` 属于**我无权改、也无权重跑认证**的文件。
   → **需要用户决定**：何时允许重跑证据认证流程、由谁负责。
2. `polars_liquidity_v2.py` 的 contracts import 是中途插入的（沿用该文件族既有风格；
   `intraday_full.py` 原本也这样，本批顺手提到了顶部）。若要求所有模块 import 集中在
   文件头，这是一次独立的整理任务。
3. 频次 1102–2863 区间的 `rewrite_needed` 算子（`zscore` 1517、`ts_spectral_entropy` 2812、
   `ts_cusum_pressure` 2863、`composition_normalized_entropy` 1304）**本批未动**——
   它们要真正重写内核，属下一批的"其次"类别。
4. `supports_lazy` / `supports_streaming` 仍一律 `False`，未做流式验证。
5. batch 1 的 9 个算子声明了 `supports_lazy/supports_streaming=True`（§6.5 的疑似超报）
   **本批未改**。
6. 行门控覆盖率只算了"本流声明的算子 + 已 real"，**未把 `rewrite_needed` 的重写纳入**。

### 9.9 本批复现命令

```bash
export ASHARE_PARQUET_ROOT=$HOME/cos_data DATA_ACCESS_SKIP_COS_MIRROR=1
export PYTHONPATH=/home/sunhaiwei/quant_projects POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4
cd /home/sunhaiwei/quant_projects

# 0) 候选内核解析（静态 token + 当前分类）
python3 evidence/_resolve_batch3.py price_impact efficiency_ratio turnover_zscore \
  fin_quarter_from_cumulative relative_volume intra_realized_skewness amihud_illiquidity \
  intra_realized_variance ts_pct signed_sqrt sigmoid intra_realized_kurtosis log_abs
#    → evidence/_resolve_batch3.json

# 1) 生成 / 落盘补丁（本脚本只出 diff，落盘走 git apply）
python3 evidence/_patch_batch3.py --emit /tmp/batch3.patch
git apply --check /tmp/batch3.patch && git apply /tmp/batch3.patch

# 2) 分类翻转
python3 evidence/_batch3_flip_check.py          # → evidence/_batch3_classification_flip.json

# 3) 数值等价（日线 25×972）
VER_NSYM=25 VER_OUT=/home/sunhaiwei/quant_projects/evidence/_verify_batch3_specs.json \
  python3 evidence/_verify_batch_specs.py price_impact efficiency_ratio turnover_zscore \
  relative_volume amihud_illiquidity ts_pct signed_sqrt sigmoid log_abs
# 3b) 分钟（20 标的 × 5 交易日 × 240 bar）
python3 evidence/_verify_batch3_intraday.py      # → evidence/_verify_batch3_intraday.json
# 3c) fiscal（真实日期轴 + 季度 SUM(AdjAmount) YTD）
python3 evidence/_verify_batch3_fiscal.py        # → evidence/_verify_batch3_fiscal.json

# 4) 回归前后对照（注意：不要带 ASHARE_PARQUET_ROOT）
bash evidence/_batch3_test_ab.sh

# 5) 覆盖率实测
python3 evidence/_batch3_coverage.py             # → evidence/_batch3_coverage.json
```
