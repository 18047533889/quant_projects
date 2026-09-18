# R57 第二轮进展：落值实测、后端就绪矩阵、双序列窗口 bug

日期：2026-09-19
范围：本轮在正式工作树上的改动与实测，全部可复现。

---

## 1. 直接回答"其他算子都改好了吗 / 更快的后端弄好了吗"

**没有。** 下面是实测口径，不是估计。

### 1.1 编译 ≠ 能落值

R57 全量编译：113893 条非空 ID → **113768 COMPILED（99.89%）/ 125 COMPILE_FAILED**。

但编译通过不等于能算出值。用正式入口
`evidence/factor_catalog_20260915/smoke_catalog.py`（真实 DataAccessSource + `engine.run_many`）
在 8 个标的、2025-01-01 ~ 2025-06-30 上跑 240 条**日频**因子：

| 后端 | EXECUTED | 全 NaN | 硬失败 | 成功率（有结果） |
|---|---|---|---|---|
| pandas | 221 | 14 | 5 | 97.9% |
| polars | 219 | 14 | 7 | 97.1% |

- **14 个全 NaN 在两个后端是完全同一批 ID** → 是样本窗口/横截面不足，不是后端差异。
- **5 个两个后端都失败**，是真实缺陷（不是后端问题）：
  - `R65X_01322` `QQE(open, high, low, close, volume)` → `QQE.length` 收到 DataFrame
  - `R65X_02963` `FisherTransform(open, high, low, close, volume)` → `FisherTransform.window` 收到 DataFrame
  - `R66_00714` `ts_transfer_entropy(x)` → `_calculate_series()` 缺 `source`
  - `R66_00636` `ts_ridge_regression_in_sample_resid(x)` → `_calculate_series()` 缺 `x`
  - `cold51_667902ca15f40247` `ts_first_passage_hit_probability(..., 1.0, ...)` → `'float' object has no attribute 'to_numpy'`
- **2 个 polars 独有失败**：见第 2 节，已定位并修复。

### 1.2 后端就绪矩阵

`load_all()` 后统计 polars 槽位的物理实现类别：

| 类别 | 数量 |
|---|---|
| `polars_pandas_delegate`（转调 pandas，等于没加速） | 940 |
| `NO_EXPLICIT_PHYSICAL_SPEC`（无物理实现说明） | 783 |
| `polars_numpy_kernel` | 58 |
| `polars_native_expr` | 24 |
| `delegate_python` | 17 |
| 无 polars 槽 | 1 |

- **显式加速器声明：1039/1039 = `none`。GPU（L20 46GB）完全没有接入任何算子。**
- 全库搜 `cupy` / `cudf` / `torch.cuda`：生产路径 0 命中（只有测试与一份尚未实现的 numba 规划候选测试）。
- `polars_long` 流式策略：native 260 / compatible 316 / python_rolling 23 / stateful 19，但实测
  `backend_path.used_polars_long_path` **全部为 False** —— 流式长路径一次都没被走到。

按目录依赖加权（113893 行因子的算子调用次数）：
**有物理说明 193492 次 vs 无物理说明 546571 次**，即约 **74% 的算子调用落在纯 pandas 委托上**。

### 1.3 实测吞吐

同一批 240 条因子、同一窗口、两后端并发：

| 后端 | wall | 峰值 RSS |
|---|---|---|
| pandas | 516.4 s | 1.21 GB |
| polars | 627.0 s | 1.24 GB |

**polars 比 pandas 慢约 21%。** 这与"940 个 polars 槽只是转调 pandas"一致：多了一层
pandas↔polars 边界转换开销，却没有换来原生执行。

### 1.4 静默缺陷：编译绿、执行必炸

`holder_top10_pledge_ratio`(690)、`holder_top10_daily_id_churn`(22)、
`holder_top10_two_day_rank_migration`(6)、`holder_top10_weighted_std`(6)、
`holder_top10_disclosure_count`(6)，合计 **716 行**编译状态为 `COMPILED`，
但这些 canonical **没有注册任何后端**（`backends_for()` 为空）。这类行编译绿、执行必失败，
是最危险的静默问题。

---

## 2. 本轮修复：polars 双序列算子的窗口解析 bug

### 2.1 现象

`cold57_2c4a7c40a63b242a` 与 `cold_d2a997f7697d84e2` 在 polars 上失败，pandas 上成功：

- `rank(ts_corr(ret, subtract(...), 60))` → `PlanParamError: window 必须为整数 literal，收到动态输入 'subtract'`
- `ts_corr(ret, log_returns(volume), 120)` → `... 收到动态输入 'ts_log_return'`

### 2.2 根因（探针实证）

`factor_engine/backend/window_spec.py::WindowSpec.from_plan_node` 用启发式推断窗口：
从 `inputs[1:]` 里找**第一个既不是 column/materialized_series/plan_ref 的子节点**，要求它是
整数字面量。

`ts_corr` 的 plan 节点结构是：

```
op = ts_corr
  input[0] op='column'    (x = ret)
  input[1] op='subtract'  (y = 派生表达式)   <-- 被当成 window
  input[2] op='literal'   value=60           <-- 从未被看到
```

双序列算子的**第二个操作数本身就是 series 表达式**，于是被误判为 window。
`ts_corr(close, volume, 20)` 之所以正常，只是因为第二个操作数恰好是裸列（被跳过）。

pandas 侧按 Python 调用签名绑定参数，不受此启发式影响 → **同一公式两后端行为不可互换**。

受影响算子：`ts_corr`、`ts_cov`/`ts_covariance`、`ts_beta`、`ts_regression_slope/intercept/resid/r2`、
`ts_ewm_corr`、`ts_ewm_cov`。触发条件：第二个操作数不是裸列。

### 2.3 修复

让双序列契约**声明**窗口位置，而不是猜：

- `WindowSpec.from_plan_node(..., window_input_index=None)`：给出下标时只在该下标取值，
  且仍要求它是整数字面量；未给出时保持原启发式 → **一元算子的动态窗口依然严格拒绝**。
- `plan_params.window_spec_from_plan_node(..., window_input_index=None)`：透传。
- `PairWindowSpec.from_plan_node`：传 `window_input_index=2`（双序列参数是 `x, y, window[, min_periods]`）。

改动文件与哈希：

| 文件 | 前 | 后 |
|---|---|---|
| `factor_engine/backend/window_spec.py` | `e05a6399…` | `8121329f…` |
| `factor_engine/backend/plan_params.py` | `76283e5d…` | `0af92ee0…` |
| `factor_engine/backend/pair_window_spec.py` | `498c894a…` | `fa6d3b1f…` |

### 2.4 验证

探针（真实 FactorEngine + polars 后端 + 真实 A 股窗口）：

| 用例 | 修复前 | 修复后 |
|---|---|---|
| `ts_corr(ret, subtract(...), 60)` | `PlanParamError` | RUN OK |
| `ts_corr(ret, log_returns(volume), 120)` | `PlanParamError` | RUN OK |
| `ts_corr(ret, subtract(...), 120)` | `PlanParamError` | RUN OK |
| `ts_corr(close, volume, 20)`（对照） | RUN OK | RUN OK |

misfire 计数：3 → 0。

---

## 3. 下一步优先级（按对"尽可能快落值"的杠杆排序）

1. 修 5 个双端真 bug：`QQE`/`FisherTransform` 的实参元数契约、`ts_transfer_entropy`、
   `ts_ridge_regression_in_sample_resid` 缺参、`ts_first_passage_hit_probability` 位置参数。
2. 把 14 个全 NaN 的样本补成合法窗口/横截面（不改断言、不放宽门槛）。
3. **补高频算子的原生物理实现**：按依赖量 `rank 90579`、`multiply 62803`、`ts_std 15209`、
   `ts_sum 12429`、`ts_zscore 9110`、`ts_mean 8239`、`ts_corr 4307`、`ts_delta 3465`、
   `ts_delay 2465`、`ts_sharpe 2283` …… 目标是把 940 个 delegate + 783 个无说明收敛掉。
4. 接通 `holder_top10_*` 或在编译期直接报错（不得留"编译绿执行炸"）。
5. R28 lower 段 104 个失败（796/900 通过）继续收敛。

---

## 4. 复现命令

编译（分片）：

```sh
cd /home/sunhaiwei/quant_projects
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  evidence/factor_catalog_20260916/launch_r57_full.py \
  --source evidence/factor_catalog_20260916/factor_catalog_review_r20_final.csv.gz \
  --source-sha 80044269dc29b209dfae5136488bde50adcfc283dd8ee2d90147746ce89463d5 \
  --outdir evidence/factor_catalog_20260916/r57_20260919_full --shards 8 --max-slots 6
```

落值实测（单后端，逐一跑以获得无干扰数字）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 PYTHONPATH=. .venv/bin/python \
  evidence/r3/test_watchdog.py --log /tmp/wd-$BE.log --max-rss-mib 3072 --timeout 2400 \
  -- .venv/bin/python evidence/factor_catalog_20260915/smoke_catalog.py \
     --input /tmp/wb-r57/exec/daily240.jsonl.gz --output /tmp/seq240-$BE.jsonl.gz \
     --limit 240 --prepare-chunk 240 --backend $BE \
     --start 2025-01-01 --end 2025-06-30 --timeout-seconds 120 \
     --max-rss-mib 3072 --batch-size 8
```

后端就绪矩阵：

```sh
PYTHONPATH=. .venv/bin/python /tmp/backend_readiness.py evidence/factor_catalog_20260916/r57_20260919_full
```

---

## 5. 追加：全量位置参数契约审计（113768 行）

`engine.compile()` **不校验位置参数元数与算子注册契约的一致性** —— 这是"编译绿、执行炸"的机制性原因。
为此写了一个只读静态审计（`audit_arity*.py`），遍历每一行的 AST 调用，把位置实参逐个对到
算子的 `metadata.param_names` 上，检查：元数超限、把 series 表达式落进标量参数、未知关键字。

### 5.1 三轮迭代（前两轮的结论是错的，记录以免重复）

| 版本 | 缺陷 | 症状 |
|---|---|---|
| v1 | 未解析别名；未识别变参算子 | 18.8 万次假 `OPERATOR_NOT_REGISTERED`（`multiply`/`subtract` 等别名） |
| v2 | 用 `calculate` 判定变参 | `calculate` 恒为 `(*args, **kwargs)` → **所有算子**被豁免，误报归零 |
| v3 | 改用 `_calculate_series` 判定变参，并解析别名 | 结果可信 |

v3 结果：**113768 行中 532 行（0.47%）命中 `SERIES_INTO_SCALAR_PARAM`**，无元数超限、无未知关键字。

### 5.2 交叉验证成立

审计独立命中了运行时实测失败的 `holder_concentration_change`（7 行）与
`holder_count_change_rate`（7 行）—— 静态审计与动态实测互相印证，不是纯启发式噪声。

### 5.3 真正的契约错配（约 287 行，需修）

同一族问题：**OHLCV 类技术指标被按"多序列"调用，但算子契约是"单序列 + 标量参数"**。

| 算子 | 命中行数 | 契约 | 目录实际调用 |
|---|---|---|---|
| `CoppockCurve` | 104 | `close, roc1, roc2, wma_window, roc_mode` | 4 个字段实参塞进 roc1/roc2/wma_window/roc_mode |
| `FisherTransform` | 96 | `high, low, window, smooth, signal_smooth, output` | 已在运行时证实必失败 |
| `QQE` | 28 | `x, length, smooth, factor, output` | 已在运行时证实必失败 |
| `ElderRay` | 14 | `high, low, close, ema, output` | ema/output 收到字段 |
| `holder_concentration_change` / `holder_count_change_rate` | 7 / 7 | `concentration, lag` | 与运行时失败一致 |
| `signed_power` | 5 | `x, c` | c 收到字段 |
| `intra_event_pre_post_contrast` / `intra_event_window_reduce` | 5 / 5 | pre/post/... 为标量 | 标量位收到字段 |
| `state_l2_partial_adjustment` / `state_l1_turnover_prox` / `state_rank_deadband` / `state_adaptive_slew_limit` | 各 4 | — | 同上 |

### 5.4 另一类独立缺陷：算子参数元数据本身写错（约 413 处调用）

这些不是公式错，是**算子的 `param_specs` 把数据参数误标成标量参数**，会让任何基于
参数域的校验/优化/搜索逻辑失真：

| 算子 | 调用数 | 问题 |
|---|---|---|
| `trade_when` | 393 | `param_specs = ['signal','fallback']`，但 `signal` 显然是序列 |
| `group_tail_lead_score` | 10 | `param_specs` 误含 `x`、`group_id`（两者都是序列） |
| `group_tail_centrality` | 10 | 同上 |

### 5.5 结论

- "编译绿、执行必炸"的真实规模是 **约 287 行 / 113768 行（0.25%）**，集中在 4 个 OHLCV 指标
  （CoppockCurve / FisherTransform / QQE / ElderRay，合计 242 行）与若干 state/intra 族算子。
- 另加第 1.4 节的 `holder_top10_*` 716 行（算子未注册），是另一种独立的静默缺陷。
- 两类合计约 **1000 行 / 113768 行（0.88%）**需要在执行前修掉，其余 99% 的编译结果在
  "依赖的算子都真实存在且元数自洽"这一层是干净的。
