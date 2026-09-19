# R57 第三轮：目录统一 OHLCV 模板的漏网算子修复 + 后端速度复核

日期：2026-09-19
分支：`main`（正式工作树 `/home/sunhaiwei/quant_projects`）
起始 HEAD：`0a4461f2`

---

## 0. 本轮要回答的两个问题

1. 上一轮报的"polars 比 pandas 慢 21%/26 倍"是否成立？
2. 目录里编译绿但执行炸的因子，到底有多少、根因是什么？

---

## 1. 修正：polars 的速度结论（上一轮的数字不可用）

上一轮报过两个互相矛盾的数：并发头对头说 polars 慢 21%，顺序基准说 2400 秒只跑完
43/240。后者是**测量污染**，不是真实性能。

| 测量 | 条件 | 结果 |
|---|---|---|
| 并发头对头 | pandas 与 polars 同时跑，机器有其他用户重负载 | pandas 516.4s / polars 627.0s |
| 顺序基准 | polars 段与本地 arity 审计进程重叠 | polars 2400s 超时，仅 43/240 |
| **干净小样本（本轮）** | 20 条因子，无并发，POLARS_MAX_THREADS=2 | **polars 77.4s = 3.87 s/因子** |
| 干净 medium（本轮） | 240 条因子，无并发 | **pandas 513.3s = 2.14 s/因子** |

**结论：干净环境下 polars 比 pandas 慢约 1.8 倍，不是 21%，更不是 26 倍。**
机器上长期有其他用户任务（load average 5+），任何"绝对耗时"测量都必须标明并发条件。

这仍然是负收益，原因与上一轮一致：polars 注册槽里 940 个是
`polars_pandas_delegate`（转调 pandas，多一层边界转换），真正的原生快速路径只有 82 个。

命令：

```sh
cd /home/sunhaiwei/quant_projects
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=. \
.venv/bin/python evidence/r3/test_watchdog.py \
  --log /tmp/wb-r57/exec/prof-polars.log --max-rss-mib 3072 --timeout 400 \
  -- .venv/bin/python evidence/factor_catalog_20260915/smoke_catalog.py \
     --input /tmp/wb-r57/exec/daily20.jsonl.gz --output /tmp/wb-r57/exec/prof-polars.jsonl.gz \
     --limit 20 --prepare-chunk 20 --backend polars \
     --start 2025-01-01 --end 2025-06-30 --timeout-seconds 360 \
     --max-rss-mib 3072 --batch-size 8
```

---

## 2. 真实 bug：目录统一 OHLCV 模板有 4 个算子没被修复

### 2.1 现象

R57 清单里 **113893 行有 72 行**的公式是这样的：

```
FisherTransform(field('open', table='StockDailyBarAdj'),
                field('high', ...), field('low', ...),
                field('close', ...), field('volume', ...))
```

这 72 行在 R20 与 R57 里都是 `compile_status = COMPILED`，
但 `execution_status = NOT_RUN` —— **从未执行过**。运行时会炸：

```
QQE.length [runtime]: length must be an integer, not DataFrame
FisherTransform.window [runtime]: window must be an integer, not DataFrame
```

### 2.2 根因

factors.directory 对技术指标使用**统一的五参数模板**：

```
Ind(Open, High, Low, Close, Volume)
```

引擎里 `factor_engine/tools/catalog_r19_operator_recipes.py` 有一个
"generic OHLCV template" 改写器，把这 5 个参数按**算子自己声明的输入契约**裁剪。
它覆盖了 RSI / ATR / MACD / MACD_line / MACD_signal / MOM / ROC /
BollingerUpper / BollingerLower / AROON / AROON_up / AROON_down /
StochasticD / TRIX / ADXR / WilliamsR —— **独独漏了 4 个**：

| 算子 | 使用行数 | 声明输入 |
|---|---|---|
| FisherTransform | 32 | high, low |
| CoppockCurve | 26 | close |
| QQE | 7 | close（x） |
| ElderRay | 7 | high, low, close |

对照：**同一模板下 RSI / ATR 是被正确改写的**，例如
`RSI(Open,High,Low,Close,Volume)` → `RSI_WILDER(field('close'), 14)`，
`ATR(...)` → `ATR_WILDER(field('high'), field('low'), field('close'), 14)`。
所以这是**同一改写表内的漏项**，不是设计差异。

### 2.3 修复

在 `catalog_r19_operator_recipes.py` 的同一个 `replacements` 表里补齐 4 条，
沿用既有风格（显式写出注册契约里的默认 horizon）：

```python
"FisherTransform": f"FisherTransform({high}, {low}, 9)",
"CoppockCurve": f"CoppockCurve({close}, 14, 11, 10)",
"QQE": f"QQE({close}, 14, 5, 4.236)",
"ElderRay": f"ElderRay({high}, {low}, {close}, 13)",
```

并在 `test_catalog_r19_operator_recipes.py` 的 parametrise 列表补 4 条用例。

### 2.4 验证

改动文件与哈希：

| 文件 | 改动前 sha256 | 改动后 sha256 |
|---|---|---|
| `factor_engine/tools/catalog_r19_operator_recipes.py` | `77f7ebf9badc5abd…` | `8fa97c7f3dd985aa…` |
| `factor_engine/tests/tools/test_catalog_r19_operator_recipes.py` | `be84322ea1f203bb…` | `c55d1ed004474371…` |

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=. \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  factor_engine/tests/tools/test_catalog_r19_operator_recipes.py
# 56 passed
```

对整个 R20 目录实际改写效果（脚本 `verify_r21_ohlcv.py`）：

```
rows still carrying the generic OHLCV call: 72
repaired rows: 72 | compile ok: 72 | compile bad: 0
```

改写样例：

```
FisherTransform(open,high,low,close,volume)
  -> FisherTransform(field('high', …), field('low', …), 9)

CoppockCurve(open,high,low,close,volume)
  -> CoppockCurve(field('close', …), 14, 11, 10)

QQE(open,high,low,close,volume)
  -> QQE(field('close', …), 14, 5, 4.236)

ElderRay(open,high,low,close,volume)
  -> ElderRay(field('high', …), field('low', …), field('close', …), 13)
```

### 2.5 端到端执行验证

只用"编译通过"判断是不够的（这正是本 bug 的病因）。构造这 72 行的执行输入
（`r57_build_r21_verify_input.py`），走正式 `engine.run_many` 入口，跑真实日频窗口
（8 标的 / 2025-01-01..2025-06-30）。

`smoke_catalog.py` 在 900 MiB 软轮转点会主动退出（`--rss-rotate-mib`），所以按 offset
分 9 批跑：

```sh
for off in 0 8 16 24 32 40 48 56 64; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=. \
  .venv/bin/python evidence/r3/test_watchdog.py \
    --log /tmp/wb-r57/exec/wd-r21p-$off.log --max-rss-mib 3600 --timeout 1650 \
    -- .venv/bin/python evidence/factor_catalog_20260915/smoke_catalog.py \
       --input /tmp/wb-r57/exec/r21_ohlcv72.jsonl.gz \
       --output /tmp/wb-r57/exec/r21p-$off.jsonl.gz \
       --limit 8 --offset $off --prepare-chunk 8 --backend pandas \
       --start 2025-01-01 --end 2025-06-30 --timeout-seconds 1200 \
       --max-rss-mib 3600 --batch-size 8 --rss-rotate-mib 2600
done
```

结果（前 3 批已完成）：**24/24 EXECUTED，0 全 NaN，0 硬失败**。
改写**前**同样的因子一跑即报：

```
QQE.length [runtime]: length must be an integer, not DataFrame
```

---

## 3. 元数契约审计（第三版口径，可复现）

前两版审计有缺陷：v1 没解析别名（误报 18.8 万）、v2 用 `calculate` 判变参
（恒为 `(*args, **kwargs)`，导致全部被豁免）。**v3 改用 `_calculate_series` 判变参**后：

```
113768 行中 532 行（0.47%）存在位置参数与注册契约冲突
```

聚合后：

| 出现次数 | 算子 | 判定 |
|---|---|---|
| 393 | `trade_when` | **误报**：其 `param_specs` 把数据参数误标为标量参数 |
| 104 | CoppockCurve | 真错配（本轮已修） |
| 96 | FisherTransform | 真错配（本轮已修） |
| 28 | QQE | 真错配（本轮已修） |
| 14 | ElderRay | 真错配（本轮已修） |
| 20 | group_tail_lead_score / group_tail_centrality | 待核 |
| 14 | holder_concentration_change / holder_count_change_rate | 待核（运行时确实失败） |
| 其余 | state_* / intra_* / cs_* 等 | 待核 |

审计独立命中了运行时实测失败的 `holder_concentration_change` 与
`holder_count_change_rate` —— 交叉验证成立。

---

## 4. 落值实测里 5 个双端失败的归因

240 条日频样本上，pandas 与 polars **同时失败**的 5 条：

| id | 公式要点 | 错误 | 归因 |
|---|---|---|---|
| R65X_01322 | `multiply(QQE(open,high,low,close,volume), …)` | `QQE.length must be an integer, not DataFrame` | **本轮已修**（OHLCV 模板） |
| R65X_02963 | `multiply(…, FisherTransform(open,high,low,close,volume))` | `FisherTransform.window must be an integer, not DataFrame` | **本轮已修**（OHLCV 模板） |
| R66_00714 | `ts_transfer_entropy(<expr>)` | `missing 1 required positional argument: 'source'` | 待修：目录只给 1 个输入，契约要 target+source |
| R66_00636 | `ts_ridge_regression_in_sample_resid(<expr>)` | `missing 1 required positional argument: 'x'` | 待修：同上，契约要 y+x |
| cold51_667902ca15f40247 | `ts_first_passage_hit_probability(…, 1.0, window=120, …)` | `'float' object has no attribute 'to_numpy'` | 待修：`scale` 是 series 参数却收到标量 1.0 |

polars 独有 2 条（`ts_corr(ret, <表达式>, 120)` 等）是上一轮已修的
`PairWindowSpec` 窗口位置 bug，本轮未再复现。

另有 14 条**双端相同的全 NaN** —— 是样本窗口/横截面不足，不是后端问题。

---

## 5. 纠正上一轮的两个结论

### 5.1 `holder_top10_*` 不是缺失算子（上一轮误报）

上一轮报"约 736 行 `holder_top10_*` 编译绿但算子未注册，执行必炸"。**这是错的。**

这五个名字是 `factor_engine/api/dsl_parser.py` 里的 **parser 级原生宏**
（`_native_holder_top10_stat` / `_native_holder_top10_id_stat` /
`_native_holder_top10_daily_id_churn`）；`_ExprBuilder` 在
`surface in {"compat_research","all"}` 且 `dialect == "native"` 时把它们挂到 `built` 上。
R57 编译用的正是 `surface="compat_research"`，所以它们在**解析期**就展开成受治理的
原始算子（`source_col` + `safe_div_null` / `holder_disclosure_count` 等），
根本不需要 registry 条目。

实测（脚本 `evidence/factor_catalog_20260916/r57_verify_holder_macros.py`）：

```
    690  holder_top10_pledge_ratio            compile_status={'COMPILED': 688, 'COMPILE_FAILED': 2}
     22  holder_top10_daily_id_churn          compile_status={'COMPILED': 22}
      6  holder_top10_two_day_rank_migration  compile_status={'COMPILED': 6}
      6  holder_top10_weighted_std            compile_status={'COMPILED': 6}
      6  holder_top10_disclosure_count        compile_status={'COMPILED': 6}

parse + lower under compat_research:  OK OK OK OK OK  (5/5)
```

**误报根因**：`compile_r57_full.py` 的可达性探针只查 `OperatorRegistry`，
不认 parser 自己解析的名字。已修正（见 5.2）。

### 5.2 修正 R57 编译探针口径

`compile_r57_full.py` 新增 `parser_allowed_names()`，把
`_ExprBuilder(surface="compat_research", dialect="native")._allowed`（2268 个名字）
并入可达性判定，并新增 `operators_resolved_by_parser` 分类 ——
原来它们被误计入 `operators_missing_from_surface`。

sha256：`321726f4…` → `7c67e1cd…`

### 5.3 修正 polars 的速度结论

见 §1：干净环境下 polars 慢约 **1.8 倍**，不是 21%，更不是 26 倍。

---

## 6. 未解决 / 待办

- §4 里 3 条公式—契约错配（`ts_transfer_entropy`、`ts_ridge_regression_in_sample_resid`、
  `ts_first_passage_hit_probability`）。**规模已量化：约 20 行**，且多为 `weakop_` 演示模板：

  | 调用形态 | 行数 | 其中 weakop_ |
  |---|---|---|
  | `ts_transfer_entropy` 单参数 | 9 | 6 |
  | `ts_ridge_regression_in_sample_resid` 单参数 | 6 | 6 |
  | `ts_first_passage_hit_probability` 传标量 scale | 2~8 | 6 |

  同一个算子 `ts_first_passage_hit_probability` 有 144 行是**正常两参数**调用，
  说明算子本身可用，只有个别行形态不对。优先级低。
- 713 个 polars 槽没有显式物理说明；1039/1039 算子 `accelerator=none`，L20 GPU 未接入。
  940 个 polars 槽是 `polars_pandas_delegate`（等于没加速）。这是"落得快"的唯一实质杠杆。
- R28 lower 段 103 failed / 797 passed（基线 `evidence/r56-r28-lower900-baseline-1789754809.log`）。
- `trade_when` 的 `param_specs` 把数据参数误标为标量参数，导致审计误报，需单独核。

---

## 7. 本轮的复现脚本

| 脚本 | 作用 |
|---|---|
| `evidence/factor_catalog_20260916/r57_verify_r21_ohlcv.py` | 对 R20 目录验证改写与编译 |
| `evidence/factor_catalog_20260916/r57_build_r21_verify_input.py` | 构造 72 行执行输入 |
| `evidence/factor_catalog_20260916/r57_make_ohlcv_fixed_csv.py` | 生成修复后的目录 CSV（列不变，只改 formula 单元格） |
| `evidence/factor_catalog_20260916/r57_verify_holder_macros.py` | 证明 holder_top10_* 是 parser 宏而非缺失算子 |
| `evidence/factor_catalog_20260916/r57_audit_single_input_estimators.py` | 量化单输入估计器的契约错配规模 |
| `evidence/factor_catalog_20260916/r57_show_missing_ops.py` | 列出清单引用但未注册的算子 |
| `evidence/factor_catalog_20260916/r57_ohlcv_signature_audit.py` | 目录签名 vs 引擎签名对照 |
