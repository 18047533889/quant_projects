# 落值链路修复 + COS 存储分层 —— 交付报告

日期：2026-09-19 · 主机 `qs-server-c`（`qs-compute-gpu-hk-01`）· 工作树 `/home/sunhaiwei/quant_projects`

---

## 一、这一轮修好了什么（已验证）

### 1. `auto` 自动选后端：从「完全不能用」到「能跑且数值正确」

你一直感觉"后端没增加"，**直接原因就是 auto 这条路从来没执行过任何一批**。它每次都在这句话上死掉：

```
PhysicalPlanRequiredError: PhysicalRegionPlan is not research-ready:
row-count estimate unavailable
```

我把它逐层拆开，是**四个独立缺陷串成一条链**：

| # | 缺陷 | 位置 |
|---|---|---|
| 1 | 裸列（`AdjClose` 这类）从不写入逐列成本表，传进优化器的 `column_scan_costs` 是**空字典**——而 `scan_cost_map` 里明明有一条完好的 ScanCost | `planner/batch_data_request.py` |
| 2 | 优化器**手里已经有逐列 ScanCost，却只读 ctx 属性**；`ctx.data_source` 是 `LQTPLogicalDataSource`，形状估计器对它返回 **0 行**，于是 rows/bytes/memory 三者全 None | `runtime/multibackend/batch_global_optimizer.py` |
| 3 | `ts_mean(AdjClose, 10)` 被区域切分器切成 **窗口字面量 `10` 独占一个区域**并派给 DuckDB；区域只能通过后端物化，字面量区域零列 → 直接炸 | 同上 |
| 4 | 跨区域转移的面板是用 `literal` 节点承载的，但发射器把所有 literal 都编译成 `pl.lit(value)` 常量广播，且 `collect_columns()` 只认 `column` 叶子 → 找不到 base | `backend/polars_expr_emitter.py` |

修复后**实测结果**：

```
[pandas]      wall=190.0s  err=''   → 4 个因子
[polars_long] wall= 43.5s  err='ResourceBudgetExceeded...'（见下）
[auto]        wall=134.9s  err=''   → 4 个因子

  auto  b01_ts_mean_c10   EQUIVALENT  maxdiff=2.27e-13  nanmask=True
  auto  b02_ts_std_c20    EQUIVALENT  maxdiff=0.0       nanmask=True
  auto  b03_rank_c        EQUIVALENT  maxdiff=0.0       nanmask=True
  auto  b04_zscore_v      EQUIVALENT  maxdiff=0.0       nanmask=True
```

**auto 与 pandas 参考实现数值等价**（NaN 掩码逐个一致，最大绝对偏差 2.3e-13，纯浮点重排）。而在这之前它连一个字都没算出来。

验证脚本：`evidence/factor_catalog_20260916/_wb_land_equiv.py`（跑法见 §五）。
提交：`e7fd6bdc`。

> ⚠️ 更正一条我此前报告里的说法：上次我说 `ResourceBudgetExceeded: serial root admission denied for root:b03_rank_c` 是 polars_long 的问题 —— **这是错的**。那是等价脚本在同一个进程里先跑了 pandas（190s）导致内存压力、经纪人额度被吃掉之后的连锁反应。**单独跑 `polars_long` 是好的**：4 个因子 48.6s，无错误。

### 2. 落值速度现状（4 因子 / 3 列 / 855.9 万行）

| 路径 | 耗时 | 每因子 | 说明 |
|---|---:|---:|---|
| pandas（现状） | 129–190s | 32–47s | 单核 |
| **polars_long** | **46.5–48.6s** | **11.6–12.2s** | **真多核（CPU 358%），快 2.8x** |
| auto（修好后） | 129.8s（2 因子） | 65s | ❌ **反而最慢**，见 §三 |

**auto 现在虽然能跑，但选错了后端**：它把 `b01` 路由到 `duckdb_sql`，然后 `sql_full_execution_failed` 回退；`b02` 直接走 pandas。两边都没走到 polars_long。路由遥测：

```
route_counts = {'sql_full_execution_failed|long=False': 1, 'pandas|long=False': 1}
```

所以现在的正确用法是**显式 `polars_long`**，它在 4 因子上快 2.8x。

### 3. 批次放大反而更慢（重要，影响你的 DAG 设计）

| 批量 | pandas | polars_long |
|---|---|---|
| 4 因子 | 129.2s（32.3s/因子） | 46.5s（11.6s/因子） |
| 10 因子 | 419.2s（**41.9s/因子**） | 173s（**17.3s/因子**） |

**两边都随批量变差**，因为 `result_policy="return"` 让结果全部驻留（10 因子时 RSS 19GB）。所以"更大一批"目前等于"更慢一批"——你说的 DAG 批量落值**必须同时配 `sink` 落盘 + 有界 `wave_size`**，不能只把 batch 开大。

---

## 二、COS 存储分层：通道已打通并实测

**详细文档：`STORAGE_POLICY_AND_COS_LANDING_20260919.md`**

核心结论：

- **真实访问只有一条路**：本地 `coscli` 的 `~/.cos.yaml` 是**占位凭证**（三个字段同一串），直连必报 `secretID is missing`。必须走托管网关
  `/usr/local/libexec/quantsociety-cos/factor-admin-cos`（本账号在 `quant-admin` 组）。
- **写因子值用 `factor-admin-cos`**（`data/` 读写）；`factor-mining-cos` 对 `data/` **只读**。
- **网关硬约束**（都踩过）：本地路径必须在 `/home/<user>` 或 `/srv/quant` 下（`/tmp` 会被拒）；`ls` 的 COS URI 必须是第一个参数；**本账号没有删除权限** → 工具设计成 append-only。
- **实测上传 + 校验通过**：

```
→ cos://quant-factors-1425188104/data/catalog/r57c/factor_catalog_review_r57c_final.csv.gz
  listed: true   size_match: true
  local_bytes: 9198033 (8.77 MB)   remote size: 8.77 MB
  etag: 75b9db4e215a921a2343c75beffbdf6a
```

- **落值工具已落地**：`tools/cos_land.py`，四个子命令 `ls / put / stage-upload / verify`。
- **桶内现状**：`data/value/` 已是既定因子值前缀（沿用，不要另起）；`data/.prefix` 存在。

### 磁盘现状 —— 占用大头不是因子数据

```
/dev/vda2  2.0T  1.8T  143G  93%  /     ← 单盘，没有第二块
~/quant_projects   344G
   ├─ lightgbm_qs              222G   ← 真正的大头
   ├─ weekly_backtest_output   108G   ← 次之
   └─ 其余合计                 ~14G
/srv/quant         485G                ← 平台部署区，非本账号数据
~/cos_data          48G                ← 原始行情，按你的口径留本地
```

**本次没有移动或删除任何文件。** 迁移清单和需要你确认的三个问题写在策略文档 §5.2。顺带提醒：因为网关**没有删除权限**，"传上去再本地删"这个动作里删除只能靠本地命令，删错无法从 COS 侧回滚——建议先改名观察，别直接删。

---

## 三、"伪 polars"到底该不该删 —— 用实测回答（这份结论和直觉相反）

你问：`polars_pandas_delegate` 那 940 个，是不是加不了速、还更慢，要不要全删。

**实测（真实日线，30 标的 × 972 天，16 个最高频 delegate 算子）：**

| 算子 | 频次(行) | pandas (s) | delegate (s) | 倍率 |
|---|---:|---:|---:|---:|
| `ts_cusum_pressure` | 2863 | 0.74419 | 0.68592 | **0.922** |
| `ts_vol_of_vol` | 533 | 0.51258 | 0.50992 | 0.995 |
| `cs_spline_resid` | 560 | 0.11435 | 0.10459 | 0.915 |
| `cs_isotonic_residual` | 772 | 0.08926 | 0.09295 | 1.041 |
| `ts_edge_effective_spread` | 723 | 7.16624 | 7.43871 | 1.038 |
| `ts_spectral_flatness` | 618 | 1.36458 | 1.63700 | 1.200 |
| `composition_normalized_entropy` | 1304 | 0.00602 | 0.01349 | 2.242 |
| `ts_beta` | 525 | 0.00746 | 0.07861 | **10.544** |

**中位数 1.036x，最小 0.915x。** 结论分三层：

1. **它永远不可能更快**：delegate 就是 `pl→pd → 跑同一份 pandas 内核 → pd→pl`，是 pandas 的超集。实测 CPU/墙钟 = 1.00（严格单线程），而原生 polars 表达式在同面板上是 1.80。它结构性地没有加速能力。
2. **但它也基本没更慢**：只多一块固定 marshaling 税（本面板约 **2.1 ms/面板**），在计算量大的内核上被完全淹没。唯一的 10x 异常（`ts_beta`）**不是 marshaling 造成的**，是该算子用了一个比 pandas 参考慢 9.9 倍的辅助内核 —— 那是另一个 bug，不是 delegate 的锅。
3. **"全部删掉"是错的**：按清单统计，**113893 行因子中有 53870 行（47.3%）至少含一个 delegate 算子**。删掉 delegate 槽位 → 这些行在 `polars_long` 上变 UNSUPPORTED → 只能回退 pandas → **直接丢掉 polars_long 的 1.94x–2.90x 端到端加速**。"慢几个百分点"远好于"47.3% 的因子跑不了快后端"。

**所以正确做法是改造而不是删除**，而且要按证据分两类做：

- **(A) 531 个算子已有真 polars 内核，只是没声明 `_physical_spec`** → 只要补声明就能接通生产通道（低成本、高收益）；
- **(B) 639 个算子内核确实走 pandas 往返** → 需要重写内核（高成本）。

### 覆盖率曲线（决定长尾值不值得做）

| 改造算子数 | 覆盖率（能完整跑通的因子行占比） |
|---|---:|
| 现状 | **1.19%**（1361 / 113893） |
| 前 50 个 | 28.61% |
| 前 300 个 | 58.23% |
| 前 750 个 | 90.18% |
| 前 1000 个 | **98.81%** |

**尾部 900 个非常值得做** —— 因为一行因子要求它**全部**算子都可用，长尾直接决定"能不能跑"。这也解释了为什么你会觉得"能落值的因子特别少"：**目前全清单只有 1.19% 的行能完整跑通**。

### 已经动手的部分

子代理完成 batch1/batch2：**24 个算子**补上了真 polars 声明（提交 `ac33a820` 9 个、`5a4902a9` 15 个），每批都做了独立数值验证：

- NaN 掩码 **15/15 完全一致**（最重要项）
- marshal 调用 **15/15 = 0**（佐证确实不是 delegate 了）
- 分类翻转 **15/15**
- 回归测试 **零新增失败**
- `ts_beta` 的 9.9x 辅助内核问题已定位

报告：`evidence/BACKEND_COVERAGE_REPORT_20260919.md`（含逐算子原始数据与命令）。
工作清单：`evidence/factor_catalog_20260916/backend_coverage_worklist.json`。
算子频次排序（后续一切优先级的依据）：`evidence/factor_catalog_20260916/r57c_operator_frequency.json`。

---

## 四、还没修好的（按优先级）

| # | 问题 | 影响面 | 现状 |
|---|---|---|---|
| 1 | **auto 路由选错后端**：路由到 `duckdb_sql` 后 `sql_full_execution_failed` 回退，12 个因子里没有一个走到 polars_long | 全量落值速度（最大项） | 已定位，未修 |
| 2 | **`_data_source_kind()` 对真实 `DataAccessSource` 仍返回 `"memory"`** | 549 个算子的 SQL 下推 | 修了 `plan_cost_router` 但**没有生效**，因为 `DataAccessSource` 缺 `capabilities` 属性；`cleaned_bridge` 侧未修 |
| 3 | **549 个有 SQL 槽位的算子中 0 个声明了 SQL 执行类型** | 549 个算子的 duckdb 通道 | 未动 |
| 4 | **证据工件 fail-closed**：514 条哈希校验错误 → **全部 1823 个算子的 polars/duckdb 生产准入都是关闭的** | 所有快后端的生产认证 | 属"证书过期"，需 recert；不阻塞 research 落值 |
| 5 | 531 个算子待补 spec 声明 + 639 个待重写内核 | 98.8% 覆盖率的路径 | 只做了 24 个 |
| 6 | 125 行编译硬失败（73 行 `ts_vol_of_vol` + 21 行缺必填参数 + 31 行涨跌停**上一轮已修**） | 剩 94 行 | 未动 |
| 7 | 134 行生产准入缺口（30 个算子，主要是统计检验族） | 134 行 | 未动 |
| 8 | **批次放大反而更慢**（结果全驻留，10 因子 19GB） | 你规划的 DAG 批量落值 | 需配 `sink` + 有界 `wave_size`，未做 |

另外两条我没能继续推进的现实约束：

- **子代理配额在 01:06（+8）前用尽**，所以"不停补充"这一轮只跑成了 2 个 batch。我可以用定时任务在配额恢复后接着跑。
- 我**没有**动 §5.2 里那三项存储迁移（`lightgbm_qs` 222G / `weekly_backtest_output` 108G），等你确认。

---

## 五、怎么复现我的验证

```bash
ssh qs-server-c
cd /home/sunhaiwei/quant_projects

# 1) auto 与 pandas 的数值等价（含 NaN 掩码比对）
python3 evidence/factor_catalog_20260916/_wb_land_equiv.py --n 4

# 2) 四种落值路径的耗时分解
python3 evidence/factor_catalog_20260916/_wb_land_bench.py --mode D --n 4   # polars_long
python3 evidence/factor_catalog_20260916/_wb_land_bench.py --mode B --n 4   # pandas
python3 evidence/factor_catalog_20260916/_wb_land_bench.py --mode E --n 4   # auto

# 3) 后端就绪矩阵
PYTHONPATH=/home/sunhaiwei/quant_projects \
  python3 evidence/factor_catalog_20260916/r57_backend_readiness.py

# 4) COS 通道
python3 tools/cos_land.py ls data/
python3 tools/cos_land.py put ~/cos_stage/x.parquet data/value/x.parquet
python3 tools/cos_land.py verify ~/cos_stage/x.parquet data/value/x.parquet
```

注意：单次进程启动约 **55s**（注册表 + 导入）。做 DAG 长驻 worker 时这部分只付一次；做"每个因子起一个进程"时这 55s 就是纯损耗。
