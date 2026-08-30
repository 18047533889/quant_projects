# 未来函数 / 口径错位 / 历史错误总集（防再犯手册）

> 孙海崴，这是把咱们这条因子项目里**犯过的所有错误**（未来函数、对不齐、口径错、时点错、以及流程类踩坑）逐条总结成的防再犯手册。以后每次改动回测/训练/因子计算的任何环节，**先对照本清单过一遍再动手**。
>
> 最后更新：2026-08-29

---

## 一、未来函数 & 数据泄露类（最严重，共 5 条）

### 1. mu/cov 用了未来收益 ← 吹出 Sharpe 2.24 假象的主因
- **错误**：决策日 d 估均值/协方差时，收益窗口用了 `ret[d] = Vwap[d+1]/Vwap[d] − 1`（明天才知道的收益）。
- **后果**：Sharpe 被虚高。修复后同参数复跑 2.24 → ~1.72，**几乎全部水分来自这一条**。
- **规则**：d 日做决策，一切统计量（mu、cov、rank_ic、标准化）**只允许用到 d−1 及之前的数据**（窗口截到 `pos−1`）。

### 2. 标签窗口Off-by-one（shift(-10)/shift(-1) vs 官方 shift(-11)/shift(-1)）
- **错误**：本地旧标签 = `AdjVwap[t+10]/AdjVwap[t+1] − 1`，COS 官方 `TargetVwapReturnH10` = `AdjVwap[t+11]/AdjVwap[t+1] − 1`（T+1 建仓、T+11 平仓）。本地这三个版本全错：`fwd_ret10`（未复权、含当日）、`fwd_ret10_adj`（少 1 天）、`fwd_adj_neu` 初版（shift(-10)）。
- **后果**：000001.SZ 2016-01-04 差 0.04；900 样本 881 个 bad。
- **规则**：**标签一律从 COS `TargetVwapReturnH10` 直接读，或用 `shift(-11)/shift(-1) − 1`**，并与官方对拍 diff=0 才准跑。禁止 shift(-10)。

### 3. w(d) 当日生效 / shift(1) 吃当日收益
- **错误**：`strat = (w.shift(1) * ret).sum()` 中 ret 定义为 `Vwap[t]/Vwap[t−1]`，导致 w(d) 隐含"当天收盘下单当天成交"，赚了 [d, d+1] 的收益；且换手率统计读错位显示 0.000。
- **规则**：**d 日预测 → d+1 日 VWAP 成交**。回测收益必须 = `w(d).shift(1) * ret(d)`，其中 `ret(d) ≡ Vwap[d+1]/Vwap[d] − 1`（即 `adj.pct_change().shift(-1)` 口径），且换手统计用的权重矩阵与生效日一致。

### 4. 旧的未复权标签把除权跳空当成收益
- **错误**：`fwd_ret10.parquet` 用未复权 Vwap，000001.SZ 2016-06 送转 −17.8% 被当成真实收益训练/评估。
- **规则**：**任何收益都必须用后复权 AdjVwap/AdjClose**。未复权 Close 只允许出现在 LQTP 输入侧（平台口径如此），评估/回测侧禁止。

### 5. rank_ic 全样本泄漏进训练
- **风险**：全样本 rank_ic 只能做**诊断**，严禁作为特征或选股入模的依据喂回训练折内。
- **规则**：训练只用 `walkforward_selection.json`（purged、train-window-only 算出的选择集）；NO_DEDUP 模式下的 `selected_factors_flipped.csv` 由**各时点可用数据**算出，且保证无 pad。

---

## 二、时点 / 对齐类（共 6 条）

### 6. rebalance 频率 `%10==0` 与 10 日持有期错配
- **错误**：调仓日按"位置 %10==0"定位，与 T+1 起持有 10 日的标签窗口错位，可能造成持仓窗口重叠错配。
- **规则**：调仓周期用交易日**计数器**（每次 rebalance 后 +10），或者显式校验调用窗口 [t+1, t+11] 与标签对齐。

### 7. 回测基准口径
- **规则**：基准 = 中证全指等权（vwap_trad_adj 等权组合），**不**用沪深300 等未复权指数溢价。策略/基准/超额三线统一用同一收益口径（`pct_change().shift(-1)`）。

### 8. `asset_returns` 与标签语义不一致（审计发现的旧问题）
- **错误**：旧 `asset_returns` 用 `vwap.pct_change()`（= Vwap[t]/Vwap[t−1]），与官方 T+1 建仓语义对不上。
- **规则**：**全链路只允许一种收益定义** = `AdjVwap.pct_change().shift(-1)`（vwap-to-vwap），build_label/build_features/train/backtest 四处脚本严禁各自再定义。

### 9. flat/单值结果补面板时的 index 对不齐
- **错误**：非 wide 结果 unstack 后若只出现单一日期（1 行），会有 reindex 到全资产时 NaN 全空行，后续 dropna 静默丢光。
- **规则**：非 wide 落盘前必须断言 panel 行数 ≥ 某个阈值（如 100）且列数 = 资产 universe 数，否则 fail loud。

### 10. 因子值日期与标签日期 join 错位
- **风险**：特征矩阵 join 标签时，date 若一边是 str 一边是 Timestamp/DATE 类型，pandas merge 会静默产出 0 行。
- **规则**：join 前强制 `pd.to_datetime(...).dt.date` 或统一用 str，且断言 join 后行数 = 特征行数（用 `validate='m:1'`）。

### 11. 股票池 tradable/universe 漂移
- **错误**：早期 pipeline 用 297 只 tradable 资产，但 LQTP 池有 5460 全市场宽；join 后大量 NaN 被静默丢弃，导致"看似正常但实际只剩部分标的"。
- **规则**：**显式声明每一次 join 的 universe**（tradable 297 vs 全市场 5460），wide↔long 之间有 universe 变换必须打日志并断言。

---

## 三、复权口径类（共 4 条）

### 12. 复权列名混淆：Close×Factor vs AdjClose
- **正确口径**（COS `StockDailyBarAdj`）：`AdjClose` 已含复权 = Close×Factor；`AdjVwap` = Vwap×Factor。**不要再自己乘**（会双重复权）。
- **规则**：adj 表直接用 `AdjClose/AdjVwap/AdjAmount`；volume 用物理列 `Volume`（股数）/Factor 未必要时不要除 F。

### 13. Return 单位是 bp 不是小数
- **规则**：COS `Return` 列单位 bp（如 12 = 0.0012）。凡是用 Return 列，先 /10000。禁止把它直接当 pct_change 用。

### 14. amount ≈ vwap × volume 的近似复权风险
- **错误**：早期用 `Vwap × Volume` 当 amount，却在复权面板下（vwap 已 ×Factor）导致 amount 虚大 / 流动性因子错。
- **规则**：amount 用物理列（adj 表用 `AdjAmount`），不要从 vwap×volume 反推。

### 15. 前复权 vs 后复权应场景
- **规则**：**训练/评估/回测 = 后复权（AdjVwap_adj）**；预测目标 = 前复权 10 日收益（官方 TargetVwapReturnH10 是后复权口径）——两者不一致时一律以后复权 AdjVwap 为主，对拍 COS 官方靶已经证实。

---

## 四、流程 / 工程类踩坑（共 7 条）

### 16. NO_DEDUP 快速路径绝不能放在采样循环后面
- **错误**：NO_DEDUP 早退分支初版放在 samples 采集后（combine 空耗 17 分钟采 737 因子样本）。
- **规则**：快速路径/早退分支**必须**放在所有昂贵的准备工作之前（本仓库 = `enumerate_candidates` 之前）。

### 17. 陈旧产物文件导致"假完成"
- **错误**：chain_now2 的等待循环 `ls preds_adj_flip_slice_*.parquet | wc -l` 立即看到**上一轮旧 529-链的 4 个文件** → 用旧预测跑了合并+回测，出了 556%/Sharpe 1.307 的假结果。
- **规则**：**每次重跑链路前，先清空下游产物**（preds/predictions/weights/best_params 等全部 rm）。等待循环必须至少校验文件 **mtime > 本轮开始时间**，而不是只看存在性。

### 18. 导出脚本偏食陈旧 CSV
- **错误**：`export_curve_outputs.py` 读的是旧 rows=1595 的 CSV，绕过新 rows=1592 nav parquet。
- **规则**：导出/绘图脚本要**显式指向最新产物**（mtime 或 manifest），或统一从 nav_curves.parquet 生成；发现行数不符就 fail。

### 19. 中断后复用 stale 状态
- **错误**：combine 被杀时留下 530 行 selected 文件，后续被当真。
- **规则**：长任务写产物用**原子写**（先写 `.tmp` 再 rename），或写一个 `BUILD_OK` 标记文件，读取方先查标记。

### 20. sleep/wait poll 脚本超时后无兜底
- **规则**：链式脚本（auto_chain 等）里每次 `wait_for_file` 必须带**超时+报错退出**，禁止死循环白等。

### 21. 内存超限
- **规则**：面板 (5460×2588) 全量 float64 ≈ 113G，**禁止**一次性读进内存。必须分块/float32/downcast；并行 worker 每 subtree 限 OMP_NUM_THREADS。

### 22. 因子遍历顺序不确定
- **规则**：所有 batch/枚举都要 `sorted()` 或固定 seed，否则 rank_ic 部分文件合并顺序不同 → selected 集不同 → 结果不可复现。

---

## 五、"这结果好得离谱"When-you-see-it 清单（泄露红旗）

出现任何一条 → **立刻停下查泄露**，而不是炫耀结果：

- [ ] rank_ic 均值 |IC| > 0.1（真实因子恢复后通常 0.02–0.05）
- [ ] 单日截面 IC 中位数远离 0（正常 ±0.05 内）而均值却极大
- [ ] Sharpe > 3 在十年回测里（本次吊打 2.263 是修完泄露后的真实上限）
- [ ] OOS 预测和标签相关系数 > 0.3
- [ ] 因子重要性第一名的因子值和标签定义里出现了同样的字段（如 AdjVwap[t+11]）
- [ ] 回测第一年就翻倍但后几年平稳（信号可能有样本特征）
- [ ] 换手率统计显示 0.000（说明统计读错矩阵，见 #3）

**验证惯例**：做一个"打乱标签"（shuffle labels across dates within asset）或"延迟 1 天的标签"的 sanity run，真实无泄露的 pipeline 应当 rank_ic 掉到 ≈0；如果还高，就一定有剩余泄露。

---

## 六、快速自查表（每次改代码后过一遍）

```
标签   = COS TargetVwapReturnH10（官方靶，或 shift(-11)/shift(-1)）？
收益   = AdjVwap.pct_change().shift(-1)，全链路唯一定义？
复权   = 直接用 Adj* 列，没有二次乘 Factor？
Return = 如果读 COS Return，单位 /10000 了？
决策   = d 日统计窗口截到 d−1（mu/cov/no tomorrow）？
执行   = w(d) 在 d+1 VWAP 生效，不是 shift(1) 吃当日？
next-day = 特征矩阵 join 标签时 validate+m:1 + 行数断言？
universe = join 两边 universe 声明一致（297 vs 5460）？
清理   = 本轮跑前删掉了上轮 preds/weights/best_params？
幂等   = 等待循环校验 mtime > start，不只是 exists？
内存   = 分块/float32，没有一把梭 5460×2588 float64？
```

---

## 七、基准（benchmark）口径错误 —— 反复犯（2026-08-29 定死）

### 23. 回测基准混用两套，报告口径不一致 ← 用户点名的错误
- **事实**：本项目历史上出现过**两种 benchmark**：
  1. `backtest_final_vectorbt.py` 默认 `--benchmark 000300.SH` → **沪深300 原始收盘价**（2020-01-23..2026-08-24 仅 +13.97%，还有 2020 前后的大回撤 -45.6%），这条被 `run_20260829_782fac`、737fac 等新链继承；
  2. `portfolio_and_backtest.py` → **全池等权 Vwap 收益**（同期 +127.88%），被 0826 (1.758) / 0827 (2.24) 存档继承。
  两者差 ~8 倍，横向比较毫无意义；且沪深300 是价格指数（不含分红），与策略的后复权 Vwap 口径（含分红）天然不公平。
- **用户的定论**：**不应该选等权（全池等权也不对）**——和全 A 股等权比等于和"满仓小票赌场"比；正确基准是**与策略可投资域一致的加权指数**（如中证500/中证全指成分内加权、或策略同 universe 的市值加权组合）。
- **规则**：
  1. **一次存档只有一个基准**，metrics/图/CSV 必须写明基准名；不同基准的结果禁止直接对比数值。
  2. 默认基准改为 `000905.SH`（中证500）或 `000985.CSI`（中证全指），**且用含分红口径**（TotalReturn 或 AdjClose）——不许再用价格指数对后复权策略。
  3. 若策略 universe ≠ 全市场，基准必须裁剪到同 universe（同 tradable 池的指数收益）。
  4. 跨版本对比前先核对 CSV 里 benchmark 终值：`126.46%` 型（等权池）vs `14.61%` 型（000300 价格）一眼可辨，混了就是口径没锁。

### 24. excess_nav 定义漂移
- **错误**：0727 存档 excess = 算术差（strategy_ret − benchmark_ret 累乘），新链 excess = `strategy_nav / benchmark_nav`（几何）。两种 excess 数值不可互比。
- **规则**：全局统一**几何超额** `strategy_nav / benchmark_nav`；若出算术版必须文件名标注 `arith`。

---

## 附：重大事故记录（时间线）

| 日期 | 事故 | 根因 | 修复 |
|---|---|---|---|
| ~2026-08-26 | Sharpe 2.24 假版本 | mu/cov 用未来收益 | 窗口截到 pos−1，同参数掉到 1.70 |
| ~2026-08-26 | 标签 881/900 bad | shift(-10) vs 官方 shift(-11) | 切换 COS 官方靶逐点对拍 diff=0 |
| ~2026-08-27 | 556%/1.307 假结果 | 陈旧 slice 文件当新预测 | 跑前清产物 + mtime 校验 |
| ~2026-08-27 | combine 空耗 17min | NO_DEDUP 早退放错位置 | 移到 enumerate_candidates 前 |
| 2026-08-28 | 737 版本 vs 2026-08-27 版差异 | 泄露修复 + 因子池不同 | 两版都存档供对比 |