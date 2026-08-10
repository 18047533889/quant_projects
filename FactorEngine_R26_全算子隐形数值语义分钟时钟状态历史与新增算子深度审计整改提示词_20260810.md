# FactorEngine R26：全算子隐形数值语义、分钟物理时钟、状态历史与新增算子深度审计整改提示词

> **用途**：直接交给另一个代码 AI，与 R24 / R25 整改并行执行。  
> **本轮定位**：只处理 **R24、R25 没有具体覆盖的新问题**。重点不是治理标签，而是逐算子查“能运行但算错”“PIT 表面正确但时间拓扑错误”“新算子数学定义悄悄偏移”“缺失值制造假信号”“状态历史少读导致结果依赖回测起点”等隐藏 bug。  
> **审计基线**：用户确认 GitHub 没有新提交；本轮按 `7b15a5e7a7a769734f7d1e003a6b0867c1ce88c9` 可见代码审查。执行时仍应先读取服务器真实 `HEAD + dirty tree`；若服务器工作树已经包含并行 AI 尚未提交的 R24/R25 修改，以实际服务器代码为准。  
> **重要**：不要重复 R24 / R25 已列问题，不要把工作重点重新放到 `production_certified`、GenuineUsability、SourceRef v2、semantic identity、R25 real recipe/parameter evidence 等已有整改项。本轮专门修新的 operator-level correctness bugs 与 shared-kernel 隐患。

---

# 0. 本轮最终目标

一个 operator 即使：

```text
能 import
能 register
能跑 synthetic smoke
shape 正确
prefix invariant
有 mining role
有 recipe
```

仍可能算错。

R26 要把以下隐藏 bug 全部消灭：

```text
1. 统计定义被实现细节悄悄改变；
2. tie / NaN / missing / gap / duplicate bar 改写因子含义；
3. 分钟数据把“观测行”错当“真实交易分钟”；
4. 会话开收盘、午休、early close 被错误替代；
5. path / duration / recovery / event age 跨缺口压缩时间；
6. 无界状态被误声明成短窗口；
7. nested rolling history requirement 少读；
8. 数值公式存在 cancellation / overflow / underflow；
9. estimator undefined 被硬剪成 0；
10. output 的数学域、符号、单位与公式不一致；
11. 默认参数组合本身 guaranteed-NaN / guaranteed-error；
12. 新增 market-language / nonlinear / intraday / tail / geometry 算子数学语义有误；
13. K 线 / event / condition 把 unknown 误写成 confirmed false；
14. composition / multi-field 算子数学能算，但经济 part-whole 定义错误。
```

最终必须达到：

```text
R26_HIDDEN_CORRECTNESS_BLOCKERS_ZERO=true
```

---

# 1. 执行原则：冻结工作树并建立“同模式全仓扫描”

## R26-001

开始时执行：

```bash
git rev-parse HEAD
git status --porcelain
```

记录：

```text
audit_head
dirty_tree_digest
canonical_set_digest
```

## R26-002

读取 R24、R25 的 finding 列表，建立：

```text
EXCLUDED_PRIOR_FINDINGS
```

禁止把旧问题重新包装成本轮新问题。

## R26-003

本轮先写静态 hazard scanner，全量扫描 retained operator implementation，至少匹配：

```text
np.nanargmax
np.nanargmin
astype(float)
fillna(0)
np.nan_to_num
np.nansum
np.nanmean
dropna
finite filtering
ffill
bfill
deque
cumsum
cummax
cummin
searchsorted
index.normalize
hour*60+minute
np.diff
np.log
sum(x*x)-sum(x)^2/n
np.lexsort
np.argsort
np.interp
clip(lower=0)
max(-cov,0)
```

## R26-004

每个命中必须分类为：

```text
SAFE_AND_INTENDED
UNKNOWN_TO_ZERO_BUG
GAP_COMPRESSION_BUG
TIME_CLOCK_BUG
STATE_HISTORY_BUG
NUMERIC_STABILITY_BUG
ESTIMATOR_DEFINITION_BUG
DOMAIN_BUG
NEEDS_MANUAL_REVIEW
```

生成：

```text
factor_engine/docs/R26_STATIC_OPERATOR_HAZARD_SCAN.csv
factor_engine/docs/R26_STATIC_OPERATOR_HAZARD_SCAN.json
```

要求：

```text
remaining_unreviewed_hits = 0
```

---

# 2. `ts_chatterjee_xi`：target-dependent tie ordering

文件：

```text
factor_engine/cleaned_operators/dependence_ext.py
```

## R26-005

当前 ξ 实现对 `(x,y)` 排序时可见：

```python
np.lexsort((yv, xv))
```

即：

```text
x 为主键，y 为 tied-x 内次键
```

## R26-006

这意味着当 x 有 ties 时，**用被解释变量 y 决定 tied-x 的内部排列**。

结果会机械降低相邻 y rank 的跳跃幅度，从而抬高 Chatterjee ξ。

A 股特别危险：

```text
零收益
连续涨跌停状态
整数事件计数
分桶信号
离散基本面状态
```

都会出现大量 ties。

## R26-007

修复必须使用：

```text
正式 tie-aware Chatterjee estimator
或完全独立于 y 的 deterministic tie handling
```

禁止用 y 解 x ties。

## R26-008

新增 golden：

```text
continuous independent x,y -> ξ≈0
discrete tied x + independent y -> 不得系统性正偏
perfect dependence -> high ξ
within-x-tie permutation y -> 不能因内部主动排序而提升 ξ
```

---

# 3. Copula MI / entropy：Jeffreys smoothing + Miller–Madow 双重修正

文件：

```text
factor_engine/cleaned_operators/cross_section_local.py
```

## R26-009

当前 copula histogram 路径同时做：

```text
Jeffreys pseudo-count 0.5
+
Miller–Madow entropy correction
```

## R26-010

Miller–Madow 是针对 raw plug-in entropy 的 finite-sample correction；已经做 Dirichlet/Jeffreys smoothing 后继续叠 MM，不再是标准同一 estimator。

潜在结果：

```text
MI 被修成负数
normalized entropy > 1
N变化带来机械漂移
double correction
```

## R26-011

固定一个定义：

```text
方案A：raw histogram plug-in + Miller–Madow
方案B：Jeffreys/Dirichlet-smoothed estimator，不再叠 MM
```

若两种都要保留，拆成不同 canonical/version。

## R26-012

测试：

```text
independent uniforms -> MI≈0
identical variables -> high MI
MI >= 0 within numerical tolerance
normalized entropy落在声明域
ties / empty bins / small N / large N convergence
```

---

# 4. 分钟→日频 shared helper：禁止从 observed deltas 猜 bar frequency

文件：

```text
factor_engine/cleaned_operators/microstructure/intraday_agg.py
```

这是本轮最高影响面之一，因为一个 helper 会污染几十个 `intra_*`。

## R26-013

当前 `_bar_width_minutes()` 通过 observed minute delta 的 mode 推断 bar width。

反例：真实 1-min source只剩：

```text
09:31
09:33
09:35
09:37
...
```

当前会被解释成合法 2-min source，而不是 1-min source 缺失 50%。

## R26-014

bar frequency必须只来自：

```text
DataContract
SessionCalendar
MarketContext
SourceRef/FieldPlan temporal grain
```

禁止 observed-delta self-certification。

---

# 5. 建立唯一 `SessionPanel / MinuteGrid`

## R26-015

新增中央结构，至少携带：

```text
market
trade_date
session_id
slot_id
bar_frequency
source_timezone
session_timezone
expected_timestamp
observed_timestamp
is_expected_slot
is_present
is_duplicate
is_valid_bar
```

## R26-016

所有 minute→daily operator先 canonicalize 到 SessionPanel，再算统计。

禁止每个模块各自：

```text
normalize()
hour*60+minute
猜 bar width
自己定义 morning/afternoon slots
```

---

# 6. physically absent timestamp 与 NaN row必须区分

## R26-017

如果 09:42 row根本不存在，当前 value-array kernel看不到 NaN。

于是：

```text
log(P09:43/P09:41)
```

会被当普通一根 bar return。

## R26-018

这会污染：

```text
realized variance
semivariance
bipower variation
jump variation
tail variation
return correlation
Amihud
Kyle proxy
path efficiency
recovery
```

## R26-019

必须 reindex 到 official SessionGrid，让 absent timestamp变成显式 missing slot。

---

# 7. `_log_returns` 禁止跨缺 bar拼成一根收益

## R26-020

只有：

```text
slot_id[t] - slot_id[t-1] == 1
AND both prices finite
```

才允许生成 1-bar return。

否则：

```text
NaN
```

---

# 8. completeness / coverage必须按 unique official slots

## R26-021

duplicate timestamp不能提高 completeness。

要求：

```text
duplicate official slot -> DQ fail
```

coverage定义：

```text
unique valid expected slots / expected slots
```

而不是 observed row count。

---

# 9. `_daily_agg_three` missing topology不一致

## R26-022

当前三输入 helper可见只按 primary `a` dropna，`b/c` 的 NaN还能进入 kernel。

后续 `nansum/nanmean/partial mask` 会悄悄改变 cohort/denominator。

## R26-023

默认 multi-input minute op必须：

```text
tuple-complete
```

即一个 required slot只有所有 required inputs同时有效才可使用。

## R26-024

若 operator允许 pairwise/partial，必须 per-canonical明确 MissingTopologyPolicy，不能由 helper偷偷决定。

---

# 10. segment return不能用“第一个有限值/最后一个有限值”偷换端点

## R26-025

例如定义：

```text
09:31–10:00 return
```

若 09:31缺失，不能变成 09:32–10:00但 canonical名字不变。

## R26-026

production默认：

```text
EndpointPolicy.EXACT
```

若确需最近有效点，定义显式 policy并进入 semantic identity。

---

# 11. 午休 gap return必须锁定官方端点

## R26-027

`intra_lunch_gap_return` 必须要求：

```text
11:30
13:01
```

按对应 market/calendar exact slot。

缺 11:30不能用11:29，缺13:01不能用13:02，除非显式另一个 endpoint policy。

---

# 12. path efficiency禁止压缩 missing gap

## R26-028

如果先：

```text
finite = values[np.isfinite(values)]
```

再算 path length，相当于把缺口两侧价格直接连起来，通常会缩短路径、抬高效率。

## R26-029

missing slot应该：

```text
break path
```

或只计算明确的 trailing contiguous complete run。

---

# 13. `high_time / low_time` 必须用 official slot ordinal

## R26-030

时间位置不能用：

```text
argmax observed-row-position / observed-row-count
```

必须：

```text
official slot_id / official session slot_count
```

否则 missing/duplicate bar会改变“最高点出现在全天什么位置”的因子。

---

# 14. limit mask tri-state bug

## R26-031

当前某些 limit mask逻辑在 limit已知、bar字段缺失时，会因为 comparison为False而输出 confirmed False。

正确规则：

```text
limit_known AND required_bar_fields_valid -> True/False
otherwise -> NaN
```

## R26-032

受影响重点查：

```text
intra_limit_duration
intra_limit_first_hit_time
intra_limit_reopen_count
所有 limit touch / lock / approach minute helpers
```

---

# 15. limit first-hit不应要求 up/down两边OHLC都存在

## R26-033

`side=up`只需：

```text
high + upper_limit
```

`side=down`只需：

```text
low + lower_limit
```

若因为另一侧panel没传而 fallback close，会漏掉“盘中触板但收盘打开”。

---

# 16. limit duration / first hit也必须使用 physical clock

## R26-034

missing bar不能被删除后少算持续分钟；first-hit时间也不能用压缩后的 row ordinal。

统一用 official slot id。

---

# 17. amount/volume share不能用 incomplete denominator

## R26-035

重点：

```text
intra_segment_amount_share
intra_segment_volume_share
intra_tail_volume_share
相关 participation/share operator
```

如果整日 denominator本身缺 bar，`np.nansum`不能照常输出正常 share。

默认：

```text
unknown denominator -> NaN
```

若允许 partial denominator，必须显式政策。

---

# 18. 全分钟 family timezone / TradeDate 审计

## R26-036

全仓扫描：

```text
index.normalize()
index.hour
index.minute
pd.Timestamp(...).normalize()
```

## R26-037

如果 source timestamp是UTC，而 session是Asia/Shanghai / America/New_York，必须先 `tz_convert(session_timezone)` 再计算 TradeDate/HHMM/slot。

## R26-038

tz-naive不能默认等于session-local。数据合同必须区分：

```text
source_timezone
storage_timezone
session_timezone
```

未知 source timezone时 production fail closed。

---

# 19. date-specific early close / special session

## R26-039

任何 generalized session helper都必须使用：

```text
calendar.for_date(trade_date)
```

得到当日：

```text
open
segments
break
close
slot_count
```

不能用一套全年固定 slot grid处理美国 early close / half day / special session。

---

# 20. `intraday_session.py` PCA residual rank gate过严

## R26-040

当前 PCA residual路径可见要求 numerical rank >= `n_components + 1`。

但“投影到前k个PC并算residual norm”只需要：

```text
rank >= k
```

`k+1`仅在要判断第k方向与下一特征值eigengap时才需要。

## R26-041

分离：

```text
PCA residual rank gate
PC score / eigen-gap identifiability gate
```

## R26-042

测试：

```text
rank exactly k -> residual可算
rank < k -> NaN
near-degenerate PC direction -> signed PC score NaN
```

同步检查 `advanced_intraday` quantile-PCA residual。

---

# 21. `intraday_profile_surprise_energy` equal-count slot会漂移wall-clock语义

文件：

```text
factor_engine/cleaned_operators/advanced_intraday.py
```

## R26-043

`_day_profile_equal_count`按当天 observed bars数等量切slot。

如果中间缺20根bar，后续slot的真实时钟全部移动。

但 operator文档描述的是控制日内时间季节性 / minute-level profile异常，这要求slot具有稳定时钟语义。

## R26-044

合法方案：

```text
A. 只有完整 SessionGrid day才允许 equal-count profile；
B. 新增 fixed-clock profile；
C. 明确重命名为 equal-observation-count profile，不再声称 fixed-time seasonality。
```

---

# 22. `intraday_profile_phase_shift` 正负号语义反向

## R26-045

当前 `_best_phase` 对 `k>0` 比较：

```text
cur[k:]
med[:-k]
```

若历史峰值slot10、今天峰值延后到slot12，`k=+2`会对齐。

因此当前实现含义是：

```text
positive k = current profile delayed
```

但文档写：

```text
正 = 高峰提前
负 = 高峰延后
```

## R26-046

用 synthetic shifted profile golden确认并统一：

```text
实现
文档
factor semantic version
```

可能修为：

```text
return -best_k / n_slots
```

---

# 23. `session_event_recovery_score` 单事件默认仍然被允许

文件：

```text
factor_engine/cleaned_operators/session_recovery.py
```

## R26-047

文档明确说单个shock统计不稳定，但当前默认：

```text
min_events = 1
```

所以一天一个事件仍能产生日度分数。

## R26-048

统一：

```text
default
ParamSpec
min-effective-sample
kernel
doc
```

选择reviewed floor（例如>=2或>=3，具体由测试政策确定），但不能继续“文档不允许、默认允许”。

---

# 24. recovery EventBool NaN不能直接跳过

## R26-049

当前 unknown event row可被 `continue` 跳过，等价于“没事件”。

需要：

```text
EventMissingPolicy
```

默认建议：

```text
BREAK/CENSOR
```

unknown event slot不能让前后episode无缝相连。

---

# 25. recovery horizon不能按压缩后的array row offset

## R26-050

`s+k`必须表示：

```text
k个真实official slots
```

不是：

```text
k个observed rows
```

缺timestamp时必须按slot distance计恢复时间。

---

# 26. EOD recovery必须证明到达session close

## R26-051

如果数据只到14:00，不能因为早间events可算就输出完整EOD factor。

默认要求：

```text
session close/full-session observability
```

partial-session需要独立context/canonical。

---

# 27. Volume clock：TradeDate、physical gaps、session open

文件：

```text
factor_engine/cleaned_operators/volume_clock.py
```

## R26-052

当前 raw `index.normalize()` 分day会在UTC存储时分错session day。改用SessionPanel.trade_date。

## R26-053

现有rows全finite不能发现“物理timestamp未出现”。Volume clock必须先过official grid completeness。

## R26-054

当前 `open_px` 取 first finite open；若官方open slot缺失，会用后续bar的open冒充session open。

更严重：如果open panel已提供但全missing，又可fallback first observed price。

必须区分：

```text
verified_session_open
first_observed_price_proxy
```

production canonical默认要求verified open。

## R26-055

zero-activity same-price检查当前用binary float exact equality。改为tick-size/price-precision-aware equality，不要随意固定EPS。

---

# 28. `intraday_activity_duration` official grid隐含1-minute contract

## R26-056

如果official grid逐分钟展开，合法5-min source会被当大量missing。

二选一：

```text
A. canonical明确只支持certified 1-minute source；
B. official grid按declared bar frequency生成。
```

---

# 29. Intraday impact decay：完全恢复点被删会系统性偏慢

文件：

```text
factor_engine/cleaned_operators/intraday_impact.py
```

## R26-057

若 decay fit 通过：

```text
valid = abs(norm) > tiny_threshold
log(abs(norm)) regression
```

那么最快恢复到0的点会被从回归中删掉。

## R26-058

这会让“快速恢复”的evidence变少，机械把decay估慢。

建议：

```text
first-recovery / censor model
或 detection-floor censored likelihood
或单独 half-life / first-passage estimator
```

不能简单drop zero recovered points。

## R26-059

测试：

```text
instant recovery
fast exponential decay
slow decay
no recovery
```

输出排序必须符合经济含义。

---

# 30. Hill lower-tail实现对正level不成立

文件：

```text
factor_engine/cleaned_operators/extreme_tail.py
```

## R26-060

当前lower path若简单做：

```text
z = -valid
```

再使用经典Hill `log(exc/u)`，对strictly-positive level的left tail会出现负threshold/负exceedance，得到负log-ratio/负xi。

## R26-061

经典Hill要求正的upper-tail magnitude。

正确区分：

```text
signed return/residual downside：loss=-x，对negative tail的positive loss magnitude做Hill；
strictly positive level的left tail：不能simple mirror，需独立lower-endpoint estimator或明确unsupported。
```

## R26-062

Golden：

```text
Pareto upper -> xi>0
-Pareto downside magnitude -> lower-tail loss xi>0
positive bounded level left-tail -> reject/另定义
```

---

# 31. Quantile regression beta的语义说明不能冒充tail-state regression

## R26-063

实现若是：

```text
Q_y(q|x)=a+b*x
```

它是 conditional quantile slope，不是“只在y自己处于q-tail状态时做OLS”。

## R26-064

修正 canonical文档/mining semantic role；如果需要tail-conditioned beta，另实现。

---

# 32. GLR change statistic catastrophic cancellation

文件：

```text
factor_engine/cleaned_operators/glr_change.py
```

## R26-065

如果SSE通过raw prefix sums：

```text
sum(x^2)-sum(x)^2/n
```

对：

```text
x=1e9+tiny signal
```

会 catastrophic cancellation。

## R26-066

GLR理论上对加常数应保持相应translation invariant性质，但实现会受level巨大影响。

修为：

```text
recentered prefix moments
Welford
Chan-Golub-LeVeque
或two-pass stable SSE
```

## R26-067

Metamorphic：

```text
GLR(x) ≈ GLR(x+1e9)
```

## R26-068

绝对 `_EPS=1e-12` 的SS gate也改成scale-aware tolerance。

---

# 33. Roll effective spread：undefined不能映射成0

文件：

```text
factor_engine/cleaned_operators/price_volume/liquidity_v2.py
factor_engine/cleaned_operators/spread_estimators.py
```

## R26-069

当前经典形式：

```text
2*sqrt(max(-Cov(Δp_t,Δp_{t-1}),0))
```

当 covariance>=0时，经典Roll implied spread没有有效real solution/assumption不成立。

当前输出0会被解释成“spread=0、极度流动”，是语义错误。

## R26-070

production canonical建议：

```text
cov>=0 -> NaN
```

如果保留clipped proxy，则单独命名，不能和经典implied spread混用。

## R26-071

`log(price)`前必须强制positive-price domain。坏价格不能变NaN后靠min_periods跳过并继续输出正常spread。

---

# 34. Corwin-Schultz也要PositiveOHLC contract

## R26-072

至少：

```text
high > 0
low > 0
high >= low
```

并继承valid OHLC geometry。零/负价格不能进入log ratio。

---

# 35. Pastor-Stambaugh gamma：`flow_scale`改变数值也改变单位

文件：

```text
factor_engine/cleaned_operators/ohlc_spread.py
```

## R26-073

如果kernel允许任意：

```text
flow_scale
```

而metadata固定：

```text
return_per_million_currency
```

则 `flow_scale != 1e6` 时unit声明错误。

## R26-074

最佳做法：source统一canonical money unit，kernel固定scale=1e6。

如果保留scale参数：

```text
flow_scale必须进入output-unit semantic identity
且不可作为自由search knob。
```

## R26-075

CNY gamma与USD gamma不能未经currency normalization直接raw cross-market rank。

---

# 36. Aroon ties：当前取第一次极值，不是最近一次极值

文件：

```text
factor_engine/cleaned_operators/price_volume/technical_extensions.py
```

## R26-076

`AroonUp/AroonDown`使用 `np.nanargmax/argmin` 时会返回窗口内第一次出现的极值。

同文件 `ts_days_since_high/low` 已经按最近一次极值处理，两套tie policy冲突。

## R26-077

例：

```text
[10,12,12]
```

最新high应是最后一个12；Aroon不能把它算成更旧的第一个12。

A股连续涨停/平价plateau尤其常见。

## R26-078

统一为latest occurrence，并做plateau golden。

---

# 37. Pivot/support/resistance family是隐藏的无界状态机

同文件。

## R26-079

这些模式：

```text
confirmed pivot -> ffill indefinitely
pivot age -> until next pivot
deque(maxlen=points) -> 最近points个pivot（时间上可能很老）
```

并不是 `left_window+right_window` bounded history。

## R26-080

动态枚举受影响canonical：

```text
ts_last_pivot_high/low
ts_pivot_high_age/low_age
ts_resistance_*
ts_support_*
pivot line level/slope/distance/break
```

## R26-081

当前若planner只预取left+right，不同回测起点/warmup/chunk会得到不同结果。

合法修法：

```text
A. full-history replay + checkpoint(last pivots, timestamps, prices, age)
B. 新增明确max_pivot_age/pivot_lookback_bars，超过后NaN
```

---

# 38. Candlestick family：NaN会经comparison变False，再astype成0

## R26-082

全量扫描所有：

```text
cdl_*
```

包括至少：

```text
doji
hammer
inverted_hammer
shooting_star
marubozu
spinning_top
engulfing
inside_bar
outside_bar
```

## R26-083

单日pattern：全部required OHLC已知才可输出0/±1；任一required input unknown -> NaN。

两日pattern：当前+前一日required OHLC都已知才可输出；否则NaN。

不能把unknown解释为confirmed no-pattern。

---

# 39. Candlestick还要验证OHLC几何

## R26-084

至少：

```text
high >= low
high >= max(open,close)
low <= min(open,close)
OHLC > 0（股票价格）
```

非法geometry不能生成负shadow/错误range/pattern；应NaN或DQ fail。

---

# 40. Spinning top zero-body被硬改成bullish +1

## R26-085

当前若：

```text
sign(body).replace(0,1)*flag
```

perfect doji/spinning-top交集会制造伪bullish方向。

## R26-086

解决：

```text
unsigned EventBool
或 event_presence + direction拆开
或SignedEvent能表达neutral event但不能和no-event混淆
```

---

# 41. Outside bar neutral event不能被0吞掉

## R26-087

如果outside bar成立但 `open==close`，`sign(close-open)*flag=0`。

若下游把0定义为no-event，就丢失neutral outside-bar。

同样要求event presence与direction拆语义。

---

# 42. `event_interval`：window实现是bars，history contract却可标event-count

文件：

```text
factor_engine/cleaned_operators/event_interval.py
```

## R26-088

implementation可见用：

```text
r-window+1
```

即trailing bar window。

closure/history contract却存在：

```text
EVENT_COUNT_WINDOW
```

## R26-089

`W bars`与`W events`是完全不同clock，会影响planner prefetch、warmup、cache identity、miner参数含义。

统一成一个定义，不能并存。

---

# 43. `event_interval` 文档最小样本与kernel不一致

## R26-090

当前可见/需重新确认的错位：

```text
memory：文档约>=4 intervals，kernel约>=6
local variation：文档约>=3，kernel更高
Fano：文档约>=2 valid blocks，default min_valid_blocks约5
```

## R26-091

统一：

```text
doc
ParamSpec
min_effective_sample
kernel
history planner
```

---

# 44. EventBool文档不能再写“非零=事件”

## R26-092

若runtime严格 `{0,1,NaN}`，文档必须同样严格。

SignedEvent/score要先显式转换，不能靠nonzero隐式变event。

---

# 45. `ts_multiscale_permutation_entropy_slope` 默认参数自相矛盾

文件：

```text
factor_engine/cleaned_operators/state_geometry.py
```

## R26-093

当前常见默认：

```text
window=120
order=3
scales含8
```

实现support floor约：

```text
5 * order! = 30 embeddings
```

最粗scale=8只有：

```text
floor(120/8)-3+1 ≈ 13
```

因此默认参数本身guaranteed infeasible。

## R26-094

建立唯一：

```python
required_window(order, scales, support_policy)
```

供default/ParamSpec/RelationalParamSpec/history/runtime共用。

## R26-095

对所有新高级operator做“默认值runtime-feasibility”全量扫描，特别是：

```text
entropy
spectral
DMD
Markov
tail
event interval
PCA
quantile
expectile
graph
intraday
state geometry
```

要求registered default不能guaranteed raise/all-NaN。

---

# 46. Composition：当前 financial_statement schema不是合法part-whole

文件：

```text
factor_engine/cleaned_operators/composition.py
```

## R26-096

当前把：

```text
revenue
operating_revenue
net_profit
net_income
total_assets
total_liabilities
equity
shareholders_equity
```

归入一个 `financial_statement` composition family。

这只是“同currency/同公司”，不是“同一个whole的互斥parts”。

## R26-097

例如：

```text
Assets = Liabilities + Equity
```

同时把Assets、Liabilities、Equity当三个parts会double count。

Revenue和NetIncome也不是同一whole的互斥parts。

## R26-098

重做真正CompositionSchema，只允许有经济part-whole关系的集合，例如在数据真实存在时：

```text
资产子项构成
负债子项构成
收入构成
现金流来源构成
holder ownership shares
intraday activity shares
```

不能只凭same unit放入composition。

---

# 47. Composition alias duplicate part

## R26-099

`net_income` / `net_profit` 等若是同一个economic concept alias，不能作为两个PartId重复计入。

要求：

```text
canonical PartId unique within composition
```

---

# 48. Standard wide panel看不到field identity，CompositionSchema验证会失效

## R26-100

当前 `_part_field_name` 主要从frame.name或single-column column name推字段。

但FE标准panel：

```text
rows=date
columns=instruments
```

所以column是股票代码，不是field id。

## R26-101

PartId/CompositionId必须来自typed IR / FieldConcept / input semantic metadata，不能依赖DataFrame股票列名。

## R26-102

如果文档声称支持custom CompositionSchema，必须真正有schema registry/object + PartId + whole_id + unit + exclusivity/coverage policy；否则删掉误导性承诺。

---

# 49. Effective Transfer Entropy surrogate移动了NaN footprint

文件：

```text
factor_engine/cleaned_operators/advanced_information.py
```

## R26-103

某effective-TE null path通过circular shift整个source window，会连NaN位置一起移动。

这只保持missing数量/形状，不保持physical-time exact missing footprint。

因此real/null的：

```text
available transition pairs
effective N
state occupation
```

会不同。

## R26-104

修法：

```text
固定NaN mask，只在finite positions中重排
```

若需要保留serial dependence，用contiguous-block circular shift / block surrogate，但不能移动censor mask。

---

# 50. `advanced_quantile_dynamics.fixed_threshold=True` 并没有真正freeze历史event label

文件：

```text
factor_engine/cleaned_operators/advanced_quantile_dynamics.py
```

## R26-105

当前fixed mode是：

```text
threshold = quantile(chunk[:-1])
然后用这个threshold给整个chunk重新打历史labels
```

这只是current-window threshold排除current row，不是“每个历史row按当时可见历史定义后永久freeze”。

## R26-106

早期row label仍依赖它之后、当前t之前才出现的数据。

这对当前rolling statistic未必构成传统future leak，但**数学定义与文档声称的event-time frozen process不一致**。

## R26-107

两种合法方案：

```text
A. 真event-time threshold：Q_s只用s之前历史，E_s一旦定义不repaint；
B. 改名/改文档为current_window_prior_threshold，停止声称past labels frozen。
```

---

# 51. Quantile spectral concentration：trailing contiguous截断后未重新检查长度

## R26-108

当前先用整个window检查finite coverage，再找到last gap，只保留trailing contiguous run。

如果：

```text
前100个finite
1个gap
最后2个finite
```

整体coverage能过，但最后可能只对2点FFT，产生无意义 concentration≈1。

## R26-109

截断后重新要求：

```text
len(trailing_run) >= min_spectral_samples
```

---

# 52. Extremogram output是signed probability difference，不是Probability

## R26-110

```text
ts_extremogram
ts_cross_extremogram
```

公式：

```text
P(event_future | event_now) - P(event)
```

可为负数。

因此数学域应该是：

```text
SignedProbabilityDifference [-1,1]
```

不能标 `[0,1] Probability`。

---

# 53. `group_tail_lead_score` 同样可为负

文件：

```text
factor_engine/cleaned_operators/tail_systemic.py
```

## R26-111

公式同样是conditional probability - baseline probability。

改为 signed probability difference。

## R26-112

`relation_diffusion_score` metadata若写 `same_as:target`，但参数只有 `x,group,alpha,steps`，应改为：

```text
same_as:x
```

并加unit-resolution test。

---

# 54. Directional Change：`scale`单位合同错误

文件：

```text
factor_engine/cleaned_operators/directional_change.py
```

## R26-113

kernel比较：

```text
absolute price difference >= threshold * scale
```

x明确是Price，因此scale也必须能产生PriceDistance。

但文档同时举：

```text
ATR
realized volatility
MAD
```

ATR/price MAD是价格距离，realized vol是dimensionless，不能混在同一公式。

## R26-114

例如price=100、vol=0.02，当前若直接threshold*scale=0.02，相当于0.02元，而不是2%。

## R26-115

拆成：

```text
absolute_scale: scale=PriceDistance
relative_scale: scale=DimensionlessVol，比较fractional/log move或乘reference price
```

不同模式必须不同semantic identity。

---

# 55. DC event-rate denominator包含了scale未知的bar

## R26-116

当前 `fin_pref` 在price finite时就计数；之后adaptive scale若NaN/<=0，clock其实break，但该bar仍进入event-rate denominator。

结果：

```text
不可观测bar被当成“可观测但没event”
-> event rate被稀释
```

## R26-117

分母必须是：

```text
clock-observable bars = price valid AND threshold/scale valid
```

---

# 56. Activity clock：nested history需求应是组合，不是max

文件：

```text
factor_engine/cleaned_operators/activity_clock.py
```

## R26-118

当前k*搜索可能回看 `max_lookback`，但每个历史 `scaled_activity[s]` 又需要它之前 `scale_window` 计算rolling median。

因此raw input history通常接近：

```text
max_lookback + scale_window (+ exact boundary)
```

而不是简单max两者。

## R26-119

精确推导history_formula，并做：

```text
full-history vs planner-declared-prefetch parity
```

---

# 57. Activity clock自定义Polars bridge不能猜metadata columns

## R26-120

当前custom转换只排除固定列名如 `date/stock_code`，真实panel若有：

```text
timestamp
trade_date
instrument
session_id
slot_id
```

可能被当value列。

## R26-121

必须走central PanelAdapter，并测试不同metadata schema/order。

---

# 58. DMD：`log(abs_b**2)` 会先underflow

文件：

```text
factor_engine/cleaned_operators/dmd.py
```

## R26-122

当前若：

```python
np.log(abs_b ** 2)
```

`abs_b=1e-200` 时平方先underflow到0，再log变-inf，把非零mode误当zero-energy。

修为：

```python
2.0 * np.log(abs_b)
```

对 `abs_b>0` 直接log-domain计算。

---

# 59. DMD：`rho=abs(lambda)**2` 也会先overflow

## R26-123

finite但很大的lambda平方后可先变inf，再进入log helper时已经丢信息。

## R26-124

直接使用：

```text
log_rho = 2*log(abs(lambda))
```

在log-rho域算finite-horizon geometric sum，避免平方。

## R26-125

如果所有mode `log_energy=-inf`，显式fail closed，不允许 `-inf - -inf -> NaN` 由偶然数值路径决定。

---

# 60. 新增高级统计 family 数值稳定性总扫

## R26-126

对：

```text
DMD
spectral
entropy
mutual information
GLR
topological
kernel
Markov
expectile
quantile
tail
```

运行适用的metamorphic：

```text
large translation
large scaling
tiny scaling
constant
near-constant
one outlier
ties
sparse missing
contiguous missing block
```

只应用该数学定义真正应满足的property。

---

# 61. “undefined estimator -> 0”专项扫描

## R26-127

全仓搜索：

```text
max(-x,0)
clip(lower=0)
np.maximum(value,0)
if invalid: return 0
```

逐个判断0是合法边界，还是undefined状态被伪装成正常数值。

Roll positive-cov是已确认实例。

---

# 62. “unknown -> confirmed false”专项扫描

## R26-128

全仓查：

```text
comparison with NaN
astype(float)
where(cond,1,0)
```

重点：

```text
candlestick
event
condition
limit
state
regime
breakout
pattern
```

所有状态必须区分：

```text
TRUE
FALSE
UNKNOWN
```

---

# 63. “time compression”专项扫描

## R26-129

全仓查：

```text
dropna
finite-filter
np.diff(filtered)
first finite
last finite
argmax index/len(observed)
event positions by row index
```

凡输出涉及：

```text
time
age
duration
lag
frequency
path
recovery
first-hit
session location
```

都必须验证删除missing row是否改变物理时间。

---

# 64. “隐式无界状态”专项扫描

## R26-130

全仓查：

```text
ffill without limit
bfill
deque of events/pivots
last_event
last_state
first_anchor
cumulative state
running extrema
```

每个命中必须具备：

```text
checkpoint
full_history
或 explicit max_age/window
```

已被R24/R25处理的只做regression，不重复报告。

---

# 65. 全量默认参数可行性扫描

## R26-131

对所有当前 retained operator，用declared default参数运行 feasibility，不重复R25整个参数证书体系，只专门找：

```text
default guaranteed-raise
default guaranteed-all-NaN
default violates own relational floor
default min sample impossible
```

状态：

```text
PASS
NO_DEFAULT_REQUIRED
FAIL_RELATIONAL
FAIL_RUNTIME
ALL_NAN_BY_CONSTRUCTION
```

生成：

```text
factor_engine/docs/R26_DEFAULT_FEASIBILITY_AUDIT.csv
```

---

# 66. adversarial ties测试

## R26-132

A股必须系统覆盖：

```text
all equal
two-value
three-value
long plateau
repeated highs
repeated lows
zero-return blocks
limit-price plateau
```

重点：

```text
rank
argmax/argmin
Aroon
Chatterjee
quantile
entropy
state bin
pivot
```

---

# 67. adversarial missing topology测试

## R26-133

不是只随机NaN，必须构造：

```text
single missing bar
missing open
missing close
missing 11:30
missing 13:01
missing every other minute
missing contiguous 10-minute block
duplicate timestamp
out-of-order timestamp
provider gap
one stock missing while peers complete
```

---

# 68. timestamp topology metamorphic test

## R26-134

构造同样numeric values：

```text
A. 完整连续时间
B. 删除若干timestamp，但保留剩余value顺序
```

如果operator理论依赖物理时间，A/B必须按missing policy产生差异/NaN，不能因为只看row position而输出一样。

---

# 69. Session boundary golden

## R26-135

A-share至少：

```text
09:31
11:30
13:01
15:00
```

US至少：

```text
normal close
early close
DST boundary
```

全部使用calendar authoritative。

---

# 70. operator arithmetic PIT补充测试

## R26-136

不重复R24/R25整体PIT框架，只对本轮修复的event/state/minute算子做：

```text
future values perturb
future missing mask perturb
future group/state perturb
```

当前row不得变化。

同时past missing topology改变时，应按声明policy正确变化。

---

# 71. Real factor impact regression

## R26-137

对本轮高影响修复family，抽真实A股窗口比较before/after：

```text
finite ratio
mean
std
quantiles
rank correlation
max abs delta
changed-cell share
```

目的不是要求after≈before，而是确认修复只改变原来错误场景，没有大面积意外漂移。

---

# 72. 新测试目录

## R26-138

新增：

```text
factor_engine/tests/operators/r26/
```

至少：

```text
test_dependence_ties.py
test_intraday_physical_clock.py
test_intraday_missing_topology.py
test_intraday_timezone.py
test_tail_estimators.py
test_technical_ties_patterns.py
test_event_interval_clock.py
test_state_geometry_defaults.py
test_composition_schema.py
test_information_surrogates.py
test_quantile_dynamics_semantics.py
test_directional_change_units.py
test_activity_clock_history.py
test_dmd_log_numerics.py
```

---

# 73. R26 machine-readable artifacts

## R26-139

生成：

```text
factor_engine/docs/R26_OPERATOR_CORRECTNESS_MATRIX.csv
factor_engine/docs/R26_OPERATOR_CORRECTNESS_MATRIX.json
factor_engine/docs/R26_OPERATOR_CORRECTNESS_MATRIX.md
```

字段至少：

```text
canonical
family
math_status
tie_status
missing_status
physical_clock_status
state_history_status
numeric_stability_status
default_feasibility_status
unit_semantics_status
economic_semantics_status
fix
evidence
```

## R26-140

生成分钟专项：

```text
factor_engine/docs/R26_INTRADAY_PHYSICAL_CLOCK_MATRIX.csv
```

字段：

```text
canonical
source_frequency
declared_bar_width
official_grid_required
missing_timestamp_policy
duplicate_policy
endpoint_policy
session_tz_policy
early_close_policy
coverage_policy
time_output_uses_slot_id
PASS/FAIL
```

## R26-141

生成 estimator专项：

```text
factor_engine/docs/R26_ESTIMATOR_DEFINITION_AUDIT.json
```

覆盖：

```text
Chatterjee
copula MI/entropy
Hill
Roll
GLR
DMD
quantile dynamics
expectile
PCA
TE
```

## R26-142

生成state/history：

```text
factor_engine/docs/R26_STATE_HISTORY_AUDIT.json
```

列：

```text
canonical
actual_state_memory
declared_history
bounded
checkpoint
full_replay
max_age
full-vs-warmup parity
```

## R26-143

生成pattern tri-state：

```text
factor_engine/docs/R26_PATTERN_TRISTATE_AUDIT.csv
```

---

# 74. 同模式全仓反查是硬要求

## R26-144

不能只修本文点名的canonical。

每修一个模式，例如：

```text
NaN compare -> false
```

必须AST/grep全仓，输出：

```text
identified_instances
fixed_instances
reviewed_safe_instances
remaining_instances
```

最终：

```text
remaining_unreviewed = 0
```

---

# 75. 禁止的“修法”

## R26-145

禁止仅：

```text
加 experimental标签
改 status
改 docs
改 artifact
```

代替实现修复。

## R26-146

禁止用clip掩盖数学bug。例如MI因双重修正变负，不能简单 `clip(0)` 后宣布修好。

## R26-147

禁止 `fillna(0)` 消灭unknown。

## R26-148

禁止只提高min_periods掩盖physical-gap/time-clock bug。

---

# 76. 已确认 concrete findings总表

执行AI应逐项在真实服务器代码重新确认，并搜索同类实例。

## R26-149

```text
ts_chatterjee_xi：tied x内部用y排序，target-dependent tie bias。
```

## R26-150

```text
copula MI/entropy：Jeffreys smoothing + Miller–Madow double correction。
```

## R26-151

```text
intraday_agg：observed delta推bar width，会把缺失1m数据自证成2m。
```

## R26-152

```text
intraday_agg：physically absent timestamp不可见，returns/path可跨gap。
```

## R26-153

```text
coverage：可能按observed rows而非unique official slots。
```

## R26-154

```text
_daily_agg_three：secondary input missing topology可被partial/nansum吞掉。
```

## R26-155

```text
segment return：first/last finite可替代required endpoints。
```

## R26-156

```text
lunch gap：缺11:30/13:01时可能移动端点。
```

## R26-157

```text
path efficiency：finite filtering会压缩missing gap。
```

## R26-158

```text
high_time/low_time：observed-row ordinal可替代official minute ordinal。
```

## R26-159

```text
limit mask：known limit + missing bar可变confirmed False。
```

## R26-160

```text
limit first-hit：up/down侧需要的OHLC输入条件不对称处理不足，可能fallback close漏intrabar touch。
```

## R26-161

```text
minute timezone：若不先转session timezone，TradeDate/HHMM可能错。
```

## R26-162

```text
PCA residual：rank>=k+1过严；rank==k时residual本可计算。
```

## R26-163

```text
profile surprise：equal-observation-count slots在incomplete session漂移wall-clock含义。
```

## R26-164

```text
profile phase shift：实现sign与文档early/late方向反向。
```

## R26-165

```text
session_event_recovery_score：文档说single event不稳，default min_events=1。
```

## R26-166

```text
session recovery：unknown EventBool被skip而非censor/break。
```

## R26-167

```text
session recovery：row offset可替代physical minute distance。
```

## R26-168

```text
session recovery：truncated EOD session仍可能产生日度分数。
```

## R26-169

```text
volume clock：raw normalize()在UTC存储时可分错TradeDate。
```

## R26-170

```text
volume clock：physical missing timestamp检测不到。
```

## R26-171

```text
volume clock：first finite open可冒充verified session open。
```

## R26-172

```text
volume clock：zero-activity same-price用binary float exact equality。
```

## R26-173

```text
activity duration：official grid可隐式hardcode 1-minute frequency。
```

## R26-174

```text
impact decay：完全/近完全恢复点从log fit删除会把恢复估慢。
```

## R26-175

```text
Hill lower tail：simple negative mirror不是positive-level left-tail的经典Hill。
```

## R26-176

```text
quantile regression beta：conditional quantile slope语义不能写成tail-state regression。
```

## R26-177

```text
GLR：raw sum/sumsq SSE有catastrophic cancellation。
```

## R26-178

```text
Roll spread：positive covariance estimator undefined却报告0 spread。
```

## R26-179

```text
Roll/Corwin：positive-price domain需hard gate。
```

## R26-180

```text
Pastor-Stambaugh：flow_scale改变数值单位但metadata可固定per-million。
```

## R26-181

```text
Aroon：ties取first extreme，不是latest extreme。
```

## R26-182

```text
pivot/support/resistance：indefinite ffill/last-pivot state实际无界，但可只声明短history。
```

## R26-183

```text
candlestick：NaN comparison -> False -> 0 no-pattern。
```

## R26-184

```text
candlestick：invalid OHLC geometry可继续参与pattern判断。
```

## R26-185

```text
spinning top：zero-body可被硬改成+1 bullish。
```

## R26-186

```text
outside bar：neutral event可collapse到0=no-event。
```

## R26-187

```text
event_interval：implementation window=bars，history contract可写event-count。
```

## R26-188

```text
event_interval：docs minimum support与kernel不一致。
```

## R26-189

```text
multiscale permutation entropy slope：default window/order/coarse scale违反自己的sample floor。
```

## R26-190

```text
composition：financial_statement schema混合非互斥、非part-whole字段。
```

## R26-191

```text
composition：alias fields可重复计算同一个economic part。
```

## R26-192

```text
composition：标准wide panel的columns是instrument，不是field id，schema检查可能失效。
```

## R26-193

```text
effective TE surrogate：circular shift移动NaN footprint，使real/null有效transition cohort不同。
```

## R26-194

```text
quantile fixed_threshold：current-window prior threshold仍会repaint历史labels，不是真event-time frozen。
```

## R26-195

```text
quantile spectral：coverage在trailing-contiguous截断前检查，截断后可能只剩极短FFT样本。
```

## R26-196

```text
extremogram：excess probability是signed，却可被标成probability。
```

## R26-197

```text
group_tail_lead_score：probability difference是signed，却可标probability。
```

## R26-198

```text
relation_diffusion_score：same_as:target引用不存在的target，应same_as:x。
```

## R26-199

```text
directional change：absolute-price threshold允许dimensionless realized-vol scale直接进入，单位不一致。
```

## R26-200

```text
directional change：event-rate denominator包含scale invalid、clock不可观测bar。
```

## R26-201

```text
activity clock：raw history requirement需要组合max_lookback+scale_window，可能under-declared。
```

## R26-202

```text
activity clock Polars：custom dataframe conversion路径可能把metadata columns当value，需central PanelAdapter。
```

## R26-203

```text
DMD：log(abs_b**2)在log前underflow。
```

## R26-204

```text
DMD：abs(lambda)**2在log-domain处理前overflow。
```

---

# 77. Final hard gates

## R26-205

以下全部必须为0：

```text
TARGET_DEPENDENT_TIE_BREAKS
UNKNOWN_TO_FALSE_PATTERN_BUGS
INTRADAY_OBSERVED_ROW_AS_CLOCK_BUGS
INTRADAY_INFERRED_BAR_WIDTH_FROM_OBSERVED_DELTAS
INTRADAY_UNDETECTED_PHYSICAL_GAPS
SESSION_ENDPOINT_SUBSTITUTION_BUGS
UNDECLARED_UNBOUNDED_STATE
UNDERDECLARED_NESTED_HISTORY
DEFAULT_GUARANTEED_INFEASIBLE_OPERATORS
UNDEFINED_ESTIMATOR_MAPPED_TO_NORMAL_ZERO
NUMERIC_CANCELLATION_BLOCKERS
PRELOG_OVERFLOW_UNDERFLOW_BLOCKERS
SIGNED_PROBABILITY_MISLABELED_AS_PROBABILITY
DIMENSIONALLY_INVALID_OPERATOR_MODES
INVALID_COMPOSITION_SCHEMAS
SURROGATE_MISSING_TOPOLOGY_DRIFT
PCA_UNNECESSARY_RANK_REJECTION
R26_UNREVIEWED_STATIC_HAZARD_HITS
```

---

# 78. Full regression gates

## R26-206

修完至少运行：

```text
all operator tests
intraday/microstructure
technical/event/stateful
statistical/nonlinear
planner/history
backend parity
mining expression tests
```

然后跑：

```text
full FactorEngine suite
```

## R26-207

对所有 history-sensitive/stateful/nested-window operator必须比较：

```text
full history compute
vs
planner-declared prefetch + target interval compute
```

完全一致，否则：

```text
HISTORY_CONTRACT_FAIL
```

## R26-208

minute operator以完整official grid为reference，人为删除timestamp、duplicate、timezone shift、session truncate后，必须按contract NaN/DQ fail/explicit partial，不能悄悄给正常因子。

---

# 79. 最终验收报告

## R26-209

生成：

```text
factor_engine/docs/R26_FINAL_ACCEPTANCE_REPORT.md
```

第一页直接回答：

```text
本轮新增 hidden operator correctness blockers 是否全部清零？
YES / NO
```

## R26-210

报告必须给：

```text
current canonical count
reviewed canonical count
static hazard hits
reviewed-safe hits
bugs fixed
remaining blockers
```

若NO，逐个列：

```text
canonical
file
function
failure mode
factor impact
exact fix
test needed
```

---

# 80. 最终机器 flags

## R26-211

只有所有gate通过才允许：

```text
R26_ALL_OPERATOR_IMPLEMENTATIONS_REVIEWED=true
R26_ALL_INTRADAY_OPERATORS_USE_PHYSICAL_SESSION_CLOCK=true
R26_ALL_INTRADAY_MISSING_TOPOLOGIES_EXPLICIT=true
R26_ALL_PATTERN_OUTPUTS_TRISTATE_CORRECT=true
R26_ALL_TIE_POLICIES_TARGET_INDEPENDENT=true
R26_ALL_STATE_HISTORY_CONTRACTS_MATCH_IMPLEMENTATION=true
R26_ALL_DEFAULTS_RUNTIME_FEASIBLE=true
R26_ALL_ESTIMATOR_DEFINITIONS_COHERENT=true
R26_ALL_NUMERIC_STABILITY_GATES_PASS=true
R26_ALL_UNIT_AND_SIGN_SEMANTICS_MATCH_FORMULA=true
R26_ALL_COMPOSITION_SCHEMAS_ECONOMICALLY_VALID=true
R26_ALL_SURROGATE_MISSING_MASKS_VALID=true
R26_ALL_NEW_OPERATOR_HIDDEN_BUGS_CLOSED=true
R26_HIDDEN_CORRECTNESS_BLOCKERS_ZERO=true
```

---

# 81. 与R24/R25联合收口

## R26-212

R26不替代R24/R25。

整个FactorEngine只有：

```text
R24 all hard flags true
AND
R25 all hard flags true
AND
R26 all hard flags true
```

才可以宣布：

```text
FACTOR_ENGINE_OPERATOR_SYSTEM_PRODUCTION_READY=true
```

---

# 82. 最后一条执行要求

## R26-213

不要只回复“分析完成”。直接：

```text
freeze server snapshot
→ enumerate all canonicals
→ run static hazard scan
→ reproduce every concrete R26 finding
→ search same bug pattern across whole registry
→ fix implementation
→ fix history/session/math contracts
→ add hostile/golden tests
→ run targeted tests
→ run full suite
→ regenerate R26 matrices
→ produce final acceptance report
```

## R26-214

用户要的是“所有算子都真正算对”。本轮尤其要确保新增：

```text
market-language
tail
quantile
expectile
information theory
intraday
state geometry
composition
directional change
DMD
event
technical pattern
```

不再出现：

```text
代码能跑，但算出来的因子并不是我们以为的那个数学/市场含义。
```

直到：

```text
R26_HIDDEN_CORRECTNESS_BLOCKERS_ZERO=true
```
