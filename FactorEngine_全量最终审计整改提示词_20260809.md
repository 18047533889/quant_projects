# FactorEngine 全量最终审计、整改与生产化闭环 Master Prompt

> **用途**：本文件不是讨论稿，而是直接交给代码 AI 执行的 FactorEngine 全库整改任务书。  
> **仓库**：`18047533889/quant_projects`  
> **审计基线**：2026-08-09 当前 `main`，审计时最新可见提交 `44b294608ad1648c0f0fa441c33e69a6d10f80d5`。  
> **核心目录**：`factor_engine/cleaned_operators/`，并包含与算子生产执行直接相关的 `factor_engine/runtime/`、planner/backend certification/manifest 层。  
> **任务原则**：不要只修这里明确点名的行。这里列出的每一个问题都代表一个“问题类别”；修完点名实例后，必须对全库做同类模式扫描，直到同类问题归零。

---

# 0. 给执行 AI 的总指令

请完整读完本文后再开始改代码。不要只挑 P0 的几项修，不要看到测试通过就提前结束，也不要只改注释、metadata 或测试来绕过问题。目标是把 FactorEngine 修到：

1. **数学定义正确**：canonical 名称、文档、公式、实现、单位、统计口径一致。
2. **严格 PIT**：无未来函数、无间接当前目标泄漏、无跨未知时间点重连。
3. **数据语义正确**：Price / Return / Flow / Stock / EventBool / ConditionBool / GroupId / FiscalPeriod / SessionGrid 等输入不能混用。
4. **参数空间真实**：没有 silent clamp、int 截断、dead parameter、等价参数、非法组合、假搜索维度。
5. **时间语义正确**：window、lag、history、event-count、report-count、session-count、递归状态都由机器契约真实描述。
6. **缺失值正确**：NaN、Inf、停牌、未上市、未知事件、未知分组、provider gap 都不能被解释成 0 或旧状态。
7. **多输入严格对齐**：任何多 panel 运算都必须 exact SameAxis / InstrumentId / TradeDate。
8. **A 股 session 正确**：1 分钟默认官方 240 bar；午休、隔夜、停牌、涨跌停、缺 bar、重复 bar 都有明确处理。
9. **minute → daily 真正可生产**：合法 grain transform 不能因为“不 shape-preserving”或 `micro_` 前缀被架构一票否决。
10. **搜索暴露正确**：有价值算子能被 AlphaProbe / AlphaMiner / LLM factor miner 真正发现；global state / condition / internal transform 不被当 terminal alpha。
11. **实现状态与最终 registry 一致**：不能源文件修好了，后处理 overlay 又回滚。
12. **测试闭环**：每一个修复都要有能击穿旧 bug 的回归测试，而不是“跑过一次没有异常”。

**不要新增用户没有的数据字段。不要为了“实现某个经典指标”凭空制造不存在的字段。**  
如果经典定义需要当前数据层不存在的真实数据，则应：
- 降级为 honest proxy 并诚实名；
- 转 research/source-blocked；
- 或删除；
不能假装已完整实现。

---

# 1. 最终目标数据结构

最终每个 canonical 至少需要形成一个机器可读的认证记录：

```yaml
canonical:
definition_id:
semantic_version:

input_semantic_types:
input_units:
panel_params:
scalar_params:
same_axis_contract:

output_unit:
output_role:

input_grain:
output_grain:
available_at:
same_session_usable:

window_semantics:
history_requirement:
current_required:
missing_policy:
state_model:
chunking:
checkpoint_schema:

param_specs:
param_roles:
relational_constraints:
active_when:

source_contract:
market_contract:
session_policy:
universe_policy:
group_membership_vintage:

min_effective_n:
effective_n_definition:

mining_role:
terminal_allowed:
cost_tier:
default_search_weight:

implementation_certified:
semantic_certified:
temporal_certified:
source_pit_certified:
edge_certified:
backend_certified:
production_certified:
mining_eligible:

duplicate_cluster_id:
compatibility_aliases:
preferred_replacement:
```

未声明的关键字段不能通过 production certification。

---

# 2. 本轮新增发现：共享内核与架构级 P0/P1

以下是**在上一版 Master Spec 之后，重新读取当前 main 后新确认的代码级问题**。这些优先级很高，因为很多会同时污染几十/几百个算子。

## NEW-001｜`ParamRole` 仍是 fail-open，而不是 fail-closed

### 位置
`factor_engine/cleaned_operators/base.py`

### 问题
`effective_param_role(spec)` 对：
- `spec is None`
- `spec.searchable=True` 但未显式声明 role

仍然可能回退到 `ECONOMIC`。

这会导致一个新 operator 忘记补 ParamSpec/ParamRole 时，参数悄悄进入默认 mining search space。

### 修复
- production/mining canonical：任何 scalar param 缺 ParamSpec 或缺 ParamRole → certification FAIL。
- compatibility/research 才允许 fallback。
- 不要再把“未声明”解释成 ECONOMIC。

### 测试
遍历 final registry：
```python
for production canonical:
    for scalar param:
        assert param in param_specs
        assert param_specs[param].param_role is not None
```

---

## NEW-002｜ParamRole 枚举还不够表达实际语义

### 当前常见角色
`ECONOMIC / HORIZON / STATE_THRESHOLD / ESTIMATOR_RESOLUTION / NUMERICAL / POLICY`

### 缺失
至少补：
- `MODEL_ORDER`
- `MISSING_POLICY`
- `MARKET_POLICY`
- `SUPPORT_POLICY`
- `SOURCE_POLICY`
- `SESSION_POLICY`

### 原因
`rank/dim/AR order` 与 estimator resolution 不是一回事；`min_periods/min_group_size/min_count` 也不是普通 policy。

### 修复
默认自动搜索仅开放：
- ECONOMIC
- HORIZON
- STATE_THRESHOLD
- MODEL_ORDER

---

## NEW-003｜`_INTEGER_PARAM_NAMES` 仍在承担 production 参数类型推断

### 位置
`base.py`

### 问题
大量旧 operator 没 ParamSpec 时仍靠参数名字猜整数。

风险：
- 新名字不在白名单；
- 同名参数在不同 canonical 语义不同；
- 让“不完整 metadata”继续进入生产。

### 修复
- production path 禁止使用 `_INTEGER_PARAM_NAMES`。
- 只允许 legacy compatibility layer 使用。
- certification 输出 `used_legacy_param_inference=True/False`，production 必须 False。

---

## NEW-004｜`rolling_pack.check_window()` 仍直接 `int(window)`

### 位置
`cleaned_operators/rolling_pack.py`

### 问题
`20.9 -> 20`。

### 修复
改用统一 `strict_int` / ParamSpec-bound value。
helper 不再自行 coercion。

---

## NEW-005｜多个 strict-int helper 并存，行为不统一

当前至少存在：
- `base.py` 参数规范化
- `overhaul/base.py::positive_int`
- `daily_panel.py::_positive_int`
- `transforms_v2.py::_pos_int`
- `fundamental/ledger.py::_pos_int`
- `technical/indicators_v2.py::_pi`
- 各 module 自己 `int()`

### 修复
建立唯一：
`cleaned_operators/common/strict_params.py`

提供：
```python
strict_int
strict_nonnegative_int
strict_positive_int
strict_float
strict_probability
strict_bool
strict_condition_bool
strict_event_bool
strict_enum
```

所有 operator 只能调用这一套。

---

## NEW-006｜fundamental `parameter_contract_v2` monkey-patch 设计实际不能覆盖已 import 的 helper

### 位置
- `fundamental/transforms_v2.py`
- `fundamental/parameter_contract_v2.py`
- `fundamental/quality_v2.py`
- `fundamental/expectation_v2.py`

### 问题
`parameter_contract_v2.py` 最后执行：
```python
transforms_v2._pos_int = _pos_int
```

但其他模块如果之前做过：
```python
from transforms_v2 import _pos_int
```
绑定的是旧函数对象。后续修改 `transforms_v2._pos_int` 不会改掉已经绑定出去的引用。

当前 loader 顺序里多种 fundamental 模块可能在 patch 前已经加载。

### 后果
同一个 FactorEngine 运行进程里，fundamental family 可能同时存在：
- 严格 integer validator
- `int(value)` 截断版 validator

### 修复
彻底删除 monkey patch。
所有 fundamental 文件直接 import 唯一 `strict_params`。

### 测试
```python
assert quality_v2.strict_int is strict_params.strict_int
assert expectation_v2.strict_int is strict_params.strict_int
...
```
更重要的是跑行为测试：
`periods=3.7` 必须在所有 fundamental operator 统一报错。

---

## NEW-007｜`load_all()` 后处理链过长，源文件正确不等于最终 runtime 正确

### 位置
`cleaned_operators/__init__.py`

当前存在类似：
1. module registration
2. dedupe
3. overhaul
4. LQTP policy
5. layer governance
6. post-governance
7. production hardening
8. fiscal re-register
9. registration audit
10. SQL attach
11. Polars gap coverage
12. evidence overlay
13. final contract hardening
14. freeze

### 问题
任何后处理都可能修改：
- canonical
- source
- status
- policy
- semantics
- backend
- alias
- history
- role

### 修复
新增 `FinalRuntimeStateAudit`，**只审 `OperatorRegistry.freeze()` 前最终状态**。

对于每个 canonical：
```text
source contract
== final catalog contract
== final backend metadata contract
== final mining manifest contract
== final history contract
```

不允许仅靠“源文件测试通过”认证。

---

## NEW-008｜多个中央手工 stateful 列表仍然并存

### 位置
- `runtime/execution_contract.py::_STATEFUL_CANONICALS`
- `production_hardening.py::FULL_HISTORY_REPLAY_CANONICALS`
- checkpoint registry
- `_DECLARED_STATEFUL`

### 当前甚至还能看到已退休名字
`state_since_reduce`

### 问题
所谓 SINGLE AUTHORITY 仍然存在 legacy 中央名单。

### 修复
新 operator 必须 self-declare：
```python
declare_execution_contract(...)
```
legacy seed 全部迁移完后删除。

最终：
```text
LEGACY_STATEFUL_SEED == empty
```

---

## NEW-009｜HistoryRequirement 同样仍依赖巨大 canonical 手工映射

### 位置
`runtime/execution_contract.py::_HISTORY_TRANSFORMS`

### 问题
新算子如果忘记在中央 map 注册，会进入名字猜测 fallback。

### 修复
每个 operator 自己提供：
```python
history_requirement_factory(params) -> HistoryRequirement
```

DAG runtime 只做组合，不猜 canonical 名。

---

## NEW-010｜`contract_hardening._LOOKBACK_PARAM_PRIORITY` 与真实现代参数不匹配

### 问题
旧推断只覆盖一部分：
`window/periods/d/lag/n/...`

但现代算子大量使用：
- recent_window
- prior_window
- history_window
- horizon
- delay
- path_window
- embedding_len
- max_age
- n_updates
- max_boundary_extension
- scale_window
等。

### 更严重
代码里有 `delay/horizon` extension 逻辑，但它们不一定能进入 controlling param 列表，形成 dead branch。

### 修复
最终 lookback 不再依赖名字优先级。

---

## NEW-011｜`production_hardening` 仍有手工 state/history override，可能覆盖新契约

### 修复
production hardening 只能读取最终统一 contract，不能再二次发明状态语义。

---

## NEW-012｜production eligibility 仍按 `micro_*` 前缀一票否决

### 位置
`cleaned_operators/operator_spec.py::is_production_denied`

### 当前
```python
if str(canon).startswith("micro_"):
    return True
```

### 问题
名字不能代表安全性。

### 修复
完全删除 prefix deny。
逐 canonical：
- source contract
- grain contract
- evidence
- certification
决定 eligibility。

---

## NEW-013｜production admission 仍要求 `shape_preserving=True`

### 位置
`operator_spec._compute_allow_in_production`

### 问题
合法的：
```text
minute panel -> daily panel
```
天然 shape-changing。

这会让真正有价值的日内派生日频因子无法 production。

### 修复
用：
```text
ShapeContract
GrainTransformContract
```

允许：
```text
minute -> daily
snapshot -> daily
event table -> entity-date panel
```
只要输出 shape **符合声明**，不要求与输入完全同形。

---

## NEW-014｜`PRODUCTION_AUTHORING_CANONICALS` 与 extended/mining production 语义存在双轨

### 问题
部分构建函数只从 `DAILY_CANONICALS` 出发，另一些准入函数又接受 daily+extended。

### 修复
分开：
```text
AuthoringTier
ExecutionEligibility
MiningEligibility
```
三者不能用同一个 set 互相代替。

---

## NEW-015｜`semantic_certification.py` 仍有超大手工 `PROMOTED_OUT_OF_EXPERIMENTAL`

### 风险
一个 operator “synthetic runtime clean” 不等于：
- 数学定义正确
- source PIT 正确
- A股 edge 正确
- 有效样本统计正确

### 修复
禁止大名单 promotion 成为任何证书的代理。
每个 certificate 必须来自独立 test artifact。

---

# 3. 本轮新增：基础 rolling kernel 的实际数学 Bug

## NEW-016｜`rolling_argmax` 方向与上层文档相反

### 位置
`cleaned_operators/_rolling_fast.py`

当前：
```python
np.nanargmax(seg)
```
返回：
```text
0 = 窗口最旧位置
```

但部分上层 `TSArgmax` 文档/语义写：
```text
0 = 最新 bar
```

### 修复
不要再一个 ambiguous canonical。

保留两个清晰算子：
```text
ts_argmax_age                 # 0=current
ts_argmax_index_from_oldest   # 0=oldest
```

旧 `ts_argmax` 只能 alias 到一个已明确定义且兼容历史语义的 canonical。

---

## NEW-017｜`rolling_argmin` 同样方向反了

同 NEW-016。

---

## NEW-018｜argmax/argmin 全 NaN 窗口返回 `0.0`

### 后果
Unknown 被伪装成合法极值位置。

### 修复
全无有效值必须 `NaN`。

---

## NEW-019｜argmax/argmin 对 Inf 的 finite-mask 与 target 求法不一致

当前：
- `np.isfinite(seg).any()` 检查
- 但 `np.nanargmax/nanmax` 仍可看到 `Inf`

存在：
```text
[finite, +Inf]
```
时 target/hits 不一致甚至异常。

### 修复
先：
```python
valid = np.isfinite(seg)
v = seg[valid]
```
再求 extreme，并用原始 physical positions 映射回去。

---

## NEW-020｜`rolling_time_slope` 在缺失值时并不是 OLS slope

### 当前
提前对完整 window 构造固定中心化 `t` 权重。

缺失时：
- 删除 x 的 NaN
- 但不重新计算实际 physical time 坐标中心
- 不重新计算 denominator

### 后果
输出不等于任何清晰的 OLS time slope。

### 修复
二选一：

A. production 默认要求 contiguous full window；

B. 对有效 physical positions：
```python
t_valid = actual row offsets
OLS(y_valid ~ 1 + t_valid)
```

绝对不能压缩成 0,1,2... 假连续时间。

---

## NEW-021｜`rolling_regression` 的 cov/var/mean 不使用同一 paired cohort

### 当前
`cov(y,x)` 用 paired support，但：
- `var(x)` 独立 rolling
- `mean(x)` 独立 rolling
- `mean(y)` 独立 rolling

当 x/y 缺失位置不同，得到的 beta/intercept 不是同一个样本上的 OLS。

### 修复
每个窗口先：
```python
mask = finite(x) & finite(y)
```
所有统计全部基于同一 mask。

---

## NEW-022｜`rolling_regression(retval="residual")` 数学对象不清晰

当前流程实质：
1. 每个 t 用含 t 的 rolling OLS；
2. 算当前 in-sample residual；
3. 再对这些由不同模型产生的 residual 做第二个 rolling mean。

它既不是：
- 当前 residual
- 单一窗口 residual mean
- prior forecast error

### 修复
删除这个混合对象。

拆：
```text
ts_regression_in_sample_resid        diagnostic only
ts_regression_forecast_error         fit <= t-1
ts_regression_forecast_error_z       prior training scale
```

---

## NEW-023｜`rolling_regression` 未知 `retval` 默认为 slope

### 问题
非法 enum 被静默解释。

### 修复
未知值直接 ValueError。

---

## NEW-024｜`rolling_beta` 默认 `min_periods=2` 统计上过弱

2 个点 slope 几乎没有意义。

### 修复
生产默认 minimum：
```text
>= 5
```
更推荐：
```text
max(5, ceil(window * support_ratio))
```
support policy 不搜索。

---

## NEW-025｜top/bottom-k kernel 会在有效样本少于 k 时自动缩成更小 k

当前：
```python
take = min(k, valid.size)
```

### 后果
同一个 `topk_5` 有时实际 top2，有时 top5。

### 修复
默认：
```text
valid.size < k -> NaN
```
如需 adaptive-k，单独 canonical 并暴露 `effective_k`。

---

## NEW-026｜`rolling_top_n_mean(x,n)` 的窗口和 k 都等于 n，退化为 rolling mean

如果 window=n、top k=n：
所有有效值都被选中。

因此：
```text
rolling_top_n_mean(x,n) ~= rolling_mean(x,n)
rolling_top_n_sum(x,n)  ~= rolling_sum(x,n)
rolling_top_n_std(x,n)  ~= rolling_std(x,n)
```

### 修复
- 删除这些语义重复 canonical；
- 或 public signature 改成 `(x, window, k)`。

---

## NEW-027｜`rolling_linear_weighted` 默认 partial warmup

### 当前
第一条数据就可输出，且缺失时重归一化。

### 风险
同一个 WMA/decay canonical 在样本前端、gap 后使用不同有效窗口。

### 修复
每个 caller 声明：
- FULL_WINDOW
- 或 minimum coverage
- CurrentRequired

---

## NEW-028｜`rolling_linear_weighted` 当前 x 为 NaN 时可能输出历史旧值

对很多“当前技术状态”不应这样。

### 修复
由 caller 的 `CurrentRequired` 控制：
```text
current x missing -> NaN
```

---

# 4. 本轮新增：core time-series / statistics

## NEW-029｜`ACF` 直接 drop finite 后重新连接时间

### 位置
`common/statistics.py`

当前：
```python
s = s[np.isfinite(s)]
```
然后再做 lag。

### 后果
真实：
```text
t, missing, t+2
```
会被当成相邻样本。

### 修复
- FULL_CONTIGUOUS_WINDOW；或
- 只使用 physical lag pair `(i, i+lag)` 均 finite。

---

## NEW-030｜ACF `min_periods=1` 与 window identity 不一致

固定 window 的 ACF 不能 N 从 2 慢慢变到 W 而仍当一个统计量。

---

## NEW-031｜Corr/Cov/Covariance 大量 `min_periods=1`

必须明确 effective pair support，而不是让 Pandas 隐式决定。

---

## NEW-032｜Median/Percentile/Sum/Var/Skew/Kurt 等基础算子仍大量 partial warmup

不是所有简单 rolling 都必须 FULL_WINDOW，但必须**显式声明**，不能依赖 Pandas 默认。

建议：
- location/sum 可 `MIN_SUPPORT_WINDOW`
- skew/kurt/corr/beta 等要求更高支持
- search/certification 记录 effective N。

---

## NEW-033｜扩展窗口算子依赖“数据集从哪里开始”

`expanding_mean/sum/std/rank/max/min/...`

同一个股票：
- 从 2010 开始加载
- 从 2020 开始加载

2026 输出不同。

### 修复
增加：
```text
OriginSemantics
```
例如：
- LISTING_ORIGIN
- DATASET_ORIGIN (research only)
- FIXED_CALENDAR_ORIGIN
- CHECKPOINT_ORIGIN

默认自动 mining 不应大量使用 DATASET_ORIGIN expanding state。

---

## NEW-034｜ExpandingMean/Sum 用 `pd.isna`，Inf 会永久污染状态

### 修复
统一 `np.isfinite`。

---

## NEW-035｜ExpandingMax/Min 与 ExpandingMean/Sum 在 current NaN 语义不同

Pandas expanding max/min 往往会继续输出旧极值；
自定义 mean/sum 当前 NaN 输出 NaN。

### 修复
所有 expanding canonical 明确 `CurrentRequired`。

---

## NEW-036｜hypothesis-test p-value 算子不适合 default alpha mining

例如：
- Bartlett
- 其他 whole/expanding iid test

金融时间序列存在自相关、异方差，p-value 假设常不成立。

### 修复
这些统一：
```text
MiningRole.DIAGNOSTIC / RESEARCH
terminal_allowed=False
```

如要因子，做 effect-size/statistic，而非裸 p-value。

---

# 5. 本轮新增：dedupe / alias 可能主动把正确数学改错

## NEW-037｜`Mad` 被 alias 到语义不同的 `ts_mad`

### 位置
- `common/statistics.py`
- `_dedupe.py`

当前源文件明确区分：
- `Mad / ts_mean_abs_deviation`：单窗口 mean absolute deviation
- `ts_mad`：历史上另一种非标准双滚动语义

但 dedupe 映射：
```text
Mad -> ts_mad
mad -> ts_mad
```

### 修复
alias 只能在**数学等价**时建立。

推荐：
```text
Mad -> ts_mean_abs_deviation
mad -> ts_mean_abs_deviation
```
若历史 DSL 兼容必须保留旧语义，则使用 semantic-versioned migration，不能静默改含义。

---

## NEW-038｜去重规则按名称手工维护，不是按数学指纹

### 修复
每条 alias 必须生成 `AliasEquivalenceCertificate`：
- parameter mapping
- output equality
- NaN equality
- ties
- edge cases
- units
- window semantics
- current inclusion
全部一致才能 alias。

---

## NEW-039｜scalar 参数顺序跨 backend 的 ABI 审计不完整

### 位置
`registration_audit.py`

当前 logical audit 主要比较 panel parameter positions，允许 trailing scalar reorder。

### 示例
Pandas：
```text
(x, window, lag)
```
Polars：
```text
(x, lag, window)
```
可能通过 panel-prefix 检查，但位置调用完全错。

### 修复
如果允许 positional scalar：
**所有 scalar positional order 必须完全相同**。

如果希望 backend 内部顺序不同：
统一 public call binding 后以 keyword dict 传给 backend，禁止 backend 直接处理 public positional ABI。

---

## NEW-040｜registration audit 注释说检查 required/optional，实际 logical signature 没记录

### 修复
LogicalParamSpec 必须包含：
```text
name
role
position
required
default
keyword_only
semantic_type
```

---

## NEW-041｜replacement history 被最终排序，丢失真实覆盖时间顺序

### 风险
调试“是谁最后覆盖了谁”时，按名字排序不是执行顺序。

### 修复
每条 replacement 记录：
```text
sequence_id
load_phase
timestamp/order
old_definition_id
new_definition_id
```
保留原始链顺序。

---

## NEW-042｜compatibility alias 如果目标 operator missing policy 改过，会产生隐形版本迁移

例：
`last/cum_last -> ffill`

如果新 ffill 加 bounded/source gate，旧 formula 语义会变化。

### 修复
alias migration 必须带 semantic version；历史持久化 factor 使用原 definition_id，不只字符串 canonical。

---

# 6. 本轮新增：Polars “backend 有槽位”与“原生执行”仍容易混淆

## NEW-043｜`register_polars_udf` 本质仍回到 pandas

### 位置
`rolling_pack.py`
`polars_gap_coverage.py`

名称虽然是 Polars UDF，但实现：
```text
pl -> pandas -> pandas operator -> pl
```

### 修复
capability 明确分：
```text
PANDAS_NATIVE
POLARS_NATIVE
POLARS_PANDAS_DELEGATE
SQL_NATIVE
```

任何性能选择器不得把 delegate 当 native。

---

## NEW-044｜“给每个 pandas-only operator 注册一个 polars slot”会扭曲 backend coverage

即使 metadata 有 delegate label，很多上层代码如果只判断：
```python
"polars" in backends
```
仍会误报。

### 修复
所有 backend consumer 必须查询 `BackendKind`，禁止只看 backend key。

---

## NEW-045｜Polars bridge 的 `_SKIP_PANEL` / `SKIP_COLUMNS` 名单不统一

不同 helper 排除：
- date
- stock_code
- timestamp
- trade_date
- datetime
- ts
- inst
- instrument
...

### 风险
某个 metadata column 被当 factor feature，或某个真实 field 因名字碰撞被删。

### 修复
不要靠列名猜 metadata。
使用 typed `Panel`：
```text
time_axis
instrument_axis
value_columns
metadata_columns
```

---

## NEW-046｜Polars bridge 对 grain-changing operator 只能报错，不代表 FactorEngine 能运行该 operator

这正说明 minute→daily 需要专用 grain-transform runtime，而不是 generic shape-preserving bridge。

---

# 7. 本轮新增：ConditionBool / EventBool 的 Inf 语义硬 Bug

## NEW-047｜`daily_panel._assert_condition_bool` 会放过 ±Inf

### 原因
```python
finite = np.isfinite(cv)
bad = finite & ...
```
Inf 不是 finite，所以不报错。

随后：
```python
condition.notna()
condition.ne(0)
bool(value)
```
又把 Inf 当真。

### 修复
合法集合严格是：
```text
0
1
NaN
```
任何其他值，包括 ±Inf，直接 ValueError。

---

## NEW-048｜同一个非法 Inf condition 在不同 family 行为不同

`daily_panel` 可能读成 True；
`stateful._common.truth_mask` 因 `np.isfinite` 读成 False。

### 修复
只能有一个共享 `TriStateBool` 解析器。

---

## NEW-049｜`conditional_ext` 复制了另一份 ConditionBool validator

重复实现本身就是漂移源。

### 修复
删掉所有 module-local boolean validator。

---

## NEW-050｜`valid_trade` 同样不是 strict bool

### 位置
`ashare/state_machine.py`

当前：
```python
finite and value != 0
```

### 后果
`0.2/-1/2` 都变成可交易。

### 修复
定义：
```text
TradableBool = {0,1,NaN}
```
坏值直接 error。

---

## NEW-051｜`is_suspend` 同样需要 strict bool

不能 finite nonzero。

---

# 8. 本轮新增：A股涨跌停状态机

## NEW-052｜`tick_tolerance=0.005` 被按相对比例使用，不是真正 tick tolerance

0.005 = 0.5%。

这远大于 A 股常见最小报价单位概念。

### 修复
涨跌停是否命中应优先基于数据源提供的：
```text
high_limit
low_limit
```
并使用：
- exact rounded price comparison
- 或 absolute tick size（如价格最小单位），不是相对 0.5%。

参数改名：
```text
price_tolerance_abs
```
或完全从 MarketAdapter 获取。

---

## NEW-053｜limit price 必须 PositivePrice

`high_limit/low_limit <= 0` 应坏数据，不可参与 ratio/tolerance。

---

## NEW-054｜`ashare_limit_asymmetry` 用 OR 定义 known day

当前类似：
```text
known = finite(up) OR finite(down)
```

如果 up 已知、down unknown，down 会通过 nansum 被当 0。

### 修复
要比较 up-down：
```text
known = finite(up) AND finite(down)
```
或明确三态逻辑。

---

## NEW-055｜`ashare_limit_event_density` 当前 `known_status=0` 仍可能输出过去密度

current unknown/non-known 状态不应继续给一个看似当前有效的 density。

### 修复
如果 CurrentRequired：
```text
known_status[t] != 1 -> NaN
```

---

## NEW-056｜limit event/count 输入未统一 strict EventBool

所有：
- limit_up_event
- limit_down_event
- touch event
- failed event
都统一 EventBool。

---

## NEW-057｜rolling limit count 随 known-day 数变化，原始 count 被 missingness 污染

### 修复
同时提供：
```text
event_count
known_day_count
event_density
```
production terminal 优先 density/coverage-gated count。

---

## NEW-058｜days_since `max_lookback` 边界语义不清

当前常见：
```python
distance < max_lookback
```
不是 `<=`。

### 修复
明确定义：
- “最多允许 age == max_lookback”
或
- “历史窗口长度 max_lookback 包含当前，因此最大 age=max_lookback-1”

写入 docs + golden test。

---

## NEW-059｜one-price board 需要 OHLC structural validity

不能只看四价接近 limit。
同时检查：
```text
O,H,L,C > 0
H >= max(O,C)
L <= min(O,C)
H >= L
```

---

## NEW-060｜A股 market-rule 时间版本必须来自 MarketAdapter

不能 operator 自己知道：
- 10%
- 20%
- 30%
- 5% ST
- IPO 特殊期
- 北交所
- 历史规则变化

operator 只消费当日官方 limit prices/state。

---

# 9. 本轮新增：minute → daily 聚合核心

## NEW-061｜bar width 仍从实际观测 gap 推断

### 位置
`microstructure/intraday_agg.py::_bar_width_minutes`

### 问题
数据缺失可以改变 mode gap，从而改变 expected slots。

### 修复
`BarFrequency` 必须来自 dataset metadata/SessionContext。

---

## NEW-062｜TradeDate 仍大量由 `index.normalize()` 推断

### 风险
- UTC index
- 夜盘/跨午夜（未来其他市场）
- timezone conversion
- 数据源时区错误

### 修复
统一：
```python
SessionCalendar.trade_date(timestamp)
```

---

## NEW-063｜`_daily_agg_two` 未在 helper 层 exact SameAxis

虽然中央 validator可能拦，但 helper direct call 仍有风险。

### 修复
helper 内 defense-in-depth `assert_same_axis`。

---

## NEW-064｜`_daily_agg_three` 只 `dropna(subset=["a"])`

b/c 缺失仍进入 group。

### 修复
按 operator 声明：
- complete-case triple
- pairwise
- optional
明确处理，不能 helper 默认只看第一路。

---

## NEW-065｜`dropna` 只过滤 NaN，Inf 仍可能进入 minute aggregate

统一 isfinite。

---

## NEW-066｜segment return 用“第一个有限值/最后一个有限值”可能跨越内部缺 bar

### 修复
声明 coverage / exact segment endpoint：
- 如果要 official segment return，应要求 segment open/close slot 存在；
- 如果允许 first/last observed，诚实名 `observed_segment_return`。

---

## NEW-067｜segment volume/amount share `nansum` 把 missing bar 当 0 activity

### 修复
缺失 bar不是零成交。
要求：
```text
session coverage >= threshold
```
并只在 coverage 合格后聚合。

---

## NEW-068｜segment enum 没有统一 ParamSpec

`segment` 必须 choices：
```text
morning
afternoon
```
且 role=SESSION_POLICY 或 ECONOMIC（若确实希望搜索时段则用 reviewed enum）。

---

## NEW-069｜minute source completeness 不应该由 operator 自己推测

统一 `SessionContext`：
```yaml
market:
trade_date:
timezone:
bar_frequency:
expected_slots:
observed_slots:
session_segments:
is_complete:
coverage:
duplicate_bar_count:
out_of_grid_count:
```

---

## NEW-070｜日频生产层需要合法接受 GrainTransform

runtime 不能继续假设所有 factor node shape-preserving。

---

# 10. 本轮新增：`safe_ops` 自身存在“安全性”不一致

## NEW-071｜`ts_valid_count` 文档说 finite，Pandas 实现却是 `rolling.count()`

`rolling.count()` 会把 ±Inf 计入。

### 修复
先 `np.isfinite` mask，再 rolling sum。

---

## NEW-072｜Polars valid count 只 `fill_nan(None)`，Inf 仍有效

同样修为 `is_finite()`。

---

## NEW-073｜ts coverage 与 cs/group coverage 有效值定义不同

必须统一 `FiniteObservation`.

---

## NEW-074｜safe argext 的 valid mask 与 `np.nanmax/nanmin` target 不一致

窗口同时有 finite 与 Inf 时可能：
- target=Inf
- valid excludes Inf
- hits empty
- 异常或错误。

### 修复
target 只能在 `seg[isfinite]` 上计算。

---

## NEW-075｜`ts_ffill_limited(lineage=None)` 仍默认允许 forward fill

### 问题
未知 lineage 应 fail closed，不应解释为 price/level。

### 修复
`lineage=None`：
- production → reject
- research compatibility → 可显式 opt-in

---

## NEW-076｜`FFILL_LINEAGE_POLICY` 粒度太粗

“financial=False”并不总对：
- balance-sheet stock as-of daily panel 通常合法 carry-forward
- earnings surprise event 不应该 daily ffill 当独立 observation

真正需要的是：
```text
FieldMissingPolicy / ObservationClock
```
而不是只看 `financial` 这个大类。

---

# 11. 本轮新增：data_cleaning forward-fill contract

## NEW-077｜`_FORWARD_FILL_ALLOWED_DEFAULT=True`

### 问题
未知字段默认允许填充，是生产 fail-open。

### 修复
默认 False/Unknown→reject。
FieldSpec 明确许可才能 ffill。

---

## NEW-078｜`max_ffill_gap=0` 当前代表无限填充，语义危险

更自然：
- 0 = 不填
- None = unlimited（通常 production 禁）
或显式 enum。

---

## NEW-079｜ffill gap 只按行数，不按 source clock

对：
- daily price
- fiscal stock
- minute quote
需要完全不同 staleness。

### 修复
`StalenessPolicy` 支持：
```text
bar_count
trading_days
calendar_days
report_periods
sessions
```

---

## NEW-080｜EWM 在 data_cleaning/technical/overhaul 有多套实现

统一到一个 state kernel，所有 alias 指向同一 definition_id。

---

# 12. 本轮新增：fundamental typing 与 PIT ledger

## NEW-081｜`transforms_v2._register` 基本没有完整 ParamSpec

大量 production-status fundamental operator 依赖中央 fallback。

### 修复
每个：
- periods
- periods_per_year
- short/long
- window_periods
- flow_type
必须明确 ParamSpec/role/active_when。

---

## NEW-082｜`fin_pct_change` / `fin_growth` 的 input semantic typing 过窄或错误

当前部分 metadata 把 x 标为 `rate`。

但真正常见用途包括：
- revenue
- profit
- assets
- inventory
等 level/flow/stock 的 period growth。

### 修复
不要把“输出是 growth rate”误写成“输入必须 rate”。

输入应该是：
```text
ComparableLevel
SinglePeriodFlow
Stock
```
按 operator definition区分。

---

## NEW-083｜`fin_log_change` 对 signed financial series需要明确正域

当前 cur/old <=0 → NaN 是合理的，但 input semantic 应写：
`PositiveComparableLevel`，不能 generic rate。

---

## NEW-084｜`fin_pct_change` 对 denominator 接近 0 只用 absolute EPS

大规模/小规模字段统一 1e-12 不够。

### 修复
field-unit-aware denominator policy 或 signed symmetric growth。

---

## NEW-085｜fundamental revision ledger并不能凭 daily as-of panel自动证明“真实公开修订”

如果数据源没有 vintage/revision history，daily panel 中数值变化可能来自：
- provider correction
- backfill
- reconstruction
而非真实上市公司 restatement。

### 修复
`RevisionLedger` production 必须要求 source contract：
```text
period_id
value
observed_at/release_at
revision/vintage identity
```
没有真实 vintage 时只能叫 `observed_value_change` proxy。

---

## NEW-086｜revision change 判断用 absolute `_EPS=1e-12`

财务数值可能 1e12，浮点序列化微差会被算 revision。

### 修复
优先使用 source revision/version key。
次选 normalized exact-source value；不能用全字段统一 absolute epsilon。

---

## NEW-087｜ledger `period_id.reindex(...)` 会隐藏轴不一致

production 应 exact SameAxis fail，而不是 silently reindex。

---

## NEW-088｜ledger `_pos_int` 仍有 `int(value)` 截断

统一 strict helper。

---

# 13. 本轮新增：group / cross-sectional 基础实现

## NEW-089｜`c_count` 使用 `.count()`，Inf 被当有效

统一 finite.

---

## NEW-090｜CS global broadcast stats必须 machine role=GLOBAL_STATE

包括：
- cs_count
- cs_mean
- cs_std
- cs_sum
- cs_median
- cs_quantile（若广播）
- cs_valid_count
- cs_coverage

不能作为 standalone rank alpha。

---

## NEW-091｜`aggr_top_n` tie boundary依赖列顺序

当前 top-N 通过排序后直接 `[:top]`。

### 修复
- tie-inclusive threshold + effective_k
或
- stable InstrumentId secondary key（定义必须明确）

不能依赖 DataFrame column arrival order。

---

## NEW-092｜`aggr_top_n` 的 fallback改变 operator 含义

例如：
- `sort_col=None -> x`
- `x=None -> empty DataFrame`

### 修复
public canonical 必需输入缺失就报错。
便利 fallback 放 recipe/compiler，不放 kernel。

---

## NEW-093｜`aggr_top_n` 更像 routing/mask，不应直接当普通 terminal alpha

标记：
`RECIPE_INTERNAL / CROSS_SECTION_SELECTOR`

---

## NEW-094｜`group_ts_decay_linear` 是“时间 + group”混合算子，但 scope 很容易被推成 group

### 风险
history contract 可能得到 0 lookback。

### 修复
增加组合 scope：
```text
TIME_THEN_GROUP
GROUP_THEN_TIME
```
并由 operator self-declare history。

---

## NEW-095｜`group_ts_decay_linear` 仍 `max(1,int(window))`

silent clamp/truncate。

---

## NEW-096｜`normalize/fallback_policy` 是隐藏 scalar controls

metadata必须声明，role=MISSING/GROUP_POLICY，non-search。

---

## NEW-097｜group z-score/normalize 缺 minimum group size

2只股票也能产生 ±1 类极端 z-score。

### 修复
所有 group statistic 有 SupportPolicy。

---

## NEW-098｜group ex-self mean/weighted mean 未统一 missing group key

`group_ext.py` 某些路径直接 `pd.unique(g_row)` / `g_row == label`，没有统一使用 `is_missing_group_key`。

`None` / `""` 等可能形成 phantom peer group。

### 修复
所有 group family只能调用一个 `GroupKey` parser。

---

## NEW-099｜hierarchical neutralize 合法但只有1个成员的 subgroup 会 residual=0

单成员组没有“中性化信息”，0 会被误读为完美中性。

### 修复
min_group_size >=2，回归/多参数更高。

---

## NEW-100｜trimmed OLS 的 trim 边界 ties可依赖股票列顺序

`np.argsort(xs)[:cut]` 在 ties 中任意切。

### 修复
按 value threshold tie-inclusive，或 deterministic instrument-id tie contract。

---

## NEW-101｜trimmed OLS 仅按 x tail 截尾，名称/文档必须明确

如果不是双变量 robust regression，不要让用户理解成一般“稳健残差”。

---

# 14. 本轮新增：cross-section extension

## NEW-102｜KNN/Local Moran仍需要 UniverseMask

rank-standardized feature graph不能把不在研究 universe 的 physical columns算 peer。

---

## NEW-103｜KNN local Moran 需要 effective neighbor count输出/门控

tie-inclusive kNN 可 >k；这会改变局部统计的尺度。

---

## NEW-104｜isotonic residual 的单日方向选择阈值 `_RHO_EPS=0.05` 是隐藏 estimator policy

版本化，non-search。

---

## NEW-105｜isotonic方向是在同一个 y cross-section上选择并拟合

作为“非线性中性化 residual”可以保留，但它不是 OOS predictive model。
role应明确：
`CROSS_SECTION_TRANSFORM`，不要被文档描述成预测。

---

## NEW-106｜group tail coexceedance current-members retrospective语义必须进入 identity

今天的 group membership 被用于历史窗口，与 historical contemporaneous membership 是两个不同统计量。

---

# 15. 本轮新增：relation

## NEW-107｜`group_signal_attraction_share` 仍把 signed signal `max(sig, EPS)`

负信号全部被压成几乎0。

### 修复
输入类型只能：
`PositiveWeight`；
或要求 caller 显式 positive transform。

---

## NEW-108｜relation module 遇到 axis mismatch时 reindex，而不是 fail

`ops_ext.py` 当前会把 group reindex 到 x。

production multi-input必须 exact axis。

---

## NEW-109｜deprecated `relation_pagerank_centrality` 只能 alias，不得作为独立 active canonical/证书

保持 research alias，并确保所有 manifest/resolver都只看到一个 canonical identity。

---

# 16. 本轮新增：conditional operators

## NEW-110｜`conditional_ext` 多处直接 `int(window)` / clamp min_periods

统一 ParamSpec strict。

---

## NEW-111｜conditional corr/beta默认允许非常少 selected observations

至少使用 support ratio + hard floor。

---

## NEW-112｜`ts_corr_if/ts_beta_if` 是否包含 current sample必须机器声明

当前统计包含当前 selected point；
`ts_regression_resid_if` 却明确 fit 排除 current。

这些不是同一 current-inclusion contract。

### 修复
metadata:
```text
reference_window=current_inclusive / strict_prior
```

---

## NEW-113｜`ts_regression_resid_if` 当前实现方向较正确，但 training support要基于 pair + condition complete-case

继续保持，并增加 rank/condition number checks。

---

# 17. 本轮新增：execution/history contract 的具体漏洞

## NEW-114｜`declare_stateful(... minimum_history=int(...))` 自己也可能截断

声明 API 也必须 strict typed。

---

## NEW-115｜`_STATEFUL_CANONICALS` 中有已不存在 canonical

最终 CI：
```text
legacy_stateful_set - final_registry == empty
final_stateful_registry - declared_contracts == empty
```

---

## NEW-116｜history fallback仍按参数名字猜 window-like

任何未显式 HistoryTransform 的 production operator都不应走 name-based fallback。

---

## NEW-117｜event/report/session count不能简单在 DAG里当无限 history而不记录 observation count

保留：
```text
kind
count
minimum_bar_floor
staleness cap
```

---

## NEW-118｜HistoryRequirement 与 `window_semantics` 必须交叉验证

例如：
```text
CONTIGUOUS_FULL_WINDOW window=60
```
history floor至少59 prior rows，且 gap后需要重新warm。

---

## NEW-119｜recursive operator warmup不能用简单 W-1替代

EMA/Wilder/KAMA/GARCH/Kalman：
history与 state initialization是不同概念。

---

## NEW-120｜同一 factor DAG 组合 history不能只取 max，某些串联窗口必须相加

已有部分 compound transform，但最终应由每个 node 的 history transform composition自动完成。

---

# 18. 本轮新增：final registry / certification invariants

## NEW-121｜源 metadata 与 final catalog必须 hash一致

建立：
`LogicalContractHash`

backend替换只能改变 implementation hash/capability，不得改变 logical hash。

---

## NEW-122｜任何 late overlay 改 logical contract必须重新 semantic version

不能 mutate in-place 后仍 definition_id不变。

---

## NEW-123｜final frozen registry必须没有 unclassified operator

```text
UNCLASSIFIED == 0
```

---

## NEW-124｜surface membership不能兼任 mining role

`daily/extended/research` 是 authoring tier；
`ALPHA/STATE/CONDITION/...` 是 mining role。

两者独立。

---

## NEW-125｜`research` 不应该成为“有问题但先放着”的垃圾桶

factor-shaped、有价值且可修的 operator：
修后 promotion；
纯 diagnostic：
明确 diagnostic role；
无意义：
删除。

---

# 19. 之前 Master Spec 中必须继续保留并逐项验证的问题

下面开始合并此前所有轮次的算子审计。即使某项你已经改过，也必须由 AI **对当前 final runtime 重新验证**，不能因为注释写着“R11 fixed”就直接勾掉。


## 19.1 历史 Master Spec：必须重新验证的算子问题总表

> 本节不是“旧问题可以忽略”，而是 **回归测试清单**。执行 AI 必须在当前 final registry 上逐项判断：`已修复 / 部分修复 / 未修复 / 修复错误 / 被 alias/overlay 重新污染`。凡是没有用测试证明的，不得标记“已完成”。

### HIST-001｜`downside_risk`：current drawdown duration 不得跨 NaN gap
- **问题**：历史实现遇到 NaN 时仅 `continue`，连续 underwater episode 可能穿过未知区间继续计数。
- **修复**：NaN 必须 `output[t]=NaN` 且 reset episode state；如要 carry，必须另建显式 `missing_policy=carry` canonical，默认生产禁用。
- **测试**：构造 `1.0,0.9,NaN,0.8`，gap 后的 duration 不得继承 gap 前长度。

### HIST-002｜`ts_best_lag_corr`：不要把 raw max-|corr| 与多重比较校正后的统计量混在一个 canonical
- **问题**：不同 lag 数量会自然提高最大相关；Fisher-z/heuristic null 又与原始最大相关是不同统计定义。
- **修复**：拆成 `ts_best_lag_corr_raw`、`ts_best_lag_corr_excess/null_adjusted`；输出最佳 lag 单独 canonical。
- **测试**：白噪声下随着 `max_lag` 增大，raw max 会抬升；校正版应保持近似稳定。

### HIST-003｜relation/distribution 全族：`window/lag/bins/min_periods` 必须补齐 ParamSpec + ParamRole
- 禁止运行时 `int()`/`max()` 静默把非法搜索点改成另一个合法点。
- `bins/grid/n_surrogates` 等应归 `ESTIMATOR_RESOLUTION`，默认不进入自动挖掘。

### HIST-004｜kurtosis 命名必须区分 Pearson 与 excess
- 正态分布 Pearson≈3，excess≈0。
- 所有 `kurt/kurtosis`、relation kurt、idio kurt、weighted kurt 都必须声明 convention；建议 canonical 明确 `*_pearson_kurtosis` / `*_excess_kurtosis`。

### HIST-005｜entropy 输出单位必须统一
- raw Shannon entropy = `nats`（若自然对数）；normalized entropy = `dimensionless [0,1]`。
- 同一 canonical 不允许通过 bool 参数切换输出单位。

### HIST-006｜distance covariance / covariance / beta 的单位代数
- `Cov(x,y)=unit(x)*unit(y)`；`dCov` 不是 generic level；`Beta(y~x)=unit(y)/unit(x)`；`R2/corr/tstat` dimensionless。
- 建立自动 unit-algebra golden test。

### HIST-007｜MI/dCorr/TE 不得通过内部 `max(min_periods,10)` 制造假参数
- 合法域应由 ParamSpec 拒绝；搜索器不应生成 3 后在 kernel 内被改成 10。

### HIST-008｜lagged MI / TE 的 raw history 必须显式包含 lag
- 如果统计窗口需要 W 个有效 lagged pairs，则原始历史至少 `W + lag`；history requirement 不能仅 W。

### HIST-009｜tail coexceedance / expected shortfall 的最小 tail count 不能静默 clamp
- `min_tail_count` 是 support policy；非法值拒绝。
- 还应增加 tail effective mass / N_eff，而不只 raw count。

### HIST-010｜Effective Transfer Entropy 不得在 masked/dropna 后压缩时间轴
- lag/embedding 拓扑必须基于原始 physical rows；任何 NaN gap 不得使 `t-2` 与 `t` 重新成为相邻点。

### HIST-011｜ordinal-pattern “ratio” 不是 ratio 时必须诚实重命名
- 如果实现是 `observed - null`、positive excess、clipped excess，则 canonical 必须体现 `excess`/`positive_excess`。
- 不允许把负 excess clip 掉后仍叫“deviation”。

### HIST-012｜spectral family：raw price 与 return spectrum 不应共享一个无模式 identity 的 canonical
- 必须由输入 SemanticType 区分 `PriceLevelSpectrum` / `ReturnSpectrum`，或拆 canonical。
- PSD 频率单位、detrend、demean、sampling grain 必须进入 definition_id。

### HIST-013｜KAMA / Supertrend / PSAR 等递归技术指标：gap 后重新 warmup
- current missing -> current output NaN；state 不得偷偷冻结后继续。
- gap 后至少重新形成 ER/ATR/initial trend 所需历史。

### HIST-014｜技术指标 ordering relation compile-time prune
- `fast < slow`、`short < medium < long`、`signal < slow` 等放 RelationalParamSpec；搜索前剪枝。

### HIST-015｜Supertrend `multiplier=0` 等退化点
- 如果为数学恒等/无意义边界，应从搜索域删除；若保留，必须 nonsearch + compatibility。

### HIST-016｜regression_models：样本内残差与 prior predictive error 分离
- `*_resid` 若使用当前样本参与拟合，就是 diagnostic；生产 alpha 应优先 `*_forecast_error` / `*_prior`。

### HIST-017｜variance ratio proxy 不得冒充 Lo–MacKinlay VR
- 若只是 `var(k-period return)/(k var(1-period))` 的简化版，canonical 使用 `_proxy`；标准版本另实现 finite-sample/overlap convention。

### HIST-018｜不要对“已经是 return”的输入再次 `diff()`
- 输入 SemanticType 必须决定是否先构造 increment。

### HIST-019｜level/vol shift 不得 `dropna()` 后把前后两半重新切分
- change-point/shift 统计要保持 physical cohort；缺失应 censor 或采用明确 aligned blocks。

### HIST-020｜CUSUM proxy / break score 的命名必须诚实
- heuristic score 不能叫标准统计量；若不是标准 CUSUM test，使用 `_score` / `_proxy`。

### HIST-021｜state_latch / state_hold / event_refractory / conditional 全族必须用唯一 ConditionBool
- 合法集合仅 `{0,1,NaN}`；`±Inf`、`-1`、`0.2`、`2` 均报错。

### HIST-022｜state_slew_limit / deadband 等退化 scalar
- 0 若导致永久冻结或恒等，必须明确是否有经济意义；无意义则搜索剪枝。

### HIST-023｜state operators 的 threshold/limit 单位必须 same_as input
- 不能把价格阈值、收益阈值作为无量纲 free scalar 混用。

### HIST-024｜`ts_lag_of_peak_corr` 输出不要除以 `max_lag` 混入参数尺度
- 建议主 canonical 输出物理 `bars`；另建 normalized lag ratio。

### HIST-025｜high-lag candidate 的支持数不平衡
- 选最佳 lag 时每个 lag 应使用可比 cohort / 最小 support；否则高 lag 少样本的极端估计更易胜出。

### HIST-026｜tie 不能默认偏低 lag
- lag 相关并列时使用明确 deterministic policy（例如返回 tie set 中位数、或 all-tie summary），不能暗含“最先出现”。

### HIST-027｜Directional Change 初始 state
- 在第一次确认方向前也要维护 running high/low extrema；不能从首个确认点才开始，导致 first event 偏移。

### HIST-028｜Directional Change 要求 PositivePrice + threshold domain
- threshold 必须正且有合理上界；价格<=0 fail closed。

### HIST-029｜state episode MFE/MAE/efficiency/retrace 的单位与定义拆分
- sum/mean/count/last 不应混在一个 canonical 的 mode 参数里造成输出单位改变。

### HIST-030｜survival/active state 不得 `finite & !=0` 接受任意分类数
- 对 EventBool/ConditionBool 严格 0/1；对 categorical state 使用独立 StateCode 类型。

### HIST-031｜dynamic KNN 所有 feature/target 必须 SameAxis
- kNN 特征 panel 任何 date/stock shift 都必须 fail-closed。

### HIST-032｜KNN tie-inclusive 邻居数可能 >k，必须暴露 effective peer count
- 不要假装恰好 k；提供 diagnostic 或在 statistic 中使用实际 count。

### HIST-033｜graph Dirichlet energy / peer mean 的单位
- peer mean = same_as target；Dirichlet energy 一般 `unit(target)^2`；不要写 ratio。

### HIST-034｜intraday realized beta ddof 要一致
- `cov(ddof=1)/var(ddof=0)` 会引入 `n/(n-1)` 偏差；使用同一 convention。

### HIST-035｜分钟 log-return 不得全局 `shift(1)` 跨 overnight
- 每个 SessionGrid 内重新计算；首 bar return 应 NaN 或使用明确 overnight component，绝不能混入 intraday RV/beta。

### HIST-036｜ex-self market return 的 numerator/denominator 必须同一有效 mask
- 排除自身后重新计算总权重时，missing/negative/zero weight cohort 必须一致。

### HIST-037｜A股分钟 session 必须以官方 240-bar grid 测试
- 09:31–11:30、13:01–15:00（按你们当前 timestamp convention）；午休 gap 不算 missing；239/241/重复/越界 bar 都要 fixture。

### HIST-038｜Markov persistence/state entropy 的 support 应基于 outgoing transitions
- 仅 occupancy 足够不代表某 state 有 transition；smoothing 不能凭先验制造 transition statistic。

### HIST-039｜transition surprisal 同理
- 当前 transition 没有真实 support 时不能只靠 pseudocount 输出有限值。

### HIST-040｜entropy production 的负值
- 理论上应非负；小数值负可视 floating error clip，显著负值必须 fail/test，不可无脑 `max(0,x)`。

### HIST-041｜AIS / Markov grid 不可行参数域必须提前剪枝
- 不要让大面积合法“表面参数”实际全部 NaN。

### HIST-042｜Kramers–Moyal D1/D2 单位
- `D1 = mean(dx)/lag` → `unit(x)/bar`; `D2 = mean(dx^2)/(2lag)` → `unit(x)^2/bar`。

### HIST-043｜全库 multi-input `.to_numpy()` 前必须 SameAxis
- 这是全局硬 gate；不要依赖“调用者应该对齐”。

### HIST-044｜Piotroski issuance 条件
- `cap_growth <= 5%` 是 tolerant proxy，不是“没有发行新股”；严格版与 proxy 版拆开。

### HIST-045｜Applicability/Mask 必须 Bool type
- `_masked` panel 不允许 arbitrary nonzero。

### HIST-046｜`(cur-old)/abs(old)` 不等于标准 growth
- signed change over absolute base 应诚实名；标准增长分母通常 old，且负 base 的经济解释不同。

### HIST-047｜cash burn runway 单位
- 若 cash / monthly burn，则 months；若 annual flow，则 years；必须由 grain/annualization 决定。

### HIST-048｜rough volatility p-variation 必须使用 mean/structure function，不要 raw SUM
- raw sum 随 scale 的 pair count N_delta 变化，会把采样数量趋势误读为 roughness slope。

### HIST-049｜rough-vol 每个 scale 的有效 pair count/coverage
- 每个尺度必须 minimum support；不能某尺度 3 pairs、另一尺度 50 pairs 后直接回归 log variation。

### HIST-050｜intrinsic dimension Theiler window 使用原始 physical embedding endpoint
- drop missing 后的 compressed index 不能拿来判断 temporal separation。

### HIST-051｜intrinsic dimension 去重不能固定 `round(10)`
- 数值尺度改变会改变唯一点数；使用 scale-aware tolerance / exact robust hashing。

### HIST-052｜Hankel / wavelet / path-signature “trailing contiguous” helper 当前 NaN 时不得回退到前一段
- current missing → empty block → current output NaN。

### HIST-053｜multifractal common cohort 也必须 current-required
- 当前样本缺失时不能用旧历史给今天输出一个新值。

### HIST-054｜persistent homology/H0 entropy current missing 不得 stale
- topology current-required contract统一。

### HIST-055｜marked event censor：要 latest/current episode，不要选窗口内 longest segment
- 选择 longest 会把“很久以前的长事件”当成当前 event state。

### HIST-056｜event interval coupling 不得读取窗口前无限远的 previous event
- history requirement 必须覆盖该 previous event，或明确 left-censored。

### HIST-057｜update-clock `n_updates` 不得仍 hardcode 5
- 所有 event-clock history count 都由参数 factory 返回。

### HIST-058｜kernel Granger 的 sigma_x / sigma_y
- X kernel 必须用自己的 bandwidth；同时 normalize kernel trace，避免 kernel scale改变 ridge 正则含义。

### HIST-059｜residualized HSIC train/test split 要 blocked/purged
- even/odd 切分会把相邻时点交叉泄漏；时序数据用 chronological split + embargo。

### HIST-060｜cross-spectral phase 必须有 coherence gate
- `min_coherence=0` 会在无共同谱能量时输出随机 phase；默认要 meaningful floor。

### HIST-061｜expectile output unit / solver convergence
- slope = unit(y)/unit(x)；未收敛必须 NaN + diagnostic，不可返回最后一次迭代。

### HIST-062｜expectile tau support
- 极端 tau 需 `N*min(tau,1-tau)` 样本下限；tau grid 需审阅。

### HIST-063｜binned response curvature 3 bins 问题
- 3点拟合二次曲线是 exact interpolation，几乎必然“有曲率”；提高 bins/support 或换离散二阶差分并做 uncertainty。

### HIST-064｜binned response tie handling
- 大量相同 sorter 值不能被硬切到低 bin；使用 tie-aware/weighted quantile bins。

### HIST-065｜normalized beta*sdx/sdy 实际就是 correlation
- 如果公式代数等价 corr，删除重复 canonical，不能用新名字扩大“假搜索空间”。

### HIST-066｜multifractal width/curvature 命名
- `H(1)-H(4)` 不是标准 singularity-spectrum width；未做 Legendre transform 就不要叫 spectrum width。

### HIST-067｜EVT threshold stability 的 runtime/metadata feasibility 一致
- k_min/k_max/threshold count必须有足够 tail observations；无效 grid 不进入搜索。

### HIST-068｜HVG raw entropy / graph size
- raw entropy随 N 变；优先 normalized variant或固定 full window。

### HIST-069｜local Lyapunov 输出单位
- 指数通常 `1/bar`；不要 ratio。

### HIST-070｜local Lyapunov 最近邻 eligibility 要先屏蔽未来 horizon
- 不能先选最近邻再发现其未来路径不完整；eligible candidate mask 应先建立。

### HIST-071｜first-passage scale 与 x 同单位
- threshold/scale必须 same_as x；anchors与side hit support单独定义。

### HIST-072｜structural nearest-level volatility scale 支持
- 只有2个return估出来的 vol 不可靠；提高最小 support。

### HIST-073｜structural level strength 的固定 proximity cutoff
- 隐藏 5% 等常量必须进入 definition/version；最好以 volatility/ATR 归一化。

### HIST-074｜state density bandwidth silent clamp
- invalid bandwidth直接拒绝；bandwidth属于 estimator resolution，默认不 search。

### HIST-075｜L-moments / dip statistic 不能4个点就生产输出
- 使用 estimator-specific minimum sample，最好>=20/30或文献建议。

### HIST-076｜bicoherence top-decile null 对 FFT resolution/window 敏感
- `n_segments` 等是 estimator resolution；固定/版本化，null calibration 必测。

### HIST-077｜advanced topology Wasserstein-1 definition统一
- nonempty/nonempty若用 Hungarian total cost，empty/nonempty也必须 total diagonal cost；不能一边 sum 一边 mean。

### HIST-078｜Takens dedup/sampling 不得按 lexicographic decimation 改变几何
- 若需降采样，用 deterministic geometry-aware subsampling，并进入 definition_id。

### HIST-079｜SPD Fisher information geometry优先 affine-invariant metric
- log-Euclidean/Frobenius 会有坐标/尺度依赖；若保留必须诚实名。

### HIST-080｜jump-robust / intraday kernel 必须声明 SessionCalendar + grain
- rolling window跨 session 时要明确 reset；不能把分钟样本当连续跨夜序列。

### HIST-081｜intraday volume entropy/HHI 的 missing 不得当 0 activity
- 缺失 bar≠零成交；必须 SessionGrid reindex + coverage。

### HIST-082｜intraday block curvature 要保留最新 remainder
- trailing 因子不能为了整块划分把最靠近当前时点的数据扔掉。

### HIST-083｜3 scales quadratic fit 不足以证明 curvature
- 三点二次精确拟合是参数化插值；需更多 scale 或降低模型阶数。

### HIST-084｜HVG forward/backward asymmetry 的 finite-window bias
- 需要 null calibration；样本短时 research only。

### HIST-085｜RQA effective graph size/window confounding
- recurrence matrix M 变化会改变 RR/entropy/trapping；production应固定/高覆盖窗口。

### HIST-086｜state episode excursion scale SameAxis + same unit
- scale/x axes和units必须严检；retrace>1语义说明。

### HIST-087｜threshold-cycle lower/upper 不应是 field-agnostic absolute constants
- 应 same-unit typed threshold，或使用 zscore/ATR-relative threshold。

### HIST-088｜cross-spectrum effective run长度变化导致 frequency bins变化
- 24 bars与120 bars不是同一个频率分辨率；生产需固定 support fraction/full window。

### HIST-089｜intraday impact shock overlap
- overlapping shocks不能重复占用同一 recovery path；refractory/episode collapse统一。

### HIST-090｜multifractal asymmetry 负 q 对零增量不可用
- exact zero increment + q<0 发散；需明确 zero handling，不可 epsilon 伪造。

### HIST-091｜advanced structure fixed random directions
- 即便 seeded，也要把实际 direction matrix hash放 definition_id；更推荐 deterministic quadrature方向。

### HIST-092｜candle-state Mahalanobis center/covariance一致
- median center + mean-centered covariance混搭估计器需改成一致 robust location/covariance。

### HIST-093｜Matrix Profile motif age 端点索引
- subsequence endpoint应 `(r-L+1)-best_s`，不是 `r-best_s`。

### HIST-094｜research spectral residualized/Granger/BDS 必须 golden-reference
- 这些统计的 finite-sample convention 很容易“看起来对”；必须与官方/论文参考实现比对。

### HIST-095｜AR mean-reversion：AR(p) 不能只拿 beta1解释全部
- half-life只对特定 AR(1)/continuous approximation有清晰含义；高阶模型拆开。

### HIST-096｜GARCH variance-targeted 要诚实名
- 无 mean model、单起点 Nelder-Mead、variance targeting 均是模型定义；不要叫 generic GARCH forecast。

### HIST-097｜HAR `*_rv_*` 若返回 sqrt(pred) 名称错误
- RV通常variance；sqrt是 realized volatility。重命名/拆分。

### HIST-098｜Kalman/state-space q/r 绝对尺度不能 field-agnostic自由搜索
- q/r应相对观测尺度参数化，或标准化输入；missing gap 要 max_gap/state expiration。

### HIST-099｜dynamic regression optional features 不能让 coefficient_index 语义漂移
- 使用命名输出 `beta_x1/beta_x2/...`；不要压缩 None 后让“第2系数”指向不同 exposure。

### HIST-100｜local moments/Markov/graph等所有“研究统计”都要 null calibration
- 白噪声/随机游走/独立双序列下输出应符合理论/模拟基线；不通过则 research only。

---

# 20. 本轮继续深挖后的新增问题（NEW-126 起）

> 下面是这一轮继续读取当前 `main` operator surface / shared helper 后新增的代码级问题。原则上优先于纯风格重构，因为其中很多会产生错误数值、错误搜索空间或错误生产行为。

## NEW-126｜`activity_clock.py` 仍在 runtime 静默截断/夹紧整数参数
**位置**：`cleaned_operators/activity_clock.py`

当前多处仍类似：
```python
sw = max(2, int(scale_window))
ml = max(1, int(max_lookback))
```
这与 strict ParamSpec 架构冲突：`5.9 -> 5`、`0 -> 2/1` 会让搜索器以为评估了原参数，实际评估另一个参数。

**解决**：统一调用 `strict_positive_int`；合法域由 ParamSpec/RelationalSpec提前拒绝，kernel只接受已经验证的 typed value。

**测试**：`5.9, True, "20", NaN, Inf, 0, -1` 全部必须按契约拒绝而不是变形。

## NEW-127｜activity-clock `budget` 缺完整 finite/domain contract
`budget=float(budget)` 后只检查 `<=EPS`；`NaN` 不满足 `<=`，可能进入全 NaN 死配置，`+Inf` 永远无法达到。

**解决**：`ParamSpec(dtype=float, min=..., max=..., finite=True, role=STATE_THRESHOLD/ECONOMIC)`；明确单位为 normalized activity units。

## NEW-128｜activity-clock `max_lookback` / `scale_window` / `budget` 的 ParamRole 仍不完整
- `scale_window`：HORIZON；
- `max_lookback`：HORIZON/support cap；
- `budget`：ECONOMIC/STATE_THRESHOLD；
- `include_current`：definition policy，不建议默认搜索。

## NEW-129｜activity scale 的“有限观测中位数”会让有效 lookback 随 missingness 改变
scale window 内会 drop missing prior activity；20-bar参数可能实际用5、10、20个观测。

**解决**：明确 `window_semantics="finite_observations"` + `min_scale_coverage`；生产默认 coverage>=80% 或 contiguous policy。提供 `ts_activity_clock_scale_coverage` diagnostic。

## NEW-130｜`intraday_activity_duration_curvature` 把 `calendar` 对象放进 operator param_names
**问题**：`SessionCalendar` 是 runtime execution context，不是 factor expression 的普通 scalar；对象不可稳定序列化/hash，也不应进入 AlphaProbe DSL。

**解决**：从 operator 参数移除 `calendar`；统一从 `ExecutionContext.session_calendar` / `MarketContext` 注入。definition_id只记录 calendar policy/version/hash。

## NEW-131｜`intraday_activity_duration` 仍硬编码 `Asia/Shanghai`
模块自称可由 calendar定义 session，但 timezone转换仍固定 `_SESSION_TZ`。

**解决**：timezone必须来自 SessionCalendar；未知 timezone fail closed，不猜。

## NEW-132｜官方 SessionGrid helper只理解“1分钟”
它根据 segment逐分钟展开，没有 bar frequency/granularity contract；5min/15min分钟源不适用。

**解决**：SessionCalendar输出 `expected_bar_timestamps(trade_date, bar_size, convention)`；operator不自己生成 minute-of-day。

## NEW-133｜`buckets` 是 estimator resolution，不应成为默认挖掘维度
activity-duration/volume-clock 中 buckets改变插值/曲率估计器本身。

**解决**：`ParamRole.ESTIMATOR_RESOLUTION, searchable=False`；只保留少数版本化配置，或每个 resolution有独立 definition_id。

## NEW-134｜`volume_clock.py` 等 activity-clock resampling对 B 高度敏感
即使要求 distinct points >= B+1，线性插值仍会制造未观测价格路径。

**解决**：明确 canonical 名为 `*_linearly_resampled_*`；B固定；增加 nearest/step empirical版本做 robustness。不要让模型通过搜索 B “挖插值器”。

## NEW-135｜volume-clock 左端点 convention 不明确
`Q` 第一个值通常 >0，但 grid 从0开始，`np.interp(0,Q,logp)` 实际拿第一个正activity价作为 p(0)。

**解决**：明确 `activity_clock_origin = first_positive_activity`，或显式插入 session-open price at Q=0。两种定义不能混在同一 canonical。

## NEW-136｜`composition.py` 的“financial_statement” composition family 在经济学上不成立
当前 allow-list把 Revenue / NetIncome / TotalAssets / Liabilities / Equity 放为同一 composition family。它们不是同一 part-whole：
- Revenue/NetIncome = flow；
- Assets/Liabilities/Equity = stock；
- Assets = Liabilities + Equity；
- Revenue + Assets + NetIncome 没有可加总的“整体”。

**解决**：CompositionSchema必须是明确 part-whole schema，例如：
- BalanceSheetFunding = {Liabilities, Equity} 相对 Assets；
- RevenueBreakdown = 各收入分部；
- CostBreakdown = COGS/SG&A/R&D/...；
- HolderShareComposition = holder shares；
- IntradayActivityComposition = time-bucket volume shares。
严禁泛化“同币种就可 composition”。

## NEW-137｜CompositionSchema 对未知字段是 fail-open
当前未知 named field直接通过，注释假定“caller supplies own schema”，但函数并没有真正接收/验证结构化 schema。

**解决**：production必须要求 `CompositionId + ordered PartId list + common unit + closure rule`；未知字段无 schema时 fail closed。

## NEW-138｜不能从 DataFrame `.name` / 单列 column name 推断 PartId
宽 panel columns是股票代码；字段语义来自 lineage/FieldConcept，不应读 dataframe labels猜。

**解决**：PartId随 IR child semantic metadata传入，runtime data container不承担字段身份推断。

## NEW-139｜composition optional x1..x8 positional API 容易 identity 漂移
某个 optional part缺失后，后续位置整体前移，Aitchison distance配对存在风险。

**解决**：引入 `CompositionVector(parts=[(PartId,panel), ...])` typed input；排序按 schema order，禁止裸位置猜测。

## NEW-140｜`cs_hartigan_dip` 实际输出 `sqrt(N)*dip`，名字不诚实
**解决**：拆：
- `cs_hartigan_dip_raw`
- `cs_hartigan_dip_sqrt_n_scaled`
不要让文献中的 raw dip 与 sample-size scaled score共用名字。

## NEW-141｜Hartigan `min_cross` 是 support policy，不是 alpha parameter
默认 nonsearch；不要允许算法把“20只股票就算/200只股票才算”当 alpha 搜索。

## NEW-142｜group Wasserstein barycenter 仍需强制 SameAxis
`x.to_numpy()` 与 `group.to_numpy()` 前必须 exact axes；central validator外也做 defense-in-depth。

## NEW-143｜`current_members_retrospective` membership vintage 必须进 semantic identity
今天属于A行业的股票，把过去60天历史也塞进A barycenter，与“historical contemporaneous membership”是两个不同因子。

**解决**：canonical或 metadata `membership_vintage`明确，并纳入 definition hash。

## NEW-144｜`cs_universe_coverage` 的 universe mask 不是严格 UniverseBool
任意非零数字/字符串都会被视为 in-universe。

**解决**：UniverseBool只接受 `{0,1,NaN}`；指数权重等不能误作 universe mask。

## NEW-145｜`distribution_break._stack_feats` 没有 SameAxis
三特征 date/columns shift 会被 positional stack。

**解决**：`aligned_pd(f1,f2,f3)` 后再 stack。

## NEW-146｜Energy distance 当前使用 biased V-statistic，且 recent/prior N不同时有样本量偏差
`dxx`/`dyy` 包含 diagonal self-distance并除 `m*m/n*n`。

**解决**：
- 若目标是 empirical energy distance，明确 `biased_v_stat`；
- production更推荐 unbiased U-stat（m,n>1时排 diagonal）；
- 用 Gaussian/同分布 null test检查不同样本数下baseline。

## NEW-147｜joint-energy prior MAD→STD fallback 会在运行中切换估计器定义
某个 feature MAD=0就改std，导致同一 canonical随样本形态切 estimator。

**解决**：明确 robust-scale fallback policy并纳入 definition；或者 MAD=0直接该维不可辨识 -> NaN。

## NEW-148｜`ts_copula_central_asymmetry.grid` 是 estimator resolution
`grid` 不应默认search；`G=max(4,int(grid))` 禁止静默 clamp。

## NEW-149｜group spectrum 所有 f1/f2/f3/group 缺 SameAxis defense
`np.stack`前 exact axes。

## NEW-150｜group spectrum 未统一 missing GroupKey
`pd.unique(g_row)` + `g_row==label` 会让 `""`、可能的 inf/脏编码成为真实组。

**解决**：统一 `is_missing_group_key`；GroupId类型不接受 Inf/空串。

## NEW-151｜group spectrum 实际是 stateful operator，但没有把 breadth history 注册为执行状态
`breadth_history[label] = deque(60)` 使结果依赖此前60天 valid-member count；chunked evaluation从chunk起点重置后会变结果。

**解决**：二选一：
1. 声明 recursive/checkpoint state，checkpoint存每个 GroupId的60天breadth；
2. 更简单：不要内部递归state，直接从显式 trailing 60-day breadth panel计算，使operator成为有界 stateless rolling。
推荐2。

## NEW-152｜group spectrum breadth history 只按 label，没绑定 GroupTaxonomyId/vintage
行业分类体系变更、同一个数字代码复用，会把不同taxonomy历史混在一起。

**解决**：GroupKey = `(TaxonomyId, TaxonomyVersion, GroupId)`。

## NEW-153｜group spectrum 输出是 group-level state，不应默认充当个股 terminal alpha
每组所有股票同值；这更适合 regime/gate/interaction。

**解决**：`role=GROUP_STATE`；默认 grammar不直接终止为个股因子，除非策略明确允许行业轮动因子。

## NEW-154｜RQA `dim*delay<=4` 不是正确的 embedding feasibility
真正 embedding span是 `(dim-1)*delay`，还要留至少M个相空间点。

**解决**：RelationalSpec：
```text
M = effective_len - (dim-1)*delay
M >= min_embedding_points
```
不要用任意 `dim*delay<=4` 代替。

## NEW-155｜`ts_recurrence_rate` 本身也受有效 M 影响
目前line entropy/trapping/divergence有80% coverage gate，但 recurrence rate没有；不同 M 的 RR sampling variance/edge denominator仍不同。

**解决**：生产统一固定/full-window或至少80% contiguous coverage；提供 effective_M diagnostic。

## NEW-156｜RQA 所有 int仍通过 `max(..., int(...))`
补 ParamSpecs：window/dim/delay/min_line/min_periods；`min_periods` support policy nonsearch。

## NEW-157｜`return_decomp.price_basis` 不应是 factor DSL 的普通 scalar
当前宽panel股票代码无法从列名解析price basis，于是依赖调用者传一个字符串。

**解决**：PriceBasis属于 child lineage semantic type；compiler验证所有 price children basis一致。runtime scalar删除。

## NEW-158｜return decomposition 的 `available_at` 不同但未精细声明
- overnight_return：开盘后可知；
- open_to_vwap若全日VWAP：close后；
- open_close_return：close后；
- vwap_to_close_return：close后。

**解决**：每个 canonical设置 machine-readable `available_at` / same_session_usable；不要统一 daily PIT-safe就结束。

## NEW-159｜注释说 `price_basis` keyword-only，但代码/metadata并没有真正 keyword-only ABI
**解决**：从参数移到 ExecutionContext最好；若保留，函数签名用 `*, price_basis=...`，registration audit检查 keyword-only。

## NEW-160｜turnover-survival：注释说负换手率 invalid，代码却直接 clamp 为0
当前：
```python
u = np.maximum(turns, 0.0)
```
负换手率被解释成“零换手”。

**解决**：窗口内任何 finite turnover<0 -> NaN/数据质量错误；不要修数据。

## NEW-161｜turnover-survival price 与 turnover 未 SameAxis
在 `.to_numpy()`前 strict align。

## NEW-162｜turnover-survival 历史 price 只检查 finite，未要求 >0
负价格/0价格可进入 acquisition cost权重。

**解决**：所有参与 chip cost 的 historical price `>0`；否则当前窗口 fail closed或该lag zero-weight策略必须明确。建议 fail closed。

## NEW-163｜Poisson replacement hazard 是模型假设，不是“真实筹码”
`surv=exp(-u)`是一个 survival model。canonical/definition必须写明 `poisson_hazard`，避免用户把输出当真实持仓成本。

## NEW-164｜`_MAX_OLD_MASS=0.3`、`_COST_LOG_EDGES` 是隐藏估计器政策
- 30% cutoff决定大量数据是否NaN；
- cost entropy/mode高度依赖固定bins。

**解决**：版本化；role=MISSING_POLICY/ESTIMATOR_RESOLUTION；默认不search；catalog导出实际常量。

## NEW-165｜weighted chip quantile 的线性插值会制造从未观察到的 acquisition price
如果目标是 empirical cost quantile，应使用 inverse weighted ECDF step quantile；若坚持 interpolation，canonical明确 `_interpolated_quantile`。

## NEW-166｜`weighted_tail.py` 多输入无 SameAxis
`target/sorter`、`x/weight` 都直接 `.to_numpy()`进入 `map_pair_rolling`。

**解决**：全族 `_align()`。

## NEW-167｜weighted-tail 对非有限权重的处理与 weighted-moment 不一致
`aligned_pairs`把 Inf/NaN weight直接丢掉；weighted-moment则非有限weight使整窗口fail。

**解决**：全库统一 `NonNegativeFiniteWeight`：
- NaN weight是 unknown，默认 fail window；
- 若允许 pairwise missing，必须明确 separate MissingWeightPolicy；
- Inf永远非法。

## NEW-168｜`ts_stratified_mean_spread` cutoff ties 依赖 row order
sorter大量相同值时 `argsort`后前/后k硬切，哪个日期样本进入top/bottom由原始顺序决定。

**解决**：fractional tie participation 或 threshold tie-inclusive + weight normalization。

## NEW-169｜stratified `k=round(q*N)` 使实际tail fraction随N跳变
**解决**：输出/记录 effective tail mass；或采用 fractional quantile cut使总权重精确 q*N。

## NEW-170｜`ts_weighted_semivariance` 名字与公式不一致
代码对 squared downside取均值后再 `sqrt`，这是 **downside deviation / semideviation**，不是 semivariance。

**解决**：重命名 `ts_weighted_downside_deviation`；若要semivariance，去sqrt并输出 unit(x)^2。

## NEW-171｜weighted-tail 三个单位声明错误
- stratified mean spread → `same_as:target`；
- weighted downside deviation → `same_as:x`；
- weighted expected shortfall → `same_as:x`。
不是 generic level/ratio。

## NEW-172｜weighted downside `target` 必须与 x 同单位
field-agnostic `target=0` 对return合理，对price/fundamental未必。

**解决**：input semantic constraint或使用 standardized x；scalar threshold unit = same_as:x。

## NEW-173｜weighted-tail 缺 N_eff gate
raw count足够但99%权重集中在1个样本时，风险估计实际只有1个有效样本。

**解决**：`N_eff=(Σw)^2/Σw²`，ES tail还要 tail_N_eff。

## NEW-174｜`ts_cpt_value` unit 标错
输入 return无量纲，CPT value在固定 power utility下也是“return^alpha”型无量纲 score，不能叫 generic level。

**解决**：`dimensionless_behavioral_score`；注意 alpha≠1时它也不是普通return，可不要写 same_as:return。

## NEW-175｜CPT hidden `min_periods=max(5,w//10)` 不在 contract
support policy需machine-readable并默认nonsearch；同时5/60样本与60/60样本的empirical distribution不是同一精度。

## NEW-176｜BVC `_rolling_scale` 在分钟gap后会压缩有限return继续估计scale
这会把隔着缺失minute的return样本当连续rolling cohort。

**解决**：SessionGrid reindex；scale使用 trailing contiguous finite run或明确 finite-observation estimator + coverage gate。

## NEW-177｜BVC 用 `r/(scale+EPS)` 会把零scale变成极大 z
常数价格区间scale=0时，分类应该 undefined/unclassified，不是靠1e-12得到巨大z。

**解决**：`scale <= tolerance -> unclassified`。

## NEW-178｜BVC `locked` 仍是 finite-nonzero truthiness
严格 LockedStateBool `{0,1,NaN}`；Inf/-1/2报错。

## NEW-179｜`intraday_bvc_imbalance` whole-panel 扫负volume会形成数据依赖的“未来异常影响过去调用”
对整张未来面板 `np.any(volume<0)` 后 raise，意味着同一prefix在长数据集与短数据集上程序成功/失败不同。

**解决**：数据源 ingestion先全局DQ；operator内部只对当前 session/window fail closed，不扫描future rows决定过去。

## NEW-180｜flow-impact close/volume/locked 未 exact SameAxis + TradeDate仍用 normalize
统一 SessionContext；禁止靠 close index去切 positions再拿另一panel同位置。

## NEW-181｜BVC缺 classified coverage gate
只有很少bar形成scale也可能输出日度imbalance。

**解决**：输出 `classified_volume_share` / `classified_bar_share`；production min coverage，例如>=70%/80%，具体由测试决定并版本化。

## NEW-182｜`micro_bvc_vpin` 若每天重新切 equal-volume buckets，不等于经典跨日 volume-clock VPIN
**解决**：诚实命名 `intraday_session_bvc_vpin_proxy`，或另实现连续 volume-bucket VPIN；不要让用户误解。

## NEW-183｜impact beta输出 unit 应是 `unit(return)/unit(flow)`
若 flow 是 shares、amount、normalized imbalance，lambda量纲完全不同；必须由 flow semantic type传播。

## NEW-184｜quantile dynamics `fixed_threshold=True` 并没有跨日期真正“冻结”threshold
每个 rolling output仍对当前 chunk的 `chunk[:-1]` 重新算阈值；过去hit标签随着rolling window变动仍可能重新分类。

**解决**：明确两种定义：
1. `window_reestimated_prior_threshold`：每个t用t前历史重估；
2. `calibration_frozen_threshold`：在明确 calibration date/state估一次后持久化。
当前 `fixed_threshold` 名称误导，需重命名。

## NEW-185｜默认 current-inclusive quantile threshold存在 self-inclusion
当前极端值会抬/压自己的阈值，削弱extreme signal。

**解决**：生产主版本优先 strictly-prior threshold；inclusive版本保留但明确 identity。

## NEW-186｜quantile crossing spectral：coverage gate之后提取 trailing contiguous run，却没再次检查run长度
窗口总共60个finite，但最新gap后只剩2个连续点，也可能做2点FFT。

**解决**：对最终 `h` 再要求 `len(h)>=min_contiguous` 且 `len(h)/window>=coverage_floor`。

## NEW-187｜quantilogram lag pair support不应写成 `ok.sum() >= lag+2`
lag决定可行pair数，但估计所需minimum pair count应独立参数/政策，例如>=20；不要把lag本身当统计可靠度阈值。

## NEW-188｜legacy microstructure 多个方法接受 `window/min_periods`，metadata却没有这些param_names
典型如 `micro_realized_vol` metadata只有close，而kernel接受window/min_periods。

**解决**：全库 `inspect.signature(kernel)` vs metadata param_names strict equality审计；隐藏kwargs一律禁止。

## NEW-189｜`micro_mid_return` 不是 bid-ask mid return
`(high+low)/2` 是 HL2/price-range midpoint，不是 microstructure midquote。

**解决**：重命名 `hl2_return` / `range_midpoint_return`；真正 midquote必须用 bid/ask。

## NEW-190｜`micro_amihud_hf` 使用 close*volume 只能叫 dollar-volume proxy
如果数据有 `amount`，标准Amihud更应使用实际成交额；close*volume是近似。

**解决**：`micro_amihud_close_volume_proxy`；另建 amount版标准定义。

## NEW-191｜`micro_trade_imbalance` 不是 trade imbalance
`sign(return)*volume` 是 tick-rule-like signed-volume proxy，不是真实buy/sell trade flow。

**解决**：重命名 `return_sign_volume_imbalance_proxy`。

## NEW-192｜`micro_jump_indicator` 实际返回 jump variation magnitude，不是 indicator bool
`max(RV-BV,0)`是非负幅度。

**解决**：重命名 `micro_jump_variation_proxy`；如果要indicator，再用统计阈值生成Bool。

## NEW-193｜`microstructure/session.py` 用 `index.normalize()` 作为 session authority
UTC index、夜盘、其他市场均可能分错TradeDate。

**解决**：彻底替换为 SessionCalendar/TradeDateMapper；helper不再自己推断。

## NEW-194｜非 DatetimeIndex 时 session helper把整段数据当成一个session
这会让跨多天数据rolling/pct_change污染。

**解决**：没有显式 SessionKey就 fail closed；绝不返回常量0 session。

## NEW-195｜`pct_change_by_session` 未显式 `fill_method=None`
不同 pandas版本/默认可能 forward-fill missing price，制造0收益。

**解决**：所有 pct_change显式 `fill_method=None`。

## NEW-196｜`rolling_cov_by_session` x/y不做SameAxis
必须 exact index；否则 `.loc`可能 silently align/NaN。

## NEW-197｜`session_cum_vwap` 会 `pd.Series(volume,index=p.index)` 隐式reindex
**解决**：先 strict axis；不允许 silent repair。

## NEW-198｜session cumulative VWAP 对 missing bar会在后续继续cumsum，等价“跨未知分钟恢复”
**解决**：定义 gap policy：生产建议 unknown bar后 current cumulative state NaN，直到新session；或只有明确volume=0才允许连续。

## NEW-199｜`price_volume.cumulative_returns` 缺 PositivePrice
anchor/current <=0 不可计算price return。

## NEW-200｜`log_returns` 只替换 prior=0，负price可能产生正ratio后被log成合法值
**解决**：current和previous都必须 `>0`。

## NEW-201｜`max_drawdown(returns)` 未拒绝 simple return <= -1
`1+r <=0` 后wealth path失去金融意义。

**解决**：ReturnSimple domain `r>-1`（-1为归零吸收态，之后需要特殊处理/NaN），不能让<-100%进入。

## NEW-202｜max_drawdown gap后虽重新anchor，但最后用全历史 `expanding().min()` 又把gap前worst带到gap后
这是“segment drawdown”与“full-history worst”混合定义。

**解决**：
- `segment_max_drawdown`：gap reset并重置historical worst；
- `full_history_max_drawdown`：gap后保持unknown直到重新建立完整连续历史，或明确carry historical record。
拆 canonical。

## NEW-203｜Sharpe/volatility硬编码 `sqrt(252)`
不是所有grain/market/calendar一年252 observation。

**解决**：annualization来自 SamplingCalendar/Grain；或输出 unannualized rolling mean/std ratio，再用单独 annualize operator。

## NEW-204｜`open_gap`/`close_gap` 与 return_decomp 重复，但契约更弱
缺SameAxis、PriceBasis、PositivePrice、availability。

**解决**：去重 alias到严格 return-decomp canonical，不要保留弱实现。

## NEW-205｜VWAP whole-panel negative-volume scan + `notna`而非finite
- future bad volume可使整个call raise；
- Inf volume/price可能被视为valid；
- axes未严对齐；
- min_periods默认1。

**解决**：per-window finite+nonnegative validation、SameAxis、coverage/N_eff、合理min support。

## NEW-206｜benchmark单列广播不检查时间轴
`align_benchmark_to_ret` 单列benchmark会concat并换columns，但没有先要求index==ret.index。

**解决**：BroadcastSpec只允许 **same time axis + single benchmark factor column**；date shift直接报错。

## NEW-207｜多列benchmark会 silent `reindex(columns=ret.columns)`
缺股票/多股票会被隐藏成NaN。

**解决**：多列exact columns；若允许subset broadcast，必须显式 join policy，不得偷偷reindex。

## NEW-208｜`capm_helpers.apply_capm_kernel_panel(min_obs=5)` 参数完全未使用
这是典型 dead parameter / 假安全门。

**解决**：要么删除；要么传入kernel并真正约束每个窗口 aligned pairs；加 ParamInjectivity test。

## NEW-209｜cross-sectional `c_neutralize` group缺失时自动退化为global demean
这会把“行业中性化”静默变成“市场去均值”。

**解决**：生产默认 group missing -> NaN/fail；global demean用独立 `cs_demean`。

## NEW-210｜group family大量 `group.reindex(index=x.index, columns=x.columns)`
这会隐藏错位的行业panel。

**解决**：生产exact SameAxis；真正broadcast/AsOf join在数据绑定层完成，不在算子内部修。

## NEW-211｜group operator 的 `fallback_policy` 经常不在 metadata param_names
kernel接收隐藏keyword但契约没声明，造成backend/validator不一致。

**解决**：所有实际kernel参数必须metadata完整；fallback policy nonsearch enum。

## NEW-212｜`group_ts_decay_linear(normalize=True/False)` 在同一 canonical 中切换定义
False输出same_as:x；True输出dimensionless zscore，单位改变。

**解决**：拆 `group_ts_decay_linear_value` 与 `group_ts_decay_linear_zscore`。

## NEW-213｜group aggregate / rank-weighted 对 missing group key 仍有多套规则
空字符串、Inf、pd.NA处理不统一。

**解决**：所有 group family唯一 `GroupKeyValidator`。

## NEW-214｜robust_stats 的 `ts_quantile_range` / `ts_trimmed_mean` unit写成 ratio
- quantile range → same_as:x；
- trimmed mean → same_as:x。

## NEW-215｜`ts_robust_zscore` 没有最小统计support
只要一个finite样本，median可算；MAD=0后NaN，但2-3点也会产生极不稳定z。

**解决**：production最少例如8/10，具体通过稳健性测试决定；`min_periods`显式support policy。

## NEW-216｜`ts_robust_zscore` center/scale组合会产生四种不同估计器
median+MAD、median+std、mean+MAD、mean+std不是“一个参数小变化”，是不同定义。

**解决**：center/scale非默认search，或拆少数明确 canonical；避免AlphaProbe把 estimator choice当alpha经济维度。

## NEW-217｜direction ratio threshold是 absolute scalar，却domain写price_volume泛化
`x>0`对return有意义，对price几乎恒真。

**解决**：限制输入 SemanticType为Return/Change/StandardizedSignal，或 threshold same_as:x并由typed recipe绑定。

## NEW-218｜`ts_abs_entropy(normalize)` 同一 canonical输出两种单位
normalize=True dimensionless；False=nats。

**解决**：拆 raw/normalized canonical，`normalize`从搜索面删除。

## NEW-219｜`event_decay_asof(event_kind=bool/marked)` 同一 canonical也会切换单位
bool输出decayed count；marked输出same_as mark。

**解决**：拆 `event_bool_decay_count` 与 `marked_event_decay_sum`。

## NEW-220｜expectation_v2 仍从 `transforms_v2 import _pos_int`，受前述 monkey-patch import-binding问题影响
这是 NEW-006 的具体受害模块之一。

**解决**：统一从 `strict_params` import final function，不从可被后patch的模块import。

## NEW-221｜expectation family 多处 `target_period_id.reindex(...)` 隐藏错位
actual/expected/target_period必须SameAxis；AsOf对齐由provider完成。

## NEW-222｜`fin_expectation_revision` 第一行直接定义“无修订=0”需要明确baseline语义
如果第一行只是数据截断起点，无法知道之前是否有过estimate revision。

**解决**：左删失数据集第一行应NaN，除非source提供“这是该target首次estimate”的Release flag。新增 `left_censor_policy`，production默认NaN。

## NEW-223｜expectation revision speed/count/magnitude `rolling(..., min_periods=1)` support漂移
60-day speed前几天只有1-2天就输出，与完整60日统计不可比。

**解决**：full window或minimum coverage；support policy非search。

## NEW-224｜`fin_surprise` scale_base只 `abs()` + zero guard，缺 near-zero 与 semantic scale定义
EPS/share、revenue、price等scale意义不同。

**解决**：ScaleBase必须typed（price/abs actual/std/consensus mean等）；near-zero scale采用relative floor或NaN，不能generic。

## NEW-225｜valuation ratio分母需要 Positive/NonNegative domain
free_cap/capitalization/circulating_cap不能只 `replace(0,NaN)`；负股本数量是数据错误。

## NEW-226｜`valuation_cashflow_disagreement` 多输入没有显式SameAxis
五个panel stack前必须exact axes。

## NEW-227｜valuation disagreement 1%/99% winsor是隐藏 estimator policy
固定阈值、min breadth=5都进入 definition；breadth=5做1/99分位极不稳定，生产应提高breadth。

## NEW-228｜`valuation_quality_mismatch` WLS minimum 5太弱
2参数回归仅5只股票残差极不稳定；沿用全库 cross-sectional regression breadth标准（>=10/20且N/K ratio）。

## NEW-229｜shareholder ratio没有系统约束合法范围
ShareRatio通常应 `[0,1]`（或provider percent conversion后ratio）；负数、>1必须DQ或明确字段单位。

## NEW-230｜Shareholder top-K 缺失slot与真实0持股必须区分
absence是“不在披露top-K”，不能通用 `nan_to_num=0` 做所有指标。ID-matched新版本较好，但 legacy helper必须永久隔离/删除。

## NEW-231｜`holder_*` top-K K=10属于 source/disclosure policy，必须进入 identity
如果未来provider提供Top20，不能仍叫同一 canonical但统计口径变化。

## NEW-232｜index member/is_suspend等仍需 strict Bool
`member=2/-1/Inf` 不应参与 entry/exit/frequency。

## NEW-233｜`index_reconstitution_churn` unknown transition被写0后rolling sum仍可能伪装“低churn”
虽然不计跨unknown transition，但窗口包含大量unknown时仍输出一个count。

**解决**：同时要求 membership coverage floor；unknown比例高则NaN，或输出 rate/count with exposure。

## NEW-234｜`multi_index_entry_intensity` 三个输入必须 Strict EventBool + SameAxis
当前只是notna+直接相加；2、-1可制造“进入数量”。

## NEW-235｜`index_event_decay` 事件域应严格 `{-1,0,1,NaN}`
否则任意marked数值会改变信号；若想MarkedEvent另建版本。

## NEW-236｜`index_event_decay(missing_policy='carry')` 在unknown日广播旧值，必须标 stale
若允许carry，输出 lineage需 staleness age；默认生产建议break。

## NEW-237｜liquidity_v2 `_check_volume_nonneg` whole-panel扫描同样有future-call行为问题
原则同 NEW-179；operator对prefix的输出/成功状态不应受未来坏值影响。

## NEW-238｜liquidity_v2 多输入普遍缺SameAxis
`ret/close/volume`、OHLC、turnover等算术依赖pandas label auto-align；必须 central SameAxis gate + helper defense。

## NEW-239｜`adv(close,volume)` 用 `close.abs()*volume` 会把负price变成正dollar volume
负price不是合法价格，不该abs修复。

**解决**：PositivePrice；若close missing/invalid当前window fail/paired missing。

## NEW-240｜`bounded_nvi/pvi` 用 `r.clip(lower=-0.999999)` 静默修复非法<-100% return
**解决**：simple return <=-1必须domain处理；不要clip成“几乎归零”。

## NEW-241｜`zero_return_ratio(epsilon=1e-12)` epsilon单位不适合泛化输入
对return可以；对price/fundamental不应拿1e-12当“零”。

**解决**：限制输入Return/Change；epsilon=NUMERICAL fixed，不进入search。

## NEW-242｜Corwin–Schultz / high-low spread必须 OHLC domain
`high>0, low>0, high>=low`；错价不能仅靠replace zero后继续。

## NEW-243｜`data_cleaning.EWM/ewm_mean` 与 `ts_ema` 重复且execution contract可能不同
统一canonical；statefulness/full-history/checkpoint只能有一个权威实现。

## NEW-244｜forward-fill `forward_fill_allowed` 仍通过裸 `bool()`
`"false"` 会变True；必须strict bool parser。

## NEW-245｜`ffill max_ffill_gap=0` 代表 unlimited 是危险的反直觉默认
已在前面指出，进一步要求 API 改为：`None=unlimited`（research only），`0=no fill`，positive int=limited。

## NEW-246｜group rank-weighted value 的 unit应 same_as:x
rank weight dimensionless；不要generic level。

## NEW-247｜group rank-weighted `x*normalized_rank_weight` 不是“组内聚合”而是 position-dependent shrink
应明确是 transform，不是 group statistic；否则搜索器可能与 rank*value 等大量代数重复。

## NEW-248｜`group_decay_linear.window` 是明确 dead parameter compatibility surface
虽然已标nonsearch，最终应从 canonical ABI彻底删除，只保留 parser migration alias，避免历史factor identity继续包含无效window。

## NEW-249｜group TS linear decay 当前missing样本权重由共享 `rolling_linear_weighted` 决定
需跟 NEW-027/028 一起修：最新样本missing不能返回stale加权值；权重应按physical positions还是按valid observations必须明确。

## NEW-250｜cross-section/global/group broadcast统计需要 terminal-role审计
`c_mean/c_std/c_sum/c_count`、Hartigan/global coverage、group spectrum等广播值在同行内可能无个股排序信息。

**解决**：
- GLOBAL_STATE：只能 gate/interaction；
- GROUP_STATE：默认只能与个股signal interaction或显式行业轮动；
- DIAGNOSTIC：不能进入alpha terminal。

---

# 21. 按模块族继续执行的“不可遗漏审计清单”

下面这些不是“可选优化”，而是执行 AI 必须逐模块跑的审计模板。即使本文件未点名某个函数，只要它满足搜索模式，就必须纳入整改。

## 21.1 所有 rolling/time-series operator
逐个检查：
1. `window` 是 physical rows、finite observations、events、reports还是sessions？
2. current row 是否参与？
3. current missing 是否仍输出旧值？
4. gap 是否 break topology？
5. min_periods是否support policy而非搜索维度？
6. `window=20` 是否可能只用2个样本输出？
7. ddof/convention是否明确？
8. unit是否正确？
9. history requirement是否与真实raw rows一致？
10. `int/max/min/clip` 是否偷改参数？

## 21.2 所有 pair/multi-panel operator
必须自动扫描函数体中的：
```text
.to_numpy(
np.stack(
pd.concat(
reindex(
shifted benchmark broadcast
```
任何多panel数学在进入这些操作前都必须 `SameAxis`；允许broadcast必须是显式 BroadcastSpec。

## 21.3 所有 condition/event/member/valid_trade/is_suspend/locked/universe inputs
统一类型：
```text
ConditionBool: {0,1,NaN}
EventBool:     {0,1,NaN}
SignedEvent:   {-1,0,1,NaN}
UniverseBool:  {0,1,NaN}
ApplicabilityBool: {0,1,NaN}
```
严禁 `finite & !=0`。

## 21.4 所有 group inputs
必须有：
- SameAxis；
- GroupKey validity；
- taxonomy id/version；
- membership vintage；
- min group size；
- unknown-group policy；
- singleton group policy；
- broadcast group-state role。

## 21.5 所有 weight inputs
统一 `NonNegativeFiniteWeight`：
- negative finite → invalid；
- Inf → invalid；
- NaN → missing policy明确；
- zero合法；
- weighted statistic要 N_eff；
- numerator/denominator同cohort。

## 21.6 所有 quantile/bin/top-k operator
必须检查：
- ties；
- effective bins；
- cutoff fractional membership；
- minimum breadth；
- estimator resolution ParamRole；
- quantile interpolation是否制造未观察值；
- q bounds；
- tail raw count和effective mass。

## 21.7 所有 regression operator
必须检查：
- in-sample vs prior/OOS；
- intercept；
- predictor standardization；
- collinearity；
- N/K DOF margin；
- weight mask；
- units；
- coefficient identity；
- solver convergence；
- regularization参数是模型定义还是搜索参数。

## 21.8 所有 stateful/recursive operator
必须登记唯一 ExecutionContract：
```text
state_model
chunking
checkpoint_schema
minimum_history
history_kind
state_expiration/max_gap
```
禁止靠中央名字白名单“猜stateful”。

## 21.9 所有 intraday/minute→daily operator
必须：
- SessionCalendar；
- TradeDate mapper；
- timezone；
- timestamp convention；
- bar size；
- official grid；
- duplicate/missing/off-grid bar policy；
- source coverage；
- multi-field bar validity；
- output `available_at=session_close` 或更细时点；
- `input_grain=minute/output_grain=daily` typed GrainTransform。

## 21.10 所有 fundamental operator
必须：
- value-as-of release time；
- report period id；
- revision ledger；
- fiscal ordinal；
- stock vs flow vs cumulative-YTD flow；
- units；
- consecutive-period requirement；
- left censoring；
- expected target-period identity；
- no trading-day shift substitute for fiscal lag。

## 21.11 所有 A-share special-state operator
必须：
- official high_limit/low_limit from provider；
- board/ST/rule vintage from market adapter；
- tick size absolute comparison；
- suspension/tradable strict bool；
- listing/relisting/special-treatment dates PIT；
- current status unknown -> NaN，不猜。

## 21.12 所有 expanding/full-history operator
默认不允许把数据集起点当自然经济起点。
- 要么 `requires_full_history=True` 并保证统一历史起点；
- 要么改为bounded window；
- 要么明确 `since_dataset_start`，从默认 mining排除。

---

# 22. 必须建立的自动化代码扫描（Static Audit）

执行 AI 应增加一个 `tools/audit_operator_source.py` 或等价脚本，对 `cleaned_operators` 全目录运行 AST/文本检查；以下命中不能直接算错误，但必须输出审计报告并人工/规则分类。

## 22.1 参数静默变形扫描
搜索：
```regex
\bint\(
max\([^\n]*int\(
min\([^\n]*int\(
np\.clip\(
\.clip\(
```
对每个命中确认：这是 numerical output clipping，还是非法参数修正？后者禁止。

## 22.2 time-axis compression扫描
搜索：
```regex
dropna\(
\[np\.isfinite\(
valid_values\(
aligned_pairs\(
```
如果后续做 lag/FFT/embedding/path/event interval/recurrence/transition，默认判高风险。

## 22.3 silent reindex扫描
搜索：
```regex
\.reindex\(
pd\.concat\(
set_axis\(
```
多输入数学层出现reindex必须证明这是合法 BroadcastSpec；否则改SameAxis fail closed。

## 22.4 truthiness扫描
搜索：
```regex
!=\s*0
bool\(
astype\(bool\)
\.ne\(0\)
```
凡condition/event/member/state输入全部改 strict semantic validator。

## 22.5 hidden parameter扫描
用 `inspect.signature` 比较：
- public calculate signature；
- kernel signature；
- metadata.param_names；
- metadata.param_specs；
- backend signatures。
任何 kernel可用但metadata没声明的参数 -> fail。

## 22.6 EPS扫描
搜索 `1e-12`, `EPS`, `+ EPS`：
- denominator numerical stabilization可接受但需scale-aware；
- domain undefined（0 variance/0 vol/0 distance）不能通过EPS硬变成有限统计量。

## 22.7 hidden constant扫描
搜索硬编码：
```text
0.05, 0.1, 0.3, 0.5, 0.8, 1.345, 1.4826,
20, 60, 120, 252,
512, 200
```
判断它属于：经济阈值 / horizon / estimator resolution / numerical / support / market policy，并写入 definition contract。

## 22.8 future-dependent whole-panel validation扫描
如果 operator 在计算每个t前先对 **整个 DataFrame** 做 `any/all/min/max` 并据此 raise，需检查未来坏值是否改变过去prefix的程序行为。

生产原则：
- source DQ可以全数据集先验验证；
- factor kernel的prefix causality不仅要求“数值不看未来”，也要求 **控制流/成功失败不由未来数据决定**。

---

# 23. 必须新增的测试套件（不要只测“能跑”）

## T01 参数类型 Fuzz
对每个scalar ParamSpec自动生成：
```text
valid min/default/max
bool
integral float
fractional float
str numeric
str arbitrary
None
NaN
+Inf/-Inf
below min / above max
```
断言：非法值必须拒绝；不能被kernel改成另一参数。

## T02 Parameter Injectivity / Dead Dimension
对所有 `searchable=True` 参数：
- 在设计好的synthetic fixture上取至少3个不同值；
- 如果输出完全一致，参数可能dead；
- dead search dimension CI fail。

允许例外必须标：`compatibility_only/searchable=False`。

## T03 Relational feasibility
自动覆盖：
```text
lag < window
fast < slow
short < medium < long
min_periods <= window
(dim-1)*delay + M_min <= window
q_low < q_high
k <= breadth/effective support
```
search grammar生成前就剪掉。

## T04 Default Viability
每个 production candidate默认参数在至少一个合理synthetic fixture上：
- 有finite输出；
- 不是全常数；
- 不是全0；
- coverage合理。

## T05 Whole-grid Dead Region
随机/网格采100–500个合法参数组合，统计：
```text
all_nan_rate
constant_rate
low_coverage_rate
identical_cluster_rate
```
超过阈值说明搜索空间定义有问题。

## T06 SameAxis Metamorphic
对每个2+ panel operator：
1. 正常对齐；
2. columns permute；
3. 日期shift 1 day；
4. 少一只股票；
5. 多一只股票；
6. duplicate timestamp；
7. reverse index。
除明确BroadcastSpec外，2–7都必须fail closed；仅columns同集合不同顺序也不应偷偷按位置配。

## T07 Current-Missing
对 CurrentRequired operator：把仅 `x[t]` 改NaN，`output[t]`必须NaN；不能与原输出相同。

## T08 Gap Topology
在：
- t-1；
- t-5；
- window middle；
插NaN。对lag/embedding/FFT/path/episode/recurrence/event interval类断言不重连。

## T09 Recursive Rewarm
KAMA/Supertrend/PSAR/Kalman/Wilder/EMA-if-required：
- 连续数据 baseline；
- 中间插gap；
- gap后检查state reset/expire + warmup长度。

## T10 Strict Bool Fuzz
输入：
```text
-Inf,-2,-1,-0.0,0,0.2,1,2,Inf,NaN,True,False
```
只有0/1/NaN合法（SignedEvent另按{-1,0,1,NaN}）。

## T11 Unit Algebra
自动生成相同数学但改变输入scale：
- x→10x；
- y→100y；
检查 covariance/beta/resid/range/tstat/entropy/lag 的尺度律。

## T12 Synthetic Golden Set
至少：
- constant；
- linear trend；
- sine；
- white noise；
- AR(1)；
- random walk；
- GARCH；
- two-state Markov；
- change point；
- motif repeat；
- one outlier；
- all ties；
- limit-up sequence；
- suspension gap。

## T13 Prefix Causality（数值 + 控制流）
计算完整T；随机改 `t+1:`；断言：
- `output[:t+1]`完全不变；
- 不应从“成功”变“raise”；
- metadata/certification不因future values改变。

## T14 Column Permutation invariance
对不依赖stock identity order的算子，随机permute columns再逆permute输出，应完全相同。
TopK ties、isotonic、KNN、group、rank重点测。

## T15 Tie Metamorphic
构造：
- 50只股票相同return；
- 涨停0 return ties；
- 相同财务ratio；
- KNN等距；
- TopK cutoff大tie。
输出不能由ticker列顺序决定。

## T16 Scale/Shift Metamorphic
根据operator contract测试：
```text
x -> 10x
x -> x+100
x -> -x
```
例如 corr不变；cov*100；zscore不变；price log-return scale不变；PositivePrice operator对-x应fail。

## T17 Effective-N Audit
每个统计输出必须同时可获得或内部记录：
- raw window rows；
- finite N；
- aligned pair N；
- weighted N_eff；
- group breadth；
- event count；
- tail count；
- embedding M；
- classified session coverage。

## T18 Null Calibration
MI/TE/HSIC/BDS/RQA/HVG/Granger/bicoherence/copula/tail/energy/topology：
至少1000组模拟null（离线golden artifact可以），保存：mean/std/quantiles/sample-size curve。
生产输出若严重偏离理论null，先research only。

## T19 Semantic Duplicate Graph
对所有 canonical用synthetic battery生成输出fingerprint，检测：
- exact equal；
- affine equal；
- monotonic equal；
- rank equal；
- >0.999 correlation across broad fixtures。
然后人工决定 alias/delete/keep。**参数变化不能被当成“不同因子”。**

## T20 Practical A-share Coverage
真实A股约500只×5年（或你们标准样本）统计：
```text
coverage
cross-sectional std
unique values
Inf count
longest NaN streak
all-zero dates
all-same dates
```
按market board/ST/suspension/new listing分层。

## T21 A-share SessionGrid
fixture：
- 完整240；
- 少开盘bar；
- 少收盘bar；
- 午休不存在bar；
- 午休多非法bar；
- duplicate minute；
- UTC timestamp；
- 239/241；
- suspension full-day；
- volume=0但bar存在；
- bar absent vs bar value NaN。

## T22 Support-count
event/tail/state/neighbor/model estimate不得用1–2个样本伪装稳定统计。CI扫描 production candidate 的 minimum support。

## T23 Runtime Final-State Audit
必须在 `load_all()` **所有 overlay完成后、freeze前** 导出最终状态，然后重新跑：
```text
aliases
status
surface
role
ParamSpec
units
history
backend
certification
implementation source/hash
semantic version
```
源文件正确但final registry错误，同样CI fail。

## T24 Chunk Equivalence
对所有stateless bounded rolling：full run vs chunked with declared warmup应一致。
对stateful checkpoint：full run vs checkpoint resume一致。
对full-history：chunk execution必须禁止或强制replay。

## T25 Grain Transform
minute→daily：
- pandas/reference输出index必须TradeDate daily；
- 不允许generic shape-preserving rebuild；
- planner能识别node改变grain；
- 下游daily operator可安全消费。

---

# 24. 需要新增的统一语义类型 / 运行时对象

执行 AI 不要继续在每个模块复制validator，建议最少建立：

```python
ConditionBool
EventBool
SignedEvent
UniverseBool
ApplicabilityBool
TradableBool
NonNegativeFiniteWeight
PositivePrice
ReturnSimple
ReturnLog
GroupId(taxonomy_id, version, value)
PriceBasis
FiscalPeriodId
ReportFlowType
CompositionSchema
SessionContext
MarketRuleContext
BroadcastSpec
HistoryRequirementFactory
```

## 24.1 SessionContext
最少字段：
```text
market
exchange
timezone
trade_date_mapper
timestamp_convention
bar_size
segments
expected_grid(trade_date)
market_rule_version
```

## 24.2 Operator semantic contract
每个 final canonical 最少必须导出：
```text
canonical
definition_id
semantic_version
input_semantic_types
panel_params
scalar_params
ParamSpec + ParamRole
relational_constraints
output_unit
window_semantics
current_included/current_required
missing_policy
min_effective_support
input_grain/output_grain
available_at
role
broadcast_specs
history_requirement_factory
state_model/chunking/checkpoint_schema
market/session policy
implementation hash by backend
production certificates
```

---

# 25. 推荐的机器可读 Certification Table

最终生成 CSV/Parquet/JSON，字段至少：

```text
canonical
definition_id
semantic_version
surface
role
input_semantic_types
input_units
output_unit
panel_params
scalar_params
param_roles
searchable_params
relational_constraints
window_semantics
current_included
current_required
missing_policy
effective_n_rule
min_effective_support
input_grain
output_grain
available_at
same_session_usable
membership_vintage
group_taxonomy_requirement
session_policy
market_policy
history_kind
history_expression
state_model
chunking
checkpoint_schema
golden_test_pass
null_test_pass
axis_test_pass
tie_test_pass
unit_test_pass
future_randomization_pass
chunk_equivalence_pass
backend_parity_pass
duplicate_cluster_id
production_eligible
```

默认 AlphaProbe / AlphaMiner grammar **只加载 `production_eligible=True` 且 role允许进入对应位置的 canonical**。

---

# 26. 修复优先级与执行顺序

## P0-A：先修共享正确性内核
1. `_rolling_fast` argmax/argmin、time_slope、regression cohort；
2. SameAxis universal gate；
3. strict Param validator；
4. strict Bool/Weight/PositivePrice；
5. SessionContext；
6. history requirement；
7. fundamental strict params monkey-patch；
8. final-registry logical-contract freeze。

原因：这些会同时污染几十/几百个上层算子，先修收益最大。

## P0-B：再修高频基础算子
```text
ts_mean/std/corr/cov/beta/regression
rank/zscore/group_neutralize
return/log_return/volatility/vwap
open/close/overnight decomposition
A-share limit/suspension
fundamental period transforms
```

## P0-C：再修 minute→daily
统一SessionGrid后再碰：
```text
intraday_agg
realized beta/vol
BVC/VPIN
impact/recovery
activity clock/volume clock
minute profiles
```
否则每个模块单独补session都是重复劳动。

## P1：高级统计
```text
RQA/HVG/topology
MI/TE/HSIC/Granger
EVT/multifractal
expectile/quantile regression
energy/copula
Markov/Kramers-Moyal
```
全部在 synthetic/null golden通过后才promotion。

## P2：搜索空间质量
1. semantic duplicate graph；
2. ParamRole清理；
3. estimator resolution退出search；
4. aliases清理；
5. group/global/diagnostic role约束；
6. practical A-share coverage。

---

# 27. 执行 AI 的工作方式（必须遵守）

1. **不要只改注释/metadata掩盖数学问题。**
2. **不要用 `np.clip/max/min/int` 把非法输入修成合法输入。**
3. **不要为了“测试通过”降低最小样本或把NaN填0。**
4. **不要用epsilon让本来undefined的统计量变成有限值。**
5. **不要新增参数来解决每个问题**；很多参数应固定/versioned，而不是扩大搜索面。
6. **不要为了后端覆盖把pandas delegate叫native Polars。**
7. **不要把research-only当垃圾桶**；能修的修，没经济信息的删，diagnostic明确角色。
8. **不要按名字去重**；必须数学fingerprint/golden equivalence。
9. **不要把分钟→日频强塞进shape-preserving框架。**
10. **不要用observed data推断官方market/session规则。**
11. 每修一族同时补对应 test；禁止最后才补测试。
12. 每完成一轮必须重新 `load_all -> final state audit -> freeze` 验证 overlay未回滚修复。

---

# 28. 最终 Definition of Done（全部满足才算 FactorEngine 这一轮完成）

- [ ] 所有 canonical metadata.param_names 与真实 public kernel signature一致；无hidden kwargs。
- [ ] 所有 scalar 参数有完整 ParamSpec；无 runtime silent truncation/clamp。
- [ ] 所有参数有 ParamRole；default mining只搜 ECONOMIC/HORIZON/STATE_THRESHOLD/MODEL_ORDER。
- [ ] 所有 relational constraints compile-time prune。
- [ ] 所有2+ panel数学都有 SameAxis或显式BroadcastSpec。
- [ ] 所有Condition/Event/Universe/Tradable严格typed；Inf不再当True/False。
- [ ] 所有weight统一NonNegativeFiniteWeight + N_eff规则。
- [ ] 所有price-derived factor有PositivePrice/PriceBasis。
- [ ] 所有CurrentRequired current-NaN test通过。
- [ ] 所有lag/path/embedding/FFT/event interval不跨NaN重连。
- [ ] 所有recursive operator execution/history/chunking唯一声明。
- [ ] full run/chunk/checkpoint equivalence通过。
- [ ] A股minute official SessionGrid测试全通过。
- [ ] minute→daily GrainTransform可被planner/runtime正确执行。
- [ ] fundamental value-as-of/revision/fiscal-period tests通过。
- [ ] A-share limit/suspension rule-vintage tests通过。
- [ ] unit algebra audit无已知错误。
- [ ] no dead searchable dimensions。
- [ ] no estimator-resolution parameters in default mining。
- [ ] alias equivalence均有golden proof。
- [ ] global/group/diagnostic operator不能错误作为个股terminal alpha。
- [ ] advanced statistics synthetic/null goldens通过。
- [ ] semantic duplicate graph已清理 exact/affine/monotonic/rank duplicates。
- [ ] prefix future randomization数值与控制流都通过。
- [ ] final frozen registry `UNCLASSIFIED == 0`。
- [ ] final registry logical contract hash与所有backend一致。
- [ ] production_eligible只由完整certification决定，不按名称/手工promotion白名单决定。
- [ ] A股真实样本 practical coverage/unique/std/NaN-streak报告通过。

---

# 29. 交付物要求

执行 AI 最终必须输出/生成：

1. **Operator Fix Report**
   - 每条本文件编号：`fixed / not-applicable / removed / pending-research`；
   - 修改文件；
   - 对应测试；
   - 不允许只写“已优化”。

2. **Final Operator Certification Table**
   - 使用第25节字段。

3. **Semantic Duplicate Report**
   - exact duplicates；
   - near duplicates；
   - aliases；
   - removed canonicals；
   - 保留理由。

4. **Parameter Surface Report**
   - searchable params；
   - ParamRole；
   - dead-region比例；
   - relational pruning统计。

5. **A-share Practical Coverage Report**
   - 每canonical覆盖率、横截面方差、unique、Inf、NaN streak；
   - 分板块/ST/停牌/新股。

6. **Session/Grain Report**
   - minute operator的SessionCalendar、bar size、availability、input/output grain。

7. **Golden/Null Test Report**
   - standard stats参考实现；
   - research stats null calibration。

8. **Final Runtime State Dump**
   - 必须是所有 overlay 后 freeze 前/后的最终 registry，不是源码静态表。

---

# 30. 最后给执行 AI 的一句总要求

**不要把任务理解成“修掉几个会报错的算子”。真正目标是：让 FactorEngine 的每一个可进入自动因子挖掘搜索空间的 operator，都同时满足数学定义正确、参数语义真实、PIT/时间拓扑正确、A股市场规则正确、输入类型与单位正确、history/chunking正确、最终 registry 不被 overlay 污染，并且经过可重复的 machine certification。任何一个条件未满足，该 canonical 就不得进入默认 AlphaProbe/AlphaMiner production grammar。**


---

# 31. 最后一轮共享 helper 反查新增问题（NEW-251 起）

## NEW-251｜`stateful._common.panel_or_scalar` 对 threshold panel 按 iloc 位置读取，却不验证与主 x SameAxis
**位置**：`cleaned_operators/stateful/_common.py`

如果 state operator 传入 ATR/volatility threshold panel，而 threshold panel 日期或股票列发生shift，`panel_or_scalar(value,row,col)` 会直接读取错误cell。

**解决**：
- operator入口把所有 panel-valued scalar/threshold 当真正 panel input纳入SameAxis；
- `panel_or_scalar`不负责猜轴；
- metadata需要 `broadcast_specs`：scalar可broadcast，panel必须SameAxis。

**测试**：threshold columns permute/date shift必须fail；不能“算出另一个股票的ATR阈值”。

## NEW-252｜`panel_or_scalar(pd.Series)` 的语义也不明确
Series被按 `col` 读取，默认解释成“跨股票静态向量”，但Series也可能是“随时间的一维序列”。

**解决**：禁止模糊Series；使用显式：
```text
ScalarConstant
CrossSectionScalarByInstrument
TimeScalarByDate
PanelScalar
```
或统一先broadcast成严格panel。

## NEW-253｜`assert_condition_bool` 全局共享版本仍让 ±Inf漏过validator
它只检查“finite且不在0/1”；Inf不是finite，因此不报错。后续不同caller有的把Inf当missing，有的notna后当True。

**解决**：ConditionBool合法判定应是：
```python
is_missing = is_nan/null
is_valid = value in {0,1}
anything_else including +/-Inf => error
```
不能用 `np.isfinite` 作为“需要验证的值”的前提。

## NEW-254｜`finite_panel()` 名称与实现相反
当前 helper 只是 `return panel.to_numpy()`，并不返回finite mask。容易被未来调用者误用。

**解决**：改名 `panel_values()`；真正 `finite_mask()` 返回 `np.isfinite`。避免语义陷阱。

## NEW-255｜`common.cs_broadcast` 的 valid/count 仍用 `notna/count`，把 ±Inf 当有效截面股票
`cs_rank_01`、row count、singleton判断可能把Inf视为正常极值。

**解决**：生产numeric factor统一 FiniteValue；`np.isfinite`决定统计样本。若某operator有意允许Inf，必须独立SemanticType且默认禁止。

## NEW-256｜`broadcast_row_stat(x, values)` 会把 values按x.index重新构造，可能静默修复错位row statistic
内部正常调用问题不大，但作为shared helper应该fail-safe。

**解决**：如果values已有index，先要求 `.index.equals(x.index)`；不要通过 `pd.Series(values,index=x.index)`覆盖来源index。

## NEW-257｜`robust_stats._rolling_apply_2d(..., min_periods=...)` 参数本身没有被helper使用
前两个caller在fn里自己检查，`ts_robust_zscore`则没有support参数；这是API假象。

**解决**：要么helper统一执行 finite-count gate，要么删除无效 `min_periods` 参数；不允许“接口写了support但实际各caller随意”。

## NEW-258｜`ParamRole` 当前枚举仍缺 `MODEL_ORDER / MISSING_POLICY / MARKET_POLICY`
当前base只有 ECONOMIC/HORIZON/STATE_THRESHOLD/ESTIMATOR_RESOLUTION/NUMERICAL/POLICY，导致：
- AR order/embedding dim与普通economic参数混在一起；
- missing policy、market policy全塞进泛化POLICY；
- search/audit无法精细区分。

**解决**：扩展角色，并兼容迁移旧POLICY：
```text
MODEL_ORDER
MISSING_POLICY
MARKET_POLICY
```
默认search：MODEL_ORDER可coarse/full reviewed grid；MISSING/MARKET永不search。

## NEW-259｜`effective_param_role(spec=None)` 仍 fail-open 到 ECONOMIC
任何没写ParamSpec的scalar都会被视为可搜索经济参数，是当前搜索空间污染的重要根源。

**解决**：生产surface **spec=None => excluded + certification fail**。只有legacy/research兼容层可fallback。

## NEW-260｜`_INTEGER_PARAM_NAMES` 仍是第二套参数类型权威
即使注释说fallback，它仍可能把同名但实际float语义参数强制当int。

**解决**：完成迁移后生产路径彻底移除name whitelist；CI要求 production/extended candidate `scalar_params ⊆ param_specs.keys()`。

---

# 32. 本次审计实际覆盖的当前代码区域（供执行 AI 继续同类扫描）

本任务书不是根据“理论想象”写的；本轮及前序轮次已直接检查过当前 `cleaned_operators` 的大量具体实现/共享层，至少包括：

```text
base.py
_rolling_fast.py
registry / registration_audit / operator_spec / operator_policy
execution_contract / semantic_certification
operator_overhaul / cleanup / polars_gap_coverage / rolling_pack
common/time_series.py
common/cross_sectional.py
common/cs_broadcast.py
common/group.py
common/data_cleaning.py
common/daily_panel.py
safe_ops.py
statistics.py
_dedupe.py
price_volume/ops.py
price_volume/beta_helpers.py
price_volume/capm_helpers.py
price_volume/liquidity_v2.py
return_decomp.py
robust_stats.py
direction_concentration.py
state_event.py
stateful/_common.py
stateful rule/event/sequential/episode/survival families
conditional_ext.py
weighted_moment_ext.py
weighted_tail.py
activity_clock.py
volume_clock.py
intraday_activity_duration.py
session_recovery.py
microstructure/session.py
microstructure/intraday_agg.py
microstructure/flow_impact.py
microstructure/ops.py
intraday realized beta/volatility/jump/impact families
ashare/state_machine.py
fundamental/transforms_v2.py
fundamental/parameter_contract_v2.py
fundamental/ledger.py
fundamental/expectation_v2.py
valuation/ops_v2.py
shareholder/churn_network.py
index_listing/ops_v2.py
relation/ops_ext.py
cross_section_ext.py
group_ext.py
group_spectrum.py
distribution_break.py
recurrence_analysis.py
advanced_quantile_dynamics.py
prospect_theory.py
turnover_survival.py
composition.py
advanced topology / topology / intrinsic dimension / hankel / multifractal
research spectral / Granger / HSIC / BDS
AR / volatility / state-space / dynamic-regression / path-signature model families
Markov / EVT / rough-vol / local-Lyapunov / event-interval families
```

**执行 AI 仍必须以第22节 static scan覆盖整个目录**。上面列表只表示重点人工审计覆盖，不意味着未列文件可以跳过。

---

# 33. 对“已经写了 R11/R13 fixed 注释”的特别要求

当前代码里大量注释写着 `R5/R6/R11/R13 P0 fixed`。这些注释只能作为“作者意图”，**不能作为认证证据**。本次审计已经多次看到：

- 注释说 strict，但实际还有 clamp；
- 注释说 ConditionBool，但 Inf 漏过；
- 注释说 SameAxis，但下层helper仍reindex；
- 注释说 stateful authority single，但仍有 legacy name set；
- 注释说 negative invalid，但实现 `maximum(x,0)`；
- 注释说 fixed threshold frozen，实际每个rolling window重估；
- 注释说 keyword-only，实际ABI仍允许位置参数。

因此执行 AI 的规则是：

> **任何“已修”只能由当前代码行为 + regression test + final registry contract共同证明。注释编号不是证据。**

