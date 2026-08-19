# FactorEngine R30：全算子逐项隐形风险终审、可修即修、随机/未来物理删除、时钟/复权/股票池/语义身份与生产准入闭环

- 仓库：`18047533889/quant_projects`
- 主要目录：`factor_engine`
- 必要联动：`data_access` / `dataaccess`（以仓库实际目录为准）
- 审计基线 HEAD：`58490a634eef36efc92ecaec945fc4a26016661c`
- 日期：2026-08-10
- 当前 catalog 快照：1362 canonical；Daily 1185 / Extended 74 / Research 92 / Unsafe 7 / Legacy 1 / Internal 3
- 本文件为新的独立整改轮次，不覆盖 R28 / R29。

R28 解决逐算子测试、模型 Walk-Forward/PIT、证据闭环；R29 解决默认 runtime 瘦身、危险算子物理删除和 Research 隔离；R30 继续审计“算子代码本身看起来没有未来函数，但真实量化使用仍可能错误”的隐形问题，重点覆盖交易可用时间、复权历史重写、动态股票池/生存者偏差、语义 hash、证据失效、参数角色、样本支持、状态机和数值极值。

## 1. 本轮处理原则

最终只允许以下处理：

1. **天然不应该用于生产因子搜索**：物理删除 executable implementation。
2. **有真实量化价值且可以严格修正**：修好后保留。
3. **有研究价值但无法建立生产契约**：物理隔离到 ResearchToolRegistry。
4. **纯内部数学工具**：移到 InternalKernelLayer。
5. 旧公式需要迁移提示时，仅保留 **不可调用 tombstone**。

禁止最终留下 `DENIED_BUT_EXECUTABLE`、`RESEARCH_BUT_PUBLIC`、`UNSAFE_BUT_CALLABLE`、`UNKNOWN`、`MAYBE`。


## 2. 随机数、未来函数、非因果填充：必须直接删除 executable

以下类型最终必须没有任何可执行 operator：

```text
rand_uniform
rand_normal
rand_lognormal
rand_poisson
rand_exp
shuffle
sample

Lead
next

bfill
causal_bfill
fillna_interpolate
任何使用未来端点完成的 interpolation
```

当前代码仍在治理集合中出现这些名字，例如 `INTENTIONALLY_PANDAS_ONLY` 仍包含多个 `rand_*` / `shuffle`，`PIT_UNSAFE_CANONICALS` 仍包含 `Lead/next/bfill/...`。R30 不把它们继续当“特殊类别 operator”，而是清出 executable 系统。

硬门：

```text
ACTIVE_RANDOM_FACTOR_PRIMITIVES == 0
ACTIVE_FUTURE_REFERENCE_PRIMITIVES == 0
ACTIVE_NONCAUSAL_FILL_PRIMITIVES == 0
```

删除后不得出现在默认 loader、OperatorRegistry runtime、DSL allowlist、所有 miner、backend emitter、composite lowering、cold-start generation。只可出现在 tombstone、negative tests、migration docs。


## 3. Tombstone 不是 operator

建议：

```python
@dataclass(frozen=True)
class OperatorTombstone:
    name: str
    removed_since: str
    risk_class: str
    reason: str
    replacement: str | None
```

Tombstone 不得拥有 `calculate`、backend、operator class，不计入 canonical，不得被 mining。`Lead(x,1)` 应报 `RemovedOperatorError`，而不是执行，也不能静默替换成 `Delay`。


## 4. 必须重新逐 current canonical 审计

执行 R30 时必须 fresh process 获取当前 HEAD 和真实 registry，不得硬编码 1362。每个 canonical 一行输出：

```text
canonical / aliases / source / backends
surface / lifecycle / role
public_loaded / raw_registry_callable
pit_safe / randomness / future_reference
panel_params / scalar_params / parameter_roles
input/output semantic types
missing/current-row/history/sample-support contracts
state/chunk/checkpoint
event_time / knowledge_time / available_time / earliest_execution
source/universe/corporate-action contracts
unit/economic meaning
numerical domain
test node ids
final disposition / fix / deletion reason
```

最终 disposition 只能是：

```text
KEEP_PRODUCTION
KEEP_PRODUCTION_HIGH_COST
KEEP_PRODUCTION_CONTEXTUAL
KEEP_STATE_CONDITION_EVENT
KEEP_INTERMEDIATE
MOVE_RESEARCH_ISOLATED
MOVE_INTERNAL
DELETE_TOMBSTONE
DELETE_COMPLETELY
```


## 5. 当前 HEAD 已确认：Production admission 没有硬检查 pit_safe

`_compute_allow_in_production(..., pit_safe, ...)` 当前接收 `pit_safe`，但函数体没有显式 `if not pit_safe: return False`。安全性不应完全依赖 `production_certified` 等下游证据。

修复：

```python
if not pit_safe:
    return False
```

同时 lifecycle 必须 fail-closed。当前只显式拒绝 experimental/deprecated/stub/doc_only，没有显式拒绝 research。生产准入建议只接受明确 `status=="production"`（或封闭的 reviewed production lifecycle 枚举）。

硬门：

```text
PRODUCTION_ADMISSION_REQUIRES_PIT_SAFE
RESEARCH_STATUS_PRODUCTION_ZERO
```


## 6. 当前 daily surface 中 pit_safe=false 的算子必须逐项解决

当前 catalog 中仍存在 `surface=daily` 但 `pit_safe=false` 的条目，例如 ADL、ADX、CMF、CMO 等。不能批量改成 true。

逐个做：

```text
prefix invariance
future perturbation
missing-gap behavior
stateful chunk/full-history behavior
```

若实现本身 causal，则修 metadata/evidence；若实现错误则重写；若本质无法 causal，则移 Research 或删除。最终生产可达集合不得有 `pit_safe=false`。


## 7. 当前默认 production loader 仍加载 research 模块

当前 `cleaned_operators/__init__.py` 自称“唯一 production runtime 层”，但 `_LOAD_MODULES` 仍包含 `research_polars`、`ts_model.*` 多个研究模块、`cross_section.panel_model`、`research_transform`、`dmd`、`research_spectral` 等。

必须拆成：

```text
PRODUCTION_LOAD_MODULES
RESEARCH_LOAD_MODULES
INTERNAL_KERNEL_MODULES
```

普通 `import factor_engine` 只加载 production。Research 必须显式 opt-in。


## 8. 当前 OperatorRegistry.get 仍可绕过生产门禁

`OperatorRegistry.get(name)` 当前直接 resolve alias 后返回 runtime instance，没有 production/research/unsafe mode gate。

整改优先级：

1. 把 raw registry 变内部 API；或
2. `get(name, mode="production")` 默认执行 production admission；Research 需显式 `mode="research"`。

被删除的随机/未来名字调用 `get()` 必须返回 removed error / None，不得返回 executable。


## 9. panel/scalar 参数类型不能继续靠“无默认值=panel”猜

`_infer_panel_params()` 在没有显式 `panel_params` / `input_fields` 时，会把函数签名中没有默认值的参数推断成 panel。对 `ts_threshold_cycle_period(x, lower, upper, window=...)` 之类会把 `lower/upper` 错认成 panel。

所有 production operator 必须显式声明：

```python
metadata.panel_params = (...)
metadata.scalar_params = (...)
```

更优可引入 `ParameterKind.PANEL / SCALAR / CONTEXT / CALENDAR / SCHEMA`。

生产认证中只要参数类型来自 heuristic inference 就失败。


## 10. 中央 multi-panel alignment 是当前正确方向

当前标准 `SeriesOperator.calculate()` 已经统一经过 `validate_operator_call()` 和 `_validate_panel_axes()`，能挡重复 index/columns、普通多 panel 轴错配。这个中央门应成为唯一 authority，不要再在所有模块复制一套 alignment。

但特殊 broadcast 仍需下一节修正。


## 11. BroadcastSpec 当前按 frame 顺序验证，而不是按参数名绑定

`BroadcastSpec` 已定义 `source_param/target_param`，但 `_verify_broadcast_specs()` 当前把第一个 frame 当 base，然后对所有后续 frame 套所有 spec。3+ panel、多个 broadcast source 或 kwargs 重排时会验证错对象。

修复：从统一 canonical bound 建立 `param_name -> panel`，然后：

```python
source = bound_panels[spec.source_param]
target = bound_panels[spec.target_param]
verify(source, target, spec)
```

一个 spec 只检查它声明的 pair；参数名不存在、重复或不是 panel 均 fail closed。kwargs 顺序变化不能影响结果。


## 12. contract / semantic hash 覆盖仍不完整

当前 `_contract_hash()` 注释已经承认尚未纳入 input/output semantic types、ClockContract、HistoryTransform、missing_policy、universe_requirement。当前代码虽纳入 `available_at`，但仍需确保 `same_session_usable`、role、BroadcastSpec、参数种类、完整 history/sample contract、market/source/universe/corporate-action 语义全部进入 semantic identity。

任何会改变因子真实含义的变化，都必须改变 digest，从而使旧 evidence/cache 失效。


## 13. implementation hash 的 closure cell 排序会丢绑定关系

当前 `_fn_payload()` 把 closure cell 冻结后排序再 hash。闭包 cell 的位置与 `co_freevars` / `LOAD_DEREF` 是有语义关系的；相同值集合但不同变量绑定可能碰撞。

改成：

```python
[(co_freevars[i], freeze(cell_i)), ...]
```

保持原顺序，不排序。

Generic Mapping 的 key 也不要用 `repr` 排序任意 object；使用 canonical frozen key，或限制 production contract map key 类型。


## 14. contract hash 读取关键契约失败不能 broad-except 后 pass

当前读取 execution/edge/cost contract 的部分路径有 broad `except Exception: pass`。生产 evidence 生成时，关键契约解析失败必须 fail closed；Research 最多记录 `UNKNOWN:<error>`，不能生成一个“缺字段但看起来正常”的 hash。


## 15. DMD：旧平方溢出已修，但 log geometric sum 仍有新 overflow

当前 DMD 已改为 `2*log(abs(lambda))` 和 `2*log(abs(b))`，旧 R28 的 square-before-log 问题不再成立。

但 `_log_finite_horizon_sum(log_rho,K)` 仍先执行 `r=exp(log_rho)`，并使用 `expm1(log_rho)`。当 `log_rho=1000` 等有限大值时仍 overflow。

修复必须全程不 materialize `rho`：

- `lr>0`：`(K-1)lr + log(1-exp(-Klr)) - log(1-exp(-lr))`
- `lr<0`：`log(1-exp(Klr)) - log(1-exp(lr))`
- `lr≈0`：`log(K)`

实现稳定 `log1mexp`。测试 `0, ±1e-15, ±1e-9, ±1e-6, 1,10,100,1000,-1000`。


## 16. path_signature：current row missing 仍会退回旧 run

当前 `_trailing_contiguous_xy()` 如果最后一行无效，会向前回退到最近的 finite run，于当前 t 输出旧路径统计。

必须：

```python
if n == 0 or not valid[-1]:
    return None
```

Depth-2 norm 的 robust scale也必须基于同一个 joint trailing contiguous run，不得分别在带 NaN 的 x/y suffix 上求 scale。


## 17. stateful survival：gap 后 episode age 仍可能串线

当前逻辑在 ACTIVE → NaN 时设置 `prev_active=False`，但不清 `cur`。随后若先出现 INACTIVE，因为 `prev_active=False` 不执行 `cur=0`；再出现 ACTIVE 时会继续 `cur+=1`，新 episode 可继承 gap 前年龄。

不要只补一行。重写为显式状态机，例如：

```text
INACTIVE
ACTIVE_OBSERVED_ENTRY
ACTIVE_LEFT_CENSORED
UNKNOWN
```

为每条 transition 定义 age、completed-run update、left/gap censor、输出。专项 golden：

```text
0,1,1,0
0,1,NaN,1,0
0,1,NaN,0,1
1,1,0
NaN,1,1,0
0,1,1,NaN,0,1
0,1,1,NaN,1,1,0
```

并修正 docstring 中“inactive output 0”和默认 `inactive_policy="nan"` 的漂移。


## 18. fin_component_score：FALSE 仍被当成 MISSING

当前 `_score_component()` 用 `np.where(finite & condition, 1, NaN)`。因此完整数据但条件不成立会变 NaN；所有条件 false 时 score 可能 NaN，而正确的三值逻辑应该是：

```text
TRUE -> 1
FALSE -> 0
UNKNOWN/MISSING -> NaN
```

同时明确 component missing policy：`require_full_components` 或 `score_available_components`。若允许 partial score，必须记录 effective component count，否则不同 completeness 的 score 不可比。


## 19. fin_component_score 的经济语义也要收紧

若它继续叫 `fin_component_score`，panel input 应限定为 FundamentalFeature/FundamentalCondition semantic type，不能让 LLM 把 momentum、volatility、PE、volume 任意混进去仍叫 fundamental score。

若需要通用评分，单独建立 `generic_condition_score`。


## 20. composition：当前 financial_statement family 仍然经济上错误

当前 schema 把 revenue、operating_revenue、net_income/net_profit、total_assets、liabilities、equity 放进同一个 family。这里混合了 flow 与 stock，且 `Assets = Liabilities + Equity`，把 whole 与 components 一起放入 composition 不成立。

必须使用 `PartOfWholeContract`：

```text
composition_id
total_concept
part_id
mutually_exclusive
stock_or_flow
period_semantics
currency/unit
strict_positive
```

Production 不再接受 anonymous positional composition。没有真实 part-of-whole 数据 schema 时，整类移 Research，不要伪造“financial_statement composition”。


## 21. KAMA：gap rewarm 数学支持 off-by-one

当前 KAMA 的 ER 使用 `close-close.shift(er_window)` 和 `diff().rolling(er_window)`，需要 `er_window+1` 个连续价格点。当前 rewarm 在 `contiguous==er_window` 就允许 seed 输出，实际 ER/alpha 尚未完整定义。

修复：完整 warmup 至少 `er_window+1` 点，并固定第一个合法 KAMA seed 定义（current close / SMA / 标准参考），首次和 gap 后都用同一规则并做 golden test。


## 22. Supertrend：gap 后硬编码 bullish re-seed

当前缺失后下一有效 bar会 `trend=+1` 并输出 lower band，形成系统性 bullish restart bias。

推荐 gap 后进入 UNKNOWN，待足够连续 ATR history 和价格/band关系确认方向后再发布。若必须固定初始化，也不得把初始化 bar 当真实方向 alpha，且必须在 contract/golden test中明确。


## 23. PSAR：gap 后也硬编码 bull=True

当前 PSAR mid-series re-seed 设 `bull=True, sar=low`，并可能当日直接输出。与 Supertrend 同类。建议 gap 后 UNKNOWN，至少使用足够连续有效 bars 确定初始方向；不能让数据缺口自动制造多头状态。


## 24. ParamRole：所有 production scalar 必须显式

当前很多 `ParamSpec` 只声明 dtype/min/max，没有 `ParamRole`，如技术指标 `_WIN_GE2` / `_WIN_GE1` / `_POS_FLOAT`。Production 不能依赖 fallback 把未声明角色自动视为 ECONOMIC。

逐项声明：

```text
window/horizon -> HORIZON
真实触发阈值 -> STATE_THRESHOLD
support floor -> SUPPORT_POLICY
session/calendar -> SESSION_POLICY
source/vintage -> SOURCE_POLICY
缺失策略 -> MISSING_POLICY
纯估计器分辨率 -> ESTIMATOR_RESOLUTION
模型阶数 -> MODEL_ORDER
数值epsilon -> NUMERICAL
```

Support/Policy/Session/Source/Missing/Numerical 不进入因子搜索。


## 25. HistoryRequirement：production 不能继续依赖 legacy 名字 fallback

`runtime/execution_contract.py` 当前仍保留 `_STATEFUL_CANONICALS` 和 `_WINDOW_LIKE_PARAM_NAMES` fallback。作为 migration 兼容可以存在，但生产 operator 必须自声明 statefulness、chunking、history kind/formula、minimum effective samples。

硬门：

```text
PRODUCTION_LEGACY_STATEFUL_SEED_FALLBACK_ZERO
PRODUCTION_HISTORY_NAME_GUESS_FALLBACK_ZERO
```


## 26. ts_regression minimum effective samples 当前存在 unreachable parameter-aware branch

当前静态 map 已有 `ts_regression: 2`，函数随后才有 `if canonical=="ts_regression": return n_regressors+1`。前面的 map return 会把后面的 parameter-aware floor遮住。

改为参数化 contract，例如：

```text
minimum_effective_samples = n_regressors + 1
```

测试：1→2、2→3、5→6、10→11。所有高维 regression/PCA/covariance/KNN 同样检查样本地板是否随维数/阶数变化。


## 27. PIT 不等于可交易：新增 Availability / Decision / Execution Clock

一个 EOD 因子可能完全 causal，但如果用同日 close 计算再假设按同一 close 成交，仍是 execution-time lookahead。

每个生产输入/因子至少区分：

```text
event_time
knowledge_time
factor_available_time
decision_time
execution_time
```

监督模型另加 `label_maturity_time`。

基本约束：

```text
knowledge_time <= factor_available_time <= decision_time <= execution_time
```

具体等号按市场/订单规则声明。


## 28. 日线 close/high/low/volume 的 same-bar 使用必须机械拒绝

日终才完整的 close/high/low/volume-derived factor 默认：

```text
available_at=session_close
same_bar_usable=False
earliest_execution=next_legal_execution_point
```

不要让 backtest 通过一个普通日期 index 就默认同 close 成交。


## 29. minute→daily 全部声明 EOD availability

所有需要完整 session 的 intraday aggregate：

```text
available_at=session_close
same_session_usable=False
```

DAG 要传播 available time：父节点 availability = children 最大 availability + 自身约束。Composite factor/materialized factor必须记录 earliest execution。


## 30. session_recovery 当前仍硬编码 market='ashare'

当前代码一方面支持 CN/HK/US timezone 解析，另一方面 `build_session_panel(..., market="ashare")` 仍硬编码 A股。

两种合法修法：

A. 真多市场：market 从 `calendar.market` / ExecutionContext / provenance 获取；
B. 明确 A_SHARE_ONLY，其他市场直接 `MarketNotSupportedError`。

禁止“timezone看似支持US但真正session builder仍按A股”的半通用状态。


## 31. 所有 intraday 算子统一消费 SessionPanel

不要每个模块自行 `index.normalize()`、猜 timezone、猜 bar frequency、猜 close。统一 SessionPanel 必须包含：

```text
market
trade_date
session timezone
official slots
timestamp convention
auction/session segments
missing slot mask
duplicate slot check
session complete
```

“整行分钟不存在”与“该分钟值是 NaN”都必须在 official-grid reindex 后可见。


## 32. Corporate Action / 复权历史是一个必须新增的 PIT 门

今天导出的前/后复权历史可能用未来分红、送转、拆股信息重写过去价格。回测若直接使用当前 vintage 的整段 adjusted history，会把未来公司行为写回过去。

Production 输入必须区分：

```text
RawPrice
PITAdjustedPrice(as_of)
CurrentVintageAdjustedPrice
PITTotalReturnIndex
```

DataAccess 应携带 corporate-action effective date、knowledge/announcement time、adjustment vintage。

未来公司行为扰动测试：在 cutoff 后新增/删除 corporate action，要求所有 `t<=cutoff` 的 factor 完全不变。


## 33. 如果没有 PIT adjustment vintage，不要伪装安全

优先使用 raw price + PIT corporate-action transform；或把依赖复权 vintage 的 production factor阻断。这个问题应由 SourceContract/typed input解决，不是在每个 `ts_return` 里猜。


## 34. Dynamic Universe / Survivorship Bias 必须成为 production contract

Cross-sectional rank/zscore/neutralize/group/KNN/index relation 即使数学 causal，如果历史股票池用“今天仍存活的股票”，仍会有生存者偏差。

需要 UniverseContract：

```text
membership effective time
membership knowledge time
listing/delist state
suspension/tradability state
market/board
```

未来 IPO、未来 delist、未来 index member、未来 industry change 的扰动不能改变 cutoff 前 factor。


## 35. index/listing operator：输入 panel正确还不等于 source PIT被证明

当前 index/listing 模块已经正确处理不少局部语义，如 member NaN不当 non-member、上市前 age NaN、unknown suspension 不当正常日，这些应保留。

但 operator 本身只看到 member/listing/is_suspend panel，无法证明 panel 是 as-of 构造。生产必须要求 provenance：

```text
effective_from/effective_to
announcement_time
knowledge_time
source_snapshot
```

指数“公告”与“生效”是不同事件语义，不能混。


## 36. Industry / group membership 同样要 as-of

未来行业分类变更不能回填过去。Group/KNN/peer/neutralization需要 GroupMembershipContract。Cross-sectional cohort每个 t 只能由 t 当时存在且可见的股票构成。


## 37. Fundamental / Consensus / Shareholder source PIT

继续严格区分：

```text
period_end
announcement_time
knowledge_time
revision/restatement time
consensus vintage
shareholder PubDate
```

未来财报重述不能覆盖过去当时已知版本；`fin_surprise` 的 expectation snapshot必须严格早于 actual knowledge time；historical consensus必须是真 vintage。

匿名 DataFrame默认不能执行 production contextual operator，Research可允许但明确 uncertified。


## 38. Label / Feature Firewall

ForwardLabel / FutureOutcome / EvaluationTarget semantic type 不得作为 production factor operator 的 feature input。Event-response 中用于历史路径计算的 response 应是逐时点 realized response series，而不是已经预制的 forward-H label。

LLM/GP grammar层直接禁止 label semantic type进入 feature DAG。


## 39. 模型类继续执行 R28 Walk-Forward 门

所有 predictive/innovation/surprise model：

```text
fit/scaler/PCA/hyperparameter data <= t-1
current t only for scoring
```

监督 label按 maturity time，重叠 label按需要 purge/embargo。生产时间序列模型禁止 random KFold/shuffle。Kalman/HMM只允许 filtered state，不允许 full-sample smoothed state作为历史 factor。


## 40. Pivot / ZigZag / 图形确认

需要未来 bars 才能确认的 pivot可以保留，但信号只能在 confirmation time首次发布，绝不能回写原 pivot timestamp。无法定义明确 confirmation time 的 pattern 移 Research或删除。

未来 perturbation必须证明 cutoff 前已发布历史不被未来更极端 pivot重写。


## 41. CurrentRowRequirement 全算子显式化

每个 production operator声明：

```text
CURRENT_REQUIRED
CURRENT_OPTIONAL
STRICTLY_PRIOR_ONLY
EVENT_MATURED_ONLY
```

CURRENT_REQUIRED 遇当前 row NaN 必须 NaN，不能回退昨日；path_signature就是当前反例。


## 42. Missing semantic 不只一个 NaN

长期建议 provenance/validity mask区分：

```text
MISSING_DATA
UNKNOWN_STATE
NOT_APPLICABLE
```

例如停牌、未上市、provider缺失虽然都可能数值 NaN，但经济含义不同。至少 source-sensitive operator不能把未知状态自动当 False/0。


## 43. ffill 必须 typed

Causal ffill本身可以存在，但不能无条件用于所有字段。财务 as-of value通常允许公告后向后 carry；trade price无约束 ffill可能错误。建议由 input semantic type决定是否允许，并要求 max-gap/source policy。


## 44. 数值极端和 EdgeContract

每个 production canonical必须有 NaN、±Inf、zero、domain-invalid 处理。多 backend必须一致。

Fuzz至少覆盖：

```text
1e-300,1e-200,1e-100,1e-12,0,1e12,1e100,1e200,1e300,±inf,nan
```

Probability输出约束 [0,1]；correlation [-1,1]；Condition/Event {0,1,NaN}；State在声明状态集合内。


## 45. Ratio 的 epsilon 不能无脑固定

除零/near-zero 的 `_EPS` 若直接进入经济公式，可能产生量纲相关的隐含定义。逐算子决定 exact-zero、relative tolerance或scale-aware policy，并进入 semantic contract；不同 backend一致。


## 46. ACF/统计缺失语义逐项核实

当前 ACF已经避免 dropna 后压缩时间，使用 physical lag pair，这是正向修复。但其均值/方差支持集与 numerator pair支持集是特定定义，必须写成 golden/reference并保证所有 backend一致。所有 variance/std/cov/skew/kurt明确 ddof、最小有限样本、tie/NaN规则。


## 47. Stateful / expanding / cumulative 的 anchor contract

Expanding/cumulative虽然 causal，但依赖 dataset origin。每个算子明确 AnchorPolicy：

```text
since_listing
since_explicit_reset
since_dataset_origin
bounded_rolling
```

需要 dataset-origin的不能伪装成有限 warmup；增量运行必须 full history或 checkpoint。


## 48. Market scope 和 currency

A股专用 operator显式 `market_scope=ashare`；US/HK同理。跨财务/估值字段还要明确 currency；需要 FX 时 FX本身也有 knowledge/availability time，不能拿当天收盘 FX给当天收盘前 decision。


## 49. Recipe/DAG 语义要一起审

单个 operator正确不够。Composite DAG必须验证 input semantic type、role、unit、market、time、universe/source contracts。Global/group state不能作为独立 cross-sectional terminal alpha；terminal role需要横截面有意义。

Lineage hash必须包含 node semantic hash + params + child hashes + source snapshot + universe snapshot + corporate-action vintage。


## 50. Cache key 也要带 semantic/source/universe/vintage

否则代码不变但 source snapshot、复权 vintage、universe版本变化时，会错误复用旧 factor cache。任何上述 contract变化都必须使 evidence和cache同时失效。


## 51. 每个 retained canonical 的测试不能只“能运行”

按风险分级：

LOW：
- execute
- param
- NaN/domain

MEDIUM：
- + prefix
- + missing topology
- + golden/metamorphic

HIGH：
- + future perturbation
- + source PIT
- + decision/execution clock
- + chunk/checkpoint
- + reference oracle
- + numeric extreme

HIGH包括 model/fundamental/consensus/intraday/stateful/event/pivot/relation/index/universe/corporate-action/spectral/topology/high-dimensional estimator。


## 52. 加 Mutation Testing 验证测试真的能抓泄漏

故意注入：

```text
FUTURE_SHIFT
CURRENT_INCLUDED_IN_FIT
DROPNA_COMPRESS
NAN_TO_ZERO
CURRENT_MISSING_BACKOFF
REVERSE_TIE_RULE
SILENT_INT_CAST
FUTURE_UNIVERSE_MEMBER
FUTURE_CORPORATE_ACTION
SAME_CLOSE_EXECUTION
```

高风险测试必须能把这些 mutation杀死，否则“有测试”不等于测试有效。


## 53. A股专项 fixtures

至少覆盖：

```text
一字涨停/跌停
停牌复牌
午休
整分钟缺失
重复bar
除权除息
上市首日/短历史
ST/板块规则状态
指数调入调出
财报盘前/盘中/盘后发布
```


## 54. 当前已确认问题编号

基于 HEAD `58490a634eef36efc92ecaec945fc4a26016661c`：

```text
R30-P0-001  random/future/noncausal names must leave executable system
R30-P0-002  default production loader still imports research modules
R30-P0-003  raw OperatorRegistry.get bypasses production mode
R30-P0-004  production admission does not hard-check pit_safe
R30-P0-005  research lifecycle not hard-rejected in production admission
R30-P0-006  panel/scalar fallback can misclassify required scalars
R30-P0-007  BroadcastSpec validation not bound to source_param/target_param
R30-P0-008  semantic/contract hash omits behavior-critical contracts
R30-P0-009  closure hash sorts cells and loses freevar binding
R30-P0-010  path_signature current missing backs off to stale prior run
R30-P0-011  DMD log geometric sum still materializes exp(log_rho)
R30-P0-012  survival gap state can carry prior episode age
R30-P0-013  fin_component_score conflates FALSE and MISSING
R30-P0-014  composition financial_statement family economically invalid
R30-P0-015  session_recovery still hardcodes market="ashare"
R30-P0-016  KAMA rewarm support off-by-one
R30-P0-017  Supertrend gap re-seed hardcodes bullish state
R30-P0-018  PSAR gap re-seed hardcodes bullish state
R30-P0-019  ts_regression dynamic sample floor shadowed by static floor=2
R30-P0-020  no universal decision/execution-time gate for same-close misuse
R30-P0-021  no universal corporate-action vintage gate
R30-P0-022  no universal as-of universe/survivorship gate
R30-P0-023  label semantic type needs hard feature firewall
R30-P1-024  production history/state still has legacy heuristic fallback debt
R30-P1-025  many production scalar ParamSpecs lack explicit ParamRole
R30-P1-026  contract hash component resolution can be silently omitted
R30-P1-027  generic Mapping hash uses repr-based key ordering
R30-P1-028  available_at/same_session fields need complete tradability propagation
R30-P1-029  index/member/listing source provenance not mechanically proven by operator
R30-P1-030  daily pit_safe=false catalog rows still need behavioral resolution
```


## 55. 必须生成的 R30 artifacts

固定目录：

```text
factor_engine/docs/evidence/r30/
```

至少：

```text
R30_HEAD.json
R30_CURRENT_CANONICAL_INVENTORY.csv/json
R30_PER_CANONICAL_FINAL_REVIEW.csv/json
R30_ALIAS_AUDIT.csv
R30_DELETION_MANIFEST.csv
R30_TOMBSTONE_MANIFEST.json
R30_RESEARCH_ISOLATION_AUDIT.json
R30_PUBLIC_REGISTRY_AUDIT.json
R30_PRODUCTION_ADMISSION_AUDIT.json
R30_PARAMETER_KIND_AUDIT.csv
R30_PARAMETER_ROLE_AUDIT.csv
R30_BROADCAST_CONTRACT_AUDIT.csv
R30_SEMANTIC_HASH_AUDIT.json
R30_EVIDENCE_INVALIDATION_TESTS.json
R30_AVAILABILITY_CLOCK_AUDIT.csv
R30_SAME_CLOSE_AUDIT.json
R30_CORPORATE_ACTION_PIT_AUDIT.csv
R30_UNIVERSE_PIT_AUDIT.csv
R30_HISTORY_SUPPORT_AUDIT.csv
R30_STATE_MACHINE_AUDIT.csv
R30_NUMERICAL_EXTREME_AUDIT.csv
R30_PREFIX_INVARIANCE_RESULTS.csv
R30_FUTURE_PERTURBATION_RESULTS.csv
R30_MUTATION_TEST_RESULTS.csv
R30_PER_CANONICAL_TEST_COVERAGE.csv
R30_PER_CANONICAL_TEST_RESULTS.csv
R30_PYTEST_JUNIT.xml
R30_ARTIFACT_MANIFEST.json
R30_FINAL_ACCEPTANCE_REPORT.md
```

全部绑定最终 `git_sha`、canonical-set digest、semantic-contract digest、test-suite digest，并用 `git check-ignore` + `git ls-files` 验证在 GitHub tracked。


## 56. 建议测试目录

```text
factor_engine/tests/operators/r30/
```

重点测试：

```text
test_random_future_physically_removed.py
test_raw_registry_cannot_get_removed.py
test_production_loader_has_no_research.py
test_production_admission_requires_pit_safe.py
test_research_status_cannot_production.py
test_all_public_param_kinds_explicit.py
test_all_production_param_roles_explicit.py
test_broadcast_spec_named_binding.py
test_semantic_contract_hash_complete.py
test_closure_hash_freevar_order.py
test_contract_hash_failure_is_not_silent.py
test_path_signature_current_missing.py
test_dmd_log_geometric_extremes.py
test_survival_transition_table.py
test_component_score_three_valued_logic.py
test_composition_part_of_whole.py
test_kama_rewarm_support.py
test_supertrend_gap_initialization.py
test_psar_gap_initialization.py
test_session_recovery_market_context.py
test_availability_clock_same_close.py
test_dag_availability_propagation.py
test_corporate_action_future_perturbation.py
test_universe_future_perturbation.py
test_industry_membership_future_perturbation.py
test_index_membership_effective_time.py
test_regression_parameter_aware_support.py
test_label_feature_firewall.py
test_production_legacy_history_fallback_zero.py
test_all_current_canonicals_have_disposition.py
test_all_retained_canonicals_have_real_tests.py
```


## 57. 全算子 family checklist

对 current set 全量分桶，逐个 canonical 执行相应 checklist：

- Common elementwise：domain/unit/overflow/NaN/duplicate/constant。
- Time-series：window/lag/current inclusion/missing topology/history。
- Cross-sectional：as-of universe/ties/column permutation/availability。
- Group：group key/as-of classification/singleton/broadcast role。
- Technical：warmup/recursive init/gap rewarm/OHLC/parameter role。
- Candlestick：OHLC validity/NaN≠False/pattern sign。
- Structure/pattern：confirmation/backdating/supersession/staleness。
- Intraday：calendar/timezone/official slots/auction/lunch/close。
- Fundamental：period/knowledge/revision/flow-stock/currency/restatement。
- Valuation：price time vs fundamental knowledge/share vintage/currency。
- Shareholder：PubDate/holder identity/top-K/snapshot completeness。
- Index/listing：announcement vs effective/member PIT/survivorship。
- Relation：edge/direction/weight/validity/knowledge time。
- Stateful：transition/gap/censor/checkpoint/chunk。
- Model：walk-forward/label/scaler/PCA/hyperparameter。
- Information theory：sample/bin/kernel/surrogate reproducibility。
- Spectral：trailing-only/frequency units/gap。
- Geometry/topology：scale/current query exclusion/sample floor。
- Composition：real part-of-whole。
- Research：default production不可见。
- Internal：DSL/mining不可见。


## 58. R30 Hard Gates

最终全部必须为 TRUE：

```text
R30_CURRENT_HEAD_BOUND
R30_EVERY_CURRENT_CANONICAL_REVIEWED
R30_EVERY_CURRENT_CANONICAL_FINAL_DISPOSITION

R30_ACTIVE_RANDOM_FACTOR_PRIMITIVES_ZERO
R30_ACTIVE_FUTURE_REFERENCE_PRIMITIVES_ZERO
R30_ACTIVE_NONCAUSAL_FILL_PRIMITIVES_ZERO

R30_DEFAULT_PRODUCTION_LOADER_RESEARCH_ZERO
R30_RAW_REGISTRY_PRODUCTION_BYPASS_ZERO

R30_PRODUCTION_ADMISSION_REQUIRES_PIT_SAFE
R30_RESEARCH_STATUS_PRODUCTION_ZERO

R30_PUBLIC_PARAM_KIND_EXPLICIT
R30_ALL_PRODUCTION_SCALAR_PARAM_ROLES_EXPLICIT
R30_BROADCAST_SPEC_BOUND_BY_PARAMETER_NAME

R30_SEMANTIC_HASH_COVERS_BEHAVIOR_CONTRACTS
R30_CLOSURE_HASH_PRESERVES_FREEVAR_ORDER
R30_CONTRACT_HASH_FAILS_CLOSED

R30_PATH_SIGNATURE_CURRENT_MISSING_FAILS_CLOSED
R30_DMD_GEOMETRIC_LOGSUM_OVERFLOW_ZERO
R30_SURVIVAL_GAP_STATE_BUG_CLOSED
R30_COMPONENT_SCORE_FALSE_VS_MISSING_CLOSED
R30_COMPOSITION_ECONOMIC_SEMANTICS_VALID
R30_KAMA_WARMUP_OFF_BY_ONE_CLOSED
R30_SUPERTREND_GAP_RESEED_DIRECTION_UNBIASED
R30_PSAR_GAP_RESEED_DIRECTION_UNBIASED
R30_SESSION_RECOVERY_MARKET_CONTEXT_CORRECT

R30_ALL_PRODUCTION_FACTORS_HAVE_AVAILABILITY_CLOCK
R30_SAME_CLOSE_LOOKAHEAD_ZERO
R30_DAG_AVAILABILITY_PROPAGATION_PASS

R30_FUTURE_CORPORATE_ACTION_RESTATEMENT_ZERO
R30_SURVIVORSHIP_BIAS_ZERO
R30_CROSS_SECTION_UNIVERSE_ASOF_REQUIRED

R30_PRODUCTION_LEGACY_STATEFUL_SEED_FALLBACK_ZERO
R30_PRODUCTION_HISTORY_NAME_GUESS_FALLBACK_ZERO
R30_REGRESSION_SUPPORT_FLOOR_PARAMETER_AWARE
R30_ALL_SUPPORT_POLICY_PARAMS_NOT_SEARCHED

R30_LABEL_FEATURE_FIREWALL_PASS

R30_HIGH_RISK_PREFIX_INVARIANCE_PASS
R30_HIGH_RISK_FUTURE_PERTURBATION_PASS
R30_HIGH_RISK_MUTATION_TESTS_PASS
R30_RETAINED_CANONICAL_ZERO_REAL_TESTS_ZERO

R30_EVIDENCE_CURRENT_SHA
R30_EVIDENCE_CURRENT_SEMANTIC_DIGEST
R30_EVIDENCE_GITHUB_TRACKED

R30_HARD_BLOCKERS_ZERO
```


## 59. 执行顺序

建议严格按顺序：

```text
Phase 1  Fresh current inventory
Phase 2  Physical deletion of random/future/noncausal primitives
Phase 3  Production/Research/Internal loader split
Phase 4  Close raw Registry bypass
Phase 5  Fix production admission gate
Phase 6  Explicit panel/scalar kinds + ParamRole
Phase 7  BroadcastSpec named binding
Phase 8  SemanticIdentity / evidence invalidation V2
Phase 9  Fix concrete current bugs (path/DMD/survival/component/composition/KAMA/Supertrend/PSAR/session)
Phase 10 Availability/Decision/Execution clock
Phase 11 Corporate-action/source PIT
Phase 12 Universe/survivorship PIT
Phase 13 History/sample support migration
Phase 14 Full per-canonical family audit
Phase 15 Property/future-perturbation/mutation tests
Phase 16 Full regression + final evidence
Phase 17 Fresh-process acceptance
```

不要在中间版本生成最终 evidence；最后一次 clean fresh process 重新生成。


## 60. 给代码 AI 的直接执行口径

请实际修改代码，不要只改 status/tag/catalog：

```text
A. 基于当前 main HEAD fresh-process 获取完整 canonical set。
B. 每个 canonical逐项形成审计行，不能以 module smoke代替。
C. rand_*/shuffle/sample/Lead/next/bfill/noncausal interpolation等物理删除 executable。
D. 能严格修成生产可用的算子优先修好保留。
E. Research工具物理隔离出默认 runtime。
F. 修复本文已确认的 current-HEAD P0/P1。
G. 建立 factor availability / decision / execution clock，堵 same-close lookahead。
H. 建立 PIT corporate-action adjustment/vintage contract。
I. 建立 as-of universe/group/index membership contract。
J. Production不再依赖 panel/scalar、state/history、ParamRole的 heuristic。
K. Semantic hash覆盖所有行为语义，任何关键契约变化使 evidence/cache失效。
L. retained canonical至少一条真实执行测试，高风险必须 prefix + future perturbation + mutation adequacy。
M. 最终 evidence 提交 `factor_engine/docs/evidence/r30/` 并绑定最终 HEAD。
```


## 61. 最终 DoD

完成 R30 后，应达到：

```text
永远不用的随机/未来/非因果 operator 已物理删除；
Research 不在默认 production runtime；
默认可调用到的 operator 本身就是可用的；

每个保留 production operator明确知道：
- 吃什么语义的数据；
- 数据何时真正可见；
- 当前行缺失时怎么办；
- 需要多少历史和有效样本；
- 属于哪个 as-of 股票池；
- 是否受未来复权/公司行为重写；
- 什么时候形成因子；
- 最早什么时候可以合法交易；
- 哪些参数可搜索，哪些只是估计/治理；
- 极端值和缺失如何处理；
- 测试和 evidence对应当前 HEAD。

没有 UNKNOWN、DENIED_BUT_EXECUTABLE、RESEARCH_BUT_PUBLIC、ZERO_TESTS。
```

最终联合验收：

```text
R24 AND R25 AND R26 AND R27 AND R28 AND R29 AND R30
```

才允许：

```text
FACTOR_ENGINE_OPERATOR_SYSTEM_PRODUCTION_READY = true
```
