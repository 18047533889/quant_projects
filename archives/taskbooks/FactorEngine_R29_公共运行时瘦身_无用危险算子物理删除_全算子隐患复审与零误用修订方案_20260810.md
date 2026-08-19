# FactorEngine R29：公共运行时瘦身、无用/危险算子物理删除、全算子隐患复审与零误用修订方案

> **执行对象**：`18047533889/quant_projects/factor_engine`
>
> **本文件生成时基线 HEAD**：`37008c7ff4eff3823480443b6325577eae394250`
>
> **日期**：2026-08-10
>
> **定位**：R29 是一个新的独立整改轮次，不覆盖 R28。
>
> R28 的重点是“全 canonical 测试 + 模型 Walk-Forward/PIT + 证据闭环”；  
> R29 的重点进一步收紧为：
>
> **即使使用者完全不了解 FactorEngine 的 mining policy、surface、DirectUse、certification，也不能轻易调用到危险、研究、非 PIT、随机、废弃或语义不成立的算子。**

---

# 0. 本轮最终原则

FactorEngine 的安全边界不能建立在：

```text
“大家都知道不要用这个”
“它虽然注册了，但是 status=research”
“它虽然能执行，但是 mining 不会返回它”
“它虽然在 registry 里，但是 policy=denied”
“default_search_weight=0，所以应该没事”
```

而必须建立在：

```text
默认 production runtime 中根本不存在危险 executable operator。
```

因此本轮最核心的准则是：

```text
不用的，删。
危险的，删。
天然未来函数，删。
随机因子，删。
无法 PIT 化的，删。
经济语义不成立的，删或移出 production。
没有数据源且短期没有合法数据契约的，删或隔离。
纯 research 的，不进入 production loader。
纯 internal kernel 的，不进入 public registry。
```

只有为了：

```text
旧公式报错
旧 manifest 迁移
告诉用户 replacement
```

才允许保留：

```text
NON-CALLABLE TOMBSTONE
```

**Tombstone 不是 operator。**

---

# 1. 为什么 R29 必须做“物理删除”，不能只做 denied

当前架构已经有：

```text
surface
ProductionCertification
MiningRole
DirectUseStatus
PRODUCTION_DENIED_CANONICALS
PERMANENTLY_FORBIDDEN_CANONICALS
```

这些治理层是有价值的。

但它们不能解决：

```python
from cleaned_operators.registry import OperatorRegistry

op = OperatorRegistry.get("某个不该用的算子")
op.calculate(...)
```

这种直接调用。

如果一个新人：

```text
不知道 mining API
不知道 DirectUse
不知道 surface
不知道哪些是 research
```

他很可能：

```text
直接看 registry 有什么就用什么。
```

所以：

```text
“被拒绝，但仍可执行”
```

本身就是 R29 要消灭的状态。

---

# 2. 当前代码已经证明这个风险真实存在

当前：

```text
cleaned_operators/__init__.py
```

文件顶部自称：

```text
统一算子库（唯一 production runtime 层）
```

但是 `_LOAD_MODULES` 同时加载了：

```text
cleaned_operators.research_polars

cleaned_operators.ts_model.dynamic_regression
cleaned_operators.ts_model.ar_meanrev
cleaned_operators.ts_model.state_space
cleaned_operators.ts_model.volatility
cleaned_operators.ts_model.complexity
cleaned_operators.ts_model.wavelet_spectral
cleaned_operators.ts_model.sequence_anomaly
cleaned_operators.ts_model.path_signature

cleaned_operators.cross_section.panel_model

cleaned_operators.research_transform

cleaned_operators.dmd
cleaned_operators.research_spectral
```

以及大量：

```text
experimental / research-only
```

模块。

这意味着：

```text
“production loader”
和
“research implementation loader”
```

目前没有真正物理隔离。

---

# 3. R29 的目标架构

最终拆成四层：

```text
ProductionOperatorRegistry
ResearchToolRegistry
InternalKernelLayer
LegacyTombstoneRegistry
```

其中：

```text
ProductionOperatorRegistry
```

才是普通 FactorEngine 默认可见的唯一 operator registry。

---

# 4. 四层定义

## 4.1 ProductionOperatorRegistry

允许包含：

```text
生产 ALPHA
高成本生产 ALPHA
合法 STATE
合法 CONDITION
合法 EVENT
合法 INTERMEDIATE
合法 SOURCE TRANSFORM
经过 Source/PIT Context Gate 的 contextual operator
```

必须满足：

```text
可执行
语义明确
PIT/时间语义明确
参数严格
输入严格
有测试
有 source/economic contract
```

---

## 4.2 ResearchToolRegistry

只在：

```python
FactorEngine(mode="research")
```

或显式：

```python
load_research_tools()
```

时加载。

默认 production Python process：

```text
不 import
不 register
不 expose
```

---

## 4.3 InternalKernelLayer

包含：

```text
FFT kernel
SVD helper
PCA helper
matrix inverse
raw convolution
private statistical helper
backend utility
shared rolling kernel
```

可以被生产算子内部调用。

但：

```text
不是 public operator
不能 DSL author
不能 mining
不能 registry.get
```

---

## 4.4 LegacyTombstoneRegistry

只保存：

```python
RemovedOperator(
    name="Lead",
    risk_class="NONCAUSAL",
    removed_since="R29",
    reason="uses future observation",
    replacement=None,
)
```

它没有：

```text
calculate
backend
operator class
callable
```

---

# 5. R29 的合法最终 disposition

每个 current canonical 必须唯一落入以下之一：

```text
KEEP_PRODUCTION_ALPHA
KEEP_PRODUCTION_HIGH_COST
KEEP_PRODUCTION_CONTEXTUAL
KEEP_STATE_CONDITION_EVENT
KEEP_INTERMEDIATE
MOVE_RESEARCH_ISOLATED
MOVE_INTERNAL
DELETE_KEEP_TOMBSTONE
DELETE_COMPLETELY
```

不再允许最终状态：

```text
DENIED_BUT_EXECUTABLE
RESEARCH_BUT_PUBLIC_REGISTRY
UNSAFE_BUT_CALLABLE
LEGACY_BUT_CALLABLE
UNKNOWN
UNRESOLVED
PENDING_FOREVER
```

---

# 6. 第一原则：PERMANENTLY_FORBIDDEN 必须 executable = 0

当前永久禁止中包括：

```text
Lead
next

bfill
causal_bfill
fillna_interpolate
interpolate

shuffle
sample

rand_exp
rand_lognormal
rand_normal
rand_poisson
rand_uniform

dropna
constant

norm
norm_l1
norm_linf
```

R29 要求：

```text
这些名字不能再对应 executable operator。
```

---

# 7. 对 future function 的处理

例如：

```text
Lead
next
```

直接：

```text
DELETE_EXECUTABLE
```

只在 tombstone 中保留名字。

调用：

```text
Lead(close, 1)
```

返回：

```text
RemovedOperatorError:
Lead was removed because it is non-causal and leaks future information.
```

---

# 8. 对随机 factor primitive 的处理

例如：

```text
rand_uniform
rand_normal
rand_poisson
sample
shuffle
```

直接：

```text
DELETE_EXECUTABLE
```

因为它们作为因子 primitive：

```text
没有稳定经济意义
破坏可复现性
污染自动搜索
容易产生数据挖掘假象
```

---

# 9. 不允许“放 unsafe surface 就算解决”

如果：

```text
默认 production loader
```

仍然注册：

```text
unsafe canonical
```

R29 判失败。

最终硬约束：

```text
DEFAULT_RUNTIME_UNSAFE_EXECUTABLES = 0
```

---

# 10. 当前 unsafe 7 个重新审

当前：

```text
arg
tan
cot
sec
csc
cosh
sinh
```

不要因为：

```text
“数学库里存在”
```

就保留。

逐个问：

```text
有没有真实因子构造用途？
是不是容易出现奇点？
是不是容易 overflow？
是不是只是函数全集凑数？
现有冷启动/论文/recipe有没有真实用到？
有没有更稳定变换可替代？
```

默认建议：

```text
tan/cot/sec/csc
→ DELETE / RESEARCH ONLY

cosh/sinh
→ 若没有实际 recipe，DELETE / INTERNAL

arg
→ 如果只是 complex utility，MOVE_INTERNAL
```

最终以：

```text
真实 call graph + recipe use + semantic review
```

决定。

---

# 11. raw matrix primitive 不公开

以下：

```text
fft
ifft
wavelet
convolve
correlate

mat_inverse
eig
svd
pca
```

不要作为公共 FactorEngine operator。

正确模式：

```text
internal kernel
→ 明确语义的 factor operator
```

例如：

```text
PCA helper
→ rolling PCA residual

FFT helper
→ trailing spectral concentration
```

---

# 12. Tombstone 必须完全不可调用

建议新增：

```text
factor_engine/compat/tombstones.py
```

数据结构：

```python
@dataclass(frozen=True)
class OperatorTombstone:
    name: str
    removed_reason: str
    risk_class: str
    replacement: str | None
    removed_since: str
    previous_semantic_hash: str | None
```

---

# 13. Tombstone invariant

必须：

```python
set(TOMBSTONES) & set(ProductionOperatorRegistry.canonicals()) == set()
set(TOMBSTONES) & set(ProductionOperatorRegistry.aliases()) == set()
```

---

# 14. Tombstone 不进入 backend

不得进入：

```text
Pandas registry
Polars registry
DuckDB SQL emitter
ClickHouse emitter
Numba kernel registry
composite lowering
```

---

# 15. Tombstone 不进入 mining

不得进入：

```text
AlphaProbe
AlphaMiner
FactorMiner
CogAlpha
AlphaSage
EvoAlpha
AlphaCFG
QuantaAlpha
cold-start
recipe search
```

---

# 16. Tombstone 不进入 DSL allowlist

parser遇到：

```text
已删除名字
```

先查 tombstone。

输出：

```text
明确 removed 原因
```

而不是：

```text
Unknown operator
```

---

# 17. safe alias 与 unsafe alias 分开

允许保留 executable compat alias 的唯一情况：

```text
canonical 本身 production-safe
alias 与 canonical 语义 100% 相同
alias 不产生第二个 mining candidate
```

---

# 18. research alias 不允许 public executable

当前：

```text
relation_pagerank_centrality
```

仍通过：

```text
OperatorRegistry.register_alias(...)
```

指向：

```text
group_signal_attraction_share
```

而这个算子本身是 research-only。

R29：

```text
把 relation_pagerank_centrality 从 public OperatorRegistry 删除。
```

如果需要兼容：

```text
Tombstone → group_signal_attraction_share (research only)
```

---

# 19. misleading legacy model alias 同理

例如：

```text
ts_ar_forecast
```

实际是：

```text
in-sample fitted value
```

虽然代码里已经有真正：

```text
ts_ar_prior_forecast
```

但旧名字极容易误导。

建议：

```text
production public registry移除旧 alias
```

保留 tombstone：

```text
ts_ar_forecast
→ removed misleading diagnostic alias
→ use ts_ar_prior_forecast
```

---

# 20. Default Production Loader 必须重写

当前一个 `_LOAD_MODULES`：

```text
把 production / experimental / research
全部装进一个 registry
```

这是 R29 的根问题。

---

# 21. 推荐 loader

```python
def load_production_operators():
    ...

def load_research_tools():
    ...

def load_internal_kernels():
    ...
```

---

# 22. 默认 import

```python
from factor_engine import FactorEngine
```

只触发：

```text
load_production_operators()
```

---

# 23. Research 只有显式 opt-in

例如：

```python
engine = FactorEngine(mode="research")
```

才：

```text
load_research_tools()
```

---

# 24. production process 中 research canonical 必须不存在

硬测试：

```python
assert "ts_dmd_dominant_growth_rate" not in ProductionOperatorRegistry
assert "ts_wavelet_lowpass_reconstruct" not in ProductionOperatorRegistry
```

不是：

```text
存在，但是 mining_eligible=False
```

---

# 25. 当前 research_transform 的问题

当前：

```text
research_transform.py
```

里的：

```text
ts_wavelet_lowpass_reconstruct
ts_signature_mahalanobis_anomaly
...
```

仍作为：

```text
register_operator
```

进入统一 cleaned registry。

虽然：

```text
default_search_weight=0
research surface
```

但直接 registry 用户还是可能拿到。

R29：

```text
MOVE_RESEARCH_ISOLATED
```

---

# 26. DMD / research_spectral 同理

研究型：

```text
DMD
bicoherence
kernel Granger
BDS
research spectral
```

不要默认注册进 production registry。

---

# 27. ts_model state_space / volatility

如果最终仍被定义为：

```text
research
```

则：

```text
默认 loader不 import
```

如果未来某个具体 canonical经过完整 certification：

```text
只把这个具体 canonical
迁到 production module
```

不要把整个 research module一起装入 production。

---

# 28. Production module 不能靠“注册后再过滤”

最佳模式：

```text
production module
只包含 production operator。
```

而不是：

```text
先全部 register
再依赖 policy过滤。
```

---

# 29. 这轮必须重新扫描全部 current canonical

本文件生成时 catalog：

```text
1362 canonicals
```

但真正执行 R29 时：

```text
不要硬编码1362。
```

必须 fresh process：

```text
git SHA
load current code
enumerate registry
```

得到：

```text
CURRENT_CANONICAL_SET
```

---

# 30. 为什么旧 audit不能证明 current all-audited

现有：

```text
current catalog = 1362

R19 = 1430
R23 = 1436
```

因此：

```text
必须重新逐 canonical生成 R29 inventory。
```

---

# 31. R29_OPERATOR_DISPOSITION_MATRIX

每一行：

```text
canonical
aliases
source_file
source_symbol

current_surface
current_status

current_public_callable
current_default_loaded

risk_family

actual_use_case
economic_meaning

time_semantics
PIT_status

input_contract
source_contract
market_contract
unit_contract

param_contract
missing_contract
state_contract

test_count

new_disposition
replacement

delete_reason
migration_action
```

---

# 32. 所有 canonical 必须 disposition

硬门：

```text
CURRENT_CANONICALS_WITHOUT_R29_DISPOSITION = 0
```

---

# 33. 不能只审 registered canonical

还要审：

```text
unregistered helper
dead private implementation
legacy compatibility function
unused wrapper
duplicate kernel
```

---

# 34. 新增 Dead Code Audit

建立：

```text
scripts/audit_r29_dead_code.py
```

输出：

```text
R29_DEAD_CODE_CANDIDATES.csv
```

---

# 35. Dead code分两类

## A. registered dead canonical

存在 registry：

```text
但没有合法 production/research use。
```

直接删。

## B. unregistered stale helper

例如旧实现已经被新 ID-matched实现替换，

但私有旧函数还在文件中。

如果 call graph证明无人调用：

```text
物理删除。
```

---

# 36. 当前 shareholder 文件就是例子

`shareholder/churn_network.py` 已经把公开注册改成：

```text
按 ShareholderId 跨期匹配
```

这是正确方向。

但文件里仍保留旧 slot-based private helper：

```text
_cur_prev_ranked
_holder_weighted_churn
_entry_share
_exit_share
_net_entry_share
```

它们仍会：

```text
按 rank slot
np.nan_to_num
```

做旧逻辑。

如果当前 call graph已无人调用：

```text
DELETE。
```

否则以后维护者很容易重新误接。

---

# 37. Dead helper 删除前必须做全 repo引用检查

扫描：

```text
imports
direct calls
getattr
string references
recipes
tests
notebooks/scripts
```

动态反射不可证明安全：

```text
先列 blocked
```

不要误删。

---

# 38. Universal MultiInputAlignmentGate

这是当前继续扫代码时发现的最大一类隐患之一。

只要 operator有：

```text
2个或以上 DataFrame panel输入
```

都必须先：

```text
检查完全一致的 index + columns。
```

---

# 39. 不允许“shape一样就直接 to_numpy”

错误：

```python
xv = x.to_numpy()
yv = y.to_numpy()
```

如果：

```text
x columns = [A, B, C]
y columns = [C, A, B]
```

shape完全一样，

但算出来全错。

---

# 40. Central alignment API

建议统一：

```python
align_panel_inputs_exact(
    *frames,
    names=...,
    allow_optional=True,
)
```

---

# 41. exact alignment默认行为

要求：

```text
index.equals
columns.equals
```

不允许：

```text
silent reindex
silent sort
silent intersection
```

---

# 42. 如果业务确实需要 as-of / reindex

必须：

```text
由 DataAccess / SourceAdapter 显式完成。
```

算子内部不猜。

---

# 43. 当前 dynamic_knn P0

当前：

```text
cs_knn_peer_mean_ex_self
cs_knn_neighbor_retention
cs_knn_graph_dirichlet_energy
```

直接：

```python
np.stack([f.to_numpy(...) for f in (f1, f2, f3)], axis=2)
```

没有先 exact align。

---

# 44. dynamic_knn 修复

所有：

```text
target
f1
f2
f3
```

先：

```text
align_panel_inputs_exact
```

---

# 45. dynamic_knn 测试

```text
f2 columns reorder
→ 必须 raise

f3 index shift one day
→ 必须 raise

same axes
→ 正常
```

---

# 46. 当前 event_response P0

`event_response.py`：

```text
response
event
```

直接：

```text
to_numpy
```

没有 DataFrame axis gate。

---

# 47. event_response 修复

进入 kernel前：

```text
response,event = align_panel_inputs_exact(...)
```

---

# 48. 当前 first_passage P0

```text
x
scale
```

直接：

```text
to_numpy
```

没有 alignment。

---

# 49. first_passage 修复

统一 exact alignment。

---

# 50. 当前 stateful.rotation P0

```text
x
group
```

当前：

```text
group.to_numpy
```

没有先确保：

```text
group与x是同一股票/日期轴。
```

必须修。

---

# 51. 当前 extrema_divergence P0

```text
x
y
```

直接：

```text
to_numpy
```

没有 strict alignment。

---

# 52. 当前 intraday_session P0

```text
x
session_id
```

当前只验证：

```text
session_id值是不是整数
```

但没有先验证：

```text
session_id axes == x axes。
```

必须修。

---

# 53. path_signature 同样统一

即使目前部分实现自己检查：

```text
也统一走 central gate。
```

减少不同模块各写一套。

---

# 54. Alignment AST Lint

新增：

```text
audit_r29_multi_input_alignment.py
```

扫描 public operator：

```text
有2+ panel参数
+
在 align之前出现 .to_numpy()
```

直接 flag。

---

# 55. 硬门

```text
PUBLIC_MULTI_PANEL_OPERATOR_WITHOUT_ALIGNMENT_GATE = 0
```

---

# 56. Strict Parameter Gate

第二大系统性隐患：

```text
metadata可能有 ParamSpec
但直接实例化 operator时仍可能绕过 binder。
```

---

# 57. 当前大量实现存在

例如：

```python
window = max(2, int(window))
lag = max(1, int(lag))
k = max(1, int(k))
alpha = max(0.0, float(alpha))
```

---

# 58. 为什么危险

用户输入：

```text
lag = 3.9
```

会静默：

```text
3
```

于是：

```text
3.1
3.9
3
```

可能变成同一因子。

---

# 59. 直接调用更危险

如果使用者绕开 parser：

```python
Operator().calculate(..., lag=-10)
```

有些 kernel：

```text
直接 clamp成1
```

而不是告诉用户错了。

---

# 60. 参数错误与数据缺失必须区分

参数错误：

```text
raise ValueError
```

数据缺失：

```text
NaN/fail closed
```

不要：

```text
错误参数 → NaN
```

因为使用者会以为数据不够。

---

# 61. 所有 public param必须同一 binder

建议：

```text
bind_operator_params()
```

由：

```text
public calculate
DSL
mining
direct registry
```

共同调用。

---

# 62. 不允许 kernel自行偷偷修参数

删除：

```text
max(1,int(...))
min(...)
silent clip
```

用于 contract 参数。

---

# 63. dynamic_knn 参数问题

当前：

```text
lag=max(1,int(lag))
```

需要：

```text
StrictInt
lag>=1
```

直接非法就 raise。

---

# 64. first_passage 参数问题

kernel中：

```text
w=max(2,int(window))
H=max(1,int(horizon))
ma=max(1,int(min_anchors))
```

必须改成严格 gate。

---

# 65. stateful.survival 参数问题

当前：

```text
history_window=max(2,int(...))
min_completed_runs=max(2,int(...))
alpha=max(0,float(alpha))
```

非法 direct call被静默纠正。

全部改。

---

# 66. state_slew_limit参数问题

当前：

```text
负 scalar limit
```

不会直接 raise，

而是循环里：

```text
output NaN
reset state
```

这是错误参数被伪装成数据缺失。

---

# 67. state_deadband 同类

负：

```text
band
```

应该：

```text
raise
```

---

# 68. threshold_cycle

当前：

```text
window=int(window)
```

`120.9` 也能跑。

改 strict int。

---

# 69. volume_clock

虽然 metadata：

```text
buckets choices=(8,16,32)
```

但 runtime：

```text
b=max(4,int(buckets))
```

直接调用可以绕过。

必须 runtime也使用同一 binder。

---

# 70. session_recovery

metadata：

```text
min_events >= 3
```

但 runtime：

```text
me=max(1,int(min_events))
```

直接调用：

```text
min_events=1
```

仍然可以跑。

这正是“绕过治理直接调用”的现实例子。

R29必须封死。

---

# 71. Unknown kwargs

很多：

```python
def _calculate_series(..., **_: Any)
```

会吞未知参数。

public operator：

```text
unknown parameter
```

必须：

```text
raise。
```

---

# 72. 只允许内部 wrapper吞内部 kwargs

如果 backend内部需要：

```text
execution context kwargs
```

必须有：

```text
reserved namespace
```

而不是任意吞。

---

# 73. Parameter hard gate

```text
PUBLIC_OPERATOR_SILENT_PARAM_COERCION = 0
PUBLIC_OPERATOR_UNKNOWN_KWARG_SWALLOWED = 0
```

---

# 74. 新发现：stateful survival 状态机 bug

当前：

```text
stateful/survival.py
```

处理 active episode 遇到 NaN 时：

```text
gap_censored=True
prev_active=False
prev_finite=False
```

但：

```text
cur没有清零。
```

---

# 75. bug路径

例如：

```text
ACTIVE
ACTIVE
NaN
INACTIVE
ACTIVE
```

NaN后：

```text
cur仍然保留旧 run age。
```

到了 INACTIVE：

```text
prev_active=False
```

所以：

```text
if prev_active:
    cur=0
```

不会执行。

随后下一次 ACTIVE：

```text
cur != 0
```

会继续：

```text
cur += 1
```

---

# 76. 后果

新的 episode 可能继承：

```text
gap之前旧 episode的年龄。
```

导致：

```text
ts_state_age_percentile
ts_state_exit_hazard
ts_state_residual_life
```

时序错误。

---

# 77. survival 修复

不要继续靠：

```text
prev_active
prev_finite
cur
gap_censored
```

松散组合。

建议显式：

```python
class EpisodeObservationState(Enum):
    INACTIVE
    ACTIVE
    UNKNOWN

@dataclass
class EpisodeMemory:
    current_age
    entry_observed
    gap_censored
    ...
```

---

# 78. 明确 transition table

至少：

```text
INACTIVE -> ACTIVE
ACTIVE -> ACTIVE
ACTIVE -> INACTIVE
ACTIVE -> UNKNOWN
UNKNOWN -> ACTIVE
UNKNOWN -> INACTIVE
UNKNOWN -> UNKNOWN
```

每条都有：

```text
age
completed-run update
censor flag
output
```

定义。

---

# 79. gap后 finite inactive

必须：

```text
强制丢弃 censored old run
cur=0
```

---

# 80. gap后 active

如果：

```text
真实 entry不可观察
```

新 run：

```text
left-censored
```

不能沿用旧 age。

---

# 81. survival 专项 golden tests

```text
0,1,1,0
0,1,NaN,1,0
0,1,NaN,0,1
1,1,0
NaN,1,1,0
0,1,1,NaN,0,1
```

逐行 hand expected。

---

# 82. survival 文档也要修

当前 class docstring仍存在：

```text
inactive output 0
```

而默认：

```text
inactive_policy="nan"
```

文档/默认语义漂移。

一并修。

---

# 83. StateMachineGate 推广到所有 stateful

覆盖：

```text
stateful.rule_language
stateful.events
stateful.sequential
stateful.episode
stateful.survival
stateful.rotation
stateful.drawdown_path
directional_change
markov_dynamics
threshold_cycle
```

---

# 84. 每个 stateful operator必须声明

```text
state variables
initialization
missing policy
reset policy
checkpoint policy
chunk policy
left-censor semantics
right-censor semantics
```

---

# 85. full-history != checkpoint

如果：

```text
required_full_history
```

就不要宣称：

```text
incremental checkpoint支持。
```

当前 survival 对此已经较诚实。

继续保持。

---

# 86. 新发现：fin_component_score FALSE 与 MISSING混淆

当前：

```text
fin_component_score
```

里的 `_score_component()`：

对：

```text
up
```

只有：

```text
finite & >0
```

返回1，

其余包括：

```text
finite <=0
```

返回 NaN。

---

# 87. 这是错误的 score 语义

Piotroski-style criterion：

```text
条件成立 → 1
条件不成立 → 0
数据缺失 → NaN
```

当前却：

```text
条件不成立 → NaN
```

---

# 88. 直接后果

如果：

```text
所有组件都有数据
但所有条件都 false
```

当前：

```text
contributions全部NaN
any_finite=False
score=NaN
```

正确应该：

```text
score=0
```

---

# 89. component score修复

例如：

```python
if not finite:
    NaN
elif condition:
    1
else:
    0
```

---

# 90. partial missing policy也要明确

两种可选：

### A sum-available

```text
缺失组件不计
```

需要附带：

```text
observed_component_count
```

避免不同 completeness不可比。

### B require-full

有一个 missing：

```text
score NaN
```

用于严格 score。

不要隐式。

---

# 91. Boolean / Missing / Unknown 三值体系

所有条件类统一：

```text
TRUE = 1
FALSE = 0
UNKNOWN = NaN
```

不允许：

```text
FALSE -> NaN
NaN -> False
```

---

# 92. 这个 gate还要覆盖

```text
candlestick
event
state condition
limit condition
component score
filter mask
applicability mask
```

---

# 93. 新发现：composition 经济语义 gate仍然不够

当前 `composition.py` 已经做了：

```text
正值
同 family
CompositionSchema
PartId
```

等很多修正。

但：

```text
_COMPOSITION_SCHEMA_FIELDS
```

仍把：

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

统一归入：

```text
financial_statement
```

---

# 94. 这在经济上仍不成立

例如：

```text
Revenue
Net Income
Total Assets
```

不是同一个 total 的互斥组成部分。

---

# 95. Flow / Stock也混了

```text
revenue
net income
```

是期间流量。

```text
assets
liabilities
equity
```

是期末存量。

不能因为：

```text
都是钱
```

就做 Aitchison composition。

---

# 96. assets/liabilities/equity还存在恒等关系

```text
Assets = Liabilities + Equity
```

如果把：

```text
Assets
Liabilities
Equity
```

同时作为组成部分，

等于：

```text
whole + components一起放入 composition。
```

数学能算，

经济意义错误。

---

# 97. 新增 EconomicMeaningGate

不再只看：

```text
unit
family
```

必须有：

```python
PartOfWholeContract(
    composition_id=...,
    total_concept=...,
    part_ids=...,
    mutually_exclusive=True,
    stock_or_flow=...,
    fiscal_period_semantics=...,
    currency_semantics=...,
)
```

---

# 98. Composition合法条件

至少：

```text
同一个 total_concept
同一个 stock/flow语义
同一个 period
同一个 currency/unit
parts互斥
不能重复 whole
严格正值
```

---

# 99. 如果数据没有合法 composition

不要为了保留算子：

```text
允许任意财务字段。
```

直接：

```text
MOVE_RESEARCH_ISOLATED
```

或者：

```text
DELETE_PUBLIC_OPERATOR
```

---

# 100. Composition可保留的真实例子

只有数据契约明确时：

```text
收入分业务构成
成本分项
资产类别分项
股东持股构成
成交量按时段 share
订单流按方向/类型 share
```

---

# 101. EconomicMeaningGate 不只用于 composition

还要用于：

```text
ratio
spread
weighted average
normalization
score
distance
regression
```

---

# 102. 数学合法 != 经济合法

R29新增原则：

```text
MathematicallyExecutable
!=
QuantitativelyMeaningful
```

---

# 103. 新发现：first_passage 文档语义冲突

实现中：

```text
未触 barrier 且 full observed 的 anchor
```

加入：

```text
0
```

所以 bias本质：

```text
方向 × 速度 × 命中概率
```

---

# 104. 但同文件后续文档又写

```text
bias只对已命中锚点求方向×速度均值
```

这与实现冲突。

---

# 105. 修复

确定唯一正式定义。

推荐保留实现当前逻辑：

```text
unconditional over fully observed anchors
```

因为：

```text
未命中=0
```

有清晰概率含义。

然后全文件：

```text
统一文档
测试
metadata
recipe docs。
```

---

# 106. first_passage unit contract还有隐患

当前 metadata类似：

```text
x = price_or_log_price
scale = volatility_with_same_unit_as_x
```

又允许：

```text
return_volatility
```

但：

```text
raw price + return volatility
```

量纲不一致。

---

# 107. relational unit contract

必须：

```text
if x=log_price:
    scale=log_return_volatility 可合法

if x=return/return index:
    对应同单位 scale

if x=raw price:
    scale必须是 price-unit volatility
```

不要靠：

```text
泛化字符串
```

---

# 108. 新增 UnitRelationSpec

类似：

```python
UnitRelationSpec(
    expression="unit(scale) == unit(x)"
)
```

或：

```text
scale semantic transform can map return-vol to log-price increment
```

必须显式。

---

# 109. Source Provenance Gate

这是阻止“别人随便拿 DataFrame喂进去”的关键。

现在一些算子：

```text
数学 kernel本身 PIT
```

但：

```text
输入 DataFrame是不是 PIT
```

它不知道。

---

# 110. fundamental expectation 是典型例子

当前代码已经正确意识到：

```text
fin_surprise
```

需要：

```text
PreEventExpectationSnapshot
```

revision需要：

```text
ConsensusVintageSource
```

---

# 111. 但 direct call仍可能绕过

如果有人：

```python
op.calculate(actual_today_db, consensus_today_db)
```

算子本身无法证明：

```text
consensus是历史时点当时可见版本。
```

---

# 112. metadata tag不够

```text
requires:ConsensusVintageSource
```

只有 admission层看它才有效。

绕开 admission：

```text
仍能算。
```

---

# 113. 新增 DataProvenanceContext

production source-sensitive operator必须接收：

```python
DataProvenanceContext(
    dataset_id,
    snapshot_id,
    market,
    grain,
    knowledge_clock,
    effective_clock,
    vintage_support,
    session_calendar,
    field_concepts,
    units,
)
```

---

# 114. anonymous DataFrame默认不能跑 contextual PIT算子

例如 production API：

```python
engine.run(fin_surprise, anonymous_dataframe)
```

应该：

```text
SourceProvenanceError
```

---

# 115. 研究模式可以放宽

```python
engine = FactorEngine(mode="research", allow_anonymous_panels=True)
```

可以。

但输出：

```text
not production certified
```

---

# 116. 需要 provenance gate 的 family

至少：

```text
fundamental
consensus
shareholder
relation
index membership
listing status
event source
intraday session
A-share limit/status
benchmark/index composition
```

---

# 117. 当前 fundamental expectation 状态建议

在真实：

```text
ConsensusVintageSource
```

没有被 DataAccess证明以前：

```text
KEEP_PUBLIC_CONTEXTUAL
```

而不是：

```text
普通 public ALPHA。
```

---

# 118. pre-event consensus

必须证明：

```text
expectation snapshot timestamp
<
actual announcement knowledge timestamp
```

同日：

```text
不能只看 date label。
```

---

# 119. financial revision

必须：

```text
历史 revision vintage
```

而不是：

```text
数据库今天最后版本回填历史。
```

---

# 120. shareholder

必须：

```text
PubDate / availability
```

做 as-of。

不能：

```text
按 report end date回填。
```

---

# 121. relation

必须：

```text
edge validity interval
edge source
edge target
weight
direction
knowledge time
```

---

# 122. 当前 fake PageRank处理

当前：

```text
group_signal_attraction_share
```

已经诚实承认：

```text
没有真正 PIT graph
```

继续：

```text
research isolated
```

不要 public executable alias。

---

# 123. SessionClockGate

继续扫 intraday 后发现：

```text
不同模块仍有不同 session 处理方式。
```

必须统一。

---

# 124. volume_clock 当前问题

当前 `_volume_clock_daily_agg`：

```text
joined.index.normalize()
```

按 calendar day分组。

---

# 125. 风险一：timezone

如果 index储存：

```text
UTC
```

normalize以后按 UTC day，

不一定是交易所 session day。

---

# 126. 风险二：缺失 timestamp不可见

如果某分钟：

```text
整行根本不存在
```

当前只检查：

```text
现存行是否 finite
```

无法知道：

```text
官方 grid缺了一格。
```

---

# 127. volume_clock修复

先：

```text
DataAccess / runtime SessionPanel
```

重建 official slots。

再：

```text
缺失 slot显式 NaN
```

最后：

```text
volume clock。
```

---

# 128. EOD factor必须 session complete

volume clock：

```text
available_at=session_close
```

只有：

```text
official close已观测
完整 session contract通过
```

才输出。

---

# 129. session_recovery 当前跨市场 bug

代码已有：

```text
US timezone resolution
```

但 `_calculate_series` 后面仍：

```python
build_session_panel(... market="ashare")
```

硬编码 A-share。

---

# 130. 后果

即使调用者给：

```text
US calendar
America/New_York
```

实际 session panel仍宣称：

```text
market="ashare"
```

这是跨市场语义风险。

---

# 131. session_recovery修复

market必须来自：

```text
ExecutionContext
calendar.market
DataProvenanceContext.market
```

不允许硬编码。

---

# 132. 如果 operator本来只支持 A股

那就诚实：

```text
A_SHARE_ONLY
```

US直接：

```text
MarketNotSupportedError
```

不要做半通用。

---

# 133. intraday_session timezone风险

当前：

```python
x.index.to_numpy(dtype="datetime64[ns]")
```

然后直接算：

```text
date
minute_of_day
```

---

# 134. tz-aware index风险

timezone在转 NumPy时：

```text
可能被转成 UTC-naive representation。
```

之后拿它和：

```text
本地交易所 official slot
```

比较可能错。

---

# 135. intraday_session修复

统一：

```text
SessionClockAdapter
```

先：

```text
convert to session-local clock
```

再构造：

```text
trade_date
official_slot
session_id
```

---

# 136. session_id也要 exact align

前面提到：

```text
x / session_id
```

必须同轴。

---

# 137. 一个统一 SessionPanel

推荐所有 intraday→daily operator都消费：

```python
SessionPanel(
    market,
    trade_date,
    session_tz,
    official_slots,
    values,
    validity_mask,
    close_reached,
)
```

---

# 138. 不允许每个 operator自己猜 session

禁止：

```text
normalize()
observed modal frequency
裸 date转换
默认 Asia/Shanghai
```

---

# 139. availability time

所有 minute→daily：

```text
available_at = session_close
```

---

# 140. mid-session不能误用

如果 FactorEngine目标频率：

```text
minute realtime
```

EOD operator：

```text
不可达。
```

---

# 141. Confirmed Pivot / Pattern Gate

结构形态类是未来函数高风险区。

当前 `structural_levels.py` 方向是比较好的：

```text
pivot k
只有到 k+confirmation
才可用。
```

---

# 142. R29 不要求删 confirmed pivot

只要：

```text
信号 timestamp = confirmation time
```

它就是 causal delayed information。

---

# 143. 禁止 backdate

绝对不能：

```text
在 k+confirmation 才知道 pivot
却把输出回写到 k。
```

---

# 144. 全 pivot family统一测试

覆盖：

```text
structural_levels
extrema_divergence
pivot high/low
support/resistance
double top/bottom
head & shoulders
triangle
wedge
flag
```

---

# 145. Future Perturbation Pivot Test

固定 cutoff t：

```text
改变 t+1以后价格
```

要求：

```text
<=t所有 published pivot-derived output不变。
```

---

# 146. confirmation row golden

构造：

```text
明显 local max
confirmation=3
```

要求：

```text
pivot day
t+1
t+2
→ 不可用

t+3
→ 首次可用
```

---

# 147. extrema_divergence 当前 alignment问题

除 pivot语义外：

```text
x/y
```

也要先 align。

---

# 148. extrema参数strict问题

当前：

```text
int(window)
int(confirmation)
int(match_lag)
```

对 fractional会静默截断。

统一 strict binder。

---

# 149. Event Response Gate

当前 event-response的成熟 horizon设计：

```text
s + H <= t
```

总体方向正确。

---

# 150. 但必须 exact align

```text
response / event
```

先对齐。

---

# 151. Event response source semantics

如果：

```text
response本身是未来H日收益
```

不能直接喂。

operator定义的 response路径应该是：

```text
逐日 realized response series
```

然后 kernel自己从历史 event后切片。

---

# 152. label semantic gate

如果输入 semantic type：

```text
ForwardLabel
```

直接禁止作为 response factor input。

---

# 153. Event cohort maturity

所有事件：

```text
完整 H-path成熟
```

才进入历史 cohort。

当前部分实现已有，

统一测试。

---

# 154. right censoring

近期事件：

```text
不足 H bars
```

不能：

```text
当作0 response。
```

---

# 155. missing response path

必须：

```text
exclude/censor
```

不能：

```text
drop NaN后缩短 horizon。
```

---

# 156. Markov dynamics

当前 `markov_dynamics.py` 已经采用：

```text
严格过去 [t-W,t-1]
```

估计 transition，

当前值只选 state。

这是正确模板。

---

# 157. Markov 还要验证

```text
quantile edges
stationary distribution
missing gap
state bins
Jeffreys prior
minimum transitions
chunk/full equivalence
```

---

# 158. state binning不能用当前值构造 edge

当前方向：

```text
past only
```

保持。

---

# 159. stationary distribution

必须：

```text
P基于历史
```

不能：

```text
含 current transition。
```

---

# 160. Dynamic KNN

同日 cross-section：

```text
不是未来信息。
```

但需要：

```text
as-of universe
as-of style fields
axis alignment
tie invariance。
```

---

# 161. KNN tie handling

当前使用：

```text
average tie rank
tie-inclusive kth radius
```

方向合理。

---

# 162. KNN membership

如果 feature来自：

```text
industry/index/fundamental
```

必须：

```text
as-of。
```

---

# 163. kNN output role

例如：

```text
peer mean
```

本身通常：

```text
INTERMEDIATE / GROUP-RELATIVE BUILDING BLOCK
```

而：

```text
target - peer_mean
```

才是更自然 alpha recipe。

逐个 role再审。

---

# 164. Activity Clock

当前 `activity_clock.py` 有 exact axis gate，

这是应当推广的模板。

---

# 165. current-inclusive lagged value

```text
ts_activity_clock_lagged_value
```

允许 current bar参与 budget。

在高 activity日可能：

```text
k=0
lagged_value=x_t
```

---

# 166. 如果用于 momentum

```text
x - lagged_value
```

会在 shock日变成0。

所以当前已有：

```text
_prior
```

版本更适合 momentum。

---

# 167. Role建议

current-inclusive：

```text
INTERMEDIATE / STATE
```

不要默认被 LLM当“过去值”。

prior版本：

```text
才允许 strict prior recipe。
```

---

# 168. Financial Component Score

除 FALSE/MISSING bug外，

还需要：

```text
组件之间经济定义
方向
权重
```

合法。

---

# 169. generic score不要允许任意 panel随便拼

如果 LLM把：

```text
price momentum
volume zscore
PE
```

都扔进：

```text
fin_component_score
```

名字就不成立。

---

# 170. 两种方案

### A
限制：

```text
fin_component_score
```

只能 financial component typed inputs。

### B
改成：

```text
generic_condition_score
```

并明确不是 fundamental。

不要名字与用途错位。

---

# 171. Research 与 Production 的语义隔离

研究工具可以：

```text
计算 retrospective statistic
做诊断
做学术分析
```

但不要和 production operators混一个 registry。

---

# 172. research mode输出必须带标签

建议返回 metadata：

```text
execution_mode=research
production_eligible=False
```

---

# 173. 不能 materialize成 production factor lake

research operator输出：

```text
默认不允许写 production factor namespace。
```

---

# 174. Data lake namespace隔离

例如：

```text
factors/production/
factors/research/
```

避免下游误读。

---

# 175. Production Public API

普通用户只看到：

```python
engine.list_operators()
```

返回：

```text
production-safe/public roles。
```

---

# 176. 研究列表单独

```python
engine.list_research_tools()
```

只有 research mode。

---

# 177. registry不要直接 export为公共主入口

如果：

```text
OperatorRegistry
```

仍然可以任意 get，

用户仍然可能绕开。

---

# 178. 推荐把 raw registry变 internal

公开：

```text
OperatorCatalog
FactorEngine
```

内部：

```text
_InternalOperatorRegistry
```

---

# 179. 如果必须保留兼容

`OperatorRegistry.get()`：

```text
默认 enforcement="production"
```

research必须：

```python
get(..., mode="research")
```

---

# 180. Direct-Call Safety Gate

所有调用路径：

```text
DSL
FactorEngine.run
registry.get
operator.calculate
backend compile
recipe execution
materialization
```

最终都不能绕过：

```text
same safety authority。
```

---

# 181. 这叫 defense in depth

即使一层 bug：

```text
下一层仍能拒绝。
```

---

# 182. Public execution context

建议统一：

```python
ExecutionContext(
    mode="production",
    market="ashare",
    frequency="daily",
    provenance=...,
    calendar=...,
    available_at=...,
)
```

---

# 183. contextual operator没有 context

直接：

```text
ContextRequiredError
```

---

# 184. production factor不是裸函数库

这是关键设计思想。

FactorEngine是：

```text
有时间/市场/数据语义的计算引擎。
```

不能退化成：

```text
“给我几个 DataFrame我什么都算”。
```

---

# 185. 继续审计所有剩余 operator family

R29执行脚本必须把当前所有 module按 family分桶：

```text
common elementwise
common rolling
cross-sectional
group
technical
price-volume
candlestick
structure/pattern
A-share
microstructure
intraday
fundamental
valuation
shareholder
index/listing
relation
event
stateful
regression/model
distribution/tail
information theory
spectral
geometry/topology
composition
research
unsafe
legacy/internal
```

---

# 186. 每个 family有专门审计 checklist

不能只跑一个 smoke。

---

# 187. Common math checklist

```text
domain
zero division
overflow
underflow
NaN
inf
dtype
unit transform
parameter identity
```

---

# 188. Rolling checklist

```text
trailing only
min periods
physical gap
window
ddof
tie
current-inclusive/prior
```

---

# 189. Cross-sectional checklist

```text
same-day only
universe
column permutation
tie
group membership
broadcast vs per-stock
```

---

# 190. Technical checklist

```text
OHLC contract
warmup
TA-Lib fallback parity
recursive init
gap behavior
parameter strictness
```

---

# 191. Candlestick checklist

```text
NaN != false
pattern sign
OHLC validity
gap
binary/event role
```

---

# 192. Pattern checklist

```text
confirmation delay
backdating
stale pivot
state reset
overlap
```

---

# 193. Intraday checklist

```text
calendar
timezone
official slots
auction
lunch
half day
missing timestamp
duplicate timestamp
bar label convention
session close
```

---

# 194. Fundamental checklist

```text
knowledge date
period date
revision vintage
flow/stock
TTM
quarter
currency
restatement
consensus vintage
```

---

# 195. Shareholder checklist

```text
PubDate
ShareholderId
top-K disclosure semantics
duplicate ID
missing ratio
previous snapshot
```

---

# 196. Relation checklist

```text
real edge
direction
weight
validity interval
knowledge time
peer exclusion
```

---

# 197. Stateful checklist

```text
transition table
unknown state
left censor
right censor
checkpoint
chunk
state reset
```

---

# 198. Model checklist

沿用 R28：

```text
walk-forward
fit cutoff
label maturity
scaler
PCA
hyperparams
purge
embargo
filter vs smoother
```

---

# 199. Spectral checklist

```text
trailing window
no centered filter
frequency grid
window length
gap
demean/detrend
normalization
```

---

# 200. Topology checklist

```text
distance metric
scale
sample floor
numerical stability
current query exclusion
```

---

# 201. Composition checklist

```text
real part-of-whole
stock/flow
same period
mutually exclusive
positive
unit/currency
```

---

# 202. Information theoretic checklist

```text
estimator bias
finite sample
binning
kernel bandwidth
surrogate
normalization
causality
```

---

# 203. Model-like算子不要只按路径识别

代码扫描：

```text
lstsq
polyfit
svd
eig
pinv
optimize
kernel
nearest neighbors
quantile regression
covariance inversion
state filtering
```

---

# 204. Economic meaningless自动扫描做不到

这一部分必须：

```text
family-level人工规则 + typed contract
```

而不是纯 AST。

---

# 205. 全算子用例证明

每个保留 canonical必须回答：

```text
这个算子在量化里为什么存在？
```

如果回答只是：

```text
“可以算”
```

不够。

---

# 206. RetentionReason

建议必填：

```text
retention_reason
typical_use
legal_role
```

---

# 207. DELETE_USELESS criteria

任一：

```text
没有明确用途
永远全 NaN
默认参数不可运行
输出永远常数
和其他 canonical exact duplicate
只有 raw utility
没有合法数据源
经济语义错误
未来函数
随机 factor
无法 PIT
严重误导且没有必要兼容
```

删除。

---

# 208. 删除不是失败

目标不是：

```text
算子数量越多越好。
```

而是：

```text
有效搜索空间越好。
```

---

# 209. 1000个真算子优于1362个混杂算子

自动 factor mining尤其如此。

垃圾 operator会：

```text
扩大无效搜索
浪费算力
增加过拟合
污染 LLM语义
制造假多样性
```

---

# 210. Exact Duplicate Audit

建立：

```text
implementation hash
semantic hash
parameter mapping
output contract
```

---

# 211. duplicate

保留：

```text
一个 canonical
```

其余：

```text
安全 exact alias
或 tombstone
```

---

# 212. misleading duplicate

如果旧名会误导：

```text
不要 executable alias
```

只 tombstone。

---

# 213. Randomized internal estimator

如果某 research statistic内部需要：

```text
surrogate
```

可以有 RNG。

但：

```text
固定 seed
local generator
不读 global RNG
```

---

# 214. production deterministic invariant

```text
PUBLIC_PRODUCTION_NONDETERMINISTIC = 0
```

---

# 215. global RNG scan

扫描：

```text
np.random
random
default_rng
RandomState
```

每处必须 disposition。

---

# 216. obvious future scan

本轮初步静态 search没有发现：

```text
shift(-
center=True
filtfilt
```

这只是好现象，

不是 PIT证明。

---

# 217. semantic future仍可能存在

例如：

```text
当前输出用全样本标准差
全样本PCA
smoothed state
未成熟label
确认后回填过去
```

代码里不一定有“future”字样。

---

# 218. 所以必须行为 causality oracle

沿用 R28：

```text
prefix invariance
future perturbation
```

---

# 219. Public operator直接调用也测 causality

不要只通过：

```text
FactorEngine planner
```

测试。

还要：

```text
operator.calculate
```

本身安全。

---

# 220. Per-Canonical Direct Execution Test

每个保留 public canonical：

```text
直接调用
```

也要走：

```text
alignment
params
context
PIT
```

门禁。

---

# 221. Removed operator negative test

每个 tombstone：

```text
registry.get -> fail
parse_expr -> RemovedOperatorError
backend compile -> fail
mining discovery -> absent
recipe -> fail migration
```

---

# 222. Research isolation test

fresh production process：

```text
research canonical count in ProductionRegistry = 0
```

---

# 223. Unsafe isolation test

fresh production process：

```text
unsafe canonical count = 0
```

---

# 224. Legacy executable test

```text
legacy executable count = 0
```

安全同义 alias除外，

但不应该标成 legacy operator。

---

# 225. denied executable test

```text
denied executable count = 0
```

---

# 226. public registry surface应该只有正向类别

例如：

```text
production
contextual
state
condition
event
intermediate
```

---

# 227. public catalog不要显示“你可以拿到但不能用”

普通 `list_operators()`：

```text
只列可用的。
```

---

# 228. admin audit catalog可以列 tombstones

单独：

```text
list_removed_operators()
```

---

# 229. Migration策略

物理删除前要检查：

```text
冷启动公式
已保存 factor AST
materialized factor metadata
实验配置
benchmark
测试
notebook
其他算法引用
```

---

# 230. 不自动改 noncausal公式

例如：

```text
Lead(x,1)
```

不能静默改成：

```text
Delay(x,1)
```

两者意义完全不同。

应该：

```text
fail + 人工迁移。
```

---

# 231. duplicate safe migration

如果：

```text
旧名和新名完全 identical
```

可以：

```text
migration map
```

---

# 232. semantic-changing migration

必须：

```text
new semantic digest
```

不能：

```text
沿用旧 factor ID。
```

---

# 233. Materialized历史因子

旧危险 factor已 materialize：

```text
不要删历史 parquet本身
```

但标：

```text
recomputable=false
deprecated_reason=...
```

---

# 234. production新任务不能再引用

hard gate。

---

# 235. R29证据目录

```text
factor_engine/docs/evidence/r29/
```

---

# 236. 必须 tracked

当前 `.gitignore` 会忽略：

```text
output/
logs/
*.log
.coverage
htmlcov/
```

所以不要把关键证据放那里。

---

# 237. R29 artifacts

至少：

```text
R29_CURRENT_OPERATOR_INVENTORY.csv
R29_CURRENT_OPERATOR_INVENTORY.json

R29_OPERATOR_DISPOSITION_MATRIX.csv
R29_OPERATOR_DISPOSITION_MATRIX.json

R29_PHYSICAL_DELETION_MANIFEST.csv
R29_PHYSICAL_DELETION_MANIFEST.json

R29_TOMBSTONE_MANIFEST.json

R29_PUBLIC_REGISTRY_SNAPSHOT.json
R29_RESEARCH_REGISTRY_SNAPSHOT.json

R29_DEFAULT_LOADER_AUDIT.json

R29_DEAD_CODE_CANDIDATES.csv
R29_DEAD_CODE_RESOLUTION.csv

R29_MULTI_INPUT_ALIGNMENT_AUDIT.csv
R29_PARAMETER_STRICTNESS_AUDIT.csv

R29_ECONOMIC_MEANING_AUDIT.csv
R29_COMPOSITION_CONTRACT_AUDIT.csv

R29_SOURCE_PROVENANCE_AUDIT.csv
R29_SESSION_CLOCK_AUDIT.csv

R29_STATE_MACHINE_AUDIT.csv
R29_SURVIVAL_TRANSITION_GOLDEN.json

R29_BOOLEAN_MISSING_SEMANTICS.csv

R29_MODEL_CAUSALITY_CURRENT.csv
R29_FUTURE_PERTURBATION_CURRENT.csv

R29_PER_CANONICAL_TEST_COVERAGE.csv
R29_PER_CANONICAL_TEST_RESULTS.csv

R29_DIRECT_CALL_BYPASS_TESTS.json
R29_MINER_EXPOSURE_TESTS.json

R29_PYTEST_JUNIT.xml
R29_ARTIFACT_MANIFEST.json
R29_FINAL_ACCEPTANCE_REPORT.md
```

---

# 238. Evidence绑定当前 SHA

每个 artifact：

```text
git_sha
canonical_set_digest
production_registry_digest
research_registry_digest
test_suite_digest
generated_at
```

---

# 239. 删除后 canonical数量可以下降

最终报告必须：

```text
before
after
deleted
moved research
moved internal
tombstones
```

---

# 240. 不要为了维持1362而保留垃圾

数量下降是允许且可能应该的。

---

# 241. R29 tests目录

建议：

```text
factor_engine/tests/operators/r29/
```

---

# 242. 核心测试文件

```text
test_default_loader_only_production.py
test_removed_operators_are_non_callable.py
test_tombstones_not_registered.py
test_research_not_default_loaded.py
test_unsafe_not_default_loaded.py
test_legacy_not_executable.py

test_all_public_multi_inputs_align.py
test_direct_call_parameter_gate.py
test_unknown_kwargs_rejected.py

test_dynamic_knn_alignment.py
test_event_response_alignment.py
test_first_passage_alignment.py
test_rotation_group_alignment.py
test_extrema_alignment.py
test_intraday_session_alignment.py

test_survival_state_machine_transition_table.py
test_component_score_false_vs_missing.py

test_composition_economic_contract.py

test_source_provenance_required.py
test_consensus_vintage_required.py

test_session_recovery_market_context.py
test_volume_clock_official_session.py
test_intraday_session_timezone.py

test_pivot_confirmation_not_backdated.py

test_direct_registry_bypass.py
test_all_miners_cannot_see_removed.py

test_every_retained_canonical_executes.py
test_every_retained_canonical_has_semantic_evidence.py
```

---

# 243. 全 miner exposure

对：

```text
AlphaProbe
AlphaMiner
FactorMiner
CogAlpha
AlphaSage
EvoAlpha
AlphaCFG
QuantaAlpha
```

分别验证：

```text
removed/research/unsafe count = 0
```

---

# 244. 不只 miner

对：

```text
FactorEngine public API
DSL parser
raw registry compatibility API
backend planner
materializer
```

也验证。

---

# 245. Public API bypass matrix

建立：

```text
name
FactorEngine.run
parse_expr
OperatorRegistry.get
backend.compile
recipe.compile
materialize
miner.discovery
```

危险名字：

```text
全部 blocked。
```

---

# 246. 这比 denied list更可靠

因为它证明：

```text
真实调用路径。
```

---

# 247. Universal Direct Safety Contract

一个 public operator即使：

```text
被人直接实例化
```

也必须：

```text
参数严格
axes严格
source/context严格（需要时）
```

---

# 248. 不要把安全全部放 planner

planner不是唯一入口。

---

# 249. 代码结构建议

新增：

```text
factor_engine/operators/production_loader.py
factor_engine/operators/research_loader.py
factor_engine/operators/tombstones.py
factor_engine/operators/public_registry.py

factor_engine/contracts/panel_alignment.py
factor_engine/contracts/parameter_binding.py
factor_engine/contracts/data_provenance.py
factor_engine/contracts/economic_semantics.py
factor_engine/contracts/session_context.py
```

---

# 250. 不一定要大迁目录

如果重构成本太大，

至少先：

```text
loader物理分离
registry物理分离
```

这是 P0。

---

# 251. 当前新增问题汇总

本轮相对 R28 新明确发现至少：

```text
R29-P0-001 default production loader加载research/experimental模块

R29-P0-002 dynamic_knn多输入未对齐
R29-P0-003 event_response多输入未对齐
R29-P0-004 first_passage x/scale未对齐
R29-P0-005 stateful.rotation x/group未对齐
R29-P0-006 extrema_divergence x/y未对齐
R29-P0-007 intraday_session x/session_id未对齐

R29-P0-008 stateful.survival gap→inactive→active 状态年龄串线

R29-P0-009 fin_component_score FALSE 被当 MISSING

R29-P0-010 composition broad financial_statement family经济语义无效

R29-P0-011 session_recovery generic代码硬编码 market="ashare"

R29-P0-012 volume_clock没有统一 official SessionPanel，
               calendar-day grouping无法发现物理缺失slot

R29-P0-013 intraday_session tz-aware index直接转numpy后本地session时钟风险

R29-P0-014 source-sensitive fundamental operator可被anonymous DataFrame直接调用

R29-P0-015 research alias仍可注册到public OperatorRegistry

R29-P0-016 default_search_weight=0 / research surface并不构成物理隔离

R29-P1-017 first_passage实现/文档统计定义冲突
R29-P1-018 first_passage x-scale relational unit contract不足

R29-P1-019 大量direct-call参数silent int/clamp
R29-P1-020 unknown kwargs可能被**_吞掉

R29-P1-021 shareholder旧slot helper疑似dead code仍留在production module
```

---

# 252. R28仍继续有效的问题

包括：

```text
path_signature current missing fallback stale run
DMD log前 overflow/underflow
model walk-forward
label maturity
full evidence mismatch
per-canonical tests
```

R29不能把这些忘掉。

---

# 253. R29执行优先级

## P0-A：先缩 runtime

```text
拆 production/research loader
拆 registry
删 forbidden executable
建立 tombstone
```

---

# 254. P0-B：封 direct call

```text
parameter gate
alignment gate
context gate
```

---

# 255. P0-C：修 current concrete bugs

```text
survival state machine
component score
session recovery market
composition economic contract
```

---

# 256. P0-D：全 canonical重新 disposition

fresh current set。

---

# 257. P0-E：全 retained真实测试

包括：

```text
direct-call tests。
```

---

# 258. P1：dead code / duplicates / docs

在 P0稳定后做。

---

# 259. 推荐具体执行阶段

```text
Phase 1  Fresh Inventory
Phase 2  Public Runtime Split
Phase 3  Forbidden Physical Deletion
Phase 4  Tombstone Migration
Phase 5  Universal Alignment Gate
Phase 6  Universal Strict Parameter Gate
Phase 7  Source/Economic/Session Context Gate
Phase 8  Fix Concrete Semantic Bugs
Phase 9  Full Family Audit
Phase 10 Dead Code + Duplicate Cleanup
Phase 11 Per-Canonical Test Completion
Phase 12 Direct-Bypass / Miner Exposure Tests
Phase 13 Full Regression
Phase 14 Evidence Generation
Phase 15 Fresh-process Final Acceptance
```

---

# 260. Phase 1要求

fresh process：

```text
HEAD
canonical count
alias count
production-loaded count
research-loaded count
unsafe-loaded count
```

---

# 261. Phase 2要求

完成后 fresh production import：

```text
research=0
unsafe=0
legacy executable=0
forbidden executable=0
```

---

# 262. Phase 3 删除清单不是手写

从：

```text
current registry
DirectUse
forbidden policy
dead-code audit
semantic audit
```

联合生成候选。

逐个 resolve。

---

# 263. Phase 4 tombstone

只对：

```text
可能仍出现在旧配置里的名字
```

保留。

完全没人引用：

```text
DELETE_COMPLETELY
```

---

# 264. Phase 5 alignment

先 central API，

再批量迁移所有 multi-input。

---

# 265. Phase 6 parameters

central binder，

然后 AST lint扫：

```text
int()
max/min clamp
bool()
str()
```

---

# 266. Phase 7 context

先：

```text
fundamental/consensus
intraday
relation/shareholder
```

这些最高风险。

---

# 267. Phase 8 concrete bug

必须有：

```text
red test
修复
green test
```

不是只改代码。

---

# 268. Phase 9 family audit

每个 current canonical：

```text
review outcome
```

必须写矩阵。

---

# 269. Phase 10 dead code

删除后：

```text
重新跑 import
registry
recipes
tests。
```

---

# 270. Phase 11 test

硬要求：

```text
retained canonical zero tests = 0
```

---

# 271. Phase 12 bypass tests

这是 R29新增重点。

---

# 272. Phase 13 full regression

不只 R29 tests，

跑：

```text
factor_engine完整测试
DataAccess集成相关测试
mining consumer测试
```

---

# 273. Phase 14 evidence

最后生成，

不要中间生成陈旧 evidence。

---

# 274. Phase 15 fresh process

防：

```text
module import缓存
registry污染
旧进程状态
```

---

# 275. R29 Hard Gates

全部必须 TRUE：

```text
R29_CURRENT_HEAD_BOUND

R29_DEFAULT_RUNTIME_RESEARCH_CANONICALS_ZERO
R29_DEFAULT_RUNTIME_UNSAFE_CANONICALS_ZERO
R29_DEFAULT_RUNTIME_LEGACY_EXECUTABLE_ZERO
R29_DEFAULT_RUNTIME_DENIED_EXECUTABLE_ZERO

R29_PERMANENTLY_FORBIDDEN_EXECUTABLE_ZERO

R29_TOMBSTONES_NONCALLABLE
R29_TOMBSTONES_NOT_IN_OPERATOR_REGISTRY
R29_TOMBSTONES_NOT_IN_DSL
R29_TOMBSTONES_NOT_IN_MINING
R29_TOMBSTONES_NOT_IN_BACKENDS

R29_DIRECT_REGISTRY_BYPASS_SAFE
R29_DIRECT_OPERATOR_CALL_SAFETY_GATES_ACTIVE

R29_PUBLIC_ALIAS_POLICY_MONOTONIC
R29_RESEARCH_ALIAS_PUBLIC_EXECUTION_ZERO

R29_CURRENT_CANONICALS_ALL_DISPOSITIONED

R29_ALL_PUBLIC_MULTI_INPUT_ALIGNED
R29_MULTI_INPUT_SILENT_MISPAIR_ZERO

R29_ALL_PUBLIC_PARAMS_STRICT
R29_FRACTIONAL_INTEGER_SILENT_CAST_ZERO
R29_INVALID_PARAM_SILENT_CLAMP_ZERO
R29_UNKNOWN_PUBLIC_KWARG_SWALLOW_ZERO

R29_STATE_SURVIVAL_GAP_BUG_CLOSED
R29_COMPONENT_SCORE_FALSE_VS_MISSING_FIXED

R29_ECONOMIC_MEANING_GATE_ACTIVE
R29_COMPOSITION_PART_OF_WHOLE_VALID
R29_INVALID_FINANCIAL_COMPOSITIONS_ZERO

R29_SOURCE_PROVENANCE_GATE_ACTIVE
R29_ANONYMOUS_CONTEXTUAL_PRODUCTION_CALL_ZERO
R29_CONSENSUS_VINTAGE_ENFORCED

R29_SESSION_CLOCK_UNIFIED
R29_SESSION_RECOVERY_MARKET_HARDCODE_ZERO
R29_INTRADAY_OFFICIAL_SLOT_GATE_ACTIVE
R29_INTRADAY_TIMEZONE_GATE_ACTIVE

R29_PIVOT_CONFIRMATION_NOT_BACKDATED

R29_DEAD_HELPERS_REVIEWED
R29_DUPLICATE_CANONICALS_RESOLVED

R29_RETAINED_ZERO_TESTS_ZERO
R29_DIRECT_BYPASS_TESTS_PASS
R29_MINER_EXPOSURE_TESTS_PASS

R29_EVIDENCE_GITHUB_TRACKED
R29_EVIDENCE_NOT_GITIGNORED
R29_EVIDENCE_CURRENT_SHA

R29_HARD_BLOCKERS_ZERO
```

---

# 276. 最终 public runtime 目标

最终普通用户：

```python
engine = FactorEngine()
engine.list_operators()
```

看到的应该是：

```text
真正可用于生产量化表达的东西。
```

而不是：

```text
“这里有1300多个，里面有一些千万别碰。”
```

---

# 277. 最终研究用户

研究员：

```python
engine = FactorEngine(mode="research")
```

才可以显式获得：

```text
研究工具。
```

---

# 278. 最终 legacy用户

旧公式：

```text
碰到删除名字
```

得到：

```text
明确报错 + replacement/reason
```

而不是执行。

---

# 279. 最终新人误用风险

目标：

```text
新人即使不知道所有治理规则，
也很难使用错。
```

这才叫：

```text
safe by construction
```

而不是：

```text
safe by documentation。
```

---

# 280. 最终质量标准

完成 R29 后，FactorEngine 应该满足：

```text
不是“所有东西都能调用，再告诉你哪些不建议用”，

而是：

“默认能调用到的，就应该是可以调用的。”
```

这应该成为整个算子系统以后新增 operator 的基本原则。

---

# 281. 新 operator以后怎么进入 production

新增 operator默认：

```text
NOT LOADED
```

只有完成：

```text
semantic contract
PIT/timing
source/economic contract
parameter contract
alignment contract
test evidence
role
```

才：

```text
加入 production loader。
```

---

# 282. 不再采用“先注册后治理”

改成：

```text
先证明
后注册。
```

---

# 283. Research incubation

新想法：

```text
先 research registry
```

验证好以后：

```text
promotion PR
```

进入 production。

---

# 284. promotion必须改变物理 loader

不是：

```text
只改 status。
```

---

# 285. deletion也必须改变物理 loader/registry

不是：

```text
只改 denied。
```

---

# 286. 最终验收报告示例

```text
HEAD: ...

Before:
  unified registered canonicals = ...
  default-loaded research = ...
  default-loaded unsafe = ...
  forbidden executable = ...

After:
  production public = ...
  contextual public = ...
  state/condition/event/intermediate = ...
  research isolated = ...
  internal = ...
  tombstones = ...
  physically deleted = ...

Public research executable = 0
Public unsafe executable = 0
Public forbidden executable = 0

Multi-input alignment gaps = 0
Silent parameter coercions = 0
Anonymous contextual production calls = 0

Retained zero-test canonicals = 0

R29_HARD_BLOCKERS_ZERO = true
```

---

# 287. 最后强调

不要把本轮执行成：

```text
“再新增一个 denied list”
```

也不要：

```text
“把 status=research 的更多”
```

更不要：

```text
“所有算子都留着，只是 mining看不到”
```

本轮的核心就是：

```text
物理删掉不该存在的 executable operator，
物理隔离 research，
物理缩小 production runtime，
让 public API 本身不容易被误用。
```

---

# 288. 与 R24-R28 的最终合并标准

最终 production ready：

```text
R24
AND R25
AND R26
AND R27
AND R28
AND R29
```

其中：

```text
R28
证明“保留下来的算子有测试、模型/PIT安全”。

R29
进一步证明“本来不该用的东西已经不在默认可执行面里”。
```

两者缺一不可。

---

# 289. 代码 AI 的直接任务口径

请直接执行：

```text
1. 以当前 HEAD fresh inventory为准；
2. 不要相信旧 catalog/evidence数量；
3. 全 current canonical逐一 disposition；
4. unused / random / future / impossible-PIT / useless / invalid-economic operators直接物理删除；
5. 需要旧名兼容的仅留 non-callable tombstone；
6. research / unsafe / legacy不得进入默认 production loader；
7. 修复本文列出的当前代码 P0/P1问题；
8. 对没逐文件人工点到的 family继续用统一审计矩阵全部扫描；
9. 每个 retained canonical至少一条真实执行测试；
10. 高风险族必须有语义/PIT专项测试；
11. 真实测试结果写进 factor_engine/docs/evidence/r29；
12. 证据必须提交 GitHub、不能被 gitignore；
13. fresh process最终确认所有 R29 hard gates。
```

---

# 290. DoD

只有下面同时满足才可结束：

```text
默认 FactorEngine 不加载 research。
默认 FactorEngine 不加载 unsafe。
永久禁用名字没有 executable implementation。
危险旧 alias不能绕过。
anonymous contextual DataFrame不能伪装 production PIT。
所有多输入不可能静默错配。
所有 public参数不可能静默截断/纠正。
已发现 state/component/session/composition问题修复。
所有 current canonical有明确处置。
所有 retained canonical有真实测试。
所有删除项有 migration/tombstone证据。
GitHub上能看到最终测试/evidence。
```

否则：

```text
R29 未完成。
```
