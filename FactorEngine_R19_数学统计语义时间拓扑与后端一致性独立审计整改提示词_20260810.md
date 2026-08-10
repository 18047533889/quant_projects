# FactorEngine R19：数学 / 统计语义、时间拓扑、参数绑定与跨后端一致性独立审计整改提示词

> **用途**：本文件单独交给代码整改 AI，直接基于服务器当前 `factor_engine` 工作区执行整改。  
> **文档关系**：这是一个**全新的、独立的 R19 文档**。不要与 R17、R18 或更早文档合并。  
> **前提假设**：R17 的多市场数据/provider/PIT问题、R18 的 DirectUse/删冗/角色/直接挖掘问题都视为正在由其它并行 AI 完成，本文件不重复那些事项。  
> **当前 GitHub 可见基线**：`48aec8d20825fd50086ace0fc06b26a5f7273e49`，提交说明为 `Sync FE operator/param hardening, R17 audit prompt, and OPEN_WEEKLY report.`。服务器如果已更新，以服务器真实 HEAD 为准。  
> **本轮主题**：不再问“这个 operator 在不在 registry”，而是问：  
>
> **它的数学实现是不是等于它声称的定义？参数是不是同一套语义？时间窗口是不是正确？缺失/Inf/并列值是不是一致？Numba/Pandas/Polars/DuckDB 是否真的计算同一个东西？组合后有没有伪因子、未来信息、样本起点污染或后台路径漂移？**
>
> **执行原则**：本轮所有发现都要修改代码和测试，不要只生成报告。所有当前 retained canonical 都要动态扫描，不能只修下面点名的算子。

---

# 0. R19 与 R17 / R18 的边界

R17 已重点解决：

- A 股 / 美股 field/provider/source/PIT；
- market/session/currency/accounting semantics；
- provider coverage 与 market context；
- typed IR 的市场语义。

R18 已重点解决：

- 每个 canonical 的 DirectUseStatus；
- ALPHA / STATE / EVENT / CONDITION / INTERMEDIATE / SOURCE_TRANSFORM；
- useless/no-data/duplicate 删除；
- direct mining catalog；
- operator smoke recipe；
- mining grammar reachability；
- 参数是否值得搜索。

**R19 不重新讨论上述分类。**

R19 专门解决：

```text
Operator 宣称的数学定义
        ↓
实际 Python/Numpy/Pandas kernel
        ↓
Numba fastpath
        ↓
Polars/DuckDB lowering
        ↓
history/warmup/chunking
        ↓
parameter normalization
        ↓
missing/Inf/tie/ddof/domain
        ↓
组合表达式
```

这条链是不是一个数学对象。

最终目标：

> **同一个 canonical + 同一组 canonical parameters + 同一份 canonical inputs，在任何合法执行路径上必须代表完全同一个数学定义。**

---

# 1. 本轮必须新增的“数学语义证书”

在 R18 DirectUse evidence 之外，为每一个 retained operator 新增：

```python
MathematicalSemanticCertificate
```

至少包含：

```text
canonical

mathematical_definition
axis_semantics
sample_definition
finite_value_policy
missing_value_policy
current_row_policy
tie_policy
ddof_policy
zero_denominator_policy
domain_policy
partial_window_policy
minimum_effective_sample
time_index_semantics
history_formula

parameter_binding_hash
semantic_kernel_hash

reference_backend
fastpath_backends

metamorphic_properties
cross_backend_parity_passed
prefix_invariance_passed
chunk_invariance_passed
```

没有完整数学证书的 retained direct operator，不能 production certify。

---

# 2. 当前代码已确认的新增问题：参数系统仍然不是单一权威

---

## R19-001：`ParameterCanonicalizer.hash_key()` 当前实现会发生 `str + tuple` 类型错误

当前：

```python
def hash_key(self, attrs):
    canon = self.canonicalize(attrs)
    return (self.canonical or "") + tuple(
        sorted((k, _freeze(v)) for k, v in canon.items())
    )
```

`(self.canonical or "")` 是 `str`，右侧是 `tuple`。

这不是合法 Python tuple 构造，真正调用时会：

```text
TypeError: can only concatenate str (not "tuple") to str
```

### 必须整改

例如：

```python
return (
    self.canonical or "",
    *tuple(sorted((k, _freeze(v)) for k, v in canon.items())),
)
```

或者：

```python
return (
    self.canonical or "",
    tuple(sorted(...)),
)
```

但要选定一种稳定 schema，并加：

```text
same params -> same key
different semantic params -> different key
cross-process deterministic
```

golden test。

这是 factor dedup / CSE / cache identity 的底层错误，不能只靠当前测试没覆盖而放着。

---

## R19-002：planning-time parameter validator 与 runtime validator 仍不是完全同一输入域

`planner/canonicalize_params.py::validate_plan_params()` 当前只对：

```python
isinstance(value, (int, float, str, bool))
```

的 attrs 做 scalar validation。

因此：

```text
np.int64
np.float64
Decimal
enum scalar
tuple/list structured scalar
```

可能在 planning 被跳过，却在 runtime 被另一套规则接收/拒绝。

### 必须整改

不要手写 Python type filter。

新增统一：

```python
normalize_and_validate_scalar_param(
    canonical,
    param_name,
    value,
    phase="planning"|"runtime"|"hash"
)
```

planning/runtime 共享一套 declared ParamSpec type domain。

---

## R19-003：显式 `None` positional literal 在 planning validator 中可能被跳过

当前 planning 提取 positional literal 时类似：

```python
if child.op == "literal" and child.attrs.get("value") is not None:
    ...
```

这会把：

```text
explicit None
```

与：

```text
literal missing
```

混为一谈。

而当前 ParamSpec 已经明确区分：

```text
MISSING
None
```

### 必须整改

判断：

```python
"value" in child.attrs
```

而不是：

```python
value is not None
```

显式 None 必须进入：

- ParamSpec validation；
- active_when；
- relational constraint；
- hash identity。

---

## R19-004：当前仍存在第二套 `parameter_validation.py`

现在已有：

```text
cleaned_operators/common/strict_params.py
```

自称：

```text
THE single strict scalar-parameter authority
```

但：

```text
cleaned_operators/parameter_validation.py
```

仍存在：

```python
strict_integer
strict_finite_scalar
```

大量 operator 仍 import 它。

### 新问题

两套 strict gate 的接受域并不完全一样。

例如旧 `strict_integer`：

```python
isinstance(value, (int, float))
```

而新的 base/strict_params 还支持：

```text
np.integer
np.floating
```

等。

### 必须整改

最终：

```text
parameter_validation.py
```

只能是兼容 re-export，不能有独立实现。

全库 AST audit：

```text
所有 scalar validation
→ strict_params / central ParamSpec authority
```

0 个 module-local strict validator。

---

## R19-005：numeric-string coercion 仍存在“声明层”和“kernel层”双重规则

`base._coerce_declared_numeric_string()` 已明确：

> 字符串只有在 ParamSpec/param_types 声明 numeric 时才转换。

但：

```text
strict_params.strict_int
→ base.strict_int_param
```

而 `strict_int_param` 本身仍能解析 numeric string。

于是：

```python
operator kernel直接 strict_int("20", "window")
```

仍可能绕过“必须有 numeric declaration 才能把 string 转数字”的原则。

### 必须整改

numeric-string conversion 只能发生在：

```text
DSL/binder
```

一次。

进入 kernel 后：

```text
strict_int("20")
```

必须 reject。

建议：

```text
bind_numeric_string_if_declared()
strict_int_runtime()
```

职责完全拆开。

---

## R19-006：parameter alias 当前至少存在三套 authority

当前可见：

1. `OperatorMetadata.param_aliases`
2. `base._LEGACY_KERNEL_ALIASES`
3. `backend/parameter_aliases.py::PARAMETER_ALIASES`

这会导致：

```text
parser 认一个 alias
planner 认另一个
runtime kernel又自己 kwargs.get 一个
```

### 必须整改

唯一权威：

```python
metadata.param_aliases
```

兼容全局 alias map必须从 metadata 自动导出，不能手写第二份。

CI：

```text
GLOBAL_ALIAS_MAP
==
DERIVED_METADATA_ALIAS_MAP
```

---

## R19-007：`active_when` 当前使用的可能不是 kernel 最终收到的 canonicalized bound values

`_normalise_call()` 会先：

- numeric-string conversion；
- ParamSpec normalization；
- alias target validation。

但调用：

```python
_enforce_active_when(metadata, args, kwargs, defaults)
```

仍使用原始 `args/kwargs`。

于是存在：

```text
alias controller
numeric string controller
canonicalized float/int controller
```

与 kernel 最终 bound 值不一致的风险。

### 必须整改

构造唯一：

```python
NormalizedBoundParameters
```

然后：

```text
active_when
relational specs
history formula
hash
kernel call
```

都读取同一个 bound。

---

## R19-008：`active_when`、relation、history 三套参数绑定不能各自重新 bind

本轮建立：

```python
bind_operator_call(...) -> BoundOperatorCall
```

至少：

```text
panel_inputs
scalar_params
canonical_aliases_resolved
defaults_applied
inactive_params
normalized_values
```

之后各层只能读，不再各自重建 dict。

---

# 3. 当前 audit script 仍有 release-gate 逻辑缺口

---

## R19-009：`release_blocking` 与 CLI 真正检查的 `_HARD_INVARIANTS` 不一致

当前 audit result 中 `release_blocking` 包含：

```text
UNUSED_PUBLIC_CANONICALS
FACTOR_SHAPED_RESEARCH_ONLY
SEMANTIC_DUPLICATE_CANONICALS
DEAD_SEARCHABLE_PARAMS
...
```

但 CLI `main()` 实际 hard fail 只检查其中一部分：

```text
UNCLASSIFIED_FACTOR_CANONICALS
MINING_ROLE_UNRESOLVED
MINING_ELIGIBLE_WITHOUT_CERTIFICATION
PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY
CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST
```

于是可能：

```text
DEAD_SEARCHABLE_PARAMS != []
SEMANTIC_DUPLICATE != []
```

但 `--strict` 仍 exit 0。

### 必须整改

CLI 不再维护第二份 `_HARD_INVARIANTS`。

直接：

```python
for invariant in result["release_blocking"]:
    ...
```

---

## R19-010：duplicate “candidate” 与 “confirmed exact duplicate” 被混在一个 blocking invariant

当前 duplicate detector 会把：

```text
alias pair
name-family pair
```

都列入 candidate。

但最终 invariant 叫：

```text
SEMANTIC_DUPLICATE_CANONICALS
```

又被列为 release-blocking。

### 必须整改

拆：

```text
SEMANTIC_DUPLICATE_CANDIDATES      # advisory
CONFIRMED_EXACT_DUPLICATES        # blocking
MONOTONIC_EQUIVALENT_PAIRS        # mining dedup
RELATED_FAMILY_PAIRS              # non-blocking
```

只有证明 exact equivalent 才阻止 release。

---

## R19-011：dead-param probe 对 `choices` 的替代值构造仍可能不是合法 choice

当前逻辑类似：

```python
lo = spec.min or 1
hi = spec.max or lo + 2
kwargs[name] = hi if spec.choices else lo + 1
```

如果 `spec.choices` 是：

```python
("pearson", "spearman")
```

`hi` 根本不是 choice。

### 必须整改

choices：

```python
alternatives = [c for c in spec.choices if c != base]
```

逐个真实 choice 测。

---

## R19-012：dead-param probe 对 `MISSING` default 判断错误

当前：

```python
if spec is not None and spec.default is not None:
    kwargs[name] = spec.default
```

但 `MISSING` sentinel：

```text
is not None
```

所以可能把：

```python
MISSING
```

本身传进 kernel。

### 必须整改

必须：

```python
spec.default is not MISSING
```

---

## R19-013：dead-param coverage 当前主要按“operator”统计，而不是“parameter”

一个 operator base run 成功后：

```text
audited += 1
```

即使某个具体 param alternate run 失败，它仍被计为 audited operator。

### 必须整改

报告：

```text
total_searchable_params
successfully_probed_params
dead_params
probe_error_params
untested_params
```

release：

```text
probe_error_params == 0
untested_params == 0
```

---

## R19-014：behavior fingerprint 直接 hash 原始 float bytes，NaN payload 可能不稳定

虽然当前已经额外 hash NaN/Inf mask，这是进步，但：

```python
h.update(arr.tobytes())
```

仍把 NaN 的底层 payload bits 放进 hash。

不同 backend/process/library 可能产生不同 NaN payload，但语义完全一致。

### 必须整改

先 canonicalize：

```text
NaN -> 0
+Inf -> fixed sentinel
-Inf -> fixed sentinel
```

再 hash values，同时单独 hash：

```text
nan mask
+inf mask
-inf mask
```

---

## R19-015：`detect_manifest_gap()` 用 `admission="eligible"`，却没有给 source context

当前 mining eligibility 的设计是：

```text
available_sources=None
→ UNKNOWN
→ fail closed
```

那么：

```python
get_mining_operators(admission="eligible")
```

如果没有 market/source context，可能得到空集或过少集合。

随后会误报：

```text
CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST
```

### 必须整改

两个独立 invariant：

```text
CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST
CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST[market,source_context]
```

不要拿 context-free query 冒充运行环境。

---

# 4. 已确认的重大数学错误：`rank_corr(d=0)` 不是截面 Rank Corr，而且有 full-sample look-ahead

---

## R19-016：当前 `rank_corr(d=0)` wrapper 维度错误

当前 wrapper：

```python
for col in x.columns:
    result[col] = rank_corr_(
        x[col].values,
        y[col].values,
        d=int(d)
    )
```

也就是说：

```text
每个股票一列
→ 把整段历史 time-series 传进 rank_corr_
```

底层 `d==0` 却把传入的一维数组当成：

```text
“截面”
```

然后：

```text
对整段历史 rank
→ 算一个整样本 corr
→ 广播回整段历史
```

### 后果

对于 t 时点输出，使用了：

```text
t+1 ... T
```

的数据。

这是明确的 full-sample look-ahead。

### 必须整改

彻底删除这种 dual semantics：

```text
rank_corr(d=0)
```

拆成：

```text
ts_rank_corr(x, y, window)
cs_rank_corr(x, y)
```

---

## R19-017：真正 `cs_rank_corr` 必须逐日跨股票计算

正确：

```python
for date:
    rank x across instruments
    rank y across instruments
    corr paired ranks
```

输出如果广播全股票：

```text
GLOBAL_STATE
```

或者作为：

```text
cross-sectional diagnostic/control context
```

不能伪装个股 alpha。

---

## R19-018：真正 `ts_rank_corr` 必须只看 trailing window

每股票：

```text
[t-W+1, t]
```

独立 rank x/y，并 paired finite。

必须：

```text
prefix invariant
window local
tie policy explicit
```

禁止任何全样本 rank。

---

# 5. 内部 `rank_()` 当前并列值会产生“列顺序信号”

---

## R19-019：`_numpy_kernels.rank_()` 双 argsort 不处理 tie

当前类似：

```python
ranks = np.argsort(np.argsort(arr[valid]))
```

当：

```text
x_i == x_j
```

时，它不会给平均 rank，而会按照内部排序/原始位置给不同 rank。

### 后果

若股票列顺序固定：

```text
ticker alphabetic order
```

并列值会生成伪横截面差异。

### 必须整改

全库统一 tie authority：

```text
average
min
max
dense
first
```

Factor rank 默认推荐：

```text
average
```

并在 metadata 写明。

---

## R19-020：`rank_()` 只排除 NaN，不排除 ±Inf

当前：

```python
valid = ~np.isnan(arr)
```

但系统其它最新 cross-sectional rank 已经明确：

```text
np.isfinite
```

才算合法统计样本。

### 必须整改

内部 kernels 不得自行定义 sample validity。

统一：

```python
finite_mask()
paired_finite_mask()
```

---

# 6. 已确认的 OLS 数学错误：`ts_regression_slope_()` 的 ddof 混用

---

## R19-021：`np.cov` 样本协方差 / `np.var` 总体方差导致 slope 系统性放大

当前：

```python
cov = np.cov(x, y)[0,1]   # denominator n-1
var = np.var(x)           # denominator n
beta = cov / var
```

所以：

```text
beta_returned
=
beta_OLS * n/(n-1)
```

### 必须整改

直接用 centered sums：

```python
xc = x - mean(x)
yc = y - mean(y)
beta = dot(xc,yc) / dot(xc,xc)
```

这样不依赖 ddof。

---

## R19-022：全库所有 regression/beta/cov/corr 做 denominator-consistency audit

机器扫描：

```text
np.cov
np.var
Series.var
rolling.var
std(ddof=...)
```

组合。

任何 ratio statistic 必须确保：

```text
same cohort
same ddof convention
same finite mask
```

---

# 7. 已确认的行业+规模中性化数学问题

---

## R19-023：当前 `industry_size_resid_panel_` 不是联合行业+规模回归

当前：

```python
ind = group_demean_panel_(y, industry)
return size_resid_panel_(ind, market_cap)
```

第二步使用的是：

```text
原始 log(size)
```

而不是：

```text
industry-demeaned log(size)
```

### 为什么错误

这不满足 Frisch–Waugh–Lovell。

会让 size regressor 中的行业平均差异重新进入 residual projection。

### 必须整改

二选一。

### 方案 A：直接联合回归

```text
y ~ industry_dummies + log(size)
```

### 方案 B：FWL

```text
y_tilde = demean(y | industry)
size_tilde = demean(log_size | industry)
resid = residual(y_tilde ~ size_tilde)
```

---

## R19-024：`size_resid_panel_` 当前把非法市值 clamp 到 1

类似：

```python
log(max(market_cap, 1))
```

这会把：

```text
0
negative
invalid
```

都变成：

```text
log(size)=0
```

### 后果

数据错误不再 missing，而变成“极小公司”。

### 必须整改

合法：

```text
market_cap > 0
and finite
```

否则：

```text
NaN
```

或 strict DQ reject。

禁止 silent clip。

---

## R19-025：`cs_resid_ / cs_regression_` 只用 `np.isnan`，±Inf 会进入回归

统一：

```text
np.isfinite(y) & np.isfinite(x)
```

---

# 8. “finite sample” 目前仍不是全库统一概念

---

## R19-026：`coalesce_()` 文档说 first finite，但实现只判 NaN

如果：

```text
first candidate = +Inf
second candidate = valid finite
```

当前会保留 Inf。

### 二选一

如果 coalesce 定义是：

```text
first non-null
```

改文档/semantic name。

如果定义是：

```text
first finite
```

实现改为：

```python
~np.isfinite(result)
```

不能文档和代码各说一种。

---

## R19-027：Cross-sectional count 与 rank 对 Inf 的样本定义不一致

最新：

```text
cs_rank_01
```

明确用：

```python
np.isfinite
```

而：

```text
c_count
broadcast_row_stat_all_null_null
```

还主要依赖：

```python
DataFrame.count()
```

Pandas count 会把 Inf 当有效值。

### 必须整改

全截面统计共享：

```python
CrossSectionSampleMask
```

---

## R19-028：`c_mean/c_std/c_sum/c_percentile/cs_mad` 全部统一 finite mask

不要由 Pandas 默认 `skipna` 决定是否把 Inf 当样本。

---

## R19-029：所有 `_numpy_kernels` statistical helper 禁止只用 `np.isnan`

静态审计：

```text
np.isnan
.notna()
.count()
```

在统计 sample selection 中都需要人工确认：

```text
是不是应该 np.isfinite
```

最终每个 operator metadata 声明：

```text
sample_validity = finite | non_null | domain_specific
```

---

# 9. `ts_corr` 目前 Numba 与 Pandas slow path 在“当前行缺失”时可计算不同结果

---

## R19-030：Numba rolling corr 与 slow Pandas current-row policy 不一致

Numba kernel：

```text
在窗口里找 finite pairs
```

只要窗口有效 pair 足够，即使：

```text
当前 t 的 x/y 是 NaN
```

也可能输出 rolling corr。

而 slow path：

```python
valid = x.notna() & y.notna()
rolling_corr(...).where(valid)
```

会强制：

```text
当前行缺失 -> output NaN
```

### 必须整改

定义一个明确：

```text
CurrentObservationRole
```

例如：

```text
window_statistic_can_exist_without_current_observation
```

或者：

```text
requires_current_pair
```

然后 Pandas/Numba/Polars/DuckDB 全部一致。

---

## R19-031：`ts_cov` 同样存在 current-row 强制 mask 的特殊语义

普通 trailing covariance 数学上只要求窗口内 pair 足够，不一定要求当前 row 本身非缺失。

如果你们确实想要求 current pair：

必须是显式 semantic contract。

不能 `ts_cov` 一种、其它 rolling stats 另一种。

---

## R19-032：`ts_corr` metadata 仍使用裸 `allow_panel_broadcast`

当前 base 已经明确：

```text
bare allow_panel_broadcast
=
legacy research waiver
```

但 `ts_corr` 仍有这个 tag。

### 必须整改

实际上相关系数的两个 panel：

```text
通常应该 exact aligned
```

如果 benchmark broadcast 合法：

用结构化 `BroadcastSpec`。

否则删除 waiver。

---

# 10. `ts_beta` 仍存在旧实现路径，与最新 shared `rolling_beta` 语义漂移

---

## R19-033：`ts_beta` 仍自行使用 `min_periods=2`

而最新 shared `rolling_beta` 已明确：

```text
reviewed min_periods default = 5
```

但 `MovingBeta/ts_beta` 当前还有：

```python
mp = max(int(kwargs.get("min_periods",2)),2)
```

### 必须整改

一个 beta kernel authority。

```text
Beta
ts_beta
rolling_beta
```

最终必须统一到同一个实现/定义。

---

## R19-034：`ts_beta` 当前 `notna()` 仍把 Inf 当有效 pair

改 paired finite。

---

## R19-035：`ts_beta` 仍有 hidden `min_periods` 参数

metadata 没完整声明，却 kernel `kwargs.get()`。

这属于：

```text
hidden parameter
```

必须：

- 声明 ParamSpec；
- 或完全删除 hidden override。

---

# 11. optional fastpath 不能改变数学结果，也不能静默 fallback

---

## R19-036：当前 triple-backend parity 没有单独强制 Numba ON/OFF 对比

新增：

```text
pandas_reference_no_numba
pandas_numba_forced
polars_native_forced
duckdb_native_forced
```

四路径 parity。

---

## R19-037：`except Exception: pass` 的 fastpath fallback 会隐藏错误

例如：

```text
ts_mean
ts_std
ts_corr
```

fastpath异常后直接退 Pandas reference。

在普通 research 可以 fallback。

但在：

```text
backend certification
parity audit
production evidence
```

必须：

```text
fastpath error = fail
```

不能靠 fallback 把测试“跑绿”。

---

## R19-038：lineage/evidence 必须记录实际执行路径

每个 result：

```text
requested_backend
actual_backend
fallback_reason
native_fastpath_used
```

否则“Polars/DuckDB通过”可能实际全部跑了 Pandas fallback。

---

# 12. `ts_argmax / ts_argmin` 当前有三套互相矛盾的定义

---

## R19-039：operator doc 说 `0=最新 bar`

当前 `TSArgmax/TSArgmin` doc：

```text
0 = 窗口内最新 bar
```

---

## R19-040：numeric semantics 说 `0=window left/oldest`

当前：

```python
ts_argmax_index_origin() -> "window_left_0"
```

---

## R19-041：实际 kernel 返回的也是从窗口左侧/最旧开始计数

`rolling_argmax`：

```python
seg = window left -> right
hits = flatnonzero(...)
out = hits[-1]
```

因此：

```text
index 0 = oldest
```

与 operator doc 直接冲突。

---

## R19-042：numeric semantics 说 tie-break=`first`，kernel 却用 `hits[-1]`

kernel：

```text
并列最大值取最后/最新 occurrence
```

numeric semantics：

```text
first
```

### 必须整改

明确拆 canonical：

```text
ts_argmax_index_from_oldest
ts_argmax_age
```

其中：

```text
index_from_oldest: 0=oldest
age: 0=current/latest
```

旧 `ts_argmax` 必须选一个唯一兼容定义。

tie-break：

```text
first/latest
```

只能一个，并全 backend 一致。

---

# 13. `ts_product` 当前不是真正通用 rolling product

---

## R19-043：当前实现把 0 当 missing

当前：

```python
x.replace(0, np.nan)
log()
rolling_sum()
exp()
```

数学乘积：

```text
(... * 0 * ...)
```

应该是：

```text
0
```

而不是忽略 0。

---

## R19-044：负数输入会被 log 变 NaN

所以：

```text
(-2)*(-3)=6
```

无法正确得到。

### 必须决定

如果 canonical 叫：

```text
ts_product
```

就实现真正 signed + zero-safe product。

建议维护：

```text
zero_count
sign_parity
sum_log_abs
```

得到稳定 rolling product。

如果只想做：

```text
positive_geometric_product
```

那必须改 canonical/semantic type，不要叫 generic product。

---

# 14. decay 系列 partial-window / missing 语义仍不统一

---

## R19-045：`ts_sum_decay` partial window 使用权重方向与 WMA policy 不一致

WMA 最新已经明确：

```text
partial window
→ 使用 newest L age slots
```

但 `ts_sum_decay` 当前类似：

```python
weights[:len(s)]
```

相当于 partial history 使用另一端权重。

### 必须整改

所有 age-weighted operator 声明：

```text
oldest_to_newest orientation
partial-window anchor
renormalization
```

---

## R19-046：`ts_sum_decay` 遇 NaN 时 raw `np.dot` 容易整窗变 NaN

如果 policy 是：

```text
skip missing + reweight
```

必须实现。

如果是：

```text
any missing invalidates window
```

也必须明确。

不能由 NumPy 偶然传播决定。

---

## R19-047：`ts_decay_exp_window` 同样缺 explicit partial/missing contract

---

## R19-048：`ts_decay_exp_window.alpha` 缺完整合法域

若定义：

```text
decay weight alpha^age
```

通常需要：

```text
0 < alpha <= 1
```

或者明确允许 >1 表示反向 age preference。

不允许：

```text
alpha=0
negative alpha
NaN/Inf
```

在未定义情况下进入矿工。

---

# 15. 仍有大量 hidden/local parameter casts

---

## R19-049：`ts_argmax/argmin` 仍 local `int(kwargs.get(...))`

这与“单一 strict authority”冲突。

---

## R19-050：`ts_quantile` 仍有 hidden `window` / `p`

类似：

```python
window = int(kwargs.get("window", d))
quantile = float(kwargs.get("p", q))
```

但 metadata 的 canonical params 是：

```text
x,d,q
```

### 必须整改

所有 aliases显式 metadata声明，binder canonicalize后 kernel只收 canonical names。

---

## R19-051：`ts_quantile.q` 必须严格 `[0,1]`

用：

```text
strict_probability
```

但不能 kernel手写 float()。

---

## R19-052：`ts_topk_sum` hidden `window/n` 与 local int cast 全部去掉

---

## R19-053：`price_spread_deviation` hidden `window` alias 全部收口

---

## R19-054：`ts_moment(d,k)` local `int(d),int(k)` 收口 ParamSpec

并明确：

```text
k >= 1?
k=1 是否有意义？
odd/even?
```

---

## R19-055：任何 kernel 中 `int(parameter)` / `float(parameter)` 都做 AST audit

只有：

```text
已经由 binder 规范化的值
```

才能进入 kernel。

kernel 不再承担 parse/coerce。

---

# 16. `ts_sharpe / ts_autocorr` 有隐藏 signature 参数未进入 metadata

---

## R19-056：`ts_sharpe.min_periods` kernel 可传，但 metadata 未完整声明

这意味着：

- factor identity 可能没它；
- search-space看不到；
- planner history看不到；
- evidence可能没它。

### 必须整改

若允许自定义：

```text
param_names
ParamSpec
ParamRole.SUPPORT_POLICY
```

全声明。

否则从 public call删除该参数。

---

## R19-057：`ts_autocorr.min_periods` 同样处理

---

## R19-058：`ann_factor` 不应作为普通 alpha 搜索维度

`ann_factor` 是：

```text
frequency/session normalization policy
```

不是经济因子 knob。

设：

```text
ParamRole.SESSION_POLICY / POLICY
searchable=False
```

---

## R19-059：`ann_factor=0` 当前允许，会把整个 Sharpe 变成 0

如果 annualization factor 表示每年 period数：

```text
必须 > 0
```

不能 minimum=0。

---

# 17. returns / log-return 数值域继续收口

---

## R19-060：`ts_pct` 只检查 previous not-null/nonzero，不足以保证 current finite

至少：

```text
current finite
previous finite
previous != 0
```

否则 ±Inf 可以生成 Inf。

---

## R19-061：`ts_log_return` 把非法 price<=0 直接 mask成 NaN，会丢失“missing vs invalid”区别

R17/R18 已有数据质量概念。

这里要求 operator 输出/lineage 能区分：

```text
source missing
domain invalid
```

生产 strict DQ 可以 reject；research可以 mask但必须计数 DQ evidence。

---

## R19-062：`ts_delay` 文档说“负滞后返回 NaN”，实现实际 reject

文档/代码必须统一。

建议：

```text
negative lag = FutureReferenceError
```

不要文档继续说返回 NaN。

---

# 18. Cross-sectional neutralize 当前缺失行业时会改变经济定义

---

## R19-063：`neutralize(x, group=None)` 当前退化成 global demean

如果 canonical 描述是：

```text
行业中性化
```

那么：

```text
没有行业数据
```

不能自动变：

```text
市场去均值
```

这不是 fallback，是另一个因子。

### 必须整改

production：

```text
missing group -> fail/NaN
```

global demean 应显式调用：

```text
cs_demean
```

---

## R19-064：某日 group 全 missing 也不能 silent global demean

同理。

---

## R19-065：group membership 部分 missing

只有：

```text
group known的股票
```

参与组中性化。

unknown group保持 NaN。

不能自动塞进 market group。

---

# 19. 已存在的 exact duplicate cross-sectional canonical 做新一轮数学证明

R18 已要求 general duplicate audit，本轮给出当前代码具体候选。

---

## R19-066：`rank` 与 `cs_rank_01` 当前看起来调用同一实现

证明 exact equivalent 后：

```text
一个 canonical
另一个 alias
```

---

## R19-067：`rank_pct` 与 `cs_pct_rank` 当前同实现链

同样处理。

---

## R19-068：`c_percentile` 与 `cs_quantile` 当前继承同一数学实现

如果只有命名差异，合并。

注意：

这里是**具体数学等价候选**，不是 R18 那种 broad dedup原则。

---

# 20. Truthiness 语义当前发生直接冲突

---

## R19-069：strict ConditionBool 说 ±Inf 非法，numeric semantics 却说 Inf=True

最新 strict semantic bool：

```text
{0,1,NaN}
```

±Inf明确数据质量错误。

但 numeric semantics 还存在：

```python
truthy_inf_is_true() -> True
```

### 必须整改

不能同时存在两套逻辑：

```text
typed bool logic
generic truthy numeric logic
```

production factor DSL建议只允许 typed bool。

---

## R19-070：当前 parity test 用 `group_id=1/2` 直接作为 `where` condition

这是一个会让错误语义“测试通过”的 fixture。

应改成：

```text
condition = {0,1,NaN}
```

另加 negative test：

```text
2
-1
Inf
string
```

必须 reject。

---

## R19-071：NULL/NaN 是否 false 还是 unknown，必须显式决定

当前 numeric semantics 类似：

```text
NULL/NaN -> false
```

这对 `where` 有强经济后果：

```text
unknown condition
```

被当成：

```text
False branch
```

可能把缺失信号变成实际交易信号。

### 建议

production条件采用三值逻辑：

```text
True
False
Unknown
```

例如：

```text
where(Unknown, a, b) -> NaN
```

如果确实要 unknown=false，必须是另一个 explicit operator/policy。

---

# 21. Numeric semantics registry 本身也需要治理

---

## R19-072：`OPERATOR_SEMANTICS` 当前出现重复 `"divide"` key

Python dict后者静默覆盖前者。

### 必须整改

不要 literal dict直接允许重复 key。

用：

```python
register_numeric_semantics(canonical,...)
```

重复注册 hard fail。

---

## R19-073：所有 numeric semantics 必须进入 semantic hash

这些会改变数学结果：

```text
ddof
tie
quantile interpolation
zero std
div zero
Inf
NaN
truthiness
```

Factor identity/evidence必须包含它们的 version/hash。

---

## R19-074：全局 default semantics 不能替代 operator review

每个 retained statistical operator至少显式声明它真正依赖的关键 policy。

否则未来全局 default 一改，会批量改变历史 factor identity。

---

# 22. 所有 expanding / cumulative 算子要做“样本起点污染”全族审计

R18 重点提过 `expanding_rank`，本轮扩展到整个 family。

---

## R19-075：所有 `expanding_*` 输出依赖 dataset start

例如：

```text
expanding_mean
expanding_std
expanding_rank
expanding_zscore
```

同一股票同一天：

```text
数据从2010开始
vs
数据从2015开始
```

结果不同。

### 必须整改

每个 expanding canonical声明：

```text
anchor policy
```

例如：

```text
listing_start
fixed_global_start
campaign_start
full_available_history
```

Factor identity包含 anchor。

---

## R19-076：`cumsum/cum_*` 同样受 anchor影响

不能只把它们当普通 bounded ts operator。

---

## R19-077：自动挖掘默认不应让 campaign start date隐式改变 factor definition

研究窗口只是：

```text
evaluation window
```

不能自动成为：

```text
factor history anchor
```

---

# 23. history/warmup 仍有 legacy heuristic fallback

---

## R19-078：`execution_contract.py` 仍维护 `_WINDOW_LIKE_PARAM_NAMES`

即使已经有 ParamSpec history semantics，代码仍允许名字推断 fallback：

```text
window
d
n
lag
...
```

### 风险

`lag` 的 history extension 是：

```text
+lag
```

普通 window 是：

```text
window-1
```

fallback猜错就会少1 bar或更多。

### 必须整改

**所有 retained direct operator 禁止使用 heuristic history fallback。**

最终：

```text
history_semantics
history_formula
```

100% explicit。

legacy/research才可 fallback。

---

## R19-079：stale `ts_cusum_break_score` 仍出现在 execution stateful seed

仓库此前已有 rename到：

```text
ts_cusum_pressure
```

当前 execution contract legacy seed仍可见旧名。

### 必须整改

全局 ghost audit扩展到：

```text
execution contracts
history transforms
checkpoint registries
```

---

## R19-080：history formula 要描述“首个数学有效输出”，不只是数据窗口

例如：

```text
skew
kurtosis
regression
correlation
```

window=20不代表 1 个有效样本就能输出。

history证书增加：

```text
minimum_effective_samples
```

---

## R19-081：support floor 是 history contract 的一部分

例如：

```text
corr >= 2/3 pairs
skew >= 3
kurtosis >= 4
regression >= p+1
```

需要 machine-readable。

---

# 24. “窗口中间缺失”不能由每个算子自由决定是否 reconnect

---

## R19-082：新增 `MissingTopologyPolicy`

至少：

```text
SKIP_FINITE
PRESERVE_PHYSICAL_LAG
BREAK_EPISODE
REQUIRE_CONTIGUOUS
PAIRWISE_FINITE
```

例如：

```text
mean
```

可以 SKIP_FINITE。

但：

```text
autocorr
directional state
event spacing
```

不能把缺失两边重新当邻居。

---

## R19-083：所有 lag/statistics 明确 physical-row lag 与 finite-observation lag

```text
lag=1
```

默认应是：

```text
前一个交易bar
```

不是：

```text
前一个非缺失观测
```

---

# 25. trading bar / calendar time / session slot 三种时间不能混

---

## R19-084：所有 `days_since/age/duration` 明确单位

可能是：

```text
trading bars
calendar days
session slots
events
reports
```

metadata必须明确。

---

## R19-085：名字里 `days` 但实际按 row count 的算子要重命名或声明

自动挖掘不能把“交易日 bars”误读成自然日。

---

## R19-086：分钟 session operator 的 duration 不能默认把午休当连续分钟

A 股 11:30→13:01之间不是有效 slot。

必须使用 session slot index，而不是 wall-clock minute差，除非 canonical明确要 wall-clock。

---

# 26. regression/correlation family 统一“当前观测是否必须存在”

---

## R19-087：定义 `CurrentRowRequirement`

例如：

```text
NOT_REQUIRED_FOR_WINDOW_STAT
REQUIRED_AS_TARGET
REQUIRED_AS_PAIR
REQUIRED_AS_EVENT
```

然后：

```text
ts_mean
ts_corr
ts_cov
ts_regression_forecast_error
```

分别声明。

避免不同 backend随手 `.where(current.notna())`。

---

# 27. 所有 rank family 统一 tie / range / singleton semantics

---

## R19-088：rank输出区间必须明确

现在至少存在：

```text
0-1 with singleton=0.5
rank/count with smallest=1/n
```

这两个都合理，但必须是不同 canonical semantics。

---

## R19-089：同一个 alias不能在两个 rank semantic之间漂移

旧：

```text
rank
rank_pct
cs_rank
c_rank
```

全部审映射。

---

## R19-090：group rank / ts rank / cs rank tie method分别证书化

---

# 28. quantile / percentile family 完整统一

---

## R19-091：所有 quantile p 严格 [0,1]

---

## R19-092：interpolation method 固定并进 identity

```text
linear
nearest
lower
higher
midpoint
```

不同 backend默认值不能自己决定。

---

## R19-093：group/cs/ts quantile 样本的 finite policy一致

---

# 29. robust statistics 的命名必须等于实际公式

---

## R19-094：`cs_mad_zscore` 当前是 `(x-median)/MAD`

有些领域把 robust zscore定义为：

```text
0.6745*(x-med)/MAD
```

或用：

```text
MAD * 1.4826
```

这里不是说当前公式一定错。

要求：

```text
description 精确写 raw-MAD standardized score
```

如果要 scaled robust zscore，做独立明确定义。

不能一个名字让用户以为用的是标准正态一致性常数。

---

# 30. elementary math 的 domain error 要跨 backend一致

---

## R19-095：`asin/acos` 只在 [-1,1] 有实数定义

Pandas/Numpy可能 warning+NaN。

DuckDB/Polars可能 error/null/NaN。

统一：

```text
outside domain -> NaN/null
```

或 strict reject，选一个。

---

## R19-096：fractional `power` + negative base

这是典型跨 backend divergence。

必须声明 real-domain policy：

```text
negative base & non-integer exponent -> NaN
```

不能某 backend complex、某 backend null。

---

## R19-097：`exp` overflow policy统一

例如：

```text
exp(1000)
```

统一：

```text
Inf -> NaN
```

或者允许 Inf。

所有 backend一致。

---

# 31. Blom / inverse-normal transform finite sample audit

---

## R19-098：`blom_transform` 当前 count/rank应统一 finite mask

如果 row里有 Inf：

```text
不能既参与 n 又作为极端 rank
```

除非明确允许。

与最新 cs rank finite policy对齐。

---

# 32. group label ordering 与 deterministic identity

---

## R19-099：group operations 不应依赖 group label出现顺序

对：

```text
string
int
category
```

group labels重新排列但 membership相同，结果必须相同。

---

## R19-100：列顺序 permutation metamorphic test

对于对称/截面 operator：

```text
permute instrument columns
→ permute output columns same way
```

必须成立。

这能抓出 `rank_` tie position bug等。

---

# 33. 数学性质测试比固定 golden 更重要

---

## R19-101：为每个 operator声明 metamorphic properties

例如：

### rank

```text
strict monotonic transform invariance
column permutation equivariance
```

### zscore

```text
translation invariance
positive-scale invariance
```

### correlation

```text
translation invariance
positive-scale invariance
symmetry
```

### covariance

```text
translation invariance
scale covariance
symmetry
```

### beta

```text
y-scale covariance
x-scale inverse covariance
```

### neutralize

```text
group residual mean ≈0
size residual covariance≈0
```

---

## R19-102：每个 retained operator至少1个非平凡 metamorphic test

不能全部只靠几个手写 expected numbers。

---

# 34. prefix invariance 扩展到所有 custom numpy loops

---

## R19-103

对任意：

```text
for i in range(...)
rolling custom
event custom
pattern custom
```

自动：

```text
run on prefix T
run on prefix T+K
compare first T outputs
```

必须完全相同（允许浮点 tolerance）。

---

## R19-104：任何 full-sample statistic出现在 direct factor层都能被该测试抓出

这次 `rank_corr(d=0)` 就属于此类。

---

# 35. chunk invariance 要覆盖“缺失刚好落在 chunk boundary”

---

## R19-105

不要只随机 chunk。

专门构造：

```text
NaN at boundary
Inf at boundary
event transition at boundary
new high at boundary
group change at boundary
corporate action at boundary
session boundary
```

---

# 36. checkpoint serialization 本身也要有数值证书

---

## R19-106

对 stateful operator：

```text
full run
=
segment A
serialize
restore
segment B
```

比较：

```text
values
NaN mask
state
```

---

## R19-107：checkpoint float precision不能偷偷降低

禁止：

```text
float64 state -> JSON rounded -> restore
```

产生长期递归 drift。

---

# 37. 真实 backend parity 必须证明“native path真的跑了”

---

## R19-108

parity result记录：

```text
native_used=True/False
fallback_used=True/False
```

测试分类：

```text
native parity
fallback parity
```

不能混为一项。

---

## R19-109：production-certified native backend必须 `native_used=True`

否则它只能叫：

```text
supported_via_reference_fallback
```

不能叫 native certified。

---

# 38. backend parity hostile fixtures 继续扩充

---

## R19-110

新增：

```text
all equal
many ties
zero denominator
negative values
very large values
very small values
alternating signs
single valid sample
two valid samples
current row missing
internal hole
Inf
-Inf
long NaN block
duplicate group labels
string group labels
group missing
column permutation
```

---

# 39. 数值稳定性：高阶 moment / polynomial / exponential 类

---

## R19-111：高阶中心矩容易 overflow

`ts_moment(k)` 对大数和高 k：

```text
(x-mean)^k
```

容易 overflow。

### 必须整改

- 合法 k上限；
- scale-aware计算；
- overflow policy；
- cost/instability evidence。

---

## R19-112：poly2 regression 的时间坐标要 center/scale

否则长 index或大 window：

```text
t^2
```

condition number恶化。

使用：

```text
centered local t
```

并声明 coefficient对应的时间尺度。

---

# 40. operator description 与实际 sample support 一致

---

## R19-113

例如：

```text
Kurt
Skew
Corr
Beta
Regression
```

文档必须说明：

```text
minimum effective sample
missing topology
finite policy
ddof
```

---

# 41. 同类 helper 不应有两套不同数学 kernel

---

## R19-114

例如：

```text
rolling_beta
ts_beta local implementation
Beta wrapper
```

这种应统一。

---

## R19-115

例如：

```text
rank_ internal
cs_rank_01
pandas rank
numba rank
```

如果代表同 canonical，最终只能有一个 semantic reference。

---

# 42. 新增“数学 reference implementation”和“optimized implementation”两层

---

## R19-116

每个关键 operator family：

```text
reference
optimized
```

reference优先：

```text
清楚
正确
慢一点
```

optimized：

```text
Numba/Polars/DuckDB
```

所有 optimized 必须 differential test reference。

---

# 43. 禁止 optimized path 自己重新解释 missing/Inf/tie

---

## R19-117

fast kernel只实现：

```text
MathematicalSemanticCertificate
```

不能靠它自己的默认。

---

# 44. 参数 canonicalization 的 `positive_scale` 语义要修

---

## R19-118：`equivalence="positive_scale"` 当前 normalization 只看 total，不一定检查每个元素为正

如果允许负权重：

```text
positive_scale
```

这个名字就不准确。

### 二选一

- 真 positive scale：每个 weight >0；
- 或改为 `homogeneous_scale`，允许有符号权重按正比例系数等价。

---

## R19-119：structured/nested parameter 的 hash canonicalization要递归

例如：

```text
dict
nested tuple
list of config
```

内部 float noise也需要 canonical freeze。

不能只顶层 Sequence。

---

## R19-120：hash canonicalization不得改变 execution semantics

继续保留 R13 的正确原则：

```text
hash side
!=
execution side
```

但必须测试：

```text
same hash => declared semantic equivalent
```

---

# 45. factor identity 增加 mathematical-semantics version

---

## R19-121

至少包括：

```text
operator semantic hash
numeric semantics hash
parameter binding hash
history contract hash
backend-independent math version
```

---

## R19-122：实现修复必须让旧 factor identity失效

例如修：

```text
ts_regression_slope ddof
rank_corr
argmax origin
industry_size neutralize
```

不能复用旧 cache/evidence。

---

# 46. 自动找“数学定义和代码不一致”

---

## R19-123

新建 static/dynamic audit：

```text
scripts/audit_operator_math_contract.py
```

对每个 canonical：

```text
description
metadata
reference kernel
numeric semantics
history contract
```

做一致性检查。

---

# 47. 自动找 hidden kwargs

---

## R19-124

AST扫描所有 operator kernel：

```text
kwargs.get(...)
kwargs[...]
```

任何 key不在：

```text
param_names
param_aliases
context inputs
```

中：

```text
HIDDEN_PARAMETER
```

release fail。

---

# 48. 自动找 local casts

---

## R19-125

AST扫描：

```text
int(param)
float(param)
bool(param)
str(param)
max(...int(param)...)
min(...int(param)...)
```

如果是 declared scalar：

必须确认来自 central binder。

未经允许 local cast：

```text
LOCAL_PARAMETER_COERCION
```

---

# 49. 自动找 statistical sample masking drift

---

## R19-126

AST扫描：

```text
notna
dropna
np.isnan
np.isfinite
count
```

产物按 operator列出：

```text
sample_mask_policy
```

人工/机器确认与 certificate一致。

---

# 50. 自动找 duplicated semantic policy

---

## R19-127

检查：

```text
numeric_semantics
rank_spec
cross_section_spec
operator metadata
kernel constants
```

同一属性：

```text
tie
ddof
zero_std
Inf
```

不能多处不一致。

---

# 51. 自动做“reference vs optimized vs backend”差分

---

## R19-128

新建：

```text
scripts/audit_operator_differential_execution.py
```

每 canonical多组 hostile fixtures，比较：

```text
reference pandas
numba
polars
duckdb
composite lowering
```

---

# 52. 输出新的 R19 数学审计矩阵

---

## R19-129

生成：

```text
factor_engine/docs/R19_OPERATOR_MATH_AUDIT.json
factor_engine/docs/R19_OPERATOR_MATH_AUDIT.csv
factor_engine/docs/R19_OPERATOR_MATH_AUDIT.md
```

每 current retained canonical一行：

```text
canonical
math_definition
reference_impl
semantic_hash

axis_semantics
sample_validity
missing_topology
current_row_requirement

tie_policy
ddof
quantile_interpolation
zero_denominator
zero_std
domain_policy

partial_window_policy
minimum_effective_sample

history_kind
history_formula
anchor_policy

parameter_binding_complete
hidden_kwargs
local_casts

reference_smoke_pass
numba_native_pass
polars_native_pass
duckdb_native_pass
fallback_pass

prefix_invariance
chunk_invariance
column_permutation
metamorphic_properties

math_defect
semantic_drift
fix_action
```

---

# 53. 本轮新增 blocker

---

## R19-130

建议新增：

```text
M01_MATH_REFERENCE_MISMATCH
M02_DDOF_MISMATCH
M03_FINITE_SAMPLE_MISMATCH
M04_TIE_POLICY_MISMATCH
M05_CURRENT_ROW_POLICY_MISMATCH
M06_PARTIAL_WINDOW_MISMATCH
M07_HIDDEN_PARAMETER
M08_LOCAL_PARAMETER_COERCION
M09_PARAMETER_BINDING_DRIFT
M10_BACKEND_NATIVE_FALLBACK_HIDDEN
M11_PREFIX_INVARIANCE_FAIL
M12_CHUNK_INVARIANCE_FAIL
M13_HISTORY_FORMULA_HEURISTIC
M14_SAMPLE_ANCHOR_UNDECLARED
M15_DOMAIN_POLICY_MISMATCH
M16_NUMERIC_SEMANTICS_DUPLICATE
M17_GHOST_EXECUTION_CONTRACT
M18_NEUTRALIZATION_MATH_DEFECT
M19_RANK_AXIS_DEFECT
M20_REGRESSION_SCALE_DEFECT
```

---

# 54. 必须写的 targeted regression tests

---

## R19-131：rank_corr no-lookahead

构造：

```text
前 T 天固定
未来 T+1...改变
```

验证前 T 日不变。

---

## R19-132：regression slope exact OLS

简单：

```text
y=2*x+3
```

任何窗口有效时 slope=2。

修复前 ddof混用会暴露偏差。

---

## R19-133：industry+size FWL

构造：

```text
行业A/B size均值不同
y同时受行业+size影响
```

neutralized result：

```text
每行业均值≈0
cov(resid, demeaned log size)≈0
```

---

## R19-134：argmax tie/origin

窗口：

```text
[5,1,5]
```

明确 expected：

```text
index from oldest
age
tie latest/first
```

各 canonical固定。

---

## R19-135：ts_product zeros/sign

```text
[2,0,3]
[-2,-3]
[-2,3]
```

---

## R19-136：corr current-row missing

窗口历史 pair足够、当前row NaN。

Numba/Pandas/Polars/DuckDB必须一致。

---

## R19-137：truthiness invalid condition

```text
2
-1
Inf
```

production reject。

---

## R19-138：column permutation rank tie test

---

## R19-139：numeric-string binder/kernel separation

---

## R19-140：ParameterCanonicalizer.hash_key可调用且 deterministic

---

# 55. 所有 retained operators 的 R19 动态全审

不能只修上面点名的。

运行：

```python
load_all()
for canonical in OperatorRegistry.list_canonical():
    ...
```

每个 operator至少走：

```text
parameter contract
sample contract
history contract
numeric contract
reference execution
metamorphic test
prefix/chunk test
backend route audit
```

不适用项：

```text
N/A with reason
```

不能：

```text
SKIP because not tested
```

---

# 56. 不要为了修 parity 让所有 backend 一起变错

reference 定义优先。

流程必须：

```text
先确认数学定义
→ reference implementation
→ golden/metamorphic
→ 再修 optimized backend
```

不能看到 Pandas与DuckDB不同，就随便让 Pandas去跟DuckDB。

---

# 57. Definition of Done

全部满足才算 R19 完成：

- [ ] `ParameterCanonicalizer.hash_key()` 可用且 deterministic；
- [ ] planning/runtime/hash parameter binding共享一个 normalized bound；
- [ ] scalar validator只有一个 authority；
- [ ] numeric-string只在 declared binder阶段转换；
- [ ] parameter aliases只有 metadata-derived authority；
- [ ] active_when使用 canonicalized bound；
- [ ] CLI strict真正检查全部 release_blocking；
- [ ] duplicate candidate与confirmed exact duplicate分开；
- [ ] dead-param choices使用真实 choices；
- [ ] MISSING sentinel绝不作为 runtime default传入；
- [ ] dead-param coverage达到 parameter级100%；
- [ ] behavior fingerprint canonicalize NaN payload；
- [ ] manifest-gap intrinsic/contextual分开；
- [ ] `rank_corr(d=0)` full-sample lookahead被彻底移除；
- [ ] 真正的 ts/cs rank-corr拆分；
- [ ] internal rank tie policy正确且不依赖列顺序；
- [ ] `ts_regression_slope_` OLS ddof错误修复；
- [ ] industry+size neutralization改为 joint/FWL正确实现；
- [ ] invalid market cap不再 clamp为1；
- [ ] 所有 statistical sample统一 finite policy；
- [ ] `ts_corr` Numba/Pandas current-row policy一致；
- [ ] `ts_beta`共享 reviewed rolling_beta semantics；
- [ ] optional fastpath强制 parity test；
- [ ] native backend fallback有 lineage；
- [ ] argmax/argmin origin/tie三套定义收敛为一套；
- [ ] `ts_product`数学定义正确或重新命名；
- [ ] decay partial-window/missing semantics统一；
- [ ] hidden kwargs=0；
- [ ] local scalar coercion=0；
- [ ] ts_sharpe/ts_autocorr隐藏 min_periods修复；
- [ ] ann_factor为policy而非 alpha knob；
- [ ] returns/log-return finite/domain semantics明确；
- [ ] neutralize缺group不再 silent global demean；
- [ ] 当前具体 rank/quantile exact duplicates确认并合并；
- [ ] ConditionBool与truthiness contract不冲突；
- [ ] numeric semantics registry无重复 canonical key；
- [ ] expanding/cumulative anchor policy显式；
- [ ] retained operator不再依赖 heuristic history fallback；
- [ ] stale execution-contract names=0；
- [ ] minimum effective sample machine-readable；
- [ ] MissingTopologyPolicy完整；
- [ ] trading bar/calendar/session slot单位明确；
- [ ] rank tie/range/singleton语义完整；
- [ ] quantile interpolation完整；
- [ ] math domain policy跨 backend一致；
- [ ] 每 retained operator至少1个 metamorphic property test；
- [ ] 所有 custom rolling通过 prefix invariance；
- [ ] stateful通过 chunk/checkpoint parity；
- [ ] native backend evidence证明 native path真的执行；
- [ ] R19 math audit matrix覆盖100%；
- [ ] 全量 tests/evidence重新生成；
- [ ] R19 blockers全部清零或 operator被正确移出 direct layer。

---

# 58. 最终执行指令

请代码整改 AI 现在直接执行：

1. 读取服务器真实 HEAD；
2. 不重复 R17/R18；
3. 动态加载当前 registry；
4. 先修 R19 已确认的明确代码错误：
   - `ParameterCanonicalizer.hash_key`;
   - `rank_corr(d=0)`;
   - `ts_regression_slope_` ddof；
   - `industry_size_resid_panel_`；
   - `size_resid_panel_` invalid cap；
   - ts_corr NumPy/Numba current-row parity；
   - ts_beta old semantic path；
   - argmax/argmin origin/tie；
   - ts_product；
   - decay partial/missing；
   - hidden kwargs/local casts；
   - truthiness冲突；
   - release-blocking CLI；
5. 建立统一 MathematicalSemanticCertificate；
6. 对全部 retained canonical跑 static + dynamic math audit；
7. 为每个 operator建立/确认 reference math implementation；
8. 跑 hostile fixtures；
9. 跑 metamorphic tests；
10. 跑 prefix invariance；
11. 跑 chunk/checkpoint invariance；
12. 强制 Numba ON/OFF parity；
13. 强制 Polars/DuckDB native/fallback路径标记；
14. 修完所有差异；
15. 重生 semantic hash/cache/evidence；
16. 生成 `R19_OPERATOR_MATH_AUDIT.{json,csv,md}`；
17. 最终报告：
    - current canonical count；
    - math defects found/fixed；
    - hidden params found/fixed；
    - local casts found/fixed；
    - backend parity failures/fixed；
    - prefix failures/fixed；
    - history heuristic dependencies remaining；
    - exact duplicate confirmed；
    - R19 blocker remaining count；
    - 全量 test结果。

**本轮最终标准不是“代码能执行”，而是：同一 canonical 在所有合法路径上只存在一个数学定义，而且这个定义经得起参数、缺失、时间、chunk、backend 与组合表达式的系统性检验。**
