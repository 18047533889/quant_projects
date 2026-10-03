# FE 原生排名候选的有界交叉测速

在库目录安装 `.[production,fast,benchmark]`，再显式调用：

```python
from factor_preprocess.scripts.benchmark_fe_native_rank import benchmark_fe_native_rank
report = benchmark_fe_native_rank(values)
```

`values` 是调用者已经加载的 ndarray 小 tile，资产必须位于最后一轴。
对于真实因子值 T×N×F，可以传 `values.transpose(2, 0, 1)` 的视图，使
最后轴仍是同一日期的资产截面；不能把多个日期放进一个排名组。
函数不加载 COS、不保存因子数据、不发布 receipt、不修改默认执行路径。

先在有界分组 tile 上比较实际 FP 排名和 FE 原生排名的全部值及 NaN 掩码；
这里 FP 是等价性基线，不是独立数学 oracle。适配器另有独立排序/tie-run
参考测试。随后以 FP → FE → FE → FP 顺序运行完整函数，每次只保留一份
完整结果。计时包括输入载体转换、排序和结果分配；输入加载、指纹与结果
校验在计时外。每次前后校验输入、模块源码、实际线程池及运行库身份。
时间为 NaN/Inf/非正数或身份变化都会拒绝返回成功候选报告。

默认结果预算 256 MiB，按每次完整结果和参考比较的两份同时存活结果检查。
这不是峰值 RSS 门禁，不包含输入、SciPy/Polars 内部工作空间；真实大面板
仍需上游资源预检和读取分块。完整资产截面不能为满足 chunk 预算而拆散。

报告只表示 caller-supplied in-memory 候选测量；没有 COS 来源证明、全 FE
注册准入或跨运行时的 loaded-code closure 证明。即使某次 FE 更快，也不能
据此直接宣称全量因子最快或切换生产 auto。只有真实请求完整搬运边界、
稳定相反顺序优势、独立正确性与正式准入都满足后，才可以考虑默认接入。
