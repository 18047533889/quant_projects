# R6 算子默认调用与当前边界

正式代码只在 server-c 的 /home/sunhaiwei/quant_projects 修改。
未创建分支、工作树或仓库副本，未提交、推送、部署或发布生产因子。

## 已验证的调用方式

以下是模板：source 是你已经配置并授权的数据源，sink 是已经配置的结果接收器。
这里明确使用研究模式，不伪造生产认证。算子性能参数保持默认；不同算子需要的经济输入仍必须提供。

```python
from factor_engine.api.cleaned_ops import make_cleaned_call_factory as op
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine

engine = FactorEngine(
    backend=build_backend("pandas"),
    data_source=source,
    run_mode="research",
)
factors = [
    Factor(name="kurt_default", expr=op("ts_kurt")(col("close"))),
    Factor(name="sma_cn_default", expr=op("ts_sma_cn")(col("close"))),
    Factor(name="gaussian_default", expr=op("cs_rank_gaussian")(col("close"))),
]
report = engine.run_many(factors, result_policy="sink", sink=sink)
```

这份例子的重点是默认算子参数和统一批量落值，**不是证明 pandas 在所有形状下最快**。
实际测试 test_r6_repaired_batch_integration.py 用 pandas 和 polars_long 两条路径，
把八个不同算子的结果逐个交给内存接收器，校验数值与资源归还。
测试资源快照是受控替身，不是实际 COS 持久写入或服务器 80% 内存压力测试。
本轮没有重新证明 auto 的全部物理区域准入、GPU 或十万因子吞吐。

## 不能省略的真实输入

- 多输入算子必须给真实输入：例如 Local Moran 的四个面板、分组标签、收益面板、财务期间与版本面板。不会用价格或随机值冒充。
- 分钟/日内算子需要对应频率、时钟和日历。部分输出是日频，不是原分钟面板。
- 静态图算子支持 StaticAdjacency 或严格标记的 N×N DataFrame；自节点不参与邻居聚合。当前是直接 Python typed call，不支持动态 date×N×N 图或 DSL 矩阵字面量。
- 递归 SMA、hump 等仍要求完整历史；没有实现检查点恢复的算子不会标成可独立分块。
- 小样本、常数窗口、无邻居或缺失支持不足返回 NaN 属于明确数学边界，不以零掩盖。

## 验证状态的含义

remaining-current29.log 中 1,757/1,757 research_callable 表示研究调用契约完整，
不表示 1,757 个算子已经逐一通过完整独立数值认证。
同一快照明确给出 executable_unverified=1,757、production_callable=0。
后者表示没有链接到可信生产认证，不表示所有研究内核都不可用。

最终回归：runtime-wide29-final.log 802项通过、0跳过；operators-wide28-final.log 499项通过、0跳过。
导入顺序和 SQL 参数验证问题已修复。24个动态回归后端另外完成普通/全新加载数值对照，
dynamic-regression-full-final.log 43项通过，含独立最小二乘数值参照。
这些是测试用例数，不是已经独立认证的算子数量；详细范围见 R6_PROGRESS.md。

## 性能与转换

default-kernel-performance25.log 记录 96×4 合成面板、三次中位数，包含输入和输出转换。
五项跨后端最大数值差为0；Polars端到端比率约0.17–0.78倍，说明小面板转换开销可能更大。
这不是 GPU 测试，也不是十万因子测试。性能结论必须带数据形状和转换成本。
