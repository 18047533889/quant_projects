# FE rank 候选执行身份（2026-10-03）

`REPRESENTATION_RANK` 的横截面 average 分支执行 FE canonical `rank`，
调用注册表时固定 `backend="pandas_numpy", mode="any"`。
该路径不是 Polars 原生执行；FE 的 Polars 注册并不会因此被自动选择。

此前去重记录了不存在的 `factor_engine.adapters.cs_rank` 路径字符串。
现在去重绑定真实算子实现哈希、调用契约哈希、语义版本、算子类型、
适配器源码哈希及 NumPy/pandas 版本和输入转换约定。
执行签名仍包含参数、训练上下文、方向与自然时间尺度；不改变训练选择目标、
验证确认规则或测试集隔离。非 average 的排名仍走原有 FP 分支。

实现身份无法建立时，不进行这组候选去重，保留各候选独立执行；
不会用一个相同路由字符串假冒已认证实现。身份绑定是去重保护，不是生产因子发布资格。

验证：替换 FE rank 实现的反例使实现哈希和执行签名均变化。
正式树 `test_research_execution_dedup.py` 和 `test_repair_execution.py` 联合回归
69 passed，53 warnings，35.88 秒。警告仍包含 FE 物理实现元数据缺失。
本改动不证明全部优化方法无缺陷，也不代表所有候选都能提升收益。
