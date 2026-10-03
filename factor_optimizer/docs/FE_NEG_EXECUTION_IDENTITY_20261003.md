# FE neg 执行身份（2026-10-03）

`SIGN_ORIENTATION` 的 `flip` 实际调用 FO `_execute_fe_neg`：优先请求 FE canonical
`neg` 的 Polars 注册；仅当该注册不存在时回退到 `pandas_numpy`。去重签名现在按
该适配器真实选中的 backend 绑定 FE 算子实现哈希、契约哈希、语义版本、算子类型、
适配器源码哈希，以及 NumPy、pandas 和 Polars 版本。`keep` 仍是普通乘以 `+1`，不查找
FE 算子。若 backend 或实现不能认证，候选保留独立身份，不因同一路由名被合并。

TRAIN 上下文、变换参数、自然时间尺度和方向仍参与执行签名；此身份变化不改变
训练目标、TRAIN-only 拟合边界、验证采用规则或测试集隔离。本改动只保护候选去重，
不声称有执行性能收益或生产发布资格。

独立 CPU 测试覆盖 Polars-first、真实 pandas fallback、两种 FE 注册均缺失时的 fail-closed、
算子替换后的签名变化，以及 `keep` 路径不访问 FE。测试比较的是 FO 实际调用的注册，
替换算子继续委托原 FE kernel 计算数值。
