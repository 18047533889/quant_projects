# 缺失指示器：FE 原生 Polars 与兼容分派

## 计算口径

对每个输入值 $x_i$，输出 $m_i=1$ 当且仅当 $x_i$ 是 null/NaN，
否则 $m_i=0$。正负无穷不是缺失，有限极小值也不是缺失。
输出为 float64 Series，保留输入行顺序、原始索引和 value 列名。
时间/资产身份必须非空且唯一；重复 pandas index 本身允许。

## 路由与身份

公共入口是 `get_default_registry().get_execution("missing_indicator")`。
支持的整数及不超过 64 位浮点走 FE 注册的 `is_null` 原生 Polars 表达式，
只传值向量，不创建稠密时间×股票面板；不是 NumPy 统计后再包装 Polars。
对象、时间等不支持的类型走既有 FE pandas_numpy 路径，身份明确报告该后端。
每次原生执行从同一 registry snapshot 获取算子和 catalog，避免热替换旧算子。
冷身份仅为 planned_native_candidate，不表示已经执行。

扩展浮点（Linux float128/longdouble）不能送进 Arrow/Polars。
FE pandas 兼容路径前仅将扩展标量 boxing 为 object，解决 pandas pivot
没有 float128 kernel 的错误。没有转成 float64，不改变有限性/缺失性，
也不修改调用者 DataFrame；正常 UInt64 最大值继续走原生路径。

## 验证与速度证据

2026-10-03 修复后的三个回归文件合跑：23 passed，47.22s。
包含公共注册 float128、UInt64 最大值、nullable 数值、无穷、重复索引、
行顺序、身份校验、registry 热替换和显式 research fallback。
最终运行时源码的全库回归：912 passed、1 xfailed、95 warnings，122.18s；
API 索引检查通过。这不证明所有 dtype 都已测试或不存在其他问题。

此前 `benchmarks/missing_indicator_fe_native_20261003.json` 对 30 万行
交错 AB/BA 七轮测得原生中位 0.047853s、旧 FE pandas 0.360917s，约 7.54 倍。
该速度报告是扩展浮点分派修复之前的快照，不是当前源码指纹认证；
共享服务器业务负载未完全控制，不能推广为所有输入/机器都快 7.54 倍。
本修复不证明全部 dtype 或全部预处理方法无 bug。
