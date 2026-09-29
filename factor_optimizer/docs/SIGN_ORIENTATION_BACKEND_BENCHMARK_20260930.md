# SIGN_ORIENTATION FE 后端 A/B 基准（2026-09-30）

## 结论

保留 `SIGN_ORIENTATION` flip 的 Polars-primary 路由。此选择是明确的后端路由策略，
不是“Polars 在所有规模或环境中都最快”的承诺：本次样本中 Polars 在 100k 行有可见优势，
在 1M 行则与 pandas_numpy 基本持平。

## 方法与资源边界

在 server-c 正式工作树使用项目根 `.venv/bin/python`，基准对象是 FO 的
`ValueRepairPlan.execute(..., allow_research=True)`。每次计时包含完整执行路径：
长表校验与重复身份检查、Series 到 NumPy 的值复制、所选 FE 后端输入构造、FE `neg`
计算、结果形状检查和输出 Series 重建。计划编译在计时区间外，避免把无关搜索/编译成本
混入后端差异。

使用可复现随机种子 8128 生成 100k 和 1M 行合成长表；`asset_id` 为 int32，日期按
1000 行一个交易日生成，键无重复，值为标准正态 float64。基础 DataFrame 的 pandas
内存统计分别约 1.9 MiB 和 19.1 MiB。测量前服务器报告 48 GiB 可用内存；每个样本串行
执行，测后释放数据并回收，不复制仓库或数据集。

通过仅针对 `neg` 的 registry lookup 屏蔽另一后端，分别强制 FE Polars 与 FE
`pandas_numpy`，其余 FO adapter 路径保持一致。先各记录一次首次调用时间，再为每种后端
各做一次预热；热测进行 12 轮配对，每轮两种后端各执行一次，运行先后顺序逐轮交替。
每次计时前做垃圾回收。计时使用单进程 `perf_counter`；没有 CPU 绑核，也没有跨机器/进程
重复，故范围用于描述本次运行波动，而非统计显著性检验。

## 结果

| 输入行数 | 首次 Polars / pandas_numpy | 热测 Polars 中位数（范围） | 热测 pandas_numpy 中位数（范围） | 比率¹ |
| ---: | ---: | ---: | ---: | ---: |
| 100k | 8.20 / 3.10 ms | 3.79 ms（3.30–4.51） | 4.81 ms（3.62–5.78） | 1.270x |
| 1M | 34.40 / 36.50 ms | 31.49 ms（28.16–41.35） | 31.08 ms（27.15–35.49） | 0.987x |

¹ 比率为 pandas_numpy 热测中位数 / Polars 热测中位数，大于 1 表示 Polars 在本次样本中
更快。首次调用顺序固定为 Polars 后 pandas_numpy，且 Python/依赖导入已完成；该列是
首次 adapter 调用，不是新进程启动到完成的总冷启动成本，不能据此比较导入开销。

两种后端在两个规模都逐元素相等，并且与输入值的 NumPy unary minus 结果精确相等。
100k 样本里 Polars 中位数约快 21%；1M 的两组范围明显重叠，约 1.3% 的差异小于本次
观察到的运行波动。保留 Polars-primary 的理由是中等规模样本的实测收益及大样本未见
有意义退化，而不是普适性能排序。

## 适用范围

这是合成 float64 数值长表、当前 server-c、当前 FE/FO 工作树和 Python 环境下的
adapter 微基准。真实因子类型、机器负载、Polars/Pandas 版本、不同长表布局和更大规模
可能改变结果。没有性能断言加入 pytest；现有测试负责验证默认 Polars 路由、两种后端
数值 parity（含 `+0.0`/`-0.0`、NaN、±Inf 与空输入）以及分块结果稳定性。任何“最快”
描述都应限定到本次测量条件，不能外推到所有规模或部署环境。
