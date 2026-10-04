# 批量 Source 评估：随机配对验证协议

本协议用于补充现有真实 COS ABBA 性能资格，不替代资格、不修改历史报告。
仅有两个因子、内核微基准或 CPU/GPU 相互一致，都不能证明真实大批量请求最快。

## 模块与执行边界

- `scripts/source_profile_paired_trials.py`：纯调度与轻量结果聚合。
- `scripts/source_profile_randomized_runner.py`：串行执行后端，每个输出都经过独立 oracle、
  实际测量收据合同与配对比较；不配置 provider、不写资格缓存、不授予 auto 路由。
- 原有 ABBA profile 与新进程 provider 验证继续各自使用原有严格报告格式。

调用 `produce_source_route_profile_randomized_trials` 时需要 `oracle`、`run_kwargs`、
显式 `seed`；默认 `pairs=3`，可显式选择 3 至 16。真实运行使用现有的
`_checked_run_backend`，每一轮重新检查资源；测试 seam 不等价于真实 COS 证明。
同一批不允许改变源码、数据、标签、精度、线程/设备配置或请求参数。

调度使用局部随机生成器；每一对恰好一次 `cpu` 和一次 `cuda_strict`。
两种执行先后次序的数量差不超过 1，再按 seed 打乱。相同 seed 可重现顺序，
但不能保证运行耗时完全相同；这不是每对独立抛硬币的非平衡设计。

## 正确性与身份门槛

每次实际输出都必须满足相同的完整 typed context（运行前、运行后及跨轮次）。
context 摘要包含源内容、源码、运行环境、配置等字段，不能只使用请求名。
oracle 检查全部指标的形状、有限/NaN/正负无穷掩码、观察计数与数值误差。
每对另做 CPU/GPU 全输出比较。异常、后端冒名、漂移、OOM 重试或错误输出不得
被吞掉后当作有效测量；任何失败都不能产生获胜后端。

这里只接受 `evaluate_factor_source_batch_wall_v1` 整段 API 墙钟耗时。
不接受仅 CUDA kernel、仅 IC、仅传输或另外拼接的估算耗时。外部 oracle、
预热与 context 观察不在 API 计时内，必须明确披露这些边界及冷热缓存条件。

## 速度判定

令第 i 对的 CPU/GPU API 耗时为 c_i、g_i。耗时必须为有限正数。
CPU 的配对胜数为 sum(1[c_i < g_i])；GPU 同理；相等视为平局。
候选获胜后端必须同时满足：

1. 所有独立 oracle 与配对正确性门槛通过。
2. 严格超过一半的配对胜出。
3. 所有轮次的全局耗时中位数也严格更小。

多数与中位数不一致、平局或无严格多数时 `winner=None`，不能偷偷选 GPU。
例如 CPU=[100,1,2]、GPU=[99,0.9,200]，GPU 胜出两对但中位数更慢，不能判其获胜。
`aggregate_pass` 只表示正确性标志通过；它为 True 不保证存在速度获胜者。
这些标志及报告是调用者可信证据，不是签名证明，也不是所有形状的普遍最快保证。

## 内存、冻结与未完成事项

runner 每次只保留一对输出供比较，处理后释放；不累积完整因子面板。
source 自身仍负责分块和关闭，完整 API 测量仍必须遵守 CPU/RAM/VRAM 余量门槛。
QE、DataAccess、FactorOptimizer、FactorPreprocess 的所有 `.py`（包括 tests）均进入
当前性能资格的源码身份。必须完成这些文件的修改与测试后再冻结、重新测量，
不能修改测试后仍称旧报告是当前源码资格。此轮次实现不证明真实 F48/F61 已完成实测。

后续必须在当前源码执行真实多因子配对、独立 oracle 与新进程 provider 验证；
普通内存/衍生因子源的资格接入和实际应用调用链仍须分别实现和验证。

## 回归证据与限制

helper 32 项测试包含有限大数中位数溢出、次正规最小正数、超大整数耗时、
不可哈希后端 token、绕过长度的自定义迭代器、多数与中位数相反等反例。
其中新增边界命令先实际出现 4 failed / 1 passed，修复后同一命令 5 passed。
runner 6 项测试验证三个配对、真实比较/测量/oracle 编排、错误后端、两种
context 漂移、oracle/parity 失败及上一配对对象释放。fixture 是纯编排脚手架，
不是真实 IC 数学数据，不作为指标正确或真实性能资格的证明。

相关四文件的独立复跑命令：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  quant_evaluator/tests/test_source_profile_paired_trials_oct04.py \
  quant_evaluator/tests/test_source_profile_randomized_runner_oct04.py \
  quant_evaluator/tests/test_source_profile_measurement_oct04.py \
  quant_evaluator/tests/test_source_profile_abba_oct04.py
```
