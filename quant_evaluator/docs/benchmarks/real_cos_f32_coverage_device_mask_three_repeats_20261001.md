# F32 coverage：GPU 原生掩码与完整评估实测

原始回执：[JSON](real_cos_f32_coverage_device_mask_three_repeats_20261001.json)。
本次只认证 `coverage` 的真实大面板请求，不能外推到其他指标、设备或数据规模。

## 数据与验证

我们通过项目 DataAccess 读取已验证的 COS 因子对象，得到 `(2586, 5461, 32)` 的
float64 因子面板及 bool 有效性掩码。清单 SHA-256：
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
读取与标签构造耗时 128.613 秒，下面的评估耗时不包含这部分。

我们按 CPU、CUDA、auto、auto、CUDA、CPU 顺序启动隔离子进程，每个执行三次
完整公开评估，包含诊断及结果身份计算。18 次结果对齐、重复数值对齐及重复一致性
检查均通过，配置 hash 相同。源码和 HEAD 的前后检查一致。

QE 全套回归：4594 passed、26 skipped、108 warnings；GPU 集中回归：177 passed。
跳过项需要专用输入或当前没有对应内核，不能作为这些场景已通过的证据。

## 耗时（秒）

| 轮次 | 请求后端 | 三次实际路由 | 冷运行 | 暖运行 1 | 暖运行 2 |
|---|---|---|---:|---:|---:|
| 1 | cpu | CPU / CPU / CPU | 15.829 | 6.536 | 6.762 |
| 2 | cuda_strict | CUDA / CUDA / CUDA | 15.329 | 5.458 | 5.696 |
| 3 | auto | CPU / CUDA / CUDA | 15.824 | 6.637 | 5.400 |
| 4 | auto | CPU / CUDA / CUDA | 15.798 | 6.605 | 5.461 |
| 5 | cuda_strict | CUDA / CUDA / CUDA | 15.720 | 5.717 | 5.437 |
| 6 | cpu | CPU / CPU / CPU | 15.902 | 6.904 | 7.150 |

显式 CUDA 两轮暖运行中位数均为 5.577 秒；CPU 为 6.649、7.027 秒。
auto 为 6.019、6.033 秒。auto 首次切到 CUDA 的调用较慢，下一次 CUDA 调用为
5.400、5.461 秒。当前 auto 的冷身份缓存规则仍先选 CPU，因此不能声称 auto
已经等于本次最快后端。后续应单独认证首次即选 CUDA 的规则，而非修改本次回执。

CUDA 均使用宽度 32、单个因子 tile，峰值显存 5,083,974,144 字节。
预检估计工作内存 33,792,000,000 字节，保留 8 GiB 可用内存余量，并执行显存准入。

## 本次实现边界

`DeviceEvaluationSession.stage_masked_factors` 上传原值与 bool 掩码，创建自有设备
值缓冲区，再用 CuPy 的设备侧反转和 `copyto` 应用 NaN。它不修改调用方的 host
或 device 值数组。整数和浮点的 NaN 提升遵循原 NumPy `where` 语义，再遵循
现有 precision policy。分块估计包含同时存在的两个 bool 缓冲区：`2*T*N*tile` 字节。

probe、内存批次、单遍数据源及多周期评估都接入同一掩码上传方法。
数据源元数据未声明是否含掩码时，我们预留掩码空间；无掩码路径保留原上传接口。
OOM 时仍按现有策略释放 tile 并减小宽度。拼接后释放上一 tile 的结果引用。
标签掩码的 host 处理不在本次优化范围。

因子诊断使用 C-order `ravel`，连续面板可取视图，交错因子面板只复制单个因子。
先前实测拒绝了反复扫描 strided view 的方案：约 12.1 秒，对比连续遍历约 5.1 秒。
相关原始证据见 [诊断 A/B](real_cos_f32_diagnosis_copy_ab_20261001.json)。

## 复现

在 server-c 的 `/home/sunhaiwei/quant_projects`，先确认磁盘与内存余量，使用已配置
网关及现有镜像；脚本先预检，拒绝时不得绕过。

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_SKIP_COS_MIRROR=1 \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_metric_batch \
  --run --factors 32 --metrics coverage --batch-repeats 3 --compact \
  --output quant_evaluator/docs/benchmarks/NEW_UNIQUE_RECEIPT.json
```
