# F48 COS 源批次 GPU 宽度对照，2026-10-01

本轮在 server-c 的 NVIDIA L20 上读取既有 COS 因子，评估 2586 日 × 5461 股票 × 48 因子，指标为 Rank IC、quantile spread、factor turnover rate。四个独立严格 CUDA 进程按 `2,4,4,2` 顺序运行；本次没有运行 CPU 对照或新增 auto 路由。

| 顺序 | 宽度 | 评估总耗时（s） | 源读取墙钟累计（s） | OOM 重试 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 57.076043 | 50.635890 | 0 |
| 2 | 4 | 58.091070 | 52.237907 | 0 |
| 3 | 4 | 59.468512 | 53.594839 | 0 |
| 4 | 2 | 58.590373 | 52.488486 | 0 |

宽度 2 的中位耗时为 57.833208 秒，宽度 4 为 58.779791 秒，前者低 1.61%。两个宽度各有两次测量，共享机器的负载也会影响耗时；你不能据此认定宽度 2 在其他请求或机器上最快。

源读取累计耗时占本次评估总耗时约九成。该累计值来自每个 tile 的源读取墙钟，包含等待及源物化；它不是独立的网络下载计时。下一步需要分解上下文建立、绑定读取、Arrow/Pandas 转换和轴对齐，再选择读取优化方案。

三次相对首组的比较均通过：每次覆盖 48 个因子、三项指标的 144 个标量，以及有限掩码和观测计数。GPU 峰值分配：宽度 2 为 2,586,991,616 字节，宽度 4 的各组数值请查 [JSON](research_f48_source_gpu_width2_4_20261001.json)。本次结果没有证明 CPU/CUDA 等价，也没有检查未请求的序列型输出。

COS adapter 使用 `auto` 预取、2 个工作线程、512 MiB 预取预算和 4 GiB 源预算。共享估算器给出宽度 2 的逻辑源载荷 1,836,108,344 字节，宽度 4 为 3,135,345,776 字节；估算不等于进程峰值 RSS。控制器在启动子进程前准入两个宽度，只准备一次绑定清单及选定记录的共享轴索引，临时目录结束后释放索引。

源码前后核验见 [provenance JSON](research_f48_source_gpu_width2_4_20261001_provenance.json)：HEAD 为 `4a98bd73240036d38ab8133f29666c88dd014ad6`，7 个相关源文件的 SHA-256 前后相同。原始结果 SHA-256 为 `1153a58f14a1bcecea81f21f36677cff6e444e2a4adecfacdfd85dbdbf583587`，保存文件的最终状态为 `complete`。

## 复现

从 `/home/sunhaiwei/quant_projects` 执行，指定未存在的输出文件：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_SKIP_COS_MIRROR=1 \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_source_batch \
  --factors 48 --gpu-tile-widths 2 4 --source-adapter cos \
  --cos-prefetch auto --cos-prefetch-workers 2 \
  --max-prefetch-memory-mib 512 --max-source-memory-mib 4096 \
  --max-object-mib 128 --max-total-mib 4096 \
  --metrics rank_ic,quantile_spread,factor_turnover_rate \
  --days 2586 --assets 5461 --output NEW_UNIQUE_REPORT.json
```
