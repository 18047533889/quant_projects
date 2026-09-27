# F32 全历史双链 CPU/CUDA/auto 实测（2026-09-28）

范围：server-c 正式工作树，DataAccess 绑定 COS landing manifest
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
32 个 verified 因子，2586 个交易日 × 5461 只股票 × 32 因子，
float64；决策 t、执行 t+1、标签 `AdjVwap(t+2)/AdjVwap(t+1)-1`。
设备 NVIDIA L20。所有调用经过公开 `evaluate` 入口，
OpenBLAS/OMP/MKL 单线程。期间服务器有其它任务并发；数值对拍与
这台服务器的实测耗时成立，不把绝对秒数外推到其它机器。

## 装载和资源边界

新版逐因子装载器单独载入 F32 用时 124.916 秒，父进程 RSS 峰值
13,838,164 KiB；后续完整评估的父进程峰值约 13.7–13.9 百万 KiB。
rank 链 CPU/GPU worker 峰值约 23.7/25.4 百万 KiB，GPU pool 峰值
9,451,395,584 字节。分位链四轮 CPU worker 峰值
23,688,604/23,865,144 KiB，CUDA worker 峰值
28,326,304/28,369,552 KiB，GPU pool 峰值 4,152,787,456 字节。
运行前检查了 RAM、显存及磁盘余量；没有清理其它任务数据。

## 完整 rank 链

指标：`rank_ic`、`rank_ic_series`、`ic_std`、`ic_ir`。
先行双调用：CPU 冷/热 81.849/78.405 秒，CUDA 38.160/33.895 秒；
四项完整制品与配置哈希对拍通过。

| 交错轮次 | 请求 | 实际后端 | 冷启动 / 热调用（秒） |
| --- | --- | --- | ---: |
| 1 | CPU | CPU | 82.945 / 79.105 |
| 2 | CUDA strict | CUDA | 38.286 / 34.181 |
| 3 | CUDA strict | CUDA | 38.634 / 34.349 |
| 4 | CPU | CPU | 81.979 / 78.837 |

四轮配置哈希均为
`4ccd501dc5dfa8bfc20722aa3dc4a9072de5298c4b818086dc92acd7ce08ae80`；
四项完整制品逐轮通过严格比较器。加入精确自动路由后，
新一轮 CPU 参考为 84.180 秒，两轮 `auto` 均实际走 CUDA，
冷/热分别为 38.699/34.485 和 38.453/34.310 秒，
配置哈希相同，四项完整制品均对拍通过。

## 完整分位链

指标：`quantile_returns_full`、`quantile_returns_daily`、
`quantile_spread`、`quantile_monotonicity`、
`daily_quantile_monotonicity_rate`。
先行单调用 CPU 167.441 秒、CUDA 34.303 秒，五项完整制品对拍通过。

| 交错轮次 | 请求 | 实际后端 | 冷调用（秒） |
| --- | --- | --- | ---: |
| 1 | CPU | CPU | 168.197 |
| 2 | CUDA strict | CUDA | 34.749 |
| 3 | CUDA strict | CUDA | 34.886 |
| 4 | CPU | CPU | 167.598 |

四轮配置哈希均为
`95b04566a9f24426f4177e8b4ad09295a3540c56d4d7dbc10027eef76670d71b`；
五项完整制品逐轮通过严格比较器。加入精确自动路由后，
新一轮 CPU 参考为 167.892 秒，两轮 `auto` 均实际走 CUDA，
34.682/34.729 秒，配置哈希相同，五项完整制品均对拍通过。

## 自动路由边界

只针对上述精确形状、两个完整指标集合、float64、默认参数、
无特殊输入、单 NVIDIA L20。rank 链需实际及有效空闲显存均
不少于 14 GiB；分位链不少于 8 GiB。显存不够、设备不符、
自定义参数、子集、未经认证的混合请求、相邻形状均保留 CPU；
三个单指标另有独立认证，见 `real_cos_f32_single_routes_20260928.md`。
这是对两个完整请求的认证，不是 164 项指标逐项最快后端的证明。
默认三指标 F32 整批请求另有独立认证及资源门槛，见
`real_cos_f32_default_batch_20260928.json`。
