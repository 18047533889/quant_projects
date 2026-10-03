# 六项分层指标的新进程 provider 验证工具

## 目的与当前状态

本工具为研究验证入口，不是生产部署入口。它验证应用配置 provider 后，
相同真实请求的默认 `backend="auto"` 是否使用当前源码上合格的最快后端，
而非只命中上一次测试进程的缓存。

实现路径为 `scripts/verify_real_cos_linear_shape_provider.py`；
输出校验独立位于 `scripts/linear_shape_provider_verification.py`。
实现与独立联合测试已通过：55 passed，保留 1 个已有 fork 警告（1.99 秒）。
新增缓存守卫的红测试确认旧实现先读取 manifest；修复后拒绝路径不读
manifest/COS、不初始化运行时、不写输出，且原缓存保持不变。
首次资格状态的红测试确认旧 helper 接受缺失、None、cache_hit 或未知状态；
修复后仅接受 supplied，第二次仅接受 cache_hit。
默认真实预检和模块入口帮助命令也已执行通过，但尚未运行当前版本的
真实 COS 验证，工具本身不产生性能资格。
历史 r2 报告不能因为严格 reader 接受就自动成为当前源码资格。

## 固定研究范围

- 2586 个交易日、5461 个资产、48 个因子；请求 tile cap 为 16。
- 指标依次为 quantile_curvature、quantile_tail_asymmetry、
  quantile_adjacent_spread、quantile_extreme_cliff、top_quantile_cliff、
  bottom_quantile_cliff。
- 默认 Q=5、min_assets=10、min_periods=20，不验证非默认参数。
- 当前只接收 `real_cos_profile_abba_f48_linear_shape.v1` 类型的完整报告。
  Pearson 四指标、旧 F61 全指标及其他形状不属于此工具范围。

## 运行方式

正式目录为 server-c 的 `/home/sunhaiwei/quant_projects`，使用其中的
`.venv/bin/python`。保留已批准的研究 COS 环境和缓存配置，不猜测桶、
凭据或替代 CLI。每个进程均使用一致的 BLAS/OMP/MKL 线程设置。

本次已实际预检通过的研究配置如下；这是已有授权入口，不是新增凭据。
默认 shell 若未配置研究 CLI，工具会拒绝，不会自行改用另一个 COS 命令。

```sh
cd /home/sunhaiwei/quant_projects
export DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos
export DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

默认预检（不读取 COS、不初始化 CUDA、不安装 provider、不写报告）：

```sh
.venv/bin/python -m quant_evaluator.scripts.verify_real_cos_linear_shape_provider
```

实际验证必须显式授权运行，且应从一个新 Python 进程启动：

```sh
.venv/bin/python -m quant_evaluator.scripts.verify_real_cos_linear_shape_provider \
  --run \
  --profile-report quant_evaluator/docs/benchmarks/NEW_CURRENT_PROFILE.json \
  --axis-index quant_evaluator/docs/benchmarks/real_cos_f48_axis_index_20261004.json \
  --output quant_evaluator/docs/benchmarks/NEW_FRESH_PROVIDER_RESULT.json
```

大写文件名是占位符，不是已存在的报告。profile 必须是在工具及相关
QE/DA/FO/FP Python 修改完成、源码冻结后重新生成的报告。输出必须为新路径。

## 验证与资源约束

先通过研究 CLI、资源和严格报告检查，再读取已授权 manifest/轴/标签。
只用既有 COS source 和有界预取，不构建全量 F48 因子 cube。
独立参考计算每次仅持有一个因子 tile，源在异常和成功路径均关闭。
CPU winner 不做 CUDA 指标预热或执行显存 gate；CUDA winner 保留两者。
双臂 ABBA 报告的运行时摘要包含 CUDA context，因此即使 CPU 获胜，
验证也需重建相同上下文；这不是无 GPU 环境的性能资格。

两次请求均不传 `source_qualification`，也不清空缓存来伪造新进程结果：

1. 首次必须为 report_candidate / candidate_validated，资格已应用，
   缓存状态必须为 supplied；后端、实际 tile 宽度和前后 live context 与报告一致。
2. 第二次必须为 process_cache / cache_hit，仍满足所有输出和上下文检查。
3. 两次均要求零 OOM 重试、独立参考一致，且六指标的值、有限掩码、
   observation_counts 摘要与获胜后端完整输出绑定一致。

已有全局 provider 时拒绝运行，不替换其他调用者的配置。
严格报告检查后、manifest/COS 读取和 CUDA gate 之前，还通过
`source_qualification_cache.has_validated_records()` 检查进程中是否已有
未过期资格记录；只读、加锁，不删除或重排调用者的缓存。存在任何有效记录
就保守拒绝，即使它属于另一个请求；过期记录不会阻挡验证。因此请用新的
Python 进程执行本工具，不在已进行评估的交互进程里调用 `main()`。
这一守卫仅用于研究验证入口，不改变正常评估 API 的缓存选择。
本次安装的 provider 在退出时清理。异常只输出安全错误类型/原因码，
不输出原始路径、凭据或外部命令异常文本。报告有界、排他创建，不覆盖历史证据。

## 如何解读结果

完成新进程验证不等于所有指标或任意批次都最快；这里只证明固定请求在
报告对应源码、运行时和资源条件下，测得的合格后端已接入默认 auto。
本工具不自动测速、搜索报告、调参或发布因子。生产应用仍需在启动时
显式配置 provider；未接入应用的验证工具不会自行改变生产默认行为。
