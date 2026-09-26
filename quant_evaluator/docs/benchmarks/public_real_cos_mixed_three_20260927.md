# 真实 COS 三指标公开入口整批 A/B（2026-09-27）

本记录对应 `rank_ic`、`quantile_spread`、`factor_turnover_rate` 的同一公开 `evaluate()` 请求。只读输入使用已有 `load_real_batch(days=0, assets=5500)`，绑定的 COS manifest SHA256 为 `00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66`。该加载器逐对象校验字节数和 SHA256，并清理自己的短暂下载文件；没有持久化原始因子面板。

- 形状：2586 日 × 5461 股 × 2 因子；float64；默认指标参数。
- 硬件：server-c NVIDIA L20；运行时另有 GPU 作业，计时仅供此设备与当时负载下的路由判断。
- 交错整批运行：CPU 8.7884、8.4854 秒；CUDA strict 4.9253、4.4917 秒。随后独立完整对拍为 CPU 8.868 秒、CUDA 5.015 秒。
- 按指标拆分的计时下界（`rank_ic` CUDA、`quantile_spread` CUDA、`factor_turnover_rate` CPU；不含产物合并）：5.631、5.7158 秒。整批 CUDA 因而是本实测请求的较快路径。
- 整批 CUDA 峰值显存：11,234,754,560 字节。新 auto 门槛沿用 F2 rank IC 大面板公式，此形状要求 21,691,616,256 字节的有效空闲显存。
- 逐项对拍：三指标的产物类型、metric_id、domain、factor_axis、producer_version、完整 provenance、有限值 mask、observation_counts、每因子的 MetricValue 非数值字段完全相同。产物及 MetricValue 数值 `rtol=1e-8, atol=1e-10` 全部通过。CPU/CUDA 的 `config_hash` 相同：`3a5fd2193c79e3d9cb3ebb603805c972f8b99c6f24a8a1f01cf4e6572aa80cff`。
- 改动后真实 auto 运行 5.0932 秒，回执 `backend_used=cuda`、`auto_backend_reason=certified_batch_real_cos_mixed_three`，三个 `metric_backends` 均为 `cuda`，回执 `config_hash` 与 bundle 相同。49 项相关测试通过。

复现（在 server-c 正式主树运行；约需 30 秒加载及数次计算；不指定输出文件）：

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python - <<'PY'
import time
import numpy as np
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch
from quant_evaluator.runtime.evaluator import evaluate

batch, labels, _ = load_real_batch(days=0, assets=5500)
assert (batch.num_times, batch.num_assets, batch.num_factors) == (2586, 5461, 2)
metrics = ("rank_ic", "quantile_spread", "factor_turnover_rate")
runs = {}
for backend in ("cpu", "cuda_strict", "cpu", "cuda_strict", "auto"):
    start = time.perf_counter()
    bundle = evaluate(batch, labels, metrics=metrics, backend=backend)
    print(backend, time.perf_counter() - start,
          bundle.metadata["backend_used"], bundle.metadata.get("peak_vram"),
          bundle.metadata["auto_backend_reason"])
    runs[backend] = bundle
cpu, gpu, auto = runs["cpu"], runs["cuda_strict"], runs["auto"]
assert cpu.config_hash == gpu.config_hash == auto.config_hash
assert auto.metadata["execution_receipt"]["config_hash"] == auto.config_hash
for metric in metrics:
    a, b = cpu.artifacts[metric], gpu.artifacts[metric]
    assert type(a) is type(b)
    assert (a.metric_id, a.domain, a.factor_axis, a.producer_version,
            a.provenance) == (b.metric_id, b.domain, b.factor_axis,
                              b.producer_version, b.provenance)
    np.testing.assert_array_equal(np.isfinite(a.values), np.isfinite(b.values))
    np.testing.assert_allclose(a.values, b.values, rtol=1e-8, atol=1e-10,
                               equal_nan=True)
    for factor_id in batch.factor_ids:
        x, y = cpu.get_metric(metric, factor_id), gpu.get_metric(metric, factor_id)
        assert (x.metric_id, x.valid, x.observation_count, x.metric_version,
                x.sample_unit, x.warnings) == (
                y.metric_id, y.valid, y.observation_count, y.metric_version,
                y.sample_unit, y.warnings)
        assert np.isclose(x.value, y.value, rtol=1e-8, atol=1e-10,
                          equal_nan=True)
print("complete parity PASS")
PY
```

本证据只覆盖精确三指标组合与上述 F2 面板；其他指标组合或形状需要另行 A/B。
