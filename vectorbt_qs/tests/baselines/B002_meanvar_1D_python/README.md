# B002 accurate Python 基线

> 历史基线：该结果属于 `standard_accurate_v1` 之前的复权价格/复权股数口径，
> 不包含 V2 的原始价格、真实股数、现金分红、送转股和成交容量处理。它只用于审计
> 旧版本，不得作为当前 `standard_accurate_v2` 的正确结果。

该目录冻结了改造前 Python 状态机运行
`examples/configs/B002_meanvar_1D.yaml` 的完整结果：

- `nav.parquet`：逐日 NAV、现金和收益率；
- `orders.parquet`：vectorbt 实际订单记录；
- `final_assets.parquet`：期末逐标的持仓；
- `execution_log.parquet`：规划器的成交、阻塞和部分成交日志；
- `metadata.json`：配置、摘要和各文件 SHA-256；
- `verification_numba.json`：Numba 引擎对比结果。

在对应的历史代码版本上可使用以下命令验证：

```bash
python scripts/verify_accurate_parity.py \
  examples/configs/B002_meanvar_1D.yaml \
  tests/baselines/B002_meanvar_1D_python
```

验证使用精确数值比较，不以绩效指标四舍五入后的结果代替逐日和逐订单检查。
