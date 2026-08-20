# day_aggs_v1 + 基本面 composite 因子模板

- 价格锚点：`~/quant_projects/data/us_stock/cleaned_massive_data/us_stocks_sip/day_aggs_v1`
- 基本面辅助：`~/quant_projects/data/us_stock/cleaned_massive_data/fundamentals/*`

批量物化：

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python examples/materialize_config_directory.py \
  examples/configs/day_aggs_v1_fundamental \
  --lake-root ~/quant_projects/data/factors/lake \
  --log-file ~/quant_projects/data/factors/lake/logs/day_aggs_v1_fundamental_materialize.log
```
