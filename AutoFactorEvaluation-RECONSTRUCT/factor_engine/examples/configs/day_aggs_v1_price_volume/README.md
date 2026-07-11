# day_aggs_v1 价量因子模板

数据根目录示例（各组员 `~` 不同，monorepo 路径相同）：

- `~/quant_projects/data/us_stock/cleaned_massive_data/us_stocks_sip/day_aggs_v1/2026/03`

批量物化：

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python examples/materialize_config_directory.py \
  examples/configs/day_aggs_v1_price_volume \
  --lake-root ~/quant_projects/data/factors/lake
```
