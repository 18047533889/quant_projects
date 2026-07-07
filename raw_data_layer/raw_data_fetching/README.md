# raw_data_fetching

这个目录负责原始数据抓取和 parquet 文件校验，当前包含面向 Massive / Polygon 风格数据源的下载脚本和质量检查脚本。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `download_all_history.py` | 统一下载多个数据集并落地为 parquet |
| `download_history.py` | 面向对象存储前缀的历史文件下载与转换 |
| `validate_parquet.py` | 递归校验 parquet 可读性、日期一致性、NaN 和负值 |
| `download_results.csv` | 历史下载结果记录 |
| `rest_api_doc/` | 接口文档与参考资料 |
| `test.ipynb` / `rest_api_test.ipynb` | 交互式试验 notebook |

## 常见入口

### 1. 批量下载历史数据

```bash
python raw_data_layer/raw_data_fetching/download_all_history.py --help
```

这个脚本内部维护了多个数据集定义，适合做统一的批量拉取。

### 2. 按前缀下载对象存储历史文件

```bash
python raw_data_layer/raw_data_fetching/download_history.py --help
```

从当前参数定义看，常用参数包括：

- `--year`
- `--month`
- `--day`
- `--prefix`
- `--local-root`
- `--workers`
- `--max-files`

### 3. 校验 parquet 目录

```bash
python raw_data_layer/raw_data_fetching/validate_parquet.py /path/to/parquet/root --workers 8
```

这个脚本会递归扫描目录，并输出：

- 文件损坏或为空
- 文件名日期和时间列不一致
- 关键字段 NaN
- 关键价格或数量字段出现负值

## 使用建议

- 先下载，再校验，再交给 `raw_data_cleaning/`
- 下载型脚本通常依赖外部访问凭据或 API key，运行前先检查环境配置
- notebook 主要用于探索和调试，不建议作为正式批处理入口

## 下游衔接

完成下载和校验后，通常会继续进入：

- [../raw_data_cleaning/README.md](../raw_data_cleaning/README.md)