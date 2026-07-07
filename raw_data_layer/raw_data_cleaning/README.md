# Massive Data Cleaning Framework

这个目录提供一套按数据源清洗 Massive parquet 的脚本和配置，用来生成更适合下游因子、策略和回测使用的 cleaned parquet。

## 目标

这套清洗层不做超级总表，只做三件事：

1. 按 source 清洗原始 parquet。
2. 补统一的标准字段，例如 ticker、align_time、primary_key。
3. 尽量保留原始业务字段，作为后续因子和策略研究的基础数据层。

## 文件

- [raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py](raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py)：核心清洗脚本。
- [raw_data_layer/raw_data_cleaning/data_source_cleaning_config.yaml](raw_data_layer/raw_data_cleaning/data_source_cleaning_config.yaml)：按源配置的清洗规则。

## 默认数据根

脚本默认使用仓库同级目录下的 [../massive_parquet](../massive_parquet) 作为数据根。

也就是默认会读取：

- `../massive_parquet/raw_massive_data`

并写出到：

- `../massive_parquet/cleaned_massive_data`

如果你的目录不一样，可以显式传：

- `--data-root`
- `--raw-root`
- `--clean-root`

## 脚本做了什么

对每个配置 source，脚本按文件执行：

1. 读取原始 parquet。
2. 如有需要，展开数组列。
3. 根据配置生成标准化 ticker。
4. 根据配置解析 align_time。
5. 用配置主键列生成 primary_key。
6. 按配置过滤空 ticker 和空 align_time。
7. 按配置去重。
8. 把标准列放前面，保留原始业务列。
9. 按 raw 目录结构镜像写出到 cleaned 目录。

## 标准输出列

每个 cleaned parquet 会补这些列：

- source
- dataset_type
- frequency
- ticker
- align_time
- primary_key
- primary_key_columns_used
- align_time_source_column
- ticker_source_column
- timezone
- notes

## 运行方式

### 清洗所有默认启用的数据源

```bash
/home/yluel/share/projects/quantsociety_backend_project/.venv/bin/python raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py --overwrite
```

### 只清洗指定 source

```bash
/home/yluel/share/projects/quantsociety_backend_project/.venv/bin/python raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py --source news/news --overwrite
```

### 同时清洗多个 source

```bash
/home/yluel/share/projects/quantsociety_backend_project/.venv/bin/python raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py --source fundamentals/balance_sheet --source news/news --overwrite
```

### 冒烟测试

```bash
/home/yluel/share/projects/quantsociety_backend_project/.venv/bin/python raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py --limit-files 1 --overwrite --verbose
```

### 自定义数据根

```bash
/home/yluel/share/projects/quantsociety_backend_project/.venv/bin/python raw_data_layer/raw_data_cleaning/massive_cleaning_framework.py --data-root /home/yluel/share/projects/massive_parquet --limit-files 1 --overwrite
```

## 默认禁用的大源

下面两个源默认不参与全量清洗：

- us_stocks_sip/quotes_v1
- us_stocks_sip/trades_v1

原因是它们体量很大，重复落地一份 cleaned parquet 成本高。需要时再显式指定 `--source` 单独处理。

## 运行摘要

每次运行完成后，脚本会生成：

- `cleaned_massive_data/_cleaning_run_summary.json`

摘要里会记录：

- 处理了哪些 source。
- 处理了多少文件。
- 每个文件的 rows_in / rows_out。
- 丢弃了多少空 ticker / 空 align_time。
- 有没有缺失主键列。

## 这层数据的作用

这套清洗层的定位是：

原始 Massive parquet
→ source-aware cleaning
→ 更适合因子、面板对齐、策略和回测使用的 cleaned parquet

它不是因子计算框架，也不是统一总表工程，而是 raw 数据和下游量化研究之间的一层标准化中间层。