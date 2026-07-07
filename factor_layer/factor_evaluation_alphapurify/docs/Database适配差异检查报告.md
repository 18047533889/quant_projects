# Database.py 相对 Database_original.py 差异检查报告

## 检查目标
- 对比 `factor_layer/alphapurify/Database_original.py` 与 `factor_layer/alphapurify/Database.py`。
- 判断 `Database.py` 是否已实现“先把原始文件格式适配为 DataBase 可读取的输入格式”这一目标。

## 关键差异（Database.py 新增内容）
- 新增路径与类型依赖：`Path`、`Iterable`、`Sequence`，并定义 `AP_DATA_ROOT_ABS`（默认指向 `0431-测试数据`）。
- 新增适配辅助函数：
  - `_ap_normalize_years()`：统一年份过滤参数。
  - `_ap_collect_factor_files()`：按 `year=xxxx` 子目录收集因子 parquet。
  - `_ap_collect_market_files()`：按文件名后缀年份收集行情 parquet。
  - `_ap_read_factor_as_column()`：将原始因子列从 `datetime/asset/value` 映射成 `datetime/symbol/<因子名>`。
  - `_ap_abs_path()`：强制输入为绝对路径且存在。
  - `_ap_write_symbol_parquets()`：把大表拆分成 `{symbol}.parquet`。
- 新增两个核心静态方法（这是本次适配的主增量）：
  - `DataBase.build_alphapurify_input()`  
    将原始因子目录 + 行情目录适配成统一大表：`datetime, symbol, close, factor, exposures...`。
  - `DataBase.build_alphapurify_symbol_parquet_input()`  
    将上述大表进一步落盘为 `DataBase.read_dir_file()` 可直接读取的目录结构：
    - `base_data/{symbol}.parquet`（`datetime,symbol,close`）
    - `factors_data/{symbol}.parquet`（`datetime,symbol,factor+exposures`）
    并返回可直接喂给 `DataBase(...)` 的 `PathConfig`、`stocks_list`。
- 其他小改动：
  - `save()` 中把原版未定义的 `setup_logger()` 注释掉，避免潜在运行错误。

## 适配目标是否达成
- **结论：基本达成。**
- 理由：
  - `Database_original.py` 只能直接吃“按 symbol 切分好的 parquet 目录”；
  - `Database.py` 新增方法已补上“原始因子格式 -> symbol parquet 标准目录”的桥接链路；
  - 新增链路产物正是 `read_dir_file()` 期望的输入格式（目录结构与列名均对齐）。

## 适配逻辑梳理（从原始数据到最终可读）
- 读取原始目标因子与风格因子（要求列：`datetime, asset, value`）。
- 统一映射为 `datetime, symbol, <factor/exposure>` 并按 `datetime+symbol` 内连接。
- 读取行情 `daily_market_summary_*.parquet`，把 `ticker/trade_date/c` 映射为 `symbol/trade_date/close`。
- 用 `symbol+trade_date` 关联因子面板与行情，形成汇总大表 `panel_df`。
- 将 `panel_df` 拆分落盘到 `base_data` 和 `factors_data` 两个 symbol 级目录。
- 返回 `PathConfig + stocks_list`，可直接实例化 `DataBase(...).get()`。

## 本次检查发现的注意点
- `AP_DATA_ROOT_ABS` 是硬编码绝对路径；跨机器或目录变化时需显式传参覆盖。
- `_ap_abs_path()` 强制绝对路径，不接受相对路径。
- 适配流程默认可在 `discrete=[]` 下运行，因此没有 `industry` 也能跑通（与你当前情况一致）。
- 你当前 `0431-测试数据/factors` 目录没有实际 parquet 文件；若要纯用真实 raw factor 跑适配，需补齐该目录原始因子数据。

## 配套测试代码与结果
- 测试脚本（历史草稿）：`0431-测试文件/test_database_io_format_compare.py`
- 本次实测结果要点：
  - `Database.py` 侧已产出：
    - 汇总大表 `panel_df`
    - `base_data/{symbol}.parquet`
    - `factors_data/{symbol}.parquet`
  - `Database_original.py` 侧使用“编撰的 symbol parquet 输入”可正常跑通。
  - 两者 `get()` 输出列与类型一致：
    - 列：`['datetime', 'symbol', 'close', 'factor_value', 'size_exposure']`
    - 类型：`['datetime64[us]', 'object', 'float64', 'float64', 'float64']`
