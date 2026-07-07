# ============================================================
# DataBase Class Usage Example
# ============================================================
# This section demonstrates how to use the static method:
#
#     DataBase.get() --> Dataframe
#
# to get factor data from symbol-level parquet files.
#
# ============================================================


# ============================================================
# 1. You need to build a directory baesd on this Structure (VERY IMPORTANT)
# ============================================================
#
# Your data MUST follow this exact structure:
#
# main_dir_name/
#
# ├── sub_dir_name1/
# │   ├── 000001.parquet
# │   ├── 000002.parquet
# │   ├── 000003.parquet
# │   └── ...
# │
# ├── sub_dir_name2/
# │   ├── 000001.parquet
# │   ├── 000002.parquet
# │   └── ...
# │
# └── ...
#
#
# Key Rules:
#
# 1. Each file = ONE stock
#    File name MUST be the stock code:
#       000001.parquet
#       600000.parquet
#
# 2. File format MUST be .parquet
#
# 3. Each file must contain at least:
#       - datetime column
#       - code column
#
# 4. All datasets must be consistent in column naming
#
#
#   for example 000001.SZ.parquet in a subfolder:
#
#          datetime       code     open     high      low    close      volume  
#   0    2005-01-04  000001.SZ   164.85   164.85   161.60   163.10    17608.32   
#   1    2005-01-05  000001.SZ   163.10   163.85   158.85   161.60    32221.44   
#   2    2005-01-06  000001.SZ   162.60   164.85   161.35   163.10    26664.13   
#   3    2005-01-07  000001.SZ   164.60   165.10   161.60   162.85    18861.51   
#   4    2005-01-10  000001.SZ   162.85   164.85   159.35   164.85    26320.55   
#
# ============================================================


# ============================================================
# 2. Config
# ============================================================
# The config is a paramater of Database.
#
# The config dictionary tells DataBase:
#
# - where the data is stored
# - what datasets to load
# - which columns to keep
#
#
# Example structure:
#
# config = {
#
#     1. Root directory of all datasets
#     'main_dir_path': your_main_dir_path/,  
#
#     2. Dataset definitions: key = folder name, value = columns to load
#     'base_dir_name': {
#         "sub_dir_name1": [
#             'datetime',
#             'code',
#             'open',
#             'close',
#             'volume'
#         ]
#     },
#
#     3. Optional datasets:

#     # "sub_dir_name2": ['code','datetime','alpha001'],
#     # "sub_dir_name3": ['code','datetime','eps'],
#     ..........

#     4. Announce continuous data (highest frequency)
#     'continuous': ["sub_dir_name2", "sub_dir_name1"],
#
#     5. Announce discrete data (event-driven, e.g. financials)
#     'discrete': ["sub_dir_name3"]
# }
#
#
# ============================================================



# ============================================================
# 4. Initialize DataBase
# ============================================================
# from alphapurify import Database
#
# DB = DataBase(
#     config,
#     code_list,
#     start_time,
#     end_time,
#     freq='1d',
#     shift_n=1,
#     trade_date_col='datetime',
#     symbol_col='code'
# )
#
#
# Parameter Explanation:
#
# - config:
#     Data configuration dictionary
#
# - code_list:
#     Stock universe
#
# - start_time / end_time:
#     Backtest time range
#     Format MUST be: 'YYYY-MM-DD HH:MM:SS'
#     Example: '2010-01-01 00:00:00'
#
# - freq:
#     Data frequency (e.g., '1d', '1min')
#
# - shift_n:
#     Shift factor values to avoid look-ahead bias
#     (VERY IMPORTANT in backtesting)
#
# - trade_date_col:
#     Name of datetime column
#
# - symbol_col:
#     Name of stock code column
#
# ============================================================


# ============================================================
# 6. Load Data
# ============================================================
#
# df = DB.get()
#
#
# Output:
#
# - A unified pandas DataFrame
# - Indexed by datetime and stock code
# - All datasets merged together
#
#
# Typical structure:
#
# datetime | code | open | close | volume | factor1 | factor2 | ...
#
#
# ============================================================







# ============================================================
# Factor Saving Utility Example
# ============================================================
#
# This section demonstrates how to use the static method:
#
#     DataBase.save(...)
#
# to store factor data into symbol-level parquet files.
#
# NOTE:
# This is a fully commented example (no executable code).
#
# ============================================================



# ============================================================
# 1. Make sure you have a folder (VERY IMPORTANT)
# ============================================================
#
# The function will automatically generate the following structure:
#
# dir/
#
# ├── 000001.parquet
# ├── 000002.parquet
# ├── 000003.parquet
# └── ...
#
#
# Key Rules:
#
# 1. Each file = ONE symbol
#
# 2. File name MUST be:
#       {symbol}.parquet
#
# 3. Each file contains:
#       - datetime column
#       - symbol column
#       - multiple factor columns
#
# 4. Data is stored in columnar format (parquet)
#    → optimized for performance and compression
#
# ============================================================


# ============================================================
# 3. Input Data Requirements
# ============================================================
#
# base_df must contain:
#
# - symbol column (e.g., 'code')
# - datetime column (e.g., 'datetime')
# - factor columns (defined in factors_list)
#
#
# Example structure:
#
# datetime | code | factor_1 | factor_2 | factor_3 | ...
#
#
# Notes:
#
# - datetime must be convertible to nanosecond precision
# - data should already be cleaned and aligned
#
# ============================================================


# ============================================================
# 4. Parameters Explanation
# ============================================================
#
# DataBase.save(
# -->    base_df,
#     factors_list,
#     factors_dir,
#     trade_date_col,
#     symbol_col,
#     append=False,
#     max_workers=-1
# )
#
#
# - base_df:
#     Input DataFrame (pandas or polars supported)
#
# - factors_list:
#     List of factor column names to save
#
# - factors_dir:
#     path of output directory for parquet files
#
# - trade_date_col:
#     Name of datetime column
#
# - symbol_col:
#     Name of symbol column
#
# - append:
#     Writing mode (CRITICAL)
#
# - max_workers:
#     Number of parallel processes
#
# ============================================================


# ============================================================
# 5. Append Mode (CRITICAL DESIGN)
# ============================================================
#
# The `append` parameter controls how existing files are handled.
#
#
# append = True:
#
#     - Only NEW dates will be added
#     - Existing data will NOT be modified
#
#     Use case:
#     → Daily factor updates (production environment)
#
#
# append = False:
#
#     - Overlapping dates will be overwritten
#     - Only specified factor columns are updated
#
#     Use case:
#     → Factor recalculation / backfill
#
#
# IMPORTANT:
#
# - In append mode:
#     New data MUST have datetime strictly greater
#     than existing data in the file
#
# - Otherwise, data inconsistency may occur
#
# ============================================================

