import pandas as pd
import polars as pl
import os
import duckdb
import datetime
import re
from pathlib import Path
from typing import Iterable, Sequence
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from joblib import Parallel, delayed

AP_DATA_ROOT_ABS = Path(r"D:/基建组hyz/alphapurify适配器/quantsociety_backend/0431-测试数据")

def process_code(args):
    """
    Process and save factor data for a single symbol.

    Parameters
    ----------
    args : tuple
        (symbol, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append)

        symbol : str
            Stock symbol.
        small_df : pl.DataFrame
            Factor data for the given symbol.
        factors_dir : str
            Directory where parquet files are stored.
        factors_list : list
            List of factor column names to be updated.
        trade_date_col : str
            Name of the datetime column.
        symbol_col : str
            Name of the symbol column.
        append : bool
        Determines how new factor data is written into the existing parquet file.

        If True (append mode):
            - Only rows whose `trade_date_col` do NOT already exist
            in the existing file will be inserted.
            - Existing rows will NOT be modified.
            - New rows are concatenated to the existing dataset.
            - If no new dates are found, nothing will be written.

        If False (overwrite mode):
            - Only rows whose `trade_date_col` already exist
            in the existing file will be considered.
            - For overlapping dates, factor values in `factors_list`
            will overwrite the existing values (using coalesce logic).
            - Non-overlapping rows will NOT be inserted.
            - If no overlapping dates are found, nothing will be written.
    """
    symbol, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append = args
    parquet_file = os.path.join(factors_dir, f"{symbol}.parquet")
    small_df = small_df.select([trade_date_col,symbol_col] + factors_list)
            
    if not os.path.exists(parquet_file):
        small_df.write_parquet(parquet_file)
        return symbol
                
    existing_df = pl.read_parquet(parquet_file, use_pyarrow=False)

    if existing_df[trade_date_col].dtype != pl.Datetime:
        existing_df = existing_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime('ns')))
    small_df = small_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime("ns")))
            
    for fac in factors_list:
        if fac not in existing_df.columns:
            existing_df = existing_df.with_columns(pl.lit(None).cast(pl.Float64).alias(fac))
            
    for col in existing_df.columns:
        if col not in small_df.columns:
            small_df = small_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    small_df = small_df.select(existing_df.columns)

    if append:
        small_df = small_df.join(
            existing_df.select(trade_date_col),
            on=trade_date_col,
            how="anti"
        )
                
        if small_df.is_empty():
            return
                        
        merged_df = pl.concat([existing_df, small_df])
        merged_df = merged_df.sort(trade_date_col)
        merged_df.write_parquet(parquet_file)
            
    else:
        overlap_df = small_df.join(
            existing_df.select(trade_date_col),
            on=trade_date_col,
            how="inner"
        )

        if overlap_df.is_empty():
            return

        merged = existing_df.join(
            overlap_df,
            on=trade_date_col,
            how="left",
            suffix="_new"
        )

        merged = merged.with_columns([
            pl.coalesce(pl.col(f + "_new"), pl.col(f)).alias(f)
            for f in small_df.columns if not f == trade_date_col
        ])

        merged = merged.drop([
            f + "_new"
            for f in small_df.columns if not f == trade_date_col
        ])

        merged.write_parquet(parquet_file)


def _ap_normalize_years(years: Sequence[int] | None) -> set[int] | None:
    if years is None:
        return None
    return {int(y) for y in years}


def _ap_collect_factor_files(folder: Path, years: set[int] | None = None) -> list[Path]:
    files = sorted(folder.rglob("*.parquet"))
    if years is None:
        return files

    selected: list[Path] = []
    for p in files:
        for part in p.parts:
            if part.startswith("year="):
                try:
                    if int(part.split("=", 1)[1]) in years:
                        selected.append(p)
                        break
                except ValueError:
                    continue
    return selected


def _ap_collect_market_files(folder: Path, years: set[int] | None = None) -> list[Path]:
    files = sorted(folder.glob("*.parquet"))
    if years is None:
        return files

    selected: list[Path] = []
    for p in files:
        maybe_year = p.stem.rsplit("_", 1)[-1]
        if maybe_year.isdigit() and int(maybe_year) in years:
            selected.append(p)
    return selected


def _ap_read_factor_as_column(factor_dir: Path, out_col: str, years: set[int] | None = None) -> pl.DataFrame:
    files = _ap_collect_factor_files(factor_dir, years=years)
    if not files:
        raise ValueError(f"未找到因子文件: {factor_dir}")

    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    required = {"datetime", "asset", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"因子文件缺少列 {missing}: {factor_dir}")

    return (
        df.select(
            pl.col("datetime").cast(pl.Datetime).alias("datetime"),
            pl.col("asset").cast(pl.Utf8).alias("symbol"),
            pl.col("value").cast(pl.Float64).alias(out_col),
        )
        .drop_nulls(["datetime", "symbol", out_col])
        .sort(["symbol", "datetime"])
    )


def _ap_abs_path(p: str | Path, name: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} 必须是绝对路径: {path}")
    if not path.exists():
        raise ValueError(f"{name} 路径不存在: {path}")
    return path.resolve()


def _ap_write_symbol_parquets(
    df: pl.DataFrame,
    out_dir: Path,
    symbol_col: str = "symbol",
    trade_date_col: str = "datetime",
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        return []

    symbols = (
        df.select(pl.col(symbol_col).cast(pl.Utf8))
        .unique()
        .sort(symbol_col)
        .get_column(symbol_col)
        .to_list()
    )
    for sym in symbols:
        one = (
            df.filter(pl.col(symbol_col) == sym)
            .sort(trade_date_col)
            .with_columns(pl.col(symbol_col).cast(pl.Utf8))
        )
        one.write_parquet(out_dir / f"{sym}.parquet")
    return symbols
             
class DataBase():
    """
    DataBase

    A high-performance data loading and preprocessing utility designed for
    factor research pipelines. This class provides a unified interface for
    reading, merging, aligning, and preparing large-scale financial datasets
    stored in symbol-level parquet files.

    The class supports continuous features (e.g., price-based indicators)
    and discrete features (e.g., fundamental events, classifications), and
    automatically aligns them with the main price dataset.

    Main capabilities include:
    - Efficient parquet querying via DuckDB
    - Parallel reading of symbol-level datasets
    - Automatic time alignment of continuous and discrete features
    - Forward filling of discrete attributes
    - Time shifting of continuous indicators to prevent look-ahead bias
    - Parallel factor storage to symbol-level parquet files

    Parameters
    ----------
    PathConfig : dict
        Configuration dictionary describing the directory structure and
        feature groups. It should include:

        - main_dir_path : str
            Root directory containing all data folders.

        - base_dir_name : dict
            Dictionary specifying the main dataset directory and its features.

        - continuous : list[str]
            Directory names containing continuous features.

        - discrete : list[str]
            Directory names containing discrete features.

        Additional keys map directory names to their feature column lists.

    stocks_list : list[str]
        List of asset symbols to load.

    begin_date : str
        Start datetime of the data range (format: "YYYY-MM-DD HH:MM:SS").

    end_date : str
        End datetime of the data range (format: "YYYY-MM-DD HH:MM:SS").

    trade_date_col : str
        Column name representing the timestamp.

    symbol_col : str
        Column name representing the asset identifier.

    freq : str, default "1d"
        Data frequency used to determine feature shift durations.

    shift_n : int, default 1
        Number of periods used to shift continuous features forward in time.
        This helps prevent look-ahead bias.

    dropNaN : bool, default False
        Whether to drop rows containing missing values after merging datasets.

    max_workers : int, default -1
        Number of parallel workers used for data loading and saving.
        If -1, the system will use (CPU count - 1).

    Attributes
    ----------
    continuous_df : pl.DataFrame
        DataFrame containing merged continuous features.

    discrete_df : pl.DataFrame
        DataFrame containing merged discrete features.

    discrete_dfs : list
        Intermediate list of discrete datasets.

    continuous_dfs : list
        Intermediate list of continuous datasets.

    feature_dict : dict
        Mapping between directory names and feature column lists.

    Notes
    -----
    Continuous vs Discrete Features

    Continuous features typically include indicators derived from prices
    (e.g., returns, momentum, volatility) and are shifted forward to avoid
    look-ahead bias.

    Discrete features represent event-based or categorical information
    (e.g., industry classification, fundamental reports) and are forward
    filled using an as-of join.

    Data Alignment

    - Continuous features are merged directly on (datetime, symbol)
    - Discrete features are aligned using backward `join_asof`
    - The final dataset is sorted by (datetime, symbol)

    Example
    -------
    >>> db = DataBase(
    ...     PathConfig=config,
    ...     stocks_list=stocks,
    ...     begin_date="2018-01-01 00:00:00",
    ...     end_date="2024-01-01 00:00:00",
    ...     trade_date_col="datetime",
    ...     symbol_col="symbol"
    ... )
    >>> df = db.get()

    The returned DataFrame can be directly used in factor research
    pipelines such as FactorAnalyzer.
    """
    def __init__(self, 
                 PathConfig: dict,
                 stocks_list:list,
                 begin_date:str,
                 end_date:str,
                 trade_date_col:str,
                 symbol_col:str,
                 freq:str='1d',
                 shift_n: int = 1,
                 dropNaN: bool = False,
                 max_workers:int = -1):
        
        self.shift_n = shift_n
        self.dropNaN = dropNaN
        self.stocks_list = stocks_list
        self.begin_date:str = begin_date
        self.end_date:str = datetime.datetime.strptime(end_date,'%Y-%m-%d %H:%M:%S')
        self.freq = freq
        self.shift_duration:str = self.multiply_duration(shift_n*20,freq)
        self.base_dir_name, self.base_dir_features = next(iter(PathConfig['base_dir_name'].items()))
        
        self.main_dir_path = PathConfig.get('main_dir_path')
        self.trade_date_col:str = trade_date_col
        self.symbol_col:str = symbol_col
        self.max_workers = max_workers
        
        self.feature_dict = {
            key: val for key, val in PathConfig.items()
            if key not in ['main_dir_path', 'continuous', 'discrete','base_dir_name']
        }
        
        self.continuous_dict = {'continuous': PathConfig.get('continuous', [])}
        self.discrete_dict = {'discrete': PathConfig.get('discrete', [])}
        
        self.discrete_dfs = []
        self.continuous_dfs = []
        
        self.discrete_df:pd.DataFrame = None
        self.continuous_df:pd.DataFrame = None
    
    def read_dir_file(self,dir_name:str):
        n_jobs = max(os.cpu_count() - 1, 1) if self.max_workers == -1 else self.max_workers
        dir_path = os.path.join(self.main_dir_path,dir_name)
           
        if not dir_path.endswith(os.sep):
            dir_path += os.sep
        print(f"Data directory : {os.path.abspath(dir_path)}")
        
        stock_list = self.stocks_list
        
        con = duckdb.connect()
        con.execute(f"SET threads TO {n_jobs};")
        
        try:
            
            parquet_files = list(map(lambda name: os.path.join(dir_path, f"{name}.parquet"), stock_list))

            file_list = ", ".join(map(lambda file: f"'{file}'", parquet_files))
            
            if dir_name == self.base_dir_name:
                features = self.base_dir_features
            elif dir_name in self.discrete_dict.get('discrete', []):
                features = self.feature_dict.get(dir_name, [])
            elif dir_name in self.continuous_dict.get('continuous', []):
                features = self.feature_dict.get(dir_name, [])
            else:
                features = []
            if not features:
                print(f"No features configured for table '{dir_name}'.")
                return []
            selected_features = ", ".join(map(lambda feat: feat, features))
            
            if dir_name in self.discrete_dict.get('discrete', []):
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                begin_date = self.shift_datetime(begin_date,'150d')
                
                #print(begin_date)
                
            elif dir_name in self.continuous_dict.get('continuous', []):
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                
                begin_date = self.shift_datetime(begin_date,self.shift_duration)
                
                #print(begin_date)
            else:
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                
                #print(begin_date)
                
            query = f"""
            SELECT {selected_features}
            FROM read_parquet([{file_list}]) 
            WHERE {self.trade_date_col} >= '{begin_date}' 
            AND datetime <= '{self.end_date}'
            """
        
            df_all = con.execute(query).pl()
                   
            if not df_all.is_empty():
                
                df_all = df_all.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime))                           
                
                print(f"Successfully fetched {len(df_all)} rows of data.")
                return df_all
            else:
                for file in parquet_files[:5]:
                    exists = "exists" if os.path.exists(file) else "not found"
                    print(f"{os.path.basename(file)}: {exists}")
                    
                    print('failed to fetch data')
                
                return pl.DataFrame(schema={col: pl.Null for col in features})
            
        except Exception as e:
            print(f"Query error: {str(e)}")
            return pl.DataFrame(schema={col: pl.Null for col in features})
        
        finally:
            con.close()
    
    def read_and_merge_dfs(self):
        base_df = self.read_dir_file(self.base_dir_name).sort([self.trade_date_col, self.symbol_col])
        self.continuous_dfs = []
        for dir_name in self.continuous_dict.get("continuous", []):
            df = self.read_dir_file(dir_name).sort([self.trade_date_col, self.symbol_col])
            if not df.is_empty():
                
                df = df.with_columns(pl.col(self.trade_date_col).shift(-self.shift_n).alias(self.trade_date_col))
            self.continuous_dfs.append(df)

       
        self.discrete_dfs = []
        for dir_name in self.discrete_dict.get("discrete", []):
            df = self.read_dir_file(dir_name)
            if not df.is_empty():
                subset_cols = self.feature_dict.get(dir_name, [])
                
                if subset_cols:
                    df = df.unique(subset=subset_cols)
                df = df.sort([self.trade_date_col, self.symbol_col])
            self.discrete_dfs.append(df)

        
        self.continuous_df = base_df.clone()
        for df in self.continuous_dfs:
            
            self.continuous_df = (
                self.continuous_df.join(df, on=[self.trade_date_col, self.symbol_col], how="left")
                .sort([self.symbol_col,self.trade_date_col])
            )

        if self.discrete_dfs:
            self.discrete_df = self.discrete_dfs[0].sort([self.trade_date_col, self.symbol_col])
            for df in self.discrete_dfs[1:]:
                
                common_cols = [c for c in self.discrete_df.columns if c in df.columns]
                if not common_cols:
                    raise ValueError(
                        "No common columns found when merging discrete datasets. "
                        "pl.merge will fail in this case — please check the input."
                    )
                self.discrete_df = self.discrete_df.join(df, on=common_cols, how="outer")
            self.discrete_df = self.discrete_df.sort([self.symbol_col,self.trade_date_col])
        else:
            self.discrete_df = pl.DataFrame() 

        #print(self.continuous_df)
        #print(self.discrete_df)

    def fill_to_full_df(self):
        cont_df:pl.DataFrame = self.continuous_df
        if self.discrete_dfs:
            disc_df = self.discrete_df

            filled_df:pl.DataFrame = cont_df.join_asof(
                disc_df,
                on=self.trade_date_col,
                by=self.symbol_col,
                strategy="backward"
            )

            if self.dropNaN:
                filled_df = filled_df.drop_nulls()
                
            filled_df = filled_df.sort([self.trade_date_col,self.symbol_col]).to_pandas()
            filled_df[self.trade_date_col] = pd.to_datetime(filled_df[self.trade_date_col])
            return filled_df.sort_values([self.trade_date_col, self.symbol_col])

        else:
            cont_df = cont_df.sort([self.trade_date_col, self.symbol_col]).to_pandas()
            cont_df[self.trade_date_col] = pd.to_datetime(cont_df[self.trade_date_col])
            return cont_df
    
    def get(self):
        self.read_and_merge_dfs()
        full_df = self.fill_to_full_df()
        return full_df

    @staticmethod
    def build_alphapurify_input(
        data_root: str | Path,
        target_factor_dir: str | Path,
        exposure_factor_dirs: Iterable[str | Path],
        factor_name: str = "factor_value",
        years: Sequence[int] | None = None,
        min_symbols_per_day: int = 15,
    ) -> pl.DataFrame:
        """
        从 0431 测试数据构建 alphapurify 所需输入：
        datetime, symbol, close, factor_name, exposure_cols...
        """
        years_set = _ap_normalize_years(years)

        if data_root is None:
            data_root = AP_DATA_ROOT_ABS
        data_root = _ap_abs_path(data_root, "data_root")
        target_factor_dir = _ap_abs_path(target_factor_dir, "target_factor_dir")
        exposure_factor_dirs = [_ap_abs_path(p, "exposure_factor_dir") for p in exposure_factor_dirs]

        target_df = _ap_read_factor_as_column(target_factor_dir, factor_name, years=years_set)

        exposure_col_names: list[str] = []
        panel = target_df
        for exp_dir in exposure_factor_dirs:
            exp_col = exp_dir.name
            exposure_col_names.append(exp_col)
            exp_df = _ap_read_factor_as_column(exp_dir, exp_col, years=years_set)
            panel = panel.join(exp_df, on=["datetime", "symbol"], how="inner")

        panel = panel.with_columns(pl.col("datetime").dt.date().alias("trade_date"))

        market_root = data_root / "daily_market_summary"
        market_files = _ap_collect_market_files(market_root, years=years_set)
        if not market_files:
            raise ValueError(f"未找到行情文件: {market_root}")

        market_df = pl.concat([pl.read_parquet(f) for f in market_files], how="vertical_relaxed")
        required_market_cols = {"ticker", "trade_date", "c"}
        missing_market_cols = required_market_cols - set(market_df.columns)
        if missing_market_cols:
            raise ValueError(f"行情文件缺少列 {missing_market_cols}")

        market_df = (
            market_df.select(
                pl.col("ticker").cast(pl.Utf8).alias("symbol"),
                pl.col("trade_date").str.strptime(pl.Date, strict=False).alias("trade_date"),
                pl.col("c").cast(pl.Float64).alias("close"),
            )
            .drop_nulls(["symbol", "trade_date", "close"])
            .unique(["symbol", "trade_date"])
        )

        out = (
            panel.join(market_df, on=["symbol", "trade_date"], how="inner")
            .drop("trade_date")
            .drop_nulls(["datetime", "symbol", "close", factor_name] + exposure_col_names)
            .sort(["symbol", "datetime"])
        )

        out = out.join(
            out.group_by("datetime").agg(pl.len().alias("_n")).filter(pl.col("_n") >= min_symbols_per_day),
            on="datetime",
            how="inner",
        ).drop("_n")

        if out.is_empty():
            raise ValueError("适配结果为空，请检查年份范围、标的重叠或列映射。")

        return out

    @staticmethod
    def build_alphapurify_symbol_parquet_input(
        data_root: str | Path,
        target_factor_dir: str | Path,
        exposure_factor_dirs: Iterable[str | Path],
        output_root: str | Path,
        factor_name: str = "factor_value",
        years: Sequence[int] | None = None,
        min_symbols_per_day: int = 15,
        base_dir_name: str = "base_data",
        continuous_dir_name: str = "factors_data",
        trade_date_col: str = "datetime",
        symbol_col: str = "symbol",
    ) -> dict:
        """
        将 0431 原始数据适配并落盘为 DataBase.read_dir_file() 期望的目录格式：
        output_root/
          base_dir_name/{symbol}.parquet            # datetime, symbol, close
          continuous_dir_name/{symbol}.parquet      # datetime, symbol, factor + exposures
        """
        output_root = Path(output_root).expanduser()
        if not output_root.is_absolute():
            raise ValueError(f"output_root 必须是绝对路径: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        if data_root is None:
            data_root = AP_DATA_ROOT_ABS

        panel = DataBase.build_alphapurify_input(
            data_root=data_root,
            target_factor_dir=target_factor_dir,
            exposure_factor_dirs=exposure_factor_dirs,
            factor_name=factor_name,
            years=years,
            min_symbols_per_day=min_symbols_per_day,
        )

        exposure_cols = [Path(p).name for p in exposure_factor_dirs]
        base_df = panel.select([trade_date_col, symbol_col, "close"])
        continuous_df = panel.select([trade_date_col, symbol_col, factor_name] + exposure_cols)

        base_path = output_root / base_dir_name
        continuous_path = output_root / continuous_dir_name
        stocks_list = _ap_write_symbol_parquets(
            base_df,
            base_path,
            symbol_col=symbol_col,
            trade_date_col=trade_date_col,
        )
        _ap_write_symbol_parquets(
            continuous_df,
            continuous_path,
            symbol_col=symbol_col,
            trade_date_col=trade_date_col,
        )

        path_config = {
            "main_dir_path": str(output_root),
            "base_dir_name": {base_dir_name: [trade_date_col, symbol_col, "close"]},
            "continuous": [continuous_dir_name],
            "discrete": [],
            continuous_dir_name: [trade_date_col, symbol_col, factor_name] + exposure_cols,
        }
        return {
            "PathConfig": path_config,
            "stocks_list": stocks_list,
            "trade_date_col": trade_date_col,
            "symbol_col": symbol_col,
            "factor_name": factor_name,
            "exposure_cols": exposure_cols,
            "panel_df": panel,
        }
    
    @staticmethod
    def save(base_df: pd.DataFrame,
             factors_list,
             factors_dir,
             trade_date_col:str,
             symbol_col:str,
             append=False,
             max_workers:int = -1):
        """
        Save factor data into symbol-level parquet files using parallel processing.

        This function splits the input DataFrame by symbol and writes each symbol's
        data into an individual parquet file. Existing files will be either appended
        or partially overwritten depending on the `append` flag.

        Parameters
        ----------
        base_df : pd.DataFrame or pl.DataFrame
            Input dataset containing at least:
            - symbol column
            - datetime column
            - factor columns listed in `factors_list`

        factors_list : list[str]
            List of factor column names to be written or updated.

        factors_dir : str
            Directory where symbol-level parquet files are stored.
            Each symbol will be saved as: {symbol}.parquet

        trade_date_col : str
            Name of the datetime column.

        symbol_col : str
            Name of the symbol column used to split the dataset.

        append : bool, default=False
            Writing mode:
            - True  → append only new dates (no modification of existing rows).
            - False → overwrite factor values on overlapping dates only.

        max_workers : int, default=-1
            Number of parallel worker processes.
            - If -1, uses (CPU count - 1).
            - Otherwise uses the specified number of workers.
        """
        os.makedirs(factors_dir, exist_ok=True)
        if isinstance(base_df,pd.DataFrame):
            base_df= pl.from_pandas(base_df)
        base_df:pl.DataFrame = base_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime('ns')))
        
        # 原版文件中就存在 setup_logger 未定义问题；此处先注释，避免未使用且未定义导致告警/潜在运行错误。
        # logger = setup_logger()
       
        symbols = base_df[symbol_col].unique().to_list()
        small_dfs = base_df.partition_by(symbol_col)
        args_list = [
            (code, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append)
            for code, small_df in zip(symbols, small_dfs)
        ]

        n_jobs = max(os.cpu_count() - 1, 1) if max_workers == -1 else max_workers

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_code)(args)
            for args in tqdm(args_list, ncols=80)
        )
            
    @staticmethod
    def shift_datetime(base_time: datetime, duration: str, mode: str = "sub") -> datetime:
        if not isinstance(base_time, datetime.datetime):
            raise TypeError("base_time must be datetime")
        if not duration or not isinstance(duration, str):
            return base_time
        if mode not in ("add", "sub"):
            raise ValueError("mode must be 'add' or 'sub'")

        
        pattern = r"(\d+)(y|min|m|d|h|s)"
        matches = re.findall(pattern, duration)

        
        kwargs = {
            "years": 0,
            "months": 0,
            "days": 0,
            "hours": 0,
            "minutes": 0,
            "seconds": 0,
        }

        
        for value, unit in matches:
            value = int(value)
            if unit == "y":
                kwargs["years"] += value
            elif unit == "m":
                kwargs["months"] += value
            elif unit == "d":
                kwargs["days"] += value
            elif unit == "h":
                kwargs["hours"] += value
            elif unit == "min":
                kwargs["minutes"] += value
            elif unit == "s":
                kwargs["seconds"] += value

        delta = relativedelta(**kwargs)
        return base_time + delta if mode == "add" else base_time - delta

    @staticmethod
    def multiply_duration(multiplier: int, duration: str) -> str:
        if not duration or not isinstance(duration, str):
            return duration

        pattern = r"(\d+)(y|min|m|d|h|s)"
        matches = re.findall(pattern, duration)

        time_units = {
            "y": 0,
            "m": 0,
            "d": 0,
            "h": 0,
            "min": 0,
            "s": 0
        }

        for value, unit in matches:
            time_units[unit] += int(value) * multiplier

        carry = [
            ("s", "min", 60),
            ("min", "h", 60),
            ("h", "d", 24),
            ("d", "m", 30),
            ("m", "y", 12),
        ]

        for low, high, threshold in carry:
            if time_units[low] >= threshold:
                extra = time_units[low] // threshold
                time_units[low] %= threshold
                time_units[high] += extra

        order = ["y", "m", "d", "h", "min", "s"]
        result = "".join(f"{time_units[u]}{u}" for u in order if time_units[u] > 0)

        return result if result else "0s"

