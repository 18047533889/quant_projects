from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .datasource import DataSource
from logging_utils import ProgressLogger, get_logger


logger = get_logger("storage.parquet_source")


class ParquetSource(DataSource):
    """通用 Parquet 目录/文件数据源。
    
    参数:
        root: 根目录路径
        timestamp_column: 见函数签名（可选）
        instrument_column: 见函数签名（可选）
        fields: 逻辑列到物理列映射（可选）
        max_files: 见函数签名（可选）
        timestamp_unit: 见函数签名（可选）
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        recursive: 见函数签名（可选）
    
    
    约定每次按需读取一列，将其整理成 `(timestamp, instrument)` MultiIndex Series。
        既可用于 fundamentals 这类多文件目录，也可用于单文件 parquet。
    
        本实现优先尝试可选依赖 ``data_access.DuckDBEngine`` 批量读 parquet；
        未安装 ``data_access`` 时自动回退 pandas，行为与旧版兼容。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        timestamp_column: str,
        instrument_column: str,
        fields: dict[str, str] | None = None,
        max_files: int | None = None,
        timestamp_unit: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        recursive: bool = True,
    ) -> None:
        """初始化实例。
        
        参数:
            root: 根目录路径
            timestamp_column: 见函数签名（可选）
            instrument_column: 见函数签名（可选）
            fields: 逻辑列到物理列映射（可选）
            max_files: 见函数签名（可选）
            timestamp_unit: 见函数签名（可选）
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）
            recursive: 见函数签名（可选）
        
        返回:
            无
        """
        self.root = Path(root)
        self.timestamp_column = timestamp_column
        self.instrument_column = instrument_column
        self.fields = dict(fields or {})
        self.max_files = max_files
        self.timestamp_unit = timestamp_unit
        self.start_date = self._normalize_bound(start_date)
        self.end_date = self._normalize_bound(end_date)
        self.recursive = recursive
        # 单列缓存（load_column 旧契约）+ 批量列缓存（load_columns 新 API）
        self._column_cache: dict[str, Any] = {}
        self._panel_cache: dict[str, Any] = {}
        self._columns_resolved = False
        self._files_cache: list[Path] | None = None

    _TIMESTAMP_FALLBACKS: dict[str, tuple[str, ...]] = {
        "TradeDate": ("trade_date", "window_start", "timestamp", "date"),
        "window_start": ("TradeDate", "trade_date", "timestamp", "date"),
    }
    _INSTRUMENT_FALLBACKS: dict[str, tuple[str, ...]] = {
        "Symbol": ("Ticker", "symbol", "ticker"),
        "Ticker": ("Symbol", "symbol", "ticker"),
        "symbol": ("Symbol", "Ticker", "ticker"),
        "ticker": ("Ticker", "Symbol", "symbol"),
    }

    def _ensure_columns_resolved(self, sample_path: Path) -> None:
        """_ensure_columns_resolved。
        
        参数:
            sample_path: 见函数签名
        
        返回:
            无
        """
        if self._columns_resolved:
            return
        import pyarrow.parquet as pq

        available = {field.name for field in pq.read_schema(sample_path)}
        ts_col = self.timestamp_column
        if ts_col not in available:
            for alt in self._TIMESTAMP_FALLBACKS.get(ts_col, ()):
                if alt in available:
                    logger.info("timestamp 列 %s 不存在，改用 %s", ts_col, alt)
                    ts_col = alt
                    break
        inst_col = self.instrument_column
        if inst_col not in available:
            for alt in self._INSTRUMENT_FALLBACKS.get(inst_col, ()):
                if alt in available:
                    logger.info("instrument 列 %s 不存在，改用 %s", inst_col, alt)
                    inst_col = alt
                    break
        if ts_col not in available:
            raise ValueError(
                f"timestamp column {self.timestamp_column!r} not in {sample_path}: {sorted(available)}"
            )
        if inst_col not in available:
            raise ValueError(
                f"instrument column {self.instrument_column!r} not in {sample_path}: {sorted(available)}"
            )
        self.timestamp_column = ts_col
        self.instrument_column = inst_col
        self._columns_resolved = True

    # ---- 公共 API ----------------------------------------------------------

    def load_column(self, name: str):
        """旧契约：加载单列为 `(timestamp, instrument)` MultiIndex Series。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        
        
        PR1 起内部走 load_columns 批量读，连续读同一个 source 的多列时显著更快。
        """
        if name in self._column_cache:
            logger.debug("命中列缓存: %s", name)
            return self._column_cache[name]
        result = self.load_columns([name])
        return result[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """批量加载多列，返回 `{name: MultiIndex Series}`。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            dict[str, Any]
        
        
        这是 PR1 新增的核心性能 API：相比旧的多次 load_column（每次都重新扫
                parquet footer + 重读 ts/instrument 两列），这里一次 SQL 拿齐所有列。
        
                行为保证：
                    - 每个 Series 的 index 为 (timestamp, instrument)，与 load_column 一致
                    - name 若在 self.fields 里有映射，则读实际列但输出用 name
                    - 已缓存的列命中 cache；未缓存的列合并成一次查询
        
                容错：
                    首选「一次性读所有文件」的批量路径；若失败（如有个别坏 parquet），
                    自动降级到逐文件 + 跳过坏文件的模式，只要不是所有文件都坏就能返回。
        """
        # 分离已缓存 / 需要查询的列
        needed: list[str] = []
        cached: dict[str, Any] = {}
        for name in names:
            if name in self._column_cache:
                cached[name] = self._column_cache[name]
            else:
                needed.append(name)

        if not needed:
            return cached

        selected_files = self._selected_files()
        if not selected_files:
            raise FileNotFoundError(f"No parquet files found under {self.root}")

        self._ensure_columns_resolved(selected_files[0])

        logger.info(
            "开始加载列 %s（文件数=%d，root=%s）",
            needed, len(selected_files), self.root,
        )

        # 映射到真实列名（支持 fields 重命名）
        actual_names = [self.fields.get(n, n) for n in needed]
        dedup_actual = list(dict.fromkeys(actual_names))
        ts_col = self.timestamp_column
        inst_col = self.instrument_column

        try:
            frame = self._duckdb_read_batch(
                selected_files, ts_col, inst_col, dedup_actual,
            )
        except Exception as exc:
            # 批量失败：退回到逐文件 + 跳过坏文件，保留旧行为
            logger.warning(
                "批量读失败（%s），降级到逐文件读取（跳过坏文件模式）",
                exc,
            )
            frame = self._duckdb_read_per_file(
                selected_files, ts_col, inst_col, dedup_actual,
            )

        series_map = self._split_into_series(
            frame, needed=needed, actual_names=actual_names,
        )
        # 写缓存
        for n, s in series_map.items():
            self._column_cache[n] = s
        # 合并已缓存 + 新读取的
        series_map.update(cached)
        return series_map

    def prefetch_columns(self, names: list[str]) -> None:
        """批量预取列（委托 load_columns）。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        self.load_columns(names)

    def prefetch_panels(self, names: list[str]) -> None:
        """prefetch_panels。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        for name in names:
            self.load_column_panel(name)

    def load_column_panel(self, name: str):
        """加载单列为宽表 panel（index=时间, columns=标的）。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        level = series.index.names[-1] if series.index.names[-1] is not None else self.instrument_column
        panel = series.unstack(level=level)
        self._panel_cache[name] = panel
        return panel

    # ---- 内部：DuckDB 读 ---------------------------------------------------

    def _duckdb_read_batch(
        self,
        files: list[Path],
        ts_col: str,
        inst_col: str,
        value_cols: list[str],
    ):
        """一次 SQL 读所有文件所有列，返回 pandas DataFrame（未清洗）。
        
        参数:
            files: 见函数签名
            ts_col: 见函数签名
            inst_col: 见函数签名
            value_cols: 见函数签名
        
        返回:
            无
        """
        read_cols = list(dict.fromkeys([ts_col, inst_col, *value_cols]))
        col_clause = ", ".join(f'"{c}"' for c in read_cols)
        paths = [str(p) for p in files]

        try:
            from data_access.engine import get_shared_engine

            engine = get_shared_engine()
            sql = f"SELECT {col_clause} FROM read_parquet(?, union_by_name=true)"
            table = engine.execute_arrow(sql, [paths])
            return table.to_pandas(self_destruct=True, split_blocks=True)
        except ModuleNotFoundError:
            pass
        except Exception as exc:
            logger.warning("data_access DuckDB 批量读失败（%s），尝试直连 duckdb", exc)

        try:
            import duckdb

            sql = f"SELECT {col_clause} FROM read_parquet({paths!r}, union_by_name=true)"
            return duckdb.connect().execute(sql).df()
        except ImportError:
            return self._pandas_read_files(files, read_cols)
        except Exception as exc:
            logger.warning("duckdb 批量读失败（%s），回退 pandas", exc)
            return self._pandas_read_files(files, read_cols)

    def _duckdb_read_per_file(
        self,
        files: list[Path],
        ts_col: str,
        inst_col: str,
        value_cols: list[str],
    ):
        """逐文件读（fallback），跳过读失败的文件，最后 concat。
        
        参数:
            files: 见函数签名
            ts_col: 见函数签名
            inst_col: 见函数签名
            value_cols: 见函数签名
        
        返回:
            无
        """
        import pandas as pd
        try:
            from data_access.engine import get_shared_engine
        except ModuleNotFoundError:
            return self._pandas_read_files(files, [ts_col, inst_col, *value_cols], skip_errors=True)

        engine = get_shared_engine()
        read_cols = list(dict.fromkeys([ts_col, inst_col, *value_cols]))
        col_clause = ", ".join(f'"{c}"' for c in read_cols)
        sql = f"SELECT {col_clause} FROM read_parquet(?, union_by_name=true)"

        frames = []
        errors: list[str] = []
        progress = ProgressLogger(
            logger, desc=f"逐文件读取列 {value_cols}",
            total=len(files), unit="file",
        )
        for path in files:
            try:
                table = engine.execute_arrow(sql, [str(path)])
                frames.append(
                    table.to_pandas(self_destruct=True, split_blocks=True)
                )
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                logger.warning("读取 parquet 失败: %s - %s", path, exc)
            progress.advance(detail=path.name)

        if not frames:
            preview = "\n".join(errors[:5])
            raise RuntimeError(
                f"所有 parquet 都读失败: root={self.root}\n前几个错误:\n{preview}"
            )
        return pd.concat(frames, ignore_index=True)

    def _pandas_read_files(
        self,
        files: list[Path],
        read_columns: list[str],
        *,
        skip_errors: bool = False,
    ):
        """_pandas_read_files。
        
        参数:
            files: 见函数签名
            read_columns: 见函数签名
            skip_errors: 见函数签名（可选）
        
        返回:
            无
        """
        import pandas as pd

        frames = []
        errors: list[str] = []
        for path in files:
            try:
                frames.append(pd.read_parquet(path, columns=read_columns))
            except Exception as exc:
                if not skip_errors:
                    raise
                errors.append(f"{path}: {exc}")
                logger.warning("读取 parquet 失败: %s - %s", path, exc)

        if not frames:
            preview = "\n".join(errors[:5])
            raise RuntimeError(
                f"所有 parquet 都读失败: root={self.root}\n前几个错误:\n{preview}"
            )
        return pd.concat(frames, ignore_index=True)

    def _split_into_series(
        self,
        frame,
        *,
        needed: list[str],
        actual_names: list[str],
    ) -> dict[str, Any]:
        """把 DataFrame 里的各值列转成 `(timestamp, instrument)` MultiIndex Series。
        
        参数:
            frame: 长表 DataFrame
            needed: 见函数签名（可选）
            actual_names: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        
        
        处理：
                    - timestamp 时区/单位规范化
                    - instrument 规范化（strip + str）
                    - 丢弃 key 残缺行 + 时间窗口过滤 + 去重 + sort_index
        """
        import pandas as pd

        timestamps = self._coerce_timestamp(frame[self.timestamp_column])
        instruments = frame[self.instrument_column].map(self._normalize_instrument)
        valid_mask = timestamps.notna() & instruments.notna()

        sub = pd.DataFrame(
            {
                "timestamp": timestamps[valid_mask],
                "instrument": instruments[valid_mask],
            },
            copy=False,
        )
        for name, actual in zip(needed, actual_names):
            sub[name] = frame.loc[valid_mask, actual].to_numpy(copy=False)

        if self.start_date is not None:
            sub = sub[sub["timestamp"] >= self.start_date]
        if self.end_date is not None:
            sub = sub[sub["timestamp"] <= self.end_date]

        sub = sub.drop_duplicates(subset=["timestamp", "instrument"], keep="last")
        indexed = sub.set_index(["timestamp", "instrument"]).sort_index()

        result: dict[str, Any] = {}
        for name in needed:
            series = indexed[name]
            series.index = series.index.set_names(["timestamp", "instrument"])
            result[name] = series
            logger.info(
                "列 '%s' 就绪：行数=%d（root=%s）",
                name,
                len(series),
                self.root,
            )
        return result

    # ---- 文件发现 + 日期窗口筛选（和旧版相同） ----------------------------

    def _selected_files(self) -> list[Path]:
        """_selected_files。
        
        参数:
            无
        
        返回:
            list[Path]
        """
        if self._files_cache is not None:
            return self._files_cache
        if self.root.is_file():
            files = [self.root]
        else:
            if not self.root.exists():
                raise FileNotFoundError(f"Parquet source root not found: {self.root}")
            iterator = self.root.rglob("*.parquet") if self.recursive else self.root.glob("*.parquet")
            files = sorted(path for path in iterator if path.is_file())
        total_files = len(files)
        if self.start_date is not None or self.end_date is not None:
            files = [path for path in files if self._path_matches_requested_range(path)]
            logger.info(
                "按日期范围筛选 parquet 文件: before=%d, after=%d, start=%s, end=%s",
                total_files, len(files), self.start_date, self.end_date,
            )
        if self.max_files is not None:
            files = files[: self.max_files]
        logger.info("发现 parquet 文件 %d 个: root=%s", len(files), self.root)
        self._files_cache = files
        return files

    @staticmethod
    def _extract_partition_value(path: Path, key: str) -> int | None:
        """_extract_partition_value。
        
        参数:
            path: 文件或目录路径
            key: 缓存键
        
        返回:
            int | None
        """
        pattern = re.compile(rf"{re.escape(key)}=(\d{{1,4}})$")
        for part in reversed(path.parts):
            match = pattern.fullmatch(part)
            if match is not None:
                return int(match.group(1))
        return None

    @classmethod
    def _infer_path_time_window(cls, path: Path):
        """_infer_path_time_window。
        
        参数:
            path: 文件或目录路径
        
        返回:
            无
        """
        import pandas as pd

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if date_match is not None:
            day = pd.Timestamp(date_match.group(1))
            return day, day + pd.Timedelta(days=1)

        year = cls._extract_partition_value(path, "year")
        month = cls._extract_partition_value(path, "month")
        day = cls._extract_partition_value(path, "day")
        if year is None:
            return None

        if month is not None and day is not None:
            start = pd.Timestamp(year=year, month=month, day=day)
            return start, start + pd.Timedelta(days=1)
        if month is not None:
            start = pd.Timestamp(year=year, month=month, day=1)
            if month == 12:
                end = pd.Timestamp(year=year + 1, month=1, day=1)
            else:
                end = pd.Timestamp(year=year, month=month + 1, day=1)
            return start, end

        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        return start, end

    def _path_matches_requested_range(self, path: Path) -> bool:
        """_path_matches_requested_range。
        
        参数:
            path: 文件或目录路径
        
        返回:
            bool
        """
        time_window = self._infer_path_time_window(path)
        if time_window is None:
            return True

        file_start, file_end = time_window
        if self.end_date is not None and file_start > self.end_date:
            return False
        if self.start_date is not None and file_end <= self.start_date:
            return False
        return True

    # ---- 规范化 helpers（和旧版相同） -------------------------------------

    @staticmethod
    def _normalize_instrument(value: Any) -> str | None:
        """_normalize_instrument。
        
        参数:
            value: 缓存值
        
        返回:
            str | None
        """
        import pandas as pd

        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if pd.isna(value):
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_bound(value: str | None):
        """_normalize_bound。
        
        参数:
            value: 缓存值
        
        返回:
            无
        """
        if value is None:
            return None

        import pandas as pd

        bound = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(bound):
            raise ValueError(f"Invalid datetime boundary: {value}")
        if getattr(bound, "tz", None) is not None:
            bound = bound.tz_convert(None)
        return bound

    def _coerce_timestamp(self, series):
        """_coerce_timestamp。
        
        参数:
            series: MultiIndex Series
        
        返回:
            无
        """
        import pandas as pd

        if self.timestamp_unit:
            numeric = pd.to_numeric(series, errors="coerce")
            timestamps = pd.to_datetime(
                numeric, unit=self.timestamp_unit, utc=True, errors="coerce",
            )
        else:
            timestamps = pd.to_datetime(series, utc=True, errors="coerce")

        if getattr(timestamps.dt, "tz", None) is not None:
            timestamps = timestamps.dt.tz_convert(None)
        return timestamps
