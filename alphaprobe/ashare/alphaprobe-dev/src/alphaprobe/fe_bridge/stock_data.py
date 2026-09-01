"""基于 factor_engine parquet 数据源的 StockData 实现。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from shared.alphagen_qlib.stock_data import FeatureType
from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable


class FactorEngineStockData:
    """用 factor_engine 读 A 股 lqtp parquet，张量布局与 ``StockData`` 兼容。"""

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu"),
        max_files: Optional[int] = None,
        data_source_config: Optional[dict] = None,
    ) -> None:
        if max_files is None:
            env_max = os.getenv("ALPHAPROBE_FE_MAX_FILES")
            max_files = int(env_max) if env_max else None

        if data_source_config is not None:
            ds_cfg = dict(data_source_config)
            if max_files is not None and "max_files" not in ds_cfg:
                ds_cfg["max_files"] = max_files
        else:
            ensure_factor_engine_importable()
            from factor_engine.api.mining_integration import default_ashare_pv_data_source_config
            from factor_engine.storage.factory import build_data_source

            ds_cfg = default_ashare_pv_data_source_config(
                max_files=max_files,
                start_date=start_time,
                end_date=end_time,
            )
        ensure_factor_engine_importable()
        from factor_engine.storage.factory import build_data_source

        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._instrument = instrument
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.df_bak = None

        self._data_source = build_data_source(ds_cfg)
        self._engine = None
        self._formula_cache: dict[str, torch.Tensor] = {}

        self.data, self._dates, self._stock_ids = self._load_feature_tensor()

    def _build_engine(self):
        if self._engine is None:
            from factor_engine.backend.factory import build_backend
            from factor_engine.runtime.engine import FactorEngine

            backend = build_backend("pandas")
            self._engine = FactorEngine(backend=backend, data_source=self._data_source)
        return self._engine

    def _field_names(self) -> List[str]:
        from alphaprobe.fe_bridge.dsl_convert import FEATURE_TO_FIELD

        return [FEATURE_TO_FIELD[f] for f in self._features]

    def _load_columns_dict(self, names: List[str]) -> dict[str, pd.Series]:
        if hasattr(self._data_source, "load_columns"):
            return self._data_source.load_columns(names)
        return {name: self._data_source.load_column(name) for name in names}

    def _load_feature_tensor(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        names = self._field_names()
        columns = self._load_columns_dict(names)

        first = columns[names[0]]
        dates = first.index.get_level_values(0).unique().sort_values()
        stock_ids = pd.Index(first.index.get_level_values(1).unique().sort_values())

        planes: List[np.ndarray] = []
        for name in names:
            series = columns[name].reindex(
                pd.MultiIndex.from_product([dates, stock_ids]), fill_value=np.nan
            )
            plane = series.unstack(level=1).values
            planes.append(plane)

        values = np.stack(planes, axis=1)
        tensor = torch.tensor(values, dtype=torch.float, device=self.device)
        return tensor, dates, stock_ids

    def run_formula(self, dsl: str) -> pd.Series:
        from factor_engine.api.dsl_parser import parse_factor

        engine = self._build_engine()
        factor = parse_factor(dsl, name="alphaprobe_expr", surface="compat")
        result = engine.run(factor)["result"]
        if not isinstance(result, pd.Series):
            raise TypeError(f"factor_engine 返回类型异常: {type(result)!r}")
        return result

    def _series_to_plane(self, series: pd.Series) -> np.ndarray:
        """把 FE 返回的 MultiIndex Series 对齐到本 StockData 日历 → (T, N) 平面。"""
        dates = series.index.get_level_values(0).unique().sort_values()
        stock_ids = pd.Index(series.index.get_level_values(1).unique().sort_values())
        aligned = series.reindex(
            pd.MultiIndex.from_product([dates, stock_ids]), fill_value=np.nan
        )
        return aligned.unstack(level=1).values

    def evaluate_many(self, exprs: List[str]) -> List[torch.Tensor]:
        """批执行路径：一次 ``engine.run_many``（enable_cse=True）替代逐个 ``engine.run``。

        结果按输入顺序写入 ``_formula_cache``（共享子树只算一次，CSE 由 FE 批调度负责）。
        ``build_backend("pandas")`` 不变（不改 FE 语义），仅把批结果落缓存。
        """
        todo: list[str] = []
        for dsl in exprs:
            if dsl not in self._formula_cache:
                plane = self._try_load_value_cache(dsl)
                if plane is not None:
                    self._formula_cache[dsl] = torch.tensor(
                        plane, dtype=torch.float, device=self.device
                    )
                else:
                    todo.append(dsl)

        if todo:
            from factor_engine.api.dsl_parser import parse_factor

            engine = self._build_engine()
            factors = [
                parse_factor(dsl, name=f"expr_{i}", surface="compat")
                for i, dsl in enumerate(todo)
            ]
            if len(factors) == 1:
                # 单条退化为逐条 run（run_many 对单因子无 CSE 收益，且更慢）
                for dsl in todo:
                    series = self.run_formula(dsl)
                    self._formula_cache[dsl] = torch.tensor(
                        self._series_to_plane(series), dtype=torch.float, device=self.device
                    )
            else:
                batch = engine.run_many(factors, enable_cse=True)
                results = batch.get("results") or {}
                for i, dsl in enumerate(todo):
                    series = results.get(f"expr_{i}")
                    if isinstance(series, pd.Series):
                        self._formula_cache[dsl] = torch.tensor(
                            self._series_to_plane(series), dtype=torch.float, device=self.device
                        )
                    else:
                        # 批执行失败单条 → 降级逐条 run（旧路径兼容，不 Big Bang）
                        series = self.run_formula(dsl)
                        self._formula_cache[dsl] = torch.tensor(
                            self._series_to_plane(series), dtype=torch.float, device=self.device
                        )

        return [self._formula_cache[dsl] for dsl in exprs]

    def evaluate_dsl(self, dsl: str) -> torch.Tensor:
        """单条评估（兼容 wrapper）：内部走 ``evaluate_many`` 批量取单条。"""
        if dsl not in self._formula_cache:
            self.evaluate_many([dsl])
        return self._formula_cache[dsl]

    def _try_load_value_cache(self, dsl: str) -> np.ndarray | None:
        """若冷启动 value_cache 命中，则对齐到当前 StockData 日历后直接返回。"""
        try:
            cache_root = os.getenv("COLD_START_VALUE_CACHE")
            if not cache_root:
                # 默认与 cold_start_library 产物对齐
                cache_root = str(
                    Path.home()
                    / "quant_projects"
                    / "cold_start_library"
                    / "data"
                    / "ashare"
                    / "value_cache"
                )
            root = Path(os.path.expanduser(cache_root))
            if not root.is_dir():
                return None
            # 延迟 import，避免非冷启动场景硬依赖
            sys_path_extra = str(root.parents[2] / "src")  # cold_start_library/src
            if sys_path_extra not in __import__("sys").path:
                __import__("sys").path.insert(0, sys_path_extra)
            from cold_start_library.runtime.value_cache import (  # type: ignore
                load_factor_panel,
                panel_to_aligned_tensor,
            )

            loaded = load_factor_panel(root, dsl)
            if loaded is None:
                return None
            dates, stocks, values = loaded
            return panel_to_aligned_tensor(
                dates,
                stocks,
                values,
                self._dates,
                self._stock_ids,
            )
        except Exception:
            return None

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"tensor days ({n_days}) != StockData.n_days ({self.n_days})"
            )
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"tensor stocks ({n_stocks}) != StockData.n_stocks ({self.n_stocks})"
            )
        if len(columns) != n_columns:
            raise ValueError("columns size mismatch")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        flat = data.reshape(-1, n_columns)
        return pd.DataFrame(flat.detach().cpu().numpy(), index=index, columns=columns)
