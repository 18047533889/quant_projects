# -*- coding: utf-8
"""分钟/小时 bar → 日频面板聚合器（Phase 12 生产地基；PERF-1 快速路径）。

Design (PERF-1, 2026-08-28): 面板列全部是 MultiIndex(timestamp, instrument)
Series。旧实现 ``_stack_frame()`` 把每列 reset 成长表后 N-way outer merge，
再对长表 groupby(["trade_date", "asset"])——每个特征重复 merge 一次（12M 行
约 10s/特征）。

新快速路径把每列 ``unstack(level='instrument')`` 成宽表（行=DatetimeIndex、
列=instrument），在构造时一次性缓存；``trade_date`` 用 ``index.normalize()``
只算一次，所有特征共享。``last/first/sum/mean`` 直接用 ``resample('D')``
在宽表上算，``vwap`` 用 ``resample('D').sum()(amount)/sum()(volume)``。
宽表聚合完毕 stack 回 MultiIndex Series（索引名 ["timestamp","instrument"]）。

语义不变式（与旧长表 groupby 完全一致）：
  * 输出索引名 (timestamp, instrument) 与 dtype 不变；Series.name = 列名。
  * last/first = 每个交易日的最后一根/第一根 bar（同 day 内按 ts 升序）。
  * 全 NaN 日 → NaN（不补 0；vwap 分母 0 → replace(0, pd.NA) 语义保留）。
  * 空面板 → ValueError；缺列 → KeyError。
  * 宽表在 index 非升序时先 sort；同 (day, instrument) 存在重复 bar 时，
    resample().last() 与旧实现 sort→groupby().last() 一样取最后出现的行。

对齐假设（P0-08 session-grid 规则）：所有面板 Series 的 index 必须完全对齐
（同一组 (timestamp, instrument) 组合）。unstack 后宽表只在「真正缺失」处
产生 NaN；这一假设在 ``_require_aligned`` 中显式断言，避免静默错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from factor_engine.util.logging_utils import get_logger
from factor_engine.storage.datasource import DataSource

logger = get_logger("factor_engine.runtime.intraday_aggregator")


def _require_multiindex(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} 期望 pd.Series，得到 {type(series).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise ValueError(f"{name} 期望 MultiIndex(timestamp, instrument) Series")
    return series


def _to_daily_key(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """将 intraday timestamp 归到日历日（UTC normalize）。"""
    return pd.DatetimeIndex(ts).normalize()


@dataclass
class IntradayAggregator:
    """已对齐的 intraday 面板列；按 (trade_date, asset) 聚合为日频。

    PERF-1 快速路径：构造时把每列 unstack 成宽表并缓存（``_wide_frames``），
    日频聚合直接在宽表上 ``resample('D')`` 完成，跨特征复用。
    """

    panels: dict[str, pd.Series] = field(default_factory=dict)

    _wide_frames: dict[str, pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _trade_date: pd.DatetimeIndex | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _presence: pd.MultiIndex | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_panel(cls, panels: dict[str, pd.Series]) -> IntradayAggregator:
        """由列名 → MultiIndex Series 字典构造聚合器。"""
        normalized: dict[str, pd.Series] = {}
        for name, series in panels.items():
            normalized[name] = _require_multiindex(series, name)
        if not normalized:
            raise ValueError("IntradayAggregator 需要至少一列面板数据")
        return cls(panels=normalized)

    # ------------------------------------------------------------------ #
    # 内部：宽表缓存 + 对齐断言
    # ------------------------------------------------------------------ #

    def _reference_index(self) -> pd.MultiIndex:
        """锚定列（首列）的 index，作为对齐基准。"""
        first = next(iter(self.panels.values()))
        return first.index

    def _presence_keys(self) -> pd.MultiIndex | None:
        """构造 (day, instrument) 真实存在组合的锚点（惰性、一次性）。

        与旧长表 groupby 的 drop-absent 语义等价：锚点来自参考列 index 的
        唯一 (day, inst) 组合。宽表 stack 会在「该 instrument 其它天有数据」
        的 (day, inst) 上产生本不该存在的单元格，需要它做过滤/占位。
        """
        if self._presence is None:
            ref_index = self._reference_index()
            ts_level = pd.Index(ref_index.get_level_values(0))
            presence = pd.MultiIndex.from_arrays(
                [_to_daily_key(ts_level), ref_index.get_level_values(1)],
                names=["timestamp", "instrument"],
            )
            # 去重：同一 (day, instrument) 可能有多根 bar（一天多根 bar 是
            # 常态），仅保留唯一组合作为存在性锚点。
            self._presence = presence[~presence.duplicated()]
        return self._presence

    def _daily_wide(
        self, column: str, func: str
    ) -> pd.DataFrame:
        """宽表 resample('D') 得到日均帧。

        ``func`` ∈ {"last","first","sum","mean"}。返回的日均帧行 index =
        日历日（DatetimeIndex）、列 = instrument。
        """
        wide = self._wide_frame(column)
        return getattr(wide.resample("D"), func)()

    def _to_series(self, daily: pd.DataFrame, name: str) -> pd.Series:
        """日均帧 → MultiIndex Series，行键与旧长表 groupby 完全一致。

        用 presence 的 (day, instrument) 组合把日均帧 reindex 成完整
        行键网格（缺的行/列补 NaN 占位），再 ``stack(dropna=False)`` 并过滤
        到真实存在的组合。这样：

        * 全 NaN 源列（unstack 后 columns 为空 Index）也会产出 NaN 占位行；
        * 某 instrument 整日缺失（停牌）的 (day, inst) 不会凭空出现；
        * 输出索引名 (timestamp, instrument)、dtype、name 与旧实现一致。
        """
        presence = self._presence_keys()
        days = pd.DatetimeIndex(
            presence.get_level_values(0).drop_duplicates()
        ).sort_values()
        insts = pd.Index(presence.get_level_values(1).drop_duplicates())
        # 行键网格（days × insts）：全 NaN 列也要有占位行/占位列。用
        # ``.stack()`` 前先 ``dropna=False``（旧式 stack 保留全 NaN 位置）
        # 保证全 NaN 源列产出 NaN 占位；再按真实 (day, inst) 坐标覆盖 daily
        # 的值，模拟旧长表 outer-merge 的行键语义。
        # 注意: daily 已经是按 insts 重排（reindex）后的帧，这里直接基于它。
        daily = daily.reindex(index=days, columns=insts)
        # 把 daily 的每个单元格（含全 NaN 的 day×inst）都纳入行键，再按真实
        # 组合过滤。future_stack 不丢全 NaN 行（它是显式占位帧，无 NaN 丢弃
        # 语义），因此空列也能产出 NaN 占位。
        stacked = daily.stack(future_stack=True).rename(name)
        stacked = stacked.sort_index()
        # 过滤回真实存在组合（full 网格可能比 presence 多出无 bar 的组合）。
        keep = presence.to_flat_index()
        stacked = stacked[stacked.index.to_flat_index().isin(keep)]
        return stacked

    def _require_aligned(self) -> None:
        """P0-08 session-grid 规则：所有面板列 index 必须与首列完全对齐。

        快速路径按「宽表只对真正缺失产生 NaN」设计，依赖列间严格对齐。
        这里做一次性显式断言（构造后首次聚合时触发），把静默错位变成
        可诊断错误。
        """
        ref = self._reference_index()
        for name, series in self.panels.items():
            if not series.index.equals(ref):
                raise ValueError(
                    "IntradayAggregator 面板列未完全对齐（P0-08 session-grid "
                    "规则要求各列 index 一致）："
                    f"列 {name!r} 的 index 与首列不同 "
                    f"(长度 {len(series.index)} vs {len(ref)}, "
                    f"names {series.index.names} vs {ref.names})。"
                    "请先对列做坐标对齐（如 reindex）后再构造聚合器。"
                )

    def _wide_frame(self, column: str) -> pd.DataFrame:
        """宽表缓存：index=DatetimeIndex 行、columns=instrument、values=列值。

        惰性构建并缓存；返回内部副本的只读视图（不 copy）。
        """
        frame = self._wide_frames.get(column)
        if frame is None:
            self._require_aligned()
            series = self.panels[column]
            if series.index.has_duplicates:
                # 同 (timestamp, instrument) 出现重复行（如同一分钟两根 bar）：
                # unstack 会报 duplicate 错。保持与旧长表 sort→groupby().last()
                # 语义一致（按 index 排序后保留最后出现行），先做最后一行去重。
                series = series[~series.index.duplicated(keep="last")]
            wide = series.unstack(level="instrument")
            if not isinstance(wide.index, pd.DatetimeIndex):
                wide.index = pd.DatetimeIndex(wide.index)
            if not wide.index.is_monotonic_increasing:
                wide = wide.sort_index()
            frame = wide
            self._wide_frames[column] = frame
            if self._trade_date is None:
                # 全面板共享的 trade_date（UTC normalize），只算一次。
                self._trade_date = _to_daily_key(frame.index)
            # presence 锚点惰性构造（见 _presence_keys）。
            self._presence_keys()
        return frame

    def _fast_daily(self, column: str, func: str) -> pd.Series:
        """宽表 resample('D') 后 stack，并过滤到真实存在的 (day, instrument)。

        ``func`` ∈ {"last","first","sum","mean"}。过滤步骤与旧长表 groupby
        （drop 掉没有 bar 行的 (trade_date, asset)）等价。因锚点来自参考列
        index 的唯一 (day, inst) 组合，宽表 stack 在多出的一天没有该
        instrument 数据时（如 S00000 首日停牌），对应单元格会被剔除。
        """
        wide = self._wide_frame(column)
        daily = getattr(wide.resample("D"), func)()
        return self._to_series(daily, column)

    def _resample_last_first(
        self, column: str, how: str
    ) -> pd.Series:
        """宽表 resample('D') 后 stack 回 MultiIndex Series。

        ``how`` ∈ {"last", "first"}。与旧实现 sort→groupby(...).last()/.first()
        语义一致：每交易日内取时间序上最后/第一根 bar。同 (day, inst) 若存在
        重复 bar（同一交易日多次出现），resample 在日窗口内按行序取末行/首行，
        与旧实现的 sort→last/first 等价。
        """
        wide = self._wide_frame(column)
        resampler = wide.resample("D")
        daily = (
            getattr(resampler, how)() if how == "last" else resampler.first()
        )
        return self._to_series(daily, column)

    # ------------------------------------------------------------------ #
    # 公共 API（不变）
    # ------------------------------------------------------------------ #

    def last(self, column: str) -> pd.Series:
        """日末值（last bar of day）。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        return self._resample_last_first(column, "last")

    def first(self, column: str) -> pd.Series:
        """日初值（first bar of day）。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        return self._resample_last_first(column, "first")

    def _resample_apply(self, column: str, func: str) -> pd.Series:
        """宽表 resample('D') 后 stack 回 MultiIndex Series。"""
        wide = self._wide_frame(column)
        daily = getattr(wide.resample("D"), func)()
        return self._to_series(daily, column)

    def sum(self, column: str) -> pd.Series:
        """按 (trade_date, asset) 对列求和。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        return self._resample_apply(column, "sum")

    def mean(self, column: str) -> pd.Series:
        """按 (trade_date, asset) 对列求均值。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        return self._resample_apply(column, "mean")

    def vwap(
        self,
        *,
        amount: str = "amount",
        volume: str = "volume",
        price: str = "close",
    ) -> pd.Series:
        """成交额加权均价：优先 ``sum(Amount) / sum(Volume)``。

        ``price`` is retained for backward compatibility with callers whose
        minute source has no amount column.  Production multi-minute bars
        should always provide amount because close×volume is not bar VWAP.
        """
        if volume not in self.panels:
            raise KeyError(f"vwap 需要 {volume!r} 列")
        v_wide = self._wide_frame(volume)
        if amount in self.panels:
            numerator = self._wide_frame(amount).resample("D").sum()
        elif price in self.panels:
            numerator = (self._wide_frame(price) * v_wide).resample("D").sum()
        else:
            raise KeyError(f"vwap 需要 {amount!r} 或 {price!r} 列")
        denominator = v_wide.resample("D").sum()
        return self._to_series(
            numerator / denominator.replace(0, pd.NA), "vwap"
        )

    def aggregate_features(
        self,
        *,
        features: tuple[str, ...] = ("last_close", "sum_volume", "vwap"),
    ) -> dict[str, pd.Series]:
        """批量产出常用日频特征。"""
        out: dict[str, pd.Series] = {}
        for feat in features:
            if feat == "last_close":
                out["close"] = self.last("close")
            elif feat == "first_open":
                out["open"] = self.first("open")
            elif feat == "sum_volume":
                out["volume"] = self.sum("volume")
            elif feat == "vwap":
                out["vwap"] = self.vwap()
            elif feat == "mean_close":
                out["close_mean"] = self.mean("close")
            else:
                raise ValueError(f"未知 intraday 日频特征: {feat!r}")
        return out

    # ------------------------------------------------------------------ #
    # 兼容回退：供旧调用方使用的长表参考实现（等值基准）
    # ------------------------------------------------------------------ #

    def _stack_frame(self) -> pd.DataFrame:
        """MultiIndex Series 字典 → 长表 [datetime, asset, *cols]。

        与 PERF-1 之前版本逐字相同的参考实现。仅测试/等值基准使用；
        快速路径（last/first/sum/mean/vwap）不再经过本方法。
        """
        frames: list[pd.DataFrame] = []
        for col, series in self.panels.items():
            df = series.reset_index()
            names = list(df.columns)
            df.columns = ["datetime", "asset", col]
            frames.append(df)
        out = frames[0]
        for extra in frames[1:]:
            out = out.merge(extra, on=["datetime", "asset"], how="outer")
        out["trade_date"] = _to_daily_key(pd.DatetimeIndex(out["datetime"]))
        return out

    def _last_long(self, column: str) -> pd.Series:
        """旧 last 实现（长表 groupby），等值基准用。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        daily = (
            df.sort_values(["asset", "datetime"])
            .groupby(["trade_date", "asset"], as_index=False)
            .last()
        )
        idx = pd.MultiIndex.from_arrays(
            [daily["trade_date"], daily["asset"]],
            names=["timestamp", "instrument"],
        )
        return pd.Series(daily[column].values, index=idx, name=column)

    def _first_long(self, column: str) -> pd.Series:
        """旧 first 实现（长表 groupby），等值基准用。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        daily = (
            df.sort_values(["asset", "datetime"])
            .groupby(["trade_date", "asset"], as_index=False)
            .first()
        )
        idx = pd.MultiIndex.from_arrays(
            [daily["trade_date"], daily["asset"]],
            names=["timestamp", "instrument"],
        )
        return pd.Series(daily[column].values, index=idx, name=column)

    def _sum_long(self, column: str) -> pd.Series:
        """旧 sum 实现（长表 groupby），等值基准用。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        grouped = df.groupby(["trade_date", "asset"])[column].sum()
        grouped.index.names = ["timestamp", "instrument"]
        return grouped.rename(column)

    def _mean_long(self, column: str) -> pd.Series:
        """旧 mean 实现（长表 groupby），等值基准用。"""
        if column not in self.panels:
            raise KeyError(f"列 {column!r} 不在面板中")
        df = self._stack_frame()
        grouped = df.groupby(["trade_date", "asset"])[column].mean()
        grouped.index.names = ["timestamp", "instrument"]
        return grouped.rename(column)

    def _vwap_long(
        self,
        *,
        amount: str = "amount",
        volume: str = "volume",
        price: str = "close",
    ) -> pd.Series:
        """旧 vwap 实现（长表 groupby），等值基准用。"""
        if volume not in self.panels:
            raise KeyError(f"vwap 需要 {volume!r} 列")
        df = self._stack_frame()
        if amount in self.panels:
            numerator = df.groupby(["trade_date", "asset"])[amount].sum()
        elif price in self.panels:
            numerator = (df[price] * df[volume]).groupby(
                [df["trade_date"], df["asset"]]
            ).sum()
        else:
            raise KeyError(f"vwap 需要 {amount!r} 或 {price!r} 列")
        denominator = df.groupby(["trade_date", "asset"])[volume].sum()
        vwap = (numerator / denominator.replace(0, pd.NA)).rename("vwap")
        vwap.index.names = ["timestamp", "instrument"]
        return vwap


@dataclass
class IntradayAggregatedDataSource(DataSource):
    """包装 intraday 数据源，对外暴露日频聚合列。"""

    inner: DataSource
    features: tuple[str, ...] = ("last_close", "sum_volume", "vwap")
    bar_freq: str = "1d"
    _cache: dict[str, pd.Series] = field(default_factory=dict, init=False, repr=False)

    def execution_spec(self) -> dict[str, Any] | None:
        """#收官轮 P0：还原 intraday_daily 子源 + features 可重建配置。"""
        from factor_engine.storage.sources.datasource import clean_execution_spec

        fn = getattr(self.inner, "execution_spec", None)
        inner_spec = fn() if callable(fn) else None
        if not isinstance(inner_spec, dict):
            return None
        return clean_execution_spec(
            {
                "type": "intraday_daily",
                "source": dict(inner_spec),
                "features": tuple(self.features),
            }
        )

    def _build_aggregator(self) -> IntradayAggregator:
        load_columns = getattr(self.inner, "load_columns", None)
        base_cols = ["open", "high", "low", "close", "volume", "amount"]
        if callable(load_columns):
            try:
                panels = load_columns(base_cols)
            except (FileNotFoundError, KeyError, NotImplementedError):
                panels = {}
                for col in base_cols:
                    try:
                        panels[col] = self.inner.load_column(col)
                    except (FileNotFoundError, KeyError, NotImplementedError):
                        continue
        else:
            panels = {}
            missing: list[str] = []
            for col in base_cols:
                try:
                    panels[col] = self.inner.load_column(col)
                except (FileNotFoundError, KeyError, NotImplementedError) as exc:
                    missing.append(col)
                    logger.debug("intraday 列 %s 不可用: %s", col, exc)
            if missing:
                logger.warning(
                    "IntradayAggregatedDataSource 缺少列 %s（将跳过）；已加载 %s",
                    missing,
                    sorted(panels.keys()),
                )
        if "close" not in panels:
            raise ValueError("intraday 数据源至少需要 close 列")
        return IntradayAggregator.from_panel(panels)

    def load_column(self, name: str) -> Any:
        """加载日频聚合列（首次访问时批量计算并缓存）。"""
        if name in self._cache:
            return self._cache[name]
        agg = self._build_aggregator()
        features = agg.aggregate_features(features=self.features)
        self._cache.update(features)
        if name not in self._cache:
            raise KeyError(
                f"IntradayAggregatedDataSource 无列 {name!r}；"
                f"可用: {sorted(features.keys())}"
            )
        return self._cache[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """批量加载多个日频聚合列。"""
        return {n: self.load_column(n) for n in names}


# --------------------------------------------------------------------------- #
# IntradayFeatureCompiler (P0#5, 100k GO §8 / §110 item 5)
#
# One authoritative entry point that computes MANY minute-feature operators
# from a SINGLE scan of the minute panel.  The (day, bar, inst) grid is
# materialized once and shared across every grid-routed operator; non-grid
# operators fall back to their standalone ``_calculate_series`` path.
#
# Grid-routed operators (PERF-2 whitelisted vector kernels) reuse the shared
# grid via the ``_grid`` keyword on ``_core.daily_agg{,_two,_three}``.  The
# scan counter proves the single-scan property: it increments once per grid
# materialization, not once per operator.
# --------------------------------------------------------------------------- #

#: Operators whose vector kernel can run on a shared (day, bar, inst) grid.
#: ``fields`` = the minute panels they consume (in grid order), ``min_finite``
#: mirrors the standalone ``daily_agg*`` call, ``vec`` = the vector kernel.
_GRID_ROUTED: dict[str, dict[str, Any]] = {}


def _register_grid_routed(name: str, fields: list[str], min_finite: int, vec) -> None:
    _GRID_ROUTED[name] = {
        "fields": list(fields),
        "min_finite": int(min_finite),
        "vec": vec,
    }


def _install_grid_routed() -> None:
    """Lazily register the grid-routed operators (avoids import cycles)."""
    if _GRID_ROUTED:
        return
    from factor_engine.cleaned_operators.intraday import _core as _c

    _register_grid_routed(
        "intra_session_mean_reversion", ["close"], 5, _c._vec_session_mean_reversion
    )
    _register_grid_routed(
        "intra_price_delay", ["close", "volume"], 5, _c._vec_price_delay
    )
    _register_grid_routed(
        "intra_volume_imbalance", ["close", "volume"], 3, _c._vec_volume_imbalance
    )


@dataclass
class IntradayFeatureCompiler:
    """Compute many minute-feature operators from one scan of the minute panel.

    Parameters
    ----------
    fields : tuple[str, ...]
        The minute panel columns the compiler may consume (canonical names).
    timezone : str
        Session timezone for minute math (default ``Asia/Shanghai``).
    session : str
        Session key (default ``ashare_regular``).

    The compiler is constructed with the *available* minute fields; the actual
    panels are supplied to ``compute_many`` (either as a ``panels`` dict of
    MultiIndex Series, or via a ``source`` exposing ``load_columns``).
    """

    fields: tuple[str, ...] = (
        "Open", "High", "Low", "Close", "Volume", "Amount", "Vwap",
    )
    timezone: str = "Asia/Shanghai"
    session: str = "ashare_regular"

    _scan_count: int = field(default=0, init=False, repr=False)
    _grid_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def scan_count(self) -> int:
        """Number of (day, bar, inst) grid materializations performed.

        A single ``compute_many`` over N grid-routed operators must leave this
        at 1 (one scan), not N.
        """
        return self._scan_count

    # ------------------------------------------------------------------ #
    # Panel loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_panels(source: Any, fields: list[str]) -> dict[str, pd.Series]:
        """Load minute panels from a source exposing ``load_columns``."""
        load_columns = getattr(source, "load_columns", None)
        if callable(load_columns):
            try:
                loaded = load_columns(fields)
                if isinstance(loaded, dict):
                    return {k: v for k, v in loaded.items() if k in fields}
            except (FileNotFoundError, KeyError, NotImplementedError):
                pass
        panels: dict[str, pd.Series] = {}
        for f in fields:
            try:
                panels[f] = source.load_column(f)
            except (FileNotFoundError, KeyError, NotImplementedError):
                continue
        return panels

    @staticmethod
    def _to_wide(series: pd.Series) -> pd.DataFrame:
        """MultiIndex(timestamp, instrument) Series -> wide DataFrame."""
        if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
            raise ValueError(
                "IntradayFeatureCompiler 面板列期望 MultiIndex(timestamp, instrument) Series"
            )
        wide = series.unstack(level="instrument")
        if not isinstance(wide.index, pd.DatetimeIndex):
            wide.index = pd.DatetimeIndex(wide.index)
        if not wide.index.is_monotonic_increasing:
            wide = wide.sort_index()
        return wide

    # ------------------------------------------------------------------ #
    # Grid materialization (the single scan)
    # ------------------------------------------------------------------ #

    def _build_grid(self, panels: dict[str, pd.Series], fields: list[str]) -> tuple:
        """Materialize the (day, bar, inst) grid for ``fields`` once.

        Uses ``_core._grid3`` so the grid is byte-identical to the standalone
        vectorized path.  Increments ``_scan_count`` exactly once per call.
        """
        from factor_engine.cleaned_operators.intraday import _core as _c

        key = tuple(fields)
        if key in self._grid_cache:
            return self._grid_cache[key]
        wides = [self._to_wide(panels[f]) for f in fields]
        # _grid3 accepts up to 3 panels; pad with None.
        a = wides[0]
        b = wides[1] if len(wides) > 1 else None
        c = wides[2] if len(wides) > 2 else None
        grid = _c._grid3(a, b, c)
        self._grid_cache[key] = grid
        self._scan_count += 1
        return grid

    # ------------------------------------------------------------------ #
    # compute_many
    # ------------------------------------------------------------------ #

    def compute_many(
        self,
        features: list[Any],
        *,
        dates: Any = None,
        universe: Any = None,
        panels: dict[str, pd.Series] | None = None,
        source: Any = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute many minute-feature operators from one scan of the panel.

        Parameters
        ----------
        features : list
            Operator names (canonical) or ``(name, params)`` tuples.  Each is
            resolved through ``OperatorRegistry`` and computed.
        dates / universe : optional
            Accepted for API parity with the GO-prompt sketch; not yet used to
            filter (the panels are assumed pre-scoped).
        panels : dict[str, pd.Series], optional
            MultiIndex(timestamp, instrument) Series keyed by field name.
        source : optional
            Data source exposing ``load_columns`` / ``load_column``; used to
            load panels when ``panels`` is not given.

        Returns
        -------
        dict[str, pd.DataFrame]
            Feature name -> daily panel (index=date, columns=instrument).
        """
        _install_grid_routed()

        # Resolve feature specs to (name, params).
        specs: list[tuple[str, dict]] = []
        for feat in features:
            if isinstance(feat, (tuple, list)):
                name, params = feat[0], dict(feat[1]) if len(feat) > 1 else {}
            else:
                name, params = str(feat), {}
            specs.append((name, params))

        # Determine the union of minute fields needed by grid-routed features.
        grid_fields: list[str] = []
        grid_specs: list[tuple[str, dict]] = []
        fallback_specs: list[tuple[str, dict]] = []
        for name, params in specs:
            route = _GRID_ROUTED.get(name)
            if route is not None:
                grid_specs.append((name, params))
                for f in route["fields"]:
                    if f not in grid_fields:
                        grid_fields.append(f)
            else:
                fallback_specs.append((name, params))

        # Load panels if not supplied.
        if panels is None:
            if source is None:
                raise ValueError("compute_many 需要 panels 或 source")
            need = list(grid_fields)
            for name, _p in fallback_specs:
                op = self._resolve_operator(name)
                if op is not None:
                    for f in getattr(op.metadata, "param_names", []) or []:
                        if f not in need:
                            need.append(f)
            panels = self._load_panels(source, need)
        if not panels:
            raise ValueError("compute_many 需要至少一列面板数据")

        out: dict[str, pd.DataFrame] = {}

        # --- grid-routed operators: ONE scan, shared grid ------------------ #
        if grid_specs:
            grid = self._build_grid(panels, grid_fields)
            for name, params in grid_specs:
                route = _GRID_ROUTED[name]
                vec = route["vec"]
                min_finite = params.get("min_finite", route["min_finite"])
                # The vec kernel reads its inputs from the shared grid by
                # position: fields[0] -> a, fields[1] -> b, fields[2] -> c.
                n = len(route["fields"])
                # The vec kernel reads its inputs from the shared grid by
                # position; the panel args are only used for ``.columns``, so
                # pass the wide frames (index=timestamp, columns=instrument).
                wide = [self._to_wide(panels[f]) for f in route["fields"]]
                if n == 1:
                    result = vec(wide[0], min_finite=min_finite, _grid=grid)
                elif n == 2:
                    result = vec(wide[0], wide[1], min_finite=min_finite, _grid=grid)
                else:
                    result = vec(wide[0], wide[1], wide[2], min_finite=min_finite, _grid=grid)
                out[name] = result

        # --- fallback operators: standalone _calculate_series path ---------- #
        for name, params in fallback_specs:
            op = self._resolve_operator(name)
            if op is None:
                raise ValueError(f"未知 intraday 算子: {name!r}")
            result = self._run_fallback(op, panels, params)
            out[name] = result

        return out

    @staticmethod
    def _resolve_operator(name: str):
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.get(name, mode="any")

    @staticmethod
    def _run_fallback(op, panels: dict[str, pd.Series], params: dict) -> pd.DataFrame:
        """Run a non-grid operator through its standalone ``_calculate_series``.

        The operator's ``metadata.param_names`` declare the minute panels it
        consumes; each is passed as a wide DataFrame (index=timestamp,
        columns=instrument) — the same shape the standalone path expects.
        """
        param_names = list(getattr(op.metadata, "param_names", []) or [])
        args = []
        for p in param_names:
            if p in panels:
                args.append(IntradayFeatureCompiler._to_wide(panels[p]))
            else:
                args.append(None)
        return op.calculate(*args, **params)
