"""
Mock 输入适配器：使用随机数生成模拟数据，用于开发调试和单元测试。

实现 InputAdapter 的全部抽象方法，生成可复现的随机数据。
通过固定 seed 保证相同参数下输出一致。

生成的模拟数据包括：
- alpha 信号（正态分布）
- 行情价格（几何布朗运动模拟）
- Barra 外采因子（F, G, F_mcap, F_ret, F_spec）
- 行业分类标签、风格因子暴露
- 基准等权权重、零持仓上期持仓、全 True 可交易标志
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.contracts import InputBundle, OptimizationContext
from .input_adapter import InputAdapter


@dataclass(slots=True)
class MockInputAdapter(InputAdapter):
    """Mock 输入适配器：生成可复现的随机模拟数据。

    Attributes:
        seed: 随机种子，保证可复现
        n_dates: 生成日期数
        n_assets: 生成资产数
        alpha_is_absolute_return: True 表示绝对收益场景
    """
    seed: int = 42
    n_dates: int = 60
    n_assets: int = 100
    alpha_is_absolute_return: bool = False

    def _index(self) -> pd.DatetimeIndex:
        """生成工作日日期索引。"""
        return pd.bdate_range("2024-01-02", periods=self.n_dates)

    def _columns(self) -> list[str]:
        """生成资产代码列表。"""
        return [f"ASSET{i:03d}" for i in range(self.n_assets)]

    # ---- 必填字段 ----

    def load_alpha(self, *args, **kwargs) -> pd.DataFrame:
        """生成随机 alpha 信号（标准正态分布）。"""
        rng = np.random.default_rng(self.seed)
        index = self._index()
        columns = self._columns()
        values = rng.normal(0, 1, size=(len(index), len(columns)))
        return pd.DataFrame(values, index=index, columns=columns)

    def load_market(self, *args, **kwargs) -> pd.DataFrame:
        """生成模拟行情价格（几何布朗运动，起始价 100）。"""
        rng = np.random.default_rng(self.seed + 1)
        index = self._index()
        columns = self._columns()
        # 对数收益率累计求和后取指数，模拟价格路径
        price = 100 * np.exp(rng.normal(0, 0.01, size=(len(index), len(columns))).cumsum(axis=0))
        frame = pd.DataFrame(price, index=index, columns=columns)
        return frame

    # ---- Barra 外采因子（v0.2） ----

    def load_factor_exposure(self, *args, **kwargs) -> pd.DataFrame:
        """生成 F：MultiIndex(date, asset) x factor 的因子暴露。"""
        rng = np.random.default_rng(self.seed + 10)
        index = self._index()
        assets = self._columns()
        factors = [f"FACTOR{i:02d}" for i in range(min(8, max(2, self.n_assets // 5)))]
        exposure_index = pd.MultiIndex.from_product(
            [index, assets], names=["date", "asset"]
        )
        return pd.DataFrame(
            rng.normal(0, 1, size=(len(exposure_index), len(factors))),
            index=exposure_index,
            columns=factors,
        )

    def load_industry_exposure(self, *args, **kwargs) -> pd.DataFrame:
        """生成行业暴露矩阵 G（每资产分配一个行业编号 0-9）。"""
        index = self._index()
        columns = self._columns()
        labels = [float(i % 10) for i in range(len(columns))]
        return pd.DataFrame([labels for _ in range(len(index))], index=index, columns=columns)

    def load_mcap_exposure(self, *args, **kwargs) -> pd.DataFrame:
        """生成正值市值矩阵 F_mcap（对数正态分布）。"""
        rng = np.random.default_rng(self.seed + 11)
        index = self._index()
        columns = self._columns()
        return pd.DataFrame(
            rng.lognormal(mean=10.0, sigma=1.0, size=(len(index), len(columns))),
            index=index,
            columns=columns,
        )

    def load_factor_returns(self, *args, **kwargs) -> pd.DataFrame:
        """生成因子收益矩阵 F_ret（小方差正态分布，模拟日频因子收益）。"""
        rng = np.random.default_rng(self.seed + 12)
        index = self._index()
        columns = [f"FACTOR{i:02d}" for i in range(min(8, max(2, self.n_assets // 5)))]
        return pd.DataFrame(rng.normal(0, 0.01, size=(len(index), len(columns))), index=index, columns=columns)

    def load_specific_returns(self, *args, **kwargs) -> pd.DataFrame:
        """生成特质收益矩阵 F_spec（稍大方差，模拟特质波动）。"""
        rng = np.random.default_rng(self.seed + 13)
        index = self._index()
        columns = self._columns()
        return pd.DataFrame(rng.normal(0, 0.02, size=(len(index), len(columns))), index=index, columns=columns)

    # ---- 其他可选字段 ----

    def load_industry(self, *args, **kwargs) -> pd.DataFrame:
        """生成行业分类标签（IND00-IND09）。"""
        index = self._index()
        columns = self._columns()
        labels = [f"IND{i % 10:02d}" for i in range(len(columns))]
        return pd.DataFrame([labels for _ in range(len(index))], index=index, columns=columns)

    def load_style(self, *args, **kwargs) -> pd.DataFrame:
        """生成风格因子暴露（size, value, momentum, volatility, liquidity, beta）。"""
        rng = np.random.default_rng(self.seed + 2)
        index = self._index()
        columns = self._columns()
        style_cols = ["size", "value", "momentum", "volatility", "liquidity", "beta"]
        data = {}
        for col in style_cols:
            data[col] = pd.DataFrame(rng.normal(0, 1, size=(len(index), len(columns))), index=index, columns=columns)
        return pd.concat(data, axis=1)

    def load_benchmark(self, *args, **kwargs) -> pd.DataFrame:
        """生成基准等权权重（每个资产 1/n）。"""
        index = self._index()
        columns = self._columns()
        weights = np.ones(len(columns)) / len(columns)
        return pd.DataFrame([weights for _ in range(len(index))], index=index, columns=columns)

    def load_prev_positions(self, *args, **kwargs) -> pd.DataFrame:
        """生成全零上期持仓（模拟空仓起始）。"""
        index = self._index()
        columns = self._columns()
        return pd.DataFrame(0.0, index=index, columns=columns)

    def load_tradable_flags(self, *args, **kwargs) -> pd.DataFrame:
        """生成全 True 可交易标志。"""
        index = self._index()
        columns = self._columns()
        return pd.DataFrame(True, index=index, columns=columns)

    def build_bundle(self, *args, **kwargs) -> InputBundle:
        """加载所有 mock 数据并组装为 InputBundle。

        Returns:
            包含全部 mock 数据的 InputBundle。
        """
        alpha = self.load_alpha()
        market = self.load_market()

        # Barra 外采因子
        F = self.load_factor_exposure()
        G = self.load_industry_exposure()
        F_mcap = self.load_mcap_exposure()
        F_ret = self.load_factor_returns()
        F_spec = self.load_specific_returns()

        # 其他可选字段
        industry = self.load_industry()
        style = self.load_style()
        benchmark = self.load_benchmark()
        prev_positions = self.load_prev_positions()
        tradable = self.load_tradable_flags()
        linear_cost_bps = pd.DataFrame(5.0, index=alpha.index, columns=alpha.columns)
        impact_cost = pd.DataFrame(0.01, index=alpha.index, columns=alpha.columns)

        metadata = OptimizationContext(
            alpha_is_absolute_return=self.alpha_is_absolute_return,
            calendar_name="XNYS",
            decision_time="close",
            execution_time="next_open",
            alpha_input_type="score",
            alpha_horizon_days=1,
        )
        return InputBundle(
            alpha=alpha,
            market=market,
            F=F,
            G=G,
            F_mcap=F_mcap,
            F_ret=F_ret,
            F_spec=F_spec,
            industry=industry,
            style=style,
            benchmark=benchmark,
            prev_positions=prev_positions,
            tradable=tradable,
            linear_cost_bps=linear_cost_bps,
            impact_cost=impact_cost,
            metadata=metadata,
        )
