from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ..FactorAnalyzer import FactorAnalyzer
from .Exposures_legacy import PortfolioExposures as _PortfolioExposuresBase
from .Exposures_legacy import PureExposures as _PureExposuresBase


class PortfolioExposures(_PortfolioExposuresBase):
    """基于原实现的抽象化重构版本（行为等价）。"""

    @staticmethod
    def _cross_section_ols(sub_df: pl.DataFrame, factor_cols: list[str]) -> np.ndarray:
        y = sub_df["fut_ret"].to_numpy()
        x = sub_df.select(factor_cols).to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        xtx = x.T @ x
        xtx_inv = np.linalg.pinv(xtx)
        return xtx_inv @ (x.T @ y)

    def _prepare_future_return_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
                pl.when(pl.col(self.price_col).shift(self.rebalance_period).over(self.symbol_col) == 0)
                .then(None)
                .otherwise(pl.col(self.price_col).shift(self.rebalance_period).over(self.symbol_col))
                .alias("price_fut"),
            ]
        )
        df = df.with_columns(((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret"))
        return df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )

    def _build_rebalance_mask(self, df: pl.DataFrame) -> pl.Expr:
        all_dates = df[self.trade_date_col].unique().sort()
        rebalance_dates = all_dates[:: self.rebalance_period]
        return pl.col(self.trade_date_col).is_in(rebalance_dates.implode())

    def _attach_overnight_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(self.trade_date_col).max().over([self.symbol_col, pl.col(self.trade_date_col).dt.date()]).alias("day_last_dt"),
            (pl.col(self.trade_date_col) + self.rebalance_period * self.td).alias("target_dt"),
        )
        return df.with_columns((pl.col("target_dt") > pl.col("day_last_dt")).alias("is_overnight"))

    def _apply_overnight_rule(self, df: pl.DataFrame, rebalance_mask: pl.Expr) -> pl.DataFrame:
        if self.overnight == "on":
            mask = rebalance_mask
        elif self.overnight == "off":
            mask = (~pl.col("is_overnight")) | rebalance_mask
        elif self.overnight == "only":
            mask = pl.col("is_overnight") | rebalance_mask
        else:
            raise ValueError(f"unsupported overnight mode: {self.overnight}")
        return df.filter(mask).select([self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols)

    def _standardize_exposures(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                ((pl.col(col) - pl.col(col).mean().over(self.trade_date_col)) / pl.col(col).std().over(self.trade_date_col)).alias(col)
                for col in self.exposure_cols
            ]
        )
        return df.drop_nulls([self.factor_name] + self.exposure_cols)

    def _assign_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                pl.col(self.factor_name).rank("average", descending=True).over(self.trade_date_col).alias("rank"),
                pl.len().over(self.trade_date_col).alias("n_stocks"),
            ]
        )
        df = df.sort([self.trade_date_col, "rank"])
        df = df.with_columns(pl.arange(0, pl.len(), eager=False).over(self.trade_date_col).alias("pos_index"))
        df = df.with_columns(
            (((pl.col("pos_index") * self.bins) / pl.col("n_stocks")).floor() + 1).cast(pl.Int32).alias("quantile_temp")
        )
        df = df.with_columns(
            pl.when(pl.col("quantile_temp") > self.bins).then(self.bins).otherwise(pl.col("quantile_temp")).alias("quantile")
        )
        df = df.drop(["quantile_temp", "pos_index"]).filter(pl.col(self.factor_name).is_not_null()).drop_nulls(["fut_ret"])
        return df

    def _build_mean_exposure_panel(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.position == "ls":
            mean_df = df.filter((pl.col("quantile") == 1) | (pl.col("quantile") == self.bins))
            mean_df = mean_df.with_columns(
                [
                    pl.when(pl.col("quantile") == self.bins).then(-pl.col(col)).otherwise(pl.col(col)).alias(col)
                    for col in self.exposure_cols
                ]
            )
        elif self.position == "l":
            mean_df = df.filter(pl.col("quantile") == 1)
        elif self.position == "s":
            mean_df = df.filter(pl.col("quantile") == self.bins)
            mean_df = mean_df.with_columns(
                [pl.when(pl.col("quantile") == self.bins).then(-pl.col(col)).alias(col) for col in self.exposure_cols]
            )
        else:
            raise ValueError(f"unsupported position: {self.position}")

        return (
            mean_df.group_by(self.trade_date_col)
            .agg([pl.col(col).mean().alias(f"{col}_expo") for col in self.exposure_cols])
            .sort(self.trade_date_col)
        )

    def _build_beta_panel(self, df: pl.DataFrame) -> pl.DataFrame:
        betas: list[list[float]] = []
        for _, sub_df in df.group_by(self.trade_date_col):
            beta = self._cross_section_ols(sub_df, self.exposure_cols)
            dt = sub_df[self.trade_date_col][0]
            betas.append([dt] + beta.tolist())
        return pl.DataFrame(
            betas,
            schema=[self.trade_date_col, "intercept"] + self.exposure_cols,
            orient="row",
        )

    def _build_attribution_panel(self, mean_df: pl.DataFrame, beta_df: pl.DataFrame) -> pl.DataFrame:
        result_df = mean_df.join(beta_df, on=self.trade_date_col, how="inner").fill_null(0).fill_nan(0).sort(self.trade_date_col)
        result_df = result_df.with_columns(
            [(pl.col(f"{col}_expo") * pl.col(col)).alias(f"{col}_attr") for col in self.exposure_cols]
        )
        result_df = result_df.with_columns([pl.col(f"{col}_attr").fill_nan(0) for col in self.exposure_cols])
        result_df = result_df.with_columns(
            [((1 + pl.col(f"{col}_attr").fill_nan(0)).cum_prod() - 1).alias(f"{col}_cum_ret") for col in self.exposure_cols]
        )
        result_df = result_df.with_columns(((1 + pl.col("intercept").fill_nan(0)).cum_prod() - 1).alias("intercept_cum_ret"))
        result_df = result_df.with_columns(
            (pl.col("intercept").fill_nan(0) + sum(pl.col(f"{col}_attr") for col in self.exposure_cols)).alias("portfolio_ret")
        )
        return result_df.with_columns(((1 + pl.col("portfolio_ret")).cum_prod() - 1).alias("portfolio_cum_ret"))

    def calc_stats(self):
        df = self.base_df.clone()
        df = self._prepare_future_return_frame(df)
        rebalance_mask = self._build_rebalance_mask(df)
        df = self._attach_overnight_flags(df)
        df = self._apply_overnight_rule(df, rebalance_mask)
        df = self._standardize_exposures(df)
        df = self._assign_quantiles(df)
        mean_df = self._build_mean_exposure_panel(df)
        beta_df = self._build_beta_panel(df)
        return self._build_attribution_panel(mean_df, beta_df)

    def run(self):
        self.result_df = self.calc_stats()

    def plot_portfolio_exposures(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_expo")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    mode="lines",
                    name=col,
                    marker=dict(size=6),
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Exposures Over Time ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black"),
            ),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(font=dict(size=16)),
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.result_df[self.trade_date_col],
            title_text="Date",
            title_font=dict(size=18),
            tickfont=dict(size=8),
        )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_portfolio_returns(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_cum_ret")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    line=dict(width=2),
                    mode="lines",
                    name=col,
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Cummulative Returns of Exposures over time ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black"),
            ),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(font=dict(size=16)),
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.result_df[self.trade_date_col],
            title_text="Date",
            title_font=dict(size=18),
            tickfont=dict(size=8),
        )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))

        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_portfolio_exposures_and_returns(self, staticPlot: bool = False, return_fig: bool = False):
        n_rows = len(self.exposure_cols)

        specs = [[{"secondary_y": True}] for _ in range(n_rows)]
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False, vertical_spacing=0, specs=specs)

        gap_px = 350
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px + 200

        domains = []
        cursor = total_height_px
        for height in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - height) / total_height_px
            domains.append([bottom, top])
            cursor -= height + gap_px

        for index, domain in enumerate(domains, start=1):
            axis_index = 2 * index - 1
            yaxis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
            xaxis_name = "xaxis" if index == 1 else f"xaxis{index}"

            fig.layout[yaxis_name].update(domain=domain)

            anchor_name = "y" if axis_index == 1 else f"y{axis_index}"
            fig.layout[xaxis_name].anchor = anchor_name

        date = self.result_df[self.trade_date_col]

        for row, factor in enumerate(self.exposure_cols, start=1):
            FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)
            expo_col = f"{factor}_expo"
            attr_col = f"{factor}_cum_ret"

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[expo_col],
                    fill="tozeroy",
                    name="Exposure" if row == 1 else None,
                    marker=dict(size=6),
                    line=dict(width=2, color="lightblue"),
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[attr_col],
                    marker=dict(size=6),
                    name="Cumulative Return" if row == 1 else None,
                    line=dict(width=2, color="red"),
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=True,
            )

            fig.update_yaxes(
                title_text=f"{factor} Exposure",
                row=row,
                col=1,
                secondary_y=False,
                title_font=dict(size=18),
                tickfont=dict(size=16),
            )

            fig.update_yaxes(
                title_text="Cumulative Return",
                tickformat=".2f",
                row=row,
                col=1,
                secondary_y=True,
                title_font=dict(size=18),
                tickfont=dict(size=16),
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row,
                col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8),
            )

        height_per_row = 550

        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=100, b=60, l=60, r=60),
            title=dict(text=f"Portfolio Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),
        )
        fig.update_layout(base_layout)
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig


class PureExposures(_PureExposuresBase):
    """基于原实现的抽象化重构版本（行为等价）。"""

    def _cross_section_ols(self, sub_df: pl.DataFrame) -> np.ndarray:
        y = sub_df["fut_ret"].to_numpy()
        x = sub_df.select(self.exposure_cols).to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        xtx = x.T @ x
        xtx_inv = np.linalg.pinv(xtx)
        return xtx_inv @ (x.T @ y)

    def _cross_section_corr(self, df: pl.DataFrame):
        pd_df = df.to_pandas()
        corr_panel = pd_df.groupby(self.trade_date_col).corr()
        corr = pd_df.drop(columns=self.trade_date_col).corr()
        fac_corr = corr_panel.xs(self.factor_name, level=1).drop(columns=[self.factor_name]).reset_index()
        return fac_corr, corr

    def _prepare_future_return_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
                pl.when(pl.col(self.price_col).shift(1).over(self.symbol_col) == 0)
                .then(None)
                .otherwise(pl.col(self.price_col).shift(1).over(self.symbol_col))
                .alias("price_fut"),
            ]
        )
        df = df.with_columns(((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret"))
        return df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )

    def _attach_overnight_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(self.trade_date_col).max().over([self.symbol_col, pl.col(self.trade_date_col).dt.date()]).alias("day_last_dt"),
            (pl.col(self.trade_date_col) + self.td).alias("target_dt"),
        )
        return df.with_columns((pl.col("target_dt") > pl.col("day_last_dt")).alias("is_overnight"))

    def _apply_overnight_rule(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.overnight == "off":
            mask = ~pl.col("is_overnight")
            return df.filter(mask).select([self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols)
        if self.overnight == "only":
            mask = pl.col("is_overnight")
            return df.filter(mask).select([self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols)
        return df

    def _standardize_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                ((pl.col(col) - pl.col(col).mean().over(self.trade_date_col)) / pl.col(col).std().over(self.trade_date_col)).alias(col)
                for col in self.exposure_cols + [self.factor_name]
            ]
        )
        return df.drop_nulls([self.factor_name] + self.exposure_cols)

    def _build_weighted_exposure_panel(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns((pl.col(self.factor_name) / pl.col(self.factor_name).abs().sum().over(self.trade_date_col)).alias("weight"))
        return (
            df.with_columns([(pl.col("weight") * pl.col(col)).alias(f"{col}_w") for col in self.exposure_cols])
            .group_by(self.trade_date_col)
            .agg([pl.sum(f"{col}_w").alias(f"{col}_expo") for col in self.exposure_cols])
            .sort(self.trade_date_col)
        )

    def _build_beta_panel(self, df: pl.DataFrame) -> pl.DataFrame:
        betas: list[list[float]] = []
        for _, sub_df in df.group_by(self.trade_date_col):
            beta = self._cross_section_ols(sub_df)
            dt = sub_df[self.trade_date_col][0]
            betas.append([dt] + beta.tolist())
        return pl.DataFrame(
            betas,
            schema=[self.trade_date_col, "intercept"] + self.exposure_cols,
            orient="row",
        )

    def _build_attribution_panel(self, mean_df: pl.DataFrame, beta_df: pl.DataFrame) -> pl.DataFrame:
        result_df = mean_df.join(beta_df, on=self.trade_date_col, how="inner")
        result_df = result_df.fill_null(0).fill_nan(0).sort(self.trade_date_col)
        result_df = result_df.with_columns(
            [(pl.col(f"{col}_expo") * pl.col(col)).alias(f"{col}_attr") for col in self.exposure_cols]
        )
        result_df = result_df.with_columns([pl.col(f"{col}_attr").fill_nan(0) for col in self.exposure_cols])
        result_df = result_df.with_columns(
            [((1 + pl.col(f"{col}_attr").fill_nan(0)).cum_prod() - 1).alias(f"{col}_cum_ret") for col in self.exposure_cols]
        )
        result_df = result_df.with_columns(((1 + pl.col("intercept").fill_nan(0)).cum_prod() - 1).alias("Alpha"))
        result_df = result_df.with_columns(
            (pl.col("intercept").fill_nan(0) + sum(pl.col(f"{col}_attr") for col in self.exposure_cols)).alias("portfolio_ret")
        )
        return result_df.with_columns(((1 + pl.col("portfolio_ret")).cum_prod() - 1).alias(f"{self.factor_name}_cum_ret"))

    def calc_stats(self):
        df = self.base_df.clone()
        df = self._prepare_future_return_frame(df)
        df = self._attach_overnight_flags(df)
        df = self._apply_overnight_rule(df)
        df = self._standardize_frame(df)
        mean_df = self._build_weighted_exposure_panel(df)
        beta_df = self._build_beta_panel(df)
        result_df = self._build_attribution_panel(mean_df, beta_df)
        corr_df = df.select([self.trade_date_col, self.factor_name] + self.exposure_cols)
        corr_df, corr = self._cross_section_corr(corr_df)
        return result_df, corr_df, corr

    def run(self):
        self.result_df, self.corr_df, self.corr_matrix = self.calc_stats()

    def plot_pure_exposures(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_expo")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    mode="lines",
                    name=col,
                    marker=dict(size=6),
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Exposures Over Time ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black"),
            ),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(font=dict(size=16)),
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.result_df[self.trade_date_col],
            title_text="Date",
            title_font=dict(size=18),
            tickfont=dict(size=8),
        )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_pure_returns(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_cum_ret") or col.endswith("Alpha")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    line=dict(width=2),
                    mode="lines",
                    name=col,
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Cummulative Returns of Exposures over time ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black"),
            ),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(font=dict(size=16)),
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.result_df[self.trade_date_col],
            title_text="Date",
            title_font=dict(size=18),
            tickfont=dict(size=8),
        )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))

        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_pure_exposures_and_returns(self, staticPlot: bool = False, return_fig: bool = False):
        n_rows = len(self.exposure_cols)

        specs = [[{"secondary_y": True}] for _ in range(n_rows)]
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False, vertical_spacing=0, specs=specs)

        gap_px = 350
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px + 200

        domains = []
        cursor = total_height_px
        for height in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - height) / total_height_px
            domains.append([bottom, top])
            cursor -= height + gap_px

        for index, domain in enumerate(domains, start=1):
            axis_index = 2 * index - 1

            yaxis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
            xaxis_name = "xaxis" if index == 1 else f"xaxis{index}"

            fig.layout[yaxis_name].update(domain=domain)

            anchor_name = "y" if axis_index == 1 else f"y{axis_index}"
            fig.layout[xaxis_name].anchor = anchor_name

        date = self.result_df[self.trade_date_col]

        for row, factor in enumerate(self.exposure_cols, start=1):
            FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)
            expo_col = f"{factor}_expo"
            attr_col = f"{factor}_cum_ret"

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[expo_col],
                    fill="tozeroy",
                    name="Exposure" if row == 1 else None,
                    marker=dict(size=6),
                    line=dict(width=2, color="lightblue"),
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[attr_col],
                    marker=dict(size=6),
                    name="Cumulative Return" if row == 1 else None,
                    line=dict(width=2, color="red"),
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=True,
            )

            fig.update_yaxes(
                title_text=f"{factor} Exposure",
                row=row,
                col=1,
                secondary_y=False,
                title_font=dict(size=18),
                tickfont=dict(size=16),
            )

            fig.update_yaxes(
                title_text="Cumulative Return",
                tickformat=".2f",
                row=row,
                col=1,
                secondary_y=True,
                title_font=dict(size=18),
                tickfont=dict(size=16),
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row,
                col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8),
            )

        height_per_row = 550

        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=100, b=60, l=60, r=60),
            title=dict(text=f"Pure Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),
        )
        fig.update_layout(base_layout)
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_correlations(self, staticPlot: bool = False, return_fig: bool = False):
        n_cols = len(self.exposure_cols)
        n_rows = n_cols + 3

        specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "heatmap"}]]
        specs += [[{"type": "xy"}] for _ in range(n_cols)]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False, vertical_spacing=0, specs=specs)
        gap_px = 350
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px + 200

        domains = []
        cursor = total_height_px
        for height in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - height) / total_height_px
            domains.append([bottom, top])
            cursor -= height + gap_px

        for index, domain in enumerate(domains, start=1):
            axis_name = "yaxis" if index == 1 else f"yaxis{index}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = domain
            else:
                fig.layout[axis_name] = dict(domain=domain)

        date = self.corr_df[self.trade_date_col]
        FactorAnalyzer.add_subtitle(fig, "Factor Correlations Over Time", 1)
        for factor in self.exposure_cols:
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.corr_df[factor],
                    marker=dict(size=6),
                    name=factor,
                    line=dict(width=2),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            fig.update_yaxes(
                title_text="Value",
                row=1,
                col=1,
                secondary_y=False,
                title_font=dict(size=18),
                tickfont=dict(size=16),
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=1,
                col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8),
            )

        means = self.corr_df.mean().drop("datetime").sort_values()
        FactorAnalyzer.add_subtitle(fig, "Mean Correlations", 2)
        fig.add_trace(
            go.Bar(
                y=means.index,
                x=means.values,
                name="Mean Correlations",
                orientation="h",
                showlegend=False,
                marker=dict(color=means.values, colorscale="RdYlGn", showscale=False),
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Value", row=2, col=1, title_font=dict(size=18), tickfont=dict(size=16))

        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=means.index,
            row=2,
            col=1,
            title_font=dict(size=18),
            tickfont=dict(size=16),
        )

        FactorAnalyzer.add_subtitle(fig, "Factor Correlations Matrix", 3)

        fig.add_trace(
            go.Heatmap(
                z=self.corr_matrix.values,
                x=self.corr_matrix.columns,
                y=self.corr_matrix.index,
                colorscale="RdYlGn",
                zmin=-1,
                zmid=0,
                zmax=1,
                showscale=True,
                colorbar=dict(
                    title="Value",
                    title_side="top",
                    yanchor="middle",
                    y=1 - (3 - 0.625) / n_rows,
                    len=(1 / n_rows) * 0.7,
                    x=1.04,
                ),
                text=np.round(self.corr_matrix.values, 4),
                texttemplate="%{text}",
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.corr_matrix.index,
            row=3,
            col=1,
            title_font=dict(size=18),
            tickfont=dict(size=16),
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.corr_matrix.columns,
            row=3,
            col=1,
            title_font=dict(size=18),
            tickfont=dict(size=16),
        )

        for row, factor in enumerate(self.exposure_cols, start=4):
            FactorAnalyzer.add_subtitle(fig, f"Correlations Distribution of {factor}", row)
            fig.add_trace(
                go.Histogram(x=self.corr_df[factor], nbinsx=40, name=factor, showlegend=True),
                row=row,
                col=1,
            )

        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(text=f"Factor Correlations ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),
        )
        legend_name_for_row = lambda row_index: "legend" if row_index == 1 else f"legend{row_index}"
        traces_per_row = [n_cols, 1, 1] + [1 for _ in range(n_cols)]

        trace_index = 0
        for row_index, count in enumerate(traces_per_row, start=1):
            legend_key = legend_name_for_row(row_index)
            for _ in range(count):
                if trace_index < len(fig.data):
                    fig.data[trace_index].update(legend=legend_key)
                trace_index += 1

        legend_layouts = {}
        for row_index in range(1, n_rows + 1):
            key = "legend" if row_index == 1 else f"legend{row_index}"
            y_fixed = 1 - (row_index - 0.85) / n_rows
            legend_layouts[key] = dict(
                x=1.03,
                y=y_fixed,
                xanchor="left",
                yanchor="middle",
                orientation="v",
                tracegroupgap=6,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=16, family="Arial", color="black"),
            )
        layout_updates = base_layout.copy()
        layout_updates.update(legend_layouts)
        fig.update_layout(**layout_updates)

        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
