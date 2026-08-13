"""
Polars native implementations for state machine operators (state_* family)

State machine operators track evolving states with memory across rows:
- Use .shift() to access previous state values
- Use .when().then().otherwise() for conditional state transitions
- Use .cum_sum() for cumulative metrics
- State resets on NaN or condition changes
- All operators are causal (depend only on previous rows)
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Union

from cleaned_operators.base import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
    ParamSpec,
    ParamRole,
)


# ============================================================================
# Adaptive Deadband and Slew Limit
# ============================================================================

@register_operator(name="state_adaptive_deadband", canonical="state_adaptive_deadband", backend="polars")
class StateAdaptiveDeadbandPolarsNative(SeriesOperator):
    """Adaptive deadband filter where threshold adjusts based on recent volatility"""
    metadata = OperatorMetadata(
        name="state_adaptive_deadband",
        category="state",
        description="Adaptive deadband filter where threshold adjusts based on recent volatility",
        param_names=["signal", "base_threshold", "window"],
        return_type="series",
        tags=["state", "filter", "polars_native"],
    )
    metadata.param_specs = {
        "base_threshold": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, signal, base_threshold=0.02, window=20, **kwargs):
        """
        Signal only changes when it moves beyond adaptive threshold from last state.
        Threshold = base_threshold * rolling_std(signal, window)
        """
        df = pl.from_pandas(signal.to_frame()).with_row_count("_idx")
        col_name = signal.name

        result = (
            df.lazy()
            .with_columns([
                # Calculate adaptive threshold
                (pl.col(col_name).rolling_std(window_size=window, min_periods=1) * base_threshold)
                .fill_nan(base_threshold)
                .alias("_threshold")
            ])
            .with_columns([
                # Initialize state with first value
                pl.col(col_name).alias("_state")
            ])
            .collect()
        )

        # Apply deadband logic using pandas for stateful iteration
        state_vals = result[col_name].to_numpy()
        threshold_vals = result["_threshold"].to_numpy()
        output = np.full(len(state_vals), np.nan)

        if len(state_vals) > 0:
            output[0] = state_vals[0]
            current_state = state_vals[0] if np.isfinite(state_vals[0]) else np.nan

            for i in range(1, len(state_vals)):
                if not np.isfinite(state_vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = state_vals[i]
                    output[i] = current_state
                elif abs(state_vals[i] - current_state) > threshold_vals[i]:
                    current_state = state_vals[i]
                    output[i] = current_state
                else:
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_adaptive_slew_limit", canonical="state_adaptive_slew_limit", backend="polars")
class StateAdaptiveSlewLimitPolarsNative(SeriesOperator):
    """Adaptive slew rate limiter where max change rate adjusts to volatility"""
    metadata = OperatorMetadata(
        name="state_adaptive_slew_limit",
        category="state",
        description="Adaptive slew rate limiter where max change rate adjusts to volatility",
        param_names=["signal", "base_rate", "window"],
        return_type="series",
        tags=["state", "filter", "polars_native"],
    )
    metadata.param_specs = {
        "base_rate": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, signal, base_rate=0.1, window=20, **kwargs):
        """
        State can only change by max_rate * rolling_std per step.
        max_rate = base_rate * rolling_std(signal, window)
        """
        df = pl.from_pandas(signal.to_frame())
        col_name = signal.name

        result = (
            df.lazy()
            .with_columns([
                (pl.col(col_name).rolling_std(window_size=window, min_periods=1) * base_rate)
                .fill_nan(base_rate)
                .alias("_max_rate")
            ])
            .collect()
        )

        state_vals = result[col_name].to_numpy()
        max_rate_vals = result["_max_rate"].to_numpy()
        output = np.full(len(state_vals), np.nan)

        if len(state_vals) > 0:
            output[0] = state_vals[0]
            current_state = state_vals[0] if np.isfinite(state_vals[0]) else np.nan

            for i in range(1, len(state_vals)):
                if not np.isfinite(state_vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = state_vals[i]
                    output[i] = current_state
                else:
                    delta = state_vals[i] - current_state
                    max_change = max_rate_vals[i]
                    clamped_delta = np.clip(delta, -max_change, max_change)
                    current_state = current_state + clamped_delta
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


# ============================================================================
# Confidence and Cost Aware Operators
# ============================================================================

@register_operator(name="state_confidence_weighted_ema", canonical="state_confidence_weighted_ema", backend="polars")
class StateConfidenceWeightedEmaPolarsNative(SeriesOperator):
    """EMA where smoothing adapts to confidence level"""
    metadata = OperatorMetadata(
        name="state_confidence_weighted_ema",
        category="state",
        description="EMA where smoothing adapts to confidence level (high confidence = faster adaptation)",
        param_names=["signal", "confidence", "base_halflife"],
        return_type="series",
        tags=["state", "ema", "polars_native"],
    )
    metadata.param_specs = {
        "base_halflife": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, signal, confidence, base_halflife=10, **kwargs):
        """
        Effective alpha = base_alpha * confidence
        Higher confidence leads to faster adaptation (shorter effective halflife)
        """
        df = pl.from_pandas(pd.concat([
            signal.to_frame(),
            confidence.to_frame().rename(columns={confidence.name: "_conf"})
        ], axis=1))

        col_name = signal.name
        base_alpha = np.where(base_halflife) != 0, (1.0 - np.exp(-np.log(2)) / (base_halflife)), np.nan)

        vals = df[col_name].to_numpy()
        conf_vals = df["_conf"].to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            ema = vals[0] if np.isfinite(vals[0]) else np.nan
            output[0] = ema

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = ema
                    continue

                conf = conf_vals[i] if np.isfinite(conf_vals[i]) else 0.5
                conf = np.clip(conf, 0.0, 1.0)
                effective_alpha = base_alpha * (0.1 + 0.9 * conf)  # Min 0.1x, max 1.0x

                if not np.isfinite(ema):
                    ema = vals[i]
                else:
                    ema = effective_alpha * vals[i] + (1.0 - effective_alpha) * ema
                output[i] = ema

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_cost_aware_deadband", canonical="state_cost_aware_deadband", backend="polars")
class StateCostAwareDeadbandPolarsNative(SeriesOperator):
    """Deadband filter that considers transaction cost"""
    metadata = OperatorMetadata(
        name="state_cost_aware_deadband",
        category="state",
        description="Deadband filter where threshold reflects transaction cost and expected benefit",
        param_names=["signal", "transaction_cost", "base_threshold"],
        return_type="series",
        tags=["state", "filter", "cost", "polars_native"],
    )
    metadata.param_specs = {
        "transaction_cost": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
        "base_threshold": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, signal, transaction_cost=0.001, base_threshold=0.02, **kwargs):
        """
        Effective threshold = base_threshold + transaction_cost
        Signal must overcome both noise threshold and transaction cost
        """
        effective_threshold = base_threshold + transaction_cost

        vals = signal.to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            output[0] = vals[0]
            current_state = vals[0] if np.isfinite(vals[0]) else np.nan

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                elif abs(vals[i] - current_state) > effective_threshold:
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_cost_aware_slew", canonical="state_cost_aware_slew", backend="polars")
class StateCostAwareSlewPolarsNative(SeriesOperator):
    """Slew limiter where cost increases with adjustment speed"""
    metadata = OperatorMetadata(
        name="state_cost_aware_slew",
        category="state",
        description="Slew limiter that penalizes rapid changes with higher effective cost",
        param_names=["signal", "base_rate", "cost_factor"],
        return_type="series",
        tags=["state", "filter", "cost", "polars_native"],
    )
    metadata.param_specs = {
        "base_rate": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
        "cost_factor": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, signal, base_rate=0.1, cost_factor=0.01, **kwargs):
        """
        Max change decreases as cost_factor increases
        effective_rate = base_rate / (1 + cost_factor)
        """
        effective_rate = (base_rate) / ((1.0 + cost_factor)) if ((1.0 + cost_factor)) != 0 else np.nan

        vals = signal.to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            output[0] = vals[0]
            current_state = vals[0] if np.isfinite(vals[0]) else np.nan

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    delta = vals[i] - current_state
                    clamped_delta = np.clip(delta, -effective_rate, effective_rate)
                    current_state = current_state + clamped_delta
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


# ============================================================================
# Episode Metrics
# ============================================================================

@register_operator(name="state_episode_efficiency", canonical="state_episode_efficiency", backend="polars")
class StateEpisodeEfficiencyPolarsNative(SeriesOperator):
    """Ratio of net excursion to path length in current episode"""
    metadata = OperatorMetadata(
        name="state_episode_efficiency",
        category="state",
        description="Efficiency of price movement: abs(net_move) / total_path_length within episode",
        param_names=["price", "state"],
        return_type="series",
        tags=["state", "episode", "polars_native"],
    )

    def _calculate_series(self, price, state, **kwargs):
        """
        Episode: consecutive non-zero state values
        Efficiency = |price_now - price_entry| / sum(|price[i] - price[i-1]|)
        """
        df = pl.from_pandas(pd.concat([
            price.to_frame(),
            state.to_frame().rename(columns={state.name: "_state"})
        ], axis=1))

        price_vals = df[price.name].to_numpy()
        state_vals = df["_state"].to_numpy()
        output = np.full(len(price_vals), np.nan)

        entry_price = np.nan
        path_length = 0.0
        last_price = np.nan
        in_episode = False

        for i in range(len(price_vals)):
            s = state_vals[i]
            p = price_vals[i]

            # Check if in episode (non-zero, finite state)
            episode_active = np.isfinite(s) and abs(s) > 1e-9

            if not episode_active or not np.isfinite(p):
                # Episode ended or invalid data
                entry_price = np.nan
                path_length = 0.0
                last_price = np.nan
                in_episode = False
                output[i] = np.nan
                continue

            if not in_episode:
                # Start new episode
                entry_price = p
                path_length = 0.0
                last_price = p
                in_episode = True
                output[i] = 0.0  # No movement yet
            else:
                # Continue episode
                if np.isfinite(last_price):
                    path_length += abs(p - last_price)
                last_price = p

                net_move = abs(p - entry_price)
                if path_length > 1e-12:
                    output[i] = (net_move) / path_length if path_length != 0 else np.nan
                else:
                    output[i] = 0.0

        return pd.Series(output, index=price.index, name=price.name)


@register_operator(name="state_episode_excursion_balance", canonical="state_episode_excursion_balance", backend="polars")
class StateEpisodeExcursionBalancePolarsNative(SeriesOperator):
    """Balance between favorable and adverse excursions"""
    metadata = OperatorMetadata(
        name="state_episode_excursion_balance",
        category="state",
        description="Balance: (MFE - MAE) / (MFE + MAE) where MFE/MAE are max favorable/adverse excursions",
        param_names=["price", "state"],
        return_type="series",
        tags=["state", "episode", "polars_native"],
    )

    def _calculate_series(self, price, state, **kwargs):
        """
        MFE = max favorable excursion (best profit in direction of state)
        MAE = max adverse excursion (worst drawdown against direction)
        Balance = (MFE - MAE) / (MFE + MAE)
        """
        df = pl.from_pandas(pd.concat([
            price.to_frame(),
            state.to_frame().rename(columns={state.name: "_state"})
        ], axis=1))

        price_vals = df[price.name].to_numpy()
        state_vals = df["_state"].to_numpy()
        output = np.full(len(price_vals), np.nan)

        entry_price = np.nan
        direction = 0
        mfe = 0.0
        mae = 0.0
        in_episode = False

        for i in range(len(price_vals)):
            s = state_vals[i]
            p = price_vals[i]

            episode_active = np.isfinite(s) and abs(s) > 1e-9

            if not episode_active or not np.isfinite(p):
                entry_price = np.nan
                direction = 0
                mfe = 0.0
                mae = 0.0
                in_episode = False
                output[i] = np.nan
                continue

            if not in_episode:
                entry_price = p
                direction = 1 if s > 0 else -1
                mfe = 0.0
                mae = 0.0
                in_episode = True
                output[i] = 0.0
            else:
                # Calculate signed excursion
                excursion = direction * (p - entry_price)
                mfe = max(mfe, excursion)
                mae = max(mae, -excursion)

                total = mfe + mae
                if total > 1e-12:
                    output[i] = ((mfe - mae)) / total if total != 0 else np.nan
                else:
                    output[i] = 0.0

        return pd.Series(output, index=price.index, name=price.name)


@register_operator(name="state_episode_mae", canonical="state_episode_mae", backend="polars")
class StateEpisodeMaePolarsNative(SeriesOperator):
    """Maximum adverse excursion in current episode"""
    metadata = OperatorMetadata(
        name="state_episode_mae",
        category="state",
        description="Maximum adverse excursion (worst drawdown against position direction) in episode",
        param_names=["price", "state"],
        return_type="series",
        tags=["state", "episode", "polars_native"],
    )

    def _calculate_series(self, price, state, **kwargs):
        """MAE = max adverse excursion against the state direction"""
        df = pl.from_pandas(pd.concat([
            price.to_frame(),
            state.to_frame().rename(columns={state.name: "_state"})
        ], axis=1))

        price_vals = df[price.name].to_numpy()
        state_vals = df["_state"].to_numpy()
        output = np.full(len(price_vals), np.nan)

        entry_price = np.nan
        direction = 0
        mae = 0.0
        in_episode = False

        for i in range(len(price_vals)):
            s = state_vals[i]
            p = price_vals[i]

            episode_active = np.isfinite(s) and abs(s) > 1e-9

            if not episode_active or not np.isfinite(p):
                entry_price = np.nan
                direction = 0
                mae = 0.0
                in_episode = False
                output[i] = np.nan
                continue

            if not in_episode:
                entry_price = p
                direction = 1 if s > 0 else -1
                mae = 0.0
                in_episode = True
                output[i] = 0.0
            else:
                excursion = direction * (p - entry_price)
                mae = max(mae, -excursion)
                output[i] = mae

        return pd.Series(output, index=price.index, name=price.name)


@register_operator(name="state_episode_mfe", canonical="state_episode_mfe", backend="polars")
class StateEpisodeMfePolarsNative(SeriesOperator):
    """Maximum favorable excursion in current episode"""
    metadata = OperatorMetadata(
        name="state_episode_mfe",
        category="state",
        description="Maximum favorable excursion (best profit in position direction) in episode",
        param_names=["price", "state"],
        return_type="series",
        tags=["state", "episode", "polars_native"],
    )

    def _calculate_series(self, price, state, **kwargs):
        """MFE = max favorable excursion in the state direction"""
        df = pl.from_pandas(pd.concat([
            price.to_frame(),
            state.to_frame().rename(columns={state.name: "_state"})
        ], axis=1))

        price_vals = df[price.name].to_numpy()
        state_vals = df["_state"].to_numpy()
        output = np.full(len(price_vals), np.nan)

        entry_price = np.nan
        direction = 0
        mfe = 0.0
        in_episode = False

        for i in range(len(price_vals)):
            s = state_vals[i]
            p = price_vals[i]

            episode_active = np.isfinite(s) and abs(s) > 1e-9

            if not episode_active or not np.isfinite(p):
                entry_price = np.nan
                direction = 0
                mfe = 0.0
                in_episode = False
                output[i] = np.nan
                continue

            if not in_episode:
                entry_price = p
                direction = 1 if s > 0 else -1
                mfe = 0.0
                in_episode = True
                output[i] = 0.0
            else:
                excursion = direction * (p - entry_price)
                mfe = max(mfe, excursion)
                output[i] = mfe

        return pd.Series(output, index=price.index, name=price.name)


@register_operator(name="state_episode_retrace_ratio", canonical="state_episode_retrace_ratio", backend="polars")
class StateEpisodeRetraceRatioPolarsNative(SeriesOperator):
    """Retracement ratio from peak to current position"""
    metadata = OperatorMetadata(
        name="state_episode_retrace_ratio",
        category="state",
        description="Retracement: (peak - current) / (peak - entry) in episode",
        param_names=["price", "state"],
        return_type="series",
        tags=["state", "episode", "polars_native"],
    )

    def _calculate_series(self, price, state, **kwargs):
        """
        Retrace ratio = (peak_excursion - current_excursion) / peak_excursion
        Measures how much of the peak gain has been given back
        """
        df = pl.from_pandas(pd.concat([
            price.to_frame(),
            state.to_frame().rename(columns={state.name: "_state"})
        ], axis=1))

        price_vals = df[price.name].to_numpy()
        state_vals = df["_state"].to_numpy()
        output = np.full(len(price_vals), np.nan)

        entry_price = np.nan
        direction = 0
        peak_excursion = 0.0
        in_episode = False

        for i in range(len(price_vals)):
            s = state_vals[i]
            p = price_vals[i]

            episode_active = np.isfinite(s) and abs(s) > 1e-9

            if not episode_active or not np.isfinite(p):
                entry_price = np.nan
                direction = 0
                peak_excursion = 0.0
                in_episode = False
                output[i] = np.nan
                continue

            if not in_episode:
                entry_price = p
                direction = 1 if s > 0 else -1
                peak_excursion = 0.0
                in_episode = True
                output[i] = 0.0
            else:
                current_excursion = direction * (p - entry_price)
                peak_excursion = max(peak_excursion, current_excursion)

                if peak_excursion > 1e-12:
                    retrace = ((peak_excursion - current_excursion)) / peak_excursion if peak_excursion != 0 else np.nan
                    output[i] = retrace
                else:
                    output[i] = 0.0

        return pd.Series(output, index=price.index, name=price.name)


# ============================================================================
# Turnover and Adjustment Operators
# ============================================================================

@register_operator(name="state_l1_turnover_prox", canonical="state_l1_turnover_prox", backend="polars")
class StateL1TurnoverProxPolarsNative(SeriesOperator):
    """L1-regularized state that minimizes turnover"""
    metadata = OperatorMetadata(
        name="state_l1_turnover_prox",
        category="state",
        description="State with L1 penalty on turnover: soft-threshold changes by lambda",
        param_names=["signal", "lambda_penalty"],
        return_type="series",
        tags=["state", "regularization", "polars_native"],
    )
    metadata.param_specs = {
        "lambda_penalty": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, signal, lambda_penalty=0.01, **kwargs):
        """
        Proximal operator: new_state = soft_threshold(signal - old_state, lambda)
        Penalizes turnover with L1 regularization
        """
        vals = signal.to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            output[0] = vals[0] if np.isfinite(vals[0]) else np.nan
            current_state = output[0]

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    # Soft threshold: shrink change by lambda
                    raw_change = vals[i] - current_state
                    if abs(raw_change) <= lambda_penalty:
                        # Change too small - stay at current state
                        output[i] = current_state
                    else:
                        # Apply soft threshold
                        shrunk_change = raw_change - np.sign(raw_change) * lambda_penalty
                        current_state = current_state + shrunk_change
                        output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_l2_partial_adjustment", canonical="state_l2_partial_adjustment", backend="polars")
class StateL2PartialAdjustmentPolarsNative(SeriesOperator):
    """Partial adjustment model with L2 penalty"""
    metadata = OperatorMetadata(
        name="state_l2_partial_adjustment",
        category="state",
        description="State adjusts partially toward target: new = old + alpha * (target - old)",
        param_names=["target", "adjustment_speed"],
        return_type="series",
        tags=["state", "adjustment", "polars_native"],
    )
    metadata.param_specs = {
        "adjustment_speed": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, target, adjustment_speed=0.5, **kwargs):
        """
        Partial adjustment: state[t] = state[t-1] + alpha * (target[t] - state[t-1])
        alpha = adjustment_speed (0 = no change, 1 = full adjustment)
        """
        vals = target.to_numpy()
        output = np.full(len(vals), np.nan)
        alpha = np.clip(adjustment_speed, 0.0, 1.0)

        if len(vals) > 0:
            output[0] = vals[0] if np.isfinite(vals[0]) else np.nan
            current_state = output[0]

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = current_state
                    continue

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    # Partial adjustment toward target
                    current_state = current_state + alpha * (vals[i] - current_state)
                    output[i] = current_state

        return pd.Series(output, index=target.index, name=target.name)


# ============================================================================
# Quantile and Rank Based State Machines
# ============================================================================

@register_operator(name="state_quantile_hysteresis", canonical="state_quantile_hysteresis", backend="polars")
class StateQuantileHysteresisPolarsNative(SeriesOperator):
    """Quantile-based hysteresis state machine"""
    metadata = OperatorMetadata(
        name="state_quantile_hysteresis",
        category="state",
        description="State machine using quantile thresholds: high/low bands with hysteresis",
        param_names=["signal", "high_quantile", "low_quantile", "window"],
        return_type="series",
        tags=["state", "hysteresis", "polars_native"],
    )
    metadata.param_specs = {
        "high_quantile": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.STATE_THRESHOLD),
        "low_quantile": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, signal, high_quantile=0.8, low_quantile=0.2, window=20, **kwargs):
        """
        State = +1 when signal > rolling high quantile
        State = -1 when signal < rolling low quantile
        State persists until opposite threshold crossed (hysteresis)
        """
        df = pl.from_pandas(signal.to_frame())
        col_name = signal.name

        result = (
            df.lazy()
            .with_columns([
                pl.col(col_name).rolling_quantile(
                    quantile=high_quantile,
                    window_size=window,
                    min_periods=1
                ).alias("_hi"),
                pl.col(col_name).rolling_quantile(
                    quantile=low_quantile,
                    window_size=window,
                    min_periods=1
                ).alias("_lo"),
            ])
            .collect()
        )

        vals = result[col_name].to_numpy()
        hi_vals = result["_hi"].to_numpy()
        lo_vals = result["_lo"].to_numpy()
        output = np.zeros(len(vals))

        current_state = 0
        for i in range(len(vals)):
            if not np.isfinite(vals[i]):
                output[i] = 0
                current_state = 0
                continue

            if vals[i] > hi_vals[i]:
                current_state = 1
            elif vals[i] < lo_vals[i]:
                current_state = -1
            # else: maintain current_state (hysteresis)

            output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_rank_deadband", canonical="state_rank_deadband", backend="polars")
class StateRankDeadbandPolarsNative(SeriesOperator):
    """Deadband filter operating on rolling rank"""
    metadata = OperatorMetadata(
        name="state_rank_deadband",
        category="state",
        description="Deadband applied to rolling rank percentile instead of raw value",
        param_names=["signal", "threshold", "window"],
        return_type="series",
        tags=["state", "rank", "polars_native"],
    )
    metadata.param_specs = {
        "threshold": ParamSpec(dtype=float, min=0.0, max=1.0, param_role=ParamRole.STATE_THRESHOLD),
        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
    }

    def _calculate_series(self, signal, threshold=0.1, window=20, **kwargs):
        """
        Convert to rolling rank percentile, then apply deadband
        State only changes when rank moves beyond threshold
        """
        df = pl.from_pandas(signal.to_frame())
        col_name = signal.name

        # Calculate rolling rank percentile
        result = (
            df.lazy()
            .with_columns([
                pl.col(col_name).rolling_map(
                    function=lambda s: pd.Series(s).rank(pct=True).iloc[-1] if len(s) > 0 else np.nan,
                    window_size=window,
                    min_periods=1
                ).alias("_rank_pct")
            ])
            .collect()
        )

        rank_vals = result["_rank_pct"].to_numpy()
        output = np.full(len(rank_vals), np.nan)

        if len(rank_vals) > 0:
            output[0] = rank_vals[0]
            current_state = rank_vals[0] if np.isfinite(rank_vals[0]) else np.nan

            for i in range(1, len(rank_vals)):
                if not np.isfinite(rank_vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = rank_vals[i]
                    output[i] = current_state
                elif abs(rank_vals[i] - current_state) > threshold:
                    current_state = rank_vals[i]
                    output[i] = current_state
                else:
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


# ============================================================================
# Since-Event Tracking Operators
# ============================================================================

@register_operator(name="state_since_mean", canonical="state_since_mean", backend="polars")
class StateSinceMeanPolarsNative(SeriesOperator):
    """Mean value since last reset event"""
    metadata = OperatorMetadata(
        name="state_since_mean",
        category="state",
        description="Running mean of values since last reset event",
        param_names=["value", "reset_flag"],
        return_type="series",
        tags=["state", "since", "polars_native"],
    )

    def _calculate_series(self, value, reset_flag, **kwargs):
        """
        Compute running mean since last reset
        Reset when reset_flag != 0
        """
        df = pl.from_pandas(pd.concat([
            value.to_frame(),
            reset_flag.to_frame().rename(columns={reset_flag.name: "_reset"})
        ], axis=1))

        vals = df[value.name].to_numpy()
        reset_vals = df["_reset"].to_numpy()
        output = np.full(len(vals), np.nan)

        running_sum = 0.0
        running_count = 0

        for i in range(len(vals)):
            # Check for reset
            if np.isfinite(reset_vals[i]) and abs(reset_vals[i]) > 1e-9:
                running_sum = 0.0
                running_count = 0

            # Accumulate if finite
            if np.isfinite(vals[i]):
                running_sum += vals[i]
                running_count += 1

                if running_count > 0:
                    output[i] = (running_sum) / running_count if running_count != 0 else np.nan

        return pd.Series(output, index=value.index, name=value.name)


@register_operator(name="state_since_sum", canonical="state_since_sum", backend="polars")
class StateSinceSumPolarsNative(SeriesOperator):
    """Cumulative sum since last reset event"""
    metadata = OperatorMetadata(
        name="state_since_sum",
        category="state",
        description="Running sum of values since last reset event",
        param_names=["value", "reset_flag"],
        return_type="series",
        tags=["state", "since", "polars_native"],
    )

    def _calculate_series(self, value, reset_flag, **kwargs):
        """
        Compute running sum since last reset
        Reset when reset_flag != 0
        """
        df = pl.from_pandas(pd.concat([
            value.to_frame(),
            reset_flag.to_frame().rename(columns={reset_flag.name: "_reset"})
        ], axis=1))

        vals = df[value.name].to_numpy()
        reset_vals = df["_reset"].to_numpy()
        output = np.full(len(vals), np.nan)

        running_sum = 0.0

        for i in range(len(vals)):
            # Check for reset
            if np.isfinite(reset_vals[i]) and abs(reset_vals[i]) > 1e-9:
                running_sum = 0.0

            # Accumulate if finite
            if np.isfinite(vals[i]):
                running_sum += vals[i]
                output[i] = running_sum

        return pd.Series(output, index=value.index, name=value.name)


@register_operator(name="state_since_trend_tstat", canonical="state_since_trend_tstat", backend="polars")
class StateSinceTrendTstatPolarsNative(SeriesOperator):
    """T-statistic of linear trend since last reset"""
    metadata = OperatorMetadata(
        name="state_since_trend_tstat",
        category="state",
        description="T-statistic of linear trend in values since last reset event",
        param_names=["value", "reset_flag"],
        return_type="series",
        tags=["state", "since", "trend", "polars_native"],
    )

    def _calculate_series(self, value, reset_flag, **kwargs):
        """
        Compute t-statistic of linear trend since last reset
        t-stat = slope / std_error
        """
        df = pl.from_pandas(pd.concat([
            value.to_frame(),
            reset_flag.to_frame().rename(columns={reset_flag.name: "_reset"})
        ], axis=1))

        vals = df[value.name].to_numpy()
        reset_vals = df["_reset"].to_numpy()
        output = np.full(len(vals), np.nan)

        segment_vals = []

        for i in range(len(vals)):
            # Check for reset
            if np.isfinite(reset_vals[i]) and abs(reset_vals[i]) > 1e-9:
                segment_vals = []

            # Accumulate if finite
            if np.isfinite(vals[i]):
                segment_vals.append(vals[i])

                if len(segment_vals) >= 3:
                    # Compute linear regression t-statistic
                    n = len(segment_vals)
                    x = np.arange(n)
                    y = np.array(segment_vals)

                    x_mean = x.mean()
                    y_mean = y.mean()

                    numerator = np.sum((x - x_mean) * (y - y_mean))
                    denominator = np.sum((x - x_mean) ** 2)

                    if denominator > 1e-12:
                        slope = (numerator) / denominator if denominator != 0 else np.nan
                        y_pred = slope * (x - x_mean) + y_mean
                        residuals = y - y_pred
                        mse = np.where((n - 2) != 0, (np.sum(residuals ** 2)) / ((n - 2)), np.nan)
                        se_slope = np.where(denominator) != 0, (np.sqrt(mse) / (denominator)), np.nan)

                        if se_slope > 1e-12:
                            t_stat = (slope) / se_slope if se_slope != 0 else np.nan
                            output[i] = t_stat

        return pd.Series(output, index=value.index, name=value.name)


# ============================================================================
# Slew Limit and Uncertainty Deadband
# ============================================================================

@register_operator(name="state_slew_limit", canonical="state_slew_limit", backend="polars")
class StateSlewLimitPolarsNative(SeriesOperator):
    """Basic slew rate limiter"""
    metadata = OperatorMetadata(
        name="state_slew_limit",
        category="state",
        description="Limit maximum rate of change per step",
        param_names=["signal", "max_rate"],
        return_type="series",
        tags=["state", "filter", "polars_native"],
    )
    metadata.param_specs = {
        "max_rate": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, signal, max_rate=0.1, **kwargs):
        """
        State can only change by max_rate per step
        """
        vals = signal.to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            output[0] = vals[0]
            current_state = vals[0] if np.isfinite(vals[0]) else np.nan

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = np.nan
                    continue

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    delta = vals[i] - current_state
                    clamped_delta = np.clip(delta, -max_rate, max_rate)
                    current_state = current_state + clamped_delta
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)


@register_operator(name="state_uncertainty_deadband", canonical="state_uncertainty_deadband", backend="polars")
class StateUncertaintyDeadbandPolarsNative(SeriesOperator):
    """Deadband filter weighted by uncertainty estimate"""
    metadata = OperatorMetadata(
        name="state_uncertainty_deadband",
        category="state",
        description="Deadband threshold scales with uncertainty: higher uncertainty = wider band",
        param_names=["signal", "uncertainty", "base_threshold"],
        return_type="series",
        tags=["state", "filter", "uncertainty", "polars_native"],
    )
    metadata.param_specs = {
        "base_threshold": ParamSpec(dtype=float, min=0.0, param_role=ParamRole.STATE_THRESHOLD),
    }

    def _calculate_series(self, signal, uncertainty, base_threshold=0.02, **kwargs):
        """
        Effective threshold = base_threshold * (1 + uncertainty)
        Higher uncertainty requires larger moves to change state
        """
        df = pl.from_pandas(pd.concat([
            signal.to_frame(),
            uncertainty.to_frame().rename(columns={uncertainty.name: "_unc"})
        ], axis=1))

        vals = df[signal.name].to_numpy()
        unc_vals = df["_unc"].to_numpy()
        output = np.full(len(vals), np.nan)

        if len(vals) > 0:
            output[0] = vals[0]
            current_state = vals[0] if np.isfinite(vals[0]) else np.nan

            for i in range(1, len(vals)):
                if not np.isfinite(vals[i]):
                    output[i] = np.nan
                    continue

                # Calculate uncertainty-weighted threshold
                unc = unc_vals[i] if np.isfinite(unc_vals[i]) else 1.0
                unc = max(0.0, unc)  # Ensure non-negative
                effective_threshold = base_threshold * (1.0 + unc)

                if not np.isfinite(current_state):
                    current_state = vals[i]
                    output[i] = current_state
                elif abs(vals[i] - current_state) > effective_threshold:
                    current_state = vals[i]
                    output[i] = current_state
                else:
                    output[i] = current_state

        return pd.Series(output, index=signal.index, name=signal.name)
