"""
Pandas adapter for quant_evaluator (reference/debug only).

This adapter provides explicit reference/debug conversions between pandas
DataFrames and QE contracts. This is NOT a production fallback and should
only be used for reference implementations, testing, and debugging.

Core quant_evaluator modules MUST NOT import this module.
Production paths MUST NOT use pandas as an implicit fallback.

When to use:
- Reference implementations of metrics
- Debug/validation workflows
- Small-scale testing with manual data
- Prototype evaluation scripts

When NOT to use:
- Production factor evaluation pipelines
- Large-scale batch processing
- As a fallback when DA/FE unavailable
"""

from typing import Optional, List, Tuple, Any, Dict
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


class PandasAdapter:
    """
    Explicit pandas adapter for reference/debug workflows only.

    This adapter is NOT an implicit production fallback. All conversions
    are explicit and documented as reference-only.

    Design principles:
    - Explicit reference/debug only; never implicit production
    - Lazy import; core QE never imports this
    - Simple conversions; no complex inference logic
    - Fail-closed on ambiguous inputs

    Usage (reference/debug only):
        adapter = PandasAdapter()
        batch = adapter.dataframe_to_factor_batch(
            df,
            factor_cols=["momentum", "value"],
            time_col="date",
            asset_col="code"
        )
    """

    def __init__(self):
        """
        Initialize adapter and verify pandas is available.

        Raises:
            OptionalDependencyMissing: If pandas not installed
        """
        try:
            import pandas as pd
            self._pd = pd
        except ImportError as e:
            raise OptionalDependencyMissing(
                "Pandas is not installed. "
                "Install pandas to use this reference/debug adapter."
            ) from e

    def dataframe_to_factor_batch(
        self,
        df: Any,  # pandas.DataFrame
        factor_cols: List[str],
        time_col: str = "date",
        asset_col: str = "code",
        validity_col: Optional[str] = None,
        layout: str = "wide",
        context_refs: Optional[Dict[str, Any]] = None,
    ) -> FactorBatch:
        """
        Convert pandas DataFrame to FactorBatch (reference/debug only).

        Args:
            df: pandas DataFrame with columns [time_col, asset_col, *factor_cols]
            factor_cols: Factor column names to extract
            time_col: Time axis column name (default "date")
            asset_col: Asset axis column name (default "code")
            validity_col: Optional validity mask column
            layout: Data layout ("wide" only currently supported)
            context_refs: Optional context metadata

        Returns:
            FactorBatch with explicit axes and values

        Raises:
            ValueError: If required columns missing, shape invalid, or data ambiguous
            OptionalDependencyMissing: If pandas not available

        Notes (reference/debug only):
            - Assumes df has rows = (time × asset), columns = factors
            - Pivots long format to (time, asset, factor) array
            - Time and asset axes extracted and sorted deterministically
            - Missing values remain as NaN in values array
            - Validity derived from finite values if validity_col not provided
            - No inference of timing, universe, or market context
            - Fails on duplicate (time, asset) pairs
        """
        pd = self._pd

        # Validate required columns
        required_cols = [time_col, asset_col] + factor_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for duplicates (fail-closed)
        duplicates = df.duplicated(subset=[time_col, asset_col])
        if duplicates.any():
            raise ValueError(
                f"Duplicate (time, asset) pairs found. "
                f"Reference adapter requires unique index."
            )

        # Extract and sort axes
        time_values = df[time_col].unique()
        time_values = np.sort(time_values)
        asset_values = df[asset_col].unique()
        asset_values = np.sort(asset_values)

        time_axis = AxisRef(
            name="time",
            dtype=str(time_values.dtype),
            size=len(time_values),
            values=time_values,
        )
        asset_axis = AxisRef(
            name="asset",
            dtype=str(asset_values.dtype),
            size=len(asset_values),
            values=asset_values,
        )

        # Build (time, asset, factor) array
        num_times = len(time_values)
        num_assets = len(asset_values)
        num_factors = len(factor_cols)

        values = np.full((num_times, num_assets, num_factors), np.nan, dtype=np.float64)

        # Create index mappings
        time_to_idx = {t: i for i, t in enumerate(time_values)}
        asset_to_idx = {a: i for i, a in enumerate(asset_values)}

        # Fill values array
        for _, row in df.iterrows():
            t_idx = time_to_idx[row[time_col]]
            a_idx = asset_to_idx[row[asset_col]]
            for f_idx, f_col in enumerate(factor_cols):
                values[t_idx, a_idx, f_idx] = row[f_col]

        # Extract or derive validity
        validity = None
        if validity_col is not None:
            if validity_col not in df.columns:
                raise ValueError(f"Validity column '{validity_col}' not found")
            validity = np.full((num_times, num_assets, num_factors), False, dtype=bool)
            for _, row in df.iterrows():
                t_idx = time_to_idx[row[time_col]]
                a_idx = asset_to_idx[row[asset_col]]
                for f_idx in range(num_factors):
                    validity[t_idx, a_idx, f_idx] = bool(row[validity_col])

        # Build factor IDs (simple names for reference)
        factor_ids = tuple(factor_cols)

        return FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            validity=validity,
            layout=layout,
            dtype="float64",
            context_refs=context_refs or {},
            value_hash=None,
        )

    def factor_batch_to_dataframe(
        self,
        batch: FactorBatch,
        time_col: str = "date",
        asset_col: str = "code",
        include_validity: bool = False,
    ) -> Any:  # pandas.DataFrame
        """
        Convert FactorBatch to pandas DataFrame (reference/debug only).

        Args:
            batch: FactorBatch to convert
            time_col: Time column name in output
            asset_col: Asset column name in output
            include_validity: Whether to include validity column

        Returns:
            pandas DataFrame in long format with columns:
                [time_col, asset_col, factor_0, factor_1, ..., validity (optional)]

        Raises:
            ValueError: If batch layout not supported
            OptionalDependencyMissing: If pandas not available

        Notes (reference/debug only):
            - Output is long format: one row per (time, asset)
            - Factor values are separate columns
            - NaN values preserved as-is
            - Validity column added if requested
            - Time/asset axes must have values; fails otherwise
        """
        pd = self._pd

        if batch.time_axis.values is None or batch.asset_axis.values is None:
            raise ValueError(
                "FactorBatch must have explicit time/asset axis values for DataFrame conversion"
            )

        # Build long format DataFrame
        records = []
        for t_idx, t_val in enumerate(batch.time_axis.values):
            for a_idx, a_val in enumerate(batch.asset_axis.values):
                record = {
                    time_col: t_val,
                    asset_col: a_val,
                }
                for f_idx, f_id in enumerate(batch.factor_ids):
                    record[f_id] = batch.values[t_idx, a_idx, f_idx]

                if include_validity and batch.validity is not None:
                    record["validity"] = batch.validity[t_idx, a_idx, 0]  # Use first factor's validity

                records.append(record)

        return pd.DataFrame(records)

    def dataframe_to_label_bundle(
        self,
        df: Any,  # pandas.DataFrame
        target_id: str,
        horizon: int,
        value_col: str = "label_value",
        decision_time_col: str = "decision_time",
        label_start_col: str = "label_start",
        label_end_col: str = "label_end",
        execution_delay: int = 0,
        validity_col: Optional[str] = None,
        calendar_ref: Optional[str] = None,
    ) -> LabelBundle:
        """
        Convert pandas DataFrame to LabelBundle (reference/debug only).

        Args:
            df: pandas DataFrame with timing columns and label values
            target_id: Label identifier (e.g., "ret_5d_vwap")
            horizon: Forward horizon in calendar units
            value_col: Column name for label values
            decision_time_col: Column name for decision timestamps
            label_start_col: Column name for label window start
            label_end_col: Column name for label window end
            execution_delay: Execution delay after decision (default 0)
            validity_col: Optional validity mask column
            calendar_ref: Optional calendar reference ID

        Returns:
            LabelBundle with explicit timing vectors

        Raises:
            ValueError: If required columns missing or timing invalid
            OptionalDependencyMissing: If pandas not available

        Notes (reference/debug only):
            - All timing columns must be present and non-empty
            - Values and timing must align row-wise
            - Timing order is not validated (caller responsibility)
            - Validity derived from finite values if not provided
        """
        pd = self._pd

        # Validate required columns
        required_cols = [value_col, decision_time_col, label_start_col, label_end_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for LabelBundle: {missing_cols}")

        # Extract arrays
        values = df[value_col].values
        decision_time = tuple(df[decision_time_col].values)
        label_start_time = tuple(df[label_start_col].values)
        label_end_time = tuple(df[label_end_col].values)

        # Extract or derive validity
        validity = None
        if validity_col is not None:
            if validity_col not in df.columns:
                raise ValueError(f"Validity column '{validity_col}' not found")
            validity = df[validity_col].values.astype(bool)

        # Build execution_time (decision_time + execution_delay)
        # For reference implementation, assume execution_time column exists or equals decision_time
        if "execution_time" in df.columns:
            execution_time = tuple(df["execution_time"].values)
        else:
            execution_time = decision_time  # Simplified for reference

        return LabelBundle(
            target_id=target_id,
            values=values,
            horizon=horizon,
            execution_delay=execution_delay,
            decision_time=decision_time,
            execution_time=execution_time,
            label_start_time=label_start_time,
            label_end_time=label_end_time,
            validity=validity,
            source_ref=None,
            calendar_ref=calendar_ref,
            metadata={},
        )

    def label_bundle_to_dataframe(
        self,
        bundle: LabelBundle,
        include_all_timing: bool = True,
    ) -> Any:  # pandas.DataFrame
        """
        Convert LabelBundle to pandas DataFrame (reference/debug only).

        Args:
            bundle: LabelBundle to convert
            include_all_timing: Whether to include all timing columns

        Returns:
            pandas DataFrame with columns:
                [decision_time, label_value, (optional timing columns), validity]

        Raises:
            OptionalDependencyMissing: If pandas not available

        Notes (reference/debug only):
            - Output has one row per observation
            - All timing vectors included if requested
            - Validity column included if present in bundle
        """
        pd = self._pd

        data = {
            "decision_time": bundle.decision_time,
            "label_value": bundle.values,
        }

        if include_all_timing:
            data["execution_time"] = bundle.execution_time
            data["label_start_time"] = bundle.label_start_time
            data["label_end_time"] = bundle.label_end_time

        if bundle.validity is not None:
            data["validity"] = bundle.validity

        data["horizon"] = [bundle.horizon] * len(bundle.decision_time)
        data["execution_delay"] = [bundle.execution_delay] * len(bundle.decision_time)

        return pd.DataFrame(data)
