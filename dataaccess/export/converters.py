# -*- coding: utf-8 -*-
"""Convert between pandas, polars, and arrow formats.

Provides zero-copy conversions where possible and handles type mapping
for contract data structures.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

__all__ = [
    "DataFrameConverter",
    "to_pandas",
    "to_polars",
    "to_arrow",
    "convert_batch",
]


class DataFrameConverter:
    """Convert contract data between pandas, polars, and arrow formats."""

    @staticmethod
    def to_pandas(data: Any, *, preserve_index: bool = False) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Args:
            data: Input data (DataFrame, Arrow Table, Polars DataFrame, dict, list)
            preserve_index: Preserve index when converting

        Returns:
            pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, pa.Table):
            return data.to_pandas(self_destruct=False)

        if POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            return data.to_pandas()

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, (list, tuple)):
            if not data:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            return pd.DataFrame({"values": data})

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({"values": data})
            return pd.DataFrame(data)

        raise TypeError(f"Cannot convert {type(data).__name__} to pandas DataFrame")

    @staticmethod
    def to_polars(data: Any) -> pl.DataFrame:
        """Convert to polars DataFrame.

        Args:
            data: Input data (DataFrame, Arrow Table, dict, list)

        Returns:
            polars DataFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("polars is not installed")

        if isinstance(data, pl.DataFrame):
            return data

        if isinstance(data, pa.Table):
            return pl.from_arrow(data)

        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)

        if isinstance(data, dict):
            return pl.DataFrame([data])

        if isinstance(data, (list, tuple)):
            if not data:
                return pl.DataFrame()
            if isinstance(data[0], dict):
                return pl.DataFrame(data)
            return pl.DataFrame({"values": data})

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pl.DataFrame({"values": data})
            return pl.DataFrame(data)

        raise TypeError(f"Cannot convert {type(data).__name__} to polars DataFrame")

    @staticmethod
    def to_arrow(data: Any, *, preserve_index: bool = False) -> pa.Table:
        """Convert to arrow Table.

        Args:
            data: Input data (DataFrame, dict, list)
            preserve_index: Preserve pandas index

        Returns:
            pyarrow Table
        """
        if isinstance(data, pa.Table):
            return data

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data, preserve_index=preserve_index)

        if POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            return data.to_arrow()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
            return pa.Table.from_pandas(df, preserve_index=False)

        if isinstance(data, (list, tuple)):
            if not data:
                return pa.Table.from_pandas(pd.DataFrame())
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
                return pa.Table.from_pandas(df, preserve_index=False)
            df = pd.DataFrame({"values": data})
            return pa.Table.from_pandas(df, preserve_index=False)

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({"values": data})
            else:
                df = pd.DataFrame(data)
            return pa.Table.from_pandas(df, preserve_index=False)

        raise TypeError(f"Cannot convert {type(data).__name__} to arrow Table")

    @staticmethod
    def infer_schema(data: Any) -> pa.Schema:
        """Infer arrow schema from data.

        Args:
            data: Input data

        Returns:
            pyarrow Schema
        """
        table = DataFrameConverter.to_arrow(data)
        return table.schema

    @staticmethod
    def validate_schema_compatible(
        source_schema: pa.Schema,
        target_schema: pa.Schema,
        *,
        strict: bool = False,
    ) -> tuple[bool, list[str]]:
        """Check if two schemas are compatible.

        Args:
            source_schema: Source schema
            target_schema: Target schema
            strict: Require exact match (field order and types)

        Returns:
            (compatible, issues) tuple
        """
        issues = []

        if strict:
            if source_schema != target_schema:
                issues.append("Schemas do not match exactly")
                for i, (sf, tf) in enumerate(zip(source_schema, target_schema)):
                    if sf.name != tf.name:
                        issues.append(f"Field {i}: name mismatch {sf.name} != {tf.name}")
                    if sf.type != tf.type:
                        issues.append(f"Field {sf.name}: type mismatch {sf.type} != {tf.type}")
            return (not issues, issues)

        target_fields = {f.name: f.type for f in target_schema}

        for field in source_schema:
            if field.name not in target_fields:
                issues.append(f"Field {field.name} not in target schema")
            elif not pa.types.is_compatible(field.type, target_fields[field.name]):
                issues.append(
                    f"Field {field.name}: incompatible types "
                    f"{field.type} vs {target_fields[field.name]}"
                )

        return (not issues, issues)


def to_pandas(data: Any, *, preserve_index: bool = False) -> pd.DataFrame:
    """Convert data to pandas DataFrame.

    Args:
        data: Input data
        preserve_index: Preserve index when converting

    Returns:
        pandas DataFrame
    """
    return DataFrameConverter.to_pandas(data, preserve_index=preserve_index)


def to_polars(data: Any) -> pl.DataFrame:
    """Convert data to polars DataFrame.

    Args:
        data: Input data

    Returns:
        polars DataFrame
    """
    return DataFrameConverter.to_polars(data)


def to_arrow(data: Any, *, preserve_index: bool = False) -> pa.Table:
    """Convert data to arrow Table.

    Args:
        data: Input data
        preserve_index: Preserve pandas index

    Returns:
        pyarrow Table
    """
    return DataFrameConverter.to_arrow(data, preserve_index=preserve_index)


def convert_batch(
    data_list: Sequence[Any],
    target_format: str = "pandas",
    *,
    preserve_index: bool = False,
) -> list[pd.DataFrame | pa.Table]:
    """Convert multiple data objects to target format.

    Args:
        data_list: Sequence of data objects
        target_format: Target format (pandas, polars, arrow)
        preserve_index: Preserve index when converting

    Returns:
        List of converted objects
    """
    if target_format not in {"pandas", "polars", "arrow"}:
        raise ValueError(f"Unsupported target format: {target_format}")

    results = []

    for data in data_list:
        if target_format == "pandas":
            results.append(to_pandas(data, preserve_index=preserve_index))
        elif target_format == "polars":
            results.append(to_polars(data))
        elif target_format == "arrow":
            results.append(to_arrow(data, preserve_index=preserve_index))

    return results
