# -*- coding: utf-8 -*-
"""Contract serialization to JSON, Parquet, and Feather formats.

Supports all major FactorEngine contracts:
- ModelOperatorSpec, ModelExecutionClass, TimingKind
- RichModelTiming, SampleAdequacyContract, LabelContract
- DecisionClock, PredictionOutputContract, PredictionBatch
- RegimeMetadata, FitFingerprint, ParameterSearchPolicy
- ParamRole enumeration values
"""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

__all__ = [
    "ContractSerializer",
    "serialize_to_json",
    "serialize_to_parquet",
    "serialize_to_feather",
    "serialize_batch",
]


class ContractSerializer:
    """Serialize FactorEngine contracts to various formats."""

    @staticmethod
    def to_dict(obj: Any, *, include_type: bool = True) -> dict[str, Any]:
        """Convert a contract object to a serializable dictionary.

        Args:
            obj: Contract object (dataclass, enum, or dict)
            include_type: Include __type__ metadata for reconstruction

        Returns:
            Dictionary with all fields serialized
        """
        if obj is None:
            return {"__type__": "none", "value": None} if include_type else None

        if isinstance(obj, dict):
            return {k: ContractSerializer.to_dict(v, include_type=include_type)
                    for k, v in obj.items()}

        if isinstance(obj, enum.Enum):
            result = {"value": obj.value}
            if include_type:
                result["__type__"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            return result

        if is_dataclass(obj):
            data = asdict(obj)
            result = {k: ContractSerializer._serialize_value(v, include_type=include_type)
                     for k, v in data.items()}
            if include_type:
                result["__type__"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            return result

        if isinstance(obj, (list, tuple)):
            return [ContractSerializer.to_dict(item, include_type=include_type) for item in obj]

        return ContractSerializer._serialize_value(obj, include_type=include_type)

    @staticmethod
    def _serialize_value(value: Any, *, include_type: bool = True) -> Any:
        """Serialize a single value to a JSON-compatible type."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        if isinstance(value, enum.Enum):
            result = {"value": value.value}
            if include_type:
                result["__type__"] = f"{value.__class__.__module__}.{value.__class__.__name__}"
            return result

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        if isinstance(value, dict):
            return {k: ContractSerializer._serialize_value(v, include_type=include_type)
                   for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [ContractSerializer._serialize_value(item, include_type=include_type)
                   for item in value]

        if is_dataclass(value):
            return ContractSerializer.to_dict(value, include_type=include_type)

        return str(value)

    @staticmethod
    def to_json(obj: Any, *, indent: int | None = 2, include_type: bool = True) -> str:
        """Serialize contract to JSON string.

        Args:
            obj: Contract object to serialize
            indent: JSON indentation (None for compact)
            include_type: Include type metadata for reconstruction

        Returns:
            JSON string representation
        """
        data = ContractSerializer.to_dict(obj, include_type=include_type)
        return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def to_parquet(
        obj: Any | Sequence[Any],
        path: str | Path,
        *,
        include_type: bool = True,
        compression: str = "snappy",
    ) -> None:
        """Serialize contract(s) to Parquet format.

        Args:
            obj: Single contract or sequence of contracts
            path: Output file path
            include_type: Include type metadata
            compression: Compression codec (snappy, gzip, zstd, lz4)
        """
        if not isinstance(obj, (list, tuple)):
            obj = [obj]

        records = [ContractSerializer.to_dict(item, include_type=include_type) for item in obj]
        df = pd.json_normalize(records, sep="__")

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression=compression)

    @staticmethod
    def to_feather(
        obj: Any | Sequence[Any],
        path: str | Path,
        *,
        include_type: bool = True,
        compression: str = "lz4",
    ) -> None:
        """Serialize contract(s) to Feather format.

        Args:
            obj: Single contract or sequence of contracts
            path: Output file path
            include_type: Include type metadata
            compression: Compression codec (lz4, zstd, uncompressed)
        """
        if not isinstance(obj, (list, tuple)):
            obj = [obj]

        records = [ContractSerializer.to_dict(item, include_type=include_type) for item in obj]
        df = pd.json_normalize(records, sep="__")

        table = pa.Table.from_pandas(df, preserve_index=False)
        feather.write_feather(table, path, compression=compression)


def serialize_to_json(
    obj: Any,
    path: str | Path | None = None,
    *,
    indent: int | None = 2,
    include_type: bool = True,
) -> str:
    """Serialize contract to JSON.

    Args:
        obj: Contract object to serialize
        path: Optional file path to write (if None, returns string)
        indent: JSON indentation
        include_type: Include type metadata

    Returns:
        JSON string (also written to file if path provided)
    """
    json_str = ContractSerializer.to_json(obj, indent=indent, include_type=include_type)

    if path is not None:
        Path(path).write_text(json_str, encoding="utf-8")

    return json_str


def serialize_to_parquet(
    obj: Any | Sequence[Any],
    path: str | Path,
    *,
    include_type: bool = True,
    compression: str = "snappy",
) -> None:
    """Serialize contract(s) to Parquet file.

    Args:
        obj: Contract object(s) to serialize
        path: Output file path
        include_type: Include type metadata
        compression: Compression codec
    """
    ContractSerializer.to_parquet(obj, path, include_type=include_type, compression=compression)


def serialize_to_feather(
    obj: Any | Sequence[Any],
    path: str | Path,
    *,
    include_type: bool = True,
    compression: str = "lz4",
) -> None:
    """Serialize contract(s) to Feather file.

    Args:
        obj: Contract object(s) to serialize
        path: Output file path
        include_type: Include type metadata
        compression: Compression codec
    """
    ContractSerializer.to_feather(obj, path, include_type=include_type, compression=compression)


def serialize_batch(
    objects: Sequence[Any],
    directory: str | Path,
    *,
    format: str = "parquet",
    prefix: str = "contract",
    include_type: bool = True,
    compression: str | None = None,
) -> list[Path]:
    """Serialize multiple contracts to a directory.

    Args:
        objects: Sequence of contract objects
        directory: Output directory path
        format: Output format (json, parquet, feather)
        prefix: Filename prefix
        include_type: Include type metadata
        compression: Compression codec (format-specific default if None)

    Returns:
        List of created file paths
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if format not in {"json", "parquet", "feather"}:
        raise ValueError(f"Unsupported format: {format}")

    paths = []

    if format == "json":
        for idx, obj in enumerate(objects):
            path = directory / f"{prefix}_{idx:04d}.json"
            serialize_to_json(obj, path, include_type=include_type)
            paths.append(path)

    elif format == "parquet":
        comp = compression or "snappy"
        path = directory / f"{prefix}_batch.parquet"
        serialize_to_parquet(objects, path, include_type=include_type, compression=comp)
        paths.append(path)

    elif format == "feather":
        comp = compression or "lz4"
        path = directory / f"{prefix}_batch.feather"
        serialize_to_feather(objects, path, include_type=include_type, compression=comp)
        paths.append(path)

    return paths
