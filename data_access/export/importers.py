# -*- coding: utf-8 -*-
"""Import and deserialize contracts from external formats.

Supports importing from JSON, Parquet, and Feather formats with automatic
type reconstruction and validation.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq

__all__ = [
    "ContractImporter",
    "import_from_json",
    "import_from_parquet",
    "import_from_feather",
    "import_batch",
]

T = TypeVar("T")


class ContractImporter:
    """Import and reconstruct FactorEngine contracts from serialized formats."""

    @staticmethod
    def from_dict(data: dict[str, Any], *, target_type: Type[T] | None = None) -> T | dict:
        """Reconstruct contract object from dictionary.

        Args:
            data: Serialized contract dictionary
            target_type: Expected type (if None, uses __type__ metadata)

        Returns:
            Reconstructed contract object
        """
        if data is None:
            return None

        if not isinstance(data, dict):
            return data

        type_name = data.get("__type__")

        if type_name and target_type is None:
            target_type = ContractImporter._resolve_type(type_name)

        if target_type is None:
            return {k: ContractImporter.from_dict(v) if isinstance(v, dict) else v
                   for k, v in data.items() if k != "__type__"}

        if hasattr(target_type, "__bases__") and any(
            base.__name__ == "Enum" for base in target_type.__bases__
        ):
            value = data.get("value", data)
            return target_type(value)

        if is_dataclass(target_type):
            field_data = {k: v for k, v in data.items() if k != "__type__"}
            field_types = {f.name: f.type for f in fields(target_type)}

            reconstructed = {}
            for field_name, value in field_data.items():
                field_type = field_types.get(field_name)

                if isinstance(value, dict) and "__type__" in value:
                    reconstructed[field_name] = ContractImporter.from_dict(
                        value, target_type=None
                    )
                elif isinstance(value, dict) and field_type:
                    reconstructed[field_name] = ContractImporter.from_dict(
                        value, target_type=field_type
                    )
                elif isinstance(value, list) and value:
                    reconstructed[field_name] = [
                        ContractImporter.from_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                elif isinstance(value, str) and field_type in (datetime,):
                    try:
                        reconstructed[field_name] = datetime.fromisoformat(value)
                    except (ValueError, AttributeError):
                        reconstructed[field_name] = value
                else:
                    reconstructed[field_name] = value

            return target_type(**reconstructed)

        return data

    @staticmethod
    def _resolve_type(type_name: str) -> Type | None:
        """Resolve type from fully qualified name."""
        if type_name == "none":
            return None

        try:
            parts = type_name.rsplit(".", 1)
            if len(parts) == 2:
                module_name, class_name = parts
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError):
            pass

        return None

    @staticmethod
    def from_json(json_str: str, *, target_type: Type[T] | None = None) -> T | dict:
        """Deserialize contract from JSON string.

        Args:
            json_str: JSON string representation
            target_type: Expected contract type

        Returns:
            Reconstructed contract object
        """
        data = json.loads(json_str)
        return ContractImporter.from_dict(data, target_type=target_type)

    @staticmethod
    def from_parquet(path: str | Path, *, target_type: Type[T] | None = None) -> list[T | dict]:
        """Import contract(s) from Parquet file.

        Args:
            path: Input file path
            target_type: Expected contract type

        Returns:
            List of reconstructed contracts
        """
        table = pq.read_table(path)
        df = table.to_pandas()

        records = []
        for _, row in df.iterrows():
            record = ContractImporter._denormalize_row(row)
            records.append(ContractImporter.from_dict(record, target_type=target_type))

        return records

    @staticmethod
    def from_feather(path: str | Path, *, target_type: Type[T] | None = None) -> list[T | dict]:
        """Import contract(s) from Feather file.

        Args:
            path: Input file path
            target_type: Expected contract type

        Returns:
            List of reconstructed contracts
        """
        table = feather.read_table(path)
        df = table.to_pandas()

        records = []
        for _, row in df.iterrows():
            record = ContractImporter._denormalize_row(row)
            records.append(ContractImporter.from_dict(record, target_type=target_type))

        return records

    @staticmethod
    def _denormalize_row(row: pd.Series) -> dict[str, Any]:
        """Convert flattened row back to nested dictionary."""
        result = {}

        for key, value in row.items():
            try:
                if pd.isna(value):
                    continue
            except (ValueError, TypeError):
                # Handle arrays or other types where isna() is ambiguous
                pass

            parts = key.split("__")
            current = result

            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result


def import_from_json(
    path: str | Path | None = None,
    json_str: str | None = None,
    *,
    target_type: Type[T] | None = None,
) -> T | dict:
    """Import contract from JSON.

    Args:
        path: JSON file path (mutually exclusive with json_str)
        json_str: JSON string (mutually exclusive with path)
        target_type: Expected contract type

    Returns:
        Reconstructed contract object
    """
    if path is not None and json_str is not None:
        raise ValueError("Provide either path or json_str, not both")

    if path is None and json_str is None:
        raise ValueError("Must provide either path or json_str")

    if path is not None:
        json_str = Path(path).read_text(encoding="utf-8")

    return ContractImporter.from_json(json_str, target_type=target_type)


def import_from_parquet(
    path: str | Path,
    *,
    target_type: Type[T] | None = None,
) -> list[T | dict]:
    """Import contract(s) from Parquet file.

    Args:
        path: Input file path
        target_type: Expected contract type

    Returns:
        List of reconstructed contracts
    """
    return ContractImporter.from_parquet(path, target_type=target_type)


def import_from_feather(
    path: str | Path,
    *,
    target_type: Type[T] | None = None,
) -> list[T | dict]:
    """Import contract(s) from Feather file.

    Args:
        path: Input file path
        target_type: Expected contract type

    Returns:
        List of reconstructed contracts
    """
    return ContractImporter.from_feather(path, target_type=target_type)


def import_batch(
    directory: str | Path,
    *,
    pattern: str = "*.json",
    target_type: Type[T] | None = None,
) -> list[T | dict]:
    """Import multiple contracts from a directory.

    Args:
        directory: Input directory path
        pattern: File glob pattern (*.json, *.parquet, *.feather)
        target_type: Expected contract type

    Returns:
        List of reconstructed contracts
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    results = []

    for path in paths:
        suffix = path.suffix.lower()

        if suffix == ".json":
            obj = import_from_json(path, target_type=target_type)
            results.append(obj)
        elif suffix == ".parquet":
            objs = import_from_parquet(path, target_type=target_type)
            results.extend(objs)
        elif suffix == ".feather":
            objs = import_from_feather(path, target_type=target_type)
            results.extend(objs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    return results
