# -*- coding: utf-8 -*-
"""Export utilities for serializing FactorEngine contracts and data.

This package provides comprehensive serialization, import, and conversion
utilities for all major FactorEngine contracts including modeling contracts,
execution contracts, and data structures.
"""
from __future__ import annotations

from .serializers import (
    ContractSerializer,
    serialize_to_json,
    serialize_to_parquet,
    serialize_to_feather,
    serialize_batch,
)
from .importers import (
    ContractImporter,
    import_from_json,
    import_from_parquet,
    import_from_feather,
    import_batch,
)
from .converters import (
    DataFrameConverter,
    to_pandas,
    to_polars,
    to_arrow,
    convert_batch,
)
from .versioning import (
    SchemaVersion,
    VersionRegistry,
    migrate_contract,
    get_current_version,
    register_migration,
)

__all__ = [
    "ContractSerializer",
    "serialize_to_json",
    "serialize_to_parquet",
    "serialize_to_feather",
    "serialize_batch",
    "ContractImporter",
    "import_from_json",
    "import_from_parquet",
    "import_from_feather",
    "import_batch",
    "DataFrameConverter",
    "to_pandas",
    "to_polars",
    "to_arrow",
    "convert_batch",
    "SchemaVersion",
    "VersionRegistry",
    "migrate_contract",
    "get_current_version",
    "register_migration",
]
