"""数据集登记与路径策略。"""
from .loader import (
    Dataset,
    DatasetRegistry,
    ParametricDataset,
    StaticDataset,
    load_registry,
)
from .paths import PathAuthorizer, canonicalize, dataset_env_root, expand_env, extra_allowed_roots_from_env
from .params_validation import ParamSpec, params_fingerprint, parse_params_schema, validate_params
from .layout_policy import LayoutPolicy, bucket_values_for_instruments, parse_layout_policy, prune_glob_paths_for_buckets, stable_bucket
from .schema_validation import check_schema, enforce_schema_or_raise, mark_validated, reset_validated_cache, schema_cache_key

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "ParametricDataset",
    "StaticDataset",
    "load_registry",
    "PathAuthorizer",
    "canonicalize",
    "dataset_env_root",
    "expand_env",
    "extra_allowed_roots_from_env",
    "ParamSpec",
    "params_fingerprint",
    "parse_params_schema",
    "validate_params",
    "LayoutPolicy",
    "parse_layout_policy",
    "stable_bucket",
    "bucket_values_for_instruments",
    "prune_glob_paths_for_buckets",
    "check_schema",
    "enforce_schema_or_raise",
    "mark_validated",
    "reset_validated_cache",
    "schema_cache_key",
]
