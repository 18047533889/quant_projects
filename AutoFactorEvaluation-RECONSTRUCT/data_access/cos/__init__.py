"""COS 镜像与远程直读。"""
from .mirror import (
    ASHARE_COS_PREFIX,
    ASHARE_DATASET_TABLE_MAP,
    DATASET_MIRROR_REGISTRY,
    US_CLEAN_COS_PREFIX,
    US_DATASET_TABLE_MAP,
    US_MASSIVE_COS_PREFIX,
    ensure_local_mirror_for_dataset,
)
from .remote import cos_cache_root, cos_remote_backend, prepare_cos_remote_paths, should_read_cos_remote

__all__ = [
    "ASHARE_COS_PREFIX",
    "ASHARE_DATASET_TABLE_MAP",
    "DATASET_MIRROR_REGISTRY",
    "US_CLEAN_COS_PREFIX",
    "US_DATASET_TABLE_MAP",
    "US_MASSIVE_COS_PREFIX",
    "ensure_local_mirror_for_dataset",
    "cos_cache_root",
    "cos_remote_backend",
    "prepare_cos_remote_paths",
    "should_read_cos_remote",
]
