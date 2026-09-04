"""Team DataAccess API with executable COS panel and PIT contracts."""
from __future__ import annotations

from data_access.core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from data_access.core.exceptions import (
    AmbiguousSemanticFieldError,
    DataAccessError,
    DataError,
    DeadlineExceeded,
    EngineError,
    ValidationError,
)
from data_access.read.key_policy import KeyPolicy
from data_access.read.aggregation import (
    AggregationItem,
    AggregationSpec,
    aggregate_minute_bundle,
    aggregate_minute_to_daily,
)
from data_access.read.session_calendar import (
    MarketCalendar,
    MarketSession,
    get_market_calendar,
    get_market_session,
)
from data_access.read.pit_event_index import (
    PITEventIndex,
    PITEventRecord,
)
from data_access.read.physical_plan import PlanNode, build_physical_plan
from data_access.read.contract_ir import ContractIR, build_contract_ir
from data_access.read.query_budget import QueryBudget
from data_access.read.read_contract import DataSnapshot, ReadResult, SqlReadResult
from data_access.read.scan_handle import ScanHandle
from data_access.read.read_handle import ReadHandle
from data_access.read.relation_handle import RelationHandle
from data_access.read.data_request import DataRequest, ReadPlan
from data_access.read.semantic_catalog import (
    FieldSemanticDescriptor,
    FieldTaxonomyProvider,
    SemanticField,
    SemanticFieldCatalog,
    SemanticFieldTaxonomyProvider,
    get_semantic_catalog,
)
from . import store as _store_module
from .store import DataAccessStore, get_store as _get_store, reset_store
from .cos_contract import (
    COSDatasetContract,
    COS_DATASET_CONTRACTS,
    get_cos_contract,
    normalize_return_values,
    require_cos_contract,
    resolve_event_clock,
    semantic_contract_fingerprint,
    validate_panel_request,
)
from .cos_runtime import install_cos_runtime


def get_store() -> DataAccessStore:
    """Return the process store with COS semantics installed exactly once."""
    return install_cos_runtime(_get_store())


# Importing data_access.store still initializes this package first; expose the
# same hardened factory there so callers cannot bypass COS contracts by import path.
_store_module.get_store = get_store


__all__ = [
    "get_store",
    "reset_store",
    "get_shared_engine",
    "reset_shared_engine",
    "DataAccessStore",
    "DuckDBEngine",
    "DataAccessError",
    "ValidationError",
    "AmbiguousSemanticFieldError",
    "DataError",
    "EngineError",
    "DeadlineExceeded",
    "AggregationSpec",
    "AggregationItem",
    "aggregate_minute_to_daily",
    "aggregate_minute_bundle",
    "MarketCalendar",
    "MarketSession",
    "get_market_calendar",
    "get_market_session",
    "PITEventIndex",
    "PITEventRecord",
    "PlanNode",
    "build_physical_plan",
    "ContractIR",
    "build_contract_ir",
    "QueryBudget",
    "KeyPolicy",
    "DataSnapshot",
    "ReadResult",
    "SqlReadResult",
    "ScanHandle",
    "ReadHandle",
    "RelationHandle",
    "DataRequest",
    "ReadPlan",
    "SemanticField",
    "SemanticFieldCatalog",
    "SemanticFieldTaxonomyProvider",
    "FieldSemanticDescriptor",
    "FieldTaxonomyProvider",
    "get_semantic_catalog",
    "COSDatasetContract",
    "COS_DATASET_CONTRACTS",
    "get_cos_contract",
    "require_cos_contract",
    "validate_panel_request",
    "resolve_event_clock",
    "normalize_return_values",
    "semantic_contract_fingerprint",
]

# Build identity is frozen into _build_info at packaging time. Importing an
# installed artifact never consults environment variables, package metadata, or Git.
from data_access.core.build_metadata import load_build_info as _load_build_metadata

_build_metadata = _load_build_metadata()
__version__ = _build_metadata.version
__build_sha__ = _build_metadata.build_sha
__build_id__ = _build_metadata.build_id
__build_time__ = _build_metadata.build_time
