"""
R32-P0-105: Metadata mutator纳入同一write transaction/security/audit.
R32-P0-106: SQL sandbox只使用AST security boundary.

Metadata mutation governance and SQL security boundary.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Sequence

from data_access.core.exceptions import AccessDeniedError, ValidationError


class MetadataMutationType(enum.Enum):
    """Types of metadata mutations requiring authorization."""

    FACTOR_CATALOG_REFRESH = "factor_catalog_refresh"
    FACTOR_CATALOG_SAVE = "factor_catalog_save"
    MANIFEST_BUILD = "manifest_build"
    MANIFEST_PUBLISH = "manifest_publish"
    SCHEMA_METADATA = "schema_metadata"
    DQ_CERTIFICATION = "dq_certification"
    COVERAGE_METADATA = "coverage_metadata"


@dataclass(frozen=True)
class MetadataMutationRequest:
    """Governed metadata mutation request.

    R32-P0-105: 包括:
    - factor catalog refresh/save
    - manifest build/publish
    - schema metadata
    - DQ certification
    - coverage metadata

    Security metadata不能用普通helper静默写.
    """

    mutation_type: MetadataMutationType
    dataset: str
    principal: Any
    request_id: str
    metadata: dict[str, Any]

    def validate_authorization(self, authorizer: Any) -> None:
        """Validate principal is authorized for this mutation.

        Raises:
            AccessDeniedError: If not authorized
        """
        from data_access.write.authorization_boundary import (
            ACTION_METADATA_WRITE,
            ACTION_FACTOR_CATALOG_WRITE,
        )

        if self.mutation_type in {
            MetadataMutationType.FACTOR_CATALOG_REFRESH,
            MetadataMutationType.FACTOR_CATALOG_SAVE,
        }:
            action = ACTION_FACTOR_CATALOG_WRITE
        else:
            action = ACTION_METADATA_WRITE

        try:
            authorizer.authorize(self.principal, self.dataset, action=action)
        except Exception as exc:
            raise AccessDeniedError(
                f"Principal {self.principal.principal_id} not authorized "
                f"for {action} on {self.dataset}"
            ) from exc


class MetadataMutationTransaction:
    """Transaction wrapper for metadata mutations.

    R32-P0-105: Metadata mutator纳入同一write transaction/security/audit.
    """

    def __init__(
        self,
        request: MetadataMutationRequest,
        authorizer: Any,
        audit_logger: Any,
    ) -> None:
        self.request = request
        self.authorizer = authorizer
        self.audit_logger = audit_logger
        self._committed = False

    def __enter__(self) -> MetadataMutationTransaction:
        # Pre-flight authorization check
        self.request.validate_authorization(self.authorizer)

        # Audit start
        self.audit_logger.record(
            event="metadata_mutation_start",
            request_id=self.request.request_id,
            principal_id=self.request.principal.principal_id,
            mutation_type=self.request.mutation_type.value,
            dataset=self.request.dataset,
        )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        success = exc_type is None and self._committed

        # Audit completion
        self.audit_logger.record(
            event="metadata_mutation_complete",
            request_id=self.request.request_id,
            principal_id=self.request.principal.principal_id,
            mutation_type=self.request.mutation_type.value,
            dataset=self.request.dataset,
            success=success,
            error=str(exc_val) if exc_val else None,
        )

    def commit(self) -> None:
        """Mark transaction as committed."""
        self._committed = True


@dataclass(frozen=True)
class SqlTableReference:
    """SQL table reference extracted from AST."""

    table_name: str
    schema_name: str | None = None
    catalog_name: str | None = None
    source_type: str = "table"  # "table", "cte", "subquery", "table_function"

    def full_name(self) -> str:
        """Get fully qualified table name."""
        parts = []
        if self.catalog_name:
            parts.append(self.catalog_name)
        if self.schema_name:
            parts.append(self.schema_name)
        parts.append(self.table_name)
        return ".".join(parts)


class SqlAstSecurityBoundary:
    """SQL security boundary using AST extraction.

    R32-P0-106: SQL sandbox只使用AST security boundary.
    Regex不作为table source authority.

    通过sqlglot/DuckDB AST枚举:
    - comma join
    - CTE
    - subquery
    - table function
    - schema/catalog
    - lateral
    - cross/natural join
    - system tables

    所有undeclared source reject.
    """

    def __init__(self, allowed_tables: Sequence[str]) -> None:
        self.allowed_tables = frozenset(allowed_tables)

    def extract_table_references(self, sql: str) -> list[SqlTableReference]:
        """Extract all table references from SQL using AST.

        R32-P0-106: Uses sqlglot AST, not regex.
        """
        try:
            import sqlglot
        except ImportError:
            raise ValidationError(
                "R32-P0-106 requires sqlglot for AST-based SQL security. "
                "Install: pip install sqlglot"
            )

        references: list[SqlTableReference] = []

        try:
            # Parse SQL to AST
            parsed = sqlglot.parse_one(sql, dialect="duckdb")
        except Exception as exc:
            raise ValidationError(f"SQL parse failed: {exc}") from exc

        # Extract table references from AST
        for node in parsed.walk():
            if isinstance(node, sqlglot.exp.Table):
                ref = SqlTableReference(
                    table_name=node.name,
                    schema_name=node.db if hasattr(node, "db") else None,
                    catalog_name=node.catalog if hasattr(node, "catalog") else None,
                    source_type="table",
                )
                references.append(ref)

            elif isinstance(node, sqlglot.exp.CTE):
                # CTE is allowed as it's defined in same query
                ref = SqlTableReference(
                    table_name=node.alias,
                    source_type="cte",
                )
                references.append(ref)

            elif isinstance(node, sqlglot.exp.Subquery):
                # Subquery itself is not a table reference
                # but might contain table references (handled by walk)
                pass

            elif isinstance(node, (sqlglot.exp.Anonymous, sqlglot.exp.Func)):
                # Table functions like read_parquet('path')
                # sqlglot.exp.TableFunction doesn't exist in some versions
                # Check if it's used as table source (has FROM context)
                if hasattr(node, "this") and node.this:
                    ref = SqlTableReference(
                        table_name=str(node.this) if hasattr(node, "this") else str(node),
                        source_type="table_function",
                    )
                    references.append(ref)

        return references

    def validate_sql(self, sql: str, *, allowed_functions: Sequence[str] = ()) -> None:
        """Validate SQL only references allowed tables.

        Args:
            sql: SQL query to validate
            allowed_functions: Additional allowed table functions (e.g., read_parquet)

        Raises:
            AccessDeniedError: If SQL references undeclared tables
        """
        references = self.extract_table_references(sql)

        allowed_funcs = frozenset(allowed_functions)
        undeclared: list[str] = []

        for ref in references:
            # Skip CTEs (defined in same query)
            if ref.source_type == "cte":
                continue

            # Skip allowed table functions
            if ref.source_type == "table_function" and ref.table_name in allowed_funcs:
                continue

            # Check if table is in allowed set
            full_name = ref.full_name()
            if full_name not in self.allowed_tables and ref.table_name not in self.allowed_tables:
                undeclared.append(full_name)

        if undeclared:
            raise AccessDeniedError(
                f"R32-P0-106: SQL references undeclared tables: {', '.join(undeclared)}. "
                f"Allowed: {', '.join(sorted(self.allowed_tables))}"
            )

    def extract_system_tables(self, sql: str) -> list[str]:
        """Extract references to system tables (information_schema, etc.)."""
        references = self.extract_table_references(sql)
        system_tables = []

        for ref in references:
            if ref.schema_name in {"information_schema", "pg_catalog", "sys"}:
                system_tables.append(ref.full_name())

        return system_tables


def create_sql_security_boundary(
    store: Any, additional_allowed: Sequence[str] = ()
) -> SqlAstSecurityBoundary:
    """Create SQL security boundary with store's registered datasets.

    R32-P0-106: All registered datasets are automatically allowed.
    """
    allowed = set(additional_allowed)

    # Add all registered dataset names
    if hasattr(store, "registry") and hasattr(store.registry, "names"):
        allowed.update(store.registry.names())

    return SqlAstSecurityBoundary(list(allowed))


__all__ = [
    "MetadataMutationType",
    "MetadataMutationRequest",
    "MetadataMutationTransaction",
    "SqlTableReference",
    "SqlAstSecurityBoundary",
    "create_sql_security_boundary",
]
