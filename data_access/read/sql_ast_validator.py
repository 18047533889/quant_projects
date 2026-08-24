"""R32-P0-106: SQL AST security boundary - 完整枚举所有表引用源。

使用 sqlglot AST 遍历所有可能的表引用，不依赖正则表达式作为安全边界。
"""
from __future__ import annotations

import logging
from typing import Any

from data_access.core.exceptions import ValidationError

logger = logging.getLogger("data_access.sql_ast")


def enumerate_sql_sources(sql: str) -> set[str]:
    """使用 sqlglot AST 枚举 SQL 中的所有表/源引用。
    
    覆盖：
    - FROM clause tables
    - JOIN tables
    - Comma joins (FROM a, b)
    - CTEs (WITH ... AS (...))
    - Subqueries
    - Table functions (read_parquet, read_csv, etc.)
    - LATERAL joins
    - System tables (information_schema, pg_catalog, etc.)
    
    Returns:
        所有表/源的完全限定名集合
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        raise ValidationError(
            "sqlglot 未安装，无法执行 SQL AST 安全验证（R32-P0-106）"
        )
    
    try:
        parsed = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception as e:
        raise ValidationError(f"SQL 解析失败（R32-P0-106）: {e}")
    
    sources: set[str] = set()
    
    def extract_table_name(node: exp.Table) -> str:
        """提取表的完全限定名。"""
        parts = []
        if node.catalog:
            parts.append(node.catalog)
        if node.db:
            parts.append(node.db)
        if node.name:
            parts.append(node.name)
        return ".".join(parts) if parts else "<unknown>"
    
    # 遍历 AST 节点
    for node in parsed.walk():
        # 1. FROM/JOIN 中的表
        if isinstance(node, exp.Table):
            table_name = extract_table_name(node)
            # 检查是否是系统表
            if any(sys in table_name.lower() for sys in [
                "information_schema", "pg_catalog", "system", "duckdb_",
                "sqlite_", "temp.", "temporary"
            ]):
                sources.add(f"SYSTEM:{table_name}")
            else:
                sources.add(table_name)
        
        # 2. Table functions (read_parquet, read_csv, etc.)
        elif isinstance(node, (exp.Anonymous, exp.Func)):
            # 检查函数名
            func_name = ""
            if hasattr(node, "this") and isinstance(node.this, str):
                func_name = node.this.lower()
            elif hasattr(node, "name"):
                func_name = node.name.lower()
            
            if func_name:
                sources.add(f"FUNCTION:{func_name}")
    
    return sources


def validate_sql_sources_against_allowlist(
    sql: str,
    allowed_sources: set[str],
) -> None:
    """验证 SQL 中的所有源都在白名单内（R32-P0-106）。
    
    Args:
        sql: 用户 SQL
        allowed_sources: 允许的源名称集合（来自 read_datasets）
    
    Raises:
        ValidationError: 发现未授权的源引用
    """
    discovered = enumerate_sql_sources(sql)
    
    # 检查函数调用（禁止所有文件函数）
    forbidden_funcs = [
        "read_parquet", "read_csv", "read_csv_auto", 
        "read_json", "read_json_auto", "read_ndjson",
        "parquet_scan", "csv_scan", "glob",
        "read_blob", "read_text",
    ]
    for source in discovered:
        if source.startswith("FUNCTION:"):
            func_name = source.split(":", 1)[1]
            if func_name in forbidden_funcs:
                raise ValidationError(
                    f"R32-P0-106 SQL 安全边界：禁止使用文件函数 '{func_name}'。"
                    f"请使用已注册的 dataset。"
                )
    
    # 检查系统表（全部禁止）
    for source in discovered:
        if source.startswith("SYSTEM:"):
            table_name = source.split(":", 1)[1]
            raise ValidationError(
                f"R32-P0-106 SQL 安全边界：禁止访问系统表 '{table_name}'。"
            )
    
    # 检查普通表是否在白名单内
    table_sources = {s for s in discovered if not s.startswith(("FUNCTION:", "SYSTEM:"))}
    unauthorized = table_sources - allowed_sources
    
    if unauthorized:
        unauthorized_str = ", ".join(sorted(unauthorized))
        allowed_str = ", ".join(sorted(allowed_sources))
        raise ValidationError(
            f"R32-P0-106 SQL 安全边界：发现未授权的表引用 [{unauthorized_str}]。"
            f"允许的表: [{allowed_str}]"
        )


def validate_sql_no_mutations(sql: str) -> None:
    """验证 SQL 不包含任何数据修改操作（R32-P0-106）。
    
    Raises:
        ValidationError: 发现 DML/DDL 操作
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        raise ValidationError("sqlglot 未安装")
    
    try:
        parsed = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception as e:
        raise ValidationError(f"SQL 解析失败: {e}")
    
    # 检查语句类型（使用实际存在的类型）
    mutation_types = (
        exp.Insert, exp.Update, exp.Delete, exp.Merge,
        exp.Create, exp.Drop, exp.Alter,
        exp.Copy, exp.Command,  # COPY, PRAGMA, etc.
    )
    
    for node in parsed.walk():
        if isinstance(node, mutation_types):
            node_type = type(node).__name__
            raise ValidationError(
                f"R32-P0-106 SQL 安全边界：禁止 {node_type} 操作。"
                f"仅支持只读 SELECT 查询。"
            )


__all__ = [
    "enumerate_sql_sources",
    "validate_sql_sources_against_allowlist",
    "validate_sql_no_mutations",
]
