# -*- coding: utf-8 -*-
"""R42-007/008: Analyzer Canonical Tables — operator-ID-based canonical membership。

本模块为 Analyzer（子树优化/规则验证/语义审计）提供 operator-ID-based canonical
membership 和 dependency tables，避免重复扫描 registry。

核心类型
    - :class:`CanonicalMembershipTable`: operator_id -> canonical_name 查询表
    - :class:`CanonicalDependencyTable`: canonical_name -> dependencies 查询表
    - :class:`AnalyzerCanonicalRegistry`: 统一查询入口

R42-007/008 要求
    编译期构造（load_all 后）：
    - operator_id -> canonical_name 映射（O(1) 反向查询）
    - canonical -> dependencies (operators/params/sources) 映射
    - canonical -> temporal/PIT/market contracts 映射
    运行时 O(1) 查询，零重复扫描。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalMembership:
    """单个 operator 的 canonical 归属元数据（R42-007）。

    Attributes:
        operator_id: Unique operator ID (sha256-based or name-based)
        canonical_name: Canonical name (production/research surface)
        surface: 'production' | 'research' | 'internal' | 'tombstone'
        backend_binding: Backend family (pandas/polars/duckdb/numba)
        parameter_hash: Parameter spec digest (区分同 canonical 不同参数实现)
    """

    operator_id: str
    canonical_name: str
    surface: str = "production"
    backend_binding: str = ""
    parameter_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "canonical_name": self.canonical_name,
            "surface": self.surface,
            "backend_binding": self.backend_binding,
            "parameter_hash": self.parameter_hash,
        }


@dataclass(frozen=True)
class CanonicalDependencies:
    """单个 canonical 的依赖元数据（R42-008）。

    Attributes:
        canonical_name: Canonical name
        dependent_operators: 该 canonical 依赖的其他 canonicals（递归依赖）
        required_parameters: 必须参数名列表
        optional_parameters: 可选参数名列表
        required_sources: 必须 data sources (table.column)
        temporal_contract: Temporal contract (causal/non_causal/state/instantaneous)
        pit_contract: PIT contract (strict/relaxed)
        market_contract: Market contract (ashare/us/generic/multi)
    """

    canonical_name: str
    dependent_operators: frozenset[str] = frozenset()
    required_parameters: frozenset[str] = frozenset()
    optional_parameters: frozenset[str] = frozenset()
    required_sources: frozenset[str] = frozenset()
    temporal_contract: str = "unknown"
    pit_contract: str = "unknown"
    market_contract: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "dependent_operators": sorted(self.dependent_operators),
            "required_parameters": sorted(self.required_parameters),
            "optional_parameters": sorted(self.optional_parameters),
            "required_sources": sorted(self.required_sources),
            "temporal_contract": self.temporal_contract,
            "pit_contract": self.pit_contract,
            "market_contract": self.market_contract,
        }


@dataclass
class CanonicalMembershipTable:
    """R42-007: operator_id -> canonical_name 映射表（O(1) 反向查询）。

    Attributes:
        memberships: operator_id -> CanonicalMembership
        canonical_to_operators: canonical_name -> list[operator_id] 正向索引
        surface_index: surface -> list[canonical_name] 表面索引
    """

    memberships: dict[str, CanonicalMembership] = field(default_factory=dict)
    canonical_to_operators: dict[str, list[str]] = field(default_factory=dict)
    surface_index: dict[str, list[str]] = field(default_factory=dict)

    def get_canonical(self, operator_id: str) -> str | None:
        """O(1) 查询 operator 的 canonical name。"""
        m = self.memberships.get(operator_id)
        return m.canonical_name if m else None

    def get_operators(self, canonical_name: str) -> list[str]:
        """O(1) 查询 canonical 的所有 operator_id（多后端实现）。"""
        return self.canonical_to_operators.get(canonical_name, [])

    def get_surface(self, canonical_name: str) -> str | None:
        """O(1) 查询 canonical 的 surface。"""
        ops = self.canonical_to_operators.get(canonical_name, [])
        if not ops:
            return None
        m = self.memberships.get(ops[0])
        return m.surface if m else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memberships": {k: v.to_dict() for k, v in self.memberships.items()},
            "canonical_to_operators": self.canonical_to_operators,
            "surface_index": self.surface_index,
        }


@dataclass
class CanonicalDependencyTable:
    """R42-008: canonical_name -> dependencies 映射表。

    Attributes:
        dependencies: canonical_name -> CanonicalDependencies
        reverse_index: operator_name -> list[dependent_canonical] 反向依赖索引
    """

    dependencies: dict[str, CanonicalDependencies] = field(default_factory=dict)
    reverse_index: dict[str, list[str]] = field(default_factory=dict)

    def get(self, canonical_name: str) -> CanonicalDependencies | None:
        """O(1) 查询 canonical 的依赖元数据。"""
        return self.dependencies.get(canonical_name)

    def get_dependents(self, operator_name: str) -> list[str]:
        """O(1) 查询依赖该 operator 的所有 canonicals（反向索引）。"""
        return self.reverse_index.get(operator_name, [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependencies": {k: v.to_dict() for k, v in self.dependencies.items()},
            "reverse_index": self.reverse_index,
        }


@dataclass
class AnalyzerCanonicalRegistry:
    """R42-007/008: Analyzer 统一查询入口（membership + dependency）。

    Attributes:
        membership_table: operator_id -> canonical mapping
        dependency_table: canonical -> dependencies mapping
        build_version: Registry build version (git HEAD + timestamp)
    """

    membership_table: CanonicalMembershipTable = field(default_factory=CanonicalMembershipTable)
    dependency_table: CanonicalDependencyTable = field(default_factory=CanonicalDependencyTable)
    build_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "membership_table": self.membership_table.to_dict(),
            "dependency_table": self.dependency_table.to_dict(),
            "build_version": self.build_version,
        }


def build_canonical_tables(registry: Any) -> AnalyzerCanonicalRegistry:
    """R42-007/008: 从 OperatorRegistry 构造 Analyzer canonical tables。

    Args:
        registry: OperatorRegistry instance (load_all 后)

    Returns:
        预构造的不可变查询表
    """
    membership = CanonicalMembershipTable()
    dependency = CanonicalDependencyTable()

    # R42-007: 构造 operator_id -> canonical membership
    all_ops = getattr(registry, "_operators", {})
    canonical_groups: dict[str, list[str]] = {}

    for op_id, op_spec in all_ops.items():
        canonical = getattr(op_spec, "canonical_name", None) or str(op_id)
        surface = getattr(op_spec, "surface", "production")
        backend = getattr(op_spec, "backend", "")
        param_hash = ""
        try:
            import hashlib
            params = getattr(op_spec, "parameters", {})
            param_str = str(sorted((k, str(v)) for k, v in params.items()))
            param_hash = hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]
        except Exception:
            pass

        m = CanonicalMembership(
            operator_id=str(op_id),
            canonical_name=canonical,
            surface=surface,
            backend_binding=backend,
            parameter_hash=param_hash,
        )
        membership.memberships[str(op_id)] = m
        canonical_groups.setdefault(canonical, []).append(str(op_id))

    # 构造正向索引和表面索引
    for canonical, op_ids in canonical_groups.items():
        membership.canonical_to_operators[canonical] = op_ids
        if op_ids:
            first = membership.memberships.get(op_ids[0])
            if first:
                surf = first.surface
                membership.surface_index.setdefault(surf, []).append(canonical)

    # R42-008: 构造 canonical -> dependencies
    for canonical, op_ids in canonical_groups.items():
        if not op_ids:
            continue
        first_op = all_ops.get(op_ids[0])
        if first_op is None:
            continue

        # 收集依赖的其他 operators
        dependent_ops = set()
        try:
            deps = getattr(first_op, "dependencies", [])
            dependent_ops.update(str(d) for d in deps)
        except Exception:
            pass

        # 收集参数
        required_params = set()
        optional_params = set()
        try:
            params = getattr(first_op, "parameters", {})
            for pname, pspec in params.items():
                is_required = getattr(pspec, "required", False)
                if is_required:
                    required_params.add(str(pname))
                else:
                    optional_params.add(str(pname))
        except Exception:
            pass

        # 收集 required sources
        required_sources = set()
        try:
            sources = getattr(first_op, "required_sources", [])
            required_sources.update(str(s) for s in sources)
        except Exception:
            pass

        # 收集 contracts
        temporal = getattr(first_op, "temporal_contract", "unknown")
        pit = getattr(first_op, "pit_contract", "unknown")
        market = getattr(first_op, "market_contract", "generic")

        dep = CanonicalDependencies(
            canonical_name=canonical,
            dependent_operators=frozenset(dependent_ops),
            required_parameters=frozenset(required_params),
            optional_parameters=frozenset(optional_params),
            required_sources=frozenset(required_sources),
            temporal_contract=str(temporal),
            pit_contract=str(pit),
            market_contract=str(market),
        )
        dependency.dependencies[canonical] = dep

        # 构造反向依赖索引
        for dep_op in dependent_ops:
            dependency.reverse_index.setdefault(dep_op, []).append(canonical)

    # 构造 build version
    build_version = ""
    try:
        import subprocess
        import time
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        ts = int(time.time())
        build_version = f"{head}@{ts}"
    except Exception:
        build_version = "unknown"

    return AnalyzerCanonicalRegistry(
        membership_table=membership,
        dependency_table=dependency,
        build_version=build_version,
    )
