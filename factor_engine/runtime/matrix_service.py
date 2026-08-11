"""factor_matrix 宽表物化编排。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

from api.factor import Factor

if TYPE_CHECKING:
    from runtime.engine import FactorEngine
    from runtime.materialization_identity_certificate import (
        MaterializationIdentityCertificate,
    )


def _matrix_factor_digest(
    engine: "FactorEngine",
    factor: Factor,
    fid: str,
    analysis: Any,
    series: Any,
    effective_config: dict | None,
    scope: Any,
    pit_enforce: bool | None,
) -> str | None:
    """按真实执行 scope + analysis + source contract 计算该 factor 的 semantic digest。

    R14 #2：matrix 的 factor_version 不能靠调用方手工传 ``factor_versions``——
    production 主链会自动绑定。这里与 ``execute_materialize`` 用同一套
    ``build_materialize_lineage`` + ``_build_semantic_identity`` 机制，保证
    Parquet 落盘、catalog、ClickHouse、matrix 四处的 version 同源。
    """
    from runtime import lineage_service
    from runtime.materialize_service import _build_semantic_identity
    from storage.catalog import compute_ir_hash

    lineage = lineage_service.build_materialize_lineage(
        factor=factor,
        analysis=analysis,
        output={"result": series},
        factor_id=fid,
        expression=getattr(factor, "source_expr", None),
        data_source_config=effective_config,
        data_source=engine.data_source,
        mode="full",
    )
    identity = _build_semantic_identity(
        engine=engine,
        factor=factor,
        ir_node=analysis.ir,
        ast_hash=compute_ir_hash(analysis.ir),
        frequency=getattr(scope, "frequency", None) or getattr(factor, "freq", None),
        data_source_config=effective_config,
        run_lineage={**lineage.to_dict(), "factor_id": fid},
        pit_enforce=pit_enforce,
        scope=scope,
    )
    if identity is None:
        return None
    return identity.identity_digest()[:16]


def _validate_matrix_scope(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    ids: list[str],
    *,
    universe: str,
    frequency: str,
) -> list[Any]:
    """R14 #2：校验 matrix 声明的 ``universe/frequency`` 与每个 factor 的实际执行
    scope（``_scope_from_factor``，优先 ``factor.semantic_identity``）一致。

    否则会把实际 5m / CSI500 的结果写进 ``freq=1d / universe=CSI300`` 路径，
    下游按错误语义读取。factor 未声明 universe（默认 ALL/动态）时由 matrix
    universe 治理，不判冲突；未声明 freq（默认 1d）时与 matrix freq=1d 一致。
    """
    from runtime.engine import _scope_from_factor

    scopes: list[Any] = []
    for factor, fid in zip(factors, ids):
        scope = _scope_from_factor(factor, data_source=engine.data_source)
        scopes.append(scope)
        sfreq = str(getattr(scope, "frequency", None) or "1d")
        if sfreq != str(frequency):
            raise ValueError(
                f"matrix frequency={frequency} 与 factor {fid} 实际执行 scope "
                f"frequency={sfreq} 不一致——禁止把 {sfreq} 结果写进 freq={frequency} "
                f"路径"
            )
        suniv = str(getattr(scope, "universe_id", None) or "ALL")
        if suniv not in {"", "ALL"} and suniv != str(universe):
            raise ValueError(
                f"matrix universe={universe} 与 factor {fid} 实际执行 scope "
                f"universe={suniv} 不一致——禁止把 {suniv} 结果写进 "
                f"universe={universe} 路径"
            )
    return scopes


def _operator_manifest_hash() -> str | None:
    """Best-effort operator manifest hash (computed once per batch)."""
    try:
        from cleaned_operators.operator_policy import compute_operator_catalog_hash

        return compute_operator_catalog_hash()
    except Exception:  # pragma: no cover - 失败按 None 记录，digest 仍权威
        return None


def build_certificates(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    ids: list[str],
    *,
    analyses: dict[str, Any],
    results: dict[str, Any],
    scopes: list[Any],
    effective_config: dict | None,
    pit_enforce: bool | None,
    universe: str,
    frequency: str,
    value_dtype: str = "float32",
) -> dict[str, "MaterializationIdentityCertificate"]:
    """R39 PERF-067: build compile-time identity certificates for a whole batch.

    Computed once (not per-factor inside the writer) and handed to the matrix
    writer so it never reconstructs large metadata dicts per factor.  The
    ``semantic_digest`` reuses ``_matrix_factor_digest`` (same lineage/source
    contract mechanism as ``execute_materialize``); other fields are extracted
    cheaply from the execution scope + engine.
    """
    from runtime.materialization_identity_certificate import (
        MaterializationIdentityCertificate,
        extract_scope_fields,
    )

    op_hash = _operator_manifest_hash()
    certs: dict[str, MaterializationIdentityCertificate] = {}
    for (factor, fid), scope in zip(zip(factors, ids), scopes):
        analysis = analyses.get(factor.name)
        series = results.get(fid)
        digest = None
        if analysis is not None and series is not None:
            digest = _matrix_factor_digest(
                engine,
                factor,
                fid,
                analysis,
                series,
                effective_config,
                scope,
                pit_enforce,
            )
        fields = extract_scope_fields(scope, engine, value_dtype)
        certs[fid] = MaterializationIdentityCertificate(
            factor_id=fid,
            semantic_digest=digest,
            source_snapshot=fields["source_snapshot"],
            calendar=fields["calendar"],
            universe=fields["universe"],
            frequency=fields["frequency"] or getattr(scope, "frequency", None),
            storage_precision=fields["storage_precision"],
            operator_manifest_hash=op_hash,
        )
    return certs


def _iter_matrix_results(
    run_out: dict[str, Any],
    factors: Sequence[Factor],
    ids: list[str],
) -> Iterator[tuple[str, Any]]:
    """R39 PERF-066: lazy per-factor results adapter for the matrix writer.

    Real streaming boundary: ``engine.run_many`` materialises the full result
    dict in memory (engine API limitation; ``runtime/engine.py`` is out of R39
    scope).  The adapter yields ``(fid, series)`` without copying, so the writer
    can consume factors one block at a time instead of building a second full
    structure.  A true scheduler/sink -> FactorBlock stream requires the engine
    to emit results incrementally.
    """
    results = run_out.get("results", {})
    for factor, fid in zip(factors, ids):
        yield fid, results[factor.name]


def materialize_matrix_streaming(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    *,
    factor_ids: Sequence[str] | None = None,
    universe: str,
    frequency: str = "1d",
    matrix_root: str | Path | None = None,
    partition_columns: list[str] | None = None,
    value_dtype: str = "float32",
    recovery: bool = False,
    factor_versions: dict[str, str] | None = None,
    expected_manifest_version: int | None = None,
    layout: str = "block",
    **run_kwargs: Any,
) -> dict[str, Any]:
    """R39 PERF-066: streaming matrix materialisation.

    ``scheduler/sink -> FactorBlock -> matrix block writer``.  Because the engine
    ``run_many`` only returns a dict, this path uses a lazy generator adapter
    (``_iter_matrix_results``) that feeds the writer per factor/block and never
    builds an additional full-results structure; the writer's block path bounds
    assembly memory to one column-block of factors at a time.  Real streaming
    (results emitted from the scheduler as they complete) requires the engine to
    expose a sink API and is out of R39 scope (``runtime/engine.py`` untouched).

    Defaults to the column-factor block layout; pass ``layout="legacy"`` for the
    monolithic single-wide-file writer.
    """
    from storage.materialize.factor_matrix_materializer import FactorMatrixMaterializer

    ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
    if len(ids) != len(factors):
        raise ValueError("factor_ids 长度必须与 factors 一致")

    from runtime.production_policy import is_production_mode
    from runtime.materialize_service import _effective_data_source_config

    production = is_production_mode(engine.run_mode)
    effective_config = _effective_data_source_config(None, engine.data_source)
    pit_enforce = getattr(engine.data_source, "pit_enforce", None)

    scopes = _validate_matrix_scope(
        engine, factors, ids, universe=universe, frequency=frequency
    )

    run_out = engine.run_many(factors, **run_kwargs)
    results = {
        fid: run_out["results"][factor.name] for factor, fid in zip(factors, ids)
    }
    analyses = run_out.get("analyses", {})

    certificates = build_certificates(
        engine,
        factors,
        ids,
        analyses=analyses,
        results=results,
        scopes=scopes,
        effective_config=effective_config,
        pit_enforce=pit_enforce,
        universe=universe,
        frequency=frequency,
        value_dtype=value_dtype,
    )

    auto_versions = {
        fid: cert.semantic_digest
        for fid, cert in certificates.items()
        if cert.semantic_digest
    }
    bound_versions = dict(factor_versions or {})
    if production:
        missing = [fid for fid in ids if fid not in auto_versions]
        if missing:
            raise RuntimeError(
                f"production factor_matrix: 无法为因子 {missing} 计算 semantic "
                f"digest（缺 analysis / identity）——拒绝无版本发布，请检查 "
                f"factor scope + analysis + source contract"
            )
        for fid, digest in auto_versions.items():
            prev = bound_versions.get(fid)
            if prev is not None and prev != digest:
                raise ValueError(
                    f"matrix factor {fid}: 调用方手工传入的 version {prev} 与 "
                    f"本次真实执行 scope/analysis/source contract 自动算出的 "
                    f"{digest} 不一致——以真实执行语义为准，拒绝沿用错误版本"
                )
            bound_versions[fid] = digest
    else:
        for fid, digest in auto_versions.items():
            bound_versions.setdefault(fid, digest)

    materializer = FactorMatrixMaterializer(matrix_root=matrix_root)
    summary = materializer.materialize_from_iterable(
        _iter_matrix_results(run_out, factors, ids),
        universe=universe,
        frequency=frequency,
        partition_columns=partition_columns,
        value_dtype=value_dtype,
        production=is_production_mode(engine.run_mode),
        recovery=recovery,
        factor_versions=bound_versions,
        expected_manifest_version=expected_manifest_version,
        certificates=certificates,
        layout=layout,
    )
    summary["run_many"] = {
        "factor_names": [f.name for f in factors],
        "shared_nodes": len(run_out.get("dag").shared_nodes)
        if run_out.get("dag")
        else 0,
        "streaming": True,
        "boundary": "engine.run_many returns a full dict; writer consumes "
        "via lazy generator adapter (block-by-block assembly).",
    }
    if "rolling_cache" in run_out:
        summary["rolling_cache"] = run_out["rolling_cache"]
    return summary


def execute_materialize_matrix(
    engine: "FactorEngine",
    factors: Sequence[Factor],
    *,
    factor_ids: Sequence[str] | None = None,
    universe: str,
    frequency: str = "1d",
    matrix_root: str | Path | None = None,
    partition_columns: list[str] | None = None,
    value_dtype: str = "float32",
    recovery: bool = False,
    factor_versions: dict[str, str] | None = None,
    expected_manifest_version: int | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """``FactorEngine.materialize_matrix`` 实现体。

    R14 #2：matrix factor-version gate 不再是「可选功能」——
      * 先校验 matrix 声明的 ``universe/frequency`` 与每个 factor 的实际执行
        scope 一致（任何模式都拒绝错写路径）；
      * 然后按本次真实执行的 factor scope + analysis + source contract 自动算
        ``semantic_digest``；production 下任何 factor 缺 version 直接拒绝，
        调用方手工传入的 version 与自动算出的不一致也拒绝。

    ``expected_manifest_version`` 仍透传给 ``FactorMatrixMaterializer.materialize``
    （P1-15 CAS），不会进入 run_many。
    """
    ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
    if len(ids) != len(factors):
        raise ValueError("factor_ids 长度必须与 factors 一致")

    from runtime.production_policy import is_production_mode
    from runtime.materialize_service import _effective_data_source_config

    production = is_production_mode(engine.run_mode)
    effective_config = _effective_data_source_config(None, engine.data_source)
    pit_enforce = getattr(engine.data_source, "pit_enforce", None)

    scopes = _validate_matrix_scope(
        engine, factors, ids, universe=universe, frequency=frequency
    )

    run_out = engine.run_many(factors, **run_kwargs)
    results = {
        fid: run_out["results"][factor.name] for factor, fid in zip(factors, ids)
    }
    analyses = run_out.get("analyses", {})

    # R39 PERF-067: 编译期一次性构建 identity certificates（writer 直接消费，
    # 不为每 factor 重建）。semantic_digest 仍与 execute_materialize 同源。
    certificates = build_certificates(
        engine,
        factors,
        ids,
        analyses=analyses,
        results=results,
        scopes=scopes,
        effective_config=effective_config,
        pit_enforce=pit_enforce,
        universe=universe,
        frequency=frequency,
        value_dtype=value_dtype,
    )
    auto_versions: dict[str, str] = {
        fid: cert.semantic_digest
        for fid, cert in certificates.items()
        if cert.semantic_digest
    }

    bound_versions = dict(factor_versions or {})
    if production:
        # production fail-closed：任何 factor 缺 version 直接拒绝发布。
        missing = [fid for fid in ids if fid not in auto_versions]
        if missing:
            raise RuntimeError(
                f"production factor_matrix: 无法为因子 {missing} 计算 semantic "
                f"digest（缺 analysis / identity）——拒绝无版本发布，请检查 "
                f"factor scope + analysis + source contract"
            )
        for fid, digest in auto_versions.items():
            prev = bound_versions.get(fid)
            if prev is not None and prev != digest:
                raise ValueError(
                    f"matrix factor {fid}: 调用方手工传入的 version {prev} 与 "
                    f"本次真实执行 scope/analysis/source contract 自动算出的 "
                    f"{digest} 不一致——以真实执行语义为准，拒绝沿用错误版本"
                )
            bound_versions[fid] = digest
    else:
        for fid, digest in auto_versions.items():
            bound_versions.setdefault(fid, digest)

    from storage.factor_matrix_materializer import FactorMatrixMaterializer

    materializer = FactorMatrixMaterializer(matrix_root=matrix_root)
    summary = materializer.materialize(
        results,
        universe=universe,
        frequency=frequency,
        partition_columns=partition_columns,
        value_dtype=value_dtype,
        # #收官轮 P0：production 由 engine.run_mode 一锤定音，物化器不再自行
        # 从环境变量猜（否则 engine=production + env=research 时 matrix 直写
        # 本地不会被拒）。
        production=is_production_mode(engine.run_mode),
        recovery=recovery,
        factor_versions=bound_versions,
        expected_manifest_version=expected_manifest_version,
        certificates=certificates,
    )
    summary["run_many"] = {
        "factor_names": [f.name for f in factors],
        "shared_nodes": len(run_out.get("dag").shared_nodes)
        if run_out.get("dag")
        else 0,
    }
    if "rolling_cache" in run_out:
        summary["rolling_cache"] = run_out["rolling_cache"]
    return summary
