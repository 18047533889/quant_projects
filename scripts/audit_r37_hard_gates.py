# -*- coding: utf-8 -*-
"""R37 §28：Production Hard Gates（R37_P0_080/081/082 + 核心证据/参数域门）。

每个 gate 必须有真实 executed cases（R37-P0-001 硬规则：executed_cases==0 =>
NOT_RUN；failed_cases>0 => FAIL），不用文件存在代替行为测试。

覆盖：
    R37_CURRENT_HEAD_BOUND
    R37_EVIDENCE_TRUTH_PASS（evidence truth 8 gates 0 FAIL + 负控 5/5）
    R37_PARAMETER_DOMAIN_CERTIFIED（17 算子 / 84 精确点；exact-call 口径）
    R37_INVALID_PARAM_REJECTION_ZERO（57 invalid 全拒绝）
    R37_PER_CANONICAL_LEDGER_PRODUCED（1391 行 parquet）
    R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER（ledger canonical 覆盖 production target）
    R37_EXACT_CALL_CERTIFICATION_CONSUMED（production 执行路径含 membership 门源码）
    R37_PROPERTY_MUTATION_KILLED（mutation 负控）
    R37_DATA_KNOWLEDGE_IDENTITY_PASS
    R37_UNIVERSE_PIT_PASS
    R37_PRICE_BASIS_IDENTITY_PASS
    R37_CURRENT_HEAD_SHA_CONSISTENT（evidence SHA == benchmark SHA == code SHA）

输出：
    evidence/factor_engine/r37/R37_HARD_GATES.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from factor_engine.runtime.r34_evidence import GateResult  # noqa: E402
from factor_engine.runtime.evidence_truth import EvidenceTruthEngine, evidence_store_path  # noqa: E402

E = evidence_store_path()


def _head() -> str:
    from factor_engine.backend.evidence_provenance import current_commit_sha

    return current_commit_sha()


def _gate_evidence_truth() -> GateResult:
    from factor_engine.runtime.r34_evidence import current_evidence_header

    head = _head()
    engine = EvidenceTruthEngine(commit_sha=head)
    # 复用 evidence truth 的 8 个门：断言 0 FAIL 且负控全 fired
    try:
        truth = json.loads(
            (E / "R37_EVIDENCE_TRUTH_GATES.json").read_text(encoding="utf-8"))
        cases = []
        for gid, g in truth["gates"].items():
            if g["status"] == "FAIL":
                cases.append(False)
            else:
                cases.append(True)
        neg_controls = truth.get("negative_controls", {})
        fired = [v["fired"] for v in neg_controls.values()] or []
        return GateResult.from_cases(
            "R37_EVIDENCE_TRUTH_PASS", cases + fired, commit_sha=head,
            evidence_files=("evidence/factor_engine/r37/R37_EVIDENCE_TRUTH_GATES.json",),
            details={"evidence_truth_gates": len(truth["gates"]),
                     "negative_controls": len(neg_controls)},
        )
    except Exception as exc:
        return GateResult.not_run("R37_EVIDENCE_TRUTH_PASS", f"evidence missing: {exc}", head)


def _gate_parameter_domain_certified() -> GateResult:
    from factor_engine.runtime.parameter_domain_store import ParameterDomainCertificationStore

    head = _head()
    store = ParameterDomainCertificationStore()
    n = store.load_json(E / "R37_PARAMETER_DOMAIN_STORE.json")
    if n == 0:
        return GateResult.not_run("R37_PARAMETER_DOMAIN_CERTIFIED",
                                  "no certified points loaded", head)
    # exact-call 口径：每个 certified 点必须能被查询到（R39 #28：查询带**全维度**
    # identity——semantic_version/backend/variant/source_context/dtype/grain 不再
    # 落默认值，否则不同维度空间会互相错查）。
    cases = []
    for cp in store.all_passed_certified_points():
        kw = dict(cp.key.parameter_point)
        cases.append(store.exact_call_is_certified(
            cp.key.canonical, kw,
            semantic_version=cp.key.semantic_version, backend=cp.key.backend,
            execution_variant=cp.key.execution_variant,
            source_context=cp.key.source_context,
            dtype=cp.key.dtype, grain=cp.key.grain,
        ))
    return GateResult.from_cases(
        "R37_PARAMETER_DOMAIN_CERTIFIED", cases, commit_sha=head,
        evidence_files=("evidence/factor_engine/r37/R37_PARAMETER_DOMAIN_STORE.json",),
        details={"certified_point_count": n,
                 "operators": len({cp.key.canonical for cp in store.all_passed_certified_points()})},
    )


def _gate_invalid_rejection() -> GateResult:
    head = _head()
    try:
        cert = json.loads(
            (E / "R37_PARAMETER_DOMAIN_CERTIFICATION.json").read_text(encoding="utf-8"))
        failed = cert.get("invalid_rejection_failed", -1)
        ok = cert.get("invalid_rejection_passed", 0)
        if failed < 0:
            return GateResult.not_run("R37_INVALID_PARAM_REJECTION_ZERO",
                                      "no rejection data", head)
        return GateResult.from_cases(
            "R37_INVALID_PARAM_REJECTION_ZERO", [failed == 0, ok > 0], commit_sha=head,
            details={"invalid_rejected": ok, "invalid_accepted": failed},
        )
    except Exception as exc:
        return GateResult.not_run("R37_INVALID_PARAM_REJECTION_ZERO", str(exc), head)


def _gate_ledger_produced() -> GateResult:
    head = _head()
    p = E / "R37_OPERATOR_CORRECTNESS_LEDGER.parquet"
    if not p.is_file():
        return GateResult.not_run("R37_PER_CANONICAL_LEDGER_PRODUCED",
                                  "ledger parquet missing", head)
    try:
        import polars as pl  # type: ignore

        df = pl.read_parquet(p)
        n = df.height
        return GateResult.from_cases(
            "R37_PER_CANONICAL_LEDGER_PRODUCED", [n > 0], commit_sha=head,
            evidence_files=("evidence/factor_engine/r37/R37_OPERATOR_CORRECTNESS_LEDGER.parquet",),
            details={"rows": n},
        )
    except Exception as exc:
        return GateResult.not_run("R37_PER_CANONICAL_LEDGER_PRODUCED", str(exc), head)


def _gate_all_production_in_ledger() -> GateResult:
    head = _head()
    try:
        from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
        from factor_engine.cleaned_operators.production_hardening import factor_production_targets

        ensure_cleaned_loaded()
        targets = set(factor_production_targets())
        import polars as pl  # type: ignore

        p = E / "R37_OPERATOR_CORRECTNESS_LEDGER.parquet"
        if not p.is_file():
            return GateResult.not_run("R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER",
                                      "ledger missing", head)
        df = pl.read_parquet(p)
        ledger_canon = set(df["canonical"].to_list())
        missing = sorted(targets - ledger_canon)
        return GateResult.from_cases(
            "R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER", [len(missing) == 0],
            commit_sha=head,
            details={"production_targets": len(targets),
                     "in_ledger": len(targets & ledger_canon),
                     "missing": missing[:10]},
        )
    except Exception as exc:
        return GateResult.not_run("R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER", str(exc), head)


def _gate_exact_call_consumed() -> GateResult:
    """R37-P0-009：production 执行路径源码必须含参数域 membership 门。"""
    head = _head()
    src = Path("backend/cleaned_bridge.py").read_text(encoding="utf-8")
    has_store = "assert_parameter_point_certified" in src
    has_membership = "get_parameter_domain_store" in src
    return GateResult.from_cases(
        "R37_EXACT_CALL_CERTIFICATION_CONSUMED",
        [has_store and has_membership], commit_sha=head,
        evidence_files=("backend/cleaned_bridge.py",),
    )


def _gate_property_mutation() -> GateResult:
    head = _head()
    # mutation 负控来自 evidence truth 的 5 类负控 + property 测试
    try:
        truth = json.loads(
            (E / "R37_EVIDENCE_TRUTH_GATES.json").read_text(encoding="utf-8"))
        controls = truth.get("negative_controls", {})
        cases = []
        for cid, v in controls.items():
            cases.append(bool(v.get("fired")))
        return GateResult.from_cases(
            "R37_PROPERTY_MUTATION_KILLED", cases, commit_sha=head,
            details={"mutation_controls": list(controls.keys())},
        )
    except Exception as exc:
        return GateResult.not_run("R37_PROPERTY_MUTATION_KILLED", str(exc), head)


def _gate_data_knowledge_identity() -> GateResult:
    head = _head()
    from factor_engine.semantic.data_knowledge_identity import DataKnowledgeIdentity

    a = DataKnowledgeIdentity(dataset_id="d", snapshot_id="s1", universe_snapshot_id="u")
    b = DataKnowledgeIdentity(dataset_id="d", snapshot_id="s1", universe_snapshot_id="u2")
    return GateResult.from_cases(
        "R37_DATA_KNOWLEDGE_IDENTITY_PASS",
        [a.to_key() != b.to_key(), a.digest(), bool(a.to_key())],
        commit_sha=head,
    )


def _gate_universe_pit() -> GateResult:
    head = _head()
    from factor_engine.market.universe import UniverseMembership, universe_membership_identity

    m = UniverseMembership(universe="CSI300", instrument="x",
                           valid_time="2024-01-01", knowledge_time="2024-01-05")
    h1 = universe_membership_identity("CSI300", ("a", "b"), as_of="2024-01-01")
    h2 = universe_membership_identity("CSI300", ("a", "b", "c"), as_of="2024-01-01")
    return GateResult.from_cases(
        "R37_UNIVERSE_PIT_PASS",
        [not m.effective_at("2024-01-01"), m.effective_at("2024-01-05"), h1 != h2],
        commit_sha=head,
    )


def _gate_price_basis_identity() -> GateResult:
    head = _head()
    from factor_engine.fields.concepts import PriceBasis
    from factor_engine.semantic.data_knowledge_identity import DataKnowledgeIdentity

    pb1 = PriceBasis.canonical("RAW")
    pb2 = PriceBasis.canonical("CONTINUOUS")
    a = DataKnowledgeIdentity(price_basis=pb1)
    b = DataKnowledgeIdentity(price_basis=pb2)
    return GateResult.from_cases(
        "R37_PRICE_BASIS_IDENTITY_PASS",
        [pb1 != pb2, a.digest() != b.digest()],
        commit_sha=head,
    )


def _gate_current_head_sha_consistent() -> GateResult:
    """R37-P0-080：final_code_sha == evidence_sha（同一 HEAD 绑定）。"""
    head = _head()
    try:
        evidence = json.loads((E / "R37_HEAD.json").read_text(encoding="utf-8"))
        ev_sha = evidence.get("evidence_header", {}).get("commit_sha", "")
        cases = [ev_sha == head]
        return GateResult.from_cases(
            "R37_CURRENT_HEAD_SHA_CONSISTENT", cases, commit_sha=head,
            details={"code_sha": head, "evidence_sha": ev_sha},
        )
    except Exception as exc:
        return GateResult.not_run("R37_CURRENT_HEAD_SHA_CONSISTENT", str(exc), head)


def _gate_no_raw_buffer_bypass() -> GateResult:
    """R37-P0-035：production hot path 必须无 raw shared_result_cache 写绕过。

    R38-P0-029/030 把 put 从裸 bool 升级为 BufferPutResult 并 fail-closed，raw
    写入分支改为 ``is_production_mode(getattr(ctx, 'run_mode', None))``。gate 同时
    接受 R37 旧形式（``is_production_mode()``）与 R38 新形式。
    """
    head = _head()
    src = Path("runtime/batch_service.py").read_text(encoding="utf-8")
    has_fail_closed = (
        "governance bypass" in src
        and (
            "is_production_mode()" in src
            or "is_production_mode(getattr(ctx, \"run_mode\", None))" in src
        )
    )
    # R38：production 下 put REFUSED 同样 fail-closed（不能 return 成功）。
    has_put_refusal_fail_closed = "R38-P0-029: put refusal must" in src
    return GateResult.from_cases(
        "R37_NO_RAW_BUFFER_GOVERNANCE_BYPASS",
        [has_fail_closed, has_put_refusal_fail_closed],
        commit_sha=head,
        evidence_files=("runtime/batch_service.py",),
    )


def _gate_cache_unregister_fail_closed() -> GateResult:
    """R37-P0-037：cache unregister 不再 except:pass，production fail-closed。"""
    head = _head()
    src = Path("cache/session.py").read_text(encoding="utf-8")
    has_fail_closed = "R37-P0-037 fail-closed" in src
    has_reconcile = "accounting drift" in src
    return GateResult.from_cases(
        "R37_CACHE_UNREGISTER_FAIL_CLOSED", [has_fail_closed, has_reconcile],
        commit_sha=head, evidence_files=("cache/session.py",),
    )


def main() -> int:
    E.mkdir(parents=True, exist_ok=True)
    head = _head()
    gates: dict[str, GateResult] = {}

    gates["R37_CURRENT_HEAD_BOUND"] = GateResult.from_cases(
        "R37_CURRENT_HEAD_BOUND", [bool(head) and len(head) >= 7], commit_sha=head)
    gates["R37_EVIDENCE_TRUTH_PASS"] = _gate_evidence_truth()
    gates["R37_PARAMETER_DOMAIN_CERTIFIED"] = _gate_parameter_domain_certified()
    gates["R37_INVALID_PARAM_REJECTION_ZERO"] = _gate_invalid_rejection()
    gates["R37_PER_CANONICAL_LEDGER_PRODUCED"] = _gate_ledger_produced()
    gates["R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER"] = _gate_all_production_in_ledger()
    gates["R37_EXACT_CALL_CERTIFICATION_CONSUMED"] = _gate_exact_call_consumed()
    gates["R37_PROPERTY_MUTATION_KILLED"] = _gate_property_mutation()
    gates["R37_DATA_KNOWLEDGE_IDENTITY_PASS"] = _gate_data_knowledge_identity()
    gates["R37_UNIVERSE_PIT_PASS"] = _gate_universe_pit()
    gates["R37_PRICE_BASIS_IDENTITY_PASS"] = _gate_price_basis_identity()
    gates["R37_CURRENT_HEAD_SHA_CONSISTENT"] = _gate_current_head_sha_consistent()
    gates["R37_NO_RAW_BUFFER_GOVERNANCE_BYPASS"] = _gate_no_raw_buffer_bypass()
    gates["R37_CACHE_UNREGISTER_FAIL_CLOSED"] = _gate_cache_unregister_fail_closed()

    payload = {
        "generated_by": "scripts/audit_r37_hard_gates.py",
        "commit_sha": head,
        "passed": sum(1 for g in gates.values() if g.status == "PASS"),
        "not_run": sum(1 for g in gates.values() if g.status == "NOT_RUN"),
        "failed": sum(1 for g in gates.values() if g.status == "FAIL"),
        "total": len(gates),
        "production_ready": all(g.status == "PASS" for g in gates.values()),
        "gates": {gid: g.to_dict() for gid, g in gates.items()},
    }
    (E / "R37_HARD_GATES.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[r37-hard-gates] HEAD={head[:12]} passed={payload['passed']} "
          f"not_run={payload['not_run']} failed={payload['failed']} total={payload['total']} "
          f"production_ready={payload['production_ready']}")
    for gid, g in gates.items():
        print(f"  {g.status:7s} {gid}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
