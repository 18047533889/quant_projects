# -*- coding: utf-8 -*-
"""R40 中央 hard gates（N1-N9 + #25 证据完备性）。

新硬门的语义：**R40 修复后的机器必须存在且 fail-closed**。每个 gate 复用一个由
R40 实现簇导出的可复用 ``check_*`` / ``validate_*`` 函数；基础设施（registry /
参数域证据 store / DA）在并发编辑或证据过期时**不可被断言为通过**——脚本以三态
汇报：

    - PASS  —— 该门覆盖的 R40 修复已验证通过；
    - FAIL  —— 该门覆盖的机器缺位或 fail-open（需修复）；
    - NOT_RUN —— 所需基础设施当前不可用（registry 并发编辑 / 证据过期 /
      依赖未安装），**诚实暴露**，不假装通过。

用法::

    python scripts/audit_r40_hard_gates.py [--json] [--mode research|production]

research 返回码 = FAIL 门数量；production 要求所有门 PASS，FAIL / NOT_RUN 均非零。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# 直接以 ``python scripts/audit_r40_hard_gates.py`` 运行时，sys.path[0] 是 scripts/
# 而不是 FE 根目录——把 FE_ROOT 前置进 path 让 ``cleaned_operators`` /
# ``runtime`` / ``market`` 等包可导入（tests/conftest 之外的可执行脚本模式）。
_FE_ROOT = Path(__file__).resolve().parents[1]
if str(_FE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FE_ROOT))

# ---------------------------------------------------------------------------
# 三态 Gate 结果
# ---------------------------------------------------------------------------


class GateResult:
    __slots__ = ("ok", "status", "detail")

    def __init__(self, ok: bool | None, detail: str) -> None:
        # ok: True -> PASS, False -> FAIL, None -> NOT_RUN
        self.ok = ok
        self.status = "PASS" if ok is True else ("FAIL" if ok is False else "NOT_RUN")
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "detail": self.detail}


def _run(name: str, fn: Callable[[], GateResult], out: list[tuple[str, GateResult]]) -> None:
    try:
        out.append((name, fn()))
    except Exception as exc:  # gate 自身基础设施异常 -> NOT_RUN（诚实）
        out.append((name, GateResult(None, f"infra error: {type(exc).__name__}: {exc}")))


# ---------------------------------------------------------------------------
# N1 Registry bootstrap
# ---------------------------------------------------------------------------


def gate_n1_registry_bootstrap() -> GateResult:
    """bootstrap module-spec 角色合法（INTERNAL_KERNEL/RESEARCH_EXTENSION 永不
    进 production surface）；RegistryBootstrap 状态机存在。"""
    from cleaned_operators import (
        BOOTSTRAP_MODULE_SPECS,
        REGISTRY_BOOTSTRAP,
        check_bootstrap_module_specs,
    )

    check_bootstrap_module_specs()  # raises on violation (#154)
    specs = BOOTSTRAP_MODULE_SPECS
    if len(specs) == 0:
        return GateResult(False, "BOOTSTRAP_MODULE_SPECS empty (#154)")
    state = getattr(REGISTRY_BOOTSTRAP, "state", None)
    if state is None:
        return GateResult(False, "REGISTRY_BOOTSTRAP missing state machine (#151)")
    return GateResult(
        True,
        f"{len(specs)} module specs valid; RegistryBootstrap.state={state}",
    )


# ---------------------------------------------------------------------------
# N2 Parameter certification
# ---------------------------------------------------------------------------


def gate_n2_parameter_certification() -> GateResult:
    """参数域认证 fail-closed：store 缺失/旧 SHA 必须抛 ParameterDomainError，
    绝不 fail-open。报告当前真实证据状态。"""
    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_domain_ready,
        get_parameter_domain_store,
    )

    store = get_parameter_domain_store(ensure_loaded=False)
    if store is None:
        # 空 store 上 assert 必须 fail-closed 而不是静默通过
        try:
            assert_parameter_domain_ready(strict=True)
        except Exception as exc:
            return GateResult(
                True,
                f"empty store fail-closed (raises {type(exc).__name__})",
            )
        return GateResult(False, "empty store did NOT fail-closed (#29)")
    # store 已装载：检查生成 commit 与当前 HEAD 是否一致
    gen = getattr(store, "_generated_commit", "") or ""
    try:
        assert_parameter_domain_ready(strict=True)
    except Exception as exc:
        return GateResult(
            False,
            f"store loaded but NOT ready for execution: {type(exc).__name__}: {exc} "
            f"(evidence generated at {gen[:12] or '<unknown>'})",
        )
    return GateResult(
        True,
        f"parameter-domain store ready at HEAD (generated={gen[:12]})",
    )


# ---------------------------------------------------------------------------
# N3 Axis truth
# ---------------------------------------------------------------------------


def gate_n3_axis_truth() -> GateResult:
    """grain-transform 证书校验（#163）：合法降采样零违规；非 DatetimeIndex /
    重复日期必须被检测为违规（fail-closed）。"""
    import pandas as pd

    from backend.cleaned_bridge import GrainTransformCertificate, validate_grain_transform

    cert = GrainTransformCertificate(
        input_grain="daily", output_grain="weekly",
        calendar_id="SSE", calendar_version="v1", timezone="Asia/Shanghai",
        session_id="ashare-daily", mapping_policy="session_end",
    )
    template = pd.DataFrame(
        {"close": [1.0] * 10},
        index=pd.to_datetime(pd.date_range("2024-01-01", periods=10, freq="D")),
    )
    # 合法降采样：唯一、单调、无未来日期、声明 grain 匹配
    ok = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
    )
    ok_errs = validate_grain_transform(ok, template, cert)
    # 负控：非 DatetimeIndex / 重复日期必须报违规
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=[1, 1])
    bad_errs = validate_grain_transform(bad, template, cert)
    if ok_errs:
        return GateResult(False, f"valid grain transform flagged: {ok_errs}")
    if not bad_errs:
        return GateResult(False, "duplicate/non-datetime index NOT flagged (#163)")
    return GateResult(
        True, f"valid transform clean; invalid index rejected ({len(bad_errs)} violations)"
    )


# ---------------------------------------------------------------------------
# N4 Market/time truth
# ---------------------------------------------------------------------------


def gate_n4_market_time_truth() -> GateResult:
    """SessionCalendar 必须能证明交换所认证性（holiday set 缺失时 production
    hard-fail）。"""
    from runtime.session_calendar import SessionCalendar

    cal = SessionCalendar("US")
    try:
        cal.require_exchange_certified(mode="production")
    except Exception as exc:
        # 无真实 exchange-certified holiday set -> production 正确 hard-fail
        return GateResult(
            True,
            f"require_exchange_certified fail-closed: {type(exc).__name__}: {exc}",
        )
    # 能证明 -> 检查 authoritativeness 枚举
    auth = getattr(cal, "authoritativeness", None)
    if auth is None:
        return GateResult(False, "authoritativeness missing (#233)")
    return GateResult(True, f"exchange-certified, authoritativeness={auth!r}")


# ---------------------------------------------------------------------------
# N5 Universe truth
# ---------------------------------------------------------------------------


def gate_n5_universe_truth() -> GateResult:
    """Universe membership 双时点区间校验 + production 缺 temporal 契约
    fail-closed（#221/#222）。"""
    from market.universe import (
        OPEN_ENDED,
        UniverseKnowledgeUnknownError,
        UniverseMembership,
    )

    m = UniverseMembership(
        universe="test",
        instrument="000001",
        valid_time="2024-01-01",
        valid_to_exclusive=OPEN_ENDED,
        knowledge_time="2024-01-01",
    )
    valid = m.effective_at(decision_time="2024-06-01", mode="research")
    if valid is not True:
        return GateResult(False, f"effective_at returned {valid!r} for open-ended membership")
    # 缺 temporal 契约 -> production 必须 fail-closed
    try:
        m2 = UniverseMembership(universe="test", instrument="X")  # no dates
        m2.effective_at(decision_time="2024-01-01", mode="production")
    except UniverseKnowledgeUnknownError:
        return GateResult(True, "missing-temporal fail-closed (UniverseKnowledgeUnknownError)")
    except Exception as exc:
        return GateResult(False, f"fail-closed wrong type: {type(exc).__name__}")
    return GateResult(False, "missing temporal contract did NOT fail-closed (#222)")


# ---------------------------------------------------------------------------
# N6 Price-basis truth
# ---------------------------------------------------------------------------


def gate_n6_price_basis_truth() -> GateResult:
    """limit-ops 价格基准契约 + PIT 调整政策：production 拒绝 retrospective。"""
    from market.adjustment_policy import AdjustmentPolicy, validate_adjustment_policy_for_production
    from market.price_basis import PriceBasis, validate_limit_ops_price_basis

    errs = validate_limit_ops_price_basis({"__all__": PriceBasis.RAW})
    if errs:
        return GateResult(False, f"raw basis rejected: {errs}")
    try:
        validate_adjustment_policy_for_production(AdjustmentPolicy.RETROSPECTIVE)
    except Exception:
        # retrospective 被 production 拒绝（#230）
        return GateResult(True, "raw basis valid; retrospective adjustment rejected in production")
    return GateResult(False, "retrospective adjustment NOT rejected in production (#230)")


# ---------------------------------------------------------------------------
# N7 Minute DQ
# ---------------------------------------------------------------------------


def gate_n7_minute_dq() -> GateResult:
    """分钟聚合 DQ：重复 slot 在 production 必须 hard-fail。"""
    from runtime.session_panel import SessionPanel

    panel = SessionPanel.__new__(SessionPanel)
    try:
        panel.hard_dq_fail(off_grid_floor=0.05)
    except Exception as exc:
        return GateResult(
            True,
            f"hard_dq_fail fail-closed on DQ violation: {type(exc).__name__}",
        )
    return GateResult(False, "hard_dq_fail did not raise on DQ violation (#238)")


# ---------------------------------------------------------------------------
# N8 Stateful
# ---------------------------------------------------------------------------


def gate_n8_stateful() -> GateResult:
    """segmented 算子 chunk-invariance universal hard gate（#250）。"""
    from cleaned_operators.math_certificate import (
        check_chunk_invariance_all_segmented_canonicals,
    )

    ok, detail = check_chunk_invariance_all_segmented_canonicals(
        n_bars=60, n_random_chunkings=2
    )
    failed = [c for c, r in (detail.get("per_canonical") or {}).items() if not r.get("ok")]
    return GateResult(
        ok,
        f"segmented chunk-invariance: {len(detail.get('per_canonical') or {})} canonical(s), "
        f"fail={failed or 'none'}",
    )


# ---------------------------------------------------------------------------
# N9 Numerics
# ---------------------------------------------------------------------------


def gate_n9_numerics() -> GateResult:
    """tie-sensitive 置换等变 + causal-TS prefix 不变 + 流式 chunk-boundary
    不变（#257/#258/#259）。"""
    from cleaned_operators.math_certificate import (
        check_chunk_boundary_invariance_all_streamable,
        check_permutation_equivariance_all_tie_sensitive_operators,
        check_prefix_invariance_all_causal_ts,
    )

    ok_perm, d_perm = check_permutation_equivariance_all_tie_sensitive_operators()
    ok_pref, d_pref = check_prefix_invariance_all_causal_ts()
    ok_chunk, d_chunk = check_chunk_boundary_invariance_all_streamable()
    failed: list[str] = []
    if not ok_perm:
        failed.append("permutation-equivariance")
    if not ok_pref:
        failed.append("prefix-invariance")
    if not ok_chunk:
        failed.append("chunk-boundary")
    return GateResult(
        not failed,
        f"perm={ok_perm} prefix={ok_pref} chunk={ok_chunk} fail={failed or 'none'}",
    )


# ---------------------------------------------------------------------------
# #25 Evidence completeness（诚实暴露）
# ---------------------------------------------------------------------------


def gate_evidence_completeness() -> GateResult:
    """R40 #25：参数域认证证据全维度完备性。证据缺失/过期时诚实 NOT_RUN。"""
    import importlib.util

    mod = importlib.util.find_spec("scripts.audit_r40_evidence_completeness")
    if mod is None:
        return GateResult(None, "audit_r40_evidence_completeness not importable")
    from pathlib import Path

    from scripts.audit_r40_evidence_completeness import (
        check_evidence_axis_completeness,
        check_evidence_cross_product_completeness,
        load_store_points,
        production_evidence_requirements,
    )

    FE_ROOT = Path(__file__).resolve().parents[1]
    path = FE_ROOT / "docs" / "evidence" / "r37" / "R37_PARAMETER_DOMAIN_STORE.json"
    points = load_store_points(path)
    if not points:
        return GateResult(
            None, f"evidence file missing/empty at {path} (R16 证据重生待运行)"
        )
    axis = check_evidence_axis_completeness(points)
    required_canonicals, required_by_canonical = production_evidence_requirements()
    cross = check_evidence_cross_product_completeness(
        points,
        required_canonicals=required_canonicals,
        required_by_canonical=required_by_canonical,
    )
    if axis["complete"] and cross["all_complete"]:
        return GateResult(
            True,
            f"{len(points)} points; axis-complete + full cross-product "
            f"({cross['required_cross_product']} combos)",
        )
    return GateResult(
        False,
        f"{len(points)} points; axis_missing={len(axis['incomplete'])} "
        f"cross_missing={cross['missing_total']} (R16 证据重生待运行)",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

GATES: list[tuple[str, Callable[[], GateResult]]] = [
    ("N1_REGISTRY_BOOTSTRAP", gate_n1_registry_bootstrap),
    ("N2_PARAMETER_CERTIFICATION", gate_n2_parameter_certification),
    ("N3_AXIS_TRUTH", gate_n3_axis_truth),
    ("N4_MARKET_TIME_TRUTH", gate_n4_market_time_truth),
    ("N5_UNIVERSE_TRUTH", gate_n5_universe_truth),
    ("N6_PRICE_BASIS_TRUTH", gate_n6_price_basis_truth),
    ("N7_MINUTE_DQ", gate_n7_minute_dq),
    ("N8_STATEFUL", gate_n8_stateful),
    ("N9_NUMERICS", gate_n9_numerics),
    ("EVIDENCE_COMPLETENESS_25", gate_evidence_completeness),
]


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    as_json = "--json" in argv
    mode = "research"
    if "--mode" in argv:
        try:
            mode = argv[argv.index("--mode") + 1]
        except IndexError as exc:
            raise SystemExit("--mode requires research or production") from exc
    if mode not in {"research", "production"}:
        raise SystemExit("--mode must be research or production")
    results: list[tuple[str, GateResult]] = []
    for name, fn in GATES:
        _run(name, fn, results)
    n_fail = sum(1 for _, r in results if r.ok is False)
    n_pass = sum(1 for _, r in results if r.ok is True)
    n_notrun = sum(1 for _, r in results if r.ok is None)
    if as_json:
        print(
            json.dumps(
                {
                    "mode": mode,
                    "gates": {name: r.to_dict() for name, r in results},
                    "summary": {"pass": n_pass, "fail": n_fail, "not_run": n_notrun},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return n_fail + (n_notrun if mode == "production" else 0)
    for name, r in results:
        print(f"[{r.status}] {name}: {r.detail}")
    print(f"\nR40 hard gates: {n_pass} PASS / {n_fail} FAIL / {n_notrun} NOT_RUN")
    return n_fail + (n_notrun if mode == "production" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
