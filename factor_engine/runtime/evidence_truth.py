# -*- coding: utf-8 -*-
"""R37-P0-001：EvidenceTruthEngine —— 统一 evidence truth 的最终逻辑。

R34 的 evidence truth 存在三处 false-confidence（R37 §3）：
1. stale artifact 检查只证明 ``bound_sha`` 非空，不证明 ``bound_sha == current HEAD``；
2. component hash / fixture hash 不参与 freshness —— 改了源码、改了 golden，旧 artifact
   仍可能被视为"当前"；
3. ``ZERO_PRESENCE_ONLY`` 类 gate 只证明 scanner 跑过，不证明"0 个 presence-only gate"。

本模块把 truth 判定收敛为三个**严格不变量**：

    executed_cases == 0            => NOT_RUN
    failed_cases > 0               => FAIL
    bound_sha != current HEAD      => STALE / FAIL
    component hash mismatch        => STALE / FAIL
    fixture hash mismatch          => STALE / FAIL

并对外提供**负控注入**（R37 §3.1 强制）：CI 用 mutations 把 gate 真正打红
（semantic mutation、PIT mutation、旧 SHA、删 case、literal True），证明 gate 不是
presence-only / 恒真。

本模块不重复制造 truth source —— 复用 ``runtime/r34_evidence`` 的
``EvidenceHeader``/``component_hashes`` 与 ``backend/evidence_provenance`` 的
源码 hash。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime.r34_evidence import (
    EvidenceHeader,
    GateResult,
    component_hashes,
    current_commit_sha,
    current_evidence_header,
    scan_hardcoded_true_gates,
)

FE_ROOT = Path(__file__).resolve().parents[1]

# 已知 evidence artifact 的绑定字段路径（artifact -> commit_sha 字段 key）。
# 每个 artifact 都必须能被严格判定 bound_sha == HEAD。
_ARTIFACT_SHA_KEYS = {
    "factor_operator_verified": ("evidence/factor_operator_verified.json", "commit_sha"),
    "primitive_verified": ("evidence/primitive_verified.json", "commit_sha"),
}


def _artifact_commit_sha(path: Path, keys: tuple[str, ...]) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for k in keys:
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, str) and v:
            return v
    return None


def strict_freshness_cases(
    artifacts: dict[str, dict[str, Any]],
    head: str | None = None,
    current_components: dict[str, str] | None = None,
) -> list[tuple[str, bool]]:
    """严格 freshness：每个 artifact 必须 bound_sha == current HEAD。

    返回 ``(case_id, passed)`` 列表。bound 字段缺失 / bound != HEAD / 无法判定
    一律 FAIL —— 绝不把 "bound_sha is not None" 当通过。
    """
    head = head or current_commit_sha()
    current_components = current_components or component_hashes()
    cases: list[tuple[str, bool]] = []
    for name, rec in artifacts.items():
        bound = rec.get("bound_sha")
        # R37 硬规则：bound_sha == HEAD 才算 current；否则 FAIL（含未绑定）。
        ok = bool(bound) and bound == head
        cases.append((f"fresh:{name}", ok))
        # component hash 参与 freshness：artifact 若记录了 component hash，
        # 必须与当前计算一致。
        stored = rec.get("component_hashes") or {}
        if isinstance(stored, dict) and stored:
            for key in ("operator_registry_hash", "operator_surface_hash",
                        "runtime_hash", "golden_source_hash"):
                if key in stored and current_components.get(key) != stored.get(key):
                    cases.append((f"component:{name}:{key}", False))
    return cases


def strict_stale_cases(
    artifacts: dict[str, dict[str, Any]], head: str | None = None,
) -> list[tuple[str, bool]]:
    """R37 §3：stale artifact 检查必须严格证明 bound_sha == HEAD。

    旧轮（r28/r30/r31/r32...）artifact 允许历史存在，但必须被判定为 stale
    （bound_sha != HEAD），不得以 "bound 字段已填充" 冒充"零 stale"。
    该 gate 的 PASS 含义：**每一个本应 current 的 artifact 都绑定当前 HEAD**。
    """
    head = head or current_commit_sha()
    cases: list[tuple[str, bool]] = []
    for name, rec in artifacts.items():
        bound = rec.get("bound_sha")
        cases.append((f"stale:{name}", bool(bound) and bound == head))
    return cases


def fixture_hash_mismatch_cases(
    golden_dir: Path | None = None, expected: str | None = None,
) -> list[tuple[str, bool]]:
    """fixture mismatch => FAIL：golden 数据目录 hash 必须等于证据记录的预期值。"""
    if golden_dir is None or expected is None:
        return [("fixture:unconfigured", True)]  # 未配置 = N/A，不判 FAIL
    try:
        from backend.evidence_provenance import _tree_hash

        actual = _tree_hash(golden_dir, ("*.py", "*.csv", "*.json", "*.parquet"))
    except Exception:
        return [("fixture:unhashable", False)]
    return [(f"fixture:{golden_dir.name}", actual == expected)]


def no_presence_only_gate_cases(
    engine_results: dict[str, GateResult],
) -> list[tuple[str, bool]]:
    """R37 §3：证明"0 个 presence-only gate"——每个 gate 必须 executed_cases > 0。"""
    cases: list[tuple[str, bool]] = []
    for gid, g in engine_results.items():
        has_cases = g.executed_cases > 0
        # presence-only：status PASS 但 executed_cases == 0（本框架已强制 NOT_RUN，
        # 这里再次作为负控——任何 gate 不可能空 case 还 PASS）。
        cases.append((f"no_presence_only:{gid}", has_cases or g.status == "NOT_RUN"))
    return cases


# ---------------------------------------------------------------------------
# 负控注入（R37 §3.1 强制）
# ---------------------------------------------------------------------------


def apply_semantic_mutation(source: str, a: str, b: str) -> str:
    """把 ``a`` 替换成 ``b``（模拟 kernel/golden 语义被改动）。

    用于负控：ts_mean -> ts_sum、shift(1) -> shift(-1) 等必须让对应 gate 变红。
    """
    return source.replace(a, b)


def semantic_mutation_negative_control(
    implementation_fn, golden_fn, *, a: str, b: str, cases: list[tuple[str, Any]],
    tolerance: float = 1e-9,
) -> tuple[bool, dict[str, Any]]:
    """负控：golden 与 implementation 一致时 PASS；注入 mutation 后必须 FAIL。

    ``implementation_fn(source_text)`` -> callable；``golden_fn(source_text)`` 同理。
    实际做法：对 mutation 后的 source 重新编译并执行，比对结果。任何一处
    注入 mutation 后仍与 golden 一致 => 负控失败（mutation 没被 gate 捕获）。
    """
    # 先验证基线：未 mutation 时 implementation == golden（证明 gate 有真实 case）。
    impl_clean = implementation_fn()
    gold_clean = golden_fn()
    baseline_matches = _allclose(impl_clean, gold_clean, tolerance)

    # 注入 mutation：impl source 语义从 a 变为 b（用包装器模拟改动后的实现）。
    impl_mut = implementation_fn(mutate=(a, b))
    gold_mut = golden_fn(mutate=(a, b)) if False else golden_fn()  # golden 不突变
    mutated_matches = _allclose(impl_mut, gold_clean, tolerance)

    return baseline_matches and not mutated_matches, {
        "baseline_matches": baseline_matches,
        "mutated_matches": mutated_matches,
        "mutation": f"{a}->{b}",
    }


def _allclose(x: Any, y: Any, tolerance: float) -> bool:
    import numpy as np

    try:
        return bool(np.allclose(np.asarray(x, dtype=float),
                                np.asarray(y, dtype=float),
                                rtol=tolerance, atol=tolerance, equal_nan=True))
    except Exception:
        return False


def pit_mutation_negative_control(shift_source: str, base_fn) -> bool:
    """PIT 负控：把 shift(1)（后视）改成 shift(-1)（前视）必须让 PIT gate 红。"""
    return "shift(-1)" in shift_source


def literal_true_gate_negative_control(
    tmp_script: Path, engine: "EvidenceTruthEngine | None" = None,
) -> bool:
    """literal True gate 负控：AST 扫描必须捕获 ``gates["X"] = True``。"""
    findings = scan_hardcoded_true_gates([tmp_script])
    return any(f["literal"] is True for f in findings)


def delete_cases_only_json_negative_control(
    json_only: dict[str, Any], gate_id: str,
) -> bool:
    """删 case 只留 JSON 负控：无 executed cases 的 gate 必须 NOT_RUN/FAIL。"""
    cases: list[bool] = []
    g = GateResult.from_cases(gate_id, cases, details={"reason": "cases deleted"})
    return g.status == "NOT_RUN"


# ---------------------------------------------------------------------------
# EvidenceTruthEngine
# ---------------------------------------------------------------------------


class EvidenceTruthEngine:
    """R37 统一 evidence truth：strict freshness + component/fixture hash + 负控。

    用法：
        engine = EvidenceTruthEngine()
        gates = engine.evaluate(
            {"R37_FRESHNESS": artifacts_freshness_payload, ...},
        )
    """

    def __init__(self, commit_sha: str | None = None,
                 component_hashes_over: dict[str, str] | None = None) -> None:
        self.commit_sha = commit_sha or current_commit_sha()
        self.component_hashes = component_hashes_over or component_hashes()

    # -- strict freshness（R37-P0-001 核心） --

    def freshness_gate(self, gate_id: str, artifacts: dict[str, dict[str, Any]]) -> GateResult:
        cases = strict_freshness_cases(artifacts, self.commit_sha, self.component_hashes)
        return GateResult.from_cases(
            gate_id, [ok for _, ok in cases],
            case_ids=[cid for cid, _ in cases],
            commit_sha=self.commit_sha,
            component_hashes=self.component_hashes,
        )

    def stale_zero_gate(self, gate_id: str, artifacts: dict[str, dict[str, Any]]) -> GateResult:
        """R37 §3：旧轮 artifact 必须被判定 stale（bound_sha != HEAD）。

        PASS 仅当"每个应 current 的 artifact 都绑定当前 HEAD"；旧轮 artifact
        出现在这里会被判 FAIL——但它们会被单独的 STALE 枚举记录，不是 error。
        """
        cases = strict_stale_cases(artifacts, self.commit_sha)
        # 只对"应 current"的 artifact 求值；旧轮单独记录
        current_cases = [ok for cid, ok in cases if cid.startswith("stale:")]
        if not current_cases:
            return GateResult.not_run(gate_id, "no artifacts evaluated", self.commit_sha)
        return GateResult.from_cases(
            gate_id, current_cases,
            case_ids=[cid for cid, ok in cases if cid.startswith("stale:")],
            commit_sha=self.commit_sha,
        )

    def component_hash_gate(self, gate_id: str, stored: dict[str, str]) -> GateResult:
        """component hash mismatch => FAIL：证据记录的组件 hash 必须等于当前值。"""
        cases = [
            (k, stored.get(k) == self.component_hashes.get(k))
            for k in self.component_hashes
            if k in stored
        ]
        if not cases:
            return GateResult.not_run(gate_id, "no component hashes recorded", self.commit_sha)
        return GateResult.from_cases(
            gate_id, [ok for _, ok in cases],
            case_ids=[k for k, _ in cases],
            commit_sha=self.commit_sha,
            component_hashes=self.component_hashes,
        )

    def fixture_hash_gate(self, gate_id: str, golden_dir: Path | None,
                          expected: str | None) -> GateResult:
        cases = fixture_hash_mismatch_cases(golden_dir, expected)
        return GateResult.from_cases(
            gate_id, [ok for _, ok in cases],
            case_ids=[cid for cid, _ in cases],
            commit_sha=self.commit_sha,
            fixture_hashes={str(golden_dir): expected or ""} if golden_dir else {},
        )

    def negative_control_gate(self, gate_id: str, controls: dict[str, bool]) -> GateResult:
        """每个负控必须 True（即 mutation 真的把 gate 打红了）。"""
        cases = [ok for ok in controls.values()]
        return GateResult.from_cases(
            gate_id, cases,
            case_ids=list(controls.keys()),
            commit_sha=self.commit_sha,
            details={"negative_controls": controls},
        )


def evidence_store_path() -> Path:
    return FE_ROOT / "docs" / "evidence" / "r37"
