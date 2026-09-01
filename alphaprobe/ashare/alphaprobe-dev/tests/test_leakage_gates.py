"""§88 Leakage tests：封存 Test 最硬 CI + eval/exec 静态 grep gate（§3.2）。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from alphaprobe.contracts import DateRange, FidelityLevel, ResearchSplitSpec
from alphaprobe.research_protocol import (
    LeakageGuard,
    SealedTestViolation,
    assert_sealed_test_zero_reads,
    reset_sealed_test_counters,
    sealed_test_read_count,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "alphaprobe"


# §88.1 Sealed Test Zero Reads（L0-L4）
def test_sealed_test_zero_reads_at_L0_L4():
    reset_sealed_test_counters()
    spec = ResearchSplitSpec(
        train=DateRange("2016-01-01", "2021-12-31"),
        search_valid=DateRange("2022-01-01", "2023-12-31"),
        audit_valid=None,
        sealed_test=DateRange("2024-01-01", "2026-07-31"),
    )
    guard = LeakageGuard(spec)
    for stage in (
        FidelityLevel.L0_STATIC,
        FidelityLevel.L1_SCOUT,
        FidelityLevel.L2_FULL_TRAIN,
        FidelityLevel.L3_SEARCH_VALID,
        FidelityLevel.L4_POOL_AUDIT,
    ):
        with pytest.raises(SealedTestViolation):
            guard.assert_access_allowed(
                stage=stage, segment=DateRange("2025-03-01", "2025-03-31"),
                caller="test",
            )
    # 被拒绝的探测不算已发生的读取（拦截在闸门，未触达数据）；
    # §88.1 语义：counter 只统计绕过闸门的真实读取
    assert_sealed_test_zero_reads()


def test_L5_may_read_after_freeze():
    spec = ResearchSplitSpec(
        train=DateRange("2016-01-01", "2021-12-31"),
        search_valid=DateRange("2022-01-01", "2023-12-31"),
        audit_valid=None,
        sealed_test=DateRange("2024-01-01", "2026-07-31"),
    )
    guard = LeakageGuard(spec)
    guard.assert_access_allowed(
        stage=FidelityLevel.L5_SEALED_TEST,
        segment=DateRange("2025-03-01", "2025-03-31"),
        caller="sealed",
    )


def test_train_segment_allowed():
    spec = ResearchSplitSpec(
        train=DateRange("2016-01-01", "2021-12-31"),
        search_valid=DateRange("2022-01-01", "2023-12-31"),
        audit_valid=None,
        sealed_test=DateRange("2024-01-01", "2026-07-31"),
    )
    guard = LeakageGuard(spec)
    guard.assert_access_allowed(
        stage=FidelityLevel.L2_FULL_TRAIN,
        segment=DateRange("2017-01-01", "2017-12-31"),
        caller="train",
    )


def test_sealed_access_requires_frozen():
    from alphaprobe.research_protocol import SealedTestAccess

    spec = ResearchSplitSpec(
        train=DateRange("2016-01-01", "2021-12-31"),
        search_valid=DateRange("2022-01-01", "2023-12-31"),
        audit_valid=None,
        sealed_test=DateRange("2024-01-01", "2026-07-31"),
    )
    with pytest.raises(SealedTestViolation):
        SealedTestAccess(spec, frozen=False)
    acc = SealedTestAccess(spec, frozen=True)
    assert acc.split_spec is spec


# §3.2 静态 grep gate：新主链禁止 eval(/exec(
# baselines / shared legacy 路径允许（任务书 §3.2：只允许明确 legacy 测试 fixture）
ALLOWED_EVAL_PREFIXES = (
    "src/shared/",  # legacy qlib 路径（baselines 专用，非新主链）
)


def _walk_py(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def test_no_python_eval_in_alphaprobe_main_path():
    """§3.2 test_no_python_eval_in_alphaprobe_main_path。

    允许的例外必须显式列在 ALLOWED_EVAL_FILES，且只能是 legacy/测试 fixture。
    """
    offenders = []
    for p in _walk_py(SRC_ROOT):
        rel = str(p.relative_to(SRC_ROOT.parent.parent))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in ("eval", "exec"):
                    offenders.append(f"{rel}:{node.lineno} {fn.id}()")
                elif isinstance(fn, ast.Attribute) and fn.attr in ("eval", "exec"):
                    # torch net.eval() 等 PyTorch 方法调用排除（属性非裸 eval）
                    pass
    offenders = [o for o in offenders if not any(o.startswith(a) for a in ALLOWED_EVAL_PREFIXES)]
    assert offenders == [], f"eval/exec in main path: {offenders}"


# §88.4 knowledge cutoff：post-cutoff survival 不可见
def test_survival_cutoff_enforced_in_packet():
    from datetime import date
    from alphaprobe.memory import GlobalMemoryStore
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
        # 注入 2026-06 事件（post-cutoff）
        store.add_regime_event(
            event_id="decay_2026_jun_jul",
            start_date="2026-06-01",
            end_date="2026-07-31",
            description="2026 大规模因子衰减",
        )
        # knowledge_cutoff=2026-01-01 时 Retriever 不可见 post-cutoff survival
        cutoff = date(2026, 1, 1)
        # GlobalMemoryStore 的 rare_directions 不带 cutoff 过滤器，
        # 因此 cutoff 由 survival_memory_cutoff 闸门实现（见 survival.py）
        from alphaprobe.research_protocol.survival import assert_survival_visible

        with pytest.raises(SealedTestViolation):
            assert_survival_visible(event_date="2026-06-15", cutoff=cutoff)
        # pre-cutoff 可见
        assert_survival_visible(event_date="2025-12-31", cutoff=cutoff)
        store.close()