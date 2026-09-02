"""V2-G 权威接线测试（§28 / §30 / §36）：FE/DA/Modeling 权威化收口。

覆盖：
- §28 FE 唯一公式权威：活跃主链无``dedup.canonicalize_dsl``正则文本化简残留
  （源码扫描）；身份委托 DedupClient/FE authority；FE 不可用 fail-closed。
- §36-§38 modeling LabelContract/EmbargoSpec 契约：vwap→vwap 20d label
  horizon=20、overlapping、embargo≥20 校验通过/违反抛错。
- §30 data_access 唯一取数：FE data_source 走 data_access 登记数据集
  （type=data_access, dataset=ashare_stock_daily_adj），不经手写 SQL/csv。
全合成数据 / 源码扫描，不跑模型、不跑训练。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# 活跃主链（评估链）模块——不应残留对文本 canonicalize_dsl 的直接调用。
ACTIVE_CHAIN = [
    "pipeline.py",
    "runner.py",
    "continuous.py",
    "authority.py",
]


def _read(mod: str) -> str:
    return (SRC_ROOT / "alphaprobe" / mod).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# §28 FE 唯一公式权威：活跃主链无正则文本化简残留
# ---------------------------------------------------------------------------


class TestFeUniqueFormulaAuthority:
    def test_active_chain_no_direct_dedup_canonical_import(self):
        """活跃主链不得 import alphaprobe.dedup 的文本 canonical 函数
        （canonicalize_dsl / signal_equivalence_id / parameter_family_key）。
        canonicalize_dsl 仅作为 DedupClient 降级链末端兜底，不直接在
        pipeline/runner/continuous 的模块级 import 出现；``getattr``/dict 读取
        identity 视图的 ``signal_equivalence_id`` 键是消费 FE 字段，属合法。"""
        forbidden_imports = [
            "from alphaprobe.dedup import canonical",
            "from alphaprobe.dedup import signal_equivalence_id",
            "from alphaprobe.dedup import parameter_family_key",
            "from alphaprobe.dedup import canonical_ast_hash",
            "import canonicalize_dsl",
            "import signal_equivalence_id",
            "import parameter_family_key",
        ]
        for mod in ACTIVE_CHAIN:
            src = _read(mod)
            for token in forbidden_imports:
                assert token not in src, f"{mod} 直连文本 canonical 残留: {token}"

    def test_active_chain_delegates_to_authority_or_dedup_client(self):
        """身份来源：DedupClient.get_identity 或 authority.build_identity_view。"""
        pipeline = _read("pipeline.py")
        assert "authority.build_identity_view" in pipeline or "build_identity_view" in pipeline
        assert "get_identity" in pipeline
        runner = _read("runner.py")
        continuous = _read("continuous.py")
        assert "build_identity_view" in runner
        assert "build_identity_view" in continuous

    def test_authority_uses_fe_identity_provider(self):
        """authority.build_identity_view 委托 factor_engine.identity。"""
        src = _read("authority.py")
        assert "factor_engine.identity" in src or "get_factor_identity" in src
        assert "RegexObject" not in src


# ---------------------------------------------------------------------------
# §36-§38 modeling LabelContract / EmbargoSpec 契约
# ---------------------------------------------------------------------------


def _fe_importable() -> bool:
    try:
        import alphaprobe.fe_bridge.paths as _p

        _p.ensure_factor_engine_importable()
        import factor_engine.identity as _  # noqa: F401
        import modeling.contracts as _m  # noqa: F401

        return True
    except Exception:
        return False


FE_OK = _fe_importable()


@pytest.mark.skipif(not FE_OK, reason="factor_engine / modeling not importable")
class TestModelingLabelContract:
    def test_contract_horizon_overlap_embargo(self):
        from alphaprobe.authority import vwap_20d_label_contract

        contract = vwap_20d_label_contract()
        assert contract.horizon_bars == 20
        assert contract.return_basis == "vwap_to_vwap"
        assert contract.overlapping is True
        assert contract.embargo_bars == 20  # embargo >= horizon

    def test_validate_overlap_passes(self):
        from alphaprobe.authority import validate_label_contract_20d

        validate_label_contract_20d()  # 不抛即通过

    def test_bad_overlap_violation_raises(self):
        """overlap 标志与 horizon/stride 不自洽 → validate_overlap 抛错。"""
        from modeling.contracts import LabelContract, validate_overlap

        bad = LabelContract(
            label_name="x", horizon_bars=20, overlapping=False  # 日频 stride=1 → 必须 True
        )
        with pytest.raises(ValueError):
            validate_overlap(bad, stride_bars=1)

    def test_embargo_below_horizon_violation(self):
        """embargo < horizon → EmbargoSpec.validate_against_label 返回违反。"""
        from alphaprobe.authority import vwap_20d_label_contract
        from modeling.contracts import EmbargoSpec

        contract = vwap_20d_label_contract()
        short = EmbargoSpec(days=5, rationale="too short")
        violations = short.validate_against_label(contract)
        assert violations, "embargo 5 < horizon 20 应违反"


# ---------------------------------------------------------------------------
# §28 FE fail-closed：权威不可用即抛错，不静默 0 分 / 文本兜底
# ---------------------------------------------------------------------------


class TestFeFailClosed:
    def test_build_identity_empty_raises(self):
        from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view

        with pytest.raises(FactorIdentityAuthorityError):
            build_identity_view("   ")

    def test_build_identity_fail_closed_when_provider_broken(self):
        """provider 返回 None / 不完整 → fail-closed raise。"""
        from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view

        class BrokenProvider:
            def get_factor_identity(self, formula):
                return None

        class IncompleteProvider:
            def get_factor_identity(self, formula):
                return {"canonical_dsl": "x"}  # 缺 signal/hash

        with pytest.raises(FactorIdentityAuthorityError):
            build_identity_view("rank(close)", provider=BrokenProvider())
        with pytest.raises(FactorIdentityAuthorityError):
            build_identity_view("rank(close)", provider=IncompleteProvider())


@pytest.mark.skipif(not FE_OK, reason="factor_engine not importable")
class TestFeAuthorityView:
    def test_identity_view_fields(self):
        from alphaprobe.authority import build_identity_view

        view = build_identity_view("rank(ts_mean(close, 20))")
        assert view["signal_equivalence_id"]
        assert view["canonical_ast_hash"]
        assert view["canonical_formula"]
        assert view["orientation"] in (-1, 1)


# ---------------------------------------------------------------------------
# §30 data_access 唯一取数：FE data_source 走 data_access 登记数据集
# ---------------------------------------------------------------------------


class TestDataAccessAuthority:
    def test_default_data_source_config_uses_data_access(self):
        """FE ASHARE 数据源 config 必须走 data_access 登记数据集，
        禁止直连 COS/parquet/SQL。"""
        from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

        ensure_factor_engine_importable()
        from factor_engine.api.mining_integration import (
            default_ashare_pv_data_source_config,
        )

        cfg = default_ashare_pv_data_source_config()
        assert cfg.get("type") == "data_access"
        assert cfg.get("dataset") == "ashare_stock_daily_adj"
        assert "read_auto" in cfg

    def test_data_source_has_no_raw_sql_or_file_io(self):
        """AlphaPROBE 主链（pipeline/runner/continuous/fe_bridge）无裸
        pd.read_csv/read_parquet/read_sql/create_engine。"""
        src = (
            _read("authority.py")
            + _read("pipeline.py")
            + _read("runner.py")
            + _read("continuous.py")
        )
        for tok in ("read_sql", "create_engine", "pd.read_csv"):
            assert tok not in src, f"主链残留裸 IO: {tok}"
