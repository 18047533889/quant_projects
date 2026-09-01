"""Phase 10 tests：Survival / Direction / Regime Intelligence（§45/§46/§47/§50）。

覆盖：
- compute_survival_metrics 手工样例（上升/下降/翻转序列）
- dna_from_formula（operators / mechanisms / horizon / complexity）
- operator_survival_stats（小样本收缩 ≈ prior；大样本趋近 raw）
- regime（cutoff 闸门 SealedTestViolation；register_2026_event 幂等）
- classify_status 全部 7 个 label 各至少一条可达路径
"""

import math

import pytest

from alphaprobe.contracts import FactorDNA, SurvivalLabel
from alphaprobe.research_protocol import SealedTestViolation
from alphaprobe.survival.dna import dna_from_formula, mechanisms_of, field_families_of
from alphaprobe.survival.profile import (
    classify_status,
    classify_status_from_sequence,
    compute_survival_metrics,
)
from alphaprobe.survival.dna_stats import (
    operator_survival_stats,
    mechanism_survival_stats,
    wilson_interval,
)
from alphaprobe.survival import regime
from alphaprobe.survival.regime import (
    DECAY_2026_EVENT_ID,
    register_2026_event,
    assert_regime_visible,
    regime_response,
    regime_family_response,
)


# ---------------------------------------------------------------------------
# §45 compute_survival_metrics：手工样例
# ---------------------------------------------------------------------------

class TestSurvivalMetrics:
    def test_empty_returns_unclassified(self):
        p = compute_survival_metrics([])
        assert p.support_periods == 0

    def test_rising_sequence_metrics(self):
        seq = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        p = compute_survival_metrics(seq)
        assert p.recent_retention is not None and p.recent_retention > 0.5
        assert p.sign_flip_rate == 0.0
        assert p.rolling_slope is not None and p.rolling_slope > 0.5
        assert p.max_deterioration == 0.0

    def test_falling_sequence_slope_low(self):
        seq = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        p = compute_survival_metrics(seq)
        assert p.rolling_slope is not None and p.rolling_slope < 0.5

    def test_flip_sequence_flip_rate(self):
        seq = [0.03, -0.02, 0.04, -0.03, 0.05, -0.04]
        p = compute_survival_metrics(seq)
        assert p.sign_flip_rate is not None and p.sign_flip_rate > 0.5

    def test_max_deterioration_positive(self):
        seq = [0.09, 0.01, 0.05, 0.02]
        p = compute_survival_metrics(seq)
        assert p.max_deterioration is not None and p.max_deterioration > 0.07


# ---------------------------------------------------------------------------
# §45.1 classify_status：全部 7 个 label 可达
# ---------------------------------------------------------------------------

def _mkprofile(seq, factor_id="f"):
    p, label = classify_status_from_sequence(seq)
    object.__setattr__(p, "factor_id", factor_id)
    return p, label


class TestClassifyStatusAllLabels:
    def test_healthy(self):
        _, label = _mkprofile([0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03])
        assert label == SurvivalLabel.HEALTHY

    def test_broken_recent_negative(self):
        _, label = _mkprofile([0.05, 0.04, 0.03, 0.02, 0.01, -0.01, -0.03, -0.02])
        assert label == SurvivalLabel.BROKEN

    def test_recovered(self):
        seq = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.07]
        _, label = _mkprofile(seq)
        assert label == SurvivalLabel.RECOVERED

    def test_persistent_alpha(self):
        seq = [0.03, 0.04, 0.02, 0.03, 0.04, 0.03, 0.02, 0.04, 0.03, 0.05, 0.04, 0.03]
        _, label = _mkprofile(seq)
        assert label == SurvivalLabel.PERSISTENT_ALPHA

    def test_degrading(self):
        seq = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        _, label = _mkprofile(seq)
        assert label == SurvivalLabel.DEGRADING

    def test_regime_dependent(self):
        # 高频符号翻转但平均仍为正（regime 切换），不足以判 BROKEN/DEGRADING
        seq = [0.04, -0.02, 0.05, -0.02, 0.06, -0.02, 0.07, -0.02, 0.08, -0.02]
        _, label = _mkprofile(seq)
        assert label == SurvivalLabel.REGIME_DEPENDENT

    def test_unclassified(self):
        seq = [0.02, 0.03, 0.02]
        _, label = _mkprofile(seq)
        assert label == SurvivalLabel.UNCLASSIFIED

    def test_all_seven_labels_reachable(self):
        reached = {
            _mkprofile([0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03])[1],
            _mkprofile([0.05, 0.04, 0.03, 0.02, 0.01, -0.01, -0.03, -0.02])[1],
            _mkprofile([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.07])[1],
            _mkprofile([0.03, 0.04, 0.02, 0.03, 0.04, 0.03, 0.02, 0.04, 0.03, 0.05, 0.04, 0.03])[1],
            _mkprofile([0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])[1],
            _mkprofile([0.04, -0.02, 0.05, -0.02, 0.06, -0.02, 0.07, -0.02, 0.08, -0.02])[1],
            _mkprofile([0.02, 0.03, 0.02])[1],
        }
        assert reached == set(SurvivalLabel)

    def test_half_life_only_when_good_fit(self):
        # 指数衰减清晰的序列 → half-life 有值
        p = compute_survival_metrics([0.100, 0.082, 0.067, 0.055, 0.045, 0.037])
        hl = p.descriptive_half_life
        assert hl is not None and 1.0 < hl < 20.0
        # 非线性（先升后降）→ R² 差 → None（不强求指数模型，§45.3）
        p2 = compute_survival_metrics([0.01, 0.02, 0.05, 0.03, 0.02, 0.01])
        assert p2.descriptive_half_life is None


# ---------------------------------------------------------------------------
# §46 FactorDNA 提取
# ---------------------------------------------------------------------------

class TestDNA:
    def test_dna_from_formula_ts_rank_ts_corr(self):
        dna = dna_from_formula("ts_rank(ts_corr(close, volume, 10), 20)")
        assert "ts_rank" in dna.operators
        assert "ts_corr" in dna.operators
        assert "volume" in dna.field_families or "price_volume" in dna.field_families
        assert "liquidity" in dna.mechanisms
        assert "momentum_or_corr" in dna.mechanisms

    def test_field_families_multi(self):
        fams = field_families_of("rank(ts_corr(close, volume, 10))")
        assert "price_volume" in fams
        fams2 = field_families_of("rank(pe) + rank(market_cap)")
        assert "valuation" in fams2

    def test_mechanisms_neg_shell_reversal(self):
        mechs = mechanisms_of("(-(rank(close)))")
        assert "reversal" in mechs

    def test_mechanisms_valuation(self):
        mechs = mechanisms_of("ts_rank(pe, 20)")
        assert "valuation" in mechs

    def test_horizon_bucket(self):
        assert dna_from_formula("ts_mean(close, 5)").horizon_bucket == "fast"
        assert dna_from_formula("ts_rank(close, 30)").horizon_bucket == "medium"
        assert dna_from_formula("ts_corr(close, volume, 120)").horizon_bucket == "slow"

    def test_complexity_bucket(self):
        assert dna_from_formula("rank(close)", {"ast_nodes": 5}).complexity_bucket == "small"
        assert dna_from_formula("rank(close)", {"ast_nodes": 30}).complexity_bucket == "mid"
        assert dna_from_formula("rank(close)", {"ast_nodes": 50}).complexity_bucket == "heavy"


# ---------------------------------------------------------------------------
# §47.1 operator_survival_stats：小样本收缩，大样本趋近 raw
# ---------------------------------------------------------------------------

def _dna(op_name):
    return dna_from_formula(f"{op_name}(close, 20)")


class TestOperatorSurvivalStats:
    def test_small_support_shrinks_to_prior(self):
        factors = [
            (_dna("ts_rank"), SurvivalLabel.HEALTHY),
            (_dna("ts_rank"), SurvivalLabel.BROKEN),
            (_dna("ts_rank"), SurvivalLabel.BROKEN),
        ]
        stats = operator_survival_stats(factors, prior=0.4)
        ts = [s for s in stats if s.key == "ts_rank"][0]
        assert ts.support_count == 3
        assert ts.raw_survival_rate == pytest.approx(1 / 3)
        assert 0.3 < ts.shrunk_survival_rate < 0.45  # 收缩后更接近 prior=0.4

    def test_big_support_approaches_raw(self):
        n = 3000
        factors = [
            (_dna("ts_mean"), SurvivalLabel.HEALTHY if i < int(0.72 * n) else SurvivalLabel.BROKEN)
            for i in range(n)
        ]
        stats = operator_survival_stats(factors, prior=0.5, shrinkage_strength=20.0)
        ts = [s for s in stats if s.key == "ts_mean"][0]
        assert ts.support_count == n
        assert ts.raw_survival_rate == pytest.approx(0.72, abs=0.01)
        assert abs(ts.shrunk_survival_rate - ts.raw_survival_rate) < 0.01

    def test_wilson_interval_bounds(self):
        lo, hi = wilson_interval(0, 10)
        assert 0.0 <= lo <= hi <= 1.0
        lo, hi = wilson_interval(10, 10)
        # Wilson 95% 双侧下界：n=10 全成 → lo≈0.722（标准公式；旧手写单侧特判
        # 1.6449²/(n+1.6449²)=0.213 会把 lo 拉到 0.213——那是错的）
        assert 0.70 <= lo <= 0.75 and hi <= 1.0
        lo, hi = wilson_interval(100, 100)
        # n=100 全成 → lo≈0.963
        assert 0.953 <= lo <= 0.973

    def test_wilson_interval_matches_standard_formula(self):
        """wilson_interval(3,10) 与标准 Wilson 双侧公式数值一致（无手写特判）。"""
        z = 1.96
        total = 10
        p = 3 / total
        denom = 1.0 + z * z / total
        centre = p + z * z / (2.0 * total)
        half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        exp_lo = max(0.0, (centre - half) / denom)
        exp_hi = min(1.0, (centre + half) / denom)
        lo, hi = wilson_interval(3, 10)
        assert lo == pytest.approx(exp_lo)
        assert hi == pytest.approx(exp_hi)

    def test_mechanism_stats(self):
        factors = [
            (dna_from_formula("rank(volume)"), SurvivalLabel.HEALTHY),
            (dna_from_formula("ts_rank(volume, 20)"), SurvivalLabel.BROKEN),
        ]
        stats = mechanism_survival_stats(factors, prior=0.5)
        liq = [s for s in stats if s.key == "liquidity"][0]
        assert liq.support_count == 2
        assert 0.4 <= liq.shrunk_survival_rate <= 0.6


# ---------------------------------------------------------------------------
# §50 regime：cutoff 闸门 + 注册幂等 + 描述统计
# ---------------------------------------------------------------------------

class TestRegime:
    def test_cutoff_blocks_post_cutoff(self):
        from datetime import date
        with pytest.raises(SealedTestViolation):
            assert_regime_visible(event_date="2026-06-15", cutoff=date(2026, 1, 1))
        # pre-cutoff 可见
        assert_regime_visible(event_date="2025-12-31", cutoff=date(2026, 1, 1))

    def test_register_2026_event_idempotent(self):
        import tempfile
        from pathlib import Path
        from alphaprobe.memory import GlobalMemoryStore

        with tempfile.TemporaryDirectory() as td:
            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            assert register_2026_event(store) is True
            assert register_2026_event(store) is False  # 幂等
            store.close()

    def test_register_event_rows(self):
        import tempfile
        from pathlib import Path
        from alphaprobe.memory import GlobalMemoryStore

        with tempfile.TemporaryDirectory() as td:
            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            register_2026_event(store)
            row = store._conn.execute(
                "SELECT event_id, start_date, end_date FROM market_regime_events"
            ).fetchone()
            assert row is not None
            assert row[0] == DECAY_2026_EVENT_ID
            assert row[1] == "2026-06-01"
            assert row[2] == "2026-07-31"
            store.close()

    def test_regime_response_cutoff_gate(self):
        from datetime import date
        event = {
            "event_id": DECAY_2026_EVENT_ID,
            "start_date": "2026-06-01",
            "end_date": "2026-07-31",
        }
        with pytest.raises(SealedTestViolation):
            regime_response(event, [], cutoff=date(2026, 1, 1))

    def test_regime_response_descriptive_stats(self):
        event = {"event_id": "x", "start_date": "2025-06-01", "end_date": "2025-07-31"}
        factors = [
            (dna_from_formula("rank(close)"), SurvivalLabel.BROKEN),
            (dna_from_formula("ts_rank(volume, 20)"), SurvivalLabel.HEALTHY),
            (dna_from_formula("ts_mean(close, 20)"), SurvivalLabel.DEGRADING),
        ]
        out = regime_response(event, factors)
        assert out["summary"]["total_support"] == 3
        assert out["summary"]["decayed"] == 2
        assert "causal_note" in out
        # price_volume 出现在 3 条 factor 的 family 里（close×2 + volume）
        pv = [f for f in out["families"] if f["family"] == "price_volume"][0]
        assert pv["support_count"] == 3
        assert pv["raw_decay_rate"] == 2 / 3

    def test_regime_family_response_shrinkage(self):
        # 构造同一 family 内 decay 集中的数据：全部 BROKEN
        factors = [
            (dna_from_formula("rank(volume)"), SurvivalLabel.BROKEN),
            (dna_from_formula("rank(amount)"), SurvivalLabel.BROKEN),
        ]
        rows = regime_family_response(factors, prior_decay_rate=0.5)
        liq = [r for r in rows if r.family == "liquidity"][0]
        # 2/2=1.0 向 prior 0.5 收缩 → 明显低于 1.0
        assert liq.raw_decay_rate == 1.0
        assert liq.shrunk_decay_rate < 0.9
