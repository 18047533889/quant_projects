import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
from types import SimpleNamespace
import json


MODULE_PATH = Path("/home/sunhaiwei/quant_projects/jobs/incremental_factor_intake.py")


def load_intake():
    spec = importlib.util.spec_from_file_location("incremental_factor_intake", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_landing_many_uses_one_streaming_run_many_call():
    intake = load_intake()
    calls = []

    class Engine:
        def run_many(self, factors, **kwargs):
            calls.append((factors, kwargs))
            for factor in factors:
                kwargs["sink"](factor.name, f"values:{factor.name}")
            return {"results": {}}

    written = {}
    records = [
        {"page_name": "first", "fe_formula": "rank(close)"},
        {"page_name": "second", "fe_formula": "ts_mean(close, 5)"},
    ]

    result = intake.land_factor_batch(
        records,
        engine=Engine(),
        sink=lambda name, values: written.setdefault(name, values),
    )

    assert len(calls) == 1
    factors, options = calls[0]
    assert [factor.name for factor in factors] == ["first", "second"]
    assert options["enable_cse"] is True
    assert options["result_policy"] == "sink"
    assert written == {"first": "values:first", "second": "values:second"}
    assert result["landed"] == ["first", "second"]


def test_load_vwap_consumes_data_access_series():
    intake = load_intake()
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-02", "2026-01-05"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )

    class Source:
        def load_column(self, field):
            assert field == "AdjVwap"
            return pd.Series([10.0, 20.0, 11.0, 18.0], index=idx, name=field)

    panel = intake.load_vwap(source=Source())

    assert panel.index.tolist() == list(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    assert panel.columns.tolist() == ["A", "B"]
    assert panel.loc[pd.Timestamp("2026-01-05"), "A"] == 11.0


def test_eval_matrix_delegates_metrics_and_direction_to_quant_evaluator(monkeypatch):
    intake = load_intake()
    dates = pd.bdate_range("2016-01-04", periods=25)
    columns = [f"S{i:02d}" for i in range(30)]
    matrix = pd.DataFrame(1.0, index=dates, columns=columns)
    vwap = pd.DataFrame(2.0, index=dates, columns=columns)
    calls = []

    def evaluate(factors, returns, **options):
        calls.append((factors.shape, returns.shape, options))
        return SimpleNamespace(
            mean_rank_ic=0.12,
            rank_ic_ir=0.8,
            rank_ic_std=0.15,
            valid_return_periods=23,
            direction=-1,
            sharpe=1.3,
            annualized_return=0.2,
            cumulative_return=0.4,
            max_drawdown=0.1,
            win_rate=0.6,
        )

    monkeypatch.setattr(intake, "evaluate_report_arrays", evaluate, raising=False)

    result = intake.eval_matrix(matrix, vwap)

    assert len(calls) == 1
    assert result["rank_ic"] == 0.12
    assert result["ic_ir"] == 0.8
    assert result["direction"] == -1
    assert result["ls_sharpe"] == 1.3


def test_evaluate_factor_batch_tiles_and_uses_one_batch_call_per_tile():
    intake = load_intake()
    dates = pd.bdate_range("2016-01-04", periods=25)
    columns = [f"S{i:02d}" for i in range(30)]
    matrices = {
        name: pd.DataFrame(index_value, index=dates, columns=columns)
        for index_value, name in enumerate(("a", "b", "c"), start=1)
    }
    vwap = pd.DataFrame(10.0, index=dates, columns=columns)
    calls = []

    def evaluator(values, returns, *, factor_ids, **options):
        calls.append((values.shape, factor_ids, options))
        return SimpleNamespace(
            backend_used="cuda",
            backend_fallback_reason=None,
            factors={name: SimpleNamespace(mean_rank_ic=0.1) for name in factor_ids},
        )

    result = intake.evaluate_factor_batch(
        [{"page_name": name} for name in ("a", "b", "c")],
        vwap=vwap,
        matrix_loader=lambda name: matrices[name],
        evaluator=evaluator,
        batch_size=2,
    )

    assert [call[1] for call in calls] == [("a", "b"), ("c",)]
    assert calls[0][0] == (25, 30, 2)
    assert result["backend_used"] == {"cuda"}
    assert set(result["factors"]) == {"a", "b", "c"}


def test_evaluate_factor_batch_keeps_data_access_authority_axes():
    intake = load_intake()
    dates = pd.bdate_range("2016-01-04", periods=25)
    columns = [f"S{i:02d}" for i in range(30)]
    vwap = pd.DataFrame(10.0, index=dates, columns=columns)
    partial = pd.DataFrame(1.0, index=dates[3:-2], columns=columns[4:-3])
    seen = {}

    def evaluator(values, returns, *, factor_ids, **options):
        seen["shape"] = values.shape
        seen["prefix_missing"] = bool(pd.isna(values[:3]).all())
        return SimpleNamespace(
            backend_used="cpu", backend_fallback_reason=None,
            factors={factor_ids[0]: SimpleNamespace(mean_rank_ic=0.1)},
        )

    result = intake.evaluate_factor_batch(
        [{"page_name": "partial"}], vwap=vwap,
        matrix_loader=lambda _: partial, evaluator=evaluator,
    )

    assert seen == {"shape": (25, 30, 1), "prefix_missing": True}
    assert result["dates"].tolist() == dates.tolist()


def test_update_index_derives_counts_from_current_pool(tmp_path, monkeypatch):
    intake = load_intake()
    index = tmp_path / "index.html"
    index.write_text(
        '<b>2</b><span>因子总数</span><b>+0</b><span>本周新挖</span>'
        '<h2 id="all-factors">全部 2 个因子</h2><table><tbody></tbody></table>'
        '<section id="robustness-2026">',
        encoding="utf-8",
    )
    monkeypatch.setattr(intake, "INDEX_HTML", index)
    monkeypatch.setattr(
        intake,
        "load_pool",
        lambda: [{"page_name": "old1"}, {"page_name": "old2"}, {"page_name": "new"}],
    )

    result = intake.update_index([
        {"page_name": "new", "rank_ic": 0.1, "ic_ir": 0.5, "cluster_id": "c1"}
    ])

    html = index.read_text(encoding="utf-8")
    assert result["total"] == 3
    assert result["n_new"] == 1
    assert "全部 3 个因子" in html
    assert "本周新挖增量因子（1）" in html


def test_landing_many_accepts_canonical_json_expression():
    intake = load_intake()
    captured = []

    class Engine:
        def run_many(self, factors, **kwargs):
            captured.extend(factors)
            return {"results": {}}

    formula = json.dumps({
        "kind": "call",
        "op": "ts_mean",
        "args": [{"kind": "column", "name": "AdjClose"}, {"kind": "literal", "value": 5}],
        "kwargs": {},
    })
    intake.land_factor_batch(
        [{"page_name": "json_factor", "fe_formula": formula}],
        engine=Engine(),
        sink=lambda *_: None,
    )

    assert captured[0].name == "json_factor"
    assert "ts_mean" in str(captured[0].expr)


def test_missing_capability_flag_does_not_bypass_factor_engine(monkeypatch, tmp_path):
    intake = load_intake()
    calls = []
    monkeypatch.setattr(intake, "MATRICES_DIR", tmp_path)
    monkeypatch.setattr(intake, "matrix_path", lambda _: None)
    monkeypatch.setattr(
        intake, "land_factor_batch_windowed",
        lambda wave, **_: calls.extend(x["page_name"] for x in wave)
        or {"landed": [x["page_name"] for x in wave]},
    )

    result = intake.land_missing_factors([
        {"page_name": "new_factor", "fe_formula": "rank(AdjClose)"},
    ])

    assert calls == ["new_factor"]
    assert result["landed"] == ["new_factor"]
    assert result["python_fallback"] == []


def test_report_manifest_is_generated_from_batch_evaluation(tmp_path):
    intake = load_intake()
    target = tmp_path / "report_manifest.json"
    evaluated = SimpleNamespace(
        mean_rank_ic=0.1,
        rank_ic_ir=0.5,
        rank_ic_std=0.2,
        valid_return_periods=100,
        direction=-1,
        sharpe=1.2,
        annualized_return=0.3,
        cumulative_return=0.7,
        max_drawdown=0.1,
        win_rate=0.55,
    )

    intake.write_report_manifest(
        [{"page_name": "factor_a", "factor_name": "Factor A", "fe_formula": "rank(AdjClose)"}],
        {"factors": {"factor_a": evaluated}, "backend_used": {"cuda"}, "fallbacks": {},
         "dates": pd.bdate_range("2016-01-04", periods=2)},
        target=target,
    )

    payload = json.loads(target.read_text())
    assert payload["schema_version"] == 1
    assert payload["backend_used"] == ["cuda"]
    assert payload["factors"]["factor_a"]["metrics"]["ls_sharpe"] == 1.2
    assert payload["factors"]["factor_a"]["direction"] == -1
    assert payload["factors"]["factor_a"]["raw_formula"] == "rank(AdjClose)"
    assert payload["factors"]["factor_a"]["effective_formula"] == "neg(rank(AdjClose))"


def test_report_manifest_persists_canonical_chart_artifact(tmp_path):
    intake = load_intake()
    target = tmp_path / "report_manifest.json"
    evaluated = SimpleNamespace(
        mean_rank_ic=0.1, rank_ic_ir=0.5, rank_ic_std=0.2,
        valid_return_periods=2, direction=1, sharpe=1.2,
        annualized_return=0.3, cumulative_return=0.02,
        max_drawdown=0.01, win_rate=0.5,
        rank_ic_series=np.array([0.1, 0.2]),
        quantile_returns=np.ones((2, 10)),
        quantile_nav=np.ones((2, 10)),
        long_short_returns=np.array([0.01, 0.01]),
        long_short_nav=np.array([1.01, 1.0201]),
        long_short_nav_aligned=np.array([1.01, 1.0201]),
    )
    payload = intake.write_report_manifest(
        [{"page_name": "factor_a", "fe_formula": "rank(AdjClose)"}],
        {"factors": {"factor_a": evaluated}, "backend_used": {"cpu"},
         "fallbacks": {}, "dates": pd.bdate_range("2016-01-04", periods=2)},
        target=target,
    )

    artifact = tmp_path / payload["factors"]["factor_a"]["artifact"]
    assert artifact.exists()
    arrays = np.load(artifact)
    assert arrays["quantile_nav"].shape == (2, 10)
    assert arrays["dates"].shape == (2,)
    assert len(payload["factors"]["factor_a"]["artifact_sha256"]) == 64


def test_publish_from_manifest_uses_verified_artifact_without_re_evaluation(tmp_path, monkeypatch):
    intake = load_intake()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact = artifacts / "factor_a.npz"
    np.savez_compressed(
        artifact,
        dates=np.asarray(pd.bdate_range("2016-01-04", periods=2), dtype="datetime64[ns]"),
        rank_ic_series=np.asarray([0.1, 0.2]),
        quantile_returns=np.ones((2, 10)),
        quantile_nav=np.ones((2, 10)),
        long_short_returns=np.asarray([0.01, 0.02]),
        long_short_nav=np.asarray([1.01, 1.03]),
        long_short_nav_aligned=np.asarray([1.01, 1.03]),
    )
    manifest = {
        "schema_version": 1,
        "factors": {
            "factor_a": {
                "factor_name": "Factor A",
                "raw_formula": "rank(AdjClose)",
                "direction": 1,
                "artifact": "artifacts/factor_a.npz",
                "artifact_sha256": intake.hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "metrics": {"rank_ic": 0.1, "ic_ir": 0.5, "ls_sharpe": 1.2,
                            "ls_annual": 0.3, "ls_mdd": 0.1, "ls_winrate": 0.5,
                            "n_days": 2},
            },
            "missing": {"factor_name": "Missing", "status": "unavailable",
                        "reason": "matrix is incomplete", "raw_formula": "rank(x)",
                        "metrics": {}},
        },
    }
    manifest_path = tmp_path / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def render(factor, eval_result, *, report_result, report_dates, out_dir):
        calls.append((factor, report_result, report_dates, out_dir))
        return {"mode": "full"}

    monkeypatch.setattr(intake, "stage_page_inject", render)
    report_dir = tmp_path / "report"
    result = intake.publish_report_from_manifest(manifest_path, report_dir=report_dir)

    assert result == {"published": 2, "available": 1, "unavailable": 1}
    assert len(calls) == 1
    assert calls[0][0]["page_name"] == "factor_a"
    assert calls[0][1].quantile_nav.shape == (2, 10)
    assert calls[0][2].tolist() == pd.bdate_range("2016-01-04", periods=2).tolist()
    assert "factor_a" in (report_dir / "index.html").read_text(encoding="utf-8")
    assert (report_dir / "factors" / "factor_missing.html").exists()
