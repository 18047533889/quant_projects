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


def test_diverse_objectives_missing_metrics_and_pareto():
    m = load_intake()
    # Load the local optimizer package just as the production entrypoint does.
    import sys
    sys.path.insert(0, str(m.ROOT / "factor_optimizer"))
    metrics = dict(mean_rankic=.03, ls_annual=.2, ls_sharpe=2., ls_mdd=.1,
        drawdown_duration=80., worst_year=.02, year_dispersion=.1, calmar=2., rankic_ir=.3,
        rolling_ic_std=.02, negative_ic_windows=.1, fee_drag=.0001, stress_annual=.18)
    scores = m.diversified_objectives(metrics)
    assert all(v["eligible"] and 0 <= v["score"] <= 1 for v in scores.values())
    missing = m.diversified_objectives(dict(metrics, fee_drag=None))
    assert missing["stability"]["eligible"] and not missing["low_cost"]["eligible"]
    assert missing["low_cost"]["score"] is None
    assert not any(v["eligible"] for v in m.diversified_objectives(dict(metrics, ls_annual=-.1)).values())
    worse = dict(metrics, ls_annual=.1, ls_sharpe=1., ls_mdd=.2, fee_drag=.0002)
    comparison = m.objective_comparison({
        "a": dict(objectives=scores, objective_metrics=metrics, expression="AdjClose"),
        "b": dict(objectives=m.diversified_objectives(worse), objective_metrics=worse, expression="AdjOpen")})
    assert comparison["pareto_frontier"] == ["a"]
    assert comparison["champions"]["balanced"]["variant"] == "a"


def test_optimizer_direction_locked_before_preprocessing(monkeypatch):
    m = load_intake()
    calls = []
    def evaluate(values, labels, **kwargs):
        calls.append((values.copy(), kwargs))
        return SimpleNamespace(direction=-1)
    monkeypatch.setattr(m, "evaluate_report_arrays", evaluate)
    raw = np.array([[1., 2.], [3., 4.], [5., 6.]])
    train = np.array([True, True, False])
    directed, sign = m.lock_optimizer_direction(raw, raw, train)
    assert sign == -1 and np.array_equal(directed, -raw)
    assert calls[0][0].shape == (2, 2)
    unchanged, sign = m.lock_optimizer_direction(directed, raw, train, already_directed=True)
    assert sign == 1 and np.array_equal(unchanged, directed)
    assert calls[-1][1]["fixed_direction"] == 1


def test_optimizer_dsl_keeps_direction_inside_preprocessing():
    m = load_intake()
    for recipe in ["raw", "cs_rank", "cs_zscore", "winsor_1pct", "winsor_5pct"]:
        formula = m.optimizer_dsl("neg(AdjClose)", recipe)
        assert formula.count("neg(") == 1
        assert "factor_preprocess." not in formula
    assert m.optimizer_dsl("neg(AdjClose)", "cs_rank") == "rank_pct(neg(AdjClose))"


def test_evaluation_version_banner_preserves_test_time(monkeypatch):
    m = load_intake()
    monkeypatch.setattr(m, "report_time", lambda: "2026-09-07T10:00:00+08:00")
    p = {"completed_at": "2026-09-06T22:00:00+08:00", "run_id": "run-1",
         "pipeline_source_sha256": "abc123"}
    html = m.evaluation_banner("<html><body><h1>factor</h1></body></html>", p)
    assert html.index("evaluation-version") < html.index("<h1>")
    assert p["completed_at"] in html and "run-1" in html and "abc123" in html
    assert "2026-09-07T10:00:00+08:00" in html
    assert "未记录，旧结果待重测" in m.evaluation_banner("<body></body>")


def test_overnight_waves_resume_without_stale_running():
    m = load_intake()
    state = {"factors": {"done": {"status": "passed"}, "old": {"status": "running"}}}
    records = [{"page_name": name} for name in ["done", "old", "new", "last"]]
    waves = m.overnight_pending_waves(records, state, 2)
    assert [[r["page_name"] for r in w] for w in waves] == [["old", "new"], ["last"]]
    assert state["factors"]["old"]["status"] == "retry_pending"
    assert state["factors"]["done"]["status"] == "passed"


def test_fallback_html_does_not_invent_metrics_or_optimization():
    m = load_intake()
    page = m._minimal_page("<name>", "where(x<0, x, 0)", "", False,
                           {"rank_ic": float("nan"), "ic_ir": None})
    assert "&lt;name&gt;" in page and "x&lt;0" in page
    assert "优化结果未在本页验证" in page
    assert "本因子已进入" not in page
    assert ">nan<" not in page and ">+0.0000<" not in page


def test_overnight_batch_keeps_per_factor_artifacts(tmp_path, monkeypatch):
    import subprocess
    m = load_intake()
    records = [{"page_name": name, "fe_formula": "AdjClose", "source_formula": "close"}
               for name in ["a", "b"]]
    monkeypatch.setattr(m, "requested_dsl_records", lambda _: (records, []))
    commands = []
    class Child:
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert len(json.loads(Path(command[command.index("--manifest") + 1]).read_text())) == 2
            target = Path(command[command.index("--output-manifest") + 1])
            target.write_text(json.dumps({"factors": {
                "a": {"metrics": {"rank_ic": .02, "rank_ic_ir": .2}},
                "b": {"status": "unavailable", "metrics": {}}}}))
        def poll(self):
            return 0
        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", Child)
    state = m.run_overnight_queue(None, tmp_path, batch_size=2, fe_backend="pandas", qe_backend="auto")
    assert len(commands) == 1
    assert commands[0][commands[0].index("--qe-backend") + 1] == "auto"
    for name in ["a", "b"]:
        assert len(json.loads((tmp_path / name / "candidate.json").read_text())) == 1
        assert list(json.loads((tmp_path / name / "report_manifest.json").read_text())["factors"]) == [name]
        assert state["factors"][name]["resource_policy"]["batch_size"] == 2
    assert state["factors"]["b"]["status"] == "unavailable"


def test_failed_refresh_excludes_old_matrix_before_evaluation():
    m = load_intake()
    records = [{"page_name": "stale"}, {"page_name": "fresh"}]
    admitted, failures = m.evaluation_candidates_after_landing(
        records, {"factor_engine_errors": {"stale": "PIT failure"}})
    assert admitted == [{"page_name": "fresh"}]
    assert "PIT failure" in failures["stale"]
    assert "old matrix excluded" in failures["stale"]
    assert m.evaluation_candidates_after_landing(records, {}) == (records, {})


def test_incomplete_existing_matrix_is_relanded(monkeypatch):
    m = load_intake()
    monkeypatch.setattr(m, "matrix_path", lambda name: Path("existing.parquet"))
    monkeypatch.setattr(m, "matrix_source", lambda name: SimpleNamespace(is_full_window=name == "complete"))
    waves = []
    def land(records, **kwargs):
        names = [r["page_name"] for r in records]
        waves.append(names)
        return {"landed": names}
    monkeypatch.setattr(m, "land_factor_batch_windowed", land)
    result = m.land_missing_factors([
        {"page_name": "incomplete", "fe_formula": "AdjClose"},
        {"page_name": "complete", "fe_formula": "AdjClose"}])
    assert waves == [["incomplete"]]
    assert result["landed"] == ["incomplete"]


def test_requested_valuation_uses_catalog_and_exact_join(tmp_path):
    m = load_intake()
    path = tmp_path / "requested.txt"
    path.write_text("rank(turnover_ratio)\nrank(market_cap)\nrank(roe)\n")
    records, deferred = m.requested_dsl_records(path)
    assert len(records) == 3
    assert not deferred
    assert {r["fe_formula"] for r in records if "valuation." in r["fe_formula"]} == {
        "rank(col('valuation.turnover_ratio'))", "rank(col('valuation.market_cap'))"}
    roe = next(r for r in records if r["source_formula"] == "rank(roe)")
    assert "__fe_source_ref_v1__" in roe["fe_formula"]
    assert "0.01" in roe["fe_formula"]
    engine = m.build_incremental_engine(start_date="2016-01-04", end_date="2016-01-05", records=records)
    # Construction must succeed using the existing FE composite/DataAccess adapter.
    assert engine is not None


def test_weekly_fields_use_ashare_contracts_and_keep_minute_gated(tmp_path):
    m = load_intake()
    path = tmp_path / "fields.txt"
    path.write_text("rank(net_profit)\nrank(dividend_yield)\nrank(free_cap)\nrank(high_limit)\nrank(minute_volume)\n")
    records, deferred = m.requested_dsl_records(path)
    assert len(records) == 4
    assert len(deferred) == 1
    assert deferred[0]["source_formula"] == "rank(minute_volume)"
    assert any("AdjHighLimit" in r["fe_formula"] for r in records)
    assert any("valuation.dividend_yield" in r["fe_formula"] for r in records)


def test_requested_minute_calls_bind_session_sources(tmp_path):
    m = load_intake()
    path = tmp_path / "minute.txt"
    path.write_text("intraday_activity_duration_curvature(minute_volume, buckets=10)\n"
                    "intra_close_participation(minute_volume, 30)\n"
                    "intraday_bvc_imbalance(minute_close, minute_volume, scale_window=20)\n")
    records, deferred = m.requested_dsl_records(path)
    assert len(records) == 3 and not deferred
    from factor_engine.api.source_ref import decode_source_ref
    import ast
    for record in records:
        tree = ast.parse(record["fe_formula"], mode="eval")
        ref = decode_source_ref(tree.body.args[0].value)
        assert ref.table == "StockMinuteBar"
        assert ref.transform_params_dict()["bar_minutes"] == 1
        assert ref.transform_params_dict()["min_coverage"] == 1.0


def test_minute_history_binding_is_incremental_and_preserves_existing(tmp_path):
    m = load_intake()
    history, mirror = tmp_path / "history", tmp_path / "mirror"
    history.mkdir(); mirror.mkdir()
    (history / "2016-01-04.parquet").touch()
    (history / "2024-01-02.parquet").touch()
    current = mirror / "2024-01-02.parquet"
    current.write_bytes(b"existing-owner-data")
    assert m.ensure_local_minute_history(mirror, history) == 1
    assert (mirror / "2016-01-04.parquet").is_symlink()
    assert current.read_bytes() == b"existing-owner-data"
    assert m.ensure_local_minute_history(mirror, history) == 0


def test_registered_minute_features_are_adjustment_invariant():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import (
        _compute_registered_session_feature, _session_calendar)
    index = pd.date_range("2016-01-05 09:31", periods=120, freq="min").append(
        pd.date_range("2016-01-05 13:01", periods=120, freq="min"))
    t = np.arange(240)
    bars = pd.DataFrame({"timestamp": index, "close": 10 + np.sin(t / 8) / 10,
                         "volume": 100 + t, "amount": (100 + t) * 10})
    calendar = _session_calendar("ashare_stock_minute", "09:30", "15:00", "bar_end")
    for feature in ["fe_activity_volume", "fe_activity_amount", "fe_close_participation", "fe_bvc_imbalance"]:
        raw = _compute_registered_session_feature(feature, bars, {}, calendar, 1.)
        adjusted = _compute_registered_session_feature(feature, bars, {}, calendar, 1.733271)
        assert np.isfinite(raw), feature
        np.testing.assert_allclose(raw, adjusted, atol=1e-12)


def test_utc_minute_labels_are_filtered_in_exchange_time():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import _session_local_minute_frame
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2016-01-06 01:31:00Z", "2016-01-06 07:00:00Z"])})
    out = _session_local_minute_frame(frame, "ashare_stock_minute")
    assert out.timestamp.dt.strftime("%H:%M").tolist() == ["09:31", "15:00"]
    frame["timestamp"] = frame.timestamp.dt.tz_convert(None)
    pd.testing.assert_frame_equal(out, _session_local_minute_frame(frame, "ashare_stock_minute", naive_utc=True))
    pd.testing.assert_frame_equal(out, _session_local_minute_frame(out, "ashare_stock_minute"))


def test_financial_conflicting_vintage_cannot_be_backdated():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    source = LQTPLogicalDataSource(SimpleNamespace())
    raw = pd.DataFrame({"Symbol": ["A", "A"], "ReportPeriodEndDate": ["2015-09-30"] * 2,
                        "PubDate": ["2015-10-20"] * 2, "Roe": [10., 20.]})
    import pytest
    with pytest.raises(ValueError, match="conflicting values"):
        source._financial_from_raw("ashare_stock_indicator", "Roe", raw, transform=None, params={})


def test_optimized_comparison_does_not_flip_saved_values_twice(monkeypatch):
    load_intake()
    import render_optimized_pages as renderer
    dates = pd.date_range("2016-01-04", periods=40)
    raw = pd.DataFrame(np.arange(120).reshape(40, 3), index=dates)
    opt = -raw
    observed = []
    def evaluate(mat, prices, *, flip=False):
        directed = -mat if flip else mat
        observed.append(directed.to_numpy())
        return dates, SimpleNamespace(quantile_nav=np.ones((40, 10)),
                                      long_short_nav_aligned=np.ones(40))
    monkeypatch.setattr(renderer, "_comparison_result", evaluate)
    monkeypatch.setattr(renderer, "fig_to_b64", lambda fig: "chart")
    renderer.plot_decile_compare(raw, opt, raw, "sample", page_flipped=True)
    renderer.plot_ls_compare(raw, opt, raw, "sample", page_flipped=True)
    for result in observed:
        np.testing.assert_array_equal(result, opt.to_numpy())


def test_distribution_keeps_zero_and_filters_nonfinite(monkeypatch):
    load_intake()
    import render_evoalpha14_pages as renderer
    means = []
    def inspect(fig):
        means.append(fig.axes[0].lines[0].get_xdata()[0])
        return "chart"
    monkeypatch.setattr(renderer, "fig_to_b64", inspect)
    renderer.plot_ic_distribution(pd.Series([0., 0., 0., .1, .2, np.nan, np.inf]), "sample")
    np.testing.assert_allclose(means, [.06])


def test_one_way_commission_charges_opening_and_both_ls_legs():
    from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_arrays
    values = np.tile(np.arange(40, dtype=float), (30, 1))
    result = evaluate_report_arrays(values, np.zeros_like(values), min_assets=1,
                                    commission_rate=.0001, fixed_direction=1)
    np.testing.assert_allclose(result.quantile_returns[0], -.0001)
    np.testing.assert_allclose(result.long_short_returns[0], -.0002)
    np.testing.assert_allclose(result.long_short_returns[1:], 0.)


def test_qe_validation_keeps_training_direction():
    from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_arrays
    rng = np.random.default_rng(12)
    values = rng.normal(size=(30, 40))
    result = evaluate_report_arrays(values, -values * .001, fixed_direction=1)
    assert result.direction == 1
    assert result.mean_rank_ic < -.99


def test_optimizer_runs_actual_libraries_on_small_panel(tmp_path, monkeypatch):
    intake = load_intake()
    rng = np.random.default_rng(77)
    dates = pd.bdate_range("2016-01-04", "2026-08-27")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (len(dates), 40)), axis=0)), index=dates)
    # Deliberately predictive synthetic fixture, not a production feature.
    values = prices.pct_change(fill_method=None).shift(-2).fillna(0) + rng.normal(0, .01, prices.shape)
    monkeypatch.setattr(intake, "load_matrix", lambda *a, **k: values)
    monkeypatch.setattr(intake, "load_vwap", lambda **k: prices)
    monkeypatch.setattr(intake, "OPT_DIR", tmp_path / "matrices")
    monkeypatch.setattr(intake, "OPT_META_JSON", tmp_path / "meta.json")
    monkeypatch.setattr(intake, "MIN_UNIVERSE", 2)  # 40 assets / 10 groups in this small fixture
    replay_calls = []
    def replay_fixture(records, *, output_dir, **kwargs):
        # Synthetic source fixture: never read production data in a unit test.
        from factor_engine.cleaned_operators.common.cross_sectional import RankPct, CrossSectionalZscore
        from factor_engine.cleaned_operators.common.elementwise import Winsorize
        expression = records[0]["fe_formula"]
        replay_calls.append(expression)
        if expression.startswith("rank_pct("):
            result = RankPct()._calculate_series(values)
        elif expression.startswith("c_zscore("):
            result = CrossSectionalZscore()._calculate_series(values)
        elif expression.startswith("winsorize("):
            lower = .01 if expression.endswith("0.01, 0.99)") else .05
            result = Winsorize()._calculate_series(values, lower=lower, upper=1-lower)
        else:
            result = values
        result.to_parquet(Path(output_dir) / "smoke.parquet")
    monkeypatch.setattr(intake, "land_factor_batch_windowed", replay_fixture)
    result = intake.stage_optimize_lite({"page_name": "smoke", "fe_formula": "ts_return(AdjClose,20)"}, None)
    meta = json.loads((tmp_path / "meta.json").read_text())["smoke"]
    assert meta["optimizer"] == "factor_optimizer.SearchRunner"
    assert len(meta["variants"]) == 5
    assert Path(result["path"]).exists()
    assert meta["dsl_preproc_ops"]
    assert len(replay_calls) == 1
    assert meta["matrix_direction_applied"]
    assert meta["direction_policy"] == "train-once-before-preprocessing-v1"
    assert meta["objective_comparison"]["default"] == "stability"
    assert {"stability", "ic_stability", "low_cost"} <= set(meta["objective_comparison"]["champions"])
    # This highly predictive fixture can have zero drawdown: undefined Calmar
    # must remain missing, not become an infinite winning balanced score.
    for name, trial in meta["variants"].items():
        assert set(trial["objectives"]) == set(intake.OBJECTIVE_PROFILES)
        if trial["objective_metrics"]["calmar"] is None:
            assert not trial["objectives"]["balanced"]["eligible"]
    assert meta["stress_commission_rate"] == intake.COMMISSION_RATE * 5
    assert all(r["direction"] == 1 for r in meta["variants"].values())


def test_windowed_landing_rejects_full_history_before_reading_data():
    import pytest
    intake = load_intake()
    intake.require_bounded_history(SimpleNamespace(lookback=20, ir=None), "bounded")
    with pytest.raises(ValueError, match="state continuity"):
        intake.require_bounded_history(
            SimpleNamespace(lookback=20, ir=None, requires_full_history=True), "stateful")


def test_retry_is_explicit_atomic_and_preserves_success_and_failure_history():
    import pytest
    intake = load_intake()
    state = {"factors": {"bad": {"status": "unavailable", "reason": "volume"},
                          "good": {"status": "passed"}}}
    before = json.loads(json.dumps(state))
    with pytest.raises(ValueError):
        intake.reopen_failed_candidates(state, ["bad", "good"])
    assert state == before
    intake.reopen_failed_candidates(state, ["bad", "bad"])
    assert state["factors"]["bad"]["status"] == "retry_pending"
    assert state["factors"]["good"]["status"] == "passed"
    assert state["retry_history"]["bad"] == [before["factors"]["bad"]]


def test_python_metadata_is_converted_to_dsl_without_executing_source(tmp_path, monkeypatch):
    intake = load_intake()
    monkeypatch.setattr(intake, "factor_from_record", lambda record: record)
    folder = tmp_path / "miner" / "20260901"
    folder.mkdir(parents=True)
    source = 'def factor_sample(df):\n    return df["close"].rolling(20).mean()'
    (folder / "metadata.json").write_text(json.dumps({
        "meta": {"market": "ashare"}, "factors": [{
            "factor_name": "python_mean", "expression_type": "python",
            "factor_expression": source,
        }]}))
    candidates, rejected = intake.discover_price_dsl_candidates(tmp_path, [], since="20260901")
    assert not rejected
    assert len(candidates) == 1
    assert candidates[0]["source_formula"] == source
    assert candidates[0]["source_language"] == "python"
    assert "safe_div_null(ts_mean(AdjClose, 20)" in candidates[0]["fe_formula"]
    assert "o[" not in candidates[0]["fe_formula"]


def test_automatic_python_conversion_rejects_uncertified_round():
    import pytest
    load_intake()
    from fe_code_transpiler import transpile_native_dsl
    with pytest.raises(ValueError, match="round"):
        transpile_native_dsl('def factor_sample(df):\n    return df["close"].round(2)')


def test_homepage_retry_does_not_duplicate_factor_or_weekly_section(tmp_path, monkeypatch):
    intake = load_intake()
    target = tmp_path / "index.html"
    target.write_text('<h2 id="all-factors">全部 1 个因子</h2>'
                      '<table><tbody></tbody></table>'
                      '<section id="robustness-2026"></section>')
    monkeypatch.setattr(intake, "INDEX_HTML", target)
    monkeypatch.setattr(intake, "load_pool", lambda: [{"page_name": "sample"}])
    records = [{"page_name": "sample", "rank_ic": .1, "ic_ir": .2}]
    intake.update_index(records)
    intake.update_index(records)
    html = target.read_text()
    assert html.count('id="new-mining"') == 1
    assert html.count('href="factors/factor_sample.html"') == 2


def test_detail_template_uses_artifact_dates_and_does_not_invent_metrics():
    from collections import defaultdict
    load_intake()
    import render_evoalpha14_pages as renderer
    base = defaultdict(float, report_start="2016-01-04", report_end="2026-08-27")
    charts = dict(svg_ts="", monthly="", decile="", ls="", dist="")
    html = renderer.build_html("sample", "", "rank(AdjClose)", "", "",
                               base, {}, {}, {}, charts, "")
    assert html.count("2016-01-04 ~ 2026-08-27") == 2
    assert "0.00%</b><span>Top10%" not in html
    assert "优化后 RankIC</td><td>0.0000" not in html
    assert "2026 RankIC 0.0000" not in html


def test_landing_parallel_preserves_dq_pit_and_streaming_sink():
    m = load_intake()
    calls = []
    class Engine:
        def run_many_parallel(self, factors, **kwargs):
            calls.append(kwargs)
            for factor in factors:
                kwargs["sink"](factor.name, factor.name)
    output = {}
    m.land_factor_batch([{"page_name": n, "fe_formula": "rank(close)"} for n in ["a", "b"]],
                        engine=Engine(), sink=lambda k, v: output.setdefault(k, v), workers=4)
    assert set(output) == {"a", "b"}
    assert calls[0]["n_jobs"] == 2
    assert calls[0]["pit_enforce"] and calls[0]["input_dq_check"]
    assert calls[0]["pit_forbid_forward_fill"] and calls[0]["enable_cse"]
    assert calls[0]["result_policy"] == "sink"
    assert "warmup_clusters" not in calls[0]


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


def test_physical_volume_dsl_is_bound_once_and_persisted():
    intake = load_intake()
    record = {"page_name": "volume_binding", "fe_formula": "safe_div_null(Volume, Factor)",
              "source_formula": "volume"}
    factor = intake.factor_from_record(record)
    assert [node.name for node in factor.expr.args] == ["Volume", "Factor"]
    assert "col('Volume')" in record["fe_formula"]
    assert "col('Factor')" in record["fe_formula"]
    assert record["source_formula"] == "volume"
    assert intake.factor_from_record(record).expr == factor.expr
    logical = intake.factor_from_record({"page_name": "logical", "fe_formula": "volume"})
    assert logical.expr.name == "volume"


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
