import sys

import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as harness


MIXED_THREE = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def _invoke_without_cos_load(monkeypatch, tmp_path, extra_args):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch",
        "--factors", "24",
        "--metrics", ",".join(MIXED_THREE),
        *extra_args,
    ])

    def stop_before_cos_load(**kwargs):
        raise RuntimeError("COS load suppressed by unit test")

    monkeypatch.setattr(harness, "load_real_batch", stop_before_cos_load)
    return harness.main()


def test_exact_f24_mixed_three_is_admitted_without_loading_cos(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="COS load suppressed"):
        _invoke_without_cos_load(
            monkeypatch, tmp_path,
            ["--run", "--output", str(tmp_path / "receipt.json")],
        )


def test_exact_f24_mixed_three_requires_explicit_run(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _invoke_without_cos_load(
            monkeypatch, tmp_path, ["--output", str(tmp_path / "receipt.json")],
        )
    assert exc.value.code == 2


def test_exact_f24_mixed_three_requires_persisted_receipt(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _invoke_without_cos_load(monkeypatch, tmp_path, ["--run"])
    assert exc.value.code == 2
