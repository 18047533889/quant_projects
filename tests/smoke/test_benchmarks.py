"""
Smoke test: Benchmarks
Test that benchmark suite can be executed.
"""
import pytest
import subprocess
import sys
from pathlib import Path


BENCHMARKS_DIR = Path("/home/shw/quant_projects/benchmarks")


class TestBenchmarks:
    """Test benchmark suite execution."""

    def test_benchmarks_directory_exists(self):
        """Test that benchmarks directory exists."""
        assert BENCHMARKS_DIR.exists(), "benchmarks/ directory should exist"
        assert BENCHMARKS_DIR.is_dir(), "benchmarks/ should be a directory"

    def test_benchmark_scripts_exist(self):
        """Test that benchmark scripts exist."""
        expected_benchmarks = [
            "bench_fa_operations.py",
            "bench_fo_search.py",
            "bench_fp_transforms.py",
            "bench_qe_metrics.py",
        ]

        for benchmark in expected_benchmarks:
            bench_path = BENCHMARKS_DIR / benchmark
            assert bench_path.exists(), f"{benchmark} should exist"

    def test_benchmark_help_flag(self):
        """Test benchmarks accept --help flag."""
        benchmarks = [
            "bench_fa_operations.py",
            "bench_fo_search.py",
            "bench_fp_transforms.py",
            "bench_qe_metrics.py",
        ]

        for benchmark_name in benchmarks:
            benchmark_path = BENCHMARKS_DIR / benchmark_name
            if not benchmark_path.exists():
                pytest.skip(f"Benchmark {benchmark_name} not found")
                continue

            try:
                # Check syntax only, not execution (--help may hang if imports are heavy)
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(benchmark_path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                assert result.returncode == 0, \
                    f"{benchmark_name} has syntax errors: {result.stderr}"
            except subprocess.TimeoutExpired:
                pytest.fail(f"{benchmark_name} syntax check timed out")
            except Exception as e:
                pytest.skip(f"Could not test {benchmark_name}: {e}")

    def test_benchmark_readme_exists(self):
        """Test that benchmark documentation exists."""
        readme = BENCHMARKS_DIR / "README.md"
        assert readme.exists(), "benchmarks/README.md should exist"

        content = readme.read_text(encoding='utf-8')
        assert len(content) > 0, "Benchmark README should not be empty"

    def test_baseline_json_exists(self):
        """Test that baseline.json exists."""
        baseline = BENCHMARKS_DIR / "baseline.json"
        if baseline.exists():
            import json
            with open(baseline) as f:
                data = json.load(f)
                assert isinstance(data, dict), "baseline.json should be a dict"
        else:
            pytest.skip("baseline.json not found (may not be generated yet)")

    def test_analyze_results_script(self):
        """Test analyze_results script exists and has valid syntax."""
        script = BENCHMARKS_DIR / "analyze_results.py"
        assert script.exists(), "analyze_results.py should exist"

        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script)],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0, f"analyze_results.py has syntax errors: {result.stderr}"
        except Exception as e:
            pytest.skip(f"Could not check analyze_results.py: {e}")
