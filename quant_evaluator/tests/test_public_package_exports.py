"""Public entry points must import without evaluator/api cycles."""

import subprocess
import sys


def test_evaluate_many_is_available_from_package_root_in_fresh_process():
    code = (
        "import quant_evaluator; "
        "from quant_evaluator.api.evaluate_many import evaluate_many; "
        "assert quant_evaluator.evaluate_many is evaluate_many"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
