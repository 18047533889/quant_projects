"""
QE-P0-03: artifact/content identity must be stable across processes.

Python's builtin ``hash()`` is salted per-process (``PYTHONHASHSEED``), so any
content identity computed from it (artifact_id, content_hash, cache keys,
dedup keys) is NOT reproducible across processes. All identity code must use
canonical SHA-256 serialization instead.

This test computes a digest of every identity surface from the fixed source in
``build/lib/quant_evaluator`` inside *separate Python subprocesses* that run
under different ``PYTHONHASHSEED`` values and asserts they are identical.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"

#: Code run inside each subprocess.  Prints a JSON digest keyed by the
#: identity surface so the parent can compare cross-process.
_SUBPROCESS_SNIPPET = r"""
import json
import sys

import numpy as np

sys.path.insert(0, "@BUILD_LIB@")

from quant_evaluator.contracts.artifact_types import (
    ExposureArtifact,
    ICSeriesArtifact,
    ProbePortfolioArtifact,
    QuantileReturnArtifact,
)
from quant_evaluator.contracts.metric_artifacts import (
    DistributionMetricArtifact,
    MatrixMetricArtifact,
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
)
from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.runtime.intermediates import CacheKey, compute_input_hash

vals = np.arange(6.0).reshape(2, 3)
matrix = np.arange(12.0).reshape(2, 2, 3)

digest = {
    "ICSeriesArtifact": hash(ICSeriesArtifact(values=vals)),
    "ExposureArtifact": hash(ExposureArtifact(values=vals)),
    "ProbePortfolioArtifact": hash(ProbePortfolioArtifact(values=vals)),
    "QuantileReturnArtifact": hash(QuantileReturnArtifact(values=vals, n_quantiles=2)),
    "ScalarMetricArtifact": hash(
        ScalarMetricArtifact(
            metric_id="ic.mean", domain="ic", artifact_kind="scalar",
            values=np.array([1.0, 2.0]),
        )
    ),
    "SeriesMetricArtifact": hash(
        SeriesMetricArtifact(
            metric_id="ic.series", domain="ic", artifact_kind="series",
            values=vals, time_index=("2024-01-01", "2024-01-02"),
        )
    ),
    "VectorMetricArtifact": hash(
        VectorMetricArtifact(
            metric_id="quantile.returns", domain="quantile",
            artifact_kind="vector", values=vals,
        )
    ),
    "MatrixMetricArtifact": hash(
        MatrixMetricArtifact(
            metric_id="transition", domain="quantile",
            artifact_kind="matrix", values=matrix,
        )
    ),
    "DistributionMetricArtifact": hash(
        DistributionMetricArtifact(
            metric_id="bootstrap", domain="ic", artifact_kind="distribution",
            samples=vals, stat_names=("p5", "p95"),
        )
    ),
    "CacheKey": hash(
        CacheKey(metric_id="ic.pearson.mean", chunk_id=3, input_hash="abc123")
    ),
    "compute_input_hash": compute_input_hash(
        ("f1", "f2"), time_slice=(0, 5), bins=4
    ),
    "chart_content_hash": ChartSpec(
        title="T", x_label="X", y_label="Y", chart_type="line",
        data={"series": [1, 2, 3], "labels": ["a", "b", "c"]},
    ).content_hash,
    # Different payload must yield a different identity (content addressed).
    "ICSeriesArtifact_ones": hash(ICSeriesArtifact(values=np.ones((2, 3)))),
}

print(json.dumps(digest, sort_keys=True))
"""


def _run_digest(seed: str) -> dict:
    """Run the digest snippet in a fresh subprocess under the given seed."""
    script = _SUBPROCESS_SNIPPET.replace("@BUILD_LIB@", str(_BUILD_LIB))
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    # Ensure the fixed source tree wins in the subprocess too.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_BUILD_LIB)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (seed={seed}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_identity_hashes_identical_across_python_hashseeds():
    """The same artifact/content identity must hold under different seeds.

    This is the core QE-P0-03 guarantee: two separate processes, one with
    ``PYTHONHASHSEED=0`` and one with ``PYTHONHASHSEED=12345``, must compute
    byte-identical identity values.
    """
    digest_a = _run_digest("0")
    digest_b = _run_digest("12345")

    assert set(digest_a) == set(digest_b)
    for key in digest_a:
        assert digest_a[key] == digest_b[key], (
            f"identity surface {key!r} differs across processes "
            f"(seed 0: {digest_a[key]!r} vs seed 12345: {digest_b[key]!r}) "
            f"--- builtin hash() (PYTHONHASHSEED) must not be used for identity"
        )


def test_identity_hashes_are_content_addressed():
    """Different payload bytes produce a different identity."""
    digest = _run_digest("0")
    assert digest["ICSeriesArtifact"] != digest["ICSeriesArtifact_ones"]
