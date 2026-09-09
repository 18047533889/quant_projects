import numpy as np
import pytest

from quant_evaluator.kernels.gpu.exposure_evidence import gpu_neutralized_rank_ic
from quant_evaluator.metrics.exposure_evidence import ExposurePanel, compute_neutralized_rank_ic


def test_gpu_neutralized_rank_ic_uses_average_ties_and_final_joint_floor():
    pytest.importorskip("cupy")
    n = 40
    risk = np.tile(np.linspace(-1, 1, n)[None, :, None], (2, 1, 1))
    tied = np.tile(np.repeat(np.arange(4.0), 10)[None, :], (2, 1))
    factor = risk[:, :, 0] + tied
    forward = tied.copy()
    panel = ExposurePanel(risk, style_names=("size",))
    cpu = compute_neutralized_rank_ic(factor, forward, panel, min_obs=10)
    gpu = gpu_neutralized_rank_ic(factor, forward, risk, min_obs=10)
    assert gpu == pytest.approx(cpu, abs=1e-12)

    sparse_forward = np.full_like(forward, np.nan)
    sparse_forward[:, :5] = forward[:, :5]
    assert np.isnan(gpu_neutralized_rank_ic(
        factor, sparse_forward, risk, min_obs=10,
    ))
