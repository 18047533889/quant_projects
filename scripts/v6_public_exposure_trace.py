#!/usr/bin/env python3
"""Actual CPU/CUDA exposure artifacts on explicitly synthetic independent gold."""
import hashlib
import json
from pathlib import Path
import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.api.requests import EvaluationBundle
from quant_evaluator.scripts.generate_capability_matrix import CapabilityRunEvidence, generate_capability_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v6'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t, n = 10, 80
    rng = np.random.default_rng(20260908)
    q = np.linalg.qr(np.column_stack((np.ones(n), rng.normal(size=(n, 7)))))[0] * np.sqrt(n)
    risk = np.tile(q[None, :, 1:7], (t, 1, 1))
    beta = np.arange(1, 7) / 6
    signal = risk @ beta + q[:, 7]
    styles = ('industry', 'size', 'beta', 'liquidity', 'volatility', 'momentum')
    times = tuple(range(t)); assets = tuple(f'a{i}' for i in range(n))
    batch = FactorBatch(('synthetic',), AxisRef('time', 'int', t, np.arange(t)),
                        AxisRef('asset', 'object', n, np.array(assets, dtype=object)), signal[:, :, None])
    labels = LabelBundle('synthetic_alpha', np.tile(q[None, :, 7], (t, 1)), 1,
                         decision_time=times, label_start_time=tuple(range(1, t+1)),
                         label_end_time=tuple(range(2, t+2)))
    panel = ExposurePanel(risk, style_names=styles, source_ref='synthetic:risk:20260908',
                          provider='independent_qr_gold', date_index=times, security_ids=assets,
                          factor_ids=batch.factor_ids, universe_snapshot_ref='synthetic:80')
    metrics = tuple(s + '_exposure' for s in styles) + ('max_absolute_style_exposure', 'purity_ratio', 'exposure_drift')
    evidence = []; hashes = {}
    for backend in ('cpu', 'cuda_strict'):
        mids = metrics + (('neutralized_rank_ic', 'residual_rank_ic') if backend == 'cpu' else ())
        bundle = evaluate(batch, labels, metrics=mids, exposure_panel=panel, backend=backend)
        for style, coefficient in zip(styles, beta):
            np.testing.assert_allclose(bundle.get_metric(style + '_exposure', 'synthetic').value,
                                       coefficient / np.sqrt(np.sum(beta**2) + 1), atol=1e-12)
        np.testing.assert_allclose(bundle.get_metric('purity_ratio', 'synthetic').value,
                                   1 / (np.sum(beta**2) + 1), atol=1e-12)
        raw = json.dumps(bundle.to_dict(), sort_keys=True).encode()
        restored = EvaluationBundle.from_dict(json.loads(raw))
        np.testing.assert_array_equal(restored.artifacts['purity_ratio'].values, bundle.artifacts['purity_ratio'].values)
        name = f'x01_public_{backend}_bundle.json'
        (OUT / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        factory = CapabilityRunEvidence.from_public_cpu if backend == 'cpu' else CapabilityRunEvidence.from_public_gpu
        evidence.extend(factory(bundle, mid, source_ref=f'evidence/v6/{name}', data_identity='synthetic_qr_20260908') for mid in mids)
    generate_capability_matrix(str(OUT / 'metric_capability_matrix.csv'), run_evidence=evidence)
    generate_capability_matrix(str(ROOT / 'quant_evaluator/docs/METRIC_CAPABILITY_MATRIX.csv'), run_evidence=evidence)
    (OUT / 'x01_public_trace_hashes.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
