"""Isolated reproductions of reviewed expressions, not quant_projects pytest.

No project modules, market database or GPU are used. A successful assertion
means the described old behaviour / counterexample was reproduced. It does
not mean the repository has been fixed or its production path is affected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

HEAD = '2bac7209c00e302728b810513c91f5475ec6954f'


def corr(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def zscore_old(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def old_confidence_band(train, valid):
    # Reviewed _confidence_band expression, inputs are per-factor summaries.
    train, valid = np.asarray(train, float), np.asarray(valid, float)
    if len(train) < 2 or len(valid) < 2:
        return (None, None)
    se_t = np.std(train, ddof=1) / np.sqrt(len(train))
    se_v = np.std(valid, ddof=1) / np.sqrt(len(valid))
    mt, mv = np.mean(train), np.mean(valid)
    if abs(mt) < 1e-12:
        return (None, None)
    rel = mv / mt
    width = 1.96 * np.sqrt((se_v / max(abs(mv), 1e-12)) ** 2
                           + (se_t / abs(mt)) ** 2) * abs(rel)
    return float(rel - width), float(rel + width)


def old_mmr(ids, quality, similarity):
    """Reviewed greedy loop; callers must supply the effective quality scores."""
    remaining, selected = list(ids), []
    calls = 0
    while remaining:
        best, best_score = None, float('-inf')
        for fid in remaining:
            sims = []
            for chosen in selected:
                calls += 1
                sims.append(abs(float(similarity(fid, chosen))))
            maximum = max(sims) if sims else 0.0
            score = .5 * quality[fid] - .5 * maximum
            if score > best_score:
                best, best_score = fid, score
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected, calls


def run():
    out = []
    def add(pid, issue, observed, required):
        out.append({'probe': pid, 'issue': issue, 'observed': observed,
                    'required_contract': required,
                    'scope': 'isolated_expression_not_repository_test'})

    # X01: an exposure panel centred at zero yields purity 1, regardless of x.
    size = np.tile(np.array([-2., -1., 0., 1., 2.]), 12)
    mu, variance = size.mean(), np.mean((size - size.mean()) ** 2)
    purity = 1. - mu ** 2 / (mu ** 2 + variance)
    true_loading = corr(size, size)
    assert purity == 1. and true_loading == 1.
    add('P01', 'X01', {'reported_size_mean': float(mu), 'reported_purity': purity,
                      'correlation_factor_equals_size': true_loading},
        'Factor exposure must depend on factor values or actual portfolio weights.')

    # X02: orthogonal signals combine perfectly but are labelled substitutable.
    a = np.tile(np.array([-1., -1., 1., 1.]), 16)
    b = np.tile(np.array([-1., 1., -1., 1.]), 16)
    y = a + b
    ic_a, ic_b = corr(a, y), corr(b, y)
    ic_comb = corr((zscore_old(a) + zscore_old(b)) / 2, y)
    ratio = abs(ic_comb) / (max(abs(ic_a), abs(ic_b)) + 1e-8)
    assert abs(corr(a, b)) < 1e-12 and ratio > 1.4 and ratio >= .9
    add('P02', 'X02', {'corr_a_b': corr(a, b), 'ic_a': ic_a, 'ic_b': ic_b,
                      'ic_combined': ic_comb, 'substitution_ratio': ratio,
                      'old_substitutable': bool(ratio >= .9)},
        'Improvement and redundancy are different relations; no one-sided >= ratio gate.')

    # X03: a constant-on-support target changes the sample, not the prediction.
    x = np.tile(np.array([-1., 1.]), 32)
    target = np.r_[np.ones(32), np.full(32, np.nan)]
    y = np.r_[x[:32], -x[32:]]
    overlap = np.isfinite(target)
    base_ic = corr(zscore_old(x), y)
    combined = (zscore_old(x[overlap]) + zscore_old(target[overlap])) / 2
    combo_ic = corr(combined, y[overlap])
    paired_base_ic = corr(zscore_old(x[overlap]), y[overlap])
    assert abs(base_ic) < 1e-12 and abs(combo_ic - 1) < 1e-12
    assert abs(combo_ic - paired_base_ic) < 1e-12
    add('P03', 'X03', {'baseline_full_ic': base_ic, 'combined_overlap_ic': combo_ic,
                      'old_apparent_lift': combo_ic - base_ic,
                      'common_support_lift': combo_ic - paired_base_ic},
        'Compare on common support; separately evaluate availability/coverage information.')

    # X04: cross-sectional pre-gate uses all factors, not the requested pair.
    ab = np.column_stack([x, x])
    abc = np.column_stack([ab, np.full(len(x), np.nan)])
    old_n_ab = int(np.all(np.isfinite(ab), axis=1).sum())
    old_n_abc = int(np.all(np.isfinite(abc), axis=1).sum())
    assert old_n_ab == 64 and old_n_abc == 0
    add('P04', 'X04', {'all_finite_before': old_n_ab, 'all_finite_after': old_n_abc,
                      'pairwise_n_unchanged': 64, 'pairwise_corr': corr(x, x)},
        'Appending an unrelated missing column must not invalidate a valid pair.')

    # X05: np.resize repeats inputs, including a synthetic zero for empty input.
    train, valid = np.array([.1, .2, .3]), np.array([.08, .16])
    resized = np.resize(valid, len(train))
    empty = np.resize(np.array([], dtype=float), 3)
    assert resized.tolist() == [.08, .16, .08] and np.all(empty == 0)
    add('P05', 'X05', {'validation_before': valid.tolist(),
                      'validation_after': resized.tolist(),
                      'empty_becomes': empty.tolist()},
        'Reject mismatched factor IDs/shapes; never repeat measurements to align evidence.')

    # X06: repeating the same factor rows falsely narrows the supposed CI.
    train, valid = np.array([.1, .2]), np.array([.08, .19])
    ci2 = old_confidence_band(train, valid)
    ci200 = old_confidence_band(np.tile(train, 100), np.tile(valid, 100))
    w2, w200 = (ci2[1] - ci2[0]) / 2, (ci200[1] - ci200[0]) / 2
    assert w200 < w2 / 10
    add('P06', 'X06', {'two_factor_ci': ci2, 'repeated_rows_ci': ci200,
                      'width_ratio': w200 / w2},
        'Per-factor temporal evidence cannot become more certain by duplicating factor rows.')

    # X07: validation mean 0 with real variation collapses old width to zero.
    train, valid = np.array([.1, .1, .1]), np.array([-.02, 0., .02])
    ci = old_confidence_band(train, valid)
    derivative_se = float(np.std(valid, ddof=1) / np.sqrt(3) / .1)
    assert ci == (0., 0.) and derivative_se > 0
    add('P07', 'X07', {'old_ci': ci, 'nonzero_ratio_delta_se': derivative_se},
        'Zero numerator mean does not remove its uncertainty; this is algebra, not CI calibration.')

    # X08: in-sample OLS fit of random labels is not OOS prediction evidence.
    rng = np.random.default_rng(20260908)
    n, k = 64, 20
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
    random_y = rng.normal(size=n)
    beta = np.linalg.lstsq(X, random_y, rcond=None)[0]
    pred = X @ beta
    Xt = np.column_stack([np.ones(4096), rng.normal(size=(4096, k))])
    yt = rng.normal(size=4096)
    add('P08', 'X08', {'same_sample_fitted_ic': corr(pred, random_y),
                      'independent_sample_ic': corr(Xt @ beta, yt)},
        'Keep same-date partial association as diagnostic; OOS model delta is a separate artifact.')

    # X09: duplicate controls are singular but span a legitimate projection.
    control = np.tile(np.array([-1., 1.]), 32)
    residual = np.tile(np.array([-1., -1., 1., 1.]), 16)
    X = np.column_stack([np.ones(64), control, control])
    y = control + residual
    normal_failed = False
    try:
        np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        normal_failed = True
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    assert normal_failed and rank == 2 and np.allclose(y - X @ beta, residual)
    add('P09', 'X09', {'normal_equation_failed': normal_failed,
                      'rank': int(rank), 'projection_residual_matches': True},
        'Use a rank-aware existing solver; report redundancy and residual degrees of freedom.')

    # X10: metadata fallback outranks measured quality values on another scale.
    ids = ['missing_quality', 'measured_A', 'measured_B']
    quality = {'missing_quality': float(len(ids)), 'measured_A': .9, 'measured_B': .8}
    ranked, _ = old_mmr(ids, quality, lambda a, b: 0.)
    assert ranked[0] == 'missing_quality'
    add('P10', 'X10', {'effective_quality': quality, 'ranking': ranked},
        'Missing quality is missing evidence, not candidate-count-sized quality.')

    # X11: one NaN comparison silently removes an otherwise valid candidate.
    ids = ['A', 'B']
    ranked, _ = old_mmr(ids, {'A': .9, 'B': .8}, lambda a, b: float('nan'))
    assert ranked == ['A']
    add('P11', 'X11', {'input_candidates': ids, 'output_candidates': ranked},
        'Nonfinite/out-of-domain similarity must become an explicit error or unresolved state.')

    # X12: source loop recomputes max similarities for the entire full ranking.
    n = 20
    _, calls = old_mmr(list(range(n)), {i: 1 - i/(2*n) for i in range(n)}, lambda a, b: 0.)
    assert calls == n * (n-1) * (n+1) // 6
    f = 100_000
    add('P12', 'X12', {'actual_calls_n20': calls,
                      'full_rank_call_formula_n100000': f*(f-1)*(f+1)//6,
                      'formula_only_not_100k_benchmark': True},
        'Stop after K feasible selections and maintain incremental max redundancy, about O(F*K).')

    # X13: factors removed later still alter the MMR greedy path.
    ids = ['A', 'B', 'C', 'D']
    quality = {'A': 1., 'B': .99, 'C': .8, 'D': .7}
    family = {'A': 'same', 'B': 'same', 'C': 'C', 'D': 'D'}
    def sim(a, b):
        return .9 if {a, b} == {'B', 'C'} else 0.
    ranking, _ = old_mmr(ids, quality, sim)
    picked, seen = [], set()
    for fid in ranking:
        if family[fid] not in seen:
            picked.append(fid); seen.add(family[fid])
    late = picked[:2]
    feasible_next = max(['C', 'D'], key=lambda c: .5*quality[c]-.5*sim(c, 'A'))
    proper = ['A', feasible_next]
    assert late == ['A', 'D'] and proper == ['A', 'C']
    add('P13', 'X13', {'full_ranking': ranking, 'late_constraints_top2': late,
                      'feasibility_aware_greedy_top2': proper},
        'Check constraints before adding a member to the set used for redundancy penalties.')

    # X14: inspected advance trusts tier_name instead of comparing current tier.
    actual_stage, supplied_stage = 'Tier1', 'Tier4'
    tiers = ['Tier1', 'Tier2', 'Tier3', 'Tier4']
    idx = tiers.index(supplied_stage)
    consecutive = 0 + 1
    completed = idx + 1 == len(tiers) and consecutive >= 0
    assert actual_stage != supplied_stage and completed
    add('P14', 'X14', {'stored_stage': actual_stage, 'supplied_stage': supplied_stage,
                      'old_completed': completed},
        'Reject outcomes for a stage that was not issued; deduplicate outcome IDs.')

    # X15: the parent ref is screened, while the destination ID is not.
    canonical_prefix = 'factor_asset:'
    legal_parent, legal_destination = 'factor_asset:alpha_v1', 'model_representation:x'
    bad_destination, alternative_parent = 'factor_asset:alpha_v1', 'resolved_ref:alpha_v1'
    add('P15', 'X15', {'legal_parent_rejected_by_prefix': legal_parent.startswith(canonical_prefix),
                      'legal_destination': legal_destination,
                      'bad_destination': bad_destination,
                      'bad_destination_not_caught_by_parent_check': not alternative_parent.startswith(canonical_prefix)},
        'Validate parent as source and destination as model representation; actual storage overwrite needs its own guard.')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('probe_results.json'))
    args = parser.parse_args()
    results = run()
    document = {'head': HEAD, 'test_scope': 'isolated_expressions_only',
                'count': len(results), 'repository_tested': False,
                'gpu_tested': False, 'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'count': len(results), 'output': str(args.out),
                      'scope': document['test_scope']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
