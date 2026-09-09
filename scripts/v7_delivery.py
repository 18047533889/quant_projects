"""Consolidate immutable V7 receipts; never mutate production or old evidence."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence/v7'


def main():
    tickets = {}
    for name in ('root_ticket_closure.json', 'fa/ticket_closure.json', 'fo_fp_13_ticket_closure.json'):
        content = json.loads((E / name).read_text())['tickets']
        if isinstance(content, list):
            content = {row['id']: row for row in content}
        assert not tickets.keys() & content.keys(), name
        tickets.update(content)
    for ticket, contract in {
        'R35': 'Existing durable run authority owns reservation/attempt/lease/terminal state. Context-local tokens, fenced publication, atomic consumption, and durable read-through prevent stale workers and duplicate consumption.',
        'R36': 'Existing generation authority stores complete serialized FeatureSet snapshots. MERGE preserves previous definitions and replaces changed provenance; explicit REPLACE, parent CAS, restart recovery and crash replay are tested.',
    }.items():
        tickets[ticket] = dict(status='INTEGRATION_VERIFIED', contract=contract,
            actual_test_log=['platform_frozen_final.log', 'r35_r36_targeted.log',
                             'r35_r36_postgres_live.log', 'r35_r36_postgres_reservations.log'],
            history_impact='Audit incomplete generation/reservation histories; no production migration or active-pointer mutation performed.',
            rollback='Preserve consumed/attempt records and immutable generations; never reset a used lease or sealed exposure.')
    expected = {f'N{i:02}' for i in range(1,14)} | {f'R{i:02}' for i in range(1,37)}
    assert tickets.keys() == expected, (expected-tickets.keys(), tickets.keys()-expected)
    tickets['R19']['actual_test_log'] = ['qe_frozen_final.log', 'root_acceptance.log',
        'stream256.json', 'stream8192.json', 'maturity64_final/receipt.json', 'maturity1024_final/receipt.json']
    tickets['R19']['status'] = 'IMPLEMENTED_SUBSET_VERIFIED_CAPABILITY_LIMITED'
    tickets['R19']['contract'] = ('Per-factor built-in IC/coverage/centered summaries; bounded synchronous IC sink; '
        'maturity queue supports exact daily Pearson/Spearman IC, preserved all_history and typed rolling_observations '
        'using last-W durable-row bounded recomputation including missing observations, restart and immutable revisions. '
        'Unknown windows fail closed. Source time/factor shards are bounded and capabilities explicit.')
    tickets['R19']['capability_limits'] = ('No algebraic remove or cross-stream merge; rolling uses exact bounded recomputation. '
        'Maturity queue certifies IC only; multi-day shards wait for maximum label maturity. Restated streams require '
        'complete target-history replay and do not certify replay completeness. StreamingEvaluator itself has no '
        'source checkpoint; large_batch still takes a full host panel. No all-metric/multi-GPU online certification.')
    tickets['R22']['history_impact'] = 'Code semantics verified; empirical production-grade calibration requires a designated unsealed development dataset, dates and universe. Sealed test data must not be used for calibration.'
    historical = {
        'N06': 'Review forward experiments whose training labels matured after validation decision; exposed holdouts cannot regain independence.',
        'N07': 'Reclassify experiments with explicit nonzero embargo that took the old timestamp early-return path.',
        'N08': 'Retain repeated-exposure campaigns and downgrade independent-test evidence; never clear attempt/exposure records.',
        'N09': 'Quarantine freeze/execution mismatches; rebuilding a handle cannot restore an exposed holdout.',
        'R05': 'Invalidate fabricated axis-alignment evidence and dependent grades, not valid raw observations.',
        'R06': 'Replace cross-factor dispersion used as temporal uncertainty; preserve valid point estimates.',
        'R07': 'Replace false zero-width intervals and dependent grades only.',
        'R14': 'Audit anomalous tier jumps and replay occurrences; no blanket invalidation.',
        'R15': 'Review invalid representation destinations; preserve canonical factor values.',
        'R31': 'Recompute values/evaluations where parameters were dropped or axes changed.',
        'R32': 'Recompute erroneous values; identity/policy-only changes create new evidence without overwriting history.',
        'R33': 'Re-evaluate affected winner decisions without reopening sealed tests.',
        'R34': 'Downgrade unresolvable bare-score evidence; retain reusable typed resolvable evidence.',
    }
    for key, row in tickets.items():
        row['ticket'] = key
        row['source_manifest'] = 'final_source/source_manifest.json'
        row.setdefault('history_impact', historical.get(key, 'See ticket-specific evidence.'))
        row.setdefault('rollback', 'Retain historical evidence and immutable refs; no production deployment or migration performed. Roll back only scoped source changes, never erase test exposure or consumed reservations.')
    manifest = (E/'final_source/source_manifest.json').read_bytes()
    payload = dict(schema='v7-ticket-closure.v1', runtime_checkout=str(ROOT),
        input_sha256='581fd57a150140c5d6332e18cda231a0524ddf06b5aef7f08eb7f3cd7124b009',
        source_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        production_changed=False, tickets={key:tickets[key] for key in sorted(tickets)},
        limitations=[
            'R22 empirical production grading calibration is blocked pending a designated unsealed development dataset, dates and universe.',
            'Single NVIDIA L20 only; no multi-GPU certification.',
            '100k-factor numerical run used T=4,N=20 and two metric instances; it is not decade-by-5000-stock-by-all-metric certification.',
            'Historical impact dry-run used synthetic references; real historical inventory and production migration were not executed.',
            'Runtime tests import remediation source explicitly; installed wheels, build copies and production services were not updated.',
        ])
    (E/'ticket_closure.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    suites = {
        'QE': 'final_regression/qe.log',
        'FO': 'final_regression/fo.log',
        'FP': 'final_regression/fp.log',
        'FA_main': 'final_regression/fa_main.log',
        'FA_isolated_concurrency': 'final_regression/fa_concurrency.log',
        'Platform': 'final_regression/platform.log',
        'Modeling_jobs': 'final_regression/modeljobs.log',
        'VectorBT': 'final_regression/vectorbt.log',
        'PostgreSQL_live': 'r35_r36_postgres_live.log',
    }
    results = {}
    for suite, name in suites.items():
        raw = (E/name).read_text()
        lines = [line.strip('= ') for line in raw.splitlines()
                 if re.search(r'\b\d+ passed\b',line)]
        assert lines, name
        assert not re.search(r'\b\d+ failed\b',lines[-1]), lines[-1]
        results[suite] = dict(log=name, result=lines[-1],
                             log_sha256=hashlib.sha256((E/name).read_bytes()).hexdigest())
    (E/'regression_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    snapshot = json.loads(manifest)
    mismatches = [r['path'] for r in snapshot['files']
                  if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() != r['sha256']]
    (E/'final_source/source_recheck.json').write_text(json.dumps(dict(
        files_checked=len(snapshot['files']), mismatches=mismatches),indent=2)+'\n')
    assert not mismatches, mismatches
    print(json.dumps(dict(tickets=len(tickets),files_checked=len(snapshot['files']),
                         source_manifest_sha256=payload['source_manifest_sha256']),indent=2))


if __name__ == '__main__':
    main()
