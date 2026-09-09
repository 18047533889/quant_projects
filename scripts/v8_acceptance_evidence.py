"""Reconcile all 71 tickets and 174 specifications without inventing passes."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/v8'
Q='quant_evaluator/tests/test_v8_numeric_goldens.py::'
R='quant_evaluator/tests/test_v8_risk_domain.py::'
EARLY_TICKETS={
 1:(15,16),2:(13,14),3:(17,30,111),4:(2,3,4,5),5:(9,10),6:(21,22),7:(20,115),
 8:(23,24),9:(6,7,8,11),10:(18,19),11:(25,),14:(56,),15:(12,30,111,115,120),
 16:(31,32),17:(33,),18:(34,35,36),19:(37,),20:(38,),21:(39,),22:(40,41),23:(42,43),
 24:(44,45,46),25:(47,48,49,50),26:(51,52),27:(53,54),28:(55,56),29:(57,58),30:(59,),
 31:(60,61),32:(62,63),33:(64,65,67,114,116,118)}
ROOT_PROOFS=(
 ((26,), 'VERIFIED', ['factor_optimizer/tests/search/test_v8_cross_layer_invariants.py'], 'Production QE permutation/single-column/F-shard outputs canonically aligned under one plan produce identical FA/FO decision identity.'),
 ((27,), 'VERIFIED', ['factor_optimizer/tests/search/test_v8_cross_layer_invariants.py'], 'Future-tail perturbation through the frozen fitted FP recipe leaves historical QE evidence and FA/FO decision identity unchanged in the executed synthetic domain.'),
 ((31,), 'VERIFIED', [Q+'test_t31_hac_bartlett_normalization'], 'Independent fixed-n Bartlett normalization witness.'),
 ((32,), 'VERIFIED', [Q+'test_t32_hac_matches_positive_semidefinite_quadratic_form'], 'Independent PSD quadratic-form n/lag grid, including smallest-eigenvector witness.'),
 ((33,), 'VERIFIED', ['quant_evaluator/tests/test_v8_clock_sink_acceptance.py::test_t33_missing_positions_never_acquire_compressed_hac_or_bootstrap_clock'], 'Missing-position witness checks original-clock behavior and explicit rejection, not a certification of every irregular-calendar estimator.'),
 ((34,35), 'VERIFIED', [Q+'test_t34_t35_joint_plan_identity_and_permutation'], 'Common plan, identical candidates and column/shard identities.'),
 ((36,), 'VERIFIED', ['quant_evaluator/tests/test_v8_clock_sink_acceptance.py::test_t36_singleton_repeats_short_span_and_oversized_blocks_cannot_be_evidence'], 'Singleton paired-bootstrap repetitions now reject; short time span and oversized blocks cannot become statistical evidence.'),
 ((37,), 'VERIFIED', [Q+'test_t37_invalid_values_cannot_change_rank_stability'], 'Invalid finite-buffer contamination leaves rank stability unchanged; existing validity suites cover turnover.'),
 ((38,), 'VERIFIED', [Q+'test_t38_nonmonotone_lag_pair_counts',Q+'test_t38_lag_counts_are_explicit'], 'Later valid lag survives earlier missing lag; pair counts are explicit.'),
 ((39,), 'VERIFIED', [Q+'test_t39_centered_persistence_translation'], 'Centered AR persistence translation invariant; zero-mean model remains distinct.'),
 ((40,41), 'VERIFIED', [Q+'test_t40_t41_membership_units_and_unknown_exit'], 'Own-day denominators and unknown exit coverage, not future intersection.'),
 ((47,48,49,50), 'VERIFIED', [Q+'test_t47_t50_factor_validity_storage_identity'], 'Typed FactorBatch and AxisRef source identity/owned immutable buffers; not a claim for all bare-array functions.'),
 ((55,), 'VERIFIED', [Q+'test_t55_johansen_reference'], 'Reference grid K1/2/3/6/12 x det(-1,0,1) x VAR-lag(1,2,4) x alpha(.01,.05,.10); statistics/eigenvalues and critical-table-derived rank checked. K1 deterministic domain explicitly rejected, not certified.'),
 ((56,), 'VERIFIED', [Q+'test_a14_ols_cusum_reference_and_unsupported_recursive_scope'], 'Unsupported recursive CUSUM rejected and uncalibrated Sup-Wald p is unavailable.'),
 ((64,67), 'VERIFIED', [Q+'test_t64_t67_qualification_domain_is_not_portable'], 'Execution-domain mismatch/SKIP cannot qualify; inventory not numerical certification.'),
 ((65,), 'VERIFIED', ['quant_evaluator/tests/metrics/test_coverage_compiler.py'], 'Required metric resolution fails closed; no silently filtered request.'),
 ((71,72), 'VERIFIED', [Q+'test_t71_t72_quantile_axes'], 'One-dimensional cross-section and F=1 standard/fast/Numba MIN/MAX goldens.'),
 ((73,74), 'VERIFIED', [Q+'test_t73_t74_t75_real_cuda_parameter_domain'], 'Real L20 nondefault MIN/MAX, finite mask and nondefault rank axis.'),
 ((75,), 'VERIFIED', [Q+'test_quantile_returns_bounded_workspace_matches_full','evidence/v8/gpu_resource_probe.json'], 'Q10/20/40 tiled scratch; measured allocator peaks for stated T/F/N only.'),
 ((76,), 'VERIFIED', ['quant_evaluator/tests/test_v8_clock_sink_acceptance.py::test_t76_oom_retile_preserves_numbers_and_never_recounts_committed_tile'], 'Real CUDA OOM injected on second production tile; first committed tile is not repeated, sink persists output and numbers equal no-OOM run.'),
 ((77,), 'VERIFIED', [Q+'test_t77_fixed_tail_mass'], 'ES95=.02 for one -.10 and 99 zeros.'),
 ((78,), 'VERIFIED', [Q+'test_t78_fractional_tail_mass'], 'Fractional boundary mass and replicated empirical distribution.'),
 ((79,), 'VERIFIED', [Q+'test_t79_unconditional_partial_moments'], 'Independent 90/10 and 10/90 unconditional UPR goldens.'),
 ((80,), 'VERIFIED', [R+'test_t80_upr_mar_translation_and_undefined_denominator'], 'MAR translation and all-positive/all-negative/zero denominator behavior.'),
 ((81,), 'VERIFIED', [Q+'test_t81_no_implicit_mixed_var_es_model'], 'Unsupported CF joint VaR/ES explicitly rejected; no hidden estimator fallback.'),
 ((82,), 'VERIFIED', [R+'test_t82_es_ratio_has_no_mechanical_annualization_or_absolute_epsilon'], 'Same-period ES ratio has no sqrt annualization or absolute epsilon cutoff.'),
 ((84,), 'VERIFIED', [R+'test_t84_psd_domain_and_rank_deficient_target_are_explicit'], 'Non-PSD/singular/missing rejected; rank-deficient PSD target achieved and marked diagnostic.'),
 ((85,), 'VERIFIED', [R+'test_t85_research_trajectory_is_typed_but_not_executable_certification'], 'Naked POSITION_REPLAY string rejected; real trajectory bridge keeps research non-executable.'),
 ((86,), 'VERIFIED', [R+'test_t86_event_identity_is_content_bound_order_invariant_not_value_dedup'], 'Canonical named event aliases only; content-bound JSON identity, candidate independence and conflict rejection.'),
 ((87,88), 'VERIFIED', [Q+'test_t87_sidak_tiny_values'], 'Stable small-p Sidak with log1p/expm1 and preserved endpoints.'),
 ((90,), 'VERIFIED', [Q+'test_t89_t90_complete_family_never_shrinks_failed_members'], 'Full family slots retained for failed/pending, no fabricated measured p.'),
 ((107,), 'VERIFIED', [Q+'test_t107_peak_not_underwater_trough'], 'Exact highwater peak does not include slightly underwater trough.'),
 ((108,), 'VERIFIED', ['quant_evaluator/tests/test_v3_drawdown_contracts.py',R+'test_t108_tiny_loss_peak_duration_and_underwater_statistics_share_events'], 'Shared exact highwater event for tiny drawdown/peak/duration/underwater fraction; initial capital, recovery and default checked.'),
 ((110,), 'VERIFIED', [R+'test_t110_tail_counts_and_small_samples_do_not_claim_confidence'], 'Descriptive finite-q counts with insufficient/degenerate status.'),
 ((116,), 'PARTIAL', [Q+'test_t116_method_version_changes_identity_and_rejects_stale_serialized_instance',Q+'test_t64_t67_qualification_domain_is_not_portable'], 'Simulated same-name method upgrade changes instance identity, rejects stale serialized instances/receipts and leaves unrelated instance unchanged; no full downstream dependency-invalidation migration executed.'),
 ((117,), 'PARTIAL', [R+'test_t86_event_identity_is_content_bound_order_invariant_not_value_dedup','factor_assets/tests/test_dlib_fa_018_composite.py'], 'Common diagnostic scenarios and non-backtest composition boundaries tested; full final-portfolio stress replay requires authorized execution scenario data.'),
 ((118,), 'VERIFIED', [Q+'test_t73_t74_t75_real_cuda_parameter_domain',Q+'test_t64_t67_qualification_domain_is_not_portable'], 'Real nondefault CUDA goldens and mismatch rejection for each of six identity fields including parameter domain; no qualification expansion to untested combinations.'),
 ((119,), 'PARTIAL', ['evidence/v8/gpu_resource_probe.json'], 'Measured T32/F8/N5000 Q10/20/40 only. No 100000-factor throughput/peak certification claimed.'),
 ((160,), 'VERIFIED', ['evidence/v8/cache_resource_instrumented_candidate2.json'], 'One million actual put/clear cycles, bounded history, measured RSS scope.'),
 ((161,), 'VERIFIED', [Q+'test_t161_cache_read_checks_budget_before_decompression',Q+'test_cache_dict_budget_and_custom_array_reducer_are_rejected_before_pickle','evidence/v8/cache_resource_instrumented_candidate2.json'], 'Admission before pickle/decompression for compressible/incompressible/dict/custom reducer payloads.'),
 ((162,), 'VERIFIED', [Q+'test_t162_cache_replacement_does_not_evict_other_key'], 'Old same-key bytes deducted before unrelated eviction.'),
 ((163,164), 'VERIFIED', [Q+'test_t163_t165_actual_device_storage_and_transfer_capability'], 'Actual staged FP64/MIXED dtype; unsupported required async/multidevice rejected; effective synchronous transfer recorded.'),
 ((165,), 'UNSUPPORTED', [Q+'test_t163_t165_actual_device_storage_and_transfer_capability'], 'Async lifetime not implemented or advertised. Mandatory async fails closed; synchronous isolation tested.'),
)

def main():
    task=(OUT/'input/task.md').read_text()
    tickets={m.group(1):dict(title=m.group(2),status='PENDING_RECONCILIATION',evidence=[])
             for m in re.finditer(r'^### (A\d{2}) · (.+)$',task,re.M)}
    specs={}
    for line in task.splitlines():
        match=re.match(r'^\| (T\d+) \|',line)
        if match:
            cells=[c.strip() for c in line.strip('|').split('|')]
            specs[match.group(1)]={'input':cells[1],'required':cells[2],
                'ticket_refs':re.findall(r'A\d{2}',cells[3]) if len(cells)>3 else [],
                'status':'PENDING_RECONCILIATION','tests':[],
                'evidence':'No per-spec executable closure recorded yet.'}
    if len(tickets)!=71 or len(specs)!=174:
        raise RuntimeError('task-book topology changed')
    for relative in ('fo/acceptance_traceability.json','fp_regime/acceptance_traceability.json',
                     'fa/acceptance_traceability.json','fa/fa_acceptance_map.json','fa/acceptance_ledger.json'):
        path=OUT/relative
        if not path.exists(): continue
        payload=json.loads(path.read_text())
        rows=payload.get('acceptances',payload.get('entries',payload.get('criteria',{})))
        if isinstance(rows,list): rows={row.get('id',row.get('spec_id')):row for row in rows}
        for key,row in rows.items():
            if key in specs:
                row=dict(row)
                if 'selectors' in row: row['tests']=row['selectors']
                if 'tests' not in row and isinstance(row.get('evidence'),list):
                    row['tests']=row['evidence']
                if 'evidence' not in row:
                    row['evidence']=row.get('gap') or row.get('domain') or 'See referenced executable assertions; scope is limited to those cases.'
                specs[key].update(row,owner_evidence=relative)
        for key,row in payload.get('tickets',{}).items():
            if key in tickets: tickets[key].update(row,owner_evidence=relative)
    for ids,status,tests,note in ROOT_PROOFS:
        for number in ids:
            specs[f'T{number:02d}'].update(status=status,tests=tests,evidence=note,owner_evidence='root-scoped-review')
    for key in ('T70','T120'):
        specs[key].update(status='BLOCKED_DATA',evidence='Awaiting user-designated unsealed development data, dates and universe for economic calibration and negative-control shadow comparison.')
    for number,refs in EARLY_TICKETS.items():
        for ref in refs:
            key=f'T{ref:02d}'
            if f'A{number:02d}' not in specs[key]['ticket_refs']:
                specs[key]['ticket_refs'].append(f'A{number:02d}')
    for key,row in tickets.items():
        linked={k:v for k,v in specs.items() if key in v['ticket_refs']}
        row['acceptance_ids']=list(linked)
        row['evidence']=list(dict.fromkeys(str(t) for v in linked.values() for t in v.get('tests',[])))
        if linked:
            statuses={v['status'] for v in linked.values()}
            row['owner_reported_status']=row['status']
            row['status']='VERIFIED' if statuses=={'VERIFIED'} else 'PARTIAL'
        row['historical_disposition']='Preserve original artifacts; changed numeric definitions require affected-lineage recomputation, policy-only changes reuse valid raw metrics. No production mutation performed.'
        row['rollback']='Use frozen source/diff as a coherent version; do not mix old policy receipts with new implementations.'
    for ticket,test in (('A12','test_a12_no_feasible_additional_break'),('A13','test_a13_rank_deficient_chow_rejected')):
        tickets[ticket].update(status='VERIFIED',evidence=[Q+test],scope='Scoped public numeric counterexample regression; not all structural-break inference certified.')
    payload=dict(task_sha256=hashlib.sha256(task.encode()).hexdigest(),tickets=tickets,acceptances=specs,
        counts={status:sum(r['status']==status for r in specs.values()) for status in sorted({r['status'] for r in specs.values()})},
        caution='Scoped tests are not universal market qualification. PENDING/PARTIAL/UNSUPPORTED are not passes. Final regression execution is recorded separately.')
    (OUT/'acceptance_ledger.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload['counts']))

if __name__=='__main__': main()
