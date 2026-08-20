# FactorEngine R11 Long-tail Closure Plan

Baseline: `ecffd565089e0979c73f039245f2bab9139f3951` (audit's stated SHA; HEAD verified identical).
Tree loads: **1362 canonicals / 0 unclassified**. Concurrent session holds 40 dirty files in
`runtime/`, `storage/`, `tests/runtime|storage/` (R10 work) — file-disjoint from this audit's targets.

## Status

- **WS-A committed `432f0c7`** (shared layer: declare_stateful, history_formula/semantics, DSL
  binder #16/#17/#18, cross_event stateless #11). 17 new tests pass.
- **WS-L §35 done**: `current_only` join method + SnapshotOnlySourcePolicy implemented in
  `composite_source.py` / `storage/factory.py` / `data_access_source.py`; default US valuation
  helper now constructs (3 integration tests pass).
- **WS-L §37 auditors done** (module + pytest wrapper): A prefix-invariance, B stateful-contract
  discovery, C default-parameter history, D unit-algebra (found 4 real `same_as:target` bugs;
  2 fixed in robust_scale.py, 2 flagged to WS-I/WS-K).
- **Agents B–K in flight** (10 parallel agents, file-disjoint). WS-B pivot ledger test landed.

## WS-A shared-layer contract (consumed by B/C/D)

| WS | Owner | Files | Findings |
|---|---|---|---|
| **A** | me (shared layer) | `runtime/execution_contract.py`, `cleaned_operators/base.py`, `cleaned_operators/contract_hardening.py`, `cleaned_operators/production_policy_extensions_v2.py`, `api/dsl_parser.py` | #11 cross_event stateless, #12 per-op ExecutionContract + CI, #13 ParamSpec history_semantics/formula, #14 compound history, #16 DSL numeric-string binder, #17 non-finite literal, #18 ComplexityBudget |
| **B** | agent | `structural_levels.py`, `extrema_divergence.py`, new `common/_pivot_ledger.py`, breakout/retest/support-resistance | #1 streaming pivot ledger (no retroactive rewrite, PivotSuperseded effective_at=t), #2 confirmed-extrema ledger, prefix-invariance tests |
| **C** | agent | `stateful/survival.py`, `stateful/sequential.py`, `state_event.py`, `stateful/events.py` | #3 survival trio stateful, #4 CUSUM recursive, #9 event_decay_asof, #10 time_since_change, #11 cross_event stateless (consume WS-A), #145 spacing NaN censor, #146 init semantics, #147 off-by-one, #148 EventBool/MarkedEvent |
| **D** | agent | `alpha_language_state.py`, `update_clock.py`, `report_timing.py` | #5 HysteresisStateKernel, #6 HysteresisMissingPolicy, #7/#44 update_clock event-count, #8 report_count, #45 acceleration scale, #46-#49 filing/revision event PIT + RevisionPair |
| **E** | agent | TE/MI + spectral + complexity + multifractal modules | #19-27, #80-91 |
| **F** | agent | `advanced_intraday.py`, `intraday_session.py`, `session_recovery.py`, `intraday_activity_duration.py` | #56-79, #183-186 |
| **G** | agent | `feature_geometry.py`, `directional_change.py`, `tail_systemic.py`, `extreme_tail.py`, `marked_event.py` | #29-43 |
| **H** | agent | fundamental + fiscal + report helpers | #50-55, #180-182 |
| **I** | agent | `gather_ext.py`, `group_ext.py`, `weighted_moment_ext.py`, `conditional_ext.py`, `cross_section_ext.py` | #131-144 |
| **J** | agent | `state_episode_excursion.py`, `downside_risk.py`, `return_decomp.py`, `hankel.py`, `vector_path.py`, fdiff | #149-164, #28, #92-96 |
| **K** | agent | relation, shareholder, rotation, coverage | #118-130, #153-157, #175-179 |
| **L** | agent | `api/mining_integration.py`, composite/coverage, shared auditors | §35 US valuation helper, §37 auditors A-I, §38 test suite |

## WS-A shared-layer contract (consumed by B/C/D)

- `runtime.execution_contract.declare_stateful(canonical, *, state_model, chunking,
  checkpoint_schema=None, minimum_history=0, history_kind="full_history")` — operators declare
  their own execution contract; `execution_contract()` resolves checkpoint-registry →
  declared → legacy seed. `declared_stateful_canonicals()` aggregates.
- `HistoryRequirement.kind` extended with `event_count` / `report_count` / `session_count`;
  those kinds are bar-incommensurable → resolve `full_history` (conservative) via
  `history_requirement()`.
- `ParamSpec.history_semantics` (exists) + new `history_formula` (e.g. `"outer_window +
  inner_window"`, `"2 * window"`) + `history_contribution`; `_own_history_extension` reads
  declared semantics/formula before the name-based default.
- DSL parser: strings never coerced at parse; `_visit_call` performs **ParamSpec-driven**
  controlled coercion (numeric dtype only). Non-finite float literal rejected.
  `ComplexityBudget` (max_ast_nodes/max_depth/max_call_arity/max_literal_magnitude/max_variadic_inputs).
- `cross_event` removed from stateful seed; declared `HistoryTransform(kind="lag", fixed=1)`.

## Verification gates (end of round)

1. `load_all` → 1362/0 intact (verify after each WS merge).
2. New targeted tests per WS (see §38 audit list; at least the 45 named cases).
3. `prefix invariance` property tests (WS-B), `full == chunk/checkpoint` parity (WS-C/D).
4. Full `pytest tests/` suite green; `scripts/audit_all_factor_production.py --strict`.
5. `python -m pytest` new shared tests (binder, DSL budget, declare_stateful CI).
6. Evidence re-generation deferred to after the tree settles (per round-7 closure plan),
   sequence: factor → primitive → recipe → 3 manifests → catalog.
