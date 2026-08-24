# New-operator authoring contract (2026-08 geometry/math expansion)

Authoring pattern = EXACTLY `cleaned_operators/interval_geometry.py` (the reference
module). Read it first. Every new module must:

1. **Own file only.** Write ONLY your assigned new module file(s) under
   `cleaned_operators/`. NEVER edit any existing file. No shared-file edits.
2. **Template** (copy from interval_geometry.py):
   - imports: `from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator`
   - `from cleaned_operators.rolling_pack import frame_like, register_polars_bridge`
   - a module-level `_metadata(name, description, params, *, unit, cost)` helper returning
     `OperatorMetadata` with tags containing `f"signature:{','.join(params)}->series"`,
     `"daily","pit_safe","causal","typed_v2","deterministic"`, `f"unit:{unit}"`, `f"cost:{cost}"`.
   - one `SeriesOperator` subclass per operator, decorated
     `@register_operator(name=..., category=..., business_category=..., canonical=..., source=<your_module_name>)`,
     with `metadata = _metadata(...)` and `def _calculate_series(self, <panels>, <params>, **_: Any) -> pd.DataFrame`.
   - bottom: `_NEW_CANONICALS = (...)` tuple, `def _register_surface()` that unions names into
     `cleaned_operators.operator_surface.EXTENDED_ONLY_CANONICALS` and calls
     `register_polars_bridge(canon)` for each, then `_register_surface()`.
   - module docstring explaining the family + shared kernels + PIT/causal/deterministic contract.
3. **Numeric contract**:
   - trailing-window per-column loops; prefix-causal (row r uses rows <= r only); deterministic.
   - all-NaN / degenerate window → np.nan. Never emit Inf or fabricate zeros.
   - invalid params raise ValueError (window<2, bins<2, k<=0, order<1, etc.).
   - panel inputs are pd.DataFrame (TradeDate x Symbol); use `.to_numpy(dtype=float)`.
   - output via `frame_like(template_frame, array)`.
   - Integer params auto-normalized by base: window,period,periods,d,lag,n,m,k,min_periods,
     bins,order,degree,max_lag. Still defensively `int(...)` them in kernels.
   - multi-input ops: kernels iterate per column; keep aligned pairs only.
   - across-input panel alignment (index/columns equal) is auto-validated by the base — do not
     re-validate, but do NOT add `allow_panel_broadcast` tag.
4. **Verification (mandatory before reporting)**:
   - `python3` import your module standalone (sys.path with quant_projects + factor_engine).
   - for each canonical: `OperatorRegistry.get(name,'pandas_numpy')` is not None;
     `'polars' in OperatorRegistry.backends_for(name)`.
   - execute each op on a synthetic panel (60 rows x 2 cols, random-walk close/high/low),
     assert finite fraction sane and no exception. Stateful ops: assert value is 0/positive
     bounded and finite.
   - Report: file path, canonical list, backends, smoke-test result lines.
5. **Do NOT** run `cleaned_operators.load_all()`, do NOT run evidence certifiers, do NOT write
   to evidence/, do NOT import anything that writes files.
6. Concurrency: another session may commit/change shared files while you work — only your own
   new file matters. Keep each operator self-contained (shared kernels private to your module).
