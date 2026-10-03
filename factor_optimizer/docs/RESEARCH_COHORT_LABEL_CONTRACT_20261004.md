# Research cohort price-label contract (2026-10-04)

This note describes the current `examples/cos_batch_audit.py` loader path and
its session-window, price-reader, label, and materialization helpers. It is a
contract summary, not a claim of full-market performance or a production
backtest result.

## Calendar and label alignment

`load_cos_sample` asks DataAccess for `get_market_calendar("ashare", store=...)`
and proceeds only when the returned calendar has data and
`source == "registry"`. The loader rejects fallback or otherwise non-registry
calendars; it does not substitute weekdays when the authoritative registered
`ashare_calendar` is unavailable.

The session-window selector takes the latest requested number of shared factor
panel dates that have two following sessions in that official calendar. For a
decision session `t`, the aligned label uses adjusted VWAP prices:

```text
r(t) = P(t+2) / P(t+1) - 1
```

Here `P(t+1)` is the label start/execution date and `P(t+2)` is the label end
date, each counted as a subsequent official trading session, not a calendar
day. Decisions without both endpoints are not selected. The loader passes the
official calendar from the first selected decision through the final `t+2`
label endpoint to the price reader, so the final endpoint is retained.

For each date/asset cell, the label is valid only if both endpoint prices are
finite and strictly positive and the computed return is finite. Otherwise the
return is `NaN` and the validity mask is false. Missing or invalid prices are
not filled with zero, forward-filled, or used to compress the session or asset
axes.

## Price reader and missing observations

`read_session_vwap` reads `AdjVwap` from `ashare_stock_daily_adj` on the exact
calendar and selected-asset axes supplied by the loader. It initializes the
whole result to `NaN`, reads in bounded calendar chunks, and reindexes each
response to the requested dates and symbols. Missing rows therefore remain
visible as `NaN`; duplicate daily asset observations and off-axis responses
are rejected rather than silently merged or discarded.

Each DataAccess query is capped at 32 MiB of result bytes and at most 64
calendar rows. The chunk bound is
`min(64, floor(32 MiB / (128 × asset_count)))`, using the reader's estimated
128 bytes per long row. This is a per-query result budget, not a cap on total
process RSS or a source-object download budget.

## Cohort and resource bounds

The loader accepts integer bounds of 1–1,260 decision sessions, 1–5,000
assets, and 1–16 factors. Its CLI defaults are 500 days, 256 assets, and 2
factors. Assets are selected from the common factor-panel symbol set using
TRAIN-only finite-value coverage; a candidate must reach 90% coverage for
every retained factor. Price-label validity does not remove or reorder the
selected axes; it is carried separately in the label mask.

There are three distinct budget layers:

| Budget | Current bounds/default | What it limits |
| --- | --- | --- |
| Source factor objects (`--max-factor-mib`, `--max-batch-mib`) | 1–128 MiB per factor (default 8); 1–2,048 MiB total (default 128) | Manifest-reported factor-object bytes checked before each factor-object read. These are separate from the price-query and decoded-memory budgets. |
| Price-reader query | 32 MiB result bytes per query; at most 64 calendar rows per chunk | One bounded VWAP read; not a total-RSS guarantee. |
| Downstream materialization (`--materialization-mib`) | 1–16,384 MiB (default 8,192) | The estimator's identified simultaneous buffers for already-read pandas/Arrow data, factor staging/freeze/masks, reindex/conversion, price panels, label arithmetic, and axis/index headroom. |

The materialization estimate is checked before the downstream price, label,
and factor-array assembly. Source Arrow decoding and pandas panel creation have
already occurred by then. The budget accounts for the named buffers; it is
not a bound on total RSS, Python/Arrow decoder peak memory, or arbitrary
third-party allocations, and it is not admission control for the earlier
source decode.

## Evidence and limits

The focused tests are mock/synthetic: the loader integration test uses a fake
COS factor response, fake price store, and constructed registry calendar; its
500- and 1,260-session cases preserve the terminal `t+2` endpoint. Reader,
label, session-window, and cohort-materialization tests likewise use fakes,
constructed calendars, in-memory arrays, or synthetic panels. The
`(1,260, 5,000, 16)` case validates the declared dimension bounds; it does not
materialize that full cohort.

Separate bounded live cohort reads were observed for one factor and 20 assets:

| Decision sessions | Source / span | Valid label cells | Estimated buffers | Notes |
| ---: | --- | ---: | ---: | --- |
| 500 | Registry calendar, 2024-08-06–2026-08-27 | 9,965 / 10,000 | 272,714,432 bytes (< 512 MiB) | No score computed; no RSS peak measurement. |
| 1,260 | `drawdown_vol_adj_stress`, 2021-06-21–2026-08-27 | 24,682 / 25,200 | 340,664,512 bytes (< 512 MiB) | Last decision 2026-08-27; label end 2026-08-31. No score computed; no RSS peak measurement. |

These observations demonstrate only those 20-asset, one-factor cases. They do
not establish a full-market 1,260-session run at the 5,000-asset / 16-factor
bounds, nor do they characterize peak RSS or scoring behavior. No “fastest” or
general no-defect claim follows from this contract or these focused checks.

## Final reader-contract recheck (2026-10-04)

The two estimates above are observations from the earlier loader source, not
estimates produced by the later query-workspace accounting. A subsequent live
read in terminal session 37833 completed with exit code 0: 1,260 decisions ×
20 assets × one factor, 24,682 valid labels, estimated buffers 341,012,672 bytes,
20 price chunks, and last label endpoint 2026-08-31. This was a loading smoke,
not an optimizer-scoring benchmark or process peak-RSS measurement.

Each price chunk records its dataset, endpoint sessions, row count, query byte
cap, DataAccess identity digest, source snapshot, and provenance status. The
live smoke had 20 identity digests and 20 snapshots; all provenance statuses
were `unknown`. Presence of a digest or snapshot does not upgrade that status
to verified source provenance. Missing identity is explicitly unavailable.
The reader closes each closable read handle even if conversion raises.

Materialization estimates additionally charge bounded price-query Arrow and
pandas workspaces, plus pivot buffers, using the same chunk dimensions as the
reader. These are declared estimates, not a total process-memory guarantee.

A cohort with no valid positive-price return labels now raises `ValueError`.
Individual missing/invalid prices remain NaN with a false validity mask; they
do not compress the session axis or move execution/label endpoints.

## Incremental source and TRAIN-workspace admission

The subsequent loader revision checks each decoded factor before accumulating
another pandas panel. If retained deep pandas bytes are R and current measured
Arrow buffers are A, the pre-conversion allowance is R + A + 2A. After
conversion, raw deep pandas bytes P are measured; index/filter/sort work is
admitted against R + A + 3P. The filtered panel is measured again before
retention. Source receipts preserve these measurements and allowances.
Unmeasurable Arrow buffers are rejected, not treated as zero. Source Arrow
aliases are released after each conversion; quarantined frame aliases are
released before final tensor materialization.

Before TRAIN coverage/asset selection, the declared numeric workspace estimate
is 17 × TRAIN_rows × sum(source_widths) + 16 × factor_count × max(source_widths)
bytes, added to the resident pandas panels. It charges reindex/selection numeric
copies, finite masks, and coverage-reduction arrays conservatively. The ranking
still uses minimum coverage, mean coverage, and asset identity as deterministic
tie-breakers; minimum/mean reductions are now computed once per asset grid.

These allowances are not pre-Arrow-decode admission, an arbitrary object/string
expansion bound, or a total-process RSS cap. Earlier live smoke estimates in this

## Later real multi-factor loading check

Session 77016, PID 640811, exited 0 on the incremental-admission source:
1,260 decisions × 256 assets × 8 factors; all eight retained, none quarantined.
There were 321,483 valid labels out of 322,560 cells and the final label endpoint
was 2026-08-31. Budget remained 4,294,967,296 bytes. Downstream identified
buffers were estimated at 1,092,753,024 bytes; TRAIN coverage workspace was
537,668,216 bytes. Eight source admissions and 20 price chunks were recorded.
All 20 price chunks had identity digests and snapshots, with status `unknown`.
Linux process peak RSS (`ru_maxrss × 1024`) was 2,691,899,392 bytes; it includes
imports, source reads and loading, and is not a bound guaranteed by admission.

This check did not run optimizer candidate scoring or measure speedup. It does
not establish 5,000-asset/16-factor feasibility. The following declared source
hashes were captured after the successful run, not a transitive-runtime closure:

| Source | SHA-256 |
| --- | --- |
| `examples/cos_batch_audit.py` | `f070c792a8c85a902b9e1576d1b4e72a531fcaa0e0edf7c39a382f8d23a75c7b` |
| `factor_optimizer/cohort_materialization.py` | `bf663447783af203e3ecedb1367f36e26b58c2a6d079c1006d3caddb8760a7cd` |
| `factor_optimizer/research_price_reader.py` | `c1bdde566da4157351a79075cfcec206239725c0c0e367ab746f2280e65d2ed1` |

Final regression session 40300 exited 0: 1,935 tests passed, 16 warnings,
174.91 seconds. Loader, cohort materialization, research batch and summary-cache
source hashes matched before/after. This run included the coverage-admission
test that rejects before TRAIN reindexing or price reads, plus decoded-source,
calendar/label and optimizer regression cases. It is regression evidence for
the tested source, not proof of universal correctness or fastest performance.
document belong to their stated source revision, not this later revision.
