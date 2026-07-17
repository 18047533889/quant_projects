# Source synchronization record

- Source repository: `18047533889/quant_projects`
- Source commit reviewed: `62fbef2af5e7f564addd8ac55836de5fc730b0af`
- Previous standalone baseline: `771b5a12cfa2fc96dba7ae7c0035b05e1130bc24`

## Absorbed from the source repository

- executable COS dataset contracts;
- fail-closed factor-panel loading;
- A-share return unit normalization;
- industry semantic filtering;
- event/PIT helper APIs;
- source-compatible `cos_factor_runtime` imports and regression concepts.

## Corrected instead of copied verbatim

- A-share dividend does not have a verified `PubDate` column;
- US dividends and split events do not have a reliable announcement timestamp;
- multi-instrument PIT selection no longer depends on fragile global `merge_asof` ordering;
- latest-period state cannot roll back when an old period is restated later;
- `columns=None` keeps the full event payload;
- industry source is explicit, not hardcoded to `sw_l1`;
- US filing `timeframe` is mandatory;
- physical registry axes/schemas and snapshot fingerprint are corrected;
- period/event/sparse COS files are not inferred from decision-date filenames;
- full recursive CLI/mirror sync requires explicit operator opt-in.

After this synchronization, active development should occur in `HKUST-QUANT-SOCIETY/data_access`. The source monorepo copy should consume a released/committed standalone version or be updated through an explicit synchronization PR, not edited concurrently.
