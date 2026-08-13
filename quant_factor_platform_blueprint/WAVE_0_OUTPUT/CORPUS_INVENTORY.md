# Corpus and Test-Pack Inventory

Audit date: 2026-08-13

Scope: read-only inspection of corpus/test-pack directories and external artifacts under `/home/shw/quant_projects`. Counts exclude `.git` and `__pycache__`. “Latest modification” is the latest retained source/artifact mtime after those exclusions. These assets are test/research corpora, not production runtime dependencies, per `AI_GUIDE/07_RESEARCH_CONTROL_AND_CORPORA.md` and `AI_GUIDE/11_PUBLIC_REPO_LESSONS_AND_LICENSE.md`.

## Per-corpus summary

| Corpus | Location | License | Version | File count | Formula count | Latest modification | Primary use |
|---|---|---|---:|---:|---:|---|---|
| GTJA-191 / GTJA185 | `/home/shw/quant_projects/gtja191` | No license file or declaration found; proprietary/unknown until provenance is recorded | FactorPack `1` | 603 | 191 DSL source formulas; 185 deliverable manifests | 2026-07-19 | FE DSL/AST regression, compat-surface parsing, materialization and batch benchmarks |
| Week2 PV factors | `/home/shw/quant_projects/week2_pv_factors` | No license file or declaration found; internal report-derived corpus | Not declared | 86 | 37 DSL formulas | 2026-07-18 | FE formula translation/regression and candidate delivery tests |
| A-share LQTP kit | `/home/shw/quant_projects/ashare_lqtp_kit` | **Unknown: no `LICENSE`, `COPYING`, or `NOTICE` in checkout or `ashare_lqtp_kit.tar.gz`** | `0.1.1` (README) | 227 | N/A | 2026-07-18 | gRPC interoperability/examples corpus; do not use as runtime dependency without license clearance |
| Raw data layer | `/home/shw/quant_projects/raw_data_layer` | No license file or declaration found | Config schema `1`; package version not declared | 71 | N/A | 2026-07-18 | Legacy data-source/config/PIT edge-case corpus; not a replacement for DataAccess |
| Factor cold start | `/home/shw/quant_projects/factor_cold_start` | No license file or declaration found; appears internal | FactorPack `1` | 24 | 4,620 catalog entries | 2026-08-04 | Deterministic cold-start, sampler, field-availability, market/surface isolation tests |
| LQTP Python gRPC examples | `/home/shw/quant_projects/lqtp-python-grpc-examples` | **Unknown: no license file, README declaration, SPDX/copyright/license headers** | Not declared; generated protobuf code says protobuf Python `6.31.1` | 26 | N/A | 2026-07-18 | Protocol/client compatibility corpus only |
| EvoAlpha mined factors | `/home/shw/quant_projects/evoalpha_mined_factors_dsl_rankic_gt0p02_pos_20260809.tar.gz` | No license artifact in archive | Snapshot dated 2026-08-09 | 3,175 archive files | 3,052 unique filtered rows; 3,006 manifests | 2026-08-09 | External mined-factor metadata/formula corpus for ingestion, dedupe and evaluation regression |
| CogAlpha report | `/home/shw/quant_projects/cogalpha_factor_backtest_report_20260710(1)(1).html` | No visible embedded license declaration found | Snapshot dated 2026-07-10 | 1 | Report-defined | 2026-07-18 | Backtest-report parser/metadata extraction corpus |
| External factor logs | `/home/shw/quant_projects/external_factors_*.log` | No license metadata established | 2026-08 snapshots | 6 | Log-defined | 2026-08-02 | Failure/replay and external materialization diagnostics corpus |

No audited directory contains `pyproject.toml` or `setup.py`. Top-level README files exist for all six directories. No audited directory contains a license file.

### Top-level inventory

- `gtja191/`: `README.md`, `autofactor/`, `candidate_pool/`, `docs/`, `dsl/`, `examples/`, `formulas/`, `lib/`, `scripts/`, `source/`, `tests/`, `.gitignore`.
- `week2_pv_factors/`: `README.md`, `candidate_pool/`, `formulas/`, `lib/`, `scripts/`, `source/`.
- `ashare_lqtp_kit/`: `README.md`, `requirements.txt`, `.env.example`, `.gitignore`, `ashare_lqtp/`, `examples/`, `protos/`, `reports/`, `tools/`, `.git/`.
- `raw_data_layer/`: `README.md`, `data_daily_update/`, `raw_data_cleaning/`, `raw_data_fetching/`.
- `factor_cold_start/`: `README.md`, `__init__.py`, `catalog.py`, `generator.py`, `model.py`, `sampler.py`, `autofactor/`, `catalogs/`, `reports/`, `scripts/`, `source/`, `tests/`.
- `lqtp-python-grpc-examples/`: `README.md`, `requirements.txt`, four result CSV files, `examples/`, `protos/`.

## GTJA191 deep dive

The directory contains exactly **191** formula files, all flat under `formulas/` and all using the `.dsl` extension:

```text
gtja191/formulas/
├── gtja191_alpha_001.dsl
├── gtja191_alpha_002.dsl
├── ...
└── gtja191_alpha_191.dsl
```

The README describes 191 source formulas translated to canonical FactorEngine DSL, with 185 A-share price/volume factors retained after excluding six index/stub formulas. The deliverable reflects that distinction: `candidate_pool/` has 185 `manifest.json` files plus one config. The DSL uses `surface=compat`, not the default daily surface.

Related organization:

- `source/gtja191_formulas.json`: 191 source formulas.
- `dsl/gtja191_dsl_catalog.json`: translated catalog.
- `dsl/manual_dsl_overrides.json`: manual translations/overrides.
- `dsl/fe_dsl_allowlist.json` and `dsl/dsl_allowlist.json`: DSL/operator snapshots.
- `dsl/audit_report.json`, `materialize_progress.json`, `materialize_report.json`: audit/materialization artifacts.
- `dsl/operator_mapping.md`: operator translation notes.
- `candidate_pool/`: disk.v1 delivery manifests.
- `scripts/`: conversion, allowlist export, audits, manifest validation, smoke/materialization and AutoFactor evaluation utilities.

The requested top-level DSL module names are: `audit_report.json`, `dsl_allowlist.json`, `fe_dsl_allowlist.json`, `gtja191_dsl_catalog.json`, `manual_dsl_overrides.json`, `materialize_progress.json`, `materialize_report.json`, and `operator_mapping.md`. There are no Python modules directly in `dsl/`.

## Week2 PV factors deep dive

The directory contains exactly **37** formula files, all flat under `formulas/` and all `.dsl`:

```text
week2_pv_factors/formulas/
├── a_001_intraday_vol_surge.dsl
├── ...                         # 9 A.pdf Elite formulas, a_001-a_009
├── w2_001_vol_quantile_stress.dsl
├── ...
└── w2_028_asymmetric_range_ema5.dsl
```

The 37 formulas comprise 28 qualified factors from the Week2 report and nine Elite factors from `A.pdf`. `candidate_pool/` contains 37 manifests plus one config. `source/week2_factors_catalog.json` is the source catalog and `source/audit_report.json` records translation audit results. The README explicitly identifies three approximate conditional-median translations (`w2_022`, `w2_027`, `w2_028`); adapters must retain these approximation notes rather than silently treating them as exact.

Scripts are `audit_formulas.py`, `build_catalog.py`, `build_delivery.py`, `validate_all.py`, and `validate_manifests.py`; helper modules are `lib/dsl_validate.py` and `lib/paths.py`.

## Factor cold-start modules

- `model.py`: catalog entry/data model contracts.
- `generator.py`: deterministic formula generation across market, surface, family, horizon and complexity dimensions.
- `catalog.py`: catalog loading and available-field filtering.
- `sampler.py`: reproducible stratified/diversity sampling with family caps.
- `catalogs/`: generated A-share/US daily and extended JSON catalogs.
- `autofactor/provider.py`: four AutoFactorEvaluation FactorPack providers.
- `scripts/build_catalog.py`: rebuild catalogs and coverage artifacts.
- `scripts/validate_catalog.py`: fail-closed full validation.
- `scripts/sample_batch.py`: command-line batch sampling.
- `scripts/report_coverage.py`: coverage report generation.
- `reports/`: machine- and human-readable coverage summaries.
- `source/existing_pack_formula_hashes.json`: structural hashes used to avoid duplicates with GTJA/Week2.
- `tests/`: catalog contract, FE parse, provider and sampler tests.

README-declared catalog sizes are A-share daily 819, US daily 1,007, A-share extended 1,359, and US extended 1,435, totaling **4,620**.

## Raw data layer

Subdirectories are:

```text
raw_data_layer/
├── data_daily_update/
│   ├── configs/
│   └── scripts/
├── raw_data_cleaning/
└── raw_data_fetching/
    ├── massive_parquet/
    ├── rest_api_doc/
    └── tests/
```

Documentation names Massive/Polygon-style APIs, REST APIs, S3/object storage, and local Massive parquet as inputs. The cleaning configuration documents 24 source datasets: aggregate daily summaries; dividends, IPOs and splits; filing risk categories/factors and SEC EDGAR index; balance sheet, cash flow, financial ratios, income statement, short interest/volume and stock floats; condition codes, exchanges and market holidays; news; ticker/reference data; and US SIP day aggregates, minute aggregates, quotes and trades. Quotes/trades are disabled by default because of size.

This tree contains legacy download, cleaning, scheduling, cursor and quality-control logic. Under the platform blueprint it should be retained only as a corpus for source-contract, schema, availability-time and failure fixtures; production access remains owned by DataAccess.

## LQTP Python gRPC examples license

The README describes runnable Python clients for authentication, factor execution/analysis, backtests and paper portfolios, plus checked-in generated protobuf modules. It makes no license statement.

Inspection of every `.py` file found no SPDX identifier, copyright statement or license header. Handwritten examples begin with usage docstrings; generated protobuf files contain only generator/version banners. There is no `LICENSE`, `COPYING`, or `NOTICE` file. Therefore the legal status is **unknown/unlicensed for reuse**, not permissive by implication. Use as a black-box interoperability/test corpus only until the upstream owner supplies explicit terms and provenance.

## A-share LQTP kit license

No `LICENSE`, `COPYING`, or `NOTICE` exists in `/home/shw/quant_projects/ashare_lqtp_kit`. The separately retained `/home/shw/quant_projects/ashare_lqtp_kit.tar.gz` also contains no license artifact. The README declares kit version `0.1.1` and subdirectories `ashare_lqtp/`, `examples/`, `protos/`, `reports/`, and `tools/`.

Conclusion: the kit license is **unknown**. Do not copy it into proprietary runtime or declare it a direct dependency until ownership, source repository/commit and license are documented. Protocol behavior can be tested through a narrow adapter using independently maintained/generated stubs after legal review.

## External tarball contents

### EvoAlpha mined factors

`file` identifies `evoalpha_mined_factors_dsl_rankic_gt0p02_pos_20260809.tar.gz` as valid gzip-compressed tar data. `tar -tzf` succeeds and reports **6,366 entries**, including **3,175 files**. Top-level content:

```text
./README.txt
./all_factors_dsl_rankic_gt0p02.csv
./all_factors_dsl_rankic_gt0p02.jsonl
./exports/
./candidate_pool/
./instances_candidate_pool/
```

The archive contains 3,171 JSON files, two CSV files, one JSONL file and one README. There are 3,006 `manifest.json` files, 164 `config.json` files, and two export metadata files under `exports/ashare_abs_rankic_ge_0p03_vwap_2024_2026/`. The README states 3,052 unique rows filtered to signed `rank_ic > 0.02`, a RankIC range of `0.020028..0.379770`, and no factor-value parquet. No license-named artifact is present.

### CogAlpha HTML

`file` identifies the artifact as a UTF-8 HTML document. Its title is **“CogAlpha 因子回测筛选汇报 · results/3 & results/4”**. No visible embedded license or copyright declaration was found after excluding scripts, styles, images and data payloads. Raw case-insensitive strings resembling GPL/AGPL occur inside encoded image/data payloads and are not license evidence.

### Other external artifacts

The scoped external logs are `external_factors_mat.log`, `external_factors_mat2.log`, `external_factors_mat3.log`, `external_factors_lqtp_mat.log`, `external_factors_lqtp_mat2.log`, and `external_factors_lqtp_mat3.log`. Treat them as diagnostic corpora with unknown provenance, not package/runtime inputs.

## GPL/AGPL findings

- **No GPL or AGPL license was identified in the audited local corpus directories, LQTP examples, A-share kit/archive, EvoAlpha archive, or CogAlpha report.**
- The CogAlpha HTML’s encoded media payload produces random case-insensitive `gpl`/`agpl` byte sequences; these are false positives and not readable legal declarations.
- The blueprint separately records that multiple public QuantSkills repositories are GPL-3.0/GPL-3.0-only. That is an architectural/legal warning about external repositories, not proof that these local corpora contain GPL code.
- Absence of GPL/AGPL does **not** mean permissive licensing. Most audited corpora have no explicit license and must remain `CORPUS`/`IDEA_ONLY`, never `DIRECT_DEPENDENCY` or `SOURCE_COPY`, until provenance and terms are recorded.

## Recommended corpus adapters

| Adapter | Input | Target/use | Required controls |
|---|---|---|---|
| `Gtja191CorpusAdapter` | `source/gtja191_formulas.json`, 191 `.dsl`, catalog and 185 manifests | FE compat parser/AST regression, 185-factor execution benchmark | Preserve source ID and 191-to-185 exclusion status; force `surface=compat`; record manual overrides and unsupported/index semantics; never auto-correct formulas |
| `Week2PvCorpusAdapter` | 37 `.dsl`, source catalog and audit report | FE parser/evaluator regression and translation fixtures | Preserve report provenance, 28+9 grouping, original formula and three explicit approximations; mark reconstructed Elite formulas |
| `ColdStartCatalogAdapter` | Four generated JSON catalogs | FO sampling/search benchmarks and FE legality tests | Enforce market/surface/field tiers, deterministic seed, structural-hash dedupe and causal constraints; catalog membership is not production admission |
| `EvoAlphaArchiveAdapter` | JSON manifests, configs, CSV/JSONL metadata | External mined-factor ingestion, identity/dedupe and QE batch tests | Stream archive without extraction into runtime tree; validate schema and paths; retain source snapshot/checksum, RankIC filter and unknown license; exclude metrics from admission truth |
| `LqtpGrpcFixtureAdapter` | `.proto`, client examples and A-share kit behavior | Protocol conformance and mock/black-box integration tests | Isolate as optional test fixture; regenerate stubs from cleared proto provenance where allowed; no runtime import from unlicensed trees; scrub endpoints/credentials |
| `RawDataContractFixtureAdapter` | YAML source metadata and documented schemas | DataAccess source-contract, PIT timestamp and DQ edge cases | Convert only declarative fixtures after provenance review; do not import downloader/S3/REST/scheduler code; map availability timestamps explicitly and flag proxies |
| `CogAlphaReportAdapter` | Standalone HTML report | Report parser and metadata/metric extraction regression | Parse inertly with scripts/network disabled; retain title/source hash and unknown license; treat report metrics as observations, not authoritative QE evidence |
| `ExternalLogAdapter` | Materialization logs | Failure taxonomy/replay fixtures | Redact credentials, endpoints, paths and factor IP; normalize nondeterministic timestamps; never execute embedded content |

Every adapter record should include source repository or artifact path, immutable digest/commit, source ID, license status, original formula/metadata, translation notes, unsupported semantics, parser surface, and corpus-only classification. Promotion from a corpus into production must pass independent FE semantics, DataAccess PIT, QE evidence and FactorAssets admission; corpus presence alone must confer no production status.
