# Leakage / Boundary Audit

Generated from a read-only scan of `/home/shw/quant_projects` on 2026-08-13. Production sources were not modified. The requested output report is the only written artifact.

## sys.path hacks

Raw scan: `rg -n "sys\.path" /home/shw/quant_projects/ -g '*.py'` returned **769 occurrences**. Rows are intentionally exhaustive; generated/build and reconstructed copies remain visible. Test/example bootstraps are accepted only under the requested exception.

| file | line | kind | verdict |
|---|---:|---|---|
| `/home/shw/quant_projects/factor_cold_start/scripts/_bootstrap.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_cold_start/scripts/_bootstrap.py` | 11 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py` | 8 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_pipeline.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_pipeline.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline_cli.py` | 73 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline_cli.py` | 74 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/lqtp-python-grpc-examples/examples/backtest_client.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/lqtp-python-grpc-examples/examples/portfolio_client.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/lqtp-python-grpc-examples/examples/auth_client.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/lqtp-python-grpc-examples/examples/factor_client.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/ashare_lqtp_kit/ashare_lqtp/client.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/ashare_lqtp/client.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run.py` | 7 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run.py` | 8 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/examples/backtest_client.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/ashare_lqtp_kit/examples/portfolio_client.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/ashare_lqtp_kit/examples/auth_client.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/ashare_lqtp_kit/examples/factor_client.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_agent/scripts/配置文件评价脚本.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_agent/scripts/配置文件评价脚本.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run_from_config.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run_from_config.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/data_daily_update/async_universal_fetcher.py` | 43 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/data_daily_update/production_update_runner.py` | 46 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/data_daily_update/production_update_runner.py` | 47 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/data_daily_update/async_incremental_update.py` | 38 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/platform_bootstrap.py` | 140 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/platform_bootstrap.py` | 141 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/platform_bootstrap.py` | 144 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/tools/demo_topk_backtest.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/tools/demo_run_factor.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/tools/probe.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/tools/probe.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/ashare_lqtp_kit/tools/probe.py` | 46 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_agent/main.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_agent/main.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/tests/test_factor_evaluation_runtime.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/tests/test_factor_evaluation_runtime.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/run_pipeline.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/run_pipeline.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 152 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 153 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 316 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 317 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 468 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 469 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 622 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 623 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/tests/test_pipeline.py` | 9 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/tests/test_pipeline.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/integrations/quant_platform.py` | 4 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/scripts/benchmark_workloads.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/scripts/compact_dataset.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_gateway_core.py` | 11 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_deduplicator.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_future_scanner.py` | 5 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/bench_pandas_vs_modin.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/bench_pandas_vs_modin.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_complexity.py` | 9 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_production_fastpath_matrix.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_production_fastpath_matrix.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_production_fastpath_matrix.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_production_fastpath_matrix.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_production_fastpath_matrix.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_validator.py` | 5 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_dsl_allowlist.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_dsl_allowlist.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/tests/test_kafka_producer.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_datasets_mining_alignment.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_datasets_mining_alignment.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_datasets_mining_alignment.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_datasets_mining_alignment.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/main.py` | 38 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_manifest.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_manifest.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_manifest.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_manifest.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_manifest.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_catalog.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_catalog.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_catalog.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_catalog.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/list_pandas_only_ops.py` | 37 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/list_pandas_only_ops.py` | 38 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/list_pandas_only_ops.py` | 40 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/list_pandas_only_ops.py` | 41 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_guide.py` | 48 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/generate_operators_guide.py` | 49 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/migrate_factor_lake_schema.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/migrate_factor_lake_schema.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/migrate_factor_lake_schema.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/migrate_factor_lake_schema.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/profile_pandas_backend.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/profile_pandas_backend.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/ops/refresh_dataset_stats.py` | 63 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/ops/refresh_dataset_stats.py` | 64 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_specs.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_specs.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_specs.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_coverage.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_coverage.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_coverage.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_coverage.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_coverage.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/check_operator_contracts.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/check_operator_contracts.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/check_operator_contracts.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_upgrade_matrix.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_operator_upgrade_matrix.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/sync_primitive_evidence.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/sync_primitive_evidence.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_fastpath_coverage.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_fastpath_coverage.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_fastpath_coverage.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_fastpath_coverage.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/report_backend_fastpath_coverage.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_public_factor_formulas.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_public_factor_formulas.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/certify_operator.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/certify_operator.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_mining_data_source_presets.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_mining_data_source_presets.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_mining_data_source_presets.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_mining_data_source_presets.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_delivery_formula.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/validate_delivery_formula.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_phase1_scope.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/export_phase1_scope.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/ops/refresh_dataset_stats.py` | 63 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/ops/refresh_dataset_stats.py` | 64 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/certify_primitive_evidence.py` | 108 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/scripts/certify_primitive_evidence.py` | 109 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/scripts/worker.py` | 147 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/scripts/worker.py` | 148 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/clickhouse_source.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/clickhouse_source.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/clickhouse_source.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/staging_loader.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/staging_loader.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/data_access_source.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/sources/data_access_source.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/trading_calendar.py` | 151 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/trading_calendar.py` | 152 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_final_closure_round9.py` | 98 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_final_closure_round9.py` | 414 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/clickhouse_materializer.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/clickhouse_materializer.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_final_closure_round10.py` | 111 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_final_closure_round10.py` | 162 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/generate_mock_data.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/generate_check_doc.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/lib/dsl_validate.py` | 56 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/lib/dsl_validate.py` | 57 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/run_three_stage_smoke.py` | 39 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/run_three_stage_smoke.py` | 40 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/run_three_stage_smoke.py` | 41 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/indicator/scripts/run_metrics.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/convert_and_build_delivery.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/convert_and_build_delivery.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/validate_manifests.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/validate_manifests.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_r32_identity_version_2026_08.py` | 94 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/gtja191/scripts/run_smoke.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_smoke.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_smoke.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_smoke.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_materialize.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_materialize.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_materialize.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_materialize.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/validate_factor_engine_coverage.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/validate_factor_engine_coverage.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/export_allowlist.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/export_allowlist.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/export_allowlist.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/export_allowlist.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/audit_semantic_consistency.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/audit_semantic_consistency.py` | 55 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/audit_gtja_dsl.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/audit_gtja_dsl.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/reconcile_data_snapshot.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/reconcile_data_snapshot.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/reconcile_data_snapshot.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/reconcile_data_snapshot.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_tests.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/run_tests.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/publish_factor.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/publish_factor.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/publish_factor.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/publish_factor.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/generate_materialize_configs.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/scripts/generate_materialize_configs.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/dataaccess/tests/unit/test_phase6_hardening.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/convert_to_lqtp.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/convert_to_lqtp.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_gtja191_factors.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_gtja191_factors.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_gtja191_factors.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_gtja191_factors.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/smoke_cos_compute_write.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/smoke_cos_compute_write.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/pipeline_worker.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/pipeline_worker.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/pipeline_worker.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/pipeline_worker.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/rollback_factor.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/rollback_factor.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/rollback_factor.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/rollback_factor.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/verify_gtja191_factor_lake.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/verify_gtja191_factor_lake.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/verify_gtja191_factor_lake.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/verify_gtja191_factor_lake.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/correlate_run_audit.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/correlate_run_audit.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/correlate_run_audit.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/correlate_run_audit.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_week2_factors.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/materialize_week2_factors.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_screening_reeval_catalog.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_screening_reeval_catalog.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_screening_reeval_catalog.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_screening_reeval_catalog.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_dsl_test.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_dsl_test.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_dsl_test.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_dsl_test.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/topup_weekly_ic02_more.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/tests/test_gtja191.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/gtja191/tests/test_gtja191.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/gtja191/tests/test_gtja191.py` | 273 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/gtja191/tests/test_gtja191.py` | 274 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/fix_platform_topk_one.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/fix_platform_topk_one.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/gtja191/tests/test_full_catalog_factor_engine.py` | 9 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/gtja191/tests/test_full_catalog_factor_engine.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_formulas_and_homepage.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_formulas_and_homepage.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_formulas_and_homepage.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_formulas_and_homepage.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/backtest_weekly_dug_panels.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/backtest_weekly_dug_panels.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/lqtp_client.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/lqtp_client.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/eval_extensions.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/eval_extensions.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_ast_translator.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_ast_translator.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_ast_translator.py` | 14 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_ast_translator.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/enrich_weekly_dug_metrics.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/enrich_weekly_dug_metrics.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/convert_to_lqtp.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/convert_to_lqtp.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/complete_remaining_factors.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/complete_remaining_factors.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/complete_remaining_factors.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/complete_remaining_factors.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/sql_pushdown/executor.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/sql_pushdown/executor.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/sql_pushdown/executor.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_report_charts.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_report_charts.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_preflight.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_preflight.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_preflight.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_preflight.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/lib/dsl_validate.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/lib/dsl_validate.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/materialize_halfyear_chunk.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/fill_one_weekly_chart.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_lqtp_platform_panel.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_lqtp_platform_panel.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_lqtp_platform_panel.py` | 82 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_lqtp_platform_panel.py` | 83 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/topup_weekly_ic02_x20.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_screening_reeval_batch.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_screening_reeval_batch.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_screening_reeval_batch.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_screening_reeval_batch.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/validate_manifests.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/validate_manifests.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/fill_weekly_dug_charts.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/add_2026_neutral_rankic_to_csv.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 913 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 914 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 915 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_production_batch.py` | 916 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/regenerate_reports_fast.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/regenerate_reports_fast.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/build_delivery.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/build_delivery.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/lqtp_connectivity_probe.py` | 51 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/factor_annotations.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/factor_annotations.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/validate_all.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/validate_all.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/eval_lake_fast.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/eval_lake_fast.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/wait_lqtp_and_resume_batch.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/wait_lqtp_and_resume_batch.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/build_catalog.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/build_catalog.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/audit_formulas.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/audit_formulas.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/audit_formulas.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/week2_pv_factors/scripts/audit_formulas.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_materialize.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_materialize.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_materialize.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_materialize.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/weekly_complex_topup_and_ui.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/regen_week3_factor_reports.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/regen_week3_factor_reports.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/factor_eval.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/add_yearly_double_neutral_to_csv.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_lqtp_platform_demo.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_lqtp_platform_demo.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_light_test.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_light_test.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 554 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/refresh_weekly_dug_ui_pack.py` | 555 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/reconcile_ic_signs.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/reconcile_ic_signs.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_eval_report.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_eval_report.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/bootstrap_lqtp_market_data.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/bootstrap_lqtp_market_data.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_eval_extensions.py` | 14 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_eval_extensions.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_to_dsl.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_to_dsl.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_to_dsl.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_to_dsl.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_lqtp_converter.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_lqtp_converter.py` | 14 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_lqtp_converter.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/test_lqtp_converter.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/patch_weekly_formula_explanations.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/data_access_panel.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/data_access_panel.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/data_access_panel.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/data_access_panel.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_factor_annotations.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/build_factor_annotations.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/materialize.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/materialize.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/materialize.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/materialize.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/crawl_candidate_pool_neutral_rankic.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/crawl_candidate_pool_neutral_rankic.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/render_rankic_screening_index.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/render_rankic_screening_index.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/run_pipeline.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/run_pipeline.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/backtest_missing_topk.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/backtest_missing_topk.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/run_pipeline.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/run_pipeline.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_batch.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_batch.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_batch.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/run_batch.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/purification/scripts/worker.py` | 183 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/purification/scripts/worker.py` | 184 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_quick.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_quick.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/validate_production_scenarios.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/tests/conftest.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/conftest.py` | 18 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/security/access.py` | 76 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/tests/unit/test_resource_governance.py` | 14 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/unit/test_run_many_streaming.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_dmd_hankel.py` | 26 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_path_signature.py` | 31 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_path_signature.py` | 32 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_e_model_lanes.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_e_model_lanes.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_kalman_stateful.py` | 44 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_kalman_stateful.py` | 45 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_b_model_oracle.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_b_model_oracle.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_a_model_truth_gates.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_a_model_truth_gates.py` | 26 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_state_families.py` | 35 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_state_families.py` | 36 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_regression_governance.py` | 39 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_ontology.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_ontology.py` | 22 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_regression.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_regression.py` | 20 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_ij_execution_traits.py` | 15 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_ij_execution_traits.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_f_numba_kernels.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_f_numba_kernels.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_kalman.py` | 40 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_kalman.py` | 41 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_knn.py` | 22 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_knn.py` | 23 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_d_model_parameter_domain.py` | 23 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_d_model_parameter_domain.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_data_factor_smoke.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_data_factor_smoke.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_sequence_anomaly.py` | 28 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_sequence_anomaly.py` | 29 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_pca_pcr.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_pca_pcr.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_usage_scenarios.py` | 28 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_usage_scenarios.py` | 29 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_param_specs.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_param_specs.py` | 26 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_unit_and_changepoint.py` | 29 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_unit_and_changepoint.py` | 30 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_c_model_causality.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_c_model_causality.py` | 20 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_gh_fast_linear_family.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_phase_gh_fast_linear_family.py` | 18 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/performance/bench_run_many_smoke.py` | 30 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/performance/bench_run_many_smoke.py` | 31 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/performance/bench_run_many_smoke.py` | 33 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/performance/bench_run_many_smoke.py` | 34 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r35/test_model_audit_volatility.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/modeling/test_model_semantic_registry.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/cache/run_simple_tests.py` | 1 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/cache/run_simple_tests.py` | 9 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/cache/run_simple_tests.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/cache/run_simple_tests.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/cache/run_tests.py` | 9 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/test_memory_leak_detection.py` | 24 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_usage_scenarios_simple.py` | 22 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/integration/test_real_usage_scenarios_simple.py` | 23 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_materializer.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_materializer.py` | 18 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_materializer.py` | 20 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/r40/test_model_audit_operator_batch2.py` | 31 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/runtime/test_phase24_platform.py` | 73 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/runtime/test_phase24_platform.py` | 74 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/storage/test_clickhouse_scan_polars_long.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/storage/test_clickhouse_scan_polars_long.py` | 18 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_executor.py` | 22 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_executor.py` | 23 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_sql_pushdown.py` | 25 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_sql_pushdown.py` | 26 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_integration.py` | 17 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_integration.py` | 18 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_integration.py` | 38 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/backend/test_clickhouse_integration.py` | 39 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/service/release_blockers.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/service/release_blockers.py` | 62 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/service/release_blockers.py` | 63 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/service/release_blockers.py` | 73 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r33_artifacts.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r33_artifacts.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/bench_pandas_vs_modin.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/bench_pandas_vs_modin.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_operator_math_contract.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/storage_tuning_benchmark.py` | 41 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_production_fastpath_matrix.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_production_fastpath_matrix.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_production_fastpath_matrix.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_production_fastpath_matrix.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_production_fastpath_matrix.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_hard_gates.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_hard_gates.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_comprehensive.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_comprehensive.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r39_benchmark_suite.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r39_benchmark_suite.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r47_evidence.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r47_evidence.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase3c_detail.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_factor_operator_evidence.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_factor_operator_evidence.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_resource_admission.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_resource_admission.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_dsl_allowlist.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_dsl_allowlist.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_market_field_manifest.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_market_field_manifest.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/backend/sql_pushdown/executor.py` | 51 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_datasets_mining_alignment.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_datasets_mining_alignment.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_datasets_mining_alignment.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_datasets_mining_alignment.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/backend/numba_kernels/kalman.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/backend/numba_kernels/kalman.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_acceptance.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_acceptance.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/backend/numba_kernels/ar_stateful.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/backend/numba_kernels/ar_stateful.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/polars_performance_examples.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_aggressive.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_aggressive.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase1_inventory.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/tests/operators/test_factor_templates.py` | 12 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/tests/operators/test_factor_templates.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_semantic_hash_completeness.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_semantic_hash_completeness.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r37_ledger.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r37_ledger.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/duckdb_parallel_tuning_benchmark.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/duckdb_parallel_tuning_benchmark.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/duckdb_parallel_tuning_benchmark.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_external_pressure.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_manifest.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_manifest.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_manifest.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_manifest.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_manifest.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_catalog.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_catalog.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_catalog.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_catalog.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r30_availability.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_demo.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_demo.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_demo.py` | 22 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r15_master.py` | 42 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_dag_parallelism.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/list_pandas_only_ops.py` | 37 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/list_pandas_only_ops.py` | 38 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/list_pandas_only_ops.py` | 40 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/list_pandas_only_ops.py` | 41 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r28_model_causality.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r28_model_causality.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/adaptive_config_demo.py` | 8 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/build_operator_market_capabilities.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_operator_market_capabilities.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_config_directory.py` | 20 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_config_directory.py` | 21 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/run_r28_execute_batches.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/batch_materialize.py` | 30 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/batch_materialize.py` | 32 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/batch_materialize.py` | 33 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/auto_generate_polars_bridges.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_guide.py` | 48 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operators_guide.py` | 49 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/migrate_factor_lake_schema.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/migrate_factor_lake_schema.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/migrate_factor_lake_schema.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/migrate_factor_lake_schema.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r31_acceptance.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r31_acceptance.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/parameter_sensitivity_audit.py` | 49 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/parameter_sensitivity_audit.py` | 50 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/performance_benchmark_calibration.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/performance_benchmark_calibration.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/sql_certification_factory.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r38_hard_gates.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r38_hard_gates.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r38_hard_gates.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_fundamental_1288.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_fundamental_1288.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/profile_pandas_backend.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/profile_pandas_backend.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r30_source_universe_pit.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_specs.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_specs.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_specs.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_coverage.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_coverage.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_coverage.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_coverage.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_coverage.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_evidence_truth.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_evidence_truth.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase1b_scan.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_stub_surface.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r30_artifacts.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r25_genuine_usability.py` | 37 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r25_genuine_usability.py` | 38 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_behavioral_certification_ledger.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_behavioral_certification_ledger.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/run_factors_joblib.py` | 13 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/run_factors_joblib.py` | 14 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_evidence.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_evidence.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_from_config.py` | 19 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/examples/materialize_from_config.py` | 20 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/classify_master_operators.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_fe_da_traces.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_fe_da_traces.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r32_artifacts.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r32_artifacts.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r21_artifacts.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r21_artifacts.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r21_artifacts.py` | 131 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r21_artifacts.py` | 132 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/sync_backend_docs.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/sync_backend_docs.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/check_operator_contracts.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/check_operator_contracts.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/check_operator_contracts.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_upgrade_matrix.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_operator_upgrade_matrix.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/benchmark_r27_batch_throughput.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_unit_normalization_once.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_unit_normalization_once.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_all_operators_direct_use.py` | 39 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/minimal_benchmark_calibration.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/sync_primitive_evidence.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/sync_primitive_evidence.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operator_gap_preflight.py` | 225 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operator_gap_preflight.py` | 226 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_fastpath_coverage.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_fastpath_coverage.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_fastpath_coverage.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_fastpath_coverage.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/report_backend_fastpath_coverage.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r38_artifacts.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r38_artifacts.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r38_artifacts.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r30_per_canonical_review.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_advanced_sql.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_fe_dataaccess_contract_drift.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_fe_dataaccess_contract_drift.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r24_semantic_continuity.py` | 202 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r28_artifact_coherence.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r30_hard_gates.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_public_factor_formulas.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_public_factor_formulas.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_recipe_evidence.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_canonical_ledger.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_canonical_ledger.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_operator_differential_execution.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_hard_gates.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_hard_gates.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_parameter_domains.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r34_parameter_domains.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r33_hard_gates.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r33_hard_gates.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase3b_surface_scan.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_lqtp_formula_pack.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_lqtp_formula_pack.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_backend_evidence_manifest.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/extract_operator_data.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_layer_manifests.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_layer_manifests.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/check_backend_coverage.py` | 4 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_operator.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_operator.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/test_stress_fixes.py` | 10 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r28_evidence.py` | 137 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r28_evidence.py` | 138 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/tests/operators/test_r19_audit_machinery.py` | 34 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_cross_market_field_resolution.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_cross_market_field_resolution.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_shard_equivalence.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operator_usage_report.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_operator_usage_report.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/sources/clickhouse_source.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_research_operator_promotion_matrix.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_research_operator_promotion_matrix.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_intraday_parity.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_intraday_parity.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase6_param_scan.py` | 9 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/sources/staging_loader.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_mining_data_source_presets.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_mining_data_source_presets.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_mining_data_source_presets.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_mining_data_source_presets.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r27_native_fusion.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_delivery_formula.py` | 12 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/validate_delivery_formula.py` | 13 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_scm_manifest.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_operator_cross_market_matrix.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_operator_cross_market_matrix.py` | 30 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/benchmarks/backend_operator_bench.py` | 39 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/benchmarks/backend_operator_bench.py` | 40 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/benchmarks/backend_operator_bench.py` | 42 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/benchmarks/backend_operator_bench.py` | 43 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_final_runtime_state.py` | 42 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_final_runtime_state.py` | 45 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_final_runtime_state.py` | 46 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r34_ledger.py` | 22 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r34_ledger.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/sources/data_access_source.py` | 527 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/sources/data_access_source.py` | 528 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/wheel_clean_install_smoke.py` | 5 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/wheel_clean_install_smoke.py` | 64 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/wheel_clean_install_smoke.py` | 66 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/wheel_clean_install_smoke.py` | 67 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r30_phase3_module_scan.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_fixes.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r36_hard_gates.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r36_hard_gates.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_provider_execution_graph.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_provider_execution_graph.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r32_hard_gates.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r32_hard_gates.py` | 15 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/trading_calendar.py` | 320 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/trading_calendar.py` | 321 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_cross_market_semantics.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_cross_market_semantics.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_parameter_domains.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r37_parameter_domains.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r37_baseline.py` | 20 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r37_baseline.py` | 21 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/data_access_loader.py` | 4 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/data_access_loader.py` | 6 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/data_access_loader.py` | 49 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/auto_fix_operators.py` | 37 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/calibrate_backend_costs.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/calibrate_backend_costs.py` | 34 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r26_default_feasibility.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/auto_fix_operators_enhanced.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_all_factor_production.py` | 23 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_all_factor_production.py` | 24 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_pit_and_period_policy.py` | 25 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_pit_and_period_policy.py` | 26 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r36_artifacts.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r36_artifacts.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r36_artifacts.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_lqtp_compatibility_manifest.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_lqtp_compatibility_manifest.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/batch_operator_operations.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/storage/materialize/clickhouse_materializer.py` | 14 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_hard_gates.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_hard_gates.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/verify_robust_stats.py` | 10 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/verify_robust_stats.py` | 11 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/verify_robust_stats.py` | 64 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r40_hard_gates.py` | 27 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r40_hard_gates.py` | 31 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r40_hard_gates.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r47_backend_matrix.py` | 111 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r47_backend_matrix.py` | 112 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r37_issue_closure.py` | 16 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/build_r37_issue_closure.py` | 17 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r28_inventory.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_r28_inventory.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_phase1_scope.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_phase1_scope.py` | 19 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_execution_semantic_preservation.py` | 495 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_execution_semantic_preservation.py` | 496 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_execution_semantic_preservation.py` | 497 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r35_evidence.py` | 28 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_r35_evidence.py` | 29 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_primitive_evidence.py` | 75 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/certify_primitive_evidence.py` | 76 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/audit_model_additional_remediation.py` | 18 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r23_audit_generate.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/r23_audit_generate.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_layer_redesign_evidence.py` | 32 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/generate_model_layer_redesign_evidence.py` | 33 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_factor_engine.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/stress_test_factor_engine.py` | 36 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/scripts/export_direct_mining_catalog.py` | 35 | production/operational code path mutation | **UNACCEPTABLE** |
| `/home/shw/quant_projects/factor_engine/tests/operators/test_r11_shared_auditors.py` | 16 | test/example project-root or generated-proto bootstrap | **ACCEPTABLE (test/example bootstrap)** |

## legacy imports

Raw scan returned **210 occurrences**. The regex also catches class names, comments, and compatibility strings; those are explicitly distinguished from imports. No legacy import/reference was found in `factor_engine/scripts/`, `factor_engine/tools/`, `factor_engine/api/`, or `dataaccess/`; `dataaccess/write/upsert.py:17` only contains a provenance comment naming old code and does not match the import regex. Public API files do not re-export `factor_layer`, AutoFactorEvaluation, `FactorAnalyzer`, or `Exposures`.

| file | line | import/reference | verdict |
|---|---:|---|---|
| `/home/shw/quant_projects/data/external_factor_packs/extracted/extra20_factors_pack/extra20_factors_pack/factors/extra_18_cogalpha_20260702043307_1c1f15bf/code.py` | 4 | `overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/data/external_factor_packs/extracted/extra20_factors_pack/extra20_factors_pack/factors/extra_12_cogalpha_20260702070916_4a2b9199/code.py` | 4 | `overnight, intraday = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/factor_layer/factor_admission/config_runner.py` | 5 | `from factor_layer.factor_admission.admission import admit_evaluation_run` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/config_runner.py` | 6 | `from factor_layer.factor_admission.config import FactorAdmissionConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/admission.py` | 7 | `from factor_layer.factor_admission.catalog import AdmissionCatalog` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/admission.py` | 8 | `from factor_layer.factor_admission.config import FactorAdmissionConfig, ThresholdConfig` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py` | 11 | `from factor_layer.factor_admission.pipeline import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/pipeline.py` | 14 | `from factor_layer.factor_admission.admission import admit_evaluation_run` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/pipeline.py` | 15 | `from factor_layer.factor_admission.config import FactorAdmissionConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/__init__.py` | 1 | `from factor_layer.factor_admission.admission import admit_evaluation_run` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/__init__.py` | 2 | `from factor_layer.factor_admission.catalog import AdmissionCatalog` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/__init__.py` | 3 | `from factor_layer.factor_admission.config import FactorAdmissionConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/__init__.py` | 4 | `from factor_layer.factor_admission.pipeline import run, run_config_directory, run_from_config, run_pipeline` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/run_pipeline.py` | 12 | `from factor_layer.factor_admission.pipeline import run_config_directory, run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline.py` | 18 | `from factor_layer.factor_admission.pipeline import run_config_directory, run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline.py` | 19 | `from factor_layer.factor_evaluation.config_runner import run_from_config as run_evaluation_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline_cli.py` | 76 | `from factor_layer.factor_evaluation.config_runner import run_from_config as run_evaluation_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py` | 18 | `from factor_layer.factor_admission.catalog import AdmissionCatalog` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py` | 19 | `from factor_layer.factor_admission.config_runner import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py` | 20 | `from factor_layer.factor_evaluation.config_runner import run_from_config as run_evaluation_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 8 | `from .FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 9 | `from .history_module.Exposures_legacy import PortfolioExposures as _PortfolioExposuresBase` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 10 | `from .history_module.Exposures_legacy import PureExposures as _PureExposuresBase` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 13 | `class PortfolioExposures(_PortfolioExposuresBase):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 175 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 217 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 276 | `FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 346 | `title=dict(text=f"Portfolio Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 358 | `class PureExposures(_PureExposuresBase):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 487 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 529 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 589 | `FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 659 | `title=dict(text=f"Pure Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 699 | `FactorAnalyzer.add_subtitle(fig, "Factor Correlations Over Time", 1)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 735 | `FactorAnalyzer.add_subtitle(fig, "Mean Correlations", 2)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 761 | `FactorAnalyzer.add_subtitle(fig, "Factor Correlations Matrix", 3)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 809 | `FactorAnalyzer.add_subtitle(fig, f"Correlations Distribution of {factor}", row)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 5 | `#   1. PortfolioExposures  → quantile portfolio attribution` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 6 | `#   2. PureExposures       → factor-weighted pure exposure analysis` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 46 | `# pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 77 | `# pure = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 101 | `# PortfolioExposures:` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 104 | `# PureExposures:` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 113 | `from alphapurify.Exposures import PureExposures, PortfolioExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 166 | `pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 182 | `pure = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 2 | `#  Class FactorAnalyzer Usage Example` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 4 | `# This section demonstrates how to use FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 33 | `# 2. Initialize FactorAnalyzer: These are all the parameters!` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 35 | `# from alphapurify import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 36 | `# from alphapurify.FactorAnalyzer import AnalysisConfig, ResearchConfig` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 38 | `# FactorAnalyzer = FactorAnalyzer(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 148 | `# FactorAnalyzer.run()` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 149 | `# FactorAnalyzer.create_long_return_sheet(staticPlot:bool=False, return_fig:bool=False)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 150 | `# FactorAnalyzer.create_long_short_return_sheet(staticPlot:bool=False, return_fig:bool=False)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 151 | `# FactorAnalyzer.create_short_return_sheet(staticPlot:bool=False, return_fig:bool=False)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 152 | `# FactorAnalyzer.create_single_fac_ic_sheet(staticPlot:bool=False, return_fig:bool=False)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 220 | `from alphapurify import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 260 | `FA = FactorAnalyzer(df,'datetime','symbol','close','factor')` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 7 | `from .FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 9 | `class PortfolioExposures():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 11 | `Portfolio_Exposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 131 | `>>> pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 423 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 471 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 541 | `FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 611 | `text=f"Portfolio Returns & Exposures ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 624 | `class PureExposures():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 626 | `Pure_Exposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 758 | `>>> pe = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1012 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1060 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1130 | `FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1200 | `text=f"Pure Returns & Exposures ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1245 | `FactorAnalyzer.add_subtitle(fig,'Factor Correlations Over Time',1)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1280 | `FactorAnalyzer.add_subtitle(fig,'Mean Correlations',2)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1315 | `FactorAnalyzer.add_subtitle(fig,'Factor Correlations Matrix',3)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 1361 | `FactorAnalyzer.add_subtitle(fig,f'Correlations Distribution of {factor}', row)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/__init__.py` | 16 | `- FactorAnalyzer: a fully vectorized, multiprocessing-powered factor research engine` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/__init__.py` | 25 | `- Exposures: a factor exposure and return attribution engine for long–short,` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/__init__.py` | 49 | `from .Exposures import PortfolioExposures, PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/__init__.py` | 50 | `from .FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Database.py` | 234 | `pipelines such as FactorAnalyzer.` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 36 | `class FactorAnalyzer():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 38 | `FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 127 | `>>> analyzer = FactorAnalyzer.simple(df, factor_name="alpha")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 192 | `self.freq = FactorAnalyzer.map_freq(self.td)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 252 | `def add_subtitle(fig, text, row, y=1.15, Exposures=False):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 254 | `if Exposures == True:` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/tests/test_FctorAnalyzer.py` | 6 | `from alphapurify import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/tests/test_FctorAnalyzer.py` | 47 | `FA = FactorAnalyzer(df,'datetime','symbol','close','factor')` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/tests/test_Exposures.py` | 3 | `from alphapurify.Exposures import PureExposures, PortfolioExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/tests/test_Exposures.py` | 60 | `pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/tests/test_Exposures.py` | 76 | `pure = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 6 | `from factor_layer.alphapurify import pipeline as _base_pipeline` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 7 | `from factor_layer.alphapurify.Exposures_new import PortfolioExposures, PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 8 | `from factor_layer.alphapurify.config import AlphaPurifyAdapterConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 16 | `original_portfolio = _base_pipeline.PortfolioExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 17 | `original_pure = _base_pipeline.PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 19 | `_base_pipeline.PortfolioExposures = PortfolioExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 20 | `_base_pipeline.PureExposures = PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 23 | `_base_pipeline.PortfolioExposures = original_portfolio` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/pipeline_0505_v_next.py` | 24 | `_base_pipeline.PureExposures = original_pure` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run_from_config.py` | 13 | `from factor_layer.factor_evaluation_alphapurify.pipeline import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run_from_config.py` | 18 | `description="Run alphapurify Database -> Exposures -> FactorAnalyzer pipeline from YAML config."` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/pipeline.py` | 425 | `factor_result["errors"].append(f"PortfolioExposures 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/pipeline.py` | 478 | `factor_result["errors"].append(f"PureExposures 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/pipeline.py` | 548 | `factor_result["errors"].append(f"FactorAnalyzer 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 9 | `from .Exposures import PortfolioExposures as OfficialPortfolioExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 10 | `from .Exposures import PureExposures as OfficialPureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 11 | `from .FactorAnalyzer import FactorAnalyzer as OfficialFactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 40 | `return OfficialPortfolioExposures, OfficialPureExposures, "official"` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 42 | `return getattr(module, "PortfolioExposures"), getattr(module, "PureExposures"), module_name` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 47 | `return OfficialFactorAnalyzer, importlib.import_module(OfficialFactorAnalyzer.__module__), "official"` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/module_selector.py` | 49 | `return getattr(module, "FactorAnalyzer"), module, module_name` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/__init__.py` | 16 | `- FactorAnalyzer: a fully vectorized, multiprocessing-powered factor research engine` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/__init__.py` | 25 | `- Exposures: a factor exposure and return attribution engine for long–short,` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/__init__.py` | 49 | `from .Exposures import PortfolioExposures, PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/__init__.py` | 50 | `from .FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/run.py` | 10 | `from factor_layer.factor_evaluation_alphapurify.run_from_config import main` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Database.py` | 333 | `pipelines such as FactorAnalyzer.` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 8 | `from ..FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 9 | `from .Exposures_legacy import PortfolioExposures as _PortfolioExposuresBase` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 10 | `from .Exposures_legacy import PureExposures as _PureExposuresBase` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 13 | `class PortfolioExposures(_PortfolioExposuresBase):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 175 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 217 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 276 | `FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 346 | `title=dict(text=f"Portfolio Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 358 | `class PureExposures(_PureExposuresBase):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 487 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 529 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 589 | `FactorAnalyzer.add_subtitle(fig, f"{factor}", row, Exposures=True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 659 | `title=dict(text=f"Pure Returns & Exposures ({self.factor_name})", font=dict(size=24, family="Arial", color="black")),` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 699 | `FactorAnalyzer.add_subtitle(fig, "Factor Correlations Over Time", 1)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 735 | `FactorAnalyzer.add_subtitle(fig, "Mean Correlations", 2)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 761 | `FactorAnalyzer.add_subtitle(fig, "Factor Correlations Matrix", 3)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py` | 809 | `FactorAnalyzer.add_subtitle(fig, f"Correlations Distribution of {factor}", row)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Database_original.py` | 234 | `pipelines such as FactorAnalyzer.` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 17 | `from factor_layer.factor_evaluation_alphapurify.Database import DataBase` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 18 | `from factor_layer.factor_evaluation_alphapurify.Exposures import PortfolioExposures, PureExposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 19 | `from factor_layer.factor_evaluation_alphapurify.FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 20 | `from factor_layer.factor_evaluation_alphapurify.config import AlphaPurifyAdapterConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 23 | `FA_MODULE = importlib.import_module("factor_layer.factor_evaluation_alphapurify.FactorAnalyzer")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 387 | `pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 430 | `factor_result["errors"].append(f"PortfolioExposures 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 433 | `pure = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 483 | `factor_result["errors"].append(f"PureExposures 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 487 | `fa = FactorAnalyzer.simple(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/pipeline_legacy.py` | 553 | `factor_result["errors"].append(f"FactorAnalyzer 失败: {exc}")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 7 | `from ..FactorAnalyzer import FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 9 | `class PortfolioExposures():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 11 | `Portfolio_Exposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 131 | `>>> pe = PortfolioExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 423 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 471 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 541 | `FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 611 | `text=f"Portfolio Returns & Exposures ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 624 | `class PureExposures():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 626 | `Pure_Exposures` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 758 | `>>> pe = PureExposures(` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1012 | `text=f"Exposures Over Time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1060 | `text=f"Cummulative Returns of Exposures over time ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1130 | `FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1200 | `text=f"Pure Returns & Exposures ({self.factor_name})",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1245 | `FactorAnalyzer.add_subtitle(fig,'Factor Correlations Over Time',1)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1280 | `FactorAnalyzer.add_subtitle(fig,'Mean Correlations',2)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1315 | `FactorAnalyzer.add_subtitle(fig,'Factor Correlations Matrix',3)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py` | 1361 | `FactorAnalyzer.add_subtitle(fig,f'Correlations Distribution of {factor}', row)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 36 | `class FactorAnalyzer():` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 38 | `FactorAnalyzer` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 127 | `>>> analyzer = FactorAnalyzer.simple(df, factor_name="alpha")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 192 | `self.freq = FactorAnalyzer.map_freq(self.td)` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 252 | `def add_subtitle(fig, text, row, y=1.15, Exposures=False):` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 254 | `if Exposures == True:` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/io.py` | 9 | `from factor_layer.factor_evaluation.config import SourceConfig` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/config_runner.py` | 5 | `from factor_layer.factor_evaluation.config import FactorEvaluationConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/config_runner.py` | 6 | `from factor_layer.factor_evaluation.pipeline import evaluate_factor, save_results` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/run_from_config.py` | 6 | `from factor_layer.factor_evaluation.config_runner import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/pipeline.py` | 14 | `from factor_layer.factor_evaluation.config import FactorEvaluationConfig` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/pipeline.py` | 15 | `from factor_layer.factor_evaluation.io import load_factor_data, load_market_data, load_universe_data` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/__init__.py` | 1 | `from factor_layer.factor_evaluation.config import FactorEvaluationConfig, load_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/__init__.py` | 2 | `from factor_layer.factor_evaluation.config_runner import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/__init__.py` | 3 | `from factor_layer.factor_evaluation.pipeline import evaluate_factor, save_results` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/tests/test_factor_evaluation_runtime.py` | 15 | `from factor_layer.factor_evaluation.config_runner import run_from_config` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/ast_translator.py` | 40 | `r"(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=\s*alpha_tools\.classify_volume_regime\(\s*([^,]+),\s*window\s*=\s*(\d+)[^)]*\)",` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/ast_translator.py` | 48 | `r"(\w+)\s*=\s*alpha_tools\.classify_volume_regime\(\s*([^,]+),\s*window\s*=\s*(\d+)[^)]*\)",` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/ast_translator.py` | 52 | `r"(\w+)\s*,\s*(\w+)\s*=\s*alpha_tools\.decompose_overnight_intraday\(\s*([^,]+),\s*([^)]+)\)",` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/repair_formulas_and_homepage.py` | 129 | `x in py for x in ("groupby", "alpha_tools", "style_gate", "classify_volume_regime")` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 52 | `from toolkit.alpha_tools.library import classify_volume_regime as _cvr` | UNACCEPTABLE legacy import outside quarantine |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 64 | `from toolkit.alpha_tools.library import decompose_overnight_intraday` | UNACCEPTABLE legacy import outside quarantine |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 112 | `alpha_tools = _AlphaToolsFacade()` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_runtime.py` | 117 | `"alpha_tools": alpha_tools,` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/scripts/cogalpha_lqtp/python_to_dsl.py` | 212 | `if "alpha_tools.classify_volume_regime" in code and (` | Legacy symbol/reference, not an import; review migration tooling only |
| `/home/shw/quant_projects/toolkit/cross_sectional.py` | 9 | `from toolkit.registry import is_allowed_transform` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 13 | `from toolkit.alpha_tools import library` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 69 | `raise AttributeError(f"alpha_tools has no active tool {name!r}.")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 75 | `raise AttributeError("alpha_tools is read-only.")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 78 | `def build_alpha_tools_facade() -> AlphaToolsFacade:` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 119 | `module = importlib.import_module("toolkit.alpha_tools.generated_library")` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 151 | `return f"Unknown parameter {parameter_name!r} for alpha_tools.{tool_name}."` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/generated_library.py` | 3 | `This file is maintained by scripts/update_alpha_tools.py. Manual edits may be` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/__init__.py` | 3 | `from toolkit.alpha_tools.registry import (` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/__init__.py` | 4 | `build_alpha_tools_facade,` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/alpha_tools/__init__.py` | 12 | `"build_alpha_tools_facade",` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/__init__.py` | 3 | `from toolkit.cross_sectional import (` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/__init__.py` | 16 | `from toolkit.registry import (` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/__init__.py` | 23 | `from toolkit.alpha_tools import (` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/__init__.py` | 24 | `build_alpha_tools_facade,` | LEGACY-INTERNAL; quarantine, do not import from new code |
| `/home/shw/quant_projects/toolkit/__init__.py` | 46 | `"build_alpha_tools_facade",` | LEGACY-INTERNAL; quarantine, do not import from new code |

## second/PIT implementation

| file | why-not | exists-in-DA/FE |
|---|---|---|
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py:72` | Independent `asof_join`; bypasses governed temporal/PIT semantics | `dataaccess/read/temporal_join.py`, `dataaccess/store.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/trading_calendar.py:16` | Second trading calendar implementation in legacy copy | `dataaccess/read/session_calendar.py`; `factor_engine/market/exchange_session_calendar.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/runtime/session_calendar.py:17` | Second session-bar calendar | `factor_engine/runtime/session_calendar.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/universe_spec.py:13` | Second universe contract | `dataaccess/r30/universe_snapshot.py`; current FE backend universe contract |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/read/read_contract.py:58` | Second DataSnapshot contract | `dataaccess/snapshot/source_snapshot.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/read/stats.py:52` | Second dataset snapshot model | `dataaccess/snapshot/resolver.py`, `dataaccess/snapshot/verifier.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/cos/remote.py:458` | Second COS remote materializer/wrapper | `dataaccess/cos/remote.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/cos/s3_duckdb.py` | Second S3/DuckDB storage wrapper | `dataaccess/cos/s3_duckdb.py`; `dataaccess/core/engine.py` |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/parquet_source.py` | Second parquet source | `factor_engine/storage/sources/parquet_source.py`; DA governed readers |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/kline_parquet_source.py` | Second kline parquet source | `factor_engine/storage/sources/kline_parquet_source.py`; DA governed readers |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/validate_parquet.py` | Standalone parquet validation path outside DA governance | `dataaccess/read/*`, `dataaccess/quality/*` |

The entire `AutoFactorEvaluation-RECONSTRUCT/data_access/` and `AutoFactorEvaluation-RECONSTRUCT/factor_engine/` trees are quarantined duplicate implementations, not alternate production authorities. The `_r38_spool/` directory is not empty (six Arrow/Parquet shards) and `_r38_spill_store/` contains one Parquet spill object; these are runtime artifacts, not source implementations, but should not be treated as authoritative storage.

## second DSL/compiler

| file | why-not |
|---|---|
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/api/dsl_parser.py` | Full second FE DSL parser; quarantine duplicate of current FE; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/ashare_lqtp_kit/ashare_lqtp/dsl_compat.py` | Compatibility DSL parser/normalizer outside FE; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/gtja191/lib/dsl_legacy_ops.py` | Legacy DSL operation mapping outside FE registry; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/gtja191/lib/dsl_normalize.py` | Second DSL normalization layer; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/gtja191/lib/dsl_validate.py` | Second DSL validation layer; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/week2_pv_factors/lib/dsl_validate.py` | Second DSL validation layer; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/planner/compiler_pass.py` | Full second compiler-pass framework; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/runtime/engine.py:112` | Second compiler entry point; canonical: `factor_engine/api/dsl_parser.py`, `factor_engine/planner/compiler_pass.py`, and current planner/compiler package. |

## second materialization

| file | why-not |
|---|---|
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py:96` | Independent atomic materialization; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/integrations/quant_platform.py:491` | Legacy staging materialization; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/materializer.py:93` | Second ParquetMaterializer; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/factor_matrix_materializer.py:61` | Second factor matrix materializer; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/clickhouse_materializer.py:43` | Second ClickHouse materializer; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/runtime/materialize_service.py:42` | Second materialization service; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |
| `/home/shw/quant_projects/gtja191/scripts/run_materialize.py:182` | Standalone legacy materialization orchestration; canonical: `factor_engine/runtime/materialize_service.py` and `factor_engine/storage/materialize/`. |

## second cache/DAG

| file | why-not |
|---|---|
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/cache_utils.py` | Standalone legacy cache utilities; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/cache.py:14` | Second CacheManager/PersistentPlanCache; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/cache/column_cache.py:12` | Second column cache; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/cache/expression_cache.py:10` | Second expression cache; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/cache/panel_cache.py:14` | Second panel cache; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/planner/dag.py:22` | Second DAGPlan; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/planner/rolling_cache.py:32` | Second rolling cache; canonical: `factor_engine/storage/cache.py`, current `factor_engine/cache/`, `factor_engine/planner/dag.py`, plus `dataaccess/runtime/cache_manager.py` for DA read caching. |

## second DuckDB wrapper

| file | why-not |
|---|---|
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/core/duckdb_config.py` | Second DuckDB configuration; canonical: `dataaccess/core/engine.py`, `dataaccess/core/duckdb_config.py`, `dataaccess/core/duckdb_capabilities.py`, and `dataaccess/cos/s3_duckdb.py`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/cos/s3_duckdb.py` | Second S3/DuckDB wrapper; canonical: `dataaccess/core/engine.py`, `dataaccess/core/duckdb_config.py`, `dataaccess/core/duckdb_capabilities.py`, and `dataaccess/cos/s3_duckdb.py`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/duckdb_pushdown_backend.py` | Second DuckDB pushdown backend; canonical: `dataaccess/core/engine.py`, `dataaccess/core/duckdb_config.py`, `dataaccess/core/duckdb_capabilities.py`, and `dataaccess/cos/s3_duckdb.py`. |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/backend/sql_pushdown/duckdb_capabilities.py` | Second DuckDB capability registry; canonical: `dataaccess/core/engine.py`, `dataaccess/core/duckdb_config.py`, `dataaccess/core/duckdb_capabilities.py`, and `dataaccess/cos/s3_duckdb.py`. |

## production code reaching into pandas in hot path

No direct pandas imports were found under `factor_engine/kernels/`, `factor_engine/operators/`, `factor_engine/backend/`, or `factor_engine/backends/`. The following production planner/runtime/storage and DA read/write paths do import pandas. These are boundary/performance review candidates; presence alone does not prove the pandas branch is selected on the fast path.

| file | line | import | assessment |
|---|---:|---|---|
| `/home/shw/quant_projects/factor_engine/runtime/ashare_intraday.py` | 7 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/intraday_aggregator.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/planner/plan_hash.py` | 492 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/planner/plan_hash.py` | 538 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/batch_service.py` | 719 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/intermediate_registry.py` | 55 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/resource_calibration_store.py` | 313 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/resource_calibration_store.py` | 326 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/run_window.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/materialize_batch.py` | 36 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/auto_shard_planner.py` | 151 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/auto_shard_planner.py` | 186 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/auto_shard_planner.py` | 207 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/auto_shard_planner.py` | 232 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/auto_shard_planner.py` | 390 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/real_data_factor_smoke.py` | 158 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/real_data_factor_smoke.py` | 247 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/planner/output_slice.py` | 73 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/engine.py` | 36 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/planner/data_shape_estimate.py` | 34 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/store.py` | 7216 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/reconcile/dual_write_service.py` | 22 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/planner/physical_plan.py` | 255 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/reconcile/dual_write_reconcile.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/resource_governor.py` | 696 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/incremental_scheduler.py` | 22 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/warmup_service.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/stateful_incremental.py` | 28 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/factor_block_ref.py` | 48 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/clickhouse/write.py` | 157 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/clickhouse/write.py` | 186 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/clickhouse/write.py` | 237 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/clickhouse/write.py` | 288 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/incremental.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/materialize_service.py` | 13 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/batch_warmup_plan.py` | 91 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/session_panel.py` | 29 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/quality/dq_gates.py` | 13 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/quality/input_dq.py` | 11 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/factor_identity.py` | 736 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/stateful_checkpoint_store.py` | 19 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/session_calendar.py` | 15 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/spill_store.py` | 137 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/spill_store.py` | 182 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/intraday_session.py` | 14 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/change_impact.py` | 34 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/source_window_contract_v2.py` | 15 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 66 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 116 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 148 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 509 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 575 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 608 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 638 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 665 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 678 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 764 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 814 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 864 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/runtime/shard_executor.py` | 899 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/write/upsert.py` | 418 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/write/upsert.py` | 569 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/cos_event_runtime.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/predicate.py` | 26 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/adapters.py` | 75 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/adapters.py` | 205 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/pit_event_index.py` | 296 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/read_handle.py` | 349 | `import pandas as pd  # noqa: F401` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/session_calendar.py` | 383 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/resample.py` | 53 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/dataaccess/read/relation_handle.py` | 307 | `import pandas as pd  # noqa: F401` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/schema_migration.py` | 10 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/lqtp_logical_source_v2.py` | 10 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/staging_loader.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/q_backend/q_executor.py` | 19 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/composite_source.py` | 711 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/composite_source.py` | 746 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/composite_source.py` | 787 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/composite_source.py` | 852 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/financial.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/q_backend/q_backend.py` | 22 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/intraday_feature_runtime_v2.py` | 18 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/q_backend/test_q_residency.py` | 12 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/relation.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/q_backend/q_adapter.py` | 29 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 359 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 411 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 454 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 552 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 612 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 638 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/parquet_source.py` | 656 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/kline_parquet_source.py` | 204 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/kline_parquet_source.py` | 249 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/kline_parquet_source.py` | 324 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/kline_parquet_source.py` | 325 | `from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/pandas_backend.py` | 28 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/data_access_source.py` | 16 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/data_access_source.py` | 1387 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/lqtp_logical_source.py` | 15 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/relation_metrics.py` | 14 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/sources/intraday_feature_extension.py` | 18 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/matrix_block_layout.py` | 37 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/factor_frame.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/trading_calendar.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/delta_store.py` | 46 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/recipe_execution_fingerprint.py` | 16 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/cache.py` | 14 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/block_lake.py` | 40 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/materialize/lake_publish.py` | 199 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/materialize/factor_matrix_materializer.py` | 44 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/materialize/materializer.py` | 43 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/materialize/materializer.py` | 2684 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/pandas_compat.py` | 57 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/materialize/write_targets.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/result_store.py` | 5 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/factor_format.py` | 8 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/time_window.py` | 11 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/plan_cost_router.py` | 435 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/plan_cost_router.py` | 438 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/plan_cost_router.py` | 441 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/partition_policy.py` | 9 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/storage/partition_stats.py` | 24 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/polars_backend_kind.py` | 150 | `supports_nulls=True,  # Inherited from pandas reference` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/fastpath_plan_probe.py` | 109 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 10 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 133 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 433 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 443 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 460 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 483 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 522 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 543 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 558 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 578 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 590 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/elementwise_semantics.py` | 603 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |
| `/home/shw/quant_projects/factor_engine/backend/polars_panel.py` | 50 | `import pandas as pd` | Review: pandas enters a production compute/read/materialization path |

## tooling inventory

| item | location | setup |
|---|---|---|
| pre-commit | `/home/shw/quant_projects/.pre-commit-config.yaml` | One local `data-access-allowlist` hook invokes `python scripts/check_data_access_allowlist.py` on Python files; excludes workspace/venv/build/dist paths. `pre-commit>=3.0` is in `requirements-dev.txt`. |
| pytest | `/home/shw/quant_projects/pytest.ini` | Defines `integration`, `perf`, and `slow` markers; comments describe excluding integration tests. No default `addopts` is configured. |
| Makefile | `/home/shw/quant_projects/Makefile` | One target: `audit-factor-engine`, invoking `bash factor_engine/scripts/audit_factor_engine.sh`. |
| runtime requirements | `/home/shw/quant_projects/requirements.txt` | PyYAML, NumPy, pandas, PyArrow, DuckDB, Polars with pinned core versions. |
| development requirements | `/home/shw/quant_projects/requirements-dev.txt` | pytest, pre-commit, pandas/PyArrow/DuckDB/Polars, scientific stack, service-test dependencies, and optional Phase 10 packages. |
| service requirements | `/home/shw/quant_projects/requirements-service.txt` | FastAPI, Uvicorn, HTTPX, Pydantic. |
