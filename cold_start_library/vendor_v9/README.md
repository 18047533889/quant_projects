# FactorEngine Cold Start Backend-Audited V9

V9在V8基础上重新完成全量FactorEngine Pandas执行、IR/Plan审计、后端能力检查和SQL编译。

- 可用库：`library/factorengine_cold_start_backend_audited_v9.json`（6266条）
- 默认核心：`library/production_default_core_v9.json`（6137条）
- 可选数据：`library/production_optional_v9.json`（129条）
- 三后端静态路径：`library/three_backend_static_path_v9.json`（5263条）
- 合同严格池：`library/contract_strict_v9.json`（5662条）
- 事件归档：`library/event_archive_v9.json`（2条）
- 详细报告：`reports/backend_audit_report_v9.md`

注意：当前隔离容器无法取得Polars/DuckDB/PyArrow安装包，所以“静态路径”不能替代真实数值parity。原生复验脚本已随包提供并会拒绝stub。
