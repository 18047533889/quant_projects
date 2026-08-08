# quant_projects monorepo 顶层验收入口（Phase 5 P1-16）。
# 全部只在服务器本地运行，不触碰 GitHub。
.PHONY: audit-factor-engine
audit-factor-engine:
	bash factor_engine/scripts/audit_factor_engine.sh
