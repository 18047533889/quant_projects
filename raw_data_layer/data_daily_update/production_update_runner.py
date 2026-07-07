#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产环境每日数据增量更新脚本
Production Daily Incremental Data Update Runner
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yaml

# ========================================
# 配置 (Configuration)
# ========================================

# --- 日志配置 ---
# 配置日志记录，确保所有输出都能被shell脚本捕获
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ProductionUpdateRunner")

# --- 路径配置 ---
# production_update_runner.py lives at raw_data_layer/data_daily_update/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MASSIVE_ROOT = Path(
    os.environ.get("MASSIVE_ROOT", "/home/yluel/share/projects/massive_parquet")
)
RAW_DATA_ROOT = MASSIVE_ROOT / "raw_massive_data"
META_ROOT = Path(__file__).resolve().parent / "_meta"
CONFIG_DIR = META_ROOT / "config"
DATA_ROOT = RAW_DATA_ROOT
REPORTS_DIR = META_ROOT / "reports"
CURSOR_DIR = META_ROOT / "cursors"
PRODUCTION_CONFIG_PATH = CONFIG_DIR / "production_config.yaml"
CRON_PAUSED_FILE = Path(__file__).resolve().parent / "CRON_PAUSED"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从主更新模块导入核心类
from raw_data_layer.data_daily_update.async_incremental_update import AsyncIncrementalUpdateMaster

# ========================================
# 数据源和系统配置
# ========================================

def get_data_sources_config():
    """
    定义所有需要更新的数据源列表。
    这份配置是我们之前在测试中最终确认的稳定版本。
    """
    return {
        'data_sources': [
            {'enabled': True, 'name': 'tickers_all', 'type': 'rest_api', 'rest_api': {'endpoint': '/v3/reference/tickers'}, 'category': 'tickers', 'sub': 'all_tickers', 'p': 'all_tickers', 'mode': 'all', 'time_column': 'last_updated_utc'},
            {'enabled': True, 'name': 's3_trades', 'type': 's3', 's3': {'prefix': 'us_stocks_sip/trades_v1/'}, 'category': 'market_data', 'sub': 'us_stocks_sip', 'p': 'trades_v1', 'mode': 'date', 'partition_freq': 'month', 'time_column': 'sip_timestamp'},
            {'enabled': True, 'name': 's3_quotes', 'type': 's3', 's3': {'prefix': 'us_stocks_sip/quotes_v1/'}, 'category': 'market_data', 'sub': 'us_stocks_sip', 'p': 'quotes_v1', 'mode': 'date', 'partition_freq': 'month', 'time_column': 'sip_timestamp'},
            {'enabled': True, 'name': 'tickers_types', 'type': 'rest_api', 'rest_api': {'endpoint': '/v3/reference/tickers/types'}, 'category': 'tickers', 'sub': 'ticker_types', 'p': 'ticker_types', 'mode': 'all'},
            {'enabled': True, 'name': 'market_exchanges', 'type': 'rest_api', 'rest_api': {'endpoint': '/v3/reference/exchanges'}, 'category': 'market_operations', 'sub': 'exchanges', 'p': 'exchanges', 'mode': 'all'},
            {'enabled': True, 'name': 'filing_sec_edgar_index', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/filings/vX/index'}, 'category': 'filing', 'sub': 'sec_edgar_index', 'p': 'sec_edgar_index', 'mode': 'all', 'time_column': 'filing_date'},
            {'enabled': True, 'name': 'filing_risk_categories', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/taxonomies/vX/risk-factors'}, 'category': 'filing', 'sub': 'risk_categories', 'p': 'risk_categories', 'mode': 'all'},
            {'enabled': True, 'name': 'filing_risk_factors', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/filings/vX/risk-factors'}, 'category': 'filing', 'sub': 'risk_factors', 'p': 'risk_factors', 'mode': 'date', 'time_column': 'filing_date'},
            {'enabled': True, 'name': 'short_volume', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/v1/short-volume'}, 'category': 'fundamentals', 'sub': 'short_volume', 'p': 'short_volume', 'mode': 'date', 'time_column': 'date'},
            {'enabled': True, 'name': 'short_interest', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/v1/short-interest'}, 'category': 'fundamentals', 'sub': 'short_interest', 'p': 'short_interest', 'mode': 'date', 'time_column': 'settlement_date'},
            {'enabled': True, 'name': 'aggs_daily_market_summary', 'type': 'rest_api', 'rest_api': {'endpoint': '/v2/aggs/grouped/locale/us/market/stocks/{date}'}, 'category': 'aggregate_bars', 'sub': 'daily_market_summary', 'p': 'daily_market_summary', 'mode': 'calendar_year', 'time_column': 'trade_date'},
            {'enabled': True, 'name': 'corp_dividends', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/v1/dividends'}, 'category': 'corporate_actions', 'sub': 'dividends', 'p': 'dividends', 'mode': 'date', 'time_column': 'ex_dividend_date'},
            {'enabled': True, 'name': 'corp_splits', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/v1/splits'}, 'category': 'corporate_actions', 'sub': 'splits', 'p': 'splits', 'mode': 'date', 'time_column': 'execution_date'},
            {'enabled': True, 'name': 'corp_ipos', 'type': 'rest_api', 'rest_api': {'endpoint': '/vX/reference/ipos'}, 'category': 'corporate_actions', 'sub': 'ipos', 'p': 'ipos', 'mode': 'all'},
            {'enabled': True, 'name': 'news_all', 'type': 'rest_api', 'rest_api': {'endpoint': '/v2/reference/news'}, 'category': 'news', 'sub': 'news', 'p': 'news', 'mode': 'all', 'time_column': 'published_utc'},
            {'enabled': True, 'name': 'market_holidays', 'type': 'rest_api', 'rest_api': {'endpoint': '/v1/marketstatus/upcoming'}, 'category': 'market_operations', 'sub': 'market_holidays', 'p': 'market_holidays', 'mode': 'all', 'time_column': 'date'},
            {'enabled': True, 'name': 'stocks_floats', 'type': 'rest_api', 'rest_api': {'endpoint': '/stocks/vX/float'}, 'category': 'fundamentals', 'sub': 'stocks_floats', 'p': 'stocks_floats', 'mode': 'all'},
            {'enabled': True, 'name': 'market_condition_codes', 'type': 'rest_api', 'rest_api': {'endpoint': '/v3/reference/conditions'}, 'category': 'market_operations', 'sub': 'condition_codes', 'p': 'condition_codes', 'mode': 'all'},
            {'enabled': True, 'name': 'balance_sheet', 'type': 'rest_or_sdk', 'rest_api': {'endpoint': '/stocks/financials/v1/balance-sheets'}, 'sdk': {'method': 'list_financials_balance_sheets'}, 'category': 'fundamentals', 'sub': 'balance_sheet', 'p': 'balance_sheet', 'mode': 'fiscal', 'time_column': 'period_end'},
            {'enabled': True, 'name': 'income_statement', 'type': 'rest_or_sdk', 'rest_api': {'endpoint': '/stocks/financials/v1/income-statements'}, 'sdk': {'method': 'list_financials_income_statements'}, 'category': 'fundamentals', 'sub': 'income_statement', 'p': 'income_statement', 'mode': 'fiscal', 'time_column': 'period_end'},
            {'enabled': True, 'name': 'cash_flow_statement', 'type': 'rest_or_sdk', 'rest_api': {'endpoint': '/stocks/financials/v1/cash-flow-statements'}, 'sdk': {'method': 'list_financials_cash_flow_statements'}, 'category': 'fundamentals', 'sub': 'cash_flow_statement', 'p': 'cash_flow_statement', 'mode': 'fiscal', 'time_column': 'period_end'},
            {'enabled': True, 'name': 'financials_ratios', 'type': 'rest_or_sdk', 'rest_api': {'endpoint': '/stocks/financials/v1/ratios'}, 'sdk': {'method': 'list_financials_ratios'}, 'category': 'fundamentals', 'sub': 'financials_ratios', 'p': 'financials_ratios', 'mode': 'all'}
        ]
    }

def get_system_config():
    """
    定义生产环境的系统级配置。
    """
    return {
        'system': {'name': 'Production Incremental Data Update System', 'version': '1.0.1', 'log_level': 'INFO', 'max_parallel_workers': 8},
        'incremental_update': {'max_retry_attempts': 3, 'retry_delay_seconds': 60, 'parallel_workers': 5, 'continue_on_failure': True},
        'performance': {'chunk_size': 50000, 'max_memory_usage_mb': 4096},
        'cursor': {'storage_path': str(CURSOR_DIR)},
        'reporting': {'enabled': True, 'output_path': str(REPORTS_DIR)},
        'output': {'path': str(DATA_ROOT), 'format': 'parquet'},
        'api': {
            'default_api_key': os.environ.get("MASSIVE_API_KEY", ""),
            'raw_data': {'path': str(DATA_ROOT), 'format': 'parquet', 'enabled': True},
            'cleaned_data': {'path': str(MASSIVE_ROOT / "cleaned_massive_data"), 'format': 'parquet'}
        }
    }

def assert_cron_allowed():
    """TASK-ENG-004: refuse to run while CRON_PAUSED or env gate unset."""
    if CRON_PAUSED_FILE.exists():
        logger.error(
            "TASK-ENG-004: incremental update PAUSED (%s). "
            "Complete P0 backfill + FEAT-008 before re-enabling.",
            CRON_PAUSED_FILE,
        )
        sys.exit(1)
    if os.environ.get("MASSIVE_CRON_ENABLED") != "1":
        logger.error(
            "TASK-ENG-004: set MASSIVE_CRON_ENABLED=1 to run production_update_runner."
        )
        sys.exit(1)
    if not os.environ.get("MASSIVE_API_KEY"):
        logger.error("MASSIVE_API_KEY is required (no default key in code).")
        sys.exit(1)

def prepare_production_environment():
    """
    准备生产环境所需的目录结构和配置文件。
    """
    logger.info("--- Preparing Production Environment ---")
    
    # 创建所有必需的目录
    for path in [CONFIG_DIR, REPORTS_DIR, CURSOR_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Meta dirs: %s | raw output root: %s", META_ROOT, DATA_ROOT)

    # 将系统配置和数据源配置合并，并写入YAML文件
    full_config = {**get_system_config(), **get_data_sources_config()}
    with open(PRODUCTION_CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(full_config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"Production configuration created at: {PRODUCTION_CONFIG_PATH}")
    
    # 生产环境不应每次都创建新的游标文件，而是依赖于上一次运行留下的游标
    # 如果游标文件不存在，master会自动处理
    if not (CURSOR_DIR / "cursor.json").exists():
        logger.warning("Cursor file not found. The master will perform a full run or use default start times.")
        
    logger.info("--- Environment Preparation Complete ---")

# ========================================
# 主执行逻辑
# ========================================

async def main():
    """
    异步主函数，负责协调整个更新流程。
    """
    assert_cron_allowed()
    prepare_production_environment()
    
    logger.info("--- Starting Production Incremental Update ---")
    
    master = None
    try:
        # 使用配置文件和生产数据根目录来创建Master实例
        master = await AsyncIncrementalUpdateMaster.create(
            config_path=PRODUCTION_CONFIG_PATH,
            data_root=DATA_ROOT
        )
        
        logger.info("AsyncIncrementalUpdateMaster initialized. Running all enabled sources...")
        
        # 运行所有在配置中启用的数据源
        results = await master.run_all_sources(batch_size=5, delay_hours=0)
        
        logger.info("--- Update Execution Summary ---")
        
        success_count = 0
        total_records = 0
        
        # 打印格式化的结果
        for result in results:
            status = result.get('status', 'unknown')
            source_name = result.get('source', 'N/A')
            records = result.get('records', 0)
            duration = result.get('duration_seconds', 0)
            
            if status in ['success', 'skipped_by_ok', 'no_data']:
                logger.info(f"✅ {source_name:<25} | Status: {status:<15} | Records: {records:<7} | Duration: {duration:.2f}s")
                if status == 'success':
                    success_count += 1
                    total_records += records
            else:
                error_msg = result.get('error', 'No error details.')
                logger.error(f"❌ {source_name:<25} | Status: {status:<15} | Error: {error_msg}")

        logger.info("--- Final Report ---")
        logger.info(f"Successfully updated sources: {success_count} / {len(results)}")
        logger.info(f"Total new records fetched: {total_records}")
        
        if success_count == 0 and len(results) > 0:
            logger.warning("❌ Update run concluded with no successful updates.")
            sys.exit(1) # 以失败状态退出，方便cron捕获
        else:
            logger.info("✅ Update run concluded successfully.")
            
    except Exception as e:
        logger.error(f"An unhandled error occurred during the update run: {e}", exc_info=True)
        sys.exit(1) # 以失败状态退出
    finally:
        if master:
            await master.close()
            logger.info("--- Update Run Finished ---")


if __name__ == "__main__":
    asyncio.run(main())