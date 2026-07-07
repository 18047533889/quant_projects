#!/usr/bin/env python3
"""
异步版本的 UniversalFetcher - 解决 ThreadPoolExecutor 阻塞问题
集成到 data_daily_update 模块
"""

import os
import sys
import asyncio
import logging
import threading
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import aiohttp
import aiofiles
import json
from urllib.parse import urlparse, parse_qs
import hashlib
import tempfile

# 导入项目内部模块
try:
    from raw_data_layer.raw_data_fetching.download_history import (
        create_s3_client, list_keys_streaming, process_one_object,
        key_to_local_path, sha256_file_hex, ALLOWED_SUFFIXES,
        PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL
    )
except ImportError:
    # 如果在 data_daily_update 目录中运行，尝试相对导入
    try:
        from ..raw_data_fetching.download_history import (
            create_s3_client, list_keys_streaming, process_one_object,
            key_to_local_path, sha256_file_hex, ALLOWED_SUFFIXES,
            PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL
        )
    except ImportError:
        # 如果仍然失败，使用绝对路径导入
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from raw_data_layer.raw_data_fetching.download_history import (
            create_s3_client, list_keys_streaming, process_one_object,
            key_to_local_path, sha256_file_hex, ALLOWED_SUFFIXES,
            PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL
        )

# 尝试导入 massive.RESTClient
try:
    from massive import RESTClient
    HAS_RESTCLIENT = True
except ImportError:
    HAS_RESTCLIENT = False


def normalize_chunk_dtypes(df):
    """标准化数据类型 - 基于download_history.py的健壮配置"""
    # 强制特定列转换为字符串，避免数据类型转换错误
    FORCE_STRING_COLUMNS = ["ticker", "conditions", "indicators"]
    if FORCE_STRING_COLUMNS:
        for col in FORCE_STRING_COLUMNS:
            if col in df.columns:
                try:
                    df[col] = df[col].astype("string")
                except Exception:
                    # 如果string类型失败，回退到普通str类型
                    try:
                        df[col] = df[col].astype(str)
                    except Exception as e:
                        # 如果转换完全失败，记录警告但继续处理
                        logger.warning(f"列 {col} 转换为字符串失败: {e}")
    
    # 将整数转换为浮点数以避免模式不匹配（跨chunk一致性）
    CAST_INT_TO_FLOAT = True
    if CAST_INT_TO_FLOAT:
        int_cols = df.select_dtypes(include=["int64", "int32", "Int64"]).columns
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype("float64")
    return df


def write_checksum_file(parquet_path, hex_digest):
    """写入校验和文件"""
    chk_path = str(parquet_path) + ".sha256"
    with open(chk_path, "w", encoding="utf-8") as f:
        f.write(hex_digest)
    return chk_path

try:
    from raw_data_layer.raw_data_fetching.download_all_history import MassiveRestDownloader
    from massive import RESTClient
    HAS_MASSIVE_REST = True
except ImportError:
    HAS_MASSIVE_REST = False

try:
    from raw_data_layer.raw_data_cleaning.massive_cleaning_framework import clean_one_file, clean_dataframe
    from raw_data_layer.raw_data_cleaning.massive_cleaning_framework import SourceConfig
    HAS_CLEANING = True
except ImportError:
    try:
        from ..raw_data_cleaning.massive_cleaning_framework import clean_one_file, clean_dataframe
        from ..raw_data_cleaning.massive_cleaning_framework import SourceConfig
        HAS_CLEANING = True
    except ImportError:
        HAS_CLEANING = False

# ==========================
# 日志配置
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AsyncUniversalFetcher")

# ==========================
# Massive API 配置
# ==========================
MASSIVE_API_ENDPOINT = "https://api.massive.com"
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")  # 正确的REST API密钥
MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_S3_BUCKET = "flatfiles"

class AsyncUniversalFetcher:
    """
    异步通用数据获取器
    支持 S3、REST API、本地文件等多种数据源
    """
    
    def __init__(self, data_root: Optional[Path] = None):
        """初始化异步获取器"""
        self.data_root = data_root or Path("../massive_parquet")
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        self.session = None
        self.s3_client = None
        self._cleanup_tasks = []
        # 降低默认并发数以避免阻塞问题
        self._s3_semaphore = asyncio.Semaphore(3)  # 限制S3并发数为3
        
        # 初始化 S3 客户端 - 增加连接池和超时配置
        self._init_s3_client()
        
        # 降低S3并发数以避免阻塞
        self._s3_semaphore = asyncio.Semaphore(3)  # 从5降到3

    def _init_s3_client(self):
        """初始化 S3 客户端 - 增强配置"""
        try:
            self.access_key = os.environ.get("MASSIVE_AWS_KEY", "f04c088e-d96f-4944-a048-8e419c0bd524")
            self.secret_key = os.environ.get("MASSIVE_AWS_SECRET", "0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")
            self.endpoint_url = "https://files.massive.com"
            
            # 使用更robust的配置
            import boto3
            from botocore.config import Config
            
            config = Config(
                signature_version="s3v4",
                connect_timeout=60,
                read_timeout=300,
                max_pool_connections=64,
                retries={
                    "max_attempts": 10,
                    "mode": "adaptive",
                },
            )
            
            session = boto3.session.Session()
            self.s3_client = session.client(
                "s3",
                endpoint_url="https://files.massive.com",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
            )
            logger.info("S3 client initialized successfully with enhanced config")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 创建 aiohttp 会话 - 降低并发数以避免429错误
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=30, limit_per_host=3)  # 从100/10降低到30/3
        )
        
        logger.info("AsyncUniversalFetcher 已启动")
        return self
    
    def _is_trading_day(self, date_str: str) -> bool:
        """
        检查是否为交易日（简化版市场日历）
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            是否为交易日
        """
        try:
            check_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # 检查周末
            if check_date.weekday() >= 5:  # 周六(5)和周日(6)
                return False
            
            # 检查主要美国市场节假日（简化版）
            year = check_date.year
            
            # 新年
            new_year = date(year, 1, 1)
            if check_date == new_year:
                return False
            
            # 马丁·路德·金纪念日（1月第三个周一）
            mlk_day = date(year, 1, 1) + timedelta(days=((2 - date(year, 1, 1).weekday()) % 7) + 14)
            if check_date == mlk_day:
                return False
            
            # 总统日（2月第三个周一）
            presidents_day = date(year, 2, 1) + timedelta(days=((2 - date(year, 2, 1).weekday()) % 7) + 14)
            if check_date == presidents_day:
                return False
            
            # 耶稣受难日（复活节前的周五）- 简化计算
            # 这里使用近似计算，实际应用中可能需要更精确的算法
            
            # 阵亡将士纪念日（5月最后一个周一）
            memorial_day = date(year, 5, 31) - timedelta(days=(date(year, 5, 31).weekday() - 1) % 7)
            if check_date == memorial_day:
                return False
            
            # 独立日
            independence_day = date(year, 7, 4)
            if check_date == independence_day:
                return False
            
            # 劳动节（9月第一个周一）
            labor_day = date(year, 9, 1) + timedelta(days=((2 - date(year, 9, 1).weekday()) % 7))
            if check_date == labor_day:
                return False
            
            # 感恩节（11月第四个周四）
            thanksgiving = date(year, 11, 1) + timedelta(days=((3 - date(year, 11, 1).weekday()) % 7) + 21)
            if check_date == thanksgiving:
                return False
            
            # 圣诞节
            christmas = date(year, 12, 25)
            if check_date == christmas:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"检查交易日时出错: {e}，默认为交易日")
            return True
    
    def _get_last_trading_day(self, date_str: str) -> str:
        """
        获取最近的交易日
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            最近的交易日日期字符串
        """
        try:
            check_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # 向前查找最近的交易日
            current_date = check_date
            while current_date >= check_date - timedelta(days=30):  # 最多回溯30天
                current_str = current_date.strftime('%Y-%m-%d')
                if self._is_trading_day(current_str):
                    return current_str
                current_date -= timedelta(days=1)
            
            # 如果找不到，返回原始日期
            logger.warning(f"无法找到最近的交易日，使用原始日期: {date_str}")
            return date_str
            
        except Exception as e:
            logger.warning(f"获取最近交易日时出错: {e}，使用原始日期")
            return date_str
    
    async def check_massive_update_status(self, dataset: str, max_wait_minutes: int = 120, check_interval_minutes: int = 10) -> bool:
        """
        检查Massive数据更新状态 - [修改] 禁用检查，直接返回True
        """
        logger.info(f"跳过对 {dataset} 的Massive更新状态检查，直接进行数据获取。")
        return True
    
    async def _check_s3_dataset_updated(self, dataset: str) -> bool:
        """检查S3数据集是否已更新"""
        try:
            # 获取昨天的日期
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 构建S3前缀
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if dataset == 'trades':
                prefix = f"us_stocks_sip/trades_v1/{current_year}/{current_month:02d}/"
            elif dataset == 'quotes':
                prefix = f"us_stocks_sip/quotes_v1/{current_year}/{current_month:02d}/"
            elif dataset == 'day_aggs':
                prefix = f"us_stocks_sip/day_aggs_v1/{current_year}/{current_month:02d}/"
            else:
                return False
            
            # 检查今天或昨天的文件是否存在
            paginator = self.s3_client.get_paginator('list_objects_v2')
            async for page in self._async_s3_pages(paginator.paginate(Bucket=MASSIVE_S3_BUCKET, Prefix=prefix)):
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    last_modified = obj['LastModified'].replace(tzinfo=None)
                    
                    # 检查文件名是否包含今天或昨天的日期
                    if (today in key or yesterday in key) and key.endswith('.parquet'):
                        # 检查文件修改时间是否在合理范围内（24小时内）
                        # 统一时区处理：确保两者都是offset-naive
                        now = datetime.now()
                        last_modified_naive = last_modified.replace(tzinfo=None) if last_modified.tzinfo else last_modified
                        if now - last_modified_naive < timedelta(hours=24):
                            logger.debug(f"找到更新的S3文件: {key}, 修改时间: {last_modified}")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查S3数据集 {dataset} 状态时出错: {e}")
            return False
    
    async def _check_rest_dataset_updated(self, dataset: str) -> bool:
        """检查REST API数据集是否已更新"""
        try:
            # 获取昨天的日期
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # 获取最近的交易日
            last_trading_day = self._get_last_trading_day(yesterday)
            
            # 根据数据集构建检查URL
            check_urls = []
            
            if dataset == 'aggs_daily_market_summary':
                check_urls.append(f"{MASSIVE_API_ENDPOINT}/v2/aggs/grouped/locale/us/market/stocks/{last_trading_day}")
            elif dataset == 'corp_dividends':
                check_urls.append(f"{MASSIVE_API_ENDPOINT}/stocks/v1/dividends")
            elif dataset == 'corp_splits':
                check_urls.append(f"{MASSIVE_API_ENDPOINT}/stocks/v1/splits")
            else:
                # 通用检查：尝试获取最新数据
                check_urls.append(f"{MASSIVE_API_ENDPOINT}/v3/reference/tickers")
            
            # 检查API是否可访问且有数据
            for url in check_urls:
                params = {'apikey': MASSIVE_API_KEY, 'limit': 1}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'results' in data and len(data['results']) > 0:
                            logger.debug(f"REST API {url} 有数据返回")
                            return True
                    else:
                        logger.debug(f"REST API {url} 返回状态: {response.status}")
            
            return False
            
        except Exception as e:
            logger.error(f"检查REST数据集 {dataset} 状态时出错: {e}")
            return False
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        # 关闭 aiohttp 会话
        if self.session:
            await self.session.close()
            logger.info("AsyncUniversalFetcher aiohttp会话已关闭")
        
        # 等待清理任务
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            logger.info(f"所有清理任务已完成，共 {len(self._cleanup_tasks)} 个")
        
        logger.info("AsyncUniversalFetcher 已关闭")
    
    async def _clean_dataframe_async(self, df: pd.DataFrame, source_config: Dict[str, Any]) -> pd.DataFrame:
        """异步清洗DataFrame - 封装同步的clean_dataframe函数"""
        if not HAS_CLEANING:
            logger.warning("清洗框架未导入，直接返回原始数据")
            return df
        
        try:
            # 将字典配置转换为SourceConfig对象
            if isinstance(source_config, dict):
                # 从massive_cleaning_framework导入SourceConfig
                from raw_data_layer.raw_data_cleaning.massive_cleaning_framework import SourceConfig
                # 创建SourceConfig实例，手动指定所有必需参数，避免传入不支持的字段（如name）
                cfg = SourceConfig(
                    source=source_config.get('source', 'universal'),
                    path_pattern=source_config.get('path_pattern', '*.parquet'),
                    dataset_type=source_config.get('dataset_type', 'unknown'),
                    frequency=source_config.get('frequency', 'unknown'),
                    enabled=True,
                    ticker_column=source_config.get('ticker_column', 'ticker'),
                    align_time_column=source_config.get('align_time_column'),
                    time_columns=source_config.get('time_columns', []),
                    time_format=source_config.get('time_format'),
                    primary_key_columns=source_config.get('primary_key_columns', ['ticker']),
                    timezone=source_config.get('timezone', 'unknown'),
                    array_columns_to_explode=source_config.get('array_columns_to_explode', []),
                    dedup_strategy=source_config.get('dedup_strategy', 'strict_primary_key'),
                    allow_null_ticker=source_config.get('allow_null_ticker', False),
                    allow_null_align_time=source_config.get('allow_null_align_time', False),
                    notes=source_config.get('notes', '')
                )
            else:
                cfg = source_config
            
            # 使用线程池执行同步的清洗函数，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            cleaned_df, summary = await loop.run_in_executor(
                None, 
                lambda: clean_dataframe(df, cfg)
            )
            
            logger.info(f"数据清洗完成: 输入行数 {summary['rows_in']}, 输出行数 {summary['rows_out']}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"数据清洗过程中出错: {e}", exc_info=True)
            # 如果清洗失败，返回原始数据
            return df
    
    async def fetch(self, source_config: Dict[str, Any], cursor_time: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """
        异步获取数据的主入口
        
        Args:
            source_config: 数据源配置
            cursor_time: 游标时间，用于增量更新
            **kwargs: 其他参数，如 delay_hours
            
        Returns:
            pd.DataFrame: 获取到的数据
        """
        source_type = source_config.get('type')
        
        if source_type == 's3':
            return await self._fetch_s3(source_config, cursor_time, **kwargs)
        elif source_type == 'rest_api':
            return await self._fetch_rest_api(source_config, cursor_time, **kwargs)
        elif source_type == 'sdk' or source_type == 'rest_or_sdk':
            return await self._fetch_sdk(source_config, cursor_time, **kwargs)
        else:
            logger.warning(f"不支持的数据源类型: {source_type}")
            return pd.DataFrame()

    async def _fetch_s3(self, source_config: Dict[str, Any], cursor_time: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """从S3异步获取数据"""
        logger.info(f"开始从S3获取数据: {source_config['name']}")
        
        s3_config = source_config.get('s3', {})
        prefix = s3_config.get('prefix')
        
        if not prefix:
            logger.error("S3配置中缺少'prefix'")
            return pd.DataFrame()
        
        # 动态生成prefix，支持{YYYY}/{MM}等占位符
        now = datetime.now()
        prefix = prefix.format(YYYY=now.year, MM=f"{now.month:02d}", DD=f"{now.day:02d}")
        
        # 检查是否需要延迟处理
        delay_hours = kwargs.get('delay_hours', 25)
        
        # 异步列出S3对象
        keys_to_process = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            async for page in self._async_s3_pages(paginator.paginate(Bucket=MASSIVE_S3_BUCKET, Prefix=prefix)):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    last_modified = obj['LastModified'].replace(tzinfo=None)
                    
                    # 检查文件后缀
                    if not key.endswith(tuple(ALLOWED_SUFFIXES)):
                        continue
                    
                    # 检查是否需要延迟处理
                    if (now - last_modified).total_seconds() < delay_hours * 3600:
                        logger.debug(f"跳过最近的文件: {key} (修改于 {last_modified})")
                        continue
                    
                    # 增量更新逻辑：如果提供了游标，只处理比游标新的文件
                    if cursor_time and last_modified <= cursor_time:
                        logger.debug(f"跳过已处理的文件: {key} (修改于 {last_modified}, 游标: {cursor_time})")
                        continue
                    
                    keys_to_process.append(obj)
            
            logger.info(f"在 {prefix} 中找到 {len(keys_to_process)} 个新文件需要处理")
            
            if not keys_to_process:
                return pd.DataFrame()
            
            # 并发下载和处理
            tasks = [self._download_and_process_s3_object(obj) for obj in keys_to_process]
            results = await asyncio.gather(*tasks)
            
            # 过滤掉None的结果并合并
            all_dfs = [df for df in results if df is not None and not df.empty]
            if not all_dfs:
                return pd.DataFrame()
            
            # 合并所有DataFrame
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"成功合并 {len(all_dfs)} 个文件，总计 {len(combined_df)} 行")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"从S3获取数据时出错: {e}", exc_info=True)
            return pd.DataFrame()

    async def _async_s3_pages(self, paginator):
        """将boto3的分页器转换为异步生成器"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                # 在线程池中运行同步的 next()
                page = await loop.run_in_executor(None, next, paginator)
                yield page
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"S3分页时出错: {e}")
                break

    async def _download_and_process_s3_object(self, s3_object: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """下载并处理单个S3对象"""
        key = s3_object['Key']
        size = s3_object['Size']
        
        async with self._s3_semaphore:
            try:
                loop = asyncio.get_event_loop()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_path = Path(temp_dir) / Path(key).name
                    
                    # 异步下载
                    await loop.run_in_executor(
                        None,
                        self.s3_client.download_file,
                        MASSIVE_S3_BUCKET,
                        key,
                        str(local_path)
                    )
                    
                    # 异步读取和处理
                    if str(local_path).endswith('.parquet'):
                        df = await loop.run_in_executor(None, pd.read_parquet, local_path)
                        return df
                    else:
                        # 对于其他文件类型，可以扩展这里的逻辑
                        logger.warning(f"不支持的文件类型: {key}")
                        return None
                        
            except Exception as e:
                logger.error(f"处理S3对象 {key} 时出错: {e}", exc_info=True)
                return None

    async def _fetch_rest_api(self, source_config: Dict[str, Any], cursor_time: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """从REST API异步获取数据"""
        if not HAS_MASSIVE_REST:
            logger.error("MassiveRestDownloader 未导入，无法从REST API获取数据")
            return pd.DataFrame()

        logger.info(f"开始从REST API获取数据: {source_config['name']}")
        
        rest_config = source_config.get('rest_api', {})
        api_key = rest_config.get('api_key', MASSIVE_API_KEY)
        
        # 提取分页参数
        page_size = rest_config.get('page_size', 1000)
        max_pages = rest_config.get('max_pages', 10) # 默认限制页数以防失控
        
        # 准备下载器
        if not HAS_RESTCLIENT:
            logger.error("massive.RESTClient 未导入，无法创建下载器")
            return pd.DataFrame()

        rest_client = RESTClient(api_key=api_key)
        downloader = MassiveRestDownloader(client=rest_client, root_dir=tempfile.gettempdir())
        
        # 准备请求参数
        params = rest_config.get('params', {})
        params['limit'] = page_size  # 将page_size作为limit参数
        
        # 增量更新逻辑 - 重构以正确处理不同模式
        mode = source_config.get('mode')
        if cursor_time and 'time_column' in source_config:
            time_col = source_config['time_column']
            
            # 根据模式构建正确的参数
            if mode == 'fiscal':
                # 对于财报模式，我们使用 fiscal_year.gte
                param_key = "fiscal_year.gte"
                param_value = cursor_time.year
                params[param_key] = param_value
                logger.info(f"为 {source_config['name']} (模式: {mode}) 设置增量参数: {param_key}={param_value}")
            elif mode == 'date':
                # 对于按日期的模式，使用 YYYY-MM-DD 格式
                param_key = f"{time_col}.gte"
                param_value = cursor_time.strftime('%Y-%m-%d')
                params[param_key] = param_value
                logger.info(f"为 {source_config['name']} (模式: {mode}) 设置增量参数: {param_key}={param_value}")
            elif 'timestamp' in time_col.lower():
                # 对于基于时间戳的模式，使用纳秒时间戳
                param_key = f"{time_col}.gte"
                param_value = int(cursor_time.timestamp() * 1e9)
                params[param_key] = param_value
                logger.info(f"为 {source_config['name']} (模式: timestamp) 设置增量参数: {param_key}={param_value}")
            else:
                # 其他情况或备用逻辑
                param_key = f"{time_col}.gte"
                param_value = cursor_time.strftime('%Y-%m-%d')
                params[param_key] = param_value
                logger.warning(f"未知的增量更新模式，默认使用日期格式: {param_key}={param_value}")

        df = pd.DataFrame()  # 初始化df
        try:
            # 动态格式化端点
            endpoint_template = source_config.get('rest_api', {}).get('endpoint')

            if mode == 'calendar_year' and endpoint_template and '{date}' in endpoint_template:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                target_date = self._get_last_trading_day(yesterday)
                endpoint = endpoint_template.format(date=target_date)
                logger.info(f"为 calendar_year 模式格式化端点: {endpoint}")
            else:
                endpoint = endpoint_template

            # 异步执行
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: pd.DataFrame(list(downloader._iter_raw(
                    endpoint=endpoint,
                    max_pages=max_pages,
                    **params
                )))
            )
        except Exception as e:
            logger.error(f"在 _fetch_rest_api 中执行下载时发生错误: {e}", exc_info=True)
            # 确保在出错时返回空的DataFrame
            return pd.DataFrame()
        
        if df is not None and not df.empty:
            logger.info(f"从 {source_config['name']} 成功获取 {len(df)} 行数据")
        else:
            logger.info(f"从 {source_config['name']} 未获取到新数据")
            return pd.DataFrame()
            
        return df

    async def _fetch_sdk(self, source_config: Dict[str, Any], cursor_time: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """
        通过SDK或REST API（对于rest_or_sdk类型）异步获取数据
        
        对于 'rest_or_sdk' 类型，此方法会优先尝试使用 'rest_api' 配置。
        如果 'rest_api' 配置不存在，它将回退到使用 'sdk' 配置。
        """
        source_type = source_config.get('type')
        
        # 如果是 rest_or_sdk 类型并且有 rest_api 配置，则优先使用 _fetch_rest_api
        if source_type == 'rest_or_sdk' and 'rest_api' in source_config:
            logger.info(f"数据源 '{source_config['name']}' 类型为 'rest_or_sdk'，优先使用 REST API 路径。")
            return await self._fetch_rest_api(source_config, cursor_time, **kwargs)
        
        # 对于 'sdk' 类型或没有 'rest_api' 配置的 'rest_or_sdk'，执行原始的SDK逻辑
        logger.info(f"开始从SDK获取数据: {source_config['name']}")
        
        if not HAS_RESTCLIENT:
            logger.error("Massive RESTClient 未安装或导入失败，无法使用SDK方法。")
            return pd.DataFrame()
            
        sdk_config = source_config.get('sdk', {})
        method_name = sdk_config.get('method')
        
        if not method_name:
            logger.error("SDK配置中缺少'method'")
            return pd.DataFrame()
            
        try:
            # 注意：这里的 massive.RESTClient 是同步的。
            # 在异步代码中直接调用同步SDK方法会阻塞事件循环。
            # 理想情况下，应该使用异步SDK或在线程池中运行同步SDK。
            # 为了保持与 download_all_history.py 的行为一致，我们暂时接受这种阻塞行为。
            
            client = RESTClient(api_key=MASSIVE_API_KEY)
            
            if not hasattr(client, method_name):
                logger.error(f"Massive RESTClient 中没有找到方法: {method_name}")
                return pd.DataFrame()
                
            method_to_call = getattr(client, method_name)
            
            # 构建SDK方法的参数
            params = {}
            mode = source_config.get('mode')
            time_column = source_config.get('time_column')

            # 处理增量更新的查询参数
            if cursor_time and time_column and mode in ['date', 'fiscal']:
                # Massive SDK 通常使用 `timestamp_gte` 或类似参数
                # 这里我们假设一个通用的 `gte` 格式，如 `period_end.gte`
                # 这需要根据具体SDK方法进行调整
                cursor_str = cursor_time.strftime('%Y-%m-%d')
                params[f'{time_column}.gte'] = cursor_str
                logger.info(f"为 '{source_config['name']}' 设置增量查询: {time_column} >= {cursor_str}")

            # 添加其他可能的参数，例如 limit
            params['limit'] = 1000  # 示例限制

            # 调用SDK方法并获取迭代器
            results_iterator = method_to_call(**params)
            
            # 将迭代器结果转换为DataFrame
            # to_dict 的实现参考了 download_all_history.py
            def _to_dict(x: object) -> dict:
                if hasattr(x, "model_dump"):
                    return x.model_dump()
                if hasattr(x, "__dict__"):
                    return x.__dict__
                return dict(x)

            records = [_to_dict(item) for item in results_iterator]
            
            if not records:
                logger.info(f"SDK方法 '{method_name}' 没有返回任何数据。")
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            logger.info(f"成功从SDK获取 {len(df)} 条记录: {source_config['name']}")
            return df

        except Exception as e:
            logger.error(f"通过SDK获取数据时出错: {source_config['name']}. 错误: {e}", exc_info=True)
            return pd.DataFrame()

    async def save_to_parquet(
        self,
        df: pd.DataFrame,
        source_name: str,
        category: str,
        subcategory: str,
        is_cleaned: bool = False
    ) -> Optional[Path]:
        """
        异步将DataFrame保存到Parquet文件
        
        Args:
            df: 要保存的DataFrame
            source_name: 数据源名称
            category: 类别
            subcategory: 子类别
            is_cleaned: 是否为已清洗的数据
            
        Returns:
            保存的文件路径
        """
        if df.empty:
            logger.info(f"DataFrame为空，跳过为 {source_name} 保存Parquet文件")
            return None
            
        try:
            # 确定是保存原始数据还是清洗后的数据
            data_type_folder = "cleaned" if is_cleaned else "raw"
            
            # 构建保存路径
            # massive_parquet/raw/{category}/{subcategory}/{source_name}/{YYYY-MM-DD}.parquet
            save_dir = self.data_root / data_type_folder / category / subcategory / source_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用当前日期作为文件名
            file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.parquet"
            file_path = save_dir / file_name
            
            # 异步写入文件
            # 使用pyarrow异步写入，需要 aiofiles
            async with aiofiles.open(file_path, 'wb') as f:
                # 将DataFrame转换为pyarrow Table
                table = pa.Table.from_pandas(df)
                # 在线程池中执行同步的写入操作
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    pq.write_table,
                    table,
                    f,
                    compression='snappy'
                )
            
            logger.info(f"成功将 {len(df)} 行数据保存到: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存Parquet文件时出错: {e}", exc_info=True)
            return None