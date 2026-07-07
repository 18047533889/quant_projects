#!/usr/bin/env python3
"""
异步版本的 IncrementalUpdateMaster - 集成 AsyncUniversalFetcher
集成到 data_daily_update 模块
"""

import asyncio
import sys
import logging
import json
import hashlib
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import yaml
from tqdm.asyncio import tqdm

# 导入核心组件 - 使用相对导入避免ModuleNotFoundError
try:
    from raw_data_layer.data_daily_update.async_universal_fetcher import AsyncUniversalFetcher
    from raw_data_layer.data_daily_update.config_validator_v3_compatible import ConfigValidatorV3Compatible, ValidationResult
    from raw_data_layer.data_daily_update.quality_controller import QualityController
    from raw_data_layer.data_daily_update.async_reporter import AsyncReporter
    from raw_data_layer.data_daily_update.cursor_manager import CursorManager
except ImportError:
    # 如果在data_daily_update目录中运行，使用相对导入
    try:
        from .async_universal_fetcher import AsyncUniversalFetcher
        from .config_validator_v3_compatible import ConfigValidatorV3Compatible, ValidationResult
        from .quality_controller import QualityController
        from .async_reporter import AsyncReporter
        from .cursor_manager import CursorManager
    except ImportError:
        # 如果仍然失败，尝试绝对路径导入
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from raw_data_layer.data_daily_update.async_universal_fetcher import AsyncUniversalFetcher
        from raw_data_layer.data_daily_update.config_validator_v3_compatible import ConfigValidatorV3Compatible, ValidationResult
        from raw_data_layer.data_daily_update.quality_controller import QualityController
        from raw_data_layer.data_daily_update.async_reporter import AsyncReporter
        from raw_data_layer.data_daily_update.cursor_manager import CursorManager

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================
# 智能更新策略配置
# ==========================
UPDATE_STRATEGIES = {
    'high_frequency': {
        'delay_hours': 1,      # 高频数据延迟1小时
        'max_retry': 3,
        'backoff_multiplier': 2,
        'priority': 1
    },
    'daily_rest': {
        'delay_hours': 2,       # 日频数据延迟2小时
        'max_retry': 2,
        'backoff_multiplier': 3,
        'priority': 2
    },
    'fundamentals': {
        'delay_hours': 24,      # 基本面数据延迟24小时
        'max_retry': 1,
        'backoff_multiplier': 4,
        'priority': 3
    },
    'default': {
        'delay_hours': 1,
        'max_retry': 2,
        'backoff_multiplier': 2,
        'priority': 2
    }
}


class AsyncIncrementalUpdateMaster:
    """异步增量更新主控器 - 使用 AsyncUniversalFetcher"""
    
    def __init__(self, config_path: Path, data_root: Optional[Path] = None):
        """
        初始化异步增量更新主控器
        
        Args:
            config_path: 配置文件路径
            data_root: 数据根目录
        """
        self.config_path = config_path
        self.data_root = data_root or Path("../massive_parquet")
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config()
        
        # 异步获取器将在create方法中初始化
        self.fetcher = None
        
        # 创建游标管理器
        self.cursor_manager = CursorManager(self.data_root)
        
        # 创建异步报告器
        report_dir = self.data_root / "reports"
        self.reporter = AsyncReporter(report_dir=report_dir)
        
        # 验证配置
        self._validate_config()
        
        logger.info(f"AsyncIncrementalUpdateMaster initialized with config: {config_path}")
    
    @classmethod
    async def create(cls, config_path: Path, data_root: Optional[Path] = None):
        """
        异步工厂方法 - 正确初始化AsyncUniversalFetcher
        
        Args:
            config_path: 配置文件路径
            data_root: 数据根目录
            
        Returns:
            AsyncIncrementalUpdateMaster实例
        """
        # 创建实例
        instance = cls(config_path, data_root)
        
        # 使用异步上下文管理器初始化fetcher
        instance.fetcher = AsyncUniversalFetcher(instance.data_root)
        
        # 进入异步上下文
        await instance.fetcher.__aenter__()
        
        return instance
    
    async def close(self):
        """
        异步清理方法 - 正确关闭fetcher和其中的aiohttp会话
        
        必须在程序结束时调用，以避免'aiohttp:Unclosed client session'警告
        """
        if self.fetcher is not None:
            try:
                # 退出异步上下文管理器，关闭aiohttp会话
                await self.fetcher.__aexit__(None, None, None)
                logger.info("AsyncIncrementalUpdateMaster 清理完成 - aiohttp会话已关闭")
            except Exception as e:
                logger.warning(f"关闭fetcher时发生错误: {e}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise
    
    def _validate_config(self):
        """验证配置 - 使用兼容的验证器V3"""
        try:
            # 使用兼容的配置验证器，支持所有现有配置格式
            validator = ConfigValidatorV3Compatible()
            result = validator.validate(self.config)
            
            if not result.is_valid:
                logger.error(f"配置验证失败: {result.errors}")
                raise ValueError(f"配置验证失败: {result.errors}")
            
            # 应用默认配置
            self.config = ConfigValidatorV3Compatible.apply_defaults(self.config)
            logger.info("✅ 配置验证通过，已应用默认配置")
            
        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            raise
    
    async def __aenter__(self):
        """异步上下文管理器进入"""
        if self.fetcher:
            await self.fetcher.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.fetcher:
            await self.fetcher.__aexit__(exc_type, exc_val, exc_tb)
    
    async def run_single_source(self, source_name: str, source_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        异步处理单个数据源 - 增强版，包含状态检测和延迟策略
        
        Args:
            source_name: 数据源名称
            source_config: 数据源配置
            **kwargs: 额外参数
            
        Returns:
            处理结果字典
        """
        start_time = datetime.now()
        result = {
            'source': source_name,
            'status': 'pending',
            'records': 0,
            'duration_seconds': 0,
            'error': None,
            'data_file': None,
            'quality_score': 0.0,
            'massive_status_checked': False,
            'delay_applied': False
        }
        
        try:
            logger.info(f"开始异步处理数据源: {source_name}")
            
            # 🆕 新增步骤：检查.ok文件是否存在（智能跳过机制）
            logger.info(f"🔍 检查 {source_name} 的.ok文件...")
            
            # 构建预期的文件路径 - 与新的保存结构一致
            dataset_name = source_config.get('dataset', source_name)
            today_date = datetime.now().strftime('%Y-%m-%d')
            
            # 原始数据的.ok文件路径
            raw_dir = self.data_root / "massive_parquet" / dataset_name
            raw_ok_file_path = raw_dir / f"{today_date}_{dataset_name}.parquet.ok"
            raw_parquet_file_path = raw_dir / f"{today_date}_{dataset_name}.parquet"
            
            # 检查.ok文件是否存在且有效
            if raw_ok_file_path.exists() and raw_parquet_file_path.exists():
                try:
                    ok_content = json.loads(raw_ok_file_path.read_text())
                    
                    # 验证.ok文件内容
                    if (ok_content.get('dataset') == dataset_name and 
                        ok_content.get('partition') == today_date and
                        ok_content.get('rows', 0) > 0):
                        
                        # 检查文件是否太旧（超过7天）
                        updated_at = datetime.fromisoformat(ok_content['updated_at'])
                        # 统一时区处理：确保两者都是offset-naive
                        now = datetime.now()
                        updated_at_naive = updated_at.replace(tzinfo=None) if updated_at.tzinfo else updated_at
                        if now - updated_at_naive <= timedelta(days=7):
                            
                            logger.info(f"✅ 找到有效的已存在数据: {raw_parquet_file_path}")
                            logger.info(f"   记录数: {ok_content['rows']}")
                            logger.info(f"   更新时间: {ok_content['updated_at']}")
                            logger.info(f"   跳过数据获取过程")
                            
                            result['status'] = 'skipped_by_ok'
                            result['records'] = ok_content['rows']
                            result['data_file'] = str(raw_parquet_file_path)
                            result['quality_score'] = 1.0  # 假设已存在的数据质量为1.0
                            
                            # 计算耗时
                            duration = (datetime.now() - start_time).total_seconds()
                            result['duration_seconds'] = duration
                            
                            # 异步报告
                            await self.reporter.generate_report(source_name, 'skipped_by_ok', ok_content['rows'], 1.0, duration)
                            
                            return result
                        else:
                            logger.info(f"⏰ 数据文件太旧，需要重新获取: {updated_at}")
                    else:
                        logger.warning(f"⚠️ .ok文件内容无效，需要重新获取数据")
                        
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"⚠️ .ok文件格式错误: {e}，需要重新获取数据")
            else:
                logger.info(f"ℹ️ 未找到.ok文件或数据文件，需要获取新数据")
            
            # 步骤1: 检查Massive更新状态
            if kwargs.get('check_massive_status', True):
                logger.info(f"🔍 检查 {source_name} 的Massive更新状态...")
                status_checked = await self.fetcher.check_massive_update_status(source_name)
                result['massive_status_checked'] = status_checked
                
                if not status_checked:
                    logger.warning(f"⚠️ {source_name} Massive数据尚未更新，跳过处理")
                    result['status'] = 'massive_not_ready'
                    return result
            
            # 步骤2: 应用延迟策略
            category = source_config.get('category', 'default')
            strategy = UPDATE_STRATEGIES.get(category, UPDATE_STRATEGIES['default'])
            delay_hours = kwargs.get('delay_hours', strategy['delay_hours'])
            
            if delay_hours > 0:
                logger.info(f"⏳ {source_name} 应用延迟策略，等待 {delay_hours} 小时...")
                await asyncio.sleep(delay_hours * 3600)
                result['delay_applied'] = True
            
            # 步骤3: 获取游标时间
            cursor_time = self.cursor_manager.get_cursor(source_name)
            logger.info(f"数据源 {source_name} 游标时间: {cursor_time}")
            
            # 步骤4: 异步获取数据（带重试机制）
            max_retries = kwargs.get('max_retry', strategy['max_retry'])
            backoff_multiplier = kwargs.get('backoff_multiplier', strategy['backoff_multiplier'])
            
            df = await self._fetch_with_retry(
                source_name, source_config, cursor_time, 
                max_retries, backoff_multiplier, **kwargs
            )
            
            if df.empty:
                logger.warning(f"数据源 {source_name} 获取到空数据")
                result['status'] = 'no_data'
                return result
            
            logger.info(f"数据源 {source_name} 获取到 {len(df)} 条记录")
            
            # 数据质量检查
            if hasattr(self, 'quality_checker') and self.quality_checker:
                quality_result = self.quality_checker.check_dataframe(df, source_config)
                
                if not quality_result.passed:
                    logger.error(f"数据源 {source_name} 质量检查失败: {quality_result.errors}")
                    result['status'] = 'quality_failed'
                    result['error'] = str(quality_result.errors)
                    return result
                
                logger.info(f"数据源 {source_name} 质量检查通过，分数: {quality_result.score:.2f}")
            else:
                # 如果没有质量检查器，执行基本的数据验证
                if len(df) == 0:
                    logger.error(f"数据源 {source_name} 数据为空")
                    result['status'] = 'quality_failed'
                    result['error'] = '数据为空'
                    return result
                
                logger.info(f"数据源 {source_name} 基本数据验证通过 (无质量检查器)")
                quality_result = type('QualityResult', (), {'passed': True, 'score': 1.0, 'errors': []})()
            
            # 保存数据
            data_file = self._save_data(df, source_name, source_config)
            result['data_file'] = str(data_file)
            
            # 更新游标
            if 'time_column' in source_config:
                time_col = source_config['time_column']
                # 确保时间列存在
                if time_col in df.columns:
                    # 强制转换为datetime，无效值变为NaT
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    
                    # 过滤掉NaT后寻找最大值
                    max_time = df[time_col].dropna().max()
                    
                    if pd.notna(max_time):
                        self.cursor_manager.update_cursor(source_name, max_time)
                        logger.info(f"数据源 {source_name} 游标更新到: {max_time}")
                    else:
                        logger.warning(f"数据源 {source_name} 的时间列 '{time_col}' 不包含有效的日期，无法更新游标。")
                else:
                    logger.warning(f"数据源 {source_name} 配置了时间列 '{time_col}'，但数据中不存在该列。")
            
            # 计算耗时
            duration = (datetime.now() - start_time).total_seconds()
            
            # 异步报告
            await self.reporter.generate_report(source_name, 'success', len(df), quality_result.score, duration)
            
            # 更新结果
            result['status'] = 'success'
            result['records'] = len(df)
            result['quality_score'] = quality_result.score
            
        except Exception as e:
            logger.error(f"数据源 {source_name} 处理失败: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
            
            # 异步报告错误
            await self.reporter.report_error(source_name, str(e))
        
        # 设置耗时
        result['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        
        return result

    async def _fetch_with_retry(self, source_name: str, source_config: Dict[str, Any], 
                              cursor_time: Optional[datetime], max_retries: int, 
                              backoff_multiplier: int, **kwargs) -> pd.DataFrame:
        """
        带重试机制的数据获取
        
        Args:
            source_name: 数据源名称
            source_config: 数据源配置
            cursor_time: 游标时间
            max_retries: 最大重试次数
            backoff_multiplier: 退避倍数
            
        Returns:
            DataFrame: 获取到的数据
        """
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"尝试获取 {source_name} 数据 (第 {attempt + 1}/{max_retries + 1} 次)")
                
                # 异步获取数据
                df = await self.fetcher.fetch(source_config, cursor_time=cursor_time, **kwargs)
                
                if df.empty:
                    logger.warning(f"{source_name} 获取到空数据")
                else:
                    logger.info(f"{source_name} 成功获取 {len(df)} 条记录")
                
                return df
                
            except Exception as e:
                if attempt < max_retries:
                    # 计算退避时间（指数退避）
                    backoff_seconds = (backoff_multiplier ** attempt) * 60  # 以分钟为单位
                    logger.warning(f"{source_name} 获取失败: {e}，等待 {backoff_seconds} 秒后重试...")
                    await asyncio.sleep(backoff_seconds)
                else:
                    logger.error(f"{source_name} 获取失败，已达到最大重试次数 {max_retries}")
                    raise  # 最后一次尝试失败，抛出异常
        
        return pd.DataFrame()  # 所有重试都失败，返回空DataFrame

    def _save_data(self, df: pd.DataFrame, source_name: str, source_config: Dict[str, Any]) -> Path:
        """
        保存原始数据和清洗后的数据，严格遵循 raw_data_fetching 的目录结构和分区策略。
        - 目录结构: {root}/{category}/{sub}/{prefix}_{partition}.parquet
        - 分区模式: all, fiscal, calendar_year, date (year/month)
        - 写入方式: 原子写入（临时文件 + 重命名 + fsync），并对分区文件进行合并更新。
        """
        # 1. 从配置中解析目录和分区元数据
        category = source_config.get("category", "uncategorized")
        sub = source_config.get("sub", source_name)
        prefix = source_config.get("p", source_name)
        mode = source_config.get("mode", "all")
        time_col = source_config.get("time_column")

        # 2. 确定分区键 (partition key)
        partition_key = ""
        if mode != "all" and time_col and time_col in df.columns:
            # 确保时间列是 datetime 类型
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            latest_time = df[time_col].max()
            
            if pd.notna(latest_time):
                if mode in ["fiscal", "calendar_year"]:
                    partition_key = str(latest_time.year)
                elif mode == "date":
                    # 默认按年分区，除非配置了更精细的频率
                    partition_freq = source_config.get("partition_freq", "year")
                    if partition_freq == "month":
                        partition_key = latest_time.strftime('%Y-%m')
                    else:
                        partition_key = str(latest_time.year)

        # 3. 构建文件名和路径
        filename = f"{prefix}_{partition_key}.parquet" if partition_key else f"{prefix}.parquet"
        
        raw_dir = self.data_root / 'raw' / category / sub
        cleaned_dir = self.data_root / 'cleaned' / category / sub
        raw_dir.mkdir(parents=True, exist_ok=True)
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        raw_file_path = raw_dir / filename
        cleaned_file_path = cleaned_dir / filename

        # 4. 清洗数据
        cleaned_df = self._clean_data(df.copy(), source_config)

        # 5. 对分区文件进行合并更新，然后执行原子写入
        for path, data in [(raw_file_path, df), (cleaned_file_path, cleaned_df)]:
            if not data.empty:
                # 如果是分区模式且文件已存在，则合并
                if partition_key and path.exists():
                    try:
                        existing_df = pd.read_parquet(path)
                        # 合并并去重
                        combined_df = pd.concat([existing_df, data]).drop_duplicates(keep='last')
                        final_df = combined_df
                        logger.info(f"合并更新: {path} (原有 {len(existing_df)}, 新增 {len(data)}, 合并后 {len(final_df)})")
                    except Exception as e:
                        logger.warning(f"读取现有Parquet文件失败: {path}, 将覆盖写入. 错误: {e}")
                        final_df = data
                else:
                    final_df = data

                # 原子写入逻辑
                temp_dir = path.parent
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix='.tmp') as tf:
                        temp_file_path = Path(tf.name)
                        # 使用文件句柄写入，以支持 fsync
                        with open(temp_file_path, 'wb') as f:
                            final_df.to_parquet(f, engine='pyarrow', index=False)
                            f.flush()
                            os.fsync(f.fileno())
                    
                    # 重命名临时文件为最终文件
                    os.rename(temp_file_path, path)
                    logger.info(f"✅ 原子写入成功: {path}")

                except Exception as e:
                    logger.error(f"原子写入失败: {path}, 错误: {e}", exc_info=True)
                    # 如果失败，尝试清理临时文件
                    if temp_file_path and temp_file_path.exists():
                        temp_file_path.unlink()
                    raise
        
        return raw_file_path

    def _clean_data(self, df: pd.DataFrame, source_config: Dict[str, Any]) -> pd.DataFrame:
        """
        数据清洗的占位方法。
        目前只做一个简单的复制，未来可以扩展。

        Args:
            df: 原始DataFrame
            source_config: 数据源配置

        Returns:
            清洗后的DataFrame
        """
        logger.info(f"正在对数据源 {source_config.get('name', 'unknown')} 进行数据清洗...")
        # 在这里，可以根据 source_config 添加复杂的清洗逻辑
        # 例如：类型转换、空值处理、异常值检测等
        
        # 示例：确保时间列是datetime类型
        time_cols = source_config.get('time_columns', [])
        for col in time_cols:
            if col in df.columns:
                try:
                    # 尝试将列转换为datetime，对于无法转换的行保持原样
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"  - 已将列 '{col}' 转换为datetime格式")
                except Exception as e:
                    logger.warning(f"  - 转换列 '{col}' 为datetime时出错: {e}")

        logger.info("✅ 数据清洗完成 (当前为基本实现)。")
        return df

    async def run_all_sources(self, batch_size: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        异步处理所有数据源，采用分批策略以控制并发量。
        
        Args:
            batch_size: 每批处理的数据源数量。
            **kwargs: 额外参数。
            
        Returns:
            所有数据源的处理结果列表。
        """
        logger.info(f"--- 开始异步处理所有数据源 (批处理大小: {batch_size}) ---")
        
        data_sources = self.config.get('data_sources', [])
        if not data_sources:
            sources_config = self.config.get('sources', {})
            data_sources = [{'name': name, **config} for name, config in sources_config.items() if isinstance(config, dict)]
        
        if not data_sources:
            logger.warning("配置中未找到 'data_sources'。")
            return []
            
        enabled_sources = [s for s in data_sources if s.get('enabled', True)]
        total_sources = len(enabled_sources)
        if not total_sources:
            logger.warning("所有数据源均被禁用。")
            return []

        logger.info(f"共找到 {total_sources} 个启用的数据源。将分批处理。")
        all_results = []
        
        # 创建批次
        batches = [enabled_sources[i:i + batch_size] for i in range(0, total_sources, batch_size)]
        
        # 使用 tqdm 包装批次迭代
        for i, batch in enumerate(tqdm(batches, desc="Overall Batch Progress", unit="batch")):
            batch_num = i + 1
            source_names_in_batch = [s['name'] for s in batch]
            logger.info(f"--- 开始处理批次 {batch_num}/{len(batches)} ---")
            logger.info(f"批次 {batch_num} 包含的数据源: {', '.join(source_names_in_batch)}")
            
            tasks = [self.run_single_source(s['name'], s, **kwargs) for s in batch]
            
            # 并发执行当前批次的任务
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(batch_results)
                logger.info(f"--- 完成处理批次 {batch_num}/{len(batches)} ---")
            except Exception as e:
                logger.error(f"处理批次 {batch_num} 时发生严重错误: {e}", exc_info=True)

            # 每处理完一个批次，短暂休眠，让系统有喘息之机
            if i < len(batches) - 1:
                 logger.info("批次处理完成，短暂休眠1秒...")
                 await asyncio.sleep(1)

        # 在所有数据源处理完毕后，统一保存一次游标
        logger.info("--- 所有批次处理完成，开始统一保存游标 ---")
        self.cursor_manager.save_cursors()
        logger.info("--- 游标已统一保存 ---")

        return all_results