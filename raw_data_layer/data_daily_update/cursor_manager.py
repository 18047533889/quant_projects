#!/usr/bin/env python3
"""
游标管理器
管理每个数据源的最后更新时间，支持历史统计数据
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CursorManager:
    """游标管理器"""
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.cursor_file = data_root / "cursors" / "cursor.json"
        self.historical_stats_file = data_root / "cursors" / "historical_stats.json"
        
        # 确保游标目录存在
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载现有游标
        self.cursors = self._load_cursors()
        self.historical_stats = self._load_historical_stats()
    
    def _load_cursors(self) -> Dict[str, Any]:
        """加载游标数据"""
        if self.cursor_file.exists():
            try:
                with open(self.cursor_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载游标文件失败: {e}")
                return {}
        return {}
    
    def _load_historical_stats(self) -> Dict[str, Any]:
        """加载历史统计数据"""
        if self.historical_stats_file.exists():
            try:
                with open(self.historical_stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载历史统计文件失败: {e}")
                return {}
        return {}
    
    def get_cursor(self, source_name: str) -> datetime:
        """获取数据源的游标时间 - 严格读取我们setup脚本设置的last_update字段"""
        cursor_data = self.cursors.get(source_name, {})
        
        # 优先读取setup_today_cursors.py中设置的last_update字段
        if 'last_update' in cursor_data:
            try:
                return datetime.fromisoformat(cursor_data['last_update'])
            except Exception as e:
                logger.warning(f"解析游标时间失败，使用备用方案: {e}")
                pass
        
        # 如果游标数据里没有last_update字段，才返回默认值，否则完全遵循我们设置的时间
        logger.warning(f"数据源 {source_name} 未找到last_update游标字段，使用默认值")
        return datetime.now() - timedelta(days=1)
    
    def update_cursor(self, source_name: str, new_cursor: datetime) -> None:
        """更新数据源游标"""
        if source_name not in self.cursors:
            self.cursors[source_name] = {}
        
        self.cursors[source_name].update({
            'last_update': new_cursor.isoformat(),
            'last_update_timestamp': datetime.now().isoformat(),
            'update_count': self.cursors[source_name].get('update_count', 0) + 1
        })
        
        # self._save_cursors() # 移除自动保存
        logger.info(f"在内存中更新 {source_name} 游标到 {new_cursor}")
        logger.info(f"【调试】更新后 self.cursors 的内容: {self.cursors}")
    
    def get_historical_stats(self, source_name: str, column: str) -> Optional[Dict[str, float]]:
        """获取历史统计数据"""
        stats = self.historical_stats.get(source_name, {}).get(column, {})
        
        if stats:
            return {
                'mean': stats.get('mean', 0.0),
                'std': stats.get('std', 1.0),
                'min': stats.get('min', 0.0),
                'max': stats.get('max', 0.0),
                'count': stats.get('count', 0)
            }
        
        return None
    
    def update_historical_stats(self, source_name: str, data: pd.DataFrame, 
                              stats_columns: List[str]) -> None:
        """更新历史统计数据"""
        if source_name not in self.historical_stats:
            self.historical_stats[source_name] = {}
        
        for column in stats_columns:
            if column in data.columns:
                col_data = data[column].dropna()
                
                if len(col_data) > 0:
                    current_stats = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': len(col_data),
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # 如果已有历史数据，进行加权平均
                    existing_stats = self.historical_stats[source_name].get(column, {})
                    if existing_stats:
                        old_count = existing_stats.get('count', 0)
                        new_count = current_stats['count']
                        total_count = old_count + new_count
                        
                        if total_count > 0:
                            # 加权平均
                            weighted_mean = (
                                (existing_stats['mean'] * old_count + current_stats['mean'] * new_count) / total_count
                            )
                            weighted_std = (
                                (existing_stats['std'] * old_count + current_stats['std'] * new_count) / total_count
                            )
                            
                            current_stats['mean'] = weighted_mean
                            current_stats['std'] = weighted_std
                            current_stats['count'] = total_count
                    
                    self.historical_stats[source_name][column] = current_stats
        
        self._save_historical_stats()
        logger.info(f"更新 {source_name} 的历史统计数据")
    
    def get_source_summary(self, source_name: str) -> Dict[str, Any]:
        """获取数据源摘要信息"""
        cursor_data = self.cursors.get(source_name, {})
        
        return {
            'source_name': source_name,
            'last_update': cursor_data.get('last_update'),
            'last_update_timestamp': cursor_data.get('last_update_timestamp'),
            'update_count': cursor_data.get('update_count', 0),
            'has_historical_stats': source_name in self.historical_stats
        }
    
    def get_all_source_summaries(self) -> List[Dict[str, Any]]:
        """获取所有数据源的摘要信息"""
        summaries = []
        for source_name in self.cursors.keys():
            summaries.append(self.get_source_summary(source_name))
        return summaries
    
    def save_cursors(self) -> None:
        """保存游标数据"""
        try:
            with open(self.cursor_file, 'w', encoding='utf-8') as f:
                json.dump(self.cursors, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存游标文件失败: {e}")
    
    def _save_historical_stats(self) -> None:
        """保存历史统计数据"""
        try:
            with open(self.historical_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.historical_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史统计文件失败: {e}")
    
    def reset_cursor(self, source_name: str, new_time: Optional[datetime] = None) -> None:
        """重置数据源游标"""
        if new_time is None:
            new_time = datetime.now() - timedelta(days=30)  # 默认30天前
        
        # 修复：无论数据源是否存在于cursors中，都要创建或更新
        if source_name not in self.cursors:
            self.cursors[source_name] = {}
        
        self.cursors[source_name]['last_update_time'] = new_time.isoformat()
        self.cursors[source_name]['last_update_timestamp'] = datetime.now().isoformat()
        self._save_cursors()
        logger.info(f"重置 {source_name} 游标到 {new_time}")
    
    def cleanup_old_cursors(self, days_to_keep: int = 90) -> None:
        """清理旧的游标记录"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        sources_to_remove = []
        for source_name, cursor_data in self.cursors.items():
            last_update = datetime.fromisoformat(cursor_data.get('last_update_timestamp', 
                                                                  datetime.now().isoformat()))
            if last_update < cutoff_date:
                sources_to_remove.append(source_name)
        
        for source_name in sources_to_remove:
            del self.cursors[source_name]
            logger.info(f"清理旧游标: {source_name}")
        
        if sources_to_remove:
            self._save_cursors()