#!/usr/bin/env python3
"""
质量风控控制器
实现所有质量检查规则，包括阻塞性和警告级别
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    """质量检查结果"""
    passed: bool
    score: float
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

class QualityController:
    """质量风控控制器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.blocking_rules = self.config.get('blocking', {})
        self.warning_rules = self.config.get('warning', {})
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """加载质量规则配置"""
        if config_path and config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'blocking': {
                    'negative_values': {
                        'enabled': True,
                        'columns': ['open', 'high', 'low', 'close', 'volume'],
                        'description': '价格相关字段不允许负值'
                    },
                    'primary_key_uniqueness': {
                        'enabled': True,
                        'description': '主键必须100%唯一'
                    },
                    'file_integrity': {
                        'enabled': True,
                        'description': '文件完整性检查'
                    }
                },
                'warning': {
                    'missing_rate': {
                        'enabled': True,
                        'threshold': 0.01,  # 1%
                        'columns': ['ticker', 'align_time'],
                        'description': '关键字段缺失率检查'
                    },
                    'volatility_anomaly': {
                        'enabled': True,
                        'threshold': 5.0,  # 5σ
                        'columns': ['volume', 'close'],
                        'description': '数据波动异常检查'
                    }
                }
            }
    
    def check_quality(self, data: pd.DataFrame, source_config: Dict) -> QualityCheckResult:
        """执行质量检查"""
        errors = []
        warnings = []
        details = {}
        
        # 检查数据源是否禁用了质量检查
        if source_config.get('quality_control', {}).get('enabled', True) == False:
            logger.info(f"数据源 {source_config.get('name')} 禁用了质量检查")
            return QualityCheckResult(
                passed=True,
                score=1.0,
                errors=[],
                warnings=[],
                details={'quality_control_disabled': True}
            )
        
        # 1. 阻塞性检查
        blocking_passed = True
        
        if self.blocking_rules.get('negative_values', {}).get('enabled', False):
            result = self._check_negative_values(data, source_config)
            details['negative_values'] = result
            if not result['passed']:
                blocking_passed = False
                errors.extend(result['issues'])
        
        if self.blocking_rules.get('primary_key_uniqueness', {}).get('enabled', False):
            result = self._check_primary_key_uniqueness(data, source_config)
            details['primary_key_uniqueness'] = result
            if not result['passed']:
                blocking_passed = False
                errors.extend(result['issues'])
        
        # 2. 警告级别检查（不阻塞，但记录）
        if self.warning_rules.get('missing_rate', {}).get('enabled', False):
            result = self._check_missing_rate(data, source_config)
            details['missing_rate'] = result
            if not result['passed']:
                warnings.extend(result['issues'])
        
        if self.warning_rules.get('volatility_anomaly', {}).get('enabled', False):
            result = self._check_volatility_anomaly(data, source_config)
            details['volatility_anomaly'] = result
            if not result['passed']:
                warnings.extend(result['issues'])
        
        # 计算综合质量分数
        score = self._calculate_quality_score(details)
        
        return QualityCheckResult(
            passed=blocking_passed,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def _check_negative_values(self, data: pd.DataFrame, source_config: Dict) -> Dict:
        """检查负值"""
        rule_config = self.blocking_rules['negative_values']
        check_columns = rule_config.get('columns', [])
        
        issues = []
        total_negative = 0
        
        for column in check_columns:
            if column in data.columns:
                negative_count = (data[column] < 0).sum()
                if negative_count > 0:
                    total_negative += negative_count
                    issues.append(f"字段 {column} 发现 {negative_count} 个负值")
                    logger.warning(f"负值检查: {column} 有 {negative_count} 个负值")
        
        return {
            'passed': total_negative == 0,
            'issues': issues,
            'total_negative': total_negative,
            'checked_columns': [col for col in check_columns if col in data.columns]
        }
    
    def _check_primary_key_uniqueness(self, data: pd.DataFrame, source_config: Dict) -> Dict:
        """检查主键唯一性"""
        pk_columns = source_config.get('primary_key_columns', [])
        
        if not pk_columns:
            return {
                'passed': True,
                'issues': [],
                'reason': '无配置主键列'
            }
        
        # 检查主键列是否存在
        available_pk_columns = [col for col in pk_columns if col in data.columns]
        
        if not available_pk_columns:
            return {
                'passed': True,
                'issues': [],
                'reason': f'主键列不存在: {pk_columns}'
            }
        
        # 检查重复
        duplicate_count = data.duplicated(subset=available_pk_columns).sum()
        total_rows = len(data)
        
        issues = []
        if duplicate_count > 0:
            duplicate_rate = duplicate_count / total_rows
            issues.append(f"主键重复: {duplicate_count} 条 ({duplicate_rate:.2%})")
            logger.warning(f"主键唯一性检查: 发现 {duplicate_count} 条重复记录")
        
        return {
            'passed': duplicate_count == 0,
            'issues': issues,
            'duplicate_count': duplicate_count,
            'duplicate_rate': duplicate_count / total_rows if total_rows > 0 else 0,
            'total_rows': total_rows,
            'pk_columns': available_pk_columns
        }
    
    def _check_missing_rate(self, data: pd.DataFrame, source_config: Dict) -> Dict:
        """检查缺失率"""
        rule_config = self.warning_rules['missing_rate']
        threshold = rule_config.get('threshold', 0.01)
        check_columns = rule_config.get('columns', [])
        
        issues = []
        total_missing_rate = 0
        
        for column in check_columns:
            if column in data.columns:
                missing_count = data[column].isna().sum()
                missing_rate = missing_count / len(data) if len(data) > 0 else 0
                
                if missing_rate > threshold:
                    issues.append(f"字段 {column} 缺失率 {missing_rate:.2%} (阈值: {threshold:.2%})")
                    logger.warning(f"缺失率检查: {column} 缺失率 {missing_rate:.2%}")
                
                total_missing_rate = max(total_missing_rate, missing_rate)
        
        return {
            'passed': total_missing_rate <= threshold,
            'issues': issues,
            'max_missing_rate': total_missing_rate,
            'threshold': threshold,
            'checked_columns': [col for col in check_columns if col in data.columns]
        }
    
    def _check_volatility_anomaly(self, data: pd.DataFrame, source_config: Dict) -> Dict:
        """检查波动异常（简化版，实际需要历史数据）"""
        rule_config = self.warning_rules['volatility_anomaly']
        threshold = rule_config.get('threshold', 5.0)
        check_columns = rule_config.get('columns', [])
        
        issues = []
        
        for column in check_columns:
            if column in data.columns:
                # 计算当前数据的统计特征
                current_mean = data[column].mean()
                current_std = data[column].std()
                
                # 这里应该有历史统计数据，现在用模拟数据
                # 实际实现中需要从cursor_manager获取历史统计
                historical_mean = current_mean * 0.95  # 模拟历史均值
                historical_std = current_std * 1.1    # 模拟历史标准差
                
                if historical_std > 0:
                    z_score = abs(current_mean - historical_mean) / historical_std
                    
                    if z_score > threshold:
                        issues.append(f"字段 {column} 波动异常 (Z-score: {z_score:.2f}, 阈值: {threshold})")
                        logger.warning(f"波动异常检查: {column} Z-score {z_score:.2f}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'threshold': threshold,
            'checked_columns': [col for col in check_columns if col in data.columns]
        }
    
    def _calculate_quality_score(self, details: Dict) -> float:
        """计算质量分数"""
        total_checks = 0
        passed_checks = 0
        
        for check_name, result in details.items():
            total_checks += 1
            if result.get('passed', False):
                passed_checks += 1
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    def update_historical_stats(self, source_name: str, data: pd.DataFrame) -> None:
        """更新历史统计数据（用于波动率检查）"""
        # 这里实现历史统计数据的更新逻辑
        # 实际实现中需要保存到cursor_manager或单独的文件
        logger.info(f"更新 {source_name} 的历史统计数据")