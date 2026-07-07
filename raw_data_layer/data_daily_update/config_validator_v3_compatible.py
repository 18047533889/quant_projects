#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证器V3 - 兼容现有配置格式
基于实际使用需求设计，适配may_2026_update_config_massive.yaml等现有配置
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ConfigValidatorV3Compatible:
    """配置验证器V3 - 兼容现有配置格式"""
    
    # 必需的主配置项 - 最小化要求
    REQUIRED_MAIN_CONFIG = {
        'system.name': str,
        'system.version': str,
    }
    
    # 默认配置值
    DEFAULT_SYSTEM_CONFIG = {
        'name': '增量数据更新系统',
        'version': '1.0.0',
        'log_level': 'INFO',
        'max_parallel_workers': 8,
        'retry_attempts': 3,
        'mode': 'async'
    }
    
    DEFAULT_OUTPUT_CONFIG = {
        'path': 'data/cleaned',
        'format': 'parquet',
        'compression': 'snappy'
    }
    
    DEFAULT_CURSOR_CONFIG = {
        'storage_path': 'cursors'
    }
    
    DEFAULT_REPORTING_CONFIG = {
        'enabled': True,
        'output_path': 'reports',
        'format': 'markdown'
    }
    
    DEFAULT_QUALITY_RULES_CONFIG = {
        'enabled': True,
        'config_path': 'configs/quality_rules_config.yaml'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置验证器"""
        self.config_path = config_path
        self.config = None
        if config_path:
            self.config = self._load_config(config_path)
    
    def validate(self, config: Optional[Dict] = None) -> ValidationResult:
        """验证配置（实例方法）"""
        if config is None:
            config = self.config
        
        if config is None:
            return ValidationResult(False, ["没有提供配置数据"], [])
        
        return self._validate_config_dict(config)
    
    @staticmethod
    def validate_incremental_config(config_path: str) -> ValidationResult:
        """验证增量更新主配置文件 - 兼容现有格式"""
        errors = []
        warnings = []
        
        try:
            # 检查文件是否存在
            if not os.path.exists(config_path):
                errors.append(f"配置文件不存在: {config_path}")
                return ValidationResult(False, errors, warnings)
            
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                errors.append("配置文件为空或格式错误")
                return ValidationResult(False, errors, warnings)
            
            # 验证必需配置项 - 只检查最基本的
            for key, expected_type in ConfigValidatorV3Compatible.REQUIRED_MAIN_CONFIG.items():
                if not ConfigValidatorV3Compatible._check_nested_key(config, key):
                    errors.append(f"缺少必需配置项: {key}")
                elif not isinstance(ConfigValidatorV3Compatible._get_nested_value(config, key), expected_type):
                    actual_type = type(ConfigValidatorV3Compatible._get_nested_value(config, key)).__name__
                    errors.append(f"配置项 {key} 类型错误，期望 {expected_type.__name__}，实际 {actual_type}")
            
            # 如果有数据源，验证数据源格式 - 兼容现有格式
            if 'data_sources' in config and isinstance(config['data_sources'], list):
                for i, source in enumerate(config['data_sources']):
                    source_errors = ConfigValidatorV3Compatible._validate_data_source_compatible(source, i)
                    errors.extend(source_errors)
            
            return ValidationResult(len(errors) == 0, errors, warnings)
            
        except yaml.YAMLError as e:
            errors.append(f"YAML解析错误: {e}")
            return ValidationResult(False, errors, warnings)
        except Exception as e:
            errors.append(f"配置文件读取错误: {e}")
            return ValidationResult(False, errors, warnings)
    
    @staticmethod
    def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认配置值"""
        # 应用系统配置默认值
        if 'system' not in config:
            config['system'] = {}
        for key, value in ConfigValidatorV3Compatible.DEFAULT_SYSTEM_CONFIG.items():
            if key not in config['system']:
                config['system'][key] = value
        
        # 应用输出配置默认值
        if 'output' not in config:
            config['output'] = {}
        for key, value in ConfigValidatorV3Compatible.DEFAULT_OUTPUT_CONFIG.items():
            if key not in config['output']:
                config['output'][key] = value
        
        # 应用游标配置默认值
        if 'cursor' not in config:
            config['cursor'] = {}
        for key, value in ConfigValidatorV3Compatible.DEFAULT_CURSOR_CONFIG.items():
            if key not in config['cursor']:
                config['cursor'][key] = value
        
        # 应用报告配置默认值
        if 'reporting' not in config:
            config['reporting'] = {}
        for key, value in ConfigValidatorV3Compatible.DEFAULT_REPORTING_CONFIG.items():
            if key not in config['reporting']:
                config['reporting'][key] = value
        
        # 应用质量规则默认值
        if 'quality_rules' not in config:
            config['quality_rules'] = {}
        for key, value in ConfigValidatorV3Compatible.DEFAULT_QUALITY_RULES_CONFIG.items():
            if key not in config['quality_rules']:
                config['quality_rules'][key] = value
        
        return config
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {e}")
    
    def _validate_config_dict(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置字典"""
        errors = []
        warnings = []
        
        # 验证必需配置项
        for key, expected_type in ConfigValidatorV3Compatible.REQUIRED_MAIN_CONFIG.items():
            if not ConfigValidatorV3Compatible._check_nested_key(config, key):
                errors.append(f"缺少必需配置项: {key}")
            elif not isinstance(ConfigValidatorV3Compatible._get_nested_value(config, key), expected_type):
                actual_type = type(ConfigValidatorV3Compatible._get_nested_value(config, key)).__name__
                errors.append(f"配置项 {key} 类型错误，期望 {expected_type.__name__}，实际 {actual_type}")
        
        # 如果有数据源，验证数据源格式
        if 'data_sources' in config and isinstance(config['data_sources'], list):
            for i, source in enumerate(config['data_sources']):
                source_errors = ConfigValidatorV3Compatible._validate_data_source_compatible(source, i)
                errors.extend(source_errors)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def _validate_data_source_compatible(source: Dict[str, Any], index: int) -> List[str]:
        """验证单个数据源配置 - 兼容现有格式"""
        errors = []
        
        if not isinstance(source, dict):
            errors.append(f"数据源 {index} 必须是字典类型")
            return errors
        
        # 必需字段 - 只要求name字段
        required_fields = ['name']
        for field in required_fields:
            if field not in source:
                errors.append(f"数据源 {index} 缺少必需字段: {field}")
        
        # 如果有type字段，验证类型 - 兼容现有配置
        if 'type' in source:
            # 扩展兼容的数据源类型，包含现有配置中使用的类型
            valid_types = [
                'rest_api', 's3', 'local_file', 'database', 
                'massive_sdk', 'massive_rest', 'massive_s3',
                'api', 'file', 'rest_or_sdk'  # 添加更多兼容类型
            ]
            if source['type'] not in valid_types:
                errors.append(f"数据源 {index} 的类型必须是 {valid_types} 之一")
            
            # 验证特定类型配置 - 兼容现有格式，只做基本检查
            source_type = source['type']
            source_name = source.get('name', f'数据源{index}')
            
            # 只对启用的数据源进行详细配置检查
            if source.get('enabled', True):
                if source_type in ['rest_api', 'massive_rest', 'api', 'rest_or_sdk']:
                    # 检查rest_api配置是否存在（兼容多种配置键名）
                    rest_config_found = False
                    for config_key in ['rest_api', 'api', 'massive_rest', 'sdk']:
                        if config_key in source:
                            rest_config_found = True
                            break
                    
                    if not rest_config_found:
                        # 对于现有配置，允许没有详细的rest_api配置
                        warnings = [f"{source_type}数据源 {source_name} 建议提供详细的API配置"]
                        # 不报错，只警告
                
                elif source_type in ['s3', 'massive_s3']:
                    # 检查s3配置是否存在（兼容多种配置键名）
                    s3_config_found = False
                    for config_key in ['s3', 'massive_s3']:
                        if config_key in source:
                            s3_config_found = True
                            break
                    
                    if not s3_config_found:
                        # 对于现有配置，允许没有详细的s3配置
                        warnings = [f"{source_type}数据源 {source_name} 建议提供详细的S3配置"]
                        # 不报错，只警告
        
        return errors
    
    @staticmethod
    def _check_nested_key(config: Dict[str, Any], key_path: str) -> bool:
        """检查嵌套键是否存在"""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        return True
    
    @staticmethod
    def _get_nested_value(config: Dict[str, Any], key_path: str) -> Any:
        """获取嵌套键的值"""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current

# 使用示例和测试函数
def test_config_validation_compatible():
    """测试兼容配置验证功能"""
    import tempfile
    
    # 创建兼容现有格式的测试配置
    test_config = {
        'system': {
            'name': '2026年5月增量数据更新试运行',
            'version': '1.0.0',
        },
        'data_sources': [
            {
                'name': 'tickers_all',
                'type': 'rest_api',
                'enabled': True,
                'rest_api': {
                    'base_url': 'https://api.massive.com',
                    'endpoint': '/v3/reference/tickers',
                    'api_key': '0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO',
                    'params': {
                        'limit': 100
                    }
                }
            },
            {
                'name': 's3_trades',
                'type': 's3',
                'enabled': True,
                's3': {
                    'bucket': 'flatfiles',
                    'prefix': 'us_stocks_sip/trades_v1/',
                    'endpoint': 'https://files.massive.com',
                    'access_key': 'f04c088e-d96f-4944-a048-8e419c0bd524',
                    'secret_key': '0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO'
                }
            }
        ]
    }
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        # 验证配置
        result = ConfigValidatorV3Compatible.validate_incremental_config(temp_path)
        print(f"兼容验证结果: {'通过' if result.is_valid else '失败'}")
        if result.errors:
            print("错误:")
            for error in result.errors:
                print(f"  - {error}")
        if result.warnings:
            print("警告:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        # 测试应用默认值
        if result.is_valid:
            validator = ConfigValidatorV3Compatible(temp_path)
            config_with_defaults = ConfigValidatorV3Compatible.apply_defaults(validator.config)
            print("应用默认值后的配置:")
            print(f"  系统配置: {config_with_defaults['system']}")
            print(f"  输出配置: {config_with_defaults['output']}")
            print(f"  游标配置: {config_with_defaults['cursor']}")
            
    finally:
        # 清理临时文件
        os.unlink(temp_path)

if __name__ == '__main__':
    test_config_validation_compatible()