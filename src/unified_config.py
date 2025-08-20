"""
统一配置管理模块
解决模型传参混乱问题，提供统一的配置解析和参数传递
"""
import yaml
import copy
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class UnifiedConfig:
    """统一配置管理器，简化参数传递流程"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        统一配置管理器
        
        Args:
            config_path: 配置文件路径，可以是单个配置文件或批处理配置文件
        """
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self._config = {}
        
        self.load_and_resolve()
    
    def load_and_resolve(self):
        """加载并解析配置文件"""
        # 加载配置文件
        self._config = self._load_yaml(self.config_path)
        
        # 检查是否存在base_config引用，如果有则合并
        base_config_ref = self._config.get('base_config')
        if base_config_ref:
            base_config_path = self.config_path.parent / base_config_ref
            base_config = self._load_yaml(base_config_path)
            self._config = self._merge_configs(base_config, self._config)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """安全加载YAML文件"""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file) or {}
        except Exception as e:
            logging.warning(f"加载配置文件失败 {path}: {e}")
        return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key == 'base_config':
                continue  # 跳过基础配置引用
            elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    

    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点分路径"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    

    
    def get_paths(self) -> Dict[str, str]:
        """获取路径配置
        
        Returns:
            Dict[str, str]: 路径配置字典，所有值自动转换为字符串
        """
        return {k: str(v) for k, v in self._config.get('paths', {}).items()}
    
    def get_enabled_models(self) -> list:
        """获取启用的模型列表"""
        return self.get('models') or self.get('modeling.models', ['XGB', 'LGB', 'RF', 'LR'])
    

    
    def to_dict(self) -> Dict[str, Any]:
        """返回完整的配置"""
        return copy.deepcopy(self._config)
    
    def validate(self) -> bool:
        """验证配置"""
        required_keys = [
            'data.input_path',
            'data.target_column', 
            'modeling.test_size',
            'modeling.random_state'
        ]
        
        missing = [k for k in required_keys if not self.get(k)]
        if missing:
            print(f"缺少必要配置项: {', '.join(missing)}")
            return False
        return True