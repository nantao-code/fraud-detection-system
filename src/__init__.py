"""
易受诈人群识别 - 机器学习训练管道

这是一个完整的机器学习训练管道，包含：
- 配置管理
- 数据加载和验证
- 特征工程
- 模型训练和优化
- 模型评估和比较
"""

__version__ = "2.0.0"
__author__ = "AI Assistant"

from .model_training import TrainingPipeline
from .unified_config import UnifiedConfig
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineeringPipeline
from .model_factory import ModelFactory
from .evaluator import ModelEvaluator

__all__ = [
    'TrainingPipeline',
    'UnifiedConfig', 
    'DataLoader',
    'FeatureEngineeringPipeline',
    'ModelFactory',
    'ModelEvaluator'
]