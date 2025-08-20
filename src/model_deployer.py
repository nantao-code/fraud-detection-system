"""
模型部署模块
包含模型保存、加载、API服务等功能
"""
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


class ModelDeployer:
    """模型部署器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths_config = config.get('paths', {})
        self.deployment_config = config.get('deployment', {})
        
    def save_model(self, model, model_name: str, metrics: Dict[str, float], 
                   feature_names: List[str], model_version: str = None) -> str:
        """
        保存模型及相关信息
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            metrics: 模型评估指标
            feature_names: 特征名称列表
            model_version: 模型版本
            
        Returns:
            模型保存路径
        """
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        model_info = {
            'model': model,
            'model_name': model_name,
            'model_version': model_version,
            'metrics': metrics,
            'feature_names': feature_names,
            'created_at': datetime.now().isoformat(),
            'config': self.config
        }
        
        # 构建模型保存路径
        model_dir = Path(self.paths_config.get('model_output', 'models'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_name}_{model_version}.joblib"
        model_path = model_dir / model_filename
        
        # 保存模型
        joblib.dump(model_info, model_path)
        
        # 保存最新模型引用
        latest_path = model_dir / f"{model_name}_latest.joblib"
        joblib.dump(model_info, latest_path)
        
        logging.info(f"模型已保存: {model_path}")
        return str(model_path)
    
    def load_model(self, model_name: str, version: str = 'latest') -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            模型信息字典
        """
        model_dir = Path(self.paths_config.get('model_output', 'models'))
        
        if version == 'latest':
            model_path = model_dir / f"{model_name}_latest.joblib"
        else:
            model_path = model_dir / f"{model_name}_{version}.joblib"
            
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        model_info = joblib.load(model_path)
        logging.info(f"模型已加载: {model_path}")
        
        return model_info
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出所有已保存的模型
        
        Returns:
            模型信息列表
        """
        model_dir = Path(self.paths_config.get('model_output', 'models'))
        models = []
        
        if model_dir.exists():
            for model_file in model_dir.glob("*.joblib"):
                if not model_file.name.endswith('_latest.joblib'):
                    try:
                        model_info = joblib.load(model_file)
                        models.append({
                            'name': model_info['model_name'],
                            'version': model_info['model_version'],
                            'created_at': model_info['created_at'],
                            'metrics': model_info['metrics'],
                            'file_path': str(model_file)
                        })
                    except Exception as e:
                        logging.warning(f"加载模型文件失败: {model_file}, 错误: {e}")
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def validate_input(self, input_data: pd.DataFrame, expected_features: List[str]) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            expected_features: 期望的特征列表
            
        Returns:
            是否有效
        """
        if input_data.empty:
            logging.error("输入数据为空")
            return False
            
        missing_features = set(expected_features) - set(input_data.columns)
        if missing_features:
            logging.error(f"缺失特征: {missing_features}")
            return False
            
        return True
    
    def predict(self, model_info: Dict[str, Any], input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        模型预测
        
        Args:
            model_info: 模型信息
            input_data: 输入数据
            
        Returns:
            预测结果
        """
        model = model_info['model']
        feature_names = model_info['feature_names']
        
        # 验证输入
        if not self.validate_input(input_data, feature_names):
            raise ValueError("输入数据验证失败")
        
        # 确保特征顺序一致
        X = input_data[feature_names]
        
        # 预测
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'model_name': model_info['model_name'],
            'model_version': model_info['model_version'],
            'timestamp': datetime.now().isoformat(),
            'input_shape': input_data.shape
        }
        
        return results
    
    def batch_predict(self, model_info: Dict[str, Any], input_file: str, 
                     output_file: str = None) -> str:
        """
        批量预测
        
        Args:
            model_info: 模型信息
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            输出文件路径
        """
        # 读取输入数据
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        if input_path.suffix.lower() == '.csv':
            input_data = pd.read_csv(input_file)
        elif input_path.suffix.lower() in ['.xls', '.xlsx']:
            input_data = pd.read_excel(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
        
        # 预测
        results = self.predict(model_info, input_data)
        
        # 构建结果DataFrame
        result_df = input_data.copy()
        result_df['prediction'] = results['predictions']
        result_df['probability'] = results['probabilities']
        result_df['model_name'] = results['model_name']
        result_df['model_version'] = results['model_version']
        result_df['prediction_time'] = results['timestamp']
        
        # 保存结果
        if output_file is None:
            output_path = input_path.parent / f"{input_path.stem}_predictions{input_path.suffix}"
        else:
            output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.csv':
            result_df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in ['.xls', '.xlsx']:
            result_df.to_excel(output_path, index=False)
        
        logging.info(f"批量预测完成，结果已保存: {output_path}")
        return str(output_path)