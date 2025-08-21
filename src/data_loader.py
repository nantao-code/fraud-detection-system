"""
数据加载模块
负责数据读取、验证和基本预处理
支持本地文件加载
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        Args:
            config: UnifiedConfig实例或配置字典
        """
        self.config = config
    
    def load_data(self, dataset_type: str = 'all') -> pd.DataFrame:
        """
        加载数据集
        
        Args:
            dataset_type: 保留参数，实际统一使用input_data配置
        
        Returns:
            加载的数据集
        """
        file_path = Path(self.config.get('paths.input_data', 'data/creditcard.csv'))
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {file_path}\n"
                f"请确保数据文件存在于指定路径，或修改config.yaml中的paths配置"
            )
        
        return self._load_from_file(file_path, dataset_type)
    
    def _load_from_file(self, file_path: Path, dataset_type: str) -> pd.DataFrame:
        """从本地文件加载数据"""
        logging.info(f"正在加载{dataset_type}数据集: {file_path}")
        
        # 根据文件扩展名选择加载方式
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logging.info(f"成功加载数据，形状: {df.shape}")
        return df
    
    def validate_data(self, df: pd.DataFrame, dataset_type: str = 'train', task_type: str = 'classification') -> bool:
        """
        验证数据完整性
        
        Args:
            df: 要验证的数据集
            dataset_type: 数据集类型
            task_type: 任务类型 ('classification' 或 'regression')
        
        Returns:
            是否通过验证
        """
        logging.info(f"正在验证{dataset_type}数据集...")
        
        # 检查空数据集
        if df.empty:
            logging.error("数据集为空")
            return False
        
        # 检查必需列
        required_columns = self.config.get('feature_engineering.required_columns', [])
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logging.error(f"缺失必需列: {missing_columns}")
                return False
        
        # 检查目标变量
        if dataset_type == 'train':
            target_col = self.get_target_column()
            if target_col not in df.columns:
                logging.error(f"训练集缺少目标变量: {target_col}")
                return False
            
            # 检查目标变量分布和任务类型的一致性
            target_values = df[target_col].dropna()
            unique_values = target_values.nunique()
            
            if task_type == 'classification':
                target_dist = target_values.value_counts()
                if len(target_dist) < 2:
                    logging.error("目标变量类别不足2个")
                    return False
                logging.info(f"目标变量分布:\n{target_dist}")
            else:  # regression
                # 检查回归任务的目标变量是否适合
                if unique_values <= 2:
                    logging.warning(
                        f"警告：回归任务的目标变量只有 {unique_values} 个唯一值 ({sorted(target_values.unique())}). "
                        f"这可能更适合分类任务。请确认任务类型设置是否正确。"
                    )
                
                logging.info(f"回归任务目标变量统计:\n{target_values.describe()}")
        
        # 检查缺失值
        missing_summary = df.isnull().sum()
        if missing_summary.any():
            missing_cols = missing_summary[missing_summary > 0]
            logging.warning(f"存在缺失值的列:\n{missing_cols}")
        
        logging.info("数据验证通过")
        return True
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """获取特征列列表"""
        target_col = self.config.get('feature_engineering.target_column', 'target')
        exclude_cols = self.config.get('feature_engineering.exclude_columns', [])
        
        feature_cols = [col for col in df.columns 
                       if col != target_col and col not in exclude_cols]
        
        return feature_cols
    
    def detect_categorical_features(self, df: pd.DataFrame, feature_cols: list = None) -> list:
        """
        自动检测categorical特征
        
        Args:
            df: 数据集
            feature_cols: 特征列列表，如果为None则使用所有列
            
        Returns:
            categorical特征列表
        """
        if feature_cols is None:
            feature_cols = df.columns.tolist()
        
        categorical_features = []
        
        for col in feature_cols:
            if col in df.columns:
                # 检测是否为object类型或category类型
                if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                    categorical_features.append(col)
                # 检测是否为数值型但唯一值较少（如性别、布尔值等）
                elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
                    categorical_features.append(col)
        
        return categorical_features
    

    
    def get_target_column(self) -> str:
        """获取目标变量列名"""
        return self.config.get('feature_engineering.target_column', 'target')


class DataSplitter:
    """数据分割器"""
    
    def __init__(self, config):
        """
        初始化数据分割器
        
        Args:
            config: UnifiedConfig实例或配置字典，支持点分路径配置访问
        """
        self.config = config
    
    def split_data(self, df: pd.DataFrame, feature_cols: list, target_col: str, task_type: str = 'classification') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割数据集
        
        Args:
            df: 数据集
            feature_cols: 特征列列表
            target_col: 目标变量列名
            task_type: 任务类型 ('classification' 或 'regression')
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        X = df[feature_cols]
        y = df[target_col]
        
        test_size = self.config.get('modeling.test_size', 0.3)
        random_state = self.config.get('modeling.random_state', 42)
        
        # 仅在分类任务中启用分层抽样
        stratify = y if task_type == 'classification' and self.config.get('modeling.stratify', True) else None
        
        logging.info(f"分割数据集，测试集比例: {test_size}, 任务类型: {task_type}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        logging.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test