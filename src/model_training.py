"""
模型训练模块
负责模型训练、评估和超参数优化
支持分类和回归任务
"""
import pandas as pd
import numpy as np
import logging
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

from model_factory import ModelFactory
from feature_engineering import FeatureEngineeringPipeline
from data_loader import DataLoader
from config_manager import UnifiedConfig


class ModelTraining:
    """模型训练类，支持分类和回归任务"""
    
    def __init__(self, config: UnifiedConfig):
        """
        初始化模型训练器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_engineering = FeatureEngineeringPipeline(config)
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """自动确定任务类型"""
        unique_values = len(y.unique())
        if unique_values <= 10 and y.dtype in ['int64', 'int32', 'object', 'category']:
            return 'classification'
        else:
            return 'regression'
    
    def _get_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def _get_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算分类评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train_model(self, model_name: str, data_path: str, feature_cols: List[str], 
                   target_col: str, model_save_path: str = None) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            data_path: 数据文件路径
            feature_cols: 特征列名列表
            target_col: 目标变量列名
            model_save_path: 模型保存路径
            
        Returns:
            训练结果字典
        """
        try:
            start_time = time.time()
            
            # 1. 加载数据
            self.logger.info("步骤1/10: 开始加载数据...")
            all_df = self.data_loader.load_data(data_path)
            self.logger.info(f"数据加载完成，数据形状: {all_df.shape}")
            
            # 2. 数据预处理
            self.logger.info("步骤2/10: 开始数据预处理...")
            all_df = self.data_loader.preprocess_data(all_df)
            self.logger.info(f"数据预处理完成，数据形状: {all_df.shape}")
            
            # 3. 特征类型检测
            categorical_features = self.data_loader.detect_categorical_features(all_df, feature_cols)
            self.logger.info(f"检测到 {len(categorical_features)} 个categorical特征: {categorical_features}")
            
            self.logger.info(f"特征列数量: {len(feature_cols)}")
            self.logger.info(f"目标变量: {target_col}")
            self.logger.info(f"categorical特征: {categorical_features}")
            
            # 4. 分割数据
            self.logger.info("步骤4/10: 开始数据分割...")
            X_train, X_test, y_train, y_test = self._split_data(all_df, feature_cols, target_col)
            self.logger.info(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
            self.logger.info(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # 确定任务类型
            task_type = self._determine_task_type(y_train)
            self.logger.info(f"检测到任务类型: {task_type}")
            
            if task_type == 'classification':
                self.logger.info(f"训练集类别分布: {dict(y_train.value_counts())}")
                self.logger.info(f"测试集类别分布: {dict(y_test.value_counts())}")
            
            # 5. 特征工程
            self.logger.info("步骤5/10: 开始特征工程处理...")
            self.feature_engineering.fit(X_train, y_train)
            X_train_processed = self.feature_engineering.transform(X_train)
            X_test_processed = self.feature_engineering.transform(X_test)
            
            final_features = self.feature_engineering.get_feature_names_out()
            self.logger.info(f"特征工程完成，最终特征数量: {len(final_features)}")
            self.logger.info(f"处理后训练集形状: {X_train_processed.shape}")
            self.logger.info(f"处理后测试集形状: {X_test_processed.shape}")
            
            # 6. 处理样本不平衡（仅分类任务）
            if task_type == 'classification':
                self.logger.info("步骤6/10: 开始处理样本不平衡...")
                X_train_balanced, y_train_balanced = self._handle_imbalance(X_train_processed, y_train)
                self.logger.info(f"处理后训练集形状: X_train_balanced={X_train_balanced.shape}, y_train_balanced={y_train_balanced.shape}")
            else:
                X_train_balanced, y_train_balanced = X_train_processed, y_train
                self.logger.info("回归任务跳过样本不平衡处理")
            
            # 7. 创建模型
            self.logger.info("步骤7/10: 开始创建模型...")
            model, param_grid = ModelFactory.create_model(model_name, self.config.to_dict())
            self.logger.info(f"模型创建完成: {type(model).__name__}")
            
            if param_grid:
                self.logger.info(f"超参数网格: {param_grid}")
            else:
                self.logger.info("使用默认参数")
            
            # 8. 训练模型
            self.logger.info("步骤8/10: 开始模型训练...")
            trained_model = self._train_single_model(model, X_train_balanced, y_train_balanced, param_grid, task_type)
            
            # 9. 模型评估
            self.logger.info("步骤9/10: 开始模型评估...")
            evaluation_results = self._evaluate_model(trained_model, X_test_processed, y_test, task_type)
            
            # 10. 保存模型
            if model_save_path:
                self.logger.info("步骤10/10: 开始保存模型...")
                self._save_model(trained_model, model_save_path, final_features, task_type)
                self.logger.info(f"模型已保存到: {model_save_path}")
            
            training_time = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'task_type': task_type,
                'metrics': evaluation_results,
                'training_time': training_time,
                'feature_count': len(final_features),
                'status': 'success'
            }
            
            self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            self.logger.info(f"评估结果: {evaluation_results}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'model_name': model_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _split_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """分割数据集"""
        X = df[feature_cols]
        y = df[target_col]
        
        test_size = self.config.get('modeling.test_size', 0.2)
        random_state = self.config.get('modeling.random_state', 42)
        
        task_type = self._determine_task_type(y)
        
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        return X_train, X_test, y_train, y_test
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """处理样本不平衡"""
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        if imbalance_method == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        elif imbalance_method == 'random_over':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            return X_resampled, y_resampled
        else:
            return X, y
    
    def _train_single_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                          param_grid: Dict, task_type: str):
        """训练单个模型"""
        use_hyperparameter_tuning = self.config.get('modeling.use_hyperparameter_tuning', False)
        
        if use_hyperparameter_tuning and param_grid:
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring='f1_weighted' if task_type == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            self.logger.info(f"最佳参数: {search.best_params_}")
            self.logger.info(f"最佳分数: {search.best_score_}")
            
            return search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model
    
    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> Dict[str, float]:
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            # 获取预测概率（如果支持）
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except:
                    y_proba = None
            
            return self._get_classification_metrics(y_test, y_pred, y_proba)
        else:
            return self._get_regression_metrics(y_test, y_pred)
    
    def _save_model(self, model, save_path: str, feature_names: List[str], task_type: str):
        """保存模型"""
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'task_type': task_type,
            'feature_engineering': self.feature_engineering
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model_data, save_path)
    
    def batch_train(self, model_configs: List[Dict], data_path: str, 
                   feature_cols: List[str], target_col: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        批量训练多个模型
        
        Args:
            model_configs: 模型配置列表
            data_path: 数据文件路径
            feature_cols: 特征列名列表
            target_col: 目标变量列名
            output_dir: 输出目录
            
        Returns:
            训练结果列表
        """
        results = []
        
        for config in model_configs:
            model_name = config['model_name']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"开始训练模型: {model_name}")
            self.logger.info(f"{'='*60}")
            
            model_save_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            
            result = self.train_model(
                model_name=model_name,
                data_path=data_path,
                feature_cols=feature_cols,
                target_col=target_col,
                model_save_path=model_save_path
            )
            
            result['model_config'] = config
            results.append(result)
        
        return results


if __name__ == "__main__":
    # 示例用法
    config = UnifiedConfig()
    trainer = ModelTraining(config)
    
    # 单个模型训练
    result = trainer.train_model(
        model_name='RIDGE',
        data_path='data.csv',
        feature_cols=['feature1', 'feature2', 'feature3'],
        target_col='target',
        model_save_path='models/ridge_model.pkl'
    )
    
    print(f"训练结果: {result}")