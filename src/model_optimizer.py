"""
模型优化模块
包含超参数优化、特征选择、模型校准等功能
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
import optuna


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modeling_config = config.get('modeling', {})
        self.optimization_config = config.get('optimization', {})
        
    def optimize_hyperparameters(self, X_train, y_train, model, model_type: str) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            model: 基础模型
            model_type: 模型类型
            
        Returns:
            最优参数
        """
        logging.info(f"开始{model_type}超参数优化...")
        
        # 获取参数空间
        param_space = self._get_param_space(model_type)
        
        # 使用Optuna进行优化
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        def objective(trial):
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            
            model.set_params(**params)
            
            # 交叉验证评估
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        study.optimize(objective, n_trials=self.optimization_config.get('n_trials', 50))
        
        logging.info(f"最优参数: {study.best_params}")
        logging.info(f"最优AUC: {study.best_value}")
        
        return study.best_params
    
    def feature_selection(self, X_train, y_train, X_test, method: str = 'mutual_info', k: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        特征选择
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            method: 选择方法 ('f_classif', 'mutual_info')
            k: 选择的特征数量
            
        Returns:
            选择后的训练集、测试集、选择的特征列表
        """
        logging.info(f"开始特征选择，方法: {method}, 特征数量: {k}")
        
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        X_train_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        logging.info(f"特征选择完成，选择了 {len(selected_features)} 个特征")
        
        return X_train_df, X_test_df, selected_features
    
    def calibrate_model(self, model, X_train, y_train, method: str = 'isotonic') -> CalibratedClassifierCV:
        """
        模型校准
        
        Args:
            model: 待校准的模型
            X_train: 训练特征
            y_train: 训练标签
            method: 校准方法 ('sigmoid', 'isotonic')
            
        Returns:
            校准后的模型
        """
        logging.info(f"开始模型校准，方法: {method}")
        
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv=3
        )
        
        calibrated_model.fit(X_train, y_train)
        
        logging.info("模型校准完成")
        return calibrated_model
    
    def _get_param_space(self, model_type: str) -> Dict[str, Any]:
        """获取参数搜索空间"""
        param_spaces = {
            'XGB': {
                'n_estimators': (100, 1000),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_weight': (1, 10),
                'gamma': (0, 0.5),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1)
            },
            'LGB': {
                'n_estimators': (100, 1000),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 300),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_samples': (5, 100),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1)
            },
            'RF': {
                'n_estimators': (100, 1000),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            }
        }
        
        return param_spaces.get(model_type, {})
    
    def evaluate_optimization(self, model, X_test, y_test) -> Dict[str, float]:
        """
        评估优化效果
        
        Args:
            model: 优化后的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # 计算KS值
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ks_value = max(tpr - fpr)
        
        metrics = {
            'auc': auc_score,
            'ks': ks_value
        }
        
        logging.info(f"优化后模型评估指标: AUC={auc_score:.4f}, KS={ks_value:.4f}")
        
        return metrics