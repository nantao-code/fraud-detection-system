"""
模型工厂模块
负责创建和管理不同类型的机器学习模型
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
import logging


class ModelFactory:
    """模型工厂，负责创建和配置不同类型的模型"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> Tuple[object, Dict[str, Any]]:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
                        分类: 'LR', 'XGB', 'RF', 'LGB'
                        回归: 'RIDGE', 'XGB_REG', 'RF_REG', 'LGB_REG'
            config: 配置字典
            
        Returns:
            (模型实例, 参数网格)
        """
        
        model_type_upper = model_type.upper()

        # 分类模型
        if model_type_upper == 'LR':
            return ModelFactory._create_logistic_regression(config)
        elif model_type_upper == 'XGB':
            return ModelFactory._create_xgboost_classifier(config)
        elif model_type_upper == 'RF':
            return ModelFactory._create_random_forest_classifier(config)
        elif model_type_upper == 'LGB':
            return ModelFactory._create_lightgbm_classifier(config)
        
        # 回归模型
        elif model_type_upper == 'RIDGE':
            return ModelFactory._create_ridge_regression(config)
        elif model_type_upper == 'XGB_REG':
            return ModelFactory._create_xgboost_regressor(config)
        elif model_type_upper == 'RF_REG':
            return ModelFactory._create_random_forest_regressor(config)
        elif model_type_upper == 'LGB_REG':
            return ModelFactory._create_lightgbm_regressor(config)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    # =================================================================
    # 分类模型 (Classification Models)
    # =================================================================

    @staticmethod
    def _create_logistic_regression(config: Dict[str, Any]) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """
        创建逻辑回归模型
        
        参数说明：
        - C: 正则化强度的倒数，值越小正则化越强
        - penalty: 正则化类型，'l1'产生稀疏解，'l2'更稳定
        - solver: 优化算法，'saga'支持L1和L2正则化
        - max_iter: 最大迭代次数
        - tol: 收敛容忍度
        - class_weight: 类别权重，'balanced'自动调整类别权重
        - random_state: 随机种子，确保结果可重现
        """
        # 获取modeling配置
        modeling_config = config.get('modeling', {})
        lr_config = modeling_config.get('logistic_regression', {})
        
        model = LogisticRegression(
            C=lr_config.get('C', 1.0),  # 从配置读取，默认1.0
            penalty=lr_config.get('penalty', 'l2'),  # 从配置读取，默认L2正则化
            solver=lr_config.get('solver', 'saga'),  # 从配置读取，默认saga优化器
            max_iter=lr_config.get('max_iter', 5000),  # 从配置读取，默认5000次
            tol=lr_config.get('tol', 1e-4),  # 从配置读取，默认1e-4
            class_weight=lr_config.get('class_weight', 'balanced'),  # 从配置读取，默认balanced
            random_state=modeling_config.get('random_state', 42)  # 从modeling配置读取随机种子
        )
        
        # 返回不带前缀的纯净参数网格
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化强度范围
            'penalty': ['l1', 'l2'],  # 正则化类型选择
            'max_iter': [5000, 10000],  # 迭代次数范围
            'tol': [1e-4, 1e-3, 1e-2]  # 收敛容忍度范围
        }
        
        return model, param_grid
    
    @staticmethod
    def _create_xgboost_classifier(config: Dict[str, Any]) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """
        创建XGBoost模型 (优化版)
        
        参数说明：
        - n_estimators: 提升树的数量，相当于训练轮数
        - objective: 学习任务类型，'binary:logistic'表示二分类逻辑回归
        - eval_metric: 评估指标，'auc'表示ROC曲线下面积
        - random_state: 随机种子，确保结果可重现
        - tree_method: 树构建方法，'hist'表示直方图算法，适合类别特征
        - enable_categorical: 是否启用内置类别特征支持
        """
        # 获取modeling配置
        modeling_config = config.get('modeling', {})
        
        # 获取XGBoost专用配置（从modeling节中读取）
        xgb_config = modeling_config.get('xgboost', {})
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=xgb_config.get('eval_metric', 'auc'),
            random_state=modeling_config.get('random_state', 42),
            tree_method="hist",
            enable_categorical=True,
            # 从配置读取默认参数，提供后备值
            n_estimators=xgb_config.get('n_estimators', 1000),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            max_depth=xgb_config.get('max_depth', 6)
            # 注意：early_stopping_rounds 和 scale_pos_weight 已移除
        )
        
        # 返回不带'classifier__'前缀的纯净参数网格
        param_grid = {
            'n_estimators': randint(100, 2000),  # 提升树数量范围
            'max_depth': randint(3, 10),  # 树的最大深度，控制模型复杂度
            'learning_rate': uniform(loc=0.01, scale=0.29),  # [0.01, 0.3]
            'subsample': uniform(loc=0.6, scale=0.4),  # [0.6, 1.0]
            'colsample_bytree': uniform(loc=0.6, scale=0.4),  # [0.6, 1.0]
            'reg_alpha': uniform(loc=0, scale=10),  # [0, 10]
            'reg_lambda': uniform(loc=0, scale=10),  # [0, 10]
            'min_child_weight': randint(1, 10),  # 子节点最小权重和
            'gamma': uniform(loc=0, scale=0.5)  # [0, 0.5]
        }
        
        return model, param_grid
    
    @staticmethod
    def _create_random_forest_classifier(config: Dict[str, Any]) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        创建随机森林模型
        
        参数说明：
        - n_estimators: 森林中树的数量，越多越稳定但计算成本越高
        - max_depth: 树的最大深度，控制模型复杂度
        - min_samples_split: 节点分裂所需的最小样本数
        - min_samples_leaf: 叶子节点所需的最小样本数
        - class_weight: 类别权重，'balanced'自动调整类别权重
        - random_state: 随机种子，确保结果可重现
        """
        # 获取modeling配置
        modeling_config = config.get('modeling', {})
        
        # 获取随机森林专用配置（从modeling节中读取）
        rf_config = modeling_config.get('random_forest', {})
        
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),  # 从配置读取，默认100
            max_depth=rf_config.get('max_depth', None),  # 从配置读取，默认None
            min_samples_split=rf_config.get('min_samples_split', 2),  # 从配置读取，默认2
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),  # 从配置读取，默认1
            class_weight=rf_config.get('class_weight', 'balanced'),  # 从配置读取，默认balanced
            random_state=modeling_config.get('random_state', 42)  # 从modeling配置读取随机种子
        )
        
        # 返回不带前缀的纯净参数网格
        param_grid = {
            'n_estimators': randint(50, 500),  # 树数量范围
            'max_depth': randint(3, 15),  # 最大深度范围
            'min_samples_split': randint(2, 20),  # 最小分裂样本数范围
            'min_samples_leaf': randint(1, 10),  # 最小叶子节点样本数范围
            'max_features': ['sqrt', 'log2', None],  # 最大特征数选择
            'bootstrap': [True, False]  # 是否使用bootstrap抽样
        }
        
        return model, param_grid
    
    @staticmethod
    def _create_lightgbm_classifier(config: Dict[str, Any]) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """
        创建LightGBM模型 (优化版)
        
        参数说明：
        - n_estimators: 提升树的数量，相当于训练轮数
        - objective: 学习任务类型，'binary'表示二分类
        - random_state: 随机种子，确保结果可重现
        - class_weight: 类别权重，'balanced'自动调整类别权重
        - max_depth: 树的最大深度，控制模型复杂度
        - num_leaves: 叶子节点数量，控制模型复杂度
        - learning_rate: 学习率，控制每次迭代的更新幅度
        - feature_fraction: 特征抽样比例，防止过拟合
        - bagging_fraction: 数据抽样比例，防止过拟合
        - bagging_freq: 每多少次迭代执行bagging
        - min_child_samples: 叶子节点最小样本数，防止过拟合
        - lambda_l1: L1正则化系数
        - lambda_l2: L2正则化系数
        - min_split_gain: 节点分裂最小增益
        """
        # 获取modeling配置
        modeling_config = config.get('modeling', {})
        
        # 获取LightGBM专用配置（从modeling节中读取）
        lgb_config = modeling_config.get('lightgbm', {})
        
        model = lgb.LGBMClassifier(
            objective='binary',  # 二分类目标函数
            random_state=modeling_config.get('random_state', 42),  # 从modeling配置读取随机种子
            class_weight=lgb_config.get('class_weight', 'balanced'),  # 从配置读取，默认balanced
            # 从配置读取默认参数，提供后备值
            n_estimators=lgb_config.get('n_estimators', 1000),
            learning_rate=lgb_config.get('learning_rate', 0.1),
            max_depth=lgb_config.get('max_depth', -1),  # -1表示不限制
            num_leaves=lgb_config.get('num_leaves', 31),
            feature_fraction=lgb_config.get('feature_fraction', 1.0),
            bagging_fraction=lgb_config.get('bagging_fraction', 1.0),
            bagging_freq=lgb_config.get('bagging_freq', 0),
            min_child_samples=lgb_config.get('min_child_samples', 20),
            lambda_l1=lgb_config.get('lambda_l1', 0.0),
            lambda_l2=lgb_config.get('lambda_l2', 0.0),
            min_split_gain=lgb_config.get('min_split_gain', 0.0)
        )
        
        param_grid = {
            'n_estimators': randint(100, 2000),
            'num_leaves': randint(16, 256),
            'learning_rate': uniform(loc=0.01, scale=0.29),
            'feature_fraction': uniform(loc=0.5, scale=0.5),
            'bagging_fraction': uniform(loc=0.5, scale=0.5),
            'bagging_freq': randint(1, 7),
            'min_child_samples': randint(5, 100),
            'lambda_l1': uniform(loc=0, scale=10),
            'lambda_l2': uniform(loc=0, scale=10),
            'min_split_gain': uniform(loc=0, scale=0.5)
        }
        
        return model, param_grid

    # =================================================================
    # 回归模型 (Regression Models)
    # =================================================================

    @staticmethod
    def _create_ridge_regression(config: Dict[str, Any]) -> Tuple[Ridge, Dict[str, Any]]:
        """
        创建岭回归模型 (回归)
        
        参数说明:
        - alpha: 正则化强度，值越大正则化越强
        - solver: 优化算法
        """
        modeling_config = config.get('modeling', {})
        ridge_config = modeling_config.get('ridge_regression', {})

        model = Ridge(
            alpha=ridge_config.get('alpha', 1.0),
            solver=ridge_config.get('solver', 'auto'),
            random_state=modeling_config.get('random_state', 42)
        )

        param_grid = {
            'alpha': uniform(loc=0.1, scale=10), # [0.1, 10.1]
            'solver': ['svd', 'cholesky', 'lsqr', 'saga']
        }

        return model, param_grid

    @staticmethod
    def _create_xgboost_regressor(config: Dict[str, Any]) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
        """创建XGBoost模型 (回归)"""
        modeling_config = config.get('modeling', {})
        xgb_reg_config = modeling_config.get('xgboost_regressor', {})

        model = xgb.XGBRegressor(
            objective='reg:squarederror', # <-- 回归目标函数
            eval_metric=xgb_reg_config.get('eval_metric', 'rmse'), # <-- 回归评估指标
            random_state=modeling_config.get('random_state', 42),
            tree_method="hist",
            enable_categorical=True,
            n_estimators=xgb_reg_config.get('n_estimators', 1000),
            learning_rate=xgb_reg_config.get('learning_rate', 0.1),
            max_depth=xgb_reg_config.get('max_depth', 6)
        )

        # 参数网格与分类器版本相同
        param_grid = {
            'n_estimators': randint(100, 2000),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(loc=0.01, scale=0.29),
            'subsample': uniform(loc=0.6, scale=0.4),
            'colsample_bytree': uniform(loc=0.6, scale=0.4),
            'reg_alpha': uniform(loc=0, scale=10),
            'reg_lambda': uniform(loc=0, scale=10)
        }
        
        return model, param_grid

    @staticmethod
    def _create_random_forest_regressor(config: Dict[str, Any]) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
        """创建随机森林模型 (回归)"""
        modeling_config = config.get('modeling', {})
        rf_reg_config = modeling_config.get('random_forest_regressor', {})
        
        model = RandomForestRegressor(
            n_estimators=rf_reg_config.get('n_estimators', 100),
            max_depth=rf_reg_config.get('max_depth', None),
            min_samples_split=rf_reg_config.get('min_samples_split', 2),
            min_samples_leaf=rf_reg_config.get('min_samples_leaf', 1),
            random_state=modeling_config.get('random_state', 42)
            # 注意：移除了 class_weight 参数
        )
        
        # 参数网格与分类器版本类似
        param_grid = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 1.0], # 1.0 相当于 None (所有特征)
            'bootstrap': [True, False]
        }
        
        return model, param_grid

    @staticmethod
    def _create_lightgbm_regressor(config: Dict[str, Any]) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
        """创建LightGBM模型 (回归)"""
        modeling_config = config.get('modeling', {})
        lgb_reg_config = modeling_config.get('lightgbm_regressor', {})

        model = lgb.LGBMRegressor(
            objective='regression', # <-- 回归目标函数
            random_state=modeling_config.get('random_state', 42),
            n_estimators=lgb_reg_config.get('n_estimators', 1000),
            learning_rate=lgb_reg_config.get('learning_rate', 0.1),
            max_depth=lgb_reg_config.get('max_depth', -1),
            num_leaves=lgb_reg_config.get('num_leaves', 31),
            feature_fraction=lgb_reg_config.get('feature_fraction', 1.0),
            bagging_fraction=lgb_reg_config.get('bagging_fraction', 1.0),
            bagging_freq=lgb_reg_config.get('bagging_freq', 0),
            min_child_samples=lgb_reg_config.get('min_child_samples', 20),
            lambda_l1=lgb_reg_config.get('lambda_l1', 0.0),
            lambda_l2=lgb_reg_config.get('lambda_l2', 0.0),
            min_split_gain=lgb_reg_config.get('min_split_gain', 0.0)
            # 注意：移除了 class_weight 参数
        )

        # 参数网格与分类器版本相同
        param_grid = {
            'n_estimators': randint(100, 2000),
            'num_leaves': randint(16, 256),
            'learning_rate': uniform(loc=0.01, scale=0.29),
            'feature_fraction': uniform(loc=0.5, scale=0.5),
            'bagging_fraction': uniform(loc=0.5, scale=0.5),
            'bagging_freq': randint(1, 7),
            'min_child_samples': randint(5, 100),
            'lambda_l1': uniform(loc=0, scale=10),
            'lambda_l2': uniform(loc=0, scale=10),
            'min_split_gain': uniform(loc=0, scale=0.5)
        }
        
        return model, param_grid


class ModelValidator:
    """模型验证器"""
    
    @staticmethod
    def validate_model_config(model_type: str, config: Dict[str, Any]) -> bool:
        """验证模型配置"""
        required_keys = ['modeling']
        
        for key in required_keys:
            if key not in config:
                logging.error(f"配置缺少必要键: {key}")
                return False
        
        modeling_config = config.get('modeling', {})
        
        # 验证必要参数
        required_params = ['test_size', 'random_state', 'target_column']
        for param in required_params:
            if param not in modeling_config:
                logging.error(f"模型配置缺少必要参数: {param}")
                return False
        
        return True