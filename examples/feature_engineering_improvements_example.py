#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程改进功能使用示例

本示例展示了如何使用改进后的特征工程功能：
1. 类别特征编码策略选择（Ordinal vs One-Hot）
2. 基于特征-目标相关性的特征选择（F检验）

运行方式:
python examples/feature_engineering_improvements_example.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineeringPipeline, DataPreprocessor
from src.config import UnifiedConfig


def create_sample_data(task_type='regression', n_samples=1000, n_features=20):
    """创建示例数据，包含数值和类别特征"""
    
    if task_type == 'regression':
        X, y = make_regression(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=10,
            noise=0.1, 
            random_state=42
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
    
    # 创建DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # 添加类别特征
    np.random.seed(42)
    df['category_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
    df['category_2'] = np.random.choice(['X', 'Y', 'Z'], n_samples)
    df['binary_cat'] = np.random.choice([0, 1], n_samples)
    
    return df, pd.Series(y, name='target')


def demonstrate_encoding_strategies():
    """演示不同的类别特征编码策略"""
    
    print("=== 类别特征编码策略演示 ===\n")
    
    # 创建回归数据
    X, y = create_sample_data(task_type='regression', n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. 序号编码（Ordinal Encoding）
    print("1. 序号编码 (Ordinal Encoding)")
    preprocessor_ordinal = DataPreprocessor(encoding_method='ordinal')
    X_train_ordinal = preprocessor_ordinal.fit_transform(X_train)
    X_test_ordinal = preprocessor_ordinal.transform(X_test)
    
    print(f"   原始特征数量: {X_train.shape[1]}")
    print(f"   编码后特征数量: {X_train_ordinal.shape[1]}")
    print(f"   类别特征: {preprocessor_ordinal.categorical_cols_}")
    
    # 训练线性回归模型
    model_ordinal = LinearRegression()
    model_ordinal.fit(X_train_ordinal, y_train)
    pred_ordinal = model_ordinal.predict(X_test_ordinal)
    score_ordinal = mean_squared_error(y_test, pred_ordinal)
    print(f"   线性回归MSE: {score_ordinal:.4f}\n")
    
    # 2. 独热编码（One-Hot Encoding）
    print("2. 独热编码 (One-Hot Encoding)")
    preprocessor_onehot = DataPreprocessor(encoding_method='onehot')
    X_train_onehot = preprocessor_onehot.fit_transform(X_train)
    X_test_onehot = preprocessor_onehot.transform(X_test)
    
    print(f"   原始特征数量: {X_train.shape[1]}")
    print(f"   编码后特征数量: {X_train_onehot.shape[1]}")
    print(f"   类别特征: {preprocessor_onehot.categorical_cols_}")
    
    # 训练线性回归模型
    model_onehot = LinearRegression()
    model_onehot.fit(X_train_onehot, y_train)
    pred_onehot = model_onehot.predict(X_test_onehot)
    score_onehot = mean_squared_error(y_test, pred_onehot)
    print(f"   线性回归MSE: {score_onehot:.4f}\n")


def demonstrate_f_regression_selection():
    """演示基于F检验的特征选择"""
    
    print("=== 基于F检验的特征选择演示 ===\n")
    
    # 创建回归数据
    X, y = create_sample_data(task_type='regression', n_samples=1000, n_features=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 配置特征工程
    config = UnifiedConfig()
    
    # 启用基于F检验的特征选择
    config.set('feature_engineering.use_f_regression_selection', True)
    config.set('feature_engineering.f_regression_k', 15)  # 选择前15个最相关特征
    config.set('feature_engineering.use_importance_selection', False)  # 禁用其他选择方法
    config.set('feature_engineering.use_pca', False)
    
    # 创建并训练特征工程管道
    fe_pipeline = FeatureEngineeringPipeline(config)
    X_train_selected = fe_pipeline.fit_transform(X_train, y_train)
    X_test_selected = fe_pipeline.transform(X_test)
    
    print(f"原始特征数量: {X_train.shape[1]}")
    print(f"F检验选择后特征数量: {X_train_selected.shape[1]}")
    print(f"选择特征: {fe_pipeline.selected_features_}")
    
    # 训练模型并评估
    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    pred = model.predict(X_test_selected)
    score = mean_squared_error(y_test, pred)
    print(f"线性回归MSE: {score:.4f}\n")


def demonstrate_classification_example():
    """演示分类任务中的F检验特征选择"""
    
    print("=== 分类任务中的F检验特征选择演示 ===\n")
    
    # 创建分类数据
    X, y = create_sample_data(task_type='classification', n_samples=1000, n_features=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 配置特征工程
    config = UnifiedConfig()
    
    # 启用基于F检验的特征选择（分类任务会自动使用f_classif）
    config.set('feature_engineering.use_f_regression_selection', True)
    config.set('feature_engineering.f_regression_k', 10)  # 选择前10个最相关特征
    config.set('modeling.task_type', 'classification')
    
    # 创建并训练特征工程管道
    fe_pipeline = FeatureEngineeringPipeline(config)
    X_train_selected = fe_pipeline.fit_transform(X_train, y_train)
    X_test_selected = fe_pipeline.transform(X_test)
    
    print(f"原始特征数量: {X_train.shape[1]}")
    print(f"F检验选择后特征数量: {X_train_selected.shape[1]}")
    print(f"任务类型: {fe_pipeline.task_type_}")
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train_selected, y_train)
    pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, pred)
    print(f"逻辑回归准确率: {accuracy:.4f}\n")


def main():
    """主函数"""
    
    print("特征工程改进功能演示")
    print("=" * 50)
    
    try:
        # 1. 演示编码策略
        demonstrate_encoding_strategies()
        
        # 2. 演示F回归特征选择
        demonstrate_f_regression_selection()
        
        # 3. 演示分类任务
        demonstrate_classification_example()
        
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()