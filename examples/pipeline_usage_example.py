"""
特征工程Pipeline使用示例
展示如何使用FeatureSelectorPipeline和FeatureGeneratorPipeline
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# 导入我们创建的pipeline
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureSelectorPipeline, FeatureGeneratorPipeline, DataPreprocessor


def classification_example():
    """分类任务示例"""
    print("=== 分类任务示例 ===")
    
    # 生成示例数据
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        random_state=42
    )
    
    # 添加缺失值和异常值
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    X.iloc[0:100, 0] = np.nan  # 添加缺失值
    X.iloc[0:50, 1] = 100  # 添加异常值
    
    # 添加类别特征
    X['category_feature'] = np.random.choice(['A', 'B', 'C'], size=len(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建完整的pipeline
    full_pipeline = Pipeline([
        ('preprocessor', DataPreprocessor()),
        ('feature_generator', FeatureGeneratorPipeline(
            generate_polynomial=True,
            polynomial_degree=2,
            generate_interaction=True,
            generate_statistical=True,
            generate_binning=True,
            bins_config={'feature_0': 5, 'feature_1': 3}
        )),
        ('feature_selector', FeatureSelectorPipeline(
            task_type='classification',
            selection_methods=['missing_rate', 'correlation', 'variance', 'mutual_info', 'importance'],
            mutual_info_k=15,
            importance_threshold=0.005
        )),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 训练模型
    full_pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"分类准确率: {accuracy:.4f}")
    
    # 获取特征选择和生成信息
    feature_selector = full_pipeline.named_steps['feature_selector']
    feature_generator = full_pipeline.named_steps['feature_generator']
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"生成特征数量: {len(feature_generator.get_generated_features())}")
    print(f"最终选择特征数量: {len(feature_selector.get_selected_features())}")
    
    # 打印特征选择摘要
    selection_summary = feature_selector.get_feature_selection_summary()
    if not selection_summary.empty:
        print("\n特征选择摘要:")
        print(selection_summary.groupby('method').size())
    
    return full_pipeline


def regression_example():
    """回归任务示例"""
    print("\n=== 回归任务示例 ===")
    
    # 生成示例数据
    X, y = make_regression(
        n_samples=1000, 
        n_features=15, 
        n_informative=8, 
        noise=0.1, 
        random_state=42
    )
    
    # 添加缺失值和异常值
    X = pd.DataFrame(X, columns=[f'reg_feature_{i}' for i in range(15)])
    X.iloc[0:50, 0] = np.nan
    X.iloc[0:30, 1] = 50
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建回归pipeline
    regression_pipeline = Pipeline([
        ('preprocessor', DataPreprocessor()),
        ('feature_generator', FeatureGeneratorPipeline(
            generate_polynomial=True,
            polynomial_degree=2,
            generate_interaction=True,
            generate_statistical=True,
            generate_binning=True,
            bins_config={'reg_feature_0': 4, 'reg_feature_1': 3}
        )),
        ('feature_selector', FeatureSelectorPipeline(
            task_type='regression',
            selection_methods=['missing_rate', 'correlation', 'variance', 'f_regression'],
            f_regression_k=12
        )),
        ('regressor', LinearRegression())
    ])
    
    # 训练模型
    regression_pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = regression_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"回归R²分数: {r2:.4f}")
    
    # 获取特征信息
    feature_selector = regression_pipeline.named_steps['feature_selector']
    feature_generator = regression_pipeline.named_steps['feature_generator']
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"生成特征数量: {len(feature_generator.get_generated_features())}")
    print(f"最终选择特征数量: {len(feature_selector.get_selected_features())}")
    
    return regression_pipeline


def standalone_usage_example():
    """独立使用示例"""
    print("\n=== 独立使用示例 ===")
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    
    # 独立使用特征选择器
    selector = FeatureSelectorPipeline(
        task_type='classification',
        selection_methods=['missing_rate', 'variance', 'mutual_info'],
        mutual_info_k=8
    )
    
    selector.fit(X, y)
    X_selected = selector.transform(X)
    print(f"特征选择后特征数量: {X_selected.shape[1]}")
    
    # 独立使用特征生成器
    generator = FeatureGeneratorPipeline(
        generate_polynomial=True,
        polynomial_degree=2,
        generate_interaction=True,
        generate_statistical=True
    )
    
    X_generated = generator.transform(X)
    print(f"特征生成后特征数量: {X_generated.shape[1]}")
    print(f"生成的特征: {generator.get_generated_features()[:5]}...")  # 显示前5个
    
    # 获取生成摘要
    generation_summary = generator.get_generation_summary()
    if not generation_summary.empty:
        print("\n特征生成摘要:")
        print(generation_summary.groupby('type').size())


if __name__ == "__main__":
    # 运行示例
    classification_pipeline = classification_example()
    regression_pipeline = regression_example()
    standalone_usage_example()
    
    print("\n=== 使用建议 ===")
    print("1. 特征选择Pipeline可以与sklearn的Pipeline无缝集成")
    print("2. 特征生成Pipeline可以生成多种类型的特征")
    print("3. 两个Pipeline都支持获取详细的特征处理信息")
    print("4. 可以根据任务类型选择合适的参数配置")