#!/usr/bin/env python3
"""
基本使用示例：展示如何使用新的FeatureSelectorPipeline和FeatureGeneratorPipeline
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 创建示例数据
import pandas as pd
import numpy as np

# 创建模拟数据
def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 创建一些基础特征
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.lognormal(10, 1, 100),
        'credit_score': np.random.normal(650, 50, 100),
        'debt_ratio': np.random.beta(2, 5, 100),
        'employment_length': np.random.randint(0, 30, 100),
        'loan_amount': np.random.lognormal(9, 1, 100),
        'feature_with_missing': np.concatenate([
            np.random.normal(100, 20, 80), 
            [np.nan] * 20  # 添加20%的缺失值
        ]),
        'low_variance_feature': [5.0] * 100,  # 零方差特征
        'correlated_feature1': np.random.normal(100, 15, 100),
        'correlated_feature2': np.random.normal(100, 15, 100) * 0.95  # 高度相关
    }
    
    df = pd.DataFrame(data)
    
    # 创建目标变量
    y = (df['credit_score'] > 650).astype(int)
    
    return df, y

def demonstrate_feature_selector():
    """演示特征选择器"""
    print("=" * 60)
    print("FeatureSelectorPipeline 演示")
    print("=" * 60)
    
    from feature_engineering import FeatureSelectorPipeline
    
    # 创建数据
    X, y = create_sample_data()
    
    print(f"原始数据形状: {X.shape}")
    print(f"特征列表: {list(X.columns)}")
    
    # 创建特征选择器
    selector = FeatureSelectorPipeline(
        task_type='classification',
        selection_methods=['missing_rate', 'variance', 'correlation', 'importance'],
        missing_rate_threshold=0.1,  # 移除缺失率>10%的特征
        variance_threshold=0.01,       # 移除方差<0.01的特征
        correlation_threshold=0.9,     # 移除相关性>0.9的特征
        importance_threshold=0.01      # 移除重要性<0.01的特征
    )
    
    # 训练选择器
    selector.fit(X, y)
    
    # 应用选择
    X_selected = selector.transform(X)
    
    print(f"\n选择后数据形状: {X_selected.shape}")
    print(f"保留的特征: {list(X_selected.columns)}")
    
    # 获取移除的特征信息
    removed_features = selector.get_removed_features()
    print(f"\n移除的特征:")
    for method, features in removed_features.items():
        if features:
            print(f"  {method}: {features}")
    
    # 获取摘要
    summary = selector.get_feature_selection_summary()
    print(f"\n特征选择摘要:")
    print(summary.head(10))
    
    return selector

def demonstrate_feature_generator():
    """演示特征生成器"""
    print("\n" + "=" * 60)
    print("FeatureGeneratorPipeline 演示")
    print("=" * 60)
    
    from feature_engineering import FeatureGeneratorPipeline
    
    # 创建数据
    X, y = create_sample_data()
    
    # 选择数值特征用于演示
    numeric_features = ['age', 'income', 'credit_score', 'debt_ratio']
    X_numeric = X[numeric_features]
    
    print(f"原始数值特征: {list(X_numeric.columns)}")
    print(f"原始数据形状: {X_numeric.shape}")
    
    # 创建特征生成器
    generator = FeatureGeneratorPipeline(
        generate_polynomial=True,
        polynomial_degree=2,
        generate_interaction=True,
        generate_statistical=True,
        generate_binning=True,
        bins_config={'age': 3, 'income': 4, 'credit_score': 3}
    )
    
    # 生成新特征
    X_generated = generator.transform(X_numeric)
    
    print(f"\n生成后数据形状: {X_generated.shape}")
    
    # 获取生成的特征信息
    generated_features = generator.get_generated_features()
    print(f"\n生成的特征数量: {len(generated_features)}")
    print(f"前10个生成的特征: {generated_features[:10]}")
    
    # 获取特征类型映射
    feature_types = generator.get_feature_types()
    print(f"\n特征类型分布:")
    type_counts = {}
    for feature, ftype in feature_types.items():
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    for ftype, count in type_counts.items():
        print(f"  {ftype}: {count}")
    
    # 获取摘要
    summary = generator.get_generation_summary()
    print(f"\n特征生成摘要:")
    print(summary.head(10))
    
    return generator

def demonstrate_combined_workflow():
    """演示组合工作流程"""
    print("\n" + "=" * 60)
    print("组合工作流程演示")
    print("=" * 60)
    
    from feature_engineering import FeatureSelectorPipeline, FeatureGeneratorPipeline
    
    # 创建数据
    X, y = create_sample_data()
    
    print(f"原始数据形状: {X.shape}")
    
    # 步骤1: 特征生成
    print("\n步骤1: 特征生成...")
    generator = FeatureGeneratorPipeline(
        generate_polynomial=True,
        polynomial_degree=2,
        generate_interaction=True,
        generate_statistical=True,
        generate_binning=True,
        bins_config={'age': 3, 'income': 3}
    )
    
    X_generated = generator.transform(X)
    print(f"生成后数据形状: {X_generated.shape}")
    
    # 步骤2: 特征选择
    print("\n步骤2: 特征选择...")
    selector = FeatureSelectorPipeline(
        task_type='classification',
        selection_methods=['missing_rate', 'variance', 'correlation', 'importance'],
        missing_rate_threshold=0.1,
        variance_threshold=0.01,
        importance_threshold=0.01
    )
    
    selector.fit(X_generated, y)
    X_final = selector.transform(X_generated)
    
    print(f"最终数据形状: {X_final.shape}")
    
    # 计算特征变化
    original_count = X.shape[1]
    generated_count = len(generator.get_generated_features())
    final_count = X_final.shape[1]
    
    print(f"\n特征变化总结:")
    print(f"  原始特征: {original_count}")
    print(f"  生成特征: {generated_count}")
    print(f"  最终选择: {final_count}")
    print(f"  特征减少率: {((original_count + generated_count - final_count) / (original_count + generated_count) * 100):.1f}%")
    
    return generator, selector

def demonstrate_sklearn_integration():
    """演示与sklearn的集成"""
    print("\n" + "=" * 60)
    print("sklearn集成演示")
    print("=" * 60)
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        from feature_engineering import FeatureSelectorPipeline, FeatureGeneratorPipeline
        
        # 创建数据
        X, y = create_sample_data()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        # 创建完整的pipeline
        full_pipeline = Pipeline([
            ('feature_generator', FeatureGeneratorPipeline(
                generate_polynomial=True,
                polynomial_degree=2,
                generate_interaction=True,
                generate_statistical=True,
                generate_binning=True,
                bins_config={'age': 3, 'income': 3}
            )),
            ('feature_selector', FeatureSelectorPipeline(
                task_type='classification',
                selection_methods=['missing_rate', 'variance', 'importance'],
                importance_threshold=0.01
            )),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # 训练和评估
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型准确率: {accuracy:.4f}")
        
        # 获取pipeline中的组件
        generator = full_pipeline.named_steps['feature_generator']
        selector = full_pipeline.named_steps['feature_selector']
        
        print(f"\n特征工程效果:")
        print(f"  生成特征数量: {len(generator.get_generated_features())}")
        print(f"  最终选择特征数量: {len(selector.get_selected_features())}")
        
    except ImportError as e:
        print(f"sklearn不可用: {e}")
        print("跳过sklearn集成演示")

if __name__ == "__main__":
    print("开始演示新的特征工程Pipeline...")
    
    try:
        # 运行各个演示
        selector = demonstrate_feature_selector()
        generator = demonstrate_feature_generator()
        gen, sel = demonstrate_combined_workflow()
        demonstrate_sklearn_integration()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        # 总结
        print("\n总结:")
        print("✓ FeatureSelectorPipeline: 支持多种特征选择方法")
        print("✓ FeatureGeneratorPipeline: 支持多种特征生成方法") 
        print("✓ 两个类都符合sklearn TransformerMixin规范")
        print("✓ 可以与sklearn Pipeline无缝集成")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()