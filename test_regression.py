#!/usr/bin/env python3
"""
回归模型测试脚本
用于验证回归模型功能是否正常
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.unified_config import UnifiedConfig
from src.pipeline_training import PipelineTraining
from src.model_factory import ModelFactory
from src.evaluator import ModelEvaluator

def create_test_data():
    """创建测试回归数据"""
    print("创建测试回归数据...")
    
    # 生成回归数据
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # 创建DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame({'fraud_amount': y})
    
    # 合并数据
    data = pd.concat([X_df, y_df], axis=1)
    
    # 添加一些分类特征
    data['category_1'] = np.random.choice(['A', 'B', 'C'], size=len(data))
    data['category_2'] = np.random.choice(['X', 'Y', 'Z'], size=len(data))
    
    # 添加缺失值
    mask = np.random.random(data.shape) < 0.05
    data[mask] = np.nan
    
    return data

def test_model_factory():
    """测试模型工厂"""
    print("测试模型工厂...")
    
    config = {
        'modeling': {
            'ridge_regression': {'alpha': 1.0},
            'xgboost_regression': {'n_estimators': 100},
            'random_forest_regression': {'n_estimators': 100},
            'lightgbm_regression': {'n_estimators': 100}
        }
    }
    
    regression_models = ['RIDGE', 'XGB_REG', 'RF_REG', 'LGB_REG']
    
    for model_type in regression_models:
        try:
            model, param_grid = ModelFactory.create_model(model_type, config)
            print(f"✓ {model_type}: {type(model).__name__}")
        except Exception as e:
            print(f"✗ {model_type}: {str(e)}")

def test_regression_pipeline():
    """测试回归管道"""
    print("测试回归管道...")
    
    # 创建测试数据
    data = create_test_data()
    
    # 保存测试数据
    test_data_path = 'data/test_regression_data.csv'
    os.makedirs('data', exist_ok=True)
    data.to_csv(test_data_path, index=False)
    print(f"测试数据已保存到: {test_data_path}")
    
    # 加载回归配置
    config = UnifiedConfig('config_regression.yaml')
    
    # 更新配置使用测试数据
    config.set('paths.input_data', test_data_path)
    config.set('modeling.regression_models', ['RIDGE', 'RF_REG'])  # 只测试两个模型
    config.set('modeling.n_iter', 2)  # 减少迭代次数
    config.set('modeling.use_hyperparameter_tuning', False)  # 关闭超参数优化
    
    # 创建训练器
    trainer = PipelineTraining(config)
    
    try:
        # 运行批量训练
        results = trainer.run_all_models()
        
        if results:
            print("✓ 回归管道测试通过")
            for result in results:
                print(f"  - {result['model_name']}: R² = {result.get('r2', 'N/A')}")
        else:
            print("✗ 回归管道测试失败: 无结果返回")
            
    except Exception as e:
        print(f"✗ 回归管道测试失败: {str(e)}")

def test_evaluator():
    """测试评估器"""
    print("测试评估器...")
    
    # 创建测试数据
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 4.9])
    
    evaluator = ModelEvaluator()
    
    try:
        metrics = evaluator._calculate_regression_metrics(y_true, y_pred)
        print("✓ 回归评估器测试通过")
        print(f"  - MSE: {metrics['mse']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - R²: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"✗ 回归评估器测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("回归模型功能测试")
    print("=" * 50)
    
    # 测试模型工厂
    test_model_factory()
    print()
    
    # 测试评估器
    test_evaluator()
    print()
    
    # 测试完整管道
    test_regression_pipeline()
    print()
    
    print("=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()