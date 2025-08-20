#!/usr/bin/env python3
"""
回归功能结构验证脚本
验证回归相关代码的语法和结构是否正确
"""

import os
import sys
import traceback

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from src.model_factory import ModelFactory
        print("✓ model_factory.py 导入成功")
    except Exception as e:
        print(f"✗ model_factory.py 导入失败: {e}")
        return False
    
    try:
        from src.pipeline_training import PipelineTraining
        print("✓ pipeline_training.py 导入成功")
    except Exception as e:
        print(f"✗ pipeline_training.py 导入失败: {e}")
        return False
    
    try:
        from src.model_training import ModelTraining
        print("✓ model_training.py 导入成功")
    except Exception as e:
        print(f"✗ model_training.py 导入失败: {e}")
        return False
    
    try:
        from src.evaluator import ModelEvaluator
        print("✓ evaluator.py 导入成功")
    except Exception as e:
        print(f"✗ evaluator.py 导入失败: {e}")
        return False
    
    try:
        from src.feature_engineering import FeatureEngineeringPipeline
        print("✓ feature_engineering.py 导入成功")
    except Exception as e:
        print(f"✗ feature_engineering.py 导入失败: {e}")
        return False
    
    return True

def test_model_factory():
    """测试模型工厂"""
    print("\n测试模型工厂...")
    
    try:
        from src.model_factory import ModelFactory
        
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
                model_name = type(model).__name__
                print(f"✓ {model_type}: {model_name}")
            except Exception as e:
                print(f"✗ {model_type}: {str(e)}")
                
    except Exception as e:
        print(f"✗ 模型工厂测试失败: {e}")
        traceback.print_exc()

def test_config_files():
    """测试配置文件"""
    print("\n测试配置文件...")
    
    config_files = [
        'config.yaml',
        'config_regression.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} 存在")
            
            # 检查文件内容
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 检查回归相关配置
                if 'regression' in content.lower():
                    print(f"  ✓ {config_file} 包含回归配置")
                else:
                    print(f"  - {config_file} 不包含回归配置")
                    
            except Exception as e:
                print(f"  ✗ 读取 {config_file} 失败: {e}")
        else:
            print(f"✗ {config_file} 不存在")

def test_class_structure():
    """测试类结构"""
    print("\n测试类结构...")
    
    try:
        from src.model_factory import ModelFactory
        
        # 检查回归模型创建方法是否存在
        methods = [
            '_create_ridge_regression',
            '_create_xgboost_regressor',
            '_create_random_forest_regressor',
            '_create_lightgbm_regressor'
        ]
        
        for method in methods:
            if hasattr(ModelFactory, method):
                print(f"✓ ModelFactory.{method} 存在")
            else:
                print(f"✗ ModelFactory.{method} 不存在")
                
    except Exception as e:
        print(f"✗ 类结构测试失败: {e}")

def main():
    """主验证函数"""
    print("=" * 60)
    print("回归功能结构验证")
    print("=" * 60)
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败，停止验证")
        return
    
    # 测试模型工厂
    test_model_factory()
    
    # 测试配置文件
    test_config_files()
    
    # 测试类结构
    test_class_structure()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)

if __name__ == "__main__":
    main()