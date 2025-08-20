#!/usr/bin/env python3
"""
回归功能语法检查脚本
验证回归相关代码的语法是否正确
"""

import ast
import os

def check_python_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析语法
        ast.parse(content)
        print(f"✓ {os.path.basename(file_path)} 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"✗ {os.path.basename(file_path)} 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)} 读取错误: {e}")
        return False

def check_regression_keywords(file_path):
    """检查回归相关关键词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        keywords = [
            'regression',
            'ridge',
            'regressor',
            'r2',
            'rmse',
            'mae',
            'mse'
        ]
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in content:
                found_keywords.append(keyword)
        
        if found_keywords:
            print(f"  ✓ 包含回归关键词: {', '.join(found_keywords)}")
        else:
            print(f"  - 未找到回归关键词")
            
    except Exception as e:
        print(f"  ✗ 关键词检查失败: {e}")

def check_model_factory():
    """检查模型工厂中的回归模型"""
    file_path = 'src/model_factory.py'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查回归模型相关代码
        regression_models = [
            'RIDGE',
            'XGB_REG',
            'RF_REG',
            'LGB_REG'
        ]
        
        found_models = []
        for model in regression_models:
            if model in content:
                found_models.append(model)
        
        if found_models:
            print(f"✓ 模型工厂包含回归模型: {', '.join(found_models)}")
        else:
            print(f"✗ 模型工厂未找到回归模型")
            
    except Exception as e:
        print(f"✗ 检查模型工厂失败: {e}")

def check_config_files():
    """检查配置文件"""
    config_files = [
        'config.yaml',
        'config_regression.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} 存在")
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查回归配置
                regression_indicators = [
                    'regression_models',
                    'ridge_regression',
                    'xgboost_regression',
                    'random_forest_regression',
                    'lightgbm_regression'
                ]
                
                found_indicators = []
                for indicator in regression_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                
                if found_indicators:
                    print(f"  ✓ 包含回归配置: {', '.join(found_indicators)}")
                else:
                    print(f"  - 未找到回归配置")
                    
            except Exception as e:
                print(f"  ✗ 读取 {config_file} 失败: {e}")
        else:
            print(f"✗ {config_file} 不存在")

def main():
    """主检查函数"""
    print("=" * 60)
    print("回归功能语法检查")
    print("=" * 60)
    
    # 检查核心Python文件
    core_files = [
        'src/model_factory.py',
        'src/pipeline_training.py',
        'src/model_training.py',
        'src/evaluator.py',
        'src/feature_engineering.py'
    ]
    
    print("\n1. 检查Python文件语法:")
    for file_path in core_files:
        if os.path.exists(file_path):
            check_python_syntax(file_path)
            check_regression_keywords(file_path)
        else:
            print(f"✗ {os.path.basename(file_path)} 不存在")
    
    print("\n2. 检查模型工厂:")
    check_model_factory()
    
    print("\n3. 检查配置文件:")
    check_config_files()
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()