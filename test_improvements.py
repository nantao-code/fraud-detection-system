#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的特征工程改进功能测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 测试导入
    from src.feature_engineering import FeatureSelector, DataPreprocessor
    
    print("✓ 成功导入改进后的模块")
    
    # 测试新添加的方法是否存在
    if hasattr(FeatureSelector, 'select_by_f_regression'):
        print("✓ FeatureSelector.select_by_f_regression 方法已添加")
    else:
        print("✗ FeatureSelector.select_by_f_regression 方法未找到")
    
    if hasattr(FeatureSelector, 'select_by_f_classif'):
        print("✓ FeatureSelector.select_by_f_classif 方法已添加")
    else:
        print("✗ FeatureSelector.select_by_f_classif 方法未找到")
    
    # 测试DataPreprocessor的encoding_method参数
    try:
        preprocessor = DataPreprocessor(encoding_method='onehot')
        print("✓ DataPreprocessor支持encoding_method参数")
        print(f"  encoding_method: {preprocessor.encoding_method}")
    except Exception as e:
        print(f"✗ DataPreprocessor初始化失败: {e}")
    
    print("\n改进功能验证完成！")
    
except ImportError as e:
    print(f"导入错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")