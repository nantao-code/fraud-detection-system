#!/usr/bin/env python3
"""
模型转换示例 - 展示如何使用ModelConverter进行PMML和评分卡转换
基于机器学习模型转换PMML.md和机器学习模型转换评分卡模型.md的完整示例
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.model_converter import ModelConverter

def main():
    """主函数 - 演示模型转换"""
    
    # 初始化转换器
    converter = ModelConverter(
        models_dir="models",
        output_dir="converted_models"
    )
    
    # 示例配置
    dataset_name = "test_creditcard"
    model_names = ["LR", "RF", "XGB", "DT"]
    
    # 示例特征分类（数值型和类别型）
    numeric_features = [f"V{i}" for i in range(1, 29)]  # 数值特征
    categorical_features = ["Gender"]  # 类别特征
    
    print("=== 模型转换演示 ===\n")
    
    # 1. 检查模型信息和支持的转换类型
    print("1. 检查模型信息和支持的转换类型:")
    for model_name in model_names:
        try:
            info = converter.get_model_info(model_name, dataset_name)
            print(f"   {model_name}:")
            print(f"     类型: {info.get('model_type', '未知')}")
            print(f"     支持PMML: {info.get('can_convert_pmml', False)}")
            print(f"     支持评分卡: {info.get('can_convert_scorecard', False)}")
            print(f"     评分卡类型: {info.get('scorecard_types', [])}")
            print()
        except Exception as e:
            print(f"   {model_name}: 错误 - {e}")
    
    print()
    
    # 2. 批量PMML转换（完整Pipeline版本）
    print("2. 批量PMML转换（完整Pipeline版本）:")
    try:
        pmml_results = converter.batch_convert_pmml(
            model_names, 
            numeric_features, 
            categorical_features, 
            dataset_name
        )
        for model_name, result in pmml_results.items():
            print(f"   {model_name}: {result}")
    except Exception as e:
        print(f"   PMML转换失败: {e}")
    
    print()
    
    # 3. 评分卡转换（支持多种模型类型）
    print("3. 评分卡转换（支持多种模型类型）:")
    try:
        # 加载训练数据
        from src.data_loader import DataLoader
        data_loader = DataLoader()
        train_data = data_loader.load_data("test_creditcard_train.csv")
        
        if train_data is not None:
            X_train = train_data.drop('Class', axis=1)
            y_train = train_data['Class']
            
            # 转换不同模型的评分卡
            for model_name in ["LR", "DT", "XGB", "RF"]:
                try:
                    print(f"   {model_name}评分卡:")
                    scorecard_report = converter.convert_to_scorecard(
                        model_name, X_train, y_train, dataset_name
                    )
                    print(f"     类型: {scorecard_report['scorecard_type']}")
                    print(f"     方法: {scorecard_report['method']}")
                    print(f"     路径: {scorecard_report['scorecard_path']}")
                    print()
                except Exception as e:
                    print(f"     转换失败: {e}")
        else:
            print("   无法加载训练数据，跳过评分卡转换")
            
    except Exception as e:
        print(f"   评分卡转换失败: {e}")
    
    print()
    
    # 4. 应用评分卡进行打分
    print("4. 应用评分卡进行打分:")
    try:
        # 加载测试数据
        test_data = data_loader.load_data("test_creditcard_test.csv")
        if test_data is not None:
            X_test = test_data.drop('Class', axis=1)
            
            # 应用LR评分卡
            try:
                lr_scorecard_path = f"converted_models/scorecard/{dataset_name}_LR_lr_scorecard.pkl"
                scored_lr = converter.apply_scorecard(lr_scorecard_path, X_test)
                print(f"   LR评分卡应用完成，样本数: {len(scored_lr)}")
                print(f"   分数统计: {scored_lr['score'].describe()}")
            except Exception as e:
                print(f"   LR评分卡应用失败: {e}")
            
            # 应用DT评分卡
            try:
                dt_scorecard_path = f"converted_models/scorecard/{dataset_name}_DT_dt_scorecard.pkl"
                scored_dt = converter.apply_scorecard(dt_scorecard_path, X_test)
                print(f"   DT评分卡应用完成，样本数: {len(scored_dt)}")
                print(f"   分数统计: {scored_dt['score'].describe()}")
            except Exception as e:
                print(f"   DT评分卡应用失败: {e}")
                
    except Exception as e:
        print(f"   评分卡应用失败: {e}")
    
    print()
    
    # 5. 验证结果
    print("5. 验证结果:")
    pmml_files = list(Path("converted_models/pmml").glob("*.pmml"))
    scorecard_files = list(Path("converted_models/scorecard").glob("*.pkl"))
    
    print(f"   PMML文件: {len(pmml_files)} 个")
    for f in pmml_files:
        print(f"     - {f.name}")
    
    print(f"   评分卡文件: {len(scorecard_files)} 个")
    for f in scorecard_files:
        print(f"     - {f.name}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()