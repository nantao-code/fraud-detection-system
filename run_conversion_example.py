#!/usr/bin/env python3
"""
模型转换使用示例

这个脚本演示如何使用转换后的评分卡
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import pandas as pd
import numpy as np


def run_example():
    """运行转换示例"""
    
    # 定义文件路径
    model_path = "models/pipeline_test_creditcard_smote_RF.joblib"
    features_path = "features/test_creditcard_smote_RF_final_features.json"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not Path(features_path).exists():
        print(f"❌ 特征文件不存在: {features_path}")
        return False
    
    print("=" * 60)
    print("开始模型转换示例...")
    print("=" * 60)
    
    # 运行简化版评分卡生成
    print("\n1. 运行简化版评分卡生成...")
    os.system(f"python simple_scorecard.py --model {model_path} --features {features_path}")
    
    # 检查生成的文件
    generated_files = [
        "scorecard_config_simple.json",
        "scorecard_simple.py",
        "feature_importance_simple.csv"
    ]
    
    missing_files = [f for f in generated_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    print("\n✓ 所有文件生成成功！")
    
    # 演示评分卡使用
    print("\n2. 演示评分卡使用...")
    
    # 导入生成的评分卡
    from scorecard_simple import CreditScorecard
    
    # 创建评分卡实例
    scorecard = CreditScorecard()
    
    # 生成示例数据
    features = ["V2", "V3", "V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18"]
    
    # 创建几个示例客户
    sample_customers = [
        {
            "V2": 1.5, "V3": -0.8, "V4": 2.1, "V7": -1.2, "V9": 0.5, "V10": -0.3,
            "V11": 1.1, "V12": -0.9, "V14": 0.7, "V16": -0.4, "V17": 0.2, "V18": -1.5
        },
        {
            "V2": -0.5, "V3": 1.2, "V4": -1.1, "V7": 0.8, "V9": -1.5, "V10": 0.9,
            "V11": -0.7, "V12": 1.3, "V14": -0.6, "V16": 0.4, "V17": -1.1, "V18": 0.8
        },
        {
            "V2": 0.1, "V3": 0.2, "V4": -0.1, "V7": 0.3, "V9": -0.2, "V10": 0.1,
            "V11": 0.0, "V12": -0.1, "V14": 0.2, "V16": 0.0, "V17": -0.3, "V18": 0.1
        }
    ]
    
    # 评估每个客户
    print("\n客户信用评估结果:")
    print("-" * 80)
    
    for i, customer in enumerate(sample_customers, 1):
        result = scorecard.predict_proba(customer)
        contributions, total = scorecard.get_feature_scores(customer)
        
        print(f"\n客户 {i}:")
        print(f"  信用分数: {result['score']} 分")
        print(f"  违约概率: {result['probability']:.2%}")
        print(f"  风险等级: {result['risk_level']}")
        
        # 显示主要贡献特征
        top_features = sorted(contributions.items(), key=lambda x: abs(x[1]['score']), reverse=True)[:3]
        print("  主要影响因素:")
        for feat, info in top_features:
            print(f"    {feat}: {info['score']:.1f} 分 (值: {info['value']:.3f})")
    
    # 显示特征重要性
    print("\n3. 特征重要性分析:")
    try:
        importance_df = pd.read_csv("feature_importance_simple.csv")
        print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"读取特征重要性失败: {e}")
    
    print("\n" + "=" * 60)
    print("模型转换演示完成！")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 评分卡文件: scorecard_simple.py")
    print("2. 配置文件: scorecard_config_simple.json")
    print("3. 特征重要性: feature_importance_simple.csv")
    print("\n在生产环境中使用:")
    print("  from scorecard_simple import CreditScorecard")
    print("  scorecard = CreditScorecard()")
    print("  score = scorecard.calculate_score(customer_data)")
    
    return True


if __name__ == "__main__":
    success = run_example()
    if success:
        print("\n✅ 所有步骤执行成功！")
    else:
        print("\n❌ 执行失败，请检查错误信息")
        sys.exit(1)