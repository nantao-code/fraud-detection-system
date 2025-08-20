#!/usr/bin/env python3
"""
回归模型训练示例
展示如何使用优化后的系统训练回归模型
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_training import ModelTraining
from config_manager import UnifiedConfig
from evaluator import ModelEvaluator


def create_sample_regression_data():
    """创建示例回归数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 生成特征
    age = np.random.normal(45, 15, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    transaction_amount = np.random.exponential(1000, n_samples)
    
    # 生成目标变量（欺诈损失金额）
    fraud_loss = (
        age * 50 + 
        income * 0.001 + 
        credit_score * 10 + 
        transaction_amount * 0.5 +
        np.random.normal(0, 5000, n_samples)
    )
    
    # 确保没有负值
    fraud_loss = np.maximum(fraud_loss, 0)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'transaction_amount': transaction_amount,
        'fraud_loss': fraud_loss
    })
    
    return data


def main():
    """主函数"""
    print("=== 回归模型训练示例 ===")
    
    # 创建示例数据
    print("1. 创建示例回归数据...")
    data = create_sample_regression_data()
    
    # 保存数据
    data_path = "regression_sample_data.csv"
    data.to_csv(data_path, index=False)
    print(f"   数据已保存至: {data_path}")
    print(f"   数据形状: {data.shape}")
    print(f"   目标变量统计:")
    print(data['fraud_loss'].describe())
    
    # 定义特征和目标变量
    feature_cols = ['age', 'income', 'credit_score', 'transaction_amount']
    target_col = 'fraud_loss'
    
    # 初始化配置
    config = UnifiedConfig()
    
    # 初始化训练器
    trainer = ModelTraining(config)
    
    # 定义要训练的回归模型
    regression_models = [
        {'model_name': 'RIDGE'},
        {'model_name': 'RF_REG'},
        {'model_name': 'XGB_REG'},
        {'model_name': 'LGB_REG'}
    ]
    
    # 创建输出目录
    output_dir = "regression_models"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n2. 开始训练回归模型...")
    
    # 批量训练模型
    results = trainer.batch_train(
        model_configs=regression_models,
        data_path=data_path,
        feature_cols=feature_cols,
        target_col=target_col,
        output_dir=output_dir
    )
    
    # 打印训练结果
    print("\n3. 训练结果:")
    for result in results:
        if result['status'] == 'success':
            print(f"   {result['model_name']}: 成功")
            print(f"      任务类型: {result['task_type']}")
            print(f"      特征数量: {result['feature_count']}")
            print(f"      训练时间: {result['training_time']:.2f}秒")
            print(f"      评估指标:")
            for metric, value in result['metrics'].items():
                print(f"        {metric}: {value:.4f}")
        else:
            print(f"   {result['model_name']}: 失败 - {result['error']}")
        print()
    
    # 评估最佳模型
    print("4. 评估最佳模型...")
    
    # 找到最佳模型（按R2排序）
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        best_model = max(successful_results, key=lambda x: x['metrics']['r2'])
        best_model_name = best_model['model_name']
        
        print(f"   最佳模型: {best_model_name} (R2 = {best_model['metrics']['r2']:.4f})")
        
        # 加载模型进行详细评估
        model_path = os.path.join(output_dir, f"{best_model_name}_model.pkl")
        
        # 重新训练并评估（为了获得详细结果）
        detailed_result = trainer.train_model(
            model_name=best_model_name,
            data_path=data_path,
            feature_cols=feature_cols,
            target_col=target_col,
            model_save_path=model_path
        )
        
        if detailed_result['status'] == 'success':
            evaluator = ModelEvaluator()
            
            # 这里可以添加详细的评估和可视化
            print("   详细评估完成")
    
    print("\n=== 回归模型训练示例完成 ===")


if __name__ == "__main__":
    main()