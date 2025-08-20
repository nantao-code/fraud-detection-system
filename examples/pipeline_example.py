#!/usr/bin/env python3
"""
Pipeline训练示例
展示如何使用新的pipeline系统进行模型训练
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_training import PipelineTraining
import yaml

def main():
    """
    演示如何使用PipelineTraining类进行模型训练
    """
    
    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以适应示例
    config['paths']['input_data'] = 'data/test_creditcard.csv'
    config['modeling']['test_size'] = 0.3
    config['modeling']['random_state'] = 42
    config['modeling']['handle_imbalance'] = True
    config['modeling']['imbalance_method'] = 'smote'
    
    # 创建输出目录
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('examples').mkdir(exist_ok=True)
    
    # 测试不同模型类型
    models_to_test = ['LR', 'RF', 'XGB', 'LGB']
    results = {}
    
    print("开始Pipeline训练示例...")
    print("=" * 60)
    
    for model_type in models_to_test:
        print(f"\n训练 {model_type} 模型...")
        print("-" * 40)
        
        try:
            # 更新模型类型
            config['modeling']['model_type'] = model_type
            
            # 创建训练器
            trainer = PipelineTraining(config)
            
            # 运行训练
            result = trainer.run_training()
            
            # 保存结果
            results[model_type] = result
            
            print(f"✓ 训练完成")
            print(f"  训练时间: {result['training_time']:.2f}秒")
            print(f"  测试集AUC: {result['metrics']['test_auc']:.4f}")
            print(f"  特征数量: {result['n_features']}")
            print(f"  模型文件: {os.path.basename(result['model_path'])}")
            
        except Exception as e:
            print(f"✗ 训练失败: {str(e)}")
            continue
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("训练结果汇总")
    print("=" * 60)
    
    for model_type, result in results.items():
        if result:
            print(f"{model_type:4s} | AUC: {result['metrics']['test_auc']:.4f} | "
                  f"训练时间: {result['training_time']:.1f}s | "
                  f"特征: {result['n_features']}")
    
    # 保存结果到文件
    results_file = 'examples/pipeline_results.json'
    import json
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {results_file}")
    
    # 演示如何加载和使用训练好的模型
    if results:
        print("\n" + "=" * 60)
        print("模型使用示例")
        print("=" * 60)
        
        # 选择最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['test_auc'])
        model_type, best_result = best_model
        
        print(f"最佳模型: {model_type}")
        print(f"模型路径: {best_result['model_path']}")
        
        # 加载模型进行预测
        from joblib import load
        
        pipeline = load(best_result['model_path'])
        print(f"Pipeline步骤: {list(pipeline.named_steps.keys())}")
        
        # 加载测试数据
        try:
            data = pd.read_csv(config['paths']['input_data'])
            X = data.drop(columns=[config['feature_engineering']['target_column']])
            y = data[config['feature_engineering']['target_column']]
            
            # 使用pipeline进行预测
            predictions = pipeline.predict_proba(X)[:, 1]
            
            print(f"预测概率范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"预测完成，共预测 {len(predictions)} 个样本")
            
        except Exception as e:
            print(f"预测演示失败: {str(e)}")

if __name__ == '__main__':
    main()