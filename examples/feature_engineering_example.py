"""
特征工程集成Pipeline示例
展示如何使用集成特征工程的完整训练流程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline_training import PipelineTraining

def main():
    """主函数：展示特征工程集成Pipeline的使用"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*60)
    print("特征工程集成Pipeline示例")
    print("="*60)
    
    try:
        # 1. 初始化Pipeline训练器
        trainer = PipelineTraining("config.yaml")
        
        # 2. 配置特征工程参数
        print("\n1. 配置特征工程参数...")
        # 在config.yaml中已配置：
        # modeling.use_feature_engineering: true
        # feature_engineering.iv_threshold: 0.1
        # feature_engineering.corr_threshold: 0.8
        # feature_engineering.missing_rate_threshold: 0.3
        
        # 3. 运行单个模型训练（带特征工程）
        print("\n2. 运行LR模型训练（带特征工程）...")
        lr_results = trainer.run_single_model("LR")
        
        if lr_results['status'] == 'success':
            print("✅ LR模型训练成功")
            
            # 显示特征选择结果
            eval_results = lr_results['evaluation_results']
            if 'feature_selection' in eval_results:
                selection_info = eval_results['feature_selection']
                print(f"\n特征选择结果:")
                print(f"  原始特征数量: {selection_info['original_features']}")
                print(f"  选择后特征数量: {selection_info['selected_features']}")
                print(f"  特征选择比例: {selection_info['selected_features']/selection_info['original_features']:.2%}")
                
                # 显示模型性能
                metrics = eval_results.get('metrics', {})
                print(f"\n模型性能:")
                print(f"  AUC: {metrics.get('auc', 'N/A')}")
                print(f"  准确率: {metrics.get('accuracy', 'N/A')}")
                print(f"  F1分数: {metrics.get('f1', 'N/A')}")
        
        # 4. 运行RF模型训练（带特征工程）
        print("\n3. 运行RF模型训练（带特征工程）...")
        rf_results = trainer.run_single_model("RF")
        
        if rf_results['status'] == 'success':
            print("✅ RF模型训练成功")
            
        # 5. 运行所有模型
        print("\n4. 运行所有模型...")
        all_results = trainer.run_all_models()
        
        print("\n训练总结:")
        for model_name, results in all_results.items():
            status = "✅" if results['status'] == 'success' else "❌"
            print(f"  {model_name}: {status}")
            
        # 6. 使用训练好的模型进行预测
        print("\n5. 使用训练好的模型进行预测...")
        try:
            predictions = trainer.predict("LR")
            print(f"预测结果形状: {predictions.shape}")
            print("前5条预测结果:")
            print(predictions.head())
        except Exception as e:
            print(f"预测测试跳过: {e}")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()