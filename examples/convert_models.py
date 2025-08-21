#!/usr/bin/env python3
"""
模型转换命令行工具 - 支持PMML和多种评分卡转换
基于机器学习模型转换PMML.md和机器学习模型转换评分卡模型.md
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.model_converter import ModelConverter
from src.data_loader import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== 模型转换工具 ===")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型转换工具')
    parser.add_argument('--dataset', '-d', default='test_creditcard', 
                       help='数据集名称')
    parser.add_argument('--models', '-m', nargs='+', 
                       default=['LR', 'RF', 'XGB', 'DT'],
                       help='要转换的模型列表')
    parser.add_argument('--type', '-t', choices=['pmml', 'scorecard', 'both'], 
                       default='both',
                       help='转换类型')
    parser.add_argument('--numeric-features', nargs='*', 
                       help='数值特征列表')
    parser.add_argument('--categorical-features', nargs='*', 
                       help='类别特征列表')
    parser.add_argument('--scorecard-type', choices=['auto', 'lr', 'dt', 'gbm'],
                       default='auto',
                       help='评分卡类型（自动或指定）')
    
    args = parser.parse_args()
    
    logger.info("=== 模型转换工具 ===")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"模型: {args.models}")
    logger.info(f"转换类型: {args.type}")
    logger.info(f"评分卡类型: {args.scorecard_type}")
    logger.info("=" * 50)
    
    # 初始化转换器
    converter = ModelConverter(
        models_dir="models",
        output_dir="converted_models"
    )
    
    # 特征分类（支持命令行参数或默认值）
    if args.numeric_features:
        numeric_features = args.numeric_features
    else:
        numeric_features = [f"V{i}" for i in range(1, 29)]
    
    if args.categorical_features:
        categorical_features = args.categorical_features
    else:
        categorical_features = []
    
    # 检查模型支持情况
    logger.info("检查模型支持情况:")
    supported_models = []
    for model_name in args.models:
        try:
            info = converter.get_model_info(model_name, args.dataset)
            if "error" not in info:
                logger.info(f"  ✅ {model_name}: {info['model_type']} - 支持")
                supported_models.append(model_name)
            else:
                logger.error(f"  ❌ {model_name}: {info['error']}")
        except Exception as e:
            logger.error(f"  ❌ {model_name}: 检查失败 - {e}")
    
    if not supported_models:
        logger.error("❌ 没有可用的模型进行转换")
        return
    
    # 加载训练数据用于评分卡
    X_train = None
    y_train = None
    
    if args.type in ['scorecard', 'both']:
        try:
            data_loader = DataLoader()
            train_data = data_loader.load_data(f"{args.dataset}_train.csv")
            
            if train_data is not None:
                target_col = 'Class' if 'Class' in train_data.columns else 'target'
                X_train = train_data.drop(target_col, axis=1)
                y_train = train_data[target_col]
                logger.info("✅ 训练数据加载成功")
            else:
                logger.warning("⚠ 无法加载训练数据，降级为仅PMML转换")
                args.type = 'pmml'
                
        except Exception as e:
            logger.error(f"❌ 加载训练数据失败: {e}")
            args.type = 'pmml'
    
    # 执行转换
    if args.type in ['pmml', 'both']:
        logger.info("开始PMML转换...")
        try:
            pmml_results = converter.batch_convert_pmml(
                supported_models, 
                numeric_features, 
                categorical_features,
                args.dataset
            )
            
            logger.info("PMML转换结果:")
            success_count = 0
            for model_name, result in pmml_results.items():
                if "失败" not in str(result):
                    success_count += 1
                    logger.info(f"  ✅ {model_name}: {result}")
                else:
                    logger.error(f"  ❌ {model_name}: {result}")
            logger.info(f"ℹ PMML转换成功率: {success_count}/{len(supported_models)}")
                
        except Exception as e:
            logger.error(f"❌ PMML转换失败: {e}")
    
    if args.type in ['scorecard', 'both']:
        logger.info("开始评分卡转换...")
        try:
            scorecard_results = converter.batch_convert_scorecards(
                supported_models, X_train, y_train, args.dataset, args.scorecard_type
            )
            
            logger.info("评分卡转换结果:")
            success_count = 0
            for model_name, result in scorecard_results.items():
                if "error" in result:
                    logger.error(f"  ❌ {model_name}: {result['error']}")
                else:
                    success_count += 1
                    logger.info(f"  ✅ {model_name}: {result['scorecard_path']}")
                    logger.info(f"     类型: {result['scorecard_type']}")
                    logger.info(f"     方法: {result['method']}")
            logger.info(f"ℹ 评分卡转换成功率: {success_count}/{len(supported_models)}")
                    
        except Exception as e:
            logger.error(f"❌ 评分卡转换失败: {e}")
    
    logger.info("转换完成！")
    logger.info("生成的文件:")
    pmml_files = list(Path("converted_models/pmml").glob("*.pmml"))
    scorecard_files = list(Path("converted_models/scorecard").glob("*.pkl"))
    
    logger.info(f"  PMML文件: {len(pmml_files)} 个")
    for f in pmml_files:
        logger.info(f"    - {f.name}")
    
    logger.info(f"  评分卡文件: {len(scorecard_files)} 个")
    for f in scorecard_files:
        logger.info(f"    - {f.name}")
    
    # 使用示例
    logger.info("使用示例:")
    logger.info("  1. 应用评分卡:")
    logger.info("     python -c \"from src.model_converter import ModelConverter; c=ModelConverter(); result=c.apply_scorecard('converted_models/scorecard/test_creditcard_LR_lr_scorecard.pkl', pd.read_csv('data/test.csv')); print(result['score'].describe())\"")
    
    logger.info("  2. 批量验证:")
    logger.info("     python -c \"from src.model_converter import ModelConverter; c=ModelConverter(); results=c.validate_all_models(); print(results)\"")

if __name__ == "__main__":
    main()