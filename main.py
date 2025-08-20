"""
易受诈人群识别 - 主程序入口

使用方法:
    python main.py --config config.yaml
    python main.py --predict
    python main.py --help
"""
import sys
import os
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from src.model_training import TrainingPipeline
from src.pipeline_training import PipelineTraining


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='易受诈人群识别 - 机器学习训练管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 根据配置自动选择训练模式
  python main.py --config custom.yaml  # 使用自定义配置文件
  python main.py --predict            # 使用配置文件中的模型进行预测
  python main.py --single-model       # 强制使用单个模型训练
  python main.py --batch-mode         # 强制使用批量训练（训练多个模型）
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--predict', 
        action='store_true',
        help='使用训练好的模型进行预测，具体模型由配置文件决定'
    )
    
    parser.add_argument(
        '--single-model', 
        action='store_true',
        help='强制使用单个模型训练模式'
    )
    
    parser.add_argument(
        '--batch-mode', 
        action='store_true',
        help='强制使用批量训练模式 (训练多个模型)'
    )
    
    args = parser.parse_args()
    
    # 检查参数冲突
    if args.single_model and args.batch_mode:
        parser.error("不能同时使用 --single-model 和 --batch-mode 参数")
    
    try:
        # 初始化训练管道
        pipeline = PipelineTraining(args.config)
        
        # 从配置文件中读取模型设置
        from src.unified_config import UnifiedConfig
        config = UnifiedConfig(args.config)
        
        if args.predict:
            # 从配置文件获取预测相关配置
            model_name = config.get('modeling.model_type', None)
            if not model_name:
                logger.error("错误: 配置文件中未指定modeling.model_type参数")
                return 1
            
            # 从input_data路径获取数据名称
            input_data_path = config.get('paths.input_data', 'data.csv')
            data_name = Path(input_data_path).stem
            
            logger.info(f"正在使用 {model_name} 模型进行预测...")
            results = pipeline.predict()
            
            # 显示预测结果
            logger.info(f"预测完成，共 {len(results)} 条记录")
            logger.info("前10条预测结果:")
            logger.info("\n" + str(results[['prediction', 'probability']].head(10)))
            
            # 保存预测结果
            output_file = f"predictions_{model_name}_{data_name}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存至: {output_file}")
            
        else:
            
            # 从input_data路径获取数据名称
            input_data_path = config.get('paths.input_data', 'data.csv')
            data_name = Path(input_data_path).stem
            model_name = config.get('modeling.model_type', None)
            
            # 决定训练模式
            if args.single_model and not args.batch_mode:
                # 强制单个模型训练
                logger.info(f"开始训练单个模型: {model_name}")
                results = pipeline.run_single_model(model_name)
                
                if results['status'] == 'success':
                    logger.info(f"✅ {model_name} 模型训练成功!")
                    metrics = results['metrics']['test_metrics']
                    logger.info(f"测试集 AUC: {metrics['auc']:.4f}")
                    logger.info(f"测试集 KS: {metrics['ks']:.4f}")
                else:
                    logger.error(f"❌ {model_name} 模型训练失败: {results['error']}")
                    return 1
            elif args.batch_mode and not args.single_model:
                # 强制批量训练
                models_to_train = config.get('modeling.models', [])
                logger.info(f"开始批量训练 {len(models_to_train)} 个模型: {', '.join(models_to_train)}")
                results = pipeline.run_all_models()
                
                # 统计成功/失败的模型
                successful = [r['model'] for r in results if 'error' not in r]
                failed = [r['model'] for r in results if 'error' in r]
                
                logger.info("训练完成!")
                logger.info(f"成功: {len(successful)} 个模型 ({', '.join(successful)})")
                if failed:
                    logger.error(f"失败: {len(failed)} 个模型 ({', '.join(failed)})")
                    return 1
            else:
                # 根据配置文件自动决定
                models_to_train = config.get('modeling.models', [])
                if len(models_to_train) == 1:
                    model_name = models_to_train[0]
                    logger.info(f"开始训练单个模型: {model_name}")
                    results = pipeline.run_single_model(model_name)
                    
                    if results['status'] == 'success':
                        logger.info(f"✅ {model_name} 模型训练成功!")
                        metrics = results['metrics']['test_metrics']
                        logger.info(f"测试集 AUC: {metrics['auc']:.4f}")
                        logger.info(f"测试集 KS: {metrics['ks']:.4f}")
                    else:
                        logger.error(f"❌ {model_name} 模型训练失败: {results['error']}")
                        return 1
                else:
                    logger.info(f"开始批量训练 {len(models_to_train)} 个模型: {', '.join(models_to_train)}")
                    results = pipeline.run_all_models()
                    
                    successful = [r['model'] for r in results if 'error' not in r]
                    failed = [r['model'] for r in results if 'error' in r]
                    
                    logger.info("训练完成!")
                    logger.info(f"成功: {len(successful)} 个模型 ({', '.join(successful)})")
                    if failed:
                        logger.error(f"失败: {len(failed)} 个模型 ({', '.join(failed)})")
                        return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)