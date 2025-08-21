"""
易受诈人群识别 - 主程序入口
支持分类与回归双任务

使用方法:
    # 分类任务
    python main.py --config config_classification.yaml
    python main.py --task classification --model RF
    
    # 回归任务
    python main.py --config config_regression.yaml
    python main.py --task regression --model RF_REG
    
    # 预测模式
    python main.py --config config_classification.yaml --predict
    python main.py --predict --task classification
    
    # 批量训练
    python main.py --batch-mode --task classification
    python main.py --batch-mode --task regression
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

from src.pipeline_training import PipelineTraining


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='易受诈人群识别 - 支持分类与回归双任务的机器学习训练管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分类任务
  python main.py --task classification --model RF
  python main.py --config config_classification.yaml
  
  # 回归任务
  python main.py --task regression --model RF_REG
  python main.py --config config_regression.yaml
  
  # 预测模式
  python main.py --config config_classification.yaml --predict
  python main.py --task classification --predict
  
  # 批量训练
  python main.py --batch-mode --task classification
  python main.py --batch-mode --task regression
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='配置文件路径 (可选，将根据任务类型自动选择)'
    )
    
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['classification', 'regression'],
        help='任务类型: classification(分类) 或 regression(回归)'
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        help='指定模型类型，如 RF, XGB, LR, LGB 或 RF_REG, XGB_REG 等'
    )
    
    parser.add_argument(
        '--predict', 
        action='store_true',
        help='使用训练好的模型进行预测'
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
        # 确定配置文件
        if args.config:
            config_path = args.config
        elif args.task == 'regression':
            config_path = 'config_regression.yaml'
        elif args.task == 'classification':
            config_path = 'config_classification.yaml'
        else:
            config_path = 'config_classification.yaml'  # 默认
        
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return 1
        
        # 初始化训练管道
        pipeline = PipelineTraining(config_path)
        
        # 从配置文件中读取模型设置
        from src.unified_config import UnifiedConfig
        config = UnifiedConfig(config_path)
        
        # 获取任务类型（从modeling配置中获取）
        task_type = config.get('modeling.task_type', 'classification')
        logger.info(f"任务类型: {task_type}")
        
        # 定义任务类型与模型的映射关系
        task_model_mapping = {
            'classification': {
                'XGB': 'XGBoost分类器',
                'RF': '随机森林分类器',
                'LR': '逻辑回归',
                'LGB': 'LightGBM分类器',
                'CAT': 'CatBoost分类器',
                'NB': '朴素贝叶斯',
                'SVM': '支持向量机',
                'DT': '决策树',
                'KNN': 'K近邻',
                'GB': '梯度提升分类器'
            },
            'regression': {
                'RIDGE': '岭回归',
                'XGB_REG': 'XGBoost回归器',
                'RF_REG': '随机森林回归器',
                'LGB_REG': 'LightGBM回归器',
                'CAT_REG': 'CatBoost回归器',
                'GB_REG': '梯度提升回归器',
                'LR_REG': '线性回归',
                'SVR': '支持向量回归',
                'DT_REG': '决策树回归',
                'KNN_REG': 'K近邻回归'
            }
        }
        
        # 获取可用模型列表
        available_models = config.get('modeling.models', 
                                    list(task_model_mapping[task_type].keys()))
        
        # 验证模型与任务类型的兼容性
        def validate_model_compatibility(model_name, task_type):
            """验证模型与任务类型的兼容性"""
            compatible_models = task_model_mapping.get(task_type, {})
            if model_name not in compatible_models:
                supported_models = list(compatible_models.keys())
                model_descriptions = [f"{k}: {v}" for k, v in compatible_models.items()]
                raise ValueError(
                    f"模型 '{model_name}' 不支持 {task_type} 任务。\n"
                    f"支持的{task_type}模型:\n" + 
                    "\n".join(f"  - {desc}" for desc in model_descriptions)
                )
            return True
        
        if args.predict:
            # 预测模式
            model_name = args.model or config.get('modeling.model_type')
            if not model_name:
                logger.error("错误: 未指定模型类型")
                return 1
            
            # 从input_data路径获取数据名称
            input_data_path = config.get('paths.input_data', 'data.csv')
            data_name = Path(input_data_path).stem
            
            logger.info(f"正在使用 {model_name} 模型进行{task_type}任务预测...")
            results = pipeline.predict()
            
            # 显示预测结果
            logger.info(f"预测完成，共 {len(results)} 条记录")
            
            if task_type == 'classification':
                logger.info("前10条预测结果:")
                logger.info("\n" + str(results[['prediction', 'probability']].head(10)))
                output_file = f"predictions_{task_type}_{model_name}_{data_name}.csv"
            else:
                logger.info("前10条预测结果:")
                logger.info("\n" + str(results[['prediction']].head(10)))
                output_file = f"predictions_{task_type}_{model_name}_{data_name}.csv"
            
            results.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存至: {output_file}")
            
        else:
            # 训练模式
            if args.model:
                # 指定单个模型
                model_name = args.model
                logger.info(f"开始训练单个{task_type}模型: {model_name}")
                
                # 验证模型与任务类型的兼容性
                try:
                    validate_model_compatibility(model_name, task_type)
                except ValueError as e:
                    logger.error(str(e))
                    return 1
                
                results = pipeline.run_single_model(model_name)
                
                if results['status'] == 'success':
                    logger.info(f"✅ {model_name} {task_type}模型训练成功!")
                    metrics = results['metrics']['test_metrics']
                    
                    if task_type == 'classification':
                        logger.info(f"测试集 AUC: {metrics['auc']}")
                        logger.info(f"测试集 KS: {metrics['ks']}")
                        logger.info(f"测试集 Accuracy: {metrics['accuracy']}")
                    else:
                        logger.info(f"测试集 RMSE: {metrics['rmse']}")
                        logger.info(f"测试集 MAE: {metrics['mae']}")
                        logger.info(f"测试集 R²: {metrics['r2']}")
                else:
                    logger.error(f"❌ {model_name} {task_type}模型训练失败: {results['error']}")
                    return 1
                    
            elif args.batch_mode:
                # 批量训练
                logger.info(f"开始批量训练{len(available_models)}个{task_type}模型: {', '.join(available_models)}")
                results = pipeline.run_all_models()
                
                # 统计成功/失败的模型
                successful = [r['model'] for r in results if 'error' not in r]
                failed = [r['model'] for r in results if 'error' in r]
                
                logger.info("训练完成!")
                logger.info(f"成功: {len(successful)} 个{task_type}模型 ({', '.join(successful)})")
                if failed:
                    logger.error(f"失败: {len(failed)} 个{task_type}模型 ({', '.join(failed)})")
                    return 1
            else:
                # 自动模式：根据模型数量决定
                if len(available_models) == 1:
                    model_name = available_models[0]
                    logger.info(f"开始训练单个{task_type}模型: {model_name}")
                    results = pipeline.run_single_model(model_name)
                    
                    if results['status'] == 'success':
                        logger.info(f"✅ {model_name} {task_type}模型训练成功!")
                        metrics = results['metrics']['test_metrics']
                        
                        if task_type == 'classification':
                            logger.info(f"测试集 AUC: {metrics['auc']}")
                            logger.info(f"测试集 KS: {metrics['ks']}")
                        else:
                            logger.info(f"测试集 RMSE: {metrics['rmse']}")
                            logger.info(f"测试集 R²: {metrics['r2']}")
                    else:
                        logger.error(f"❌ {model_name} {task_type}模型训练失败: {results['error']}")
                        return 1
                else:
                    logger.info(f"开始批量训练{len(available_models)}个{task_type}模型: {', '.join(available_models)}")
                    results = pipeline.run_all_models()
                    
                    successful = [r['model'] for r in results if 'error' not in r]
                    failed = [r['model'] for r in results if 'error' in r]
                    
                    logger.info("训练完成!")
                    logger.info(f"成功: {len(successful)} 个{task_type}模型 ({', '.join(successful)})")
                    if failed:
                        logger.error(f"失败: {len(failed)} 个{task_type}模型 ({', '.join(failed)})")
                        return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)