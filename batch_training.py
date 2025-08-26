"""
批量训练脚本 - 支持分类与回归双任务的多参数组合和多进程并行执行

使用方法:
    # 分类任务
    python batch_training.py --task classification --config batch_config.yaml
    python batch_training.py --task classification --config batch_config.yaml --dry-run
    
    # 回归任务
    python batch_training.py --task regression --config batch_config.yaml
    python batch_training.py --task regression --config batch_config.yaml --dry-run

功能特点:
    - 支持分类与回归任务配置
    - 支持YAML配置文件定义多组参数组合
    - 多进程并行执行，提高训练效率
    - 详细的进度跟踪和结果汇总
    - 支持干运行模式预览任务
    - 失败任务自动重试机制
"""
import sys
import os
import yaml
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline_training import PipelineTraining
from src.unified_config import UnifiedConfig

# 确保logs目录存在
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'batch_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchTrainer:
    """批量训练管理器"""
    
    def __init__(self, config_path):
        # 从配置文件读取设置
        self.config = UnifiedConfig(config_path)
        
        # 从配置文件获取并行设置
        self.max_workers = self.config.get('batch.max_workers', min(cpu_count(), 8))
        self.retry_times = self.config.get('batch.retry_times', 3)
        self.results = []
        
    def make_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        import numpy as np
        import pandas as pd
        
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'items') and callable(getattr(obj, 'items')):
            # 处理类似字典的对象
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__class__') and 'sklearn' in str(type(obj)):
            # 处理sklearn对象（如ColumnTransformer、Pipeline等）
            return str(obj)
        else:
            return str(obj)
    
    def generate_config_combinations(self, parameter_grid):
        """根据参数网格生成所有参数组合"""
        from itertools import product
        import copy
        
        # 生成所有组合
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combo in product(*values):
            # 创建参数字典
            params = {}
            for key, value in zip(keys, combo):
                # 处理嵌套键路径
                keys_path = key.split('.')
                current = params
                for k in keys_path[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys_path[-1]] = value
            combinations.append(params)
            
        return combinations
    
    def create_config_file(self, config, output_dir, index):
        """创建临时配置文件"""
        config_path = os.path.join(output_dir, f'temp_config_{index}.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return config_path
    
    def run_single_training(self, config_path, model_name, run_id):
        """执行单个训练任务"""
        try:
            logger.info(f"开始任务 {run_id}: 模型={model_name}, 配置={config_path}")
            
            # 使用配置文件创建训练管道
            pipeline = PipelineTraining(config_path=config_path)
            
            # 始终运行指定模型
            result = pipeline.run_single_model(model_name=model_name)
                
            # 清理结果，移除所有不可序列化的对象
            clean_result = {}
            for k, v in result.items():
                if k == 'model' or k == 'pipeline':
                    # 跳过模型和pipeline对象
                    continue
                else:
                    # 使用make_json_serializable清理其他值
                    try:
                        clean_result[k] = self.make_json_serializable(v)
                    except Exception as e:
                        logger.warning(f"清理键 '{k}' 时出错: {str(e)}")
                        clean_result[k] = str(v)  # 降级为字符串
            
            # 添加任务信息
            clean_result['run_id'] = run_id
            clean_result['config_path'] = str(config_path)  # 确保路径是字符串
            clean_result['model_name'] = model_name
            clean_result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"任务 {run_id} 完成")
            return clean_result
            
        except Exception as e:
            logger.error(f"任务 {run_id} 失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'run_id': run_id,
                'config_path': str(config_path),
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_training(self, dry_run=False, task_type = None):
        """执行批量训练"""
        
        # 获取参数网格
        parameter_grid = self.config.get('parameter_grid', {})
        logger.info(f"原始参数网格: {parameter_grid}")
        
        # 获取启用的模型列表
        enabled_models = self.config.get('models', [])
        logger.info(f"启用的模型: {enabled_models}")
        
        # 生成参数组合
        param_combinations = self.generate_config_combinations(parameter_grid)
        logger.info(f"生成 {len(param_combinations)} 组参数组合")
        
        # 生成任务列表
        tasks = []
        temp_dir = f'temp_configs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(temp_dir, exist_ok=True)
        
        task_index = 0
        for model_name in enabled_models:
            for params in param_combinations:
                # 应用参数组合到正确的嵌套路径 - 使用深拷贝避免覆盖
                import copy
                temp_config = copy.deepcopy(self.config.to_dict())
                
                # 移除批处理相关配置
                temp_config.pop('parameter_grid', None)
                temp_config.pop('models', None)
                
                # 应用参数组合到正确的嵌套路径
                def merge_nested_dict(d, key_path, value):
                    """合并嵌套字典的值，支持完整字典合并"""
                    if '.' in key_path:
                        # 点分路径，如 modeling.imbalance_method
                        keys = key_path.split('.')
                        current = d
                        for key in keys[:-1]:
                            if key not in current:
                                current[key] = {}
                            current = current[key]
                        current[keys[-1]] = value
                    else:
                        # 完整键，如 modeling
                        if isinstance(value, dict):
                            # 如果是字典，合并而不是覆盖
                            if key_path not in d:
                                d[key_path] = {}
                            d[key_path].update(value)
                        else:
                            # 如果不是字典，直接赋值
                            d[key_path] = value
                
                # 应用每个参数到正确的位置
                for key_path, value in params.items():
                    merge_nested_dict(temp_config, key_path, value)
                
                # 设置模型类型（覆盖原有值）
                if 'modeling' not in temp_config:
                    temp_config['modeling'] = {}
                temp_config['modeling']['model_type'] = model_name
                
                # 修复：确保paths配置完整，使用UnifiedConfig的get_paths方法获取正确路径
                if 'paths' not in temp_config:
                    temp_config['paths'] = {}
                
                # 获取当前的不平衡方法
                imbalance_method = temp_config.get('modeling', {}).get('imbalance_method', 'none')
                
                # 获取当前的数据文件路径
                input_data = temp_config.get('paths', {}).get('input_data', None)
                
                # 使用当前配置值生成路径，而不是原始配置
                data_name = Path(input_data).stem
                
                # 构建路径配置 - 确保每个任务有独立的日志文件
                timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # 包含微秒的高精度时间戳
                paths = {
                    'data_path': 'data',
                    'input_data': input_data,
                    'log_output': f'logs/{data_name}_{imbalance_method}_{model_name}_{timestamp_suffix}_training.log',
                    'model_output': f'models/{data_name}_{imbalance_method}_{model_name}_model.joblib',
                    'plot_output': f'plots/{data_name}_{imbalance_method}_{model_name}_{{plot_type}}_plot.png'
                }
                
                temp_config['paths'].update(paths)
                
                # 同时更新logging配置以确保一致性
                if 'logging' not in temp_config:
                    temp_config['logging'] = {}
                temp_config['logging']['file'] = paths['log_output']
                
                # 创建临时配置文件
                config_path = self.create_config_file(temp_config, temp_dir, task_index)
                run_id = f"{model_name}_{task_index}_{int(time.time())}"
                tasks.append({
                    'config_path': config_path,
                    'model_name': model_name,
                    'run_id': run_id
                })
                task_index += 1
        
        logger.info(f"共生成 {len(tasks)} 个训练任务")
        
        if dry_run:
            logger.info("干运行模式 - 预览任务:")
            parameter_grid = self.config.get('parameter_grid', {})
            logger.info(f"参数网格: {parameter_grid}")
            logger.info(f"生成的组合数量: {len(param_combinations)}")
            logger.info(f"总任务数量: {len(tasks)}")
            
            for i, task in enumerate(tasks):
                # 读取配置文件内容
                with open(task['config_path'], 'r', encoding='utf-8') as f:
                    config_content = yaml.safe_load(f)
                
                logger.info(f"任务 {i+1}/{len(tasks)}:")
                logger.info(f"  任务ID: {task['run_id']}")
                logger.info(f"  模型: {task['model_name']}")
                logger.info(f"  数据文件: {config_content.get('paths.input_data')}")
                logger.info(f"  不平衡方法: {config_content.get('modeling.imbalance_method')}")
                logger.info(f"  配置文件: {task['config_path']}")
                logger.info("  ---")
            return
        
        # 执行并行训练
        start_time = time.time()
        completed_tasks = 0
        failed_tasks = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.run_single_training, task['config_path'], task['model_name'], task['run_id']): task
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    if result['status'] == 'success':
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
                        
                    # 实时进度报告
                    progress = (completed_tasks + failed_tasks) / len(tasks) * 100
                    logger.info(f"进度: {progress:.1f}% ({completed_tasks+failed_tasks}/{len(tasks)})")
                    
                except Exception as e:
                    logger.error(f"任务执行异常: {str(e)}")
                    failed_tasks += 1
        
        # 清理临时文件
        # import shutil
        # shutil.rmtree(temp_dir)
        
        # 生成结果汇总
        elapsed_time = time.time() - start_time
        self.generate_summary_report(elapsed_time, completed_tasks, failed_tasks, task_type)
        
        return self.results
    
    def generate_summary_report(self, elapsed_time, completed_tasks, failed_tasks, task_type):
        """生成训练结果汇总报告"""
        
        # 清理所有结果中的不可序列化对象
        clean_results = []
        for result in self.results:
            clean_result = {}
            for k, v in result.items():
                try:
                    clean_result[k] = self.make_json_serializable(v)
                except Exception as e:
                    logger.warning(f"清理结果中的键 '{k}' 时出错: {str(e)}")
                    clean_result[k] = str(v)
            clean_results.append(clean_result)
        
        report = {
            'total_tasks': len(clean_results),
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / len(clean_results) if clean_results else 0,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'results': clean_results
        }
        
        # 保存详细结果到logs目录
        report_path = os.path.join(logs_dir, f'batch_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        logger.info("="*60)
        logger.info("批量训练完成!")
        logger.info(f"总任务数: {len(clean_results)}")
        logger.info(f"成功任务: {completed_tasks}")
        logger.info(f"失败任务: {failed_tasks}")
        logger.info(f"成功率: {report['success_rate']:.2%}")
        logger.info(f"总耗时: {elapsed_time:.2f}秒")
        logger.info(f"详细结果已保存至: {report_path}")
        
        # 生成全局模型比较报告
        self.generate_global_comparison_report(task_type)
        
        # 成功任务的结果摘要
        successful_results = [r for r in clean_results if r.get('status') == 'success']
        if successful_results:
            logger.info("\n成功任务摘要:")
            for result in successful_results:
                model = result.get('model_name', 'all')
                run_id = result['run_id']
                logger.info(f"  {run_id}: 模型={model}")
    
    def generate_global_comparison_report(self, task_type):
        """生成全局模型比较报告"""
        import pandas as pd
        
        logger.info("开始生成全局模型比较报告...")
        
        # 收集所有成功的模型结果
        model_results = []
        models_dir = Path('models')
        
        # 遍历所有结果文件
        for results_file in models_dir.glob('*_results.json'):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 解析文件名获取模型信息
                filename = results_file.stem  # 去掉.json和_model/results
                parts = filename.split('_')
                
                # 提取数据名、不平衡方法、模型名
                if len(parts) > 6:
                    data_name = '_'.join(parts[-6:-4])  # 合并前面的部分
                    imbalance_method = '_'.join(parts[-4:-2])
                    model_name = parts[-2]
                else:
                    data_name = '_'.join(parts[-5:-3])  # 合并前面的部分
                    imbalance_method = parts[-3]
                    model_name = parts[-2]
                
                
                # 提取关键指标 - 根据任务类型选择不同的指标
                evaluation_results = result_data.get('evaluation_results', {})
                test_metrics = evaluation_results.get('metrics', {}).get('test_metrics', {})
                
                # 基础信息
                model_info = {
                    'data_name': data_name,
                    'imbalance_method': imbalance_method,
                    'model_name': model_name,
                    'task_type': task_type,
                    'file_path': str(results_file)
                }
                
                # 根据任务类型提取相应指标
                if task_type == 'regression':
                    # 回归任务指标
                    model_info.update({
                        'rmse': test_metrics.get('rmse', 0),
                        'mae': test_metrics.get('mae', 0),
                        'r2': test_metrics.get('r2', 0),
                        'mse': test_metrics.get('mse', 0),
                        'mape': test_metrics.get('mape', 0)
                    })
                else:
                    # 分类任务指标（默认）
                    model_info.update({
                        'accuracy': test_metrics.get('accuracy', 0),
                        'precision': test_metrics.get('precision', 0),
                        'recall': test_metrics.get('recall', 0),
                        'f1': test_metrics.get('f1', 0),
                        'auc': test_metrics.get('auc', 0),
                        'ks': test_metrics.get('ks', 0)
                    })
                
                model_results.append(model_info)
                
            except Exception as e:
                logger.warning(f"跳过文件 {results_file}: {str(e)}")
        
        if not model_results:
            logger.warning("没有找到可用的模型结果文件，无法生成全局比较报告")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(model_results)
        
        # 按任务类型分组处理
        task_types = df['task_type'].unique()
        
        for task_type in task_types:
            task_df = df[df['task_type'] == task_type].copy()
            
            if task_type == 'regression':
                # 回归任务按R²排序（降序）
                task_df = task_df.sort_values('r2', ascending=False)
                sort_metric = 'r2'
            else:
                # 分类任务按AUC排序（降序）
                task_df = task_df.sort_values('auc', ascending=False)
                sort_metric = 'auc'
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存为CSV表格
            csv_path = os.path.join(logs_dir, f'global_model_comparison_{task_type}_{timestamp}.csv')
            task_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # 打印报告
            logger.info(f"\n{'='*80}")
            logger.info(f"全局模型比较报告 - {task_type.upper()}任务")
            logger.info(f"{'='*80}")
            
            # 选择要显示的列（根据任务类型）
            if task_type == 'regression':
                display_columns = ['data_name', 'imbalance_method', 'model_name', 'rmse', 'mae', 'r2', 'mape']
            else:
                display_columns = ['data_name', 'imbalance_method', 'model_name', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'ks']
            
            display_df = task_df[display_columns]
            logger.info(f"\n{display_df.to_string(index=False)}")
            
            # 生成汇总统计
            summary_stats = {
                'total_models': len(task_df),
                'best_model': task_df.iloc[0]['model_name'] if len(task_df) > 0 else None,
                'data_coverage': task_df['data_name'].nunique(),
                'method_coverage': task_df['imbalance_method'].nunique(),
                'model_coverage': task_df['model_name'].nunique()
            }
            
            # 添加最佳模型指标
            if task_type == 'regression':
                summary_stats['best_model_r2'] = task_df.iloc[0]['r2'] if len(task_df) > 0 else 0
                summary_stats['best_model_rmse'] = task_df.iloc[0]['rmse'] if len(task_df) > 0 else 0
            else:
                summary_stats['best_model_auc'] = task_df.iloc[0]['auc'] if len(task_df) > 0 else 0
                summary_stats['best_model_ks'] = task_df.iloc[0]['ks'] if len(task_df) > 0 else 0
            
            logger.info(f"\n{task_type.upper()}任务汇总统计:")
            logger.info(f"总模型数: {summary_stats['total_models']}")
            logger.info(f"最佳模型: {summary_stats['best_model']}")
            
            if task_type == 'regression':
                logger.info(f"最佳R²: {summary_stats['best_model_r2']:.4f}")
                logger.info(f"最佳RMSE: {summary_stats['best_model_rmse']:.4f}")
            else:
                logger.info(f"最佳AUC: {summary_stats['best_model_auc']:.4f}")
                logger.info(f"最佳KS: {summary_stats['best_model_ks']:.4f}")
            
            logger.info(f"覆盖数据集: {summary_stats['data_coverage']} 个")
            logger.info(f"覆盖不平衡处理方法: {summary_stats['method_coverage']} 个")
            logger.info(f"覆盖模型类型: {summary_stats['model_coverage']} 个")
            
            logger.info(f"\n报告文件已保存:")
            logger.info(f"📊 {task_type.upper()}任务CSV表格: {csv_path}")
            logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量训练脚本 - 支持多参数组合和多进程并行执行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_training.py --config batch_config.yaml
  python batch_training.py --config batch_config.yaml --dry-run
  python batch_training.py --help
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='批量配置文件路径 (将根据任务类型自动选择)'
    )
    
    parser.add_argument(
        '--task', 
        type=str,
        choices=['classification', 'regression'],
        default='classification',
        help='任务类型: classification(分类) 或 regression(回归) (默认: classification)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式，仅预览任务不执行'
    )
    
    args = parser.parse_args()
    
    try:
        # 确定配置文件
        if args.config:
            config_path = args.config
        elif args.task == 'regression':
            config_path = 'batch_config_regression.yaml'
        else:
            config_path = 'batch_config_classification.yaml'
        
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return 1
            
        # 初始化批量训练器
        trainer = BatchTrainer(config_path)
        
        # 执行批量训练
        trainer.run_batch_training(dry_run=args.dry_run, task_type = args.task)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)