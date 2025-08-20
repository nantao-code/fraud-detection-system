"""
模型转换管道模块
负责将训练好的模型转换为PMML和评分卡格式
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import json

from unified_config import UnifiedConfig
from data_loader import DataLoader
from model_converter import ModelConverter


class ModelConversionPipeline:
    """模型转换管道主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型转换管道
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_manager = UnifiedConfig(config_path)
        self.config = self.config_manager
        self.data_loader = DataLoader(self.config)
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = self.config.get('logging.level', 'INFO')
        log_file = self.config.get('paths.conversion_log', 'model_conversion.log')
        
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 清除现有的处理器，确保我们的配置生效
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def convert_model_to_pmml(self, model_name: str, dataset_name: str = None) -> str:
        """
        将指定模型转换为PMML格式
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称，如果为None则从配置中获取
            
        Returns:
            PMML文件路径
        """
        logging.info(f"开始将模型 {model_name} 转换为PMML格式")
        
        # 加载模型
        model = self._load_model(model_name)
        if model is None:
            raise ValueError(f"模型 {model_name} 未找到")
        
        # 加载特征名称
        feature_names = self._load_feature_names(model_name)
        if not feature_names:
            raise ValueError(f"无法获取模型 {model_name} 的特征名称")
        
        # 设置数据集名称
        if dataset_name is None:
            input_data_path = self.config.get('paths.input_data', 'data')
            dataset_name = Path(input_data_path).stem
        
        # 执行转换
        converter = ModelConverter()
        pmml_path = converter.convert_to_pmml(
            model_name, 
            feature_names, 
            dataset_name
        )
        
        logging.info(f"✓ PMML转换完成: {pmml_path}")
        return pmml_path
    
    def convert_model_to_scorecard(self, model_name: str, scorecard_type: str = "auto", 
                                 dataset_name: str = None) -> str:
        """
        将指定模型转换为评分卡格式
        
        Args:
            model_name: 模型名称
            scorecard_type: 评分卡类型 ('lr', 'dt', 'gbm', 'auto')
            dataset_name: 数据集名称，如果为None则从配置中获取
            
        Returns:
            评分卡文件路径
        """
        logging.info(f"开始将模型 {model_name} 转换为评分卡格式 (类型: {scorecard_type})")
        
        # 加载模型
        model = self._load_model(model_name)
        if model is None:
            raise ValueError(f"模型 {model_name} 未找到")
        
        # 加载训练数据
        all_df = self.data_loader.load_data()
        feature_cols = self.data_loader.get_feature_columns(all_df)
        target_col = self.data_loader.get_target_column()
        
        # 设置数据集名称
        if dataset_name is None:
            dataset_name = Path(self.config.get('paths.input_data', 'data')).stem
        
        # 加载特征名称
        feature_names = self._load_feature_names(model_name)
        
        # 执行转换
        converter = ModelConverter()
        scorecard_report = converter.convert_to_scorecard(
            model_name,
            model,
            all_df[feature_cols],
            all_df[target_col],
            feature_names,
            scorecard_type
        )
        
        scorecard_path = scorecard_report["scorecard_path"]
        logging.info(f"✓ 评分卡转换完成: {scorecard_path}")
        return scorecard_path
    
    def batch_convert_models(self, models: list = None, convert_pmml: bool = True, 
                           convert_scorecard: bool = True) -> Dict[str, Any]:
        """
        批量转换多个模型
        
        Args:
            models: 要转换的模型列表，如果为None则转换所有已训练的模型
            convert_pmml: 是否转换为PMML格式
            convert_scorecard: 是否转换为评分卡格式
            
        Returns:
            包含转换结果的字典
        """
        logging.info("开始批量模型转换")
        
        if models is None:
            models = self._get_trained_models()
        
        if not models:
            logging.warning("未找到任何已训练的模型")
            return {}
        
        results = {}
        input_data_path = self.config.get('paths.input_data', 'data')
        dataset_name = Path(input_data_path).stem
        
        for model_name in models:
            logging.info(f"\n{'-'*50}")
            logging.info(f"开始转换模型: {model_name}")
            
            results[model_name] = {
                'status': 'success',
                'pmml_path': None,
                'scorecard_path': None,
                'error': None
            }
            
            try:
                if convert_pmml:
                    pmml_path = self.convert_model_to_pmml(model_name, dataset_name)
                    results[model_name]['pmml_path'] = pmml_path
                
                if convert_scorecard:
                    scorecard_path = self.convert_model_to_scorecard(model_name, "auto", dataset_name)
                    results[model_name]['scorecard_path'] = scorecard_path
                    
            except Exception as e:
                logging.error(f"模型 {model_name} 转换失败: {str(e)}")
                results[model_name]['status'] = 'failed'
                results[model_name]['error'] = str(e)
        
        logging.info("\n批量模型转换完成")
        return results
    
    def _load_model(self, model_name: str):
        """加载训练好的模型"""
        # 获取配置信息用于路径模板
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 使用配置中的模板格式化模型路径
        model_output_template = self.config.get('paths.model_output', 'models/{input_data}_{imbalance_method}_{model_name}_model.joblib')
        model_filename = model_output_template.format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )
        model_file = Path(model_filename)
        
        if not model_file.exists():
            logging.warning(f"模型文件不存在: {model_file}")
            return None
        
        return joblib.load(model_file)
    
    def _load_feature_names(self, model_name: str) -> list:
        """加载模型的特征名称"""
        # 获取配置信息用于路径模板
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 使用配置中的模板格式化结果文件路径
        results_filename = f"{input_data_name}_{imbalance_method}_{model_name}_results.json"
        
        # 获取模型输出目录
        model_output_template = self.config.get('paths.model_output', 'models/{input_data}_{imbalance_method}_{model_name}_model.joblib')
        base_path = Path(model_output_template.format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )).parent
        
        results_path = base_path / results_filename
        
        if not results_path.exists():
            logging.warning(f"结果文件不存在: {results_path}")
            return []
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results.get('final_features', [])
        except Exception as e:
            logging.error(f"加载特征名称失败: {str(e)}")
            return []
    
    def _get_trained_models(self) -> list:
        """获取所有已训练的模型列表"""
        # 获取配置信息
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 获取模型输出目录
        model_output_template = self.config.get('paths.model_output', 'models/{input_data}_{imbalance_method}_{model_name}_model.joblib')
        base_path = Path(model_output_template.format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name="placeholder"  # 仅用于获取目录
        )).parent
        
        # 查找所有模型文件
        pattern = f"{input_data_name}_{imbalance_method}_*_model.joblib"
        model_files = list(base_path.glob(pattern))
        
        # 提取模型名称
        models = []
        prefix = f"{input_data_name}_{imbalance_method}_"
        suffix = "_model.joblib"
        
        for file_path in model_files:
            filename = file_path.name
            if filename.startswith(prefix) and filename.endswith(suffix):
                model_name = filename[len(prefix):-len(suffix)]
                models.append(model_name)
        
        return sorted(models)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型转换管道')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='指定要转换的模型')
    parser.add_argument('--pmml', action='store_true', help='转换为PMML格式')
    parser.add_argument('--scorecard', action='store_true', help='转换为评分卡格式')
    parser.add_argument('--batch', action='store_true', help='批量转换所有模型')
    
    args = parser.parse_args()
    
    pipeline = ModelConversionPipeline(args.config)
    
    if args.batch:
        # 批量转换所有模型
        results = pipeline.batch_convert_models(
            convert_pmml=args.pmml or True,  # 默认转换PMML
            convert_scorecard=args.scorecard or True  # 默认转换评分卡
        )
        
        print("\n转换结果:")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  状态: {result['status']}")
            if result['pmml_path']:
                print(f"  PMML: {result['pmml_path']}")
            if result['scorecard_path']:
                print(f"  评分卡: {result['scorecard_path']}")
            if result['error']:
                print(f"  错误: {result['error']}")
    
    elif args.model:
        # 转换单个模型
        results = {}
        
        if args.pmml or not args.scorecard:  # 默认转换PMML
            try:
                pmml_path = pipeline.convert_model_to_pmml(args.model)
                results['pmml_path'] = pmml_path
                print(f"✓ PMML转换完成: {pmml_path}")
            except Exception as e:
                print(f"✗ PMML转换失败: {str(e)}")
        
        if args.scorecard:
            try:
                scorecard_path = pipeline.convert_model_to_scorecard(args.model)
                results['scorecard_path'] = scorecard_path
                print(f"✓ 评分卡转换完成: {scorecard_path}")
            except Exception as e:
                print(f"✗ 评分卡转换失败: {str(e)}")
    
    else:
        print("请指定转换模式:")
        print("  --batch: 批量转换所有模型")
        print("  --model <模型名>: 转换单个模型")
        print("  --pmml: 转换为PMML格式")
        print("  --scorecard: 转换为评分卡格式")


if __name__ == "__main__":
    main()