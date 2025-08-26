"""
Pipeline训练管道模块
使用sklearn.pipeline和imblearn.pipeline进行完整的训练流程
集成特征工程（特征选择、数据预处理）
支持PMML导出
"""
import os
import logging
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

import warnings

# sklearn imports
from pandas.core.strings.accessor import cat_safe
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
set_config(transform_output="pandas")

# imblearn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 模型导入
from model_factory import ModelFactory
from unified_config import UnifiedConfig
from data_loader import DataLoader
from evaluator import ModelEvaluator, ModelComparator
from feature_engineering import CategoricalConverter, DataPreprocessor, FeatureGeneratorPipeline, FeatureSelectorPipeline, ColumnSelector

warnings.filterwarnings('ignore')

class PipelineTraining:
    """使用Pipeline的完整训练管道"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化Pipeline训练管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config = UnifiedConfig(config_path)
        self.data_loader = DataLoader(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
    
    def _setup_logging(self, model_name : str = None):
        """
        设置日志配置，从配置文件获取参数
        """
        log_level = self.config.get('logging.level', 'INFO')
        log_dir = self.config.get('paths.logs_dir', 'logs')
        data_name = self.config.get('paths.input_data', 'data').split('/')[-1].replace('.csv', '')
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        model_type = model_name or self.config.get('modeling.model_type', 'unknown_model')
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 构建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{data_name}_{imbalance_method}_{model_type}_{timestamp}.log")
        
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
        
        logging.info(f"日志文件已创建: {log_file}")
        return log_file
    

    

    
    def explore_data(self, X: pd.DataFrame, y: pd.Series, save_dir: str = None) -> List[str]:
        """
        数据探索函数：执行数据预处理、特征生成、特征选择，并保存最终特征
        
        Args:
            X: 特征数据
            y: 目标变量
            save_dir: 保存结果的目录，默认为配置中的results_dir
            
        Returns:
            最终选择的特征列表
        """
        logging.info("开始数据探索...")
        
        # 设置保存目录
        if save_dir is None:
            save_dir = self.config.get('paths.results_dir', 'results')
        os.makedirs(save_dir, exist_ok=True)
        
        # 从配置获取任务类型
        task_type = self.config.get('modeling.task_type', 'classification')
        model_type = self.config.get('modeling.model_type', 'LR')
        
        # 从配置中提取预处理参数
        preproc_config = self.config.get('feature_engineering.preprocessing', {})
        
        # 1. 执行数据预处理
        logging.info("执行数据预处理...")
        preprocessor = DataPreprocessor(
            model_type=model_type,
            missing_strategy=preproc_config.get('missing_strategy', 'median'),
            outlier_method=preproc_config.get('outlier_method', 'iqr'),
            outlier_threshold=preproc_config.get('outlier_threshold', 1.5),
            transform_method=preproc_config.get('transform_method', 'yeo-johnson'),
            scale_method=preproc_config.get('scale_method', 'standard'),
            encoding_method=preproc_config.get('encoding_method', 'ordinal'),
            categorical_threshold=preproc_config.get('categorical_threshold', 10),
            numerical_threshold=preproc_config.get('numerical_threshold', 20)
        )
        X_preprocessed = preprocessor.fit_transform(X, y)
        logging.info(f"数据预处理完成")
        
        # # 2. 执行特征生成
        # logging.info("执行特征生成...")
        # gen_config = self.config.get('feature_engineering.feature_generation', {})
        # feature_generator = FeatureGeneratorPipeline(
        #     generate_polynomial=gen_config.get('generate_polynomial', False),
        #     polynomial_degree=gen_config.get('polynomial_degree', 2),
        #     polynomial_include_bias=gen_config.get('polynomial_include_bias', False),
        #     generate_interaction=gen_config.get('generate_interaction', False),
        #     interaction_pairs=gen_config.get('interaction_pairs', None),
        #     generate_statistical=gen_config.get('generate_statistical', False),
        #     statistical_group_cols=gen_config.get('statistical_group_cols', None),
        #     generate_binning=gen_config.get('generate_binning', False),
        #     bins_config=gen_config.get('bins_config', None),
        #     generate_aggregation=gen_config.get('generate_aggregation', False),
        #     aggregation_window_cols=gen_config.get('aggregation_window_cols', None)
        # )
        # X_engineered = feature_generator.fit_transform(X_preprocessed, y)
        # logging.info(f"特征工程完成，特征数量: {X_engineered.shape[1]}")
        
        # 3. 执行特征选择
        logging.info("执行特征选择...")
        select_config = self.config.get('feature_engineering.feature_selection', {})
        feature_selector = FeatureSelectorPipeline(
            task_type=task_type,
            selection_methods=select_config.get('selection_methods', ['missing_rate', 'correlation', 'variance']),
            missing_rate_threshold=select_config.get('missing_rate_threshold', 0.5),
            correlation_threshold=select_config.get('correlation_threshold', 0.9),
            variance_threshold=select_config.get('variance_threshold', 0.01),
            iv_threshold=select_config.get('iv_threshold', 0.02),
            mutual_info_k=select_config.get('mutual_info_k', 'auto'),
            importance_threshold=select_config.get('importance_threshold', 0.01),
            model_type=model_type,
            f_regression_k=select_config.get('f_regression_k', 'auto'),
            f_classif_k=select_config.get('f_classif_k', 'auto')
        )
        feature_selector.fit(X_preprocessed, y)
        
        # 4. 获取最终特征列表
        final_features = list(feature_selector.selected_features_)
        logging.info(f"特征选择完成，最终特征数量: {len(final_features)}")
        
        # 5. 保存最终特征
        features_path = os.path.join(save_dir, 'final_features.json')
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(final_features, f, ensure_ascii=False, indent=2)
        logging.info(f"最终特征已保存到: {features_path}")
        
        return final_features
    
    def _create_model(self, model_name: str=None):
        """
        创建模型实例
        
        Args:
            model_name: 模型名称，如果为None则从配置获取
            
        Returns:
            模型实例和参数网格的元组
        """
        # 从配置获取模型类型
        model_type = model_name or self.config.get('modeling.model_type', 'unknown_model')
        
        # 使用ModelFactory创建模型和参数网格，传入完整配置
        model, param_grid = ModelFactory.create_model(model_type, self.config)
        
        return model, param_grid
    
    def _create_preprocessor(self, numeric_features: List[str], 
                           categorical_features: List[str], model_type: str = None) -> ColumnTransformer: 
        """ 
        创建预处理器，根据模型类型选择不同的编码策略 
        
        Args: 
            numeric_features: 数值特征列表 
            categorical_features: 类别特征列表 
            model_type: 模型类型
            
        Returns: 
            ColumnTransformer实例 
        """ 
        
        # 从配置获取缺失值处理策略
        missing_strategy = self.config.get('feature_engineering.preprocessing.missing_strategy', 'median')
        
        # 根据模型类型选择特征处理策略 
        model_type_upper = str(model_type or self.config.get('modeling.model_type', 'LR')).upper() 
        
        if model_type_upper in ['LR', 'RIDGE', 'LOGISTIC']: 
            # 逻辑回归/岭回归：数值特征需要标准化 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=missing_strategy)), 
                ('scaler', StandardScaler()) 
            ]) 
            logging.info(f"为{model_type_upper}模型使用StandardScaler标准化数值特征") 
            
            # 线性模型：独热编码处理类别特征 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
            ]) 
            logging.info(f"为{model_type_upper}模型使用独热编码处理类别特征") 
            
        elif model_type_upper in ['RF', 'RF_REG', 'RANDOMFOREST']: 
            # 随机森林：数值特征无需缩放，类别特征使用序号编码 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=missing_strategy)) 
            ]) 
            logging.info(f"为{model_type_upper}模型不对数值特征进行缩放处理") 
            
            # RF模型：使用序号编码处理类别特征 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
            ]) 
            logging.info(f"为{model_type_upper}模型使用序号编码处理类别特征") 
            
        elif model_type_upper in ['XGB', 'LGB', 'XGBOOST', 'LIGHTGBM', 'XGB_REG', 'LGB_REG']: 
            # XGBoost/LightGBM：数值特征无需缩放，使用专用CategoricalConverter 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=missing_strategy)) 
            ]) 
            logging.info(f"为{model_type_upper}模型不对数值特征进行缺失值填充与缩放处理") 
            
            # XGB/LGB模型：使用专为它们优化的CategoricalConverter 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                ('categorical_converter', CategoricalConverter()) 
            ]) 
            logging.info(f"为{model_type_upper}模型使用专为XGBoost/LightGBM优化的CategoricalConverter") 
            
        else: 
            # 默认处理策略
            logging.warning(f"未识别的模型类型: {model_type_upper}，使用默认处理策略") 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=missing_strategy)), 
                ('scaler', StandardScaler()) 
            ]) 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
            ]) 
        
        # 动态构建transformers列表 
        transformers = [] 
        
        # 只有当存在数值特征时，才添加数值转换器 
        if numeric_features: 
            transformers.append(('num', numeric_transformer, numeric_features)) 
            logging.info(f"为 {len(numeric_features)} 个数值特征添加了转换器。") 
        
        # 只有当存在类别特征时，才添加类别转换器 
        if categorical_features: 
            transformers.append(('cat', categorical_transformer, categorical_features)) 
            logging.info(f"为 {len(categorical_features)} 个类别特征添加了转换器。") 
        
        # 如果没有任何特征，这是一个异常情况 
        if not transformers: 
            logging.warning("警告：数据中既没有检测到数值特征，也没有检测到类别特征。预处理器将是空的。") 
        
        # 组合预处理器 
        preprocessor = ColumnTransformer( 
            transformers=transformers, 
            remainder='passthrough', 
            verbose_feature_names_out=False 
        ) 
        
        return preprocessor
    
    # 检测数据类型
    def detect_data_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        检测数据框中的数值特征和类别特征
        
        Args:
            X: 输入数据框
            
        Returns:
            数值特征列表和类别特征列表
        """
        # 初始化空列表
        numeric_features = []
        categorical_features = []
        
        # 遍历数据框的每个列
        for col in X.columns:
            # 检查列的数据类型
            if pd.api.types.is_numeric_dtype(X[col]):
                # 如果是数值类型，添加到数值特征列表
                numeric_features.append(col)
            else:
                # 如果不是数值类型，添加到类别特征列表
                categorical_features.append(col)
        
        return numeric_features, categorical_features

    
    def train_model(self, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = None) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any], List[str]]:
        """
        训练模型 (配置驱动版)
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称，如果为None则从配置获取
            
        Returns:
            (训练好的pipeline, 评估结果, 最佳参数)
        """
        import warnings
        warnings.filterwarnings('ignore')

        # 从配置文件获取任务类型
        task_type = self.config.get('modeling.task_type', 'classification')
        logging.info(f"配置文件指定任务类型: {task_type}")
        
        model_type = model_name or self.config.get('modeling.model_type', 'LR')
        logging.info(f"开始训练{model_type}模型...")

        # 1. 数据探索：获取最终特征列表
        logging.info("步骤1: 执行数据探索，获取最终特征列表...")
        final_features = self.explore_data(X_train, y_train)
        logging.info(f"数据探索完成，最终特征数量: {len(final_features)}")
        logging.info(f"最终特征列表: {final_features}")
        
        # 2. 创建完整的训练Pipeline
        logging.info("步骤2: 创建完整的训练Pipeline...")

        X_train = X_train[final_features]
        X_test = X_test[final_features]
        
        steps = []
        numeric_features, categorical_features = self.detect_data_types(X_train)
        preprocessor = self._create_preprocessor(numeric_features, categorical_features, model_type)
        steps.append(('preprocessor', preprocessor))

        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        # 2. 不平衡处理（可选）
        if imbalance_method != 'none':
            # 从配置获取随机状态
            random_state = self.config.get('modeling.random_state', 42)
            
            if imbalance_method == 'smote':
                sampler = SMOTE(random_state=random_state)
            elif imbalance_method == 'random_over':
                sampler = RandomOverSampler(random_state=random_state)
            elif imbalance_method == 'random_under':
                sampler = RandomUnderSampler(random_state=random_state)
            else:
                raise ValueError(f"不支持的不平衡处理方法: {imbalance_method}")
            
            steps.append(('sampler', sampler))
            logging.info(f"添加了{imbalance_method}采样器")

        # 创建完整的训练Pipeline
        model, param_grid = self._create_model(model_type)
        
        steps.append(('classifier', model))
        
        # 根据不平衡处理方法选择Pipeline类型
        
        if imbalance_method != 'none':
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipeline = ImbPipeline(steps=steps)
        else:
            pipeline = Pipeline(steps=steps)
        
        use_hyperparameter_tuning = self.config.get('modeling.use_hyperparameter_tuning', False)
        
        # 统一准备早停机制所需的验证集和fit_params
        fit_params = {}
        train_data = (X_train, y_train) # 默认使用全部训练数据
        
        # 检查是否需要早停机制
        needs_early_stopping = model_type.upper() in ['XGB', 'LGB', 'XGB_REG', 'LGB_REG']
        
        if needs_early_stopping:
            logging.info(f"为 {model_type} 创建早停验证集...")
            validation_split_ratio = self.config.get('modeling.validation_split_ratio', 0.1)
            
            # 根据任务类型决定是否使用分层抽样
            stratify = y_train if task_type == 'classification' else None
            
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, 
                test_size=validation_split_ratio, 
                random_state=self.config.get('modeling.random_state', 42),
                stratify=stratify
            )
            train_data = (X_train_main, y_train_main) # 更新训练数据为分割后的主训练集
            
            if model_type.upper() == 'XGB':
                fit_params = {'classifier__eval_set': [(X_val, y_val)], "classifier__verbose": False}
            elif model_type.upper() == 'LGB':
                import lightgbm as lgb
                fit_params = {
                    "classifier__eval_set": [(X_val, y_val)],
                    "classifier__callbacks": [lgb.early_stopping(
                        stopping_rounds=self.config.get('modeling.early_stopping_rounds', 100), 
                        verbose=False
                    )]
                }

        # 根据任务类型选择交叉验证器和评估指标
        if use_hyperparameter_tuning and param_grid:
            logging.info("启用超参数优化...")
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
            
            # 为参数网格添加前缀
            prefixed_param_grid = {}
            for param_name, param_values in param_grid.items():
                prefixed_param_grid[f'classifier__{param_name}'] = param_values
            
            # 根据任务类型选择交叉验证器
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.config.get('modeling.n_splits', 5), 
                                   shuffle=True, 
                                   random_state=self.config.get('modeling.random_state', 42))
                scoring = self.config.get('feature_engineering.primary_metric_classification', 'roc_auc')
            else:  # regression
                cv = KFold(n_splits=self.config.get('modeling.n_splits', 5), 
                          shuffle=True, 
                          random_state=self.config.get('modeling.random_state', 42))
                scoring = self.config.get('feature_engineering.primary_metric_regression', 'neg_root_mean_squared_error')
            
            n_iter_search = self.config.get('modeling.n_iter', 20)
            
            searcher = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=prefixed_param_grid,
                n_iter=n_iter_search,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.config.get('modeling.random_state', 42),
                refit=True
            )
            training_entity = searcher
            total_fits = cv.n_splits * n_iter_search
            
        else:
            logging.info("不启用超参数优化，使用默认参数训练。")
            training_entity = pipeline
            total_fits = 1

        # 执行训练
        from tqdm import tqdm
        with tqdm(total=total_fits, desc=f"训练 {model_type}") as pbar:
            # 将准备好的训练数据和fit_params传入
            training_entity.fit(*train_data, **fit_params)
            pbar.update(total_fits)

        # 获取最终的pipeline和最佳参数
        if use_hyperparameter_tuning and param_grid:
            final_pipeline = training_entity.best_estimator_
            best_params = training_entity.best_params_
            logging.info(f"超参数优化完成。最佳分数: {training_entity.best_score_:.4f}")
            logging.info(f"最佳参数: {best_params}")
        else:
            final_pipeline = training_entity
            best_params = final_pipeline.get_params()

        # 评估模型
        logging.info("开始模型评估...")
        evaluation_results = self.evaluator.evaluate_model(
            final_pipeline, X_train, y_train, X_test, y_test, model_type, task_type

        )
        
        return final_pipeline, evaluation_results, best_params, final_features
    
    def run_single_model(self, model_name: str = None) -> Dict[str, Any]:
        """
        运行单个模型训练（改造后：特征工程在数据分割后进行）
        
        Args:
            model_name: 模型名称，如果为None则从配置获取
            
        Returns:
            训练结果
        """


        
        # 使用参数中的模型名称或从配置获取
        model_type = model_name or self.config.get('modeling.model_type', 'LR')
        data_name = self.config.get('paths.input_data', 'data').split('/')[-1].replace('.csv', '')
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')

        self._setup_logging(model_type)
        logging.info(f"\n{'='*80}")
        logging.info(f"开始Pipeline训练: {model_type}")
        logging.info(f"不平衡处理方法: {imbalance_method}")
        logging.info(f"使用特征工程: 已启用（默认）")
        logging.info(f"{'='*80}")
        
        try:
            # 1. 加载数据
            logging.info("步骤1/8: 开始加载数据...")
            all_df = self.data_loader.load_data()
            logging.info(f"数据加载完成，数据形状: {all_df.shape}")
            logging.info(f"数据列: {list(all_df.columns)}")

            # 2. 验证数据
            logging.info("步骤2/8: 开始验证数据质量...")
            if not self.data_loader.validate_data(all_df, 'train'):
                raise ValueError("数据验证失败")
            logging.info("数据验证通过")

            # 3. 获取特征和目标变量
            logging.info("步骤3/8: 提取特征和目标变量...")
            feature_cols = self.data_loader.get_feature_columns(all_df)
            target_col = self.data_loader.get_target_column()
            
            # 自动检测categorical特征
            categorical_features = self.data_loader.detect_categorical_features(all_df, feature_cols)
            logging.info(f"检测到 {len(categorical_features)} 个categorical特征: {categorical_features}")
            
            logging.info(f"特征列数量: {len(feature_cols)}")
            logging.info(f"目标变量: {target_col}")
            logging.info(f"categorical特征: {categorical_features}")

            # 4. 分割数据 - 使用DataSplitter替代DataLoader
            logging.info("步骤4/8: 开始数据分割...")
            from data_loader import DataSplitter
            data_splitter = DataSplitter(self.config)
            
            # 从配置文件获取任务类型
            task_type = self.config.get('modeling.task_type', 'classification')
            
            X_train, X_test, y_train, y_test = data_splitter.split_data(
                all_df, feature_cols, target_col, task_type
            )
            logging.info(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
            logging.info(f"训练集类别分布: {dict(y_train.value_counts())}")
            logging.info(f"测试集类别分布: {dict(y_test.value_counts())}")

            # 5. 训练模型（新的train_model已经包含数据探索和特征工程）
            logging.info("步骤5/8: 开始模型训练...")
            start_time = time.time()
            
            
            logging.info(f"\n{'='*60}")
            logging.info(f"开始训练模型: {model_type}")
            logging.info(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                pipeline, evaluation_results, best_params, final_features = self.train_model(
                    X_train, y_train, 
                    X_test, y_test,
                    model_type
                )
                
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model_type,
                    'data_name': data_name,
                    'imbalance_method': imbalance_method,
                    'metrics': evaluation_results,
                    'best_params': best_params,
                    'training_time': training_time,
                    'feature_count': len(final_features),
                    'status': 'success'
                }
                

                
            except Exception as e:
                logging.error(f"模型 {model_type} 训练失败: {str(e)}")
                logging.error(traceback.format_exc())
                
                result = {
                    'model_name': model_type,
                    'data_name': data_name,
                    'imbalance_method': imbalance_method,
                    'error': str(e),
                    'status': 'failed'
                }
            
            training_time = time.time() - start_time
            logging.info(f"模型训练完成，耗时: {training_time:.2f}秒")

            # 7. 保存模型和结果（仅在成功时执行）
            if result['status'] == 'success':
                logging.info("步骤7/8: 保存训练好的pipeline和结果...")
                pipeline_path = self._save_pipeline(pipeline, model_type, result)
                logging.info(f"Pipeline已保存到: {pipeline_path}")

                # 8. 保存特征信息
                logging.info("步骤8/8: 保存特征信息...")
                self._save_final_features(final_features, model_type)
                logging.info("特征信息保存完成")

                # 获取评估结果用于日志输出
                evaluation_results = result.get('metrics', {})
                best_params = result.get('best_params', {})
                
                logging.info("=" * 80)
                logging.info(f"模型 {model_type} 训练完成")
                
                test_metrics = evaluation_results.get('test_metrics', {})
                task_type = evaluation_results.get('task_type', 'unknown')
                
                if task_type == 'classification':
                    logging.info(f"  AUC: {test_metrics.get('auc', 'N/A')}")
                    logging.info(f"  KS: {test_metrics.get('ks', 'N/A')}")
                    logging.info(f"  F1: {test_metrics.get('f1', 'N/A')}")
                    logging.info(f"  Precision: {test_metrics.get('precision', 'N/A')}")
                    logging.info(f"  Recall: {test_metrics.get('recall', 'N/A')}")
                    logging.info(f"  Accuracy: {test_metrics.get('accuracy', 'N/A')}")
                elif task_type == 'regression':
                    logging.info(f"  R²: {test_metrics.get('r2', 'N/A')}")
                    logging.info(f"  RMSE: {test_metrics.get('rmse', 'N/A')}")
                    logging.info(f"  MAE: {test_metrics.get('mae', 'N/A')}")
                    logging.info(f"  MSE: {test_metrics.get('mse', 'N/A')}")
                else:
                    logging.info(f"  ✓ {model_name}: {test_metrics}"
                           f"耗时={result.get('training_time', 0):.2f}s")
                
                logging.info(f"  Best_params: {best_params}")
                logging.info("=" * 80)
            else:
                logging.warning("模型训练失败，跳过保存步骤")

            return result

        except Exception as e:
            logging.error(f"训练过程中发生错误: {str(e)}")
            logging.error(traceback.format_exc())
            
            result = {
                'model_name': model_type,
                'imbalance_method': imbalance_method,
                'error': str(e),
                'status': 'failed'
            }
            
            return result
    
    def run_all_models(self):
        """
        运行所有模型的训练（配置文件驱动版本）
        
        该方法从配置文件中获取模型列表，并为每个模型运行训练。
        使用临时配置更新机制确保每个模型使用正确的配置。
        训练完成后，使用ModelComparator生成模型比较报告。
        """
        # 从配置获取基础信息
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        data_name = self.config.get('paths.input_data', 'data').split('/')[-1].replace('.csv', '')
        task_type = self.config.get('modeling.task_type', 'classification')
        
        # 获取模型列表 - 支持多种配置方式
        model_names = self.config.get('modeling.models', [])
        
        if not model_names:
            logging.warning("未找到任何启用的模型，请检查配置文件中的 'models' 或 'modeling.models'")
            return []

        logging.info("=" * 80)
        logging.info(f"开始批量模型训练")
        logging.info(f"不平衡处理方法: {imbalance_method}")
        logging.info(f"数据名称: {data_name}")
        logging.info(f"模型列表: {model_names}")
        logging.info("=" * 80)
        
        results = []
        total_models = len(model_names)
        
        # 初始化模型比较器
        comparator = ModelComparator()
        
        for idx, model_name in enumerate(model_names, 1):
            logging.info("-" * 60)
            logging.info(f"[{idx}/{total_models}] 开始训练模型: {model_name}")
            logging.info("-" * 60)
            start_time = time.time()
            try:
                # 运行单个模型训练
                result = self.run_single_model(model_name)
                result['model'] = model_name
                result['batch_index'] = idx
                
                training_time = time.time() - start_time
                result['training_time'] = training_time
                
                logging.info(f"✓ 模型 {model_name} 训练成功")
                logging.info(f"  训练时间: {training_time:.2f} 秒")
                
                # 将成功的模型结果添加到比较器中
                test_metrics = result.get('metrics', {}).get('test_metrics', {})
                if not isinstance(test_metrics, dict):
                    logging.warning(f"警告: test_metrics 不是字典类型，而是 {type(test_metrics)}")
                    test_metrics = {}
                
                model_info = {
                    'model': model_name,
                    'imbalance_method': imbalance_method,
                    'data_name': data_name,
                    'training_time': training_time,
                    **test_metrics
                }
                comparator.add_model_result(model_name, model_info)
                results.append(result)
                
            except Exception as e:
                training_time = time.time() - start_time
                logging.error(f"✗ 模型 {model_name} 训练失败: {str(e)}")
                logging.error(traceback.format_exc())
                
                # 记录失败结果
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result = {
                    'model': model_name,
                    'batch_index': idx,
                    'imbalance_method': imbalance_method,
                    'data_name': data_name,
                    'error': str(e),
                    'training_time': training_time,
                    'timestamp': timestamp_str
                }
                results.append(result)
        
        # 生成批量训练报告
        self._generate_batch_report(results)
        
        # 生成模型比较报告
        self._generate_comparison_report(comparator, results, task_type)
        
        return results
    
    def _generate_batch_report(self, results: List[Dict[str, Any]]):
        """生成批量训练汇总报告"""
        logging.info("=" * 80)
        logging.info("批量训练完成汇总")
        logging.info("=" * 80)
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        logging.info(f"总模型数: {len(results)}")
        logging.info(f"成功: {len(successful)}")
        logging.info(f"失败: {len(failed)}")
        
        if successful:
            logging.info("\n成功训练的模型:")
            for result in successful:
                model_name = result['model']
                metrics = result.get('metrics', {})
                test_metrics = metrics.get('test_metrics', {})
                task_type = metrics.get('task_type', 'classification')
                
                if task_type == 'classification':
                    logging.info(f"  ✓ {model_name}: "
                               f"AUC={test_metrics.get('auc', 'N/A')}, "
                               f"F1={test_metrics.get('f1', 'N/A')},"
                               f"KS={test_metrics.get('ks', 'N/A')}")
                elif task_type == 'regression':
                    logging.info(f"  ✓ {model_name}: "
                               f"R²={test_metrics.get('r2', 'N/A')}, "
                               f"RMSE={test_metrics.get('rmse', 'N/A')}")
                else:
                    logging.info(f"  ✓ {model_name}: {test_metrics}"
                           f"耗时={result.get('training_time', 0)}s")
        
        if failed:
            logging.warning("\n训练失败的模型:")
            for result in failed:
                logging.warning(f"  ✗ {result['model']}: {result['error']}")
        
        # 保存批量训练结果
        self._save_batch_results(results)
    
    def _save_batch_results(self, results: List[Dict[str, Any]]):
        """保存批量训练结果"""
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        data_name = self.config.get('paths.input_data', 'data').split('/')[-1].replace('.csv', '')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_path = self.config.get(
            'paths.batch_results_output',
            f'models/batch_results_{data_name}_{imbalance_method}_{timestamp}.json'
        )
        
        # 确保目录存在
        Path(batch_results_path).parent.mkdir(parents=True, exist_ok=True)
        
        batch_results = {
            'metadata': {
                'data_name': data_name,
                'imbalance_method': imbalance_method,
                'timestamp': datetime.now().isoformat(),
                'total_models': len(results),
                'successful_models': len([r for r in results if 'error' not in r]),
                'failed_models': len([r for r in results if 'error' in r])
            },
            'results': results
        }
        
        import json
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2, default=str)
        
        logging.info(f"\n批量训练结果已保存到: {batch_results_path}")
        logging.info("=" * 80)

    def _generate_comparison_report(self, comparator: ModelComparator, results: List[Dict[str, Any]], task_type: str):

        """生成模型比较报告"""
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        data_name = self.config.get('paths.input_data', 'data').split('/')[-1].replace('.csv', '')
        
        try:
            # 生成比较报告
            comparison_df = comparator.generate_comparison_report(task_type)

            
            if comparison_df.empty:
                logging.warning("没有可用的模型结果进行比较")
                return
            
            # 打印比较报告
            logging.info("=" * 80)
            logging.info("模型性能比较报告")
            logging.info("=" * 80)
            logging.info(f"数据: {data_name}")
            logging.info(f"不平衡处理方法: {imbalance_method}")
            logging.info(f"模型数量: {len(comparison_df)}")
            logging.info("")
            
            # 显示排序后的比较结果
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            logging.info("\n" + str(comparison_df))
            
            # 保存比较报告
            self._save_comparison_report(comparison_df, data_name, imbalance_method)
            
        except Exception as e:
            logging.error(f"生成模型比较报告时出错: {str(e)}")
            logging.error(traceback.format_exc())

    def _save_comparison_report(self, comparison_df, data_name: str, imbalance_method: str):
        """保存模型比较报告"""
        try:
            # 生成文件名 - 使用安全的文件名格式
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = self.config.get(
                'paths.comparison_output',
                f'models/comparison_{data_name}_{imbalance_method}_{timestamp}.csv'
            ).format(data_name=data_name, imbalance_method=imbalance_method, timestamp=timestamp)
            
            # 确保目录存在
            Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存CSV格式
            comparison_df.to_csv(comparison_path, index=False)
            logging.info(f"\n模型比较报告已保存到: {comparison_path}")
            
            # 同时保存JSON格式
            json_path = comparison_path.replace('.csv', '.json')
            comparison_dict = comparison_df.to_dict('records')
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'data_name': data_name,
                        'imbalance_method': imbalance_method,
                        'timestamp': datetime.now().isoformat(),
                        'total_models': len(comparison_df)
                    },
                    'results': comparison_dict
                }, f, ensure_ascii=False, indent=2, default=str)
            
            logging.info(f"模型比较报告(JSON)已保存到: {json_path}")
            
        except Exception as e:
            logging.error(f"保存模型比较报告时出错: {str(e)}")
    
    def _save_pipeline(self, pipeline: Pipeline, model_name: str, results: Dict[str, Any] = None) -> str:
        """
        保存训练好的pipeline，并同时保存模型结果
        
        Args:
            pipeline: 训练好的pipeline
            model_name: 模型名称
            results: 模型评估结果
            
        Returns:
            保存路径
        """
        # (此部分代码保持不变)
        # 获取输入数据名称
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        
        # 从配置获取不平衡处理方法
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 生成模型文件名，添加pipeline后缀
        model_filename = self.config.get(
            'paths.pipeline_model_output', 
            'models/pipeline_{input_data}_{imbalance_method}_{model_name}.joblib'
        ).format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )
        
        # 确保目录存在
        model_path = Path(model_filename)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存pipeline
        import joblib
        joblib.dump(pipeline, model_path)
        logging.info(f"Pipeline模型已保存到: {model_path}")


        # 保存PMML模型
        # 生成pmml文件名，添加pipeline后缀
        pmml_filename = self.config.get(
            'paths.pmml_model_output', 
            'pmmls/pipeline_{input_data}_{imbalance_method}_{model_name}.pmml'
        ).format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )
        
        # 确保目录存在
        pmml_path = Path(pmml_filename)
        pmml_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn.pipeline import Pipeline as SklearnPipeline
            from imblearn.pipeline import Pipeline as ImbPipeline
            # << 新增导入 >>
            from feature_engineering import DataPreprocessor, FeatureSelectorPipeline, FeatureGeneratorPipeline

            # 确定要转换的原始pipeline（移除采样器）
            raw_pipeline_to_convert = None
            if isinstance(pipeline, ImbPipeline):
                logging.info("检测到ImbPipeline，正在创建用于PMML的纯净预测管道...")
                prediction_steps = []
                for step_name, step_transformer in pipeline.steps:
                    if step_name != 'sampler':
                        prediction_steps.append((step_name, step_transformer))
                raw_pipeline_to_convert = SklearnPipeline(steps=prediction_steps)
            else:
                raw_pipeline_to_convert = pipeline

            logging.info(f"正在将{model_name} Pipeline保存到: {pmml_path}")
            # 使用新构建的 pmml_pipeline 进行转换
            sklearn2pmml(raw_pipeline_to_convert, pmml_path, with_repr=True) 
            logging.info(f"{model_name} PMML文件已生成,路径为{pmml_path}")

        except Exception as e:
            logging.error(f"保存{model_name} PMML文件时出错: {str(e)}")
            logging.error(traceback.format_exc())

        if results is not None:
            self._save_results(model_name, results, model_path)

        return str(model_path)
    
    def _save_results(self, model_name: str, results: Dict[str, Any], model_path: str):
        """保存训练结果（优化版）"""
        # 获取输入数据名称
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem

        # 从配置获取不平衡处理方法
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        results_path = self.config.get(
            'paths.pipeline_results_output', 
            'models/pipeline_{input_data}_{imbalance_method}_{model_name}_results.json'
        ).format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )

        # 确保目录存在
        results_path_obj = Path(results_path)
        results_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用自定义JSON编码器处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                if isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)
        
        results_data = {
            'model_name': model_name,
            'model_path': model_path,
            'evaluation_results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
        
        logging.info(f"训练结果已保存到: {results_path}")
    
    def _save_final_features(self, final_features: List[str], model_name: str):
        """
        保存最终入模特征变量列表
        
        Args:
            final_features: 最终选择的特征列表
            model_name: 模型名称
        """
        # 获取输入数据名称
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem

        # 从配置获取不平衡处理方法
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 生成特征文件名
        features_path = self.config.get(
            'paths.final_features_output',
            'models/{input_data}_{imbalance_method}_{model_name}_final_features.json'
        ).format(input_data=input_data_name, imbalance_method=imbalance_method, model_name=model_name)
        
        # 确保目录存在
        features_path_obj = Path(features_path)
        features_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存特征列表
        features_data = {
            'input_data': input_data_name,
            'final_features': final_features,
            'feature_count': len(final_features),
            'timestamp': str(pd.Timestamp.now().isoformat())
        }
        
        import json
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features_data, f, ensure_ascii=False, indent=2, default=str)
        
        logging.info(f"最终入模特征变量已保存到: {features_path}")
    
    def predict(self, data_path: str = None) -> pd.DataFrame:
        """
        使用训练好的pipeline进行预测
        
        Args:
            data_path: 数据路径，如果为None使用测试数据
            
        Returns:
            预测结果DataFrame
        """
        # 从配置获取模型名称
        model_name = self.config.get('modeling.model_type', 'LR')
        
        # 加载模型
        model_path = self._get_model_path(model_name)
        if not model_path:
            raise ValueError(f"找不到{model_name}的模型文件")
        
        import joblib
        pipeline = joblib.load(model_path)
        
        # 加载数据
        if data_path:
            df = self.data_loader.load_data(data_path)
            feature_cols = self.data_loader.get_feature_columns(df)
            X = df[feature_cols]
        else:
            # 使用测试数据
            all_df = self.data_loader.load_data()
            feature_cols = self.data_loader.get_feature_columns(all_df)
            target_col = self.data_loader.get_target_column()
            
            # 从配置获取测试集分割比例和随机状态
            test_size = self.config.get('modeling.test_size', 0.2)
            random_state = self.config.get('modeling.random_state', 42)
            
            _, X_test, _, _ = train_test_split(
                all_df[feature_cols], all_df[target_col], 
                test_size=test_size, random_state=random_state, stratify=all_df[target_col]
            )
            X = X_test
        
        # 预测
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)[:, 1]
        
        # 创建结果DataFrame
        result_df = X.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = probabilities
        
        return result_df
    
    def _get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        model_template = self.config.get(
            'paths.pipeline_model_output', 
            'models/pipeline_{input_data}_{imbalance_method}_{model_name}.joblib'
        )
        model_filename = model_template.format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name
        )
        
        return str(Path(model_filename)) if Path(model_filename).exists() else None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline训练管道')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='指定要训练的模型')
    parser.add_argument('--predict', action='store_true', 
                       help='使用训练好的模型进行预测')
    
    args = parser.parse_args()
    
    trainer = PipelineTraining(args.config)
    
    if args.predict:
        results = trainer.predict()
        print(f"预测结果预览:\n{results.head()}")
    else:
            if args.model:
                # 临时设置配置中的模型类型
                trainer.config.set('modeling.model_type', args.model)
                results = trainer.run_single_model(args.model)
            else:
                results = trainer.run_all_models()
        
    print("Pipeline训练完成！")


if __name__ == "__main__":
    main()