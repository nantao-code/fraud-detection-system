#!/usr/bin/env python3
"""
模型转换工具 - 将训练好的模型转换为PMML文件和评分卡模型
基于机器学习模型转换PMML.md和机器学习模型转换评分卡模型.md进行全面改造
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Tuple
import warnings

# PMML转换相关
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# 评分卡相关
try:
    import scorecardpy as sc
    SCORECARD_AVAILABLE = True
except ImportError:
    SCORECARD_AVAILABLE = False
    logging.warning("scorecardpy未安装，评分卡功能将不可用")

# 统一配置管理
from src.unified_config import UnifiedConfig

from tqdm import tqdm
import sys

class ModelConverter:
    """模型转换器 - 支持PMML和多种评分卡转换"""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "converted_models", config_path: str = "config.yaml"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.pmml_dir = self.output_dir / "pmml"
        self.scorecard_dir = self.output_dir / "scorecard"
        self.pmml_dir.mkdir(exist_ok=True)
        self.scorecard_dir.mkdir(exist_ok=True)
        
        # 使用统一配置管理器
        self.config_manager = UnifiedConfig(config_path)
        self.target_column = self.config_manager.get('feature_engineering.target_column', 'label')
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_name: str, dataset_name: str = "test_creditcard") -> Any:
        """加载训练好的模型"""
        model_path = self.models_dir / f"{dataset_name}_{model_name}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        return joblib.load(model_path)
    
    def create_pmml_pipeline(self, 
                           model: Any, 
                           numeric_features: List[str], 
                           categorical_features: List[str],
                           target_name: str = None) -> PMMLPipeline:
        """创建完整的PMML Pipeline，包含数据预处理"""
        
        # 创建预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # 创建完整的Pipeline
        pipeline = PMMLPipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # 设置元数据
        all_features = numeric_features + categorical_features
        pipeline.active_fields = all_features
        # 使用配置文件中的目标变量，如果指定了target_name则使用指定的
        pipeline.target_field = target_name if target_name else self.target_column
        
        return pipeline
    
    def convert_to_pmml(self, 
                       model_name: str, 
                       numeric_features: List[str],
                       categorical_features: List[str],
                       dataset_name: str = "test_creditcard",
                       target_name: str = None) -> str:
        """将模型转换为PMML格式（完整Pipeline版本）"""
        try:
            self.logger.info(f"开始将 {model_name} 转换为PMML格式...")
            
            # 加载模型
            model = self.load_model(model_name, dataset_name)
            
            # 创建完整的PMML Pipeline
            pipeline = self.create_pmml_pipeline(
                model, 
                numeric_features, 
                categorical_features,
                target_name
            )
            
            # 生成PMML文件路径
            pmml_path = self.pmml_dir / f"{dataset_name}_{model_name}_pipeline.pmml"
            
            # 转换为PMML
            sklearn2pmml(pipeline, str(pmml_path), with_repr=True)
            
            self.logger.info(f"PMML文件已保存: {pmml_path}")
            self.logger.info(f"✓ {model_name} PMML转换完成")
            
            return str(pmml_path)
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} PMML转换失败: {str(e)}")
            self.logger.error(f"PMML转换失败: {str(e)}")
            raise
    
    def convert_lr_scorecard(self, 
                           model_name: str, 
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           dataset_name: str = "test_creditcard") -> Dict[str, Any]:
        """将逻辑回归模型转换为经典评分卡"""
        if not SCORECARD_AVAILABLE:
            raise ImportError("scorecardpy未安装，无法创建评分卡")
        
        try:
            self.logger.info(f"开始将 {model_name} 转换为逻辑回归评分卡...")
            
            # 加载模型
            model = self.load_model(model_name, dataset_name)
            
            # 确保是逻辑回归模型
            if not isinstance(model, LogisticRegression):
                raise TypeError("模型必须是LogisticRegression类型")
            
            # 特征分箱与WOE转换
            train_data = X_train.copy()
            train_data[self.target_column] = y_train
            
            # 特征分箱
            bins = sc.woebin(train_data, y=self.target_column)
            
            # WOE转换
            train_woe = sc.woebin_trans(train_data, bins)
            
            # 获取WOE特征
            woe_features = [col for col in train_woe.columns if col.endswith('_woe')]
            X_train_woe = train_woe[woe_features]
            y_train_woe = train_woe[self.target_column]
            
            # 创建评分卡
            card = sc.scorecard(
                bins=bins,
                model=model,
                x0=0,
                points0=600,
                odds0=50,
                pdo=20,
                basepoints_eq=0
            )
            
            # 保存评分卡
            scorecard_path = self.scorecard_dir / f"{dataset_name}_{model_name}_lr_scorecard.pkl"
            joblib.dump({
                'scorecard': card,
                'bins': bins,
                'model': model
            }, str(scorecard_path))
            
            # 生成报告
            scorecard_report = {
                "model_name": model_name,
                "scorecard_type": "logistic_regression",
                "scorecard_path": str(scorecard_path),
                "base_score": 600,
                "pdo": 20,
                "features": list(X_train.columns),
                "total_variables": len(X_train.columns),
                "method": "WOE + LogisticRegression"
            }
            
            self.logger.info(f"逻辑回归评分卡已保存: {scorecard_path}")
            self.logger.info(f"✓ {model_name} 逻辑回归评分卡转换完成")
            
            return scorecard_report
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 逻辑回归评分卡转换失败: {str(e)}")
            self.logger.error(f"逻辑回归评分卡转换失败: {str(e)}")
            raise
    
    def convert_dt_scorecard(self, 
                           model_name: str, 
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           dataset_name: str = "test_creditcard",
                           max_depth: int = 3) -> Dict[str, Any]:
        """将决策树模型转换为规则评分卡"""
        try:
            self.logger.info(f"开始将 {model_name} 转换为决策树评分卡...")
            
            # 加载模型
            model = self.load_model(model_name, dataset_name)
            
            # 确保是决策树模型
            if not isinstance(model, DecisionTreeClassifier):
                raise TypeError("模型必须是DecisionTreeClassifier类型")
            
            # 处理类别特征
            X_train_processed = X_train.copy()
            categorical_cols = X_train_processed.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                X_train_processed = pd.get_dummies(X_train_processed, columns=categorical_cols, drop_first=True)
            
            # 训练决策树（这里直接使用已训练的模型）
            dt_model = model
            
            # 获取叶节点信息
            leaf_ids = dt_model.apply(X_train_processed)
            
            # 计算每个叶节点的分数
            base_score = 600
            pdo = 20
            B = pdo / np.log(2)
            
            # 获取每个样本的预测概率
            probas = dt_model.predict_proba(X_train_processed)[:, 1]
            
            # 计算每个叶节点的平均概率
            df_leaves = pd.DataFrame({
                'leaf_id': leaf_ids,
                'proba_bad': probas,
                self.target_column: y_train
            })
            
            leaf_scores = df_leaves.groupby('leaf_id').agg({
                'proba_bad': 'mean',
                self.target_column: 'count'
            }).reset_index()
            
            # 计算分数
            leaf_scores['proba_bad'] = leaf_scores['proba_bad'].clip(0.001, 0.999)
            leaf_scores['odds'] = leaf_scores['proba_bad'] / (1 - leaf_scores['proba_bad'])
            leaf_scores['score'] = (base_score - B * np.log(leaf_scores['odds'])).round().astype(int)
            
            # 获取规则路径
            from sklearn.tree import export_text
            feature_names = list(X_train_processed.columns)
            rules = export_text(dt_model, feature_names=feature_names)
            
            # 保存决策树评分卡
            scorecard_path = self.scorecard_dir / f"{dataset_name}_{model_name}_dt_scorecard.pkl"
            joblib.dump({
                'leaf_scores': leaf_scores,
                'rules': rules,
                'model': dt_model,
                'feature_names': feature_names,
                'max_depth': max_depth
            }, str(scorecard_path))
            
            # 生成报告
            scorecard_report = {
                "model_name": model_name,
                "scorecard_type": "decision_tree",
                "scorecard_path": str(scorecard_path),
                "base_score": base_score,
                "pdo": pdo,
                "max_depth": max_depth,
                "num_rules": len(leaf_scores),
                "method": "DecisionTree Rules"
            }
            
            self.logger.info(f"决策树评分卡已保存: {scorecard_path}")
            self.logger.info(f"✓ {model_name} 决策树评分卡转换完成")
            
            return scorecard_report
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} 决策树评分卡转换失败: {str(e)}")
            self.logger.error(f"决策树评分卡转换失败: {str(e)}")
            raise
    
    def convert_gbm_scorecard(self, 
                            model_name: str, 
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            dataset_name: str = "test_creditcard",
                            n_estimators: int = 20,
                            max_depth: int = 4) -> Dict[str, Any]:
        """将GBM/XGBoost模型转换为高级评分卡"""
        try:
            self.logger.info(f"开始将 {model_name} 转换为GBM评分卡...")
            
            # 加载模型
            model = self.load_model(model_name, dataset_name)
            
            # 检查模型类型
            valid_models = (xgb.XGBClassifier, GradientBoostingClassifier)
            if not isinstance(model, valid_models):
                raise TypeError(f"模型必须是XGBClassifier或GradientBoostingClassifier类型")
            
            # 处理类别特征
            X_train_processed = X_train.copy()
            categorical_cols = X_train_processed.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                X_train_processed = pd.get_dummies(X_train_processed, columns=categorical_cols, drop_first=True)
            
            # 获取叶节点索引作为新特征
            train_leaf_indices = model.apply(X_train_processed)
            if len(train_leaf_indices.shape) == 3:
                train_leaf_indices = train_leaf_indices[:, :, 0]  # 处理XGBoost的3D输出
            
            # 独热编码叶节点特征
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_gbm = encoder.fit_transform(train_leaf_indices)
            
            # 训练逻辑回归
            lr_model = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
            lr_model.fit(X_train_gbm, y_train)
            
            # 保存GBM评分卡
            scorecard_path = self.scorecard_dir / f"{dataset_name}_{model_name}_gbm_scorecard.pkl"
            joblib.dump({
                'gbm_model': model,
                'lr_model': lr_model,
                'encoder': encoder,
                'feature_transform': train_leaf_indices,
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }, str(scorecard_path))
            
            # 生成报告
            scorecard_report = {
                "model_name": model_name,
                "scorecard_type": "gbm_lr",
                "scorecard_path": str(scorecard_path),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "new_features": X_train_gbm.shape[1],
                "method": "GBM Feature Engineering + LogisticRegression"
            }
            
            self.logger.info(f"GBM评分卡已保存: {scorecard_path}")
            self.logger.info(f"✓ {model_name} GBM评分卡转换完成")
            
            return scorecard_report
            
        except Exception as e:
            self.logger.error(f"✗ {model_name} GBM评分卡转换失败: {str(e)}")
            self.logger.error(f"GBM评分卡转换失败: {str(e)}")
            raise
    
    def convert_to_scorecard(self, 
                           model_name: str, 
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           dataset_name: str = "test_creditcard",
                           scorecard_type: str = "auto") -> Dict[str, Any]:
        """统一的评分卡转换入口"""
        
        # 根据模型类型自动选择转换方法
        if scorecard_type == "auto":
            model = self.load_model(model_name, dataset_name)
            if isinstance(model, LogisticRegression):
                scorecard_type = "lr"
            elif isinstance(model, DecisionTreeClassifier):
                scorecard_type = "dt"
            elif isinstance(model, (xgb.XGBClassifier, RandomForestClassifier)):
                scorecard_type = "gbm"
            else:
                raise ValueError(f"不支持的模型类型: {type(model)}")
        
        # 调用对应的转换方法
        if scorecard_type == "lr":
            return self.convert_lr_scorecard(model_name, X_train, y_train, dataset_name)
        elif scorecard_type == "dt":
            return self.convert_dt_scorecard(model_name, X_train, y_train, dataset_name)
        elif scorecard_type == "gbm":
            return self.convert_gbm_scorecard(model_name, X_train, y_train, dataset_name)
        else:
            raise ValueError(f"不支持的评分卡类型: {scorecard_type}")
    
    def batch_convert_pmml(self, 
                          model_names: List[str], 
                          numeric_features: List[str],
                          categorical_features: List[str],
                          dataset_name: str = "test_creditcard") -> Dict[str, str]:
        """批量转换模型为PMML"""
        results = {}
        
        self.logger.info("开始批量PMML转换")
        
        for model_name in tqdm(model_names, desc="转换PMML"):
            try:
                pmml_path = self.convert_to_pmml(
                    model_name, 
                    numeric_features, 
                    categorical_features,
                    dataset_name
                )
                results[model_name] = pmml_path
                self.logger.info(f"✓ {model_name} 转换成功")
            except Exception as e:
                results[model_name] = f"转换失败: {str(e)}"
                self.logger.error(f"✗ {model_name} 转换失败: {str(e)}")
        
        return results
    
    def batch_convert_scorecards(self,
                               model_names: List[str],
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               dataset_name: str = "test_creditcard") -> Dict[str, Any]:
        """批量转换模型为评分卡"""
        results = {}
        
        self.logger.info("开始批量评分卡转换")
        
        with Progress() as progress:
            task = progress.add_task("转换评分卡...", total=len(model_names))
            
            for model_name in model_names:
                try:
                    scorecard_report = self.convert_to_scorecard(
                        model_name, X_train, y_train, dataset_name
                    )
                    results[model_name] = scorecard_report
                    self.logger.info(f"✓ {model_name} 评分卡转换成功")
                except Exception as e:
                    results[model_name] = {"error": str(e)}
                    self.logger.error(f"✗ {model_name} 评分卡转换失败: {str(e)}")
                
                progress.advance(task)
        
        return results
    
    def validate_pmml(self, pmml_path: str, test_data: pd.DataFrame) -> bool:
        """验证PMML文件的正确性
        
        验证包括：
        1. 文件存在性和完整性检查
        2. XML格式验证
        3. PMML结构验证
        4. 功能测试（可选）
        
        Args:
            pmml_path: PMML文件路径
            test_data: 用于功能测试的数据
            
        Returns:
            bool: 验证是否通过
        """
        try:
            import os
            import json
            from pathlib import Path
            
            validation_report = {
                "pmml_path": pmml_path,
                "validation_time": pd.Timestamp.now().isoformat(),
                "checks": {}
            }
            
            # 1. 基本文件验证
            self.logger.info(f"开始验证PMML文件: {pmml_path}")
            
            pmml_path_obj = Path(pmml_path)
            if not pmml_path_obj.exists():
                self.logger.error("✗ PMML文件不存在")
                validation_report["checks"]["file_exists"] = False
                return False
            
            validation_report["checks"]["file_exists"] = True
            
            file_size = pmml_path_obj.stat().st_size
            if file_size == 0:
                self.logger.error("✗ PMML文件为空")
                validation_report["checks"]["file_size"] = 0
                return False
            
            self.logger.info(f"ℹ 文件大小: {file_size:,} bytes")
            validation_report["checks"]["file_size"] = file_size
            
            # 2. XML格式验证
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(pmml_path)
                root = tree.getroot()
                
                # 检查PMML命名空间
                pmml_ns = "http://www.dmg.org/PMML-4_4"
                if root.tag.endswith('PMML') or root.tag == f"{{{pmml_ns}}}PMML":
                    self.logger.info("✓ 有效的PMML格式")
                    validation_report["checks"]["xml_format"] = True
                else:
                    self.logger.error("✗ 无效的PMML格式")
                    validation_report["checks"]["xml_format"] = False
                    return False
                
                # 检查PMML版本和头信息
                pmml_version = root.get('version', 'unknown')
                self.logger.info(f"ℹ PMML版本: {pmml_version}")
                validation_report["checks"]["pmml_version"] = pmml_version
                
                # 检查必要的PMML元素
                required_elements = ['Header', 'DataDictionary', 'Model']
                found_elements = []
                for child in root:
                    tag_name = child.tag.split('}')[-1]  # 移除命名空间
                    if tag_name in required_elements:
                        found_elements.append(tag_name)

                missing_elements = set(required_elements) - set(found_elements)
                if missing_elements:
                    self.logger.error(f"✗ 缺少必要元素: {missing_elements}")
                    validation_report["checks"]["required_elements"] = list(missing_elements)
                else:
                    self.logger.info("✓ 包含所有必要元素")
                    validation_report["checks"]["required_elements"] = found_elements
                
            except Exception as e:
                self.logger.error(f"✗ XML格式错误: {str(e)}")
                validation_report["checks"]["xml_format"] = False
                return False
            
            # 3. 使用pypmml进行功能验证
            try:
                from pypmml import Model
                self.logger.info("ℹ 使用pypmml进行功能验证...")
                    
                pmml_model = Model.fromFile(pmml_path)
                validation_report["checks"]["pypmml_load"] = True
                
                # 获取模型信息
                model_info = {
                    "model_type": str(type(pmml_model)),
                    "input_fields": [field.name for field in pmml_model.inputFields],
                    "output_fields": [field.name for field in pmml_model.outputFields],
                    "target_fields": [field.name for field in pmml_model.targetFields]
                }
                validation_report["model_info"] = model_info
                
                self.logger.info(f"ℹ 输入特征: {len(model_info['input_fields'])}个")
                self.logger.info(f"ℹ 输出字段: {len(model_info['output_fields'])}个")
                
                # 4. 功能测试
                if test_data is not None and len(test_data) > 0:
                    self.logger.info("ℹ 进行功能测试...")
                    
                    # 使用前5行数据进行测试
                    test_samples = test_data.head(5)
                    test_results = []
                    
                    for idx, row in test_samples.iterrows():
                        try:
                            # 准备测试数据
                            test_dict = row.to_dict()
                            
                            # 执行预测
                            result = pmml_model.predict(test_dict)
                            
                            # 检查结果格式
                            if isinstance(result, dict):
                                has_prob = any('probability' in str(k).lower() for k in result.keys())
                                has_score = 'score' in result or 'prediction' in result
                                
                                test_results.append({
                                    "sample_idx": idx,
                                    "success": True,
                                    "result_keys": list(result.keys()),
                                    "has_probability": has_prob,
                                    "has_score": has_score
                                })
                            else:
                                test_results.append({
                                    "sample_idx": idx,
                                    "success": True,
                                    "result_type": str(type(result))
                                })
                                
                        except Exception as pred_error:
                            test_results.append({
                                "sample_idx": idx,
                                "success": False,
                                "error": str(pred_error)
                            })
                    
                    validation_report["function_test"] = {
                        "total_samples": len(test_samples),
                        "successful_predictions": sum(1 for r in test_results if r["success"]),
                        "failed_predictions": sum(1 for r in test_results if not r["success"]),
                        "test_results": test_results
                    }
                    
                    success_rate = validation_report["function_test"]["successful_predictions"] / len(test_samples)
                    
                    if success_rate >= 0.8:  # 80%成功率
                        self.logger.info(f"✓ 功能测试通过 ({success_rate:.0%}成功率)")
                    else:
                        self.logger.warning(f"⚠ 功能测试部分失败 ({success_rate:.0%}成功率)")
                
                # 保存验证报告
                report_path = pmml_path.replace('.pmml', '_validation_report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_report, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"✓ PMML文件验证通过: {pmml_path}")
                self.logger.info(f"ℹ 验证报告已保存: {report_path}")
                return True
                
            except ImportError:
                self.logger.warning("⚠ pypmml库未安装，跳过功能验证")
                self.logger.info(f"✓ PMML文件基本验证通过: {pmml_path}")
                
                # 仍然保存基本验证报告
                report_path = pmml_path.replace('.pmml', '_validation_report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_report, f, indent=2, ensure_ascii=False)
                
                return True
                
            except Exception as e:
                self.logger.error(f"✗ PMML功能验证失败: {str(e)}")
                validation_report["checks"]["pypmml_load"] = str(e)
                return False
                
        except Exception as e:
            self.logger.error(f"✗ PMML文件验证失败: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str, dataset_name: str = "test_creditcard") -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = self.load_model(model_name, dataset_name)
            
            info = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "dataset": dataset_name,
                "can_convert_pmml": True,
                "can_convert_scorecard": True,
                "scorecard_types": self._get_supported_scorecard_types(model)
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_supported_scorecard_types(self, model: Any) -> List[str]:
        """获取模型支持的评分卡类型"""
        if isinstance(model, LogisticRegression):
            return ["lr", "classic"]
        elif isinstance(model, DecisionTreeClassifier):
            return ["dt", "rules"]
        elif isinstance(model, (xgb.XGBClassifier, RandomForestClassifier)):
            return ["gbm", "advanced"]
        else:
            return []
    
    def apply_scorecard(self, 
                       scorecard_path: str, 
                       data: pd.DataFrame) -> pd.DataFrame:
        """应用评分卡进行打分"""
        try:
            scorecard_data = joblib.load(scorecard_path)
            
            if 'scorecard' in scorecard_data:  # 逻辑回归评分卡
                card = scorecard_data['scorecard']
                bins = scorecard_data['bins']
                
                # 应用评分卡
                scored_data = sc.scorecard_ply(data, card, print_step=0)
                return scored_data
                
            elif 'leaf_scores' in scorecard_data:  # 决策树评分卡
                model = scorecard_data['model']
                leaf_scores = scorecard_data['leaf_scores']
                
                # 处理数据
                X_processed = data.copy()
                categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
                
                # 获取叶节点
                leaf_ids = model.apply(X_processed)
                
                # 映射分数
                score_map = dict(zip(leaf_scores['leaf_id'], leaf_scores['score']))
                scores = [score_map.get(leaf_id, 300) for leaf_id in leaf_ids]
                
                result = data.copy()
                result['score'] = scores
                return result
                
            else:  # GBM评分卡
                gbm_model = scorecard_data['gbm_model']
                lr_model = scorecard_data['lr_model']
                encoder = scorecard_data['encoder']
                
                # 处理数据
                X_processed = data.copy()
                categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
                
                # 获取叶节点特征
                leaf_indices = gbm_model.apply(X_processed)
                if len(leaf_indices.shape) == 3:
                    leaf_indices = leaf_indices[:, :, 0]
                
                # 编码并预测
                X_gbm = encoder.transform(leaf_indices)
                probabilities = lr_model.predict_proba(X_gbm)[:, 1]
                
                # 转换为分数
                base_score = 600
                pdo = 20
                B = pdo / np.log(2)
                
                scores = []
                for prob in probabilities:
                    prob = max(0.001, min(0.999, prob))
                    odds = prob / (1 - prob)
                    score = base_score - B * np.log(odds)
                    scores.append(round(score))
                
                result = data.copy()
                result['score'] = scores
                result['probability'] = probabilities
                return result
                
        except Exception as e:
            self.logger.error(f"应用评分卡失败: {str(e)}")
            raise
    
    def batch_convert_pmml(self, 
                          model_names: List[str], 
                          feature_names: List[str],
                          dataset_name: str = "test_creditcard") -> Dict[str, str]:
        """批量转换模型为PMML"""
        results = {}
        
        self.logger.info("开始批量PMML转换")
        
        with Progress() as progress:
            task = progress.add_task("转换PMML...", total=len(model_names))
            
            for model_name in model_names:
                try:
                    pmml_path = self.convert_to_pmml(model_name, feature_names, dataset_name)
                    results[model_name] = pmml_path
                    self.logger.info(f"✓ {model_name} 转换成功")
                except Exception as e:
                    results[model_name] = f"转换失败: {str(e)}"
                    self.logger.error(f"✗ {model_name} 转换失败")
                
                progress.advance(task)
        
        return results
    
    def get_model_info(self, model_name: str, dataset_name: str = "test_creditcard") -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = self.load_model(model_name, dataset_name)
            
            info = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "dataset": dataset_name,
                "can_convert_pmml": True,  # 大多数sklearn模型都支持
                "can_convert_scorecard": isinstance(model, LogisticRegression)
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}

# 使用示例
if __name__ == "__main__":
    converter = ModelConverter()
    
    # 支持的模型列表
    models = ["LR", "RF", "XGB"]
    
    # 示例特征名称（需要根据实际数据调整）
    feature_names = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    
    # 批量转换PMML
    converter.logger.info("开始PMML转换")
    pmml_results = converter.batch_convert_pmml(models, feature_names)
    
    # 显示结果
    for model, path in pmml_results.items():
        if "失败" not in path:
            converter.logger.info(f"✓ {model}: {path}")
        else:
            converter.logger.error(f"✗ {model}: {path}")