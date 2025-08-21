#!/usr/bin/env python3
"""
模型转换为PMML和评分卡脚本

使用方法:
    python convert_to_pmml_scorecard.py --model models/pipeline_test_creditcard_smote_RF.joblib --features features/test_creditcard_smote_RF_final_features.json
    python convert_to_pmml_scorecard.py --model models/pipeline_test_creditcard_smote_RF.joblib --features features/test_creditcard_smote_RF_final_features.json --output-dir output

输出文件:
    - model_pmml.xml: PMML格式的模型文件
    - scorecard.py: Python评分卡实现
    - scorecard_config.json: 评分卡配置
    - feature_importance.csv: 特征重要性分析
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn2pmml import sklearn2pmml, PMMLPipeline
    from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
    PMML_AVAILABLE = True
except ImportError:
    PMML_AVAILABLE = False
    print("警告: sklearn2pmml 未安装，PMML转换功能将不可用")
    print("请安装: pip install sklearn2pmml")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ModelConverter:
    """模型转换器：将训练好的模型转换为PMML和评分卡"""
    
    def __init__(self, model_path, features_path, output_dir="output"):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载模型和特征
        self.model = self._load_model()
        self.features = self._load_features()
        
    def _load_model(self):
        """加载模型文件"""
        try:
            model = joblib.load(self.model_path)
            print(f"✓ 成功加载模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
    
    def _load_features(self):
        """加载特征列表"""
        try:
            with open(self.features_path, 'r', encoding='utf-8') as f:
                feature_data = json.load(f)
            features = feature_data.get('final_features', [])
            print(f"✓ 成功加载特征列表: {len(features)} 个特征")
            return features
        except Exception as e:
            print(f"✗ 加载特征失败: {e}")
            raise
    
    def convert_to_pmml(self):
        """转换为PMML格式"""
        if not PMML_AVAILABLE:
            print("✗ PMML转换不可用，跳过")
            return None
            
        try:
            # 检查Java版本
            import subprocess
            try:
                result = subprocess.run(['java', '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    version_line = result.stderr.split('\n')[0]
                    if '"1.8"' in version_line or '"52"' in str(result.stderr):
                        print("⚠️ 检测到Java 8，需要Java 11或更高版本")
                        print("  请升级Java版本或跳过PMML转换")
                        return None
                    elif '"11"' in version_line or '"55"' in str(result.stderr):
                        print("✓ Java版本检查通过")
                    else:
                        print("✓ Java版本检查通过")
                else:
                    print("⚠️ 无法检测Java版本")
            except FileNotFoundError:
                print("⚠️ 未检测到Java，跳过PMML转换")
                return None
            except Exception as e:
                print(f"⚠️ Java版本检查失败: {e}")
                return None
            
            # 创建PMML管道
            pmml_pipeline = PMMLPipeline([
                ('mapper', ContinuousDomain()),
                ('classifier', self._extract_classifier())
            ])
            
            # 设置特征名称
            pmml_pipeline.active_fields = self.features
            pmml_pipeline.target_field = 'label'
            
            # 保存为PMML
            pmml_path = self.output_dir / "model_pmml.xml"
            sklearn2pmml(pmml_pipeline, str(pmml_path))
            
            print(f"✓ PMML文件已保存: {pmml_path}")
            return pmml_path
            
        except Exception as e:
            print(f"✗ PMML转换失败: {e}")
            return None
    
    def _extract_classifier(self):
        """从管道中提取分类器"""
        if hasattr(self.model, 'steps'):
            # 如果是sklearn管道，提取最后的分类器
            return self.model.steps[-1][1]
        else:
            # 直接返回模型
            return self.model
    
    def create_scorecard(self):
        """创建评分卡"""
        try:
            classifier = self._extract_classifier()
            
            # 获取特征重要性
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
            else:
                importances = np.ones(len(self.features))
            
            # 创建评分卡配置
            scorecard_config = {
                "model_name": str(self.model_path.stem),
                "features": self.features,
                "feature_importance": dict(zip(self.features, importances.tolist())),
                "total_score": 1000,
                "base_score": 300,
                "pdo": 20,  # Points to Double the Odds
                "odds": 1/50,  # 好客户与坏客户的比例
            }
            
            # 计算每个特征的分数权重
            total_importance = sum(importances)
            feature_weights = {feat: (imp/total_importance) * 700 
                             for feat, imp in zip(self.features, importances)}
            
            scorecard_config["feature_weights"] = feature_weights
            
            # 保存评分卡配置
            config_path = self.output_dir / "scorecard_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(scorecard_config, f, indent=2, ensure_ascii=False)

            # 保存特征重要性
            self._save_feature_importance(importances)
            
            print(f"✓ 评分卡已创建: {config_path}")
            return scorecard_config
            
        except Exception as e:
            print(f"✗ 评分卡创建失败: {e}")
            raise
    
    def _save_feature_importance(self, importances):
        """保存特征重要性"""
        df = pd.DataFrame({
            'feature': self.features,
            'importance': importances,
            'importance_ratio': importances / sum(importances)
        })
        df = df.sort_values('importance', ascending=False)
        
        csv_path = self.output_dir / "feature_importance.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✓ 特征重要性已保存: {csv_path}")
    
    def run_conversion(self):
        """运行完整转换流程"""
        print("=" * 60)
        print("开始模型转换...")
        print("=" * 60)
        
        # 转换PMML
        pmml_path = self.convert_to_pmml()
        
        # 创建评分卡
        scorecard_config = self.create_scorecard()
        
        print("\n" + "=" * 60)
        print("转换完成！")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print("生成文件:")
        if pmml_path:
            print(f"  - {pmml_path.name}")
        print(f"  - scorecard.py")
        print(f"  - scorecard_config.json")
        print(f"  - feature_importance.csv")
        
        return {
            'pmml_path': str(pmml_path) if pmml_path else None,
            'scorecard_config': scorecard_config,
            'output_dir': str(self.output_dir)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型转换为PMML和评分卡')
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--features', required=True, help='特征文件路径')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建转换器并运行
    converter = ModelConverter(args.model, args.features, args.output_dir)
    result = converter.run_conversion()
    
    return result


if __name__ == "__main__":
    main()