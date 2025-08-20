#!/usr/bin/env python3
"""
简化版评分卡生成器

使用方法:
    python simple_scorecard.py --model models/pipeline_test_creditcard_smote_RF.joblib --features features/test_creditcard_smote_RF_final_features.json
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleScorecardGenerator:
    """简化版评分卡生成器"""
    
    def __init__(self, model_path, features_path):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        
        # 加载模型和特征
        self.model = joblib.load(self.model_path)
        self.features = self._load_features()
        
    def _load_features(self):
        """加载特征列表"""
        with open(self.features_path, 'r', encoding='utf-8') as f:
            feature_data = json.load(f)
        return feature_data.get('final_features', [])
    
    def generate_scorecard(self):
        """生成评分卡"""
        # 提取分类器
        if hasattr(self.model, 'steps'):
            classifier = self.model.steps[-1][1]
        else:
            classifier = self.model
        
        # 获取特征重要性
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = np.ones(len(self.features)) / len(self.features)
        
        # 创建评分卡配置
        scorecard = {
            "model_info": {
                "model_name": str(self.model_path.stem),
                "model_type": type(classifier).__name__,
                "feature_count": len(self.features),
                "total_score": 1000,
                "base_score": 300,
                "pdo": 20,  # Points to Double the Odds
                "good_bad_ratio": 50
            },
            "features": [],
            "score_mapping": {}
        }
        
        # 为每个特征创建评分规则
        total_importance = sum(importances)
        
        for i, (feature, importance) in enumerate(zip(self.features, importances)):
            weight = (importance / total_importance) * 700  # 700分给特征
            
            feature_config = {
                "name": feature,
                "importance": float(importance),
                "weight": float(weight),
                "type": "continuous",
                "bins": [
                    {"min": -float('inf'), "max": -1.0, "score": -weight * 0.5},
                    {"min": -1.0, "max": -0.5, "score": -weight * 0.3},
                    {"min": -0.5, "max": 0.0, "score": 0},
                    {"min": 0.0, "max": 0.5, "score": weight * 0.3},
                    {"min": 0.5, "max": 1.0, "score": weight * 0.5},
                    {"min": 1.0, "max": float('inf'), "score": weight * 0.7}
                ]
            }
            
            scorecard["features"].append(feature_config)
            scorecard["score_mapping"][feature] = feature_config["bins"]
        
        # 保存评分卡
        output_file = Path("scorecard_config_simple.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scorecard, f, indent=2, ensure_ascii=False)
        
        # 创建评分卡类
        self._create_scorecard_class(scorecard)
        
        # 保存特征重要性
        self._save_feature_importance(importances)
        
        print(f"✓ 评分卡已保存到: {output_file}")
        return scorecard
    
    def _create_scorecard_class(self, scorecard):
        """创建评分卡类"""
        class_code = '''
import json
import numpy as np
from pathlib import Path

class CreditScorecard:
    """信用评分卡实现"""
    
    def __init__(self, config_path="scorecard_config_simple.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.base_score = self.config["model_info"]["base_score"]
        self.features = [f["name"] for f in self.config["features"]]
    
    def calculate_score(self, data):
        """计算信用分数"""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k in self.features}
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_dict()
        
        total_score = self.base_score
        
        for feature_config in self.config["features"]:
            feature_name = feature_config["name"]
            value = data.get(feature_name, 0)
            
            # 查找对应的分数
            score = 0
            for bin_config in feature_config["bins"]:
                if bin_config["min"] <= value < bin_config["max"]:
                    score = bin_config["score"]
                    break
            
            total_score += score
        
        return max(0, min(1000, int(total_score)))
    
    def predict_proba(self, data):
        """预测违约概率"""
        score = self.calculate_score(data)
        
        # 简化的概率计算
        odds = np.exp((score - 300) / 20 * np.log(2))
        probability = 1 / (1 + 1/odds)
        
        return {
            'score': score,
            'probability': probability,
            'risk_level': self._get_risk_level(score)
        }
    
    def _get_risk_level(self, score):
        """获取风险等级"""
        if score >= 800:
            return "优秀"
        elif score >= 700:
            return "良好"
        elif score >= 600:
            return "一般"
        elif score >= 500:
            return "风险"
        else:
            return "高风险"
    
    def get_feature_scores(self, data):
        """获取各特征分数"""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k in self.features}
        
        feature_scores = {}
        total_score = self.base_score
        
        for feature_config in self.config["features"]:
            feature_name = feature_config["name"]
            value = data.get(feature_name, 0)
            
            score = 0
            for bin_config in feature_config["bins"]:
                if bin_config["min"] <= value < bin_config["max"]:
                    score = bin_config["score"]
                    break
            
            feature_scores[feature_name] = {
                'value': value,
                'score': score,
                'contribution': score
            }
            total_score += score
        
        return feature_scores, total_score

# 使用示例
if __name__ == "__main__":
    scorecard = CreditScorecard()
    
    # 示例数据
    sample_data = {''' + str({f: 0.5 for f in self.features[:3]}) + '''}
    
    result = scorecard.predict_proba(sample_data)
    print(f"信用分数: {result['score']}")
    print(f"违约概率: {result['probability']:.2%}")
    print(f"风险等级: {result['risk_level']}")
    
    feature_scores, total = scorecard.get_feature_scores(sample_data)
    print("各特征分数:")
    for feat, score_info in feature_scores.items():
        print(f"  {feat}: {score_info['score']:.1f} 分 (值: {score_info['value']:.3f})")
'''
        
        # 保存评分卡类
        with open("scorecard_simple.py", 'w', encoding='utf-8') as f:
            f.write(class_code)
        
        print("✓ 评分卡类已保存到: scorecard_simple.py")
    
    def _save_feature_importance(self, importances):
        """保存特征重要性"""
        df = pd.DataFrame({
            'feature': self.features,
            'importance': importances,
            'importance_ratio': importances / sum(importances),
            'score_weight': (importances / sum(importances)) * 700
        })
        df = df.sort_values('importance', ascending=False)
        
        df.to_csv("feature_importance_simple.csv", index=False, encoding='utf-8')
        print("✓ 特征重要性已保存到: feature_importance_simple.csv")
    
    def run(self):
        """运行评分卡生成"""
        print("=" * 50)
        print("开始生成评分卡...")
        print("=" * 50)
        
        scorecard = self.generate_scorecard()
        
        print("\n" + "=" * 50)
        print("评分卡生成完成！")
        print("=" * 50)
        print("生成文件:")
        print("  - scorecard_config_simple.json")
        print("  - scorecard_simple.py")
        print("  - feature_importance_simple.csv")
        
        return scorecard


def main():
    parser = argparse.ArgumentParser(description='生成简化版评分卡')
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--features', required=True, help='特征文件路径')
    
    args = parser.parse_args()
    
    generator = SimpleScorecardGenerator(args.model, args.features)
    generator.run()


if __name__ == "__main__":
    main()