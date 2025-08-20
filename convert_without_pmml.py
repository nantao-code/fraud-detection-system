#!/usr/bin/env python3
"""
模型转换为评分卡（无PMML依赖版本）

这个脚本专门解决Java版本不兼容问题，直接创建评分卡而不使用PMML
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleScorecardConverter:
    """简化版评分卡转换器"""
    
    def __init__(self, model_path, features_path, output_dir="output"):
        """初始化转换器"""
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载模型和特征
        self.model = self._load_model()
        self.features = self._load_features()
        
    def _load_model(self):
        """加载模型"""
        try:
            model = joblib.load(self.model_path)
            print(f"✓ 成功加载模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
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
            print(f"✗ 特征文件加载失败: {e}")
            raise
    
    def _get_feature_importance(self):
        """获取特征重要性"""
        try:
            # 尝试从模型获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # 对于线性模型，使用系数绝对值
                importances = np.abs(self.model.coef_[0])
            else:
                # 默认均匀分布
                importances = np.ones(len(self.features)) / len(self.features)
            
            return importances
        except Exception as e:
            print(f"⚠️ 无法获取特征重要性，使用默认值: {e}")
            return np.ones(len(self.features)) / len(self.features)
    
    def _create_scorecard_config(self):
        """创建评分卡配置"""
        importances = self._get_feature_importance()
        
        # 标准化重要性到权重
        total_importance = np.sum(importances)
        weights = (importances / total_importance) * 100  # 总权重100分
        
        # 评分卡配置
        config = {
            "model_name": str(self.model_path.stem),
            "features": self.features,
            "feature_weights": {feat: float(weight) for feat, weight in zip(self.features, weights)},
            "base_score": 600,  # 基础分数
            "score_range": [300, 850],
            "pdo": 20,  # 点数翻倍值
            "odds_at": 600,
            "odds": 50,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # 保存配置
        config_path = self.output_dir / "scorecard_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 评分卡配置已保存: {config_path}")
        return config
    
    def _create_scorecard_class(self, config):
        """创建评分卡类"""
        features_str = json.dumps(config["features"])
        weights_str = json.dumps(config["feature_weights"])
        
        scorecard_code = f'''"""
信用评分卡 - {config["model_name"]}

这个评分卡基于机器学习模型转换而来，无需PMML依赖
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class CreditScorecard:
    """信用评分卡实现"""
    
    def __init__(self, config_path="scorecard_config.json"):
        """初始化评分卡"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # 使用默认配置
            self.config = {json.dumps(config, indent=2, ensure_ascii=False)}
        
        self.features = self.config["features"]
        self.feature_weights = self.config["feature_weights"]
        self.base_score = self.config["base_score"]
        self.pdo = self.config["pdo"]
        
    def calculate_score(self, data):
        """计算信用分数"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 检查必需特征
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"缺少特征: {{missing_features}}")
        
        # 计算每个特征的分数
        total_score = float(self.base_score)
        
        for feature in self.features:
            if feature in data.columns:
                value = data[feature].iloc[0] if hasattr(data, 'iloc') else data[feature]
                
                # 标准化特征值到合理范围
                normalized_value = self._normalize_feature(value, feature)
                
                # 计算该特征的分数贡献
                feature_score = normalized_value * self.feature_weights[feature]
                total_score += feature_score
        
        # 限制分数范围
        min_score, max_score = self.config["score_range"]
        return max(min_score, min(max_score, int(total_score)))
    
    def _normalize_feature(self, value, feature):
        """标准化特征值到0-100分"""
        # 根据特征的实际分布进行调整
        # 这里使用经验值，实际应根据训练数据调整
        
        # 假设大部分特征值在[-3, 3]之间
        clipped_value = max(-3, min(3, float(value)))
        normalized = (clipped_value + 3) * 100 / 6  # 映射到[0, 100]
        return normalized
    
    def get_risk_level(self, score):
        """根据分数返回风险等级"""
        if score >= 750:
            return "极低风险"
        elif score >= 700:
            return "低风险"
        elif score >= 650:
            return "中等风险"
        elif score >= 600:
            return "中高风险"
        elif score >= 550:
            return "高风险"
        else:
            return "极高风险"
    
    def predict_proba(self, data):
        """预测违约概率
        
        返回:
            dict: 包含分数、概率和风险等级的字典
        """
        score = self.calculate_score(data)
        
        # 使用逻辑回归公式将分数转换为概率
        # 基于评分卡标准公式
        log_odds = (score - self.base_score) / self.pdo * np.log(2)
        odds = np.exp(log_odds)
        probability = 1 / (1 + 1/odds)
        
        # 确保概率在合理范围内
        probability = max(0.001, min(0.999, probability))
        
        return {{
            'score': score,
            'probability': float(probability),
            'risk_level': self.get_risk_level(score),
            'log_odds': float(log_odds)
        }}
    
    def predict(self, data):
        """预测类别（二分类）"""
        result = self.predict_proba(data)
        return 1 if result['probability'] > 0.5 else 0
    
    def get_feature_contributions(self, data):
        """获取各特征对分数的贡献"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        contributions = {{}}
        total_score = float(self.base_score)
        
        for feature in self.features:
            if feature in data.columns:
                value = data[feature].iloc[0] if hasattr(data, 'iloc') else data[feature]
                normalized = self._normalize_feature(value, feature)
                contribution = normalized * self.feature_weights[feature]
                
                contributions[feature] = {{
                    'value': float(value),
                    'normalized': float(normalized),
                    'weight': float(self.feature_weights[feature]),
                    'contribution': float(contribution),
                    'score_impact': float(contribution)
                }}
                total_score += contribution
        
        contributions['_total_score'] = float(total_score)
        return contributions
    
    def validate(self, test_data):
        """验证评分卡性能"""
        if isinstance(test_data, dict):
            test_data = pd.DataFrame([test_data])
        
        # 简单的验证逻辑
        scores = []
        for _, row in test_data.iterrows():
            result = self.predict_proba(row.to_dict())
            scores.append(result['score'])
        
        return {{
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores)
        }}

# 使用示例
if __name__ == "__main__":
    # 创建评分卡实例
    scorecard = CreditScorecard()
    
    # 示例客户数据
    sample_customer = {{
        "V2": 1.5, "V3": -0.8, "V4": 2.1, "V7": -1.2, "V9": 0.5, "V10": -0.3,
        "V11": 1.1, "V12": -0.9, "V14": 0.7, "V16": -0.4, "V17": 0.2, "V18": -1.5
    }}
    
    # 评估客户
    result = scorecard.predict_proba(sample_customer)
    contributions = scorecard.get_feature_contributions(sample_customer)
    
    print("客户信用评估结果:")
    print(f"  信用分数: {{result['score']}} 分")
    print(f"  违约概率: {{result['probability']:.2%}}")
    print(f"  风险等级: {{result['risk_level']}}")
    print(f"  Log-Odds: {{result['log_odds']:.3f}}")
    
    print("\n特征贡献分析:")
    for feat, info in contributions.items():
        if feat != '_total_score':
            print(f"  {{feat}}: {{info['contribution']:.1f}} 分 (值: {{info['value']:.3f}})")
'''
        
        # 保存评分卡类
        scorecard_path = self.output_dir / "scorecard.py"
        with open(scorecard_path, 'w', encoding='utf-8') as f:
            f.write(scorecard_code)
        
        print(f"✓ 评分卡类已保存: {scorecard_path}")
        return scorecard_path
    
    def _save_feature_importance(self, importances):
        """保存特征重要性"""
        df = pd.DataFrame({
            'feature': self.features,
            'importance': importances,
            'importance_ratio': importances / np.sum(importances),
            'weight': importances / np.sum(importances) * 100
        })
        df = df.sort_values('importance', ascending=False)
        
        csv_path = self.output_dir / "feature_importance.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✓ 特征重要性已保存: {csv_path}")
        return csv_path
    
    def run_conversion(self):
        """运行完整转换流程"""
        print("=" * 60)
        print("开始无PMML模型转换...")
        print("=" * 60)
        
        # 创建评分卡配置
        config = self._create_scorecard_config()
        
        # 创建评分卡类
        scorecard_path = self._create_scorecard_class(config)
        
        # 保存特征重要性
        importance_path = self._save_feature_importance(
            self._get_feature_importance()
        )
        
        print("\n" + "=" * 60)
        print("转换完成！")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print("生成文件:")
        print(f"  - {scorecard_path.name}")
        print(f"  - {self.output_dir / 'scorecard_config.json'}")
        print(f"  - {importance_path.name}")
        
        return {
            'scorecard_path': str(scorecard_path),
            'config_path': str(self.output_dir / 'scorecard_config.json'),
            'importance_path': str(importance_path),
            'output_dir': str(self.output_dir)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型转换为评分卡（无PMML依赖）')
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--features', required=True, help='特征文件路径')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建转换器并运行
    converter = SimpleScorecardConverter(args.model, args.features, args.output_dir)
    result = converter.run_conversion()
    
    return result


if __name__ == "__main__":
    main()