"""
模型评估器
提供模型评估、指标计算和可视化功能
支持分类和回归任务
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, classification_report, 
    confusion_matrix, precision_recall_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import logging
import os
import joblib
from pathlib import Path


class ModelEvaluator:
    """模型评估器，支持分类和回归任务"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series, 
                      task_type: str = 'classification') -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            task_type: 任务类型 ('classification' 或 'regression')
            
        Returns:
            评估结果字典
        """
        if task_type == 'classification':
            return self._evaluate_classification(model, X_train, y_train, X_test, y_test)
        else:
            return self._evaluate_regression(model, X_train, y_train, X_test, y_test)
    
    def _evaluate_classification(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """评估分类模型"""
        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 获取预测概率
        y_train_proba = None
        y_test_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)
            except:
                pass
        
        # 计算指标
        train_metrics = self._calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = self._calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
        
        # 详细报告
        detailed_report = {
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'detailed_report': detailed_report,
            'predictions': {
                'y_true': y_test.values,
                'y_pred': y_test_pred,
                'y_proba': y_test_proba[:, 1] if y_test_proba is not None else None
            }
        }
    
    def _evaluate_regression(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """评估回归模型"""
        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算指标
        train_metrics = self._calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_regression_metrics(y_test, y_test_pred)
        
        # 计算残差
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'y_true': y_test.values,
                'y_pred': y_test_pred,
                'residuals': test_residuals.values
            },
            'residuals': {
                'train_residuals': train_residuals.values,
                'test_residuals': test_residuals.values
            }
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算分类评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE（平均绝对百分比误差）
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_classification_results(self, results: Dict[str, Any], save_path: str = None):
        """绘制分类结果图表"""
        if 'predictions' not in results or 'y_proba' not in results['predictions']:
            self.logger.warning("无法绘制分类图表：缺少预测概率")
            return
        
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        y_proba = results['predictions']['y_proba']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('真实值')
        
        # ROC曲线
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC曲线 (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('假正率')
            axes[0, 1].set_ylabel('真正率')
            axes[0, 1].set_title('ROC曲线')
            axes[0, 1].legend(loc="lower right")
        
        # 预测概率分布
        axes[1, 0].hist(y_proba, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('预测概率分布')
        
        # 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([f'Feature {i}' for i in indices])
            axes[1, 1].set_xlabel('重要性')
            axes[1, 1].set_title('特征重要性')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"分类结果图表已保存至: {save_path}")
        
        plt.show()
    
    def plot_regression_results(self, results: Dict[str, Any], save_path: str = None):
        """绘制回归结果图表"""
        if 'predictions' not in results:
            self.logger.warning("无法绘制回归图表：缺少预测结果")
            return
        
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        residuals = results['predictions']['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 实际值 vs 预测值
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('实际值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('实际值 vs 预测值')
        
        # 残差图
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差图')
        
        # 残差分布
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('残差分布')
        
        # 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([f'Feature {i}' for i in indices])
            axes[1, 1].set_xlabel('重要性')
            axes[1, 1].set_title('特征重要性')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"回归结果图表已保存至: {save_path}")
        
        plt.show()
    
    def save_evaluation_report(self, results: Dict[str, Any], task_type: str, 
                              output_path: str, model_name: str = None):
        """保存评估报告"""
        report = {
            'model_name': model_name or 'unknown',
            'task_type': task_type,
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if task_type == 'classification' and 'detailed_report' in results:
            report['classification_report'] = results['detailed_report']['classification_report']
            report['confusion_matrix'] = results['detailed_report']['confusion_matrix']
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估报告已保存至: {output_path}")


class ModelComparator:
    """模型比较器"""
    
    def __init__(self):
        self.results = []
    
    def add_model_result(self, model_name: str, metrics: Dict[str, float], 
                        task_type: str = 'classification'):
        """添加模型结果"""
        self.results.append({
            'model_name': model_name,
            'task_type': task_type,
            **metrics
        })
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """生成比较报告"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # 按任务类型分组排序
        if 'task_type' in df.columns:
            # 对于分类任务，按F1或R2排序
            if 'f1' in df.columns:
                df = df.sort_values('f1', ascending=False)
            elif 'r2' in df.columns:
                df = df.sort_values('r2', ascending=False)
        
        return df
    
    def save_comparison_report(self, output_path: str):
        """保存比较报告"""
        df = self.generate_comparison_report()
        if df.empty:
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logging.info(f"模型比较报告已保存至: {output_path}")


# 工具函数
def load_model_results(model_path: str) -> Dict[str, Any]:
    """加载模型结果"""
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return None


def determine_task_type(y: pd.Series) -> str:
    """自动确定任务类型"""
    unique_values = len(y.unique())
    if unique_values <= 10 and y.dtype in ['int64', 'int32', 'object', 'category']:
        return 'classification'
    else:
        return 'regression'


if __name__ == "__main__":
    # 示例用法
    evaluator = ModelEvaluator()
    
    # 这里可以添加测试代码
    print("模型评估器已初始化")