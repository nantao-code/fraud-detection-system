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
from sklearn.inspection import permutation_importance
import logging
import os
import joblib
import json
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
                      X_test: pd.DataFrame, y_test: pd.Series, model_name: str, 
                      task_type: str) -> Dict[str, Any]:
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
            return self._evaluate_classification(model, X_train, y_train, X_test, y_test, model_name)
        elif task_type == 'regression':
            return self._evaluate_regression(model, X_train, y_train, X_test, y_test, model_name)
        else:
            self.logger.error(f"不支持的任务类型: {task_type}")
            return None
    
    def _evaluate_classification(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """评估分类模型"""
        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        train_metrics = self._calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = self._calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
        
        # 详细报告
        detailed_report = {
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        # 绘制可视化图表
        self._plot_score_distribution(y_train_proba, y_test_proba, model_name)
        self._plot_ks_curve(y_test, y_test_proba, model_name)
        self._plot_roc_curve(y_train, y_train_proba, y_test, y_test_proba, model_name) 
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'detailed_report': detailed_report
        }
    
    def _evaluate_regression(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:

        """评估回归模型"""
        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算指标
        train_metrics = self._calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_regression_metrics(y_test, y_test_pred)
        
        # 绘图
        self._plot_regression_score_distribution(y_test, y_test_pred, model_name)

        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算分类评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_proba),
            'ks': self._calculate_ks(y_true, y_proba)
        }
        
        return metrics
    
    def _calculate_ks(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算KS统计量（使用TPR-FPR差值最大值）
        
        Args:
            y_true: 真实标签
            y_score: 预测概率
            
        Returns:
            KS统计量值（max(TPR - FPR)）
        """
        try:
            # 确保输入是numpy数组
            y_true = np.array(y_true)
            y_score = np.array(y_score)
            
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            
            # 计算KS统计量为TPR-FPR的最大差值
            ks_stat = max(tpr - fpr)
            
            return ks_stat
            
        except Exception as e:
            self.logger.error(f"计算KS统计量时出错: {str(e)}")
            return 0.0
    
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

    # 分类任务绘图函数
    def _plot_roc_curve(self, y_train: pd.Series, y_pred_proba_train: np.ndarray,
                       y_test: pd.Series, y_pred_proba_test: np.ndarray, model_name: str):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve
        
        plt.figure(figsize=(10, 8))
        
        # 训练集ROC曲线
        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
        auc_train = roc_auc_score(y_train, y_pred_proba_train)
        
        # 测试集ROC曲线
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
        auc_test = roc_auc_score(y_test, y_pred_proba_test)
        
        plt.plot(fpr_train, tpr_train, label=f'Training Set (AUC = {auc_train:.3f})', linewidth=2)
        plt.plot(fpr_test, tpr_test, label=f'Test Set (AUC = {auc_test:.3f})', linewidth=2)
        
        # 对角线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = self._get_plot_path(model_name, 'roc_curve')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"ROC curve saved to: {plot_path}")

    def _plot_ks_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str):
        """绘制测试集的KS曲线图"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            ks_values = tpr - fpr
            ks_stat = max(ks_values)
            ks_idx = np.argmax(ks_values)
            ks_threshold = thresholds[ks_idx]
            
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, ks_values, color='blue', lw=2, label=f'KS曲线 (KS={ks_stat:.3f})')
            plt.axhline(y=ks_stat, color='red', linestyle='--', linewidth=1, alpha=0.7)
            plt.plot(ks_threshold, ks_stat, 'o', color='red', markersize=6, label=f'最大KS点')
            plt.xlabel('thresholds')
            plt.ylabel('TPR - FPR')
            plt.title(f'{model_name} - KS Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            
            plot_path = self._get_plot_path(model_name, 'ks')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"KS curve saved to: {plot_path}")

        except Exception as e:
            self.logger.error(f"绘制KS曲线时出错: {str(e)}")

    def _plot_score_distribution(self, y_pred_proba_train: np.ndarray, 
                                y_pred_proba_test: np.ndarray, model_name: str):
        """绘制评分分布图（合并折线图版本）"""
        plt.figure(figsize=(10, 6))
        
        # 训练集折线图
        hist_train, bins_train = np.histogram(y_pred_proba_train, bins=50, range=(0, 1))
        bin_centers_train = (bins_train[:-1] + bins_train[1:]) / 2
        plt.plot(bin_centers_train, hist_train, label="Training Set", linewidth=2, marker='o', markersize=4)
        
        # 测试集折线图
        hist_test, bins_test = np.histogram(y_pred_proba_test, bins=50, range=(0, 1))
        bin_centers_test = (bins_test[:-1] + bins_test[1:]) / 2
        plt.plot(bin_centers_test, hist_test, label="Test Set", linewidth=2, marker='s', markersize=4)
        
        plt.title(f"{model_name} - Score Distribution", fontsize=14)
        plt.xlabel("Model Score (Probability)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 1)  # 设置x轴范围为0-1
        plt.ylim(0, max(max(hist_train), max(hist_test)) * 1.1)  # 设置合适的y轴范围
        
        plot_path = self._get_plot_path(model_name, 'score_distribution')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Score distribution plot saved to: {plot_path}")

    # 回归任务绘图函数
    def _plot_regression_score_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,  model_name: str):
        """绘制测试集的评分分布折线图"""
        try:
            # 计算评分（这里使用绝对误差作为评分）
            scores = np.abs(y_true - y_pred)
            
            # 创建评分区间
            max_score = np.max(scores)
            bins = np.linspace(0, max_score, 20)
            
            # 计算评分分布
            hist, bin_edges = np.histogram(scores, bins=bins)
            freq = hist / len(scores) * 100
            
            # 使用区间中点作为x轴
            x_points = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_points, freq, marker='o', color='green', linewidth=2)
            plt.xlabel('Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f"{model_name} - Score Distribution", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plot_path = self._get_plot_path(model_name, 'score_distribution')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Regression score distribution plot saved to: {plot_path}")
        except Exception as e:
            self.logger.error(f"绘制评分分布图时出错: {str(e)}")
    

    
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
    
    def _get_plot_path(self, model_name: str, plot_type: str) -> Path:
        """获取图表保存路径"""
        # 获取配置信息用于路径模板
        input_data_path = self.config.get('paths.input_data', 'data')
        input_data_name = Path(input_data_path).stem
        imbalance_method = self.config.get('modeling.imbalance_method', 'none')
        
        # 使用配置中的模板格式化图表输出路径
        plot_output_template = self.config.get('paths.plot_output', 'plots/{input_data}_{imbalance_method}_{model_name}_{plot_type}_plot.png')
        plot_filename = plot_output_template.format(
            input_data=input_data_name,
            imbalance_method=imbalance_method,
            model_name=model_name,
            plot_type=plot_type
        )
        base_path = Path(plot_filename).parent
        base_path.mkdir(parents=True, exist_ok=True)
        return Path(plot_filename)


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
    
    def generate_comparison_report(self, task_type: str) -> pd.DataFrame:

        """生成比较报告"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # 按任务类型分组排序
        if task_type == 'classification':
            df = df.sort_values('auc', ascending=False)
        elif task_type == 'regression':
            df = df.sort_values('r2', ascending=False)

        return df

if __name__ == "__main__":
    # 示例用法
    evaluator = ModelEvaluator()
    
    # 这里可以添加测试代码
    print("模型评估器已初始化")