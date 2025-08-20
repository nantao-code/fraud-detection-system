"""
模型监控模块
包含性能监控、数据漂移检测、告警等功能
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path
import warnings

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib


class ModelMonitor:
    """模型监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_config = config.get('monitoring', {})
        self.paths_config = config.get('paths', {})
        
        # 监控数据存储路径
        self.monitor_dir = Path(self.paths_config.get('monitor_output', 'monitoring'))
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化监控日志
        self.predictions_log = []
        self.performance_log = []
        
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        记录预测日志
        
        Args:
            prediction_data: 预测数据
        """
        prediction_data['timestamp'] = datetime.now().isoformat()
        self.predictions_log.append(prediction_data)
        
        # 保存到文件
        log_file = self.monitor_dir / "predictions_log.json"
        with open(log_file, 'a') as f:
            f.write(json.dumps(prediction_data) + '\n')
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """
        记录性能日志
        
        Args:
            performance_data: 性能数据
        """
        performance_data['timestamp'] = datetime.now().isoformat()
        self.performance_log.append(performance_data)
        
        # 保存到文件
        log_file = self.monitor_dir / "performance_log.json"
        with open(log_file, 'a') as f:
            f.write(json.dumps(performance_data) + '\n')
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame, 
                         threshold: float = 0.05) -> Dict[str, Any]:
        """
        检测数据漂移
        
        Args:
            current_data: 当前数据
            reference_data: 参考数据（训练数据）
            threshold: 显著性阈值
            
        Returns:
            漂移检测结果
        """
        drift_results = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        numerical_features = current_data.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features:
            if feature not in reference_data.columns:
                continue
                
            # 使用KS检验检测分布差异
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ks_stat, p_value = stats.ks_2samp(
                    current_data[feature].dropna(),
                    reference_data[feature].dropna()
                )
            
            drift_results['drift_scores'][feature] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
            
            if p_value < threshold:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(feature)
        
        # 分类特征使用卡方检验
        categorical_features = current_data.select_dtypes(include=['object', 'category']).columns
        
        for feature in categorical_features:
            if feature not in reference_data.columns:
                continue
                
            # 创建列联表
            current_counts = current_data[feature].value_counts()
            reference_counts = reference_data[feature].value_counts()
            
            # 对齐类别
            all_categories = set(current_counts.index) | set(reference_counts.index)
            
            current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
            reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
            
            # 卡方检验
            chi2_stat, p_value = stats.chisquare(current_aligned, reference_aligned)
            
            drift_results['drift_scores'][feature] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
            
            if p_value < threshold:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(feature)
        
        # 保存漂移检测结果
        drift_file = self.monitor_dir / "drift_results.json"
        with open(drift_file, 'a') as f:
            f.write(json.dumps(drift_results) + '\n')
        
        if drift_results['drift_detected']:
            logging.warning(f"检测到数据漂移，漂移特征: {drift_results['drifted_features']}")
        
        return drift_results
    
    def monitor_model_performance(self, y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray,
                                model_name: str) -> Dict[str, Any]:
        """
        监控模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            model_name: 模型名称
            
        Returns:
            性能监控结果
        """
        performance_metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(y_true),
            'positive_rate': np.mean(y_true),
            'predicted_positive_rate': np.mean(y_pred)
        }
        
        # 计算性能指标
        try:
            performance_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            performance_metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            performance_metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            performance_metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logging.error(f"计算性能指标失败: {e}")
            performance_metrics['error'] = str(e)
        
        # 性能阈值检查
        performance_thresholds = self.monitoring_config.get('performance_thresholds', {})
        
        alerts = []
        for metric, threshold in performance_thresholds.items():
            if metric in performance_metrics:
                if performance_metrics[metric] < threshold:
                    alerts.append({
                        'metric': metric,
                        'value': performance_metrics[metric],
                        'threshold': threshold,
                        'severity': 'warning'
                    })
        
        performance_metrics['alerts'] = alerts
        
        # 记录性能日志
        self.log_performance(performance_metrics)
        
        if alerts:
            logging.warning(f"检测到性能告警: {alerts}")
        
        return performance_metrics
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """
        生成监控报告
        
        Args:
            days: 报告天数
            
        Returns:
            监控报告
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_period_days': days,
            'summary': {},
            'details': {}
        }
        
        # 读取预测日志
        predictions_file = self.monitor_dir / "predictions_log.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = [json.loads(line) for line in f.readlines()]
            
            # 筛选最近的数据
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_predictions = [
                p for p in predictions 
                if datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            report['summary']['total_predictions'] = len(recent_predictions)
            if recent_predictions:
                report['summary']['avg_probability'] = np.mean([
                    p['probabilities'][0] if p['probabilities'] else 0 
                    for p in recent_predictions
                ])
        
        # 读取性能日志
        performance_file = self.monitor_dir / "performance_log.json"
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                performances = [json.loads(line) for line in f.readlines()]
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_performances = [
                p for p in performances 
                if datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            if recent_performances:
                report['details']['performance_trend'] = recent_performances
        
        # 读取漂移检测结果
        drift_file = self.monitor_dir / "drift_results.json"
        if drift_file.exists():
            with open(drift_file, 'r') as f:
                drifts = [json.loads(line) for line in f.readlines()]
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_drifts = [
                d for d in drifts 
                if datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            report['details']['drift_events'] = recent_drifts
            report['summary']['drift_events_count'] = len([
                d for d in recent_drifts if d['drift_detected']
            ])
        
        # 保存报告
        report_file = self.monitor_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"监控报告已生成: {report_file}")
        return report
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        检查告警
        
        Returns:
            告警列表
        """
        alerts = []
        
        # 检查性能告警
        performance_file = self.monitor_dir / "performance_log.json"
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                performances = [json.loads(line) for line in f.readlines()]
            
            # 获取最新性能
            if performances:
                latest_performance = performances[-1]
                if 'alerts' in latest_performance:
                    alerts.extend(latest_performance['alerts'])
        
        # 检查漂移告警
        drift_file = self.monitor_dir / "drift_results.json"
        if drift_file.exists():
            with open(drift_file, 'r') as f:
                drifts = [json.loads(line) for line in f.readlines()]
            
            # 获取最新漂移
            if drifts:
                latest_drift = drifts[-1]
                if latest_drift['drift_detected']:
                    alerts.append({
                        'type': 'data_drift',
                        'drifted_features': latest_drift['drifted_features'],
                        'severity': 'warning'
                    })
        
        return alerts