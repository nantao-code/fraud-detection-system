"""
模型优化、部署、API调用和监控的完整示例
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader, DataSplitter
from src.feature_engineer import FeatureEngineer
from src.model_optimizer import ModelOptimizer
from src.model_deployer import ModelDeployer
from src.model_monitor import ModelMonitor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline():
    """运行完整的模型优化、部署和监控流程"""
    
    # 1. 加载配置
    from src.unified_config import UnifiedConfig
    config = UnifiedConfig("config.yaml")
    
    logger.info("1. 开始数据加载...")
    # 2. 加载和预处理数据
    data_loader = DataLoader(config)
    
    # 使用配置中的单一数据源
    all_data = data_loader.load_data()
    
    if all_data.empty:
        logger.error("数据加载失败，请检查config.yaml中的input_data配置")
        return
    
    # 3. 特征工程和数据分割
    logger.info("2. 特征工程和数据分割...")
    feature_engineer = FeatureEngineer(config)
    data_splitter = DataSplitter(config)
    
    # 假设数据包含特征和标签
    target_col = config.feature_engineering.target_column
    feature_cols = [col for col in all_data.columns if col != target_col]
    
    X = all_data[feature_cols]
    y = all_data[target_col]
    
    # 数据分割
    X_train, X_test, y_train, y_test = data_splitter.split_data(
        all_data, feature_cols, target_col
    )
    
    # 特征处理
    X_train_processed = feature_engineer.fit_transform(X_train, y_train)
    X_test_processed = feature_engineer.transform(X_test)
    
    # 4. 模型优化
    logger.info("3. 模型优化...")
    optimizer = ModelOptimizer(config)
    
    # 超参数优化
    best_params, best_score = optimizer.optimize_hyperparameters(
        X_train_processed, y_train, n_trials=config.optimization.n_trials
    )
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳分数: {best_score}")
    
    # 训练最终模型
    final_model = optimizer.train_final_model(
        X_train_processed, y_train, best_params
    )
    
    # 5. 模型评估
    logger.info("4. 模型评估...")
    evaluation_results = optimizer.evaluate_model(
        final_model, X_test_processed, y_test
    )
    logger.info(f"模型评估结果: {evaluation_results}")
    
    # 6. 模型部署
    logger.info("5. 模型部署...")
    deployer = ModelDeployer(config)
    
    # 保存模型和特征工程器
    model_path = deployer.save_model(final_model, feature_engineer)
    logger.info(f"模型已保存到: {model_path}")
    
    # 7. 初始化监控
    logger.info("6. 初始化监控...")
    monitor = ModelMonitor(config)
    
    # 8. 批量预测和监控
    logger.info("7. 批量预测和监控...")
    predictions = deployer.batch_predict(X_test_processed)
    
    # 记录预测结果
    for i, pred in enumerate(predictions):
        monitor.log_prediction({
            'model_version': 'v1.0',
            'prediction': int(pred),
            'probabilities': [0.5, 0.5],  # 示例概率
            'input_hash': hash(str(X_test_processed.iloc[i].values))
        })
    
    # 监控模型性能
    performance_metrics = monitor.monitor_model_performance(
        y_test.values, predictions, [0.5] * len(predictions), 'fraud_detection_v1'
    )
    logger.info(f"性能监控结果: {performance_metrics}")
    
    # 9. 数据漂移检测
    logger.info("8. 数据漂移检测...")
    # 使用训练数据作为参考
    drift_results = monitor.detect_data_drift(
        X_test_processed, X_train_processed
    )
    logger.info(f"数据漂移检测结果: {drift_results}")
    
    # 10. 生成监控报告
    logger.info("9. 生成监控报告...")
    report = monitor.generate_monitoring_report(days=7)
    logger.info(f"监控报告已生成: {report['generated_at']}")
    
    # 11. 检查告警
    alerts = monitor.check_alerts()
    if alerts:
        logger.warning(f"发现告警: {alerts}")
    else:
        logger.info("无告警")
    
    logger.info("完整流程执行完成！")


def run_api_deployment():
    """运行API部署示例"""
    logger.info("启动API服务...")
    
    # 使用uvicorn启动FastAPI服务
    import subprocess
    import time
    
    # 启动API服务（在实际环境中使用命令行启动）
    cmd = [
        "uvicorn", 
        "src.api_service:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ]
    
    logger.info("API服务启动命令:")
    logger.info(" ".join(cmd))
    logger.info("请在终端中运行上述命令启动API服务")
    
    # 实际使用示例：
    # import requests
    # 
    # # 单条预测
    # response = requests.post("http://localhost:8000/predict", json={
    #     "features": [1.0, 2.0, 3.0, 4.0, 5.0]
    # })
    # print(response.json())
    # 
    # # 批量预测
    # response = requests.post("http://localhost:8000/predict/batch", json={
    #     "data": [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
    # })
    # print(response.json())


if __name__ == "__main__":
    print("=== 模型优化、部署和监控完整示例 ===")
    print("1. 运行完整流程")
    print("2. API部署示例")
    
    choice = input("请选择操作 (1/2): ").strip()
    
    if choice == "1":
        run_complete_pipeline()
    elif choice == "2":
        run_api_deployment()
    else:
        print("无效选择")