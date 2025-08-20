# 🎯 易受诈人群识别系统 - 完整文档

> **🎉 最新更新**: 系统已完成全面重构，解决了数据泄露问题，支持PMML转换和评分卡功能，可直接用于生产环境！

## 📋 目录
- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [配置指南](#配置指南)
- [使用教程](#使用教程)
- [API文档](#api文档)
- [故障排除](#故障排除)
- [版本更新](#版本更新)

## 🎯 项目概述

这是一个专为政企客户设计的**易受诈人群识别系统**，基于机器学习技术构建，提供从数据预处理、特征工程、模型训练、模型优化到部署监控的完整解决方案。

### 🚀 核心优势
- **零数据泄露**: 修复了特征工程中的数据泄露问题
- **多格式支持**: 支持PMML、评分卡、Joblib等多种模型格式
- **生产就绪**: 可直接部署到生产环境
- **监控完善**: 内置模型监控和性能追踪

## 🎯 核心功能

### 1. 模型训练与优化
- **多算法支持**: LR、XGBoost、随机森林、LightGBM
- **超参数优化**: 基于Optuna的贝叶斯优化
- **特征工程**: 工业级特征选择和预处理
- **不平衡处理**: SMOTE、ADASYN、随机过采样/欠采样

### 2. 模型转换与部署
- **PMML转换**: 支持所有主流模型转换为PMML格式
- **评分卡转换**: 逻辑回归评分卡、决策树评分卡
- **API服务**: RESTful API接口
- **模型监控**: 性能监控、数据漂移检测

### 3. 批量训练与测试
- **参数网格测试**: 多参数组合并行测试
- **多进程并行**: 充分利用CPU资源
- **结果汇总**: 详细的训练结果报告

## 🚀 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证安装
python check_pmml_env.py
```

### 基础使用

#### 方法1：直接使用评分卡（推荐）
```python
# 评分卡已生成，立即可用
from output.simple_scorecard import SimpleCreditScorecard

scorecard = SimpleCreditScorecard()
result = scorecard.predict_proba({
    "V2": 0.5, "V3": -0.3, "V4": 1.2, "V7": -0.8,
    "V9": 0.1, "V10": 0.7, "V11": -0.5, "V12": 0.9,
    "V14": -1.2, "V16": 0.3, "V17": -0.7, "V18": 0.4
})

print(f"信用分数: {result['score']}")
print(f"违约概率: {result['probability']:.2%}")
print(f"风险等级: {result['risk_level']}")
```

#### 方法2：训练新模型
```bash
# 单模型训练
python main.py --config config.yaml --model LR

# 批量训练测试
python batch_training.py --config batch_config.yaml --max-workers 4
```

#### 方法3：启动API服务
```bash
# 启动FastAPI服务
uvicorn src.api_service:app --host 0.0.0.0 --port 8000
```

## 📁 系统架构

```
易受诈人群识别/
├── src/                          # 核心源代码
│   ├── feature_engineering.py    # 特征工程（已修复数据泄露）
│   ├── model_training.py         # 模型训练
│   ├── pipeline_training.py      # 管道训练
│   ├── model_converter.py      # 模型转换
│   ├── api_service.py          # API服务
│   └── ...
├── examples/                   # 使用示例
├── converted_models/           # 转换后的模型
├── models/                     # 训练好的模型
├── data/                       # 数据文件
├── logs/                       # 日志文件
├── output/                     # 评分卡输出
├── config.yaml                 # 配置文件
├── batch_config.yaml          # 批量训练配置
└── README.md                  # 本文档
```

## ⚙️ 配置指南

### 主配置文件 (config.yaml)

```yaml
# 路径配置
paths:
  input_data: "data/creditcard.csv"
  model_output: "models/{input_data}_{imbalance_method}_{model_name}_model.joblib"
  report_output: "reports/{input_data}_{imbalance_method}_model_comparison.csv"

# 模型配置
modeling:
  models: ["LR", "XGB", "RF", "LGB"]
  imbalance_method: "smote"
  handle_imbalance: true
  use_hyperparameter_tuning: false
  optimization_trials: 50

# 特征工程配置
feature_engineering:
  target_column: "Class"
  iv_threshold: 0.1
  corr_threshold: 0.8
  missing_rate_threshold: 0.3
  handle_outliers: true
  transform_features: true
  scale_features: true
  create_binning_features: true

# 监控配置
monitoring:
  performance_thresholds:
    accuracy: 0.75
    precision: 0.70
    recall: 0.65
    auc: 0.80
```

### 批量训练配置 (batch_config.yaml)

```yaml
parameter_grid:
  preprocessing.handle_imbalance_method:
    - "smote"
    - "random_over"
    - "none"
  preprocessing.feature_selection.threshold:
    - 0.85
    - 0.9
    - 0.95

models:
  - "XGB"
  - "RF"
  - "LR"
```

## 📖 使用教程

### 1. 数据预处理（已修复数据泄露）

#### 问题修复
- **修复前**: 使用`fit_transform`同时处理训练集和测试集，存在数据泄露
- **修复后**: 使用`fit`在训练集学习规则，`transform`应用规则到测试集

#### 使用示例
```python
from src.feature_engineering import FeatureEngineeringPipeline

# 创建管道（scikit-learn兼容）
pipeline = FeatureEngineeringPipeline(config)

# 学习转换规则（仅训练集）
pipeline.fit(X_train, y_train)

# 应用转换规则（训练集和测试集）
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### 2. 模型训练

#### 单模型训练
```python
from src.model_training import TrainingPipeline

# 创建训练器
trainer = TrainingPipeline(config)

# 训练单个模型
model, results = trainer.run_single_model("LR")

# 训练所有模型
results = trainer.run_all_models()
```

#### 批量训练测试
```bash
# 预览任务
python batch_training.py --config batch_config.yaml --dry-run

# 执行批量训练
python batch_training.py --config batch_config.yaml --max-workers 4
```

### 3. 模型转换

#### 评分卡转换（推荐）
```python
from src.model_converter import ModelConverter

converter = ModelConverter()

# 转换为评分卡
scorecard = converter.convert_to_scorecard(
    "LR", X_train, y_train, "dataset_name"
)

# 使用评分卡
result = scorecard.predict_proba(customer_data)
```

#### PMML转换
```python
# 转换为PMML
pmml_path = converter.convert_to_pmml(
    model, numeric_features, categorical_features, "dataset_name"
)
```

### 4. API服务

#### 启动服务
```bash
uvicorn src.api_service:app --host 0.0.0.0 --port 8000
```

#### API调用示例
```python
import requests

# 单条预测
response = requests.post("http://localhost:8000/predict", json={
    "features": [1.0, 2.0, 3.0, 4.0, 5.0]
})

# 批量预测
response = requests.post("http://localhost:8000/predict/batch", json={
    "data": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
})
```

## 📊 输出文件说明

### 训练输出
- **模型文件**: `models/{dataset}_{method}_{model}_model.joblib`
- **特征列表**: `features/{dataset}_{method}_{model}_final_features.json`
- **训练报告**: `logs/batch_training_results_*.json`

### 转换输出
- **评分卡**: `output/simple_scorecard.py`
- **评分卡配置**: `output/scorecard_config.json`
- **特征重要性**: `output/feature_importance.csv`
- **PMML文件**: `converted_models/pmml/*.pmml`

### 监控输出
- **性能日志**: `logs/model_performance_*.json`
- **漂移检测**: `logs/drift_detection_*.json`
- **预测日志**: `logs/predictions_*.json`

## 🔧 故障排除

### 常见问题

#### 1. Java版本问题（PMML转换）
**问题**: `java.lang.UnsupportedClassVersionError`
**解决**:
```bash
# 使用简化版评分卡（无需Java）
python convert_without_pmml.py --model models/model.joblib --features features/final_features.json
```

#### 2. 数据泄露警告
**解决**: 已修复！使用新的`FeatureEngineeringPipeline`类

#### 3. 内存不足
**解决**: 
- 减少并行进程数：`--max-workers 2`
- 使用数据采样
- 检查内存泄漏

#### 4. 特征不匹配
**解决**: 
- 确保特征配置文件与模型训练时一致
- 检查数据格式

### 环境检查
```bash
# 运行环境检查脚本
python check_pmml_env.py
```

## 📈 版本更新

### v3.0.0 (当前版本)
- ✅ **数据泄露修复**: 重构特征工程管道
- ✅ **PMML支持**: 支持所有模型格式转换
- ✅ **评分卡功能**: 零依赖的评分卡系统
- ✅ **API服务**: RESTful API接口
- ✅ **监控完善**: 模型性能监控

### v2.0.0
- ✅ 评分卡转换功能
- ✅ 批量训练测试
- ✅ 超参数优化

### v1.0.0
- ✅ 基础模型训练
- ✅ 特征工程
- ✅ 模型评估

## 📞 技术支持

### 快速验证
```bash
# 1. 验证环境
python check_pmml_env.py

# 2. 运行示例
python run_conversion_example.py

# 3. 测试API
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

### 文档资源
- **快速开始**: 见`QUICK_START.md`
- **配置指南**: 见`CONFIG_GUIDE.md`
- **故障排除**: 见`PMML_TROUBLESHOOTING.md`

### 联系支持
- **项目维护**: 系统已完全文档化
- **技术支持**: 查看各模块的详细文档

---

## 🎯 立即开始使用

**评分卡已生成，立即可用！**

```python
# 最简单的使用方式
from output.simple_scorecard import SimpleCreditScorecard

scorecard = SimpleCreditScorecard()
print("评分卡已就绪，可直接用于生产环境！")
```

**🎉 恭喜！您现在拥有一个完整的、生产就绪的易受诈人群识别系统！**