# 易受诈人群识别系统

一个基于机器学习的智能反诈系统，支持欺诈检测分类任务和欺诈损失金额预测回归任务。

## 🎯 项目特色

- **双任务支持**：同时支持分类任务（欺诈/非欺诈识别）和回归任务（欺诈损失金额预测）
- **多模型集成**：集成RIDGE、XGBoost、随机森林、LightGBM四种主流机器学习模型
- **完整工作流**：涵盖数据预处理、特征工程、模型训练、评估、部署全流程
- **灵活配置**：通过YAML配置文件灵活控制整个机器学习流程
- **生产就绪**：支持模型导出、PMML转换、API服务部署

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用最小依赖
pip install -r requirements_minimal.txt
```

### 2. 运行分类任务（欺诈检测）

```python
from src.pipeline_training import PipelineTraining

# 使用默认配置运行分类任务
pipeline = PipelineTraining()
results = pipeline.run()
```

### 3. 运行回归任务（损失金额预测）

```python
from src.pipeline_training import PipelineTraining
from src.unified_config import ConfigManager

# 加载回归配置
config = ConfigManager('config_regression.yaml')
pipeline = PipelineTraining(config)
results = pipeline.run()
```

## 📊 支持的模型

### 分类模型
- **LR** (Logistic Regression) - 逻辑回归
- **XGB** (XGBoost) - 极端梯度提升
- **RF** (Random Forest) - 随机森林
- **LGB** (LightGBM) - 轻量级梯度提升
- **ET** (Extra Trees) - 极端随机树
- **GBM** (Gradient Boosting) - 梯度提升
- **NB** (Naive Bayes) - 朴素贝叶斯
- **DT** (Decision Tree) - 决策树

### 回归模型
- **RIDGE** (Ridge Regression) - 岭回归
- **XGB_REG** (XGBoost Regression) - XGBoost回归
- **RF_REG** (Random Forest Regression) - 随机森林回归
- **LGB_REG** (LightGBM Regression) - LightGBM回归

## 📈 评估指标

### 分类任务
- **Accuracy** (准确率)
- **Precision** (精确率)
- **Recall** (召回率)
- **F1-Score** (F1分数)
- **AUC-ROC** (ROC曲线下面积)
- **AUC-PR** (PR曲线下面积)

### 回归任务
- **MSE** (均方误差)
- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **R²** (决定系数)
- **MAPE** (平均绝对百分比误差)

## ⚙️ 配置文件

项目使用YAML配置文件管理系统行为：

- **config.yaml** - 默认分类任务配置
- **config_regression.yaml** - 回归任务专用配置
- **batch_config.yaml** - 批处理任务配置

### 关键配置项

```yaml
# 任务类型
task_type: "regression"  # 或 "classification"

# 目标变量
target_column: "fraud_loss_amount"  # 回归任务
# target_column: "is_fraud"  # 分类任务

# 模型选择
models: ["LR", "XGB", "RF", "LGB"]  # 分类模型
regression_models: ["RIDGE", "XGB_REG", "RF_REG", "LGB_REG"]  # 回归模型
```

## 📁 项目结构

```
易受诈人群识别/
├── src/                    # 核心源码
│   ├── pipeline_training.py # 训练管道
│   ├── model_factory.py    # 模型工厂
│   ├── evaluator.py        # 评估器
│   ├── feature_engineering.py # 特征工程
│   └── ...
├── examples/               # 使用示例
├── data/                   # 数据文件
├── models/                 # 训练好的模型
├── plots/                  # 可视化图表
├── logs/                   # 运行日志
├── config.yaml             # 配置文件
└── README.md              # 项目说明
```

## 🛠️ 高级用法

### 新的特征工程Pipeline

项目新增了符合sklearn规范的特征工程Pipeline类：

#### 1. 特征选择Pipeline
```python
from src.feature_engineering import FeatureSelectorPipeline

# 创建特征选择器
selector = FeatureSelectorPipeline(
    task_type='classification',
    selection_methods=['missing_rate', 'variance', 'correlation', 'importance'],
    missing_rate_threshold=0.3,
    importance_threshold=0.01
)

# 训练并应用
selector.fit(X_train, y_train)
X_selected = selector.transform(X_test)
```

#### 2. 特征生成Pipeline
```python
from src.feature_engineering import FeatureGeneratorPipeline

# 创建特征生成器
generator = FeatureGeneratorPipeline(
    generate_polynomial=True,
    polynomial_degree=2,
    generate_interaction=True,
    generate_binning=True,
    bins_config={'age': 5, 'income': 4}
)

# 生成新特征
X_generated = generator.transform(X)
```

#### 3. 完整工作流程
```python
from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([
    ('generator', FeatureGeneratorPipeline(
        generate_polynomial=True,
        generate_interaction=True,
        generate_binning=True
    )),
    ('selector', FeatureSelectorPipeline(
        task_type='classification',
        selection_methods=['variance', 'importance']
    )),
    ('classifier', RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)
```

### 自定义特征工程

```python
from src.feature_engineering import FeatureEngineering

# 自定义特征处理
fe = FeatureEngineering()
fe.custom_feature_engineering(data)
```

### 模型部署

```python
from src.model_deployer import ModelDeployer

# 部署模型为API服务
deployer = ModelDeployer()
deployer.deploy_model(model_path, port=8000)
```

### 批量训练

```bash
# 运行批处理训练
python batch_training.py
```

## 📚 文档

- [配置指南](CONFIG_GUIDE.md) - 详细配置说明
- [回归任务使用指南](回归任务使用指南.md) - 回归任务专用指南
- [优化改造完成报告](优化改造完成报告.md) - 项目升级记录

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。