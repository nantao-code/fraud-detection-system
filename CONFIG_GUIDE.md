# 配置系统使用指南

## 概述
本项目已将所有硬编码值改为从配置文件获取，提供了完整的配置化支持。主要配置文件为 `config.yaml`，模板文件为 `config_template.yaml`。

## 主要配置项说明

### 1. 特征工程配置 (`feature_engineering`)
```yaml
feature_engineering:
  # 数值特征检测阈值
  unique_value_threshold: 10
  
  # 缺失率阈值
  missing_rate_threshold: 0.3
  
  # 信息值阈值
  iv_threshold: 0.1
  
  # VIF阈值
  vif_threshold: 10.0
  
  # 相关性阈值
  corr_threshold: 0.8
  
  # 早停轮次
  early_stopping_rounds: 50
  
  # 分类任务指标
  primary_metric_classification: roc_auc
  scoring_metrics_classification:
    - roc_auc
    - ks
  
  # 回归任务指标
  primary_metric_regression: neg_root_mean_squared_error
  scoring_metrics_regression:
    - neg_root_mean_squared_error
    - neg_mean_absolute_error
    - r2
```

### 2. 模型配置 (`modeling`)
```yaml
modeling:
  # 随机状态
  random_state: 42
  
  # 测试集比例
  test_size: 0.3
  
  # 验证集比例（用于早停）
  validation_split_ratio: 0.1
  
  # 交叉验证
  n_splits: 5
  n_iter: 5
  
  # 早停轮次
  early_stopping_rounds: 100
  
  # 分类模型列表
  models:
    - XGB
    - RF
    - LR
    - LGB
  
  # 回归模型列表
  regression_models:
    - RIDGE
    - XGB_REG
    - RF_REG
    - LGB_REG
  
  # 模型类型映射（分类）
  model_types:
    linear_models:
      - LR
      - LASSO
      - ELASTIC
    tree_models:
      - RF
      - XGB
      - LGB
```

### 3. 回归模型配置
系统现已支持以下回归模型：

- **RIDGE**: Ridge回归（L2正则化线性回归）
- **XGB_REG**: XGBoost回归器
- **RF_REG**: 随机森林回归器
- **LGB_REG**: LightGBM回归器

```yaml
# 回归模型参数配置
modeling:
  ridge_regression:
    alpha: [0.1, 1.0, 10.0]
    solver: "auto"
    max_iter: 1000
    tol: 0.0001
  
  xgboost_regression:
    n_estimators: [500, 1000, 1500]
    learning_rate: [0.01, 0.1, 0.3]
    max_depth: [3, 6, 9]
    eval_metric: "rmse"
  
  random_forest_regression:
    n_estimators: [100, 200, 300]
    max_depth: [None, 10, 20, 30]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
  
  lightgbm_regression:
    n_estimators: [500, 1000, 1500]
    learning_rate: [0.01, 0.1, 0.3]
    max_depth: [-1, 5, 10]
    num_leaves: [31, 50, 100]
```

### 3. 不平衡处理配置
```yaml
imbalance_methods:
  smote:
    type: SMOTE
    random_state: 42
  random_over:
    type: RandomOverSampler
    random_state: 42
  random_under:
    type: RandomUnderSampler
    random_state: 42
```

## 已修复的硬编码项

### pipeline_training.py 中的修复

1. **数值特征检测阈值**
   - 原：硬编码值 `10`
   - 现：`feature_engineering.unique_value_threshold`

2. **不平衡采样器随机状态**
   - 原：硬编码值 `42`
   - 现：`modeling.random_state`

3. **验证集分割比例**
   - 原：硬编码值 `0.1`
   - 现：`modeling.validation_split_ratio`

4. **模型类型检查**
   - 原：硬编码列表 `['LR']` 和 `['RF', 'XGB', 'LGB']`
   - 现：`modeling.model_types.linear_models` 和 `modeling.model_types.tree_models`

5. **早停轮次**
   - 原：硬编码值 `100`
   - 现：`modeling.early_stopping_rounds`

6. **预测方法中的分割比例**
   - 原：硬编码值 `0.2` 和 `42`
   - 现：`modeling.test_size` 和 `modeling.random_state`

7. **日志配置参数**
   - **文件**: `model_training.py`
   - **原代码**: `_setup_logging` 方法参数 `data_name="数据"`, `imbalance_method='none'`, `model_name='model'`
   - **修复后**: 所有参数从配置文件获取，方法签名简化为 `_setup_logging()`

8. **训练管道方法参数**
   - **文件**: `model_training.py`
   - **原代码**: `run_single_model(model_name: str, data_name: str = "data")`
   - **修复后**: 移除 `data_name` 参数，改为从配置获取

9. **批量训练方法参数**
   - **文件**: `model_training.py`
   - **原代码**: `run_all_models(data_name: str = "data")`
   - **修复后**: 移除 `data_name` 参数，改为从配置获取

10. **API服务参数**
    - **文件**: `api_service.py`
    - **原代码**: `run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False)`
    - **修复后**: 移除所有参数，改为从配置 `api.host`, `api.port`, `api.reload` 获取

## 使用示例

### 分类任务（欺诈检测）
```python
from src.unified_config import UnifiedConfig
from src.pipeline_training import PipelineTraining

# 加载分类配置
config = UnifiedConfig('config.yaml')

# 创建训练器
trainer = PipelineTraining(config)

# 运行分类模型训练
trainer.run_all_models()
```

### 回归任务（欺诈损失金额预测）
```python
from src.unified_config import UnifiedConfig
from src.pipeline_training import PipelineTraining

# 加载回归配置
config = UnifiedConfig('config_regression.yaml')

# 创建训练器
trainer = PipelineTraining(config)

# 运行回归模型训练
trainer.run_all_models()
```

### 分类任务配置文件示例
```yaml
# 分类任务配置
modeling:
  model_type: XGB
  random_state: 42
  test_size: 0.3
  imbalance_method: smote
  models:  # 分类模型
    - XGB
    - RF
    - LR
    - LGB

paths:
  input_data: data/creditcard.csv
  
logging:
  level: INFO
  file: logs/training.log
  logs_dir: logs

api:
  host: 0.0.0.0
  port: 8000
  reload: false
```

### 回归任务配置文件示例
```yaml
# 回归任务配置
modeling:
  model_type: RF_REG
  random_state: 42
  test_size: 0.3
  handle_imbalance: false  # 回归任务不需要处理不平衡
  imbalance_method: none
  models: []  # 禁用分类模型
  regression_models:  # 启用回归模型
    - RIDGE
    - XGB_REG
    - RF_REG
    - LGB_REG

paths:
  input_data: data/fraud_loss_data.csv
  target_column: fraud_amount  # 回归目标变量
```

### 自定义配置
```yaml
# 分类任务自定义配置
feature_engineering:
  unique_value_threshold: 15
  
modeling:
  test_size: 0.25
  random_state: 123
  early_stopping_rounds: 200

paths:
  input_data: data/my_data.csv
```

### 运行时覆盖配置
```python
from src.unified_config import UnifiedConfig

# 分类任务
config = UnifiedConfig('config.yaml')
config.set('modeling.test_size', 0.2)
config.set('feature_engineering.unique_value_threshold', 20)

# 回归任务
config = UnifiedConfig('config_regression.yaml')
config.set('modeling.regression_models', ['XGB_REG', 'RF_REG'])  # 只运行部分回归模型
```

## 配置文件验证

系统会自动验证配置文件中的关键参数，确保：
- 数值范围合理（如测试集比例在0-1之间）
- 模型类型存在且有效
- 路径配置正确

## 最佳实践

1. **版本控制**：将配置文件纳入版本控制，便于团队协作
2. **环境分离**：为不同环境（开发、测试、生产）创建不同的配置文件
3. **文档同步**：配置变更时同步更新此文档
4. **默认值**：合理设置默认值，确保配置缺失时系统仍能正常运行

## 故障排查

### 配置未生效
1. 检查配置文件路径是否正确
2. 确认配置项名称拼写无误
3. 查看日志中的配置加载信息

### 参数类型错误
1. 检查数值参数是否为数字类型
2. 确认布尔值使用true/false（小写）
3. 列表参数使用正确的YAML格式

### 路径问题
1. 使用相对路径时确认工作目录正确
2. 检查路径分隔符（Windows使用反斜杠或正斜杠）
3. 确保目标目录存在且可写
- **文件**: `model_training.py`