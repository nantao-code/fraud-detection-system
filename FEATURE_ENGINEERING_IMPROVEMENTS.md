# 特征工程改进功能文档

## 概述

本项目针对回归任务对特征工程模块进行了两项重要改进：

1. **类别特征编码策略选择**：支持根据模型类型选择Ordinal编码或One-Hot编码
2. **基于特征-目标相关性的特征选择**：新增F检验方法评估特征与目标变量的相关性

## 改进功能详解

### 1. 类别特征编码改进

#### 问题背景
- **Ordinal编码**：将类别特征转换为整数（0,1,2...），但会引入人为的顺序关系
- **One-Hot编码**：为每个类别创建二进制特征，避免顺序假设，但会增加特征维度

#### 解决方案
在`DataPreprocessor`类中添加了`encoding_method`参数：

```python
# 初始化时选择编码策略
preprocessor = DataPreprocessor(
    encoding_method='ordinal'  # 或 'onehot'
)

# 或
preprocessor = DataPreprocessor(
    encoding_method='onehot'  # 适用于线性模型
)
```

#### 使用建议
- **树模型**（随机森林、XGBoost）：使用`ordinal`编码
- **线性模型**（线性回归、岭回归）：使用`onehot`编码

### 2. 基于特征-目标相关性的特征选择

#### 新增方法
在`FeatureSelector`类中添加了两种新方法：

```python
# 回归任务使用F回归检验
@staticmethod
def select_by_f_regression(data: pd.DataFrame, target: pd.Series, k: int = 20) -> List[str]

# 分类任务使用F分类检验
@staticmethod
def select_by_f_classif(data: pd.DataFrame, target: pd.Series, k: int = 20) -> List[str]
```

#### 配置使用
在配置文件中启用：

```yaml
feature_engineering:
  # 启用基于F检验的特征选择
  use_f_regression_selection: true
  f_regression_k: 15  # 选择前15个最相关特征
```

## 代码变更总结

### 文件修改
1. **`src/feature_engineering.py`**
   - `DataPreprocessor`类：添加`encoding_method`参数
   - `FeatureSelector`类：添加`select_by_f_regression`和`select_by_f_classif`方法
   - `FeatureEngineeringPipeline`类：添加F检验特征选择步骤

### 新增文件
1. **`config/feature_engineering_improvements.yaml`**
   - 改进功能的配置文件示例
2. **`examples/feature_engineering_improvements_example.py`**
   - 完整的使用示例和演示
3. **`FEATURE_ENGINEERING_IMPROVEMENTS.md`**
   - 本文档

## 使用方法

### 1. 类别特征编码选择

```python
from src.feature_engineering import DataPreprocessor

# 为线性模型使用独热编码
preprocessor = DataPreprocessor(
    missing_strategy='median',
    outlier_method='iqr',
    encoding_method='onehot'  # 关键参数
)

# 为树模型使用序号编码
preprocessor = DataPreprocessor(
    missing_strategy='median',
    encoding_method='ordinal'  # 默认选项
)
```

### 2. 基于F检验的特征选择

```python
from src.feature_engineering import FeatureEngineeringPipeline
from src.config import UnifiedConfig

# 配置特征工程
config = UnifiedConfig()
config.set('feature_engineering.use_f_regression_selection', True)
config.set('feature_engineering.f_regression_k', 20)

# 创建特征工程管道
fe_pipeline = FeatureEngineeringPipeline(config)
X_processed = fe_pipeline.fit_transform(X_train, y_train)
```

### 3. 完整配置示例

```yaml
# config/regression_config.yaml
feature_engineering:
  encoding_method: "onehot"  # 回归任务推荐onehot
  use_f_regression_selection: true
  f_regression_k: 15
  
modeling:
  task_type: "regression"
  model_type: "LR"  # 线性回归
```

## 性能影响

### 类别特征编码
- **Ordinal编码**：特征数量不变，内存占用小
- **One-Hot编码**：特征数量增加，但提高线性模型性能

### F检验特征选择
- **计算成本**：相对较低，适合大数据集
- **效果**：有效过滤与目标变量无关的特征
- **可解释性**：保留最有预测力的特征

## 兼容性说明

- 所有改进均为向后兼容
- 默认行为保持不变（`encoding_method='ordinal'`）
- 现有代码无需修改即可运行
- 新增功能通过配置参数启用

## 最佳实践

1. **回归任务**：
   - 使用`encoding_method='onehot'`
   - 启用`use_f_regression_selection=true`

2. **分类任务**：
   - 树模型使用`encoding_method='ordinal'`
   - 线性模型使用`encoding_method='onehot'`

3. **特征选择**：
   - 结合多种选择方法获得更好效果
   - 通过交叉验证确定最佳`k`值

## 技术支持

如遇到问题，请检查：
1. 配置文件格式是否正确
2. 特征工程管道是否正确初始化
3. 任务类型设置是否匹配实际数据