import pandas as pd
from pypmml import Model

# --- 1. 加载您之前导出的PMML模型 ---
# PMMLPipeline已经将元数据嵌入，所以模型知道它需要哪些特征
try:
    model = Model.fromFile('pmmls/puts/pipeline_test_creditcard_random_over_LR.pmml') # 替换成您的PMML文件路径
    print("PMML模型加载成功！")
except Exception as e:
    print(f"PMML模型加载失败: {e}")
    exit()

# --- 2. 准备需要预测的新数据 ---
data = pd.read_csv('./data/test_creditcard.csv')
predict_data = data.loc[:,["V14","V16","V4","V6","V7","V9","V10","V11","V12","V17","V18","V19","V21"]]
predict_data_y = data.loc[:,["label"]]
print("预测数据前五行:")
print(predict_data.head())
print("预测数据标签前五行:")
print(predict_data_y.head())



# --- 3. 执行预测 ---
# .predict() 方法会执行PMML文件中的整个预处理和模型评分流程
results = model.predict(predict_data)

# --- 4. 查看结果 ---
# 结果通常是一个包含丰富信息的DataFrame
print("\n预测结果:")
print(results)

# 结果通常会包含以下列：
# - predicted_your_target_name: 最终的预测类别 (来自 .predict())
# - probability_0: 类别0的概率 (来自 .predict_proba())
# - probability_1: 类别1的概率 (来自 .predict_proba())
# ... 以及其他可能的输出