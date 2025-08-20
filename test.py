from sklearn2pmml import sklearn2pmml
import joblib
from pathlib import Path

model_path = Path('./models/pipeline_test_creditcard_random_over_RF.joblib')

try:
    model = joblib.load(model_path)
    print(f"✓ 成功加载模型: {model_path}")
except Exception as e:
    print(f"✗ 加载模型失败: {e}")
    raise


# --- 步骤3: 将训练好的Pipeline保存为PMML文件 ---
pmml_file_path_rf = './output/pipeline_test_creditcard_random_over_RF.pmml'
print(f"正在将RF Pipeline保存到: {pmml_file_path_rf}")
sklearn2pmml(model, pmml_file_path_rf, with_repr=True)
print("随机森林PMML文件已生成。")


# (可选) 验证文件内容
with open(pmml_file_path_rf, "r") as f:
    # 打印前几行，可以看到XML结构
    print("\nPMML文件内容预览:")
    for i, line in enumerate(f):
        if i >= 10: break
        print(line, end='')
