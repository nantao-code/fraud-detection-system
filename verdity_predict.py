"""
模型预测一致性验证脚本

功能:
1. 加载通过 joblib 保存的 Scikit-learn Pipeline 模型。
2. 加载通过 sklearn2pmml 导出的 PMML 模型。
3. 使用相同的测试数据集，分别对两个模型进行预测。
4. 比较两个模型输出的概率值，以验证PMML文件的转换是否准确无误。
"""
import pandas as pd
import numpy as np
import joblib
from pypmml import Model
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def compare_model_predictions(
    joblib_model_path: str,
    pmml_model_path: str,
    full_dataset_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    加载joblib和pmml模型，对同一测试集进行预测并比较概率结果。

    Args:
        joblib_model_path (str): .joblib模型文件的路径。
        pmml_model_path (str): .pmml模型文件的路径。
        full_dataset_path (str): 原始完整数据集的.csv文件路径。
        target_column (str): 数据集中的目标变量（标签）列名。
        test_size (float): 测试集分割比例，必须与训练时完全一致。
        random_state (int): 随机种子，必须与训练时完全一致。
    """
    print("=" * 80)
    print("开始进行 Joblib 与 PMML 模型预测一致性验证")
    print("=" * 80)

    # --- 1. 加载并准备数据 ---
    try:
        print(f"正在加载完整数据集: {full_dataset_path}...")
        df = pd.read_csv(full_dataset_path)
        
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]

        # 关键步骤：使用与训练时完全相同的参数分割数据，以获得一致的测试集
        _, X_test, _, _ = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y  # 确保分层抽样与训练时一致
        )
        print(f"数据加载和分割完成。测试集大小: {X_test.shape[0]} 条记录。")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 at {full_dataset_path}")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # --- 2. 加载模型 ---
    try:
        print(f"正在加载 Joblib 模型: {joblib_model_path}...")
        joblib_model = joblib.load(joblib_model_path)
        print("Joblib 模型加载成功。")
    except FileNotFoundError:
        print(f"错误: Joblib 模型文件未找到 at {joblib_model_path}")
        return
    except Exception as e:
        print(f"加载 Joblib 模型时发生错误: {e}")
        return

    try:
        print(f"正在加载 PMML 模型: {pmml_model_path}...")
        pmml_model = Model.fromFile(pmml_model_path)
        print("PMML 模型加载成功。")
    except Exception as e:
        print(f"加载 PMML 模型时发生错误: {e}")
        return

    # --- 3. 使用模型进行预测 ---
    print("\n正在使用模型进行预测...")
    
    # Joblib 模型预测
    # .predict_proba() 返回一个数组，第二列是正类(1)的概率
    joblib_probs = joblib_model.predict_proba(X_test)[:, 1]
    print("Joblib 模型预测完成。")

    # PMML 模型预测
    # .predict() 返回一个DataFrame，我们需要找到代表正类概率的列
    pmml_results = pmml_model.predict(X_test)
    
    # 动态查找概率列名 (通常是 "probability_1" 或 "probability(1)")
    positive_class_prob_col = None
    for col in pmml_results.columns:
        if 'probability' in col and '1' in col:
            positive_class_prob_col = col
            break
            
    if not positive_class_prob_col:
        print("错误: 在PMML输出中找不到类别'1'的概率列。")
        print(f"可用列: {pmml_results.columns.tolist()}")
        return
        
    pmml_probs = pmml_results[positive_class_prob_col].values
    print("PMML 模型预测完成。")

    # --- 4. 比较结果并生成报告 ---
    print("\n" + "=" * 80)
    print("预测结果比较报告")
    print("=" * 80)

    # 检查预测数量是否一致
    if len(joblib_probs) != len(pmml_probs):
        print(f"严重错误: 预测结果数量不匹配！ Joblib: {len(joblib_probs)}, PMML: {len(pmml_probs)}")
        return

    # 使用 numpy.allclose 进行浮点数比较，允许有极小的容差
    # atol (absolute tolerance) 是最重要的参数，代表绝对误差容忍度
    are_close = np.allclose(joblib_probs, pmml_probs, atol=1e-8)

    if are_close:
        print("✅ 验证通过！两个模型的预测概率在容差范围内完全一致。")
    else:
        print("❌ 验证失败！两个模型的预测概率存在显著差异。")

    # 计算差异的统计数据
    abs_diff = np.abs(joblib_probs - pmml_probs)
    
    print("\n差异统计:")
    print(f"  - 最大绝对差异: {np.max(abs_diff):.10f}")
    print(f"  - 平均绝对差异: {np.mean(abs_diff):.10f}")
    print(f"  - 差异标准差:   {np.std(abs_diff):.10f}")

    # 显示差异最大的前5个样本
    if not are_close:
        print("\n差异最大的前5个样本:")
        comparison_df = pd.DataFrame({
            'Joblib_Prob': joblib_probs,
            'PMML_Prob': pmml_probs,
            'Absolute_Difference': abs_diff
        })
        top_5_diffs = comparison_df.sort_values(by='Absolute_Difference', ascending=False).head(5)
        print(top_5_diffs.to_string())

    print("\n" + "=" * 80)
    print("验证结束。")
    print("=" * 80)

if __name__ == "__main__":
    # --- 配置区域 ---
    # 请根据您的项目修改以下文件路径和参数
    
    # 1. 模型文件路径
    JOBLIB_MODEL_PATH = 'models/pipeline_test_creditcard_random_over_LR.joblib' # <-- 修改这里
    PMML_MODEL_PATH   = 'pmmls/puts/pipeline_test_creditcard_random_over_LR.pmml' # <-- 修改这里
    
    # 2. 数据文件路径
    FULL_DATASET_PATH = 'data/test_creditcard.csv' # <-- 修改这里
    TARGET_COLUMN     = 'label'                        # <-- 修改这里
    
    # 3. 数据分割参数 (必须与训练时完全一致！)
    TEST_SPLIT_SIZE   = 0.2
    RANDOM_STATE_SEED = 42

    # --- 运行验证 ---
    compare_model_predictions(
        joblib_model_path=JOBLIB_MODEL_PATH,
        pmml_model_path=PMML_MODEL_PATH,
        full_dataset_path=FULL_DATASET_PATH,
        target_column=TARGET_COLUMN,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE_SEED
    )