"""
特征工程模块
包含特征选择、特征评估和特征预处理相关功能
支持分类和回归任务
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CategoricalConverter(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        for col in X_copy.columns: # 现在可以安全地遍历列
            if X_copy[col].dtype == 'object':
                X_copy[col] = X_copy[col].astype('category')
        return X_copy


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """数据预处理器 - 重构为符合scikit-learn规范的转换器"""
    
    def __init__(self, model_type='unknown_model', missing_strategy='median',
                 outlier_method='iqr', outlier_threshold=1.5, transform_method='yeo-johnson',
                 scale_method='standard', encoding_method='ordinal',
                 categorical_threshold=10, numerical_threshold=20):
        """
        初始化数据预处理器，接收具体的参数
        
        Args:
            model_type: 模型类型，用于确定预处理方式
            missing_strategy: 缺失值处理策略 ('median', 'mean', 'mode', 'drop')
            outlier_method: 异常值处理方法 ('iqr', 'zscore', 'none')
            outlier_threshold: 异常值检测阈值
            transform_method: 数据变换方法 ('yeo-johnson', 'box-cox', 'quantile', 'none')
            scale_method: 特征缩放方法 ('standard', 'minmax', 'robust', 'none')
            encoding_method: 类别变量编码方法 ('ordinal', 'onehot', 'target')
            categorical_threshold: 类别变量唯一值阈值
            numerical_threshold: 数值特征阈值
        """
        # 将所有参数直接保存为实例属性
        self.model_type = model_type
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.transform_method = transform_method
        self.scale_method = scale_method
        self.encoding_method = encoding_method
        self.categorical_threshold = categorical_threshold
        self.numerical_threshold = numerical_threshold
        
        # << FIX START >>
        # This will hold the standard, PMML-compatible transformer
        self.preprocessor_ = None
        # << FIX END >>
        
    def fit(self, X: pd.DataFrame, y=None):
        """在训练数据上学习所有预处理规则"""
        
        # 1. 识别数值和类别列
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        
        # << FIX START >>
        # 2. Create and fit a standard ColumnTransformer, which is PMML-compatible
        self.preprocessor_ = self._create_preprocessor(
            numeric_features=numeric_cols, 
            categorical_features=categorical_cols
        )
        self.preprocessor_.fit(X, y)
        # << FIX END >>
        
        return self
    
    def _create_preprocessor(self, numeric_features: List[str], 
                           categorical_features: List[str]) -> ColumnTransformer: 
        """ 
        创建预处理器，根据模型类型选择不同的编码策略 
        
        Args: 
            numeric_features: 数值特征列表 
            categorical_features: 类别特征列表 
            
        Returns: 
            ColumnTransformer实例 
        """ 
        
        # 根据模型类型选择特征处理策略 
        model_type_upper = str(self.model_type).upper() 
        
        if model_type_upper in ['LR', 'RIDGE', 'LOGISTIC']: 
            # 逻辑回归/岭回归：数值特征需要标准化 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=self.missing_strategy)), 
                ('scaler', StandardScaler()) 
            ]) 
            logging.info(f"为{self.model_type}模型使用StandardScaler标准化数值特征") 
            
            # 线性模型：独热编码处理类别特征 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
            ]) 
            logging.info(f"为{self.model_type}模型使用独热编码处理类别特征") 
            
        elif model_type_upper in ['RF', 'RF_REG', 'RANDOMFOREST']: 
            # 随机森林：数值特征无需缩放，类别特征使用序号编码 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=self.missing_strategy)) 
            ]) 
            logging.info(f"为{self.model_type}模型不对数值特征进行缩放处理") 
            
            # RF模型：使用序号编码处理类别特征 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
            ]) 
            logging.info(f"为{self.model_type}模型使用序号编码处理类别特征") 
            
        elif model_type_upper in ['XGB', 'LGB', 'XGBOOST', 'LIGHTGBM', 'XGB_REG', 'LGB_REG']: 
            # XGBoost/LightGBM：数值特征无需缩放，使用专用CategoricalConverter 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=self.missing_strategy)) 
            ]) 
            logging.info(f"为{self.model_type}模型不对数值特征进行缺失值填充与缩放处理") 
            
            # XGB/LGB模型：使用专为它们优化的CategoricalConverter 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                ('categorical_converter', CategoricalConverter()) 
            ]) 
            logging.info(f"为{self.model_type}模型使用专为XGBoost/LightGBM优化的CategoricalConverter") 
            
        else: 
            # 默认处理策略
            logging.warning(f"未识别的模型类型: {self.model_type}，使用默认处理策略") 
            numeric_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy=self.missing_strategy)), 
                ('scaler', StandardScaler()) 
            ]) 
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
            ]) 
        
        # 动态构建transformers列表 
        transformers = [] 
        
        # 只有当存在数值特征时，才添加数值转换器 
        if numeric_features: 
            transformers.append(('num', numeric_transformer, numeric_features)) 
            logging.info(f"为 {len(numeric_features)} 个数值特征添加了转换器。") 
        
        # 只有当存在类别特征时，才添加类别转换器 
        if categorical_features: 
            transformers.append(('cat', categorical_transformer, categorical_features)) 
            logging.info(f"为 {len(categorical_features)} 个类别特征添加了转换器。") 
        
        # 如果没有任何特征，这是一个异常情况 
        if not transformers: 
            logging.warning("警告：数据中既没有检测到数值特征，也没有检测到类别特征。预处理器将是空的。") 
        
        # 组合预处理器 
        preprocessor = ColumnTransformer( 
            transformers=transformers, 
            remainder='passthrough', 
            verbose_feature_names_out=False 
        ) 
        
        return preprocessor
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用学到的规则转换数据"""
        # << FIX START >>
        # Delegate the transform call to the standard, PMML-compatible preprocessor
        if self.preprocessor_ is None:
            raise RuntimeError("You must call fit before calling transform.")
        
        transformed_array = self.preprocessor_.transform(X)
        feature_names = self.preprocessor_.get_feature_names_out()
        
        return pd.DataFrame(transformed_array, columns=feature_names, index=X.index)
        # << FIX END >>


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    一个简单的转换器，用于从DataFrame中选择指定的列。
    这使得特征选择步骤可以被包含在scikit-learn Pipeline中。
    """
    def __init__(self, columns: List[str]):
        """
        Args:
            columns: 需要保留的列名列表。
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        # 这个转换器没有需要学习的东西，所以fit方法什么都不做
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        执行列选择。
        
        Args:
            X: 输入的DataFrame。
            
        Returns:
            只包含指定列的DataFrame。
        """
        # 返回只包含指定列的数据
        return X[self.columns]


class FeatureSelectorPipeline(BaseEstimator, TransformerMixin):
    """
    特征选择Pipeline - 符合scikit-learn规范的转换器
    提供多种特征选择方法，支持分类和回归任务
    """
    
    def __init__(self, task_type='classification', selection_methods=None,
                 missing_rate_threshold=0.5, correlation_threshold=0.9,
                 variance_threshold=0.01, iv_threshold=0.02,
                 mutual_info_k='auto', importance_threshold=0.01,
                 model_type='LR', f_regression_k='auto', f_classif_k='auto'):
        """
        初始化特征选择Pipeline，接收具体的参数
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            selection_methods: 特征选择方法列表
            missing_rate_threshold: 缺失率阈值
            correlation_threshold: 相关性阈值
            variance_threshold: 方差阈值
            iv_threshold: IV值阈值
            mutual_info_k: 互信息选择特征数
            importance_threshold: 重要性阈值
            model_type: 模型类型
            f_regression_k: F回归选择特征数
            f_classif_k: F分类选择特征数
        """
        self.task_type = task_type
        self.selection_methods = selection_methods if selection_methods else ['missing_rate', 'correlation', 'variance']
        self.missing_rate_threshold = missing_rate_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.iv_threshold = iv_threshold
        self.mutual_info_k = mutual_info_k
        self.importance_threshold = importance_threshold
        self.model_type = model_type
        self.f_regression_k = f_regression_k
        self.f_classif_k = f_classif_k
        
        # 存储学习到的规则
        self.selected_features_ = None
        self.removed_features_ = {}
        # << FIX START >>
        # This will hold the standard, PMML-compatible transformer
        self.transformer_ = None
        # << FIX END >>
    
    @staticmethod
    def select_by_missing_rate(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """基于缺失率选择特征"""
        missing_rates = X.isnull().sum() / len(X)
        return missing_rates[missing_rates > threshold].index.tolist()
    
    @staticmethod
    def select_by_iv(X: pd.DataFrame, y: pd.Series, threshold: float = 0.02) -> List[str]:
        """基于IV值选择特征（仅分类任务）"""
        from sklearn.preprocessing import LabelEncoder
        
        iv_scores = {}
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                # 类别变量
                le = LabelEncoder()
                x_encoded = le.fit_transform(X[col].fillna('missing'))
            else:
                # 数值变量，使用分箱
                x_encoded = pd.qcut(X[col], q=10, duplicates='drop').astype(str)
            
            # 计算IV值
            df = pd.DataFrame({'x': x_encoded, 'y': y})
            iv = 0
            for val in df['x'].unique():
                good = len(df[(df['x'] == val) & (df['y'] == 0)])
                bad = len(df[(df['x'] == val) & (df['y'] == 1)])
                if good > 0 and bad > 0:
                    good_rate = good / len(df[df['y'] == 0])
                    bad_rate = bad / len(df[df['y'] == 1])
                    woe = np.log(bad_rate / good_rate)
                    iv += (bad_rate - good_rate) * woe
            
            iv_scores[col] = abs(iv)
        
        return [col for col, iv in iv_scores.items() if iv < threshold]
    
    @staticmethod
    def select_by_correlation(X: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        """基于相关性选择特征"""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        
        for col in upper_triangle.columns:
            correlated = [c for c in upper_triangle.index if upper_triangle.loc[c, col] > threshold]
            if correlated:
                to_drop.append(col)
        
        return list(set(to_drop))
    
    @staticmethod
    def select_by_variance(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """基于方差选择特征"""
        variances = X.var()
        return variances[variances < threshold].index.tolist()
    
    @staticmethod
    def select_by_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 20, task_type: str = 'classification') -> List[str]:
        """基于互信息选择特征"""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        if task_type == 'classification':

            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        
        scores = pd.Series(scores, index=X.columns)
        return scores.nlargest(k).index.tolist()
    
    @staticmethod
    def select_by_importance(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01, 
                           model_type: str = 'LR', task_type: str = 'classification') -> List[str]:
        """基于特征重要性选择特征"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if task_type == 'classification':
            if model_type.upper() in ['RF', 'XGB', 'LGB']:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LogisticRegression(random_state=42)
        else:
            if model_type.upper() in ['RF', 'XGB', 'LGB']:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = Ridge(random_state=42)
        
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # 处理线性模型的系数
            coef = model.coef_
            if len(coef.shape) == 1:
                # 单输出回归
                importances = np.abs(coef)
            elif len(coef.shape) == 2:
                if coef.shape[0] == 1:
                    # 二分类或多输出回归的单个系数
                    importances = np.abs(coef[0])
                else:
                    # 多分类，取每个特征的最大重要性
                    importances = np.abs(coef).max(axis=0)
            else:
                importances = np.abs(coef).flatten()
        
        # 确保importances长度与特征数量匹配
        if len(importances) != len(X.columns):
            importances = importances[:len(X.columns)]
        
        feature_importance = pd.Series(importances, index=X.columns)
        return feature_importance[feature_importance >= threshold].index.tolist()
    
    @staticmethod
    def select_by_f_regression(X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
        """基于F检验回归选择特征"""
        from sklearn.feature_selection import f_regression, SelectKBest
        
        selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        scores = pd.Series(selector.scores_, index=X.columns)
        return scores.nlargest(k).index.tolist()
    
    @staticmethod
    def select_by_f_classif(X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
        """基于F检验分类选择特征"""
        from sklearn.feature_selection import f_classif, SelectKBest
        
        selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        scores = pd.Series(selector.scores_, index=X.columns)
        return scores.nlargest(k).index.tolist()
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        在训练数据上学习特征选择规则
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            self: 返回自身实例
        """
        if y is None:
            raise ValueError("特征选择需要目标变量y")
            
        X_copy = X.copy()
        original_features = set(X_copy.columns)
        remaining_features = set(X_copy.columns)
        
        # 记录每一步移除的特征
        self.removed_features_ = {}
        
        logging.info("=" * 50)
        logging.info("特征选择开始")
        logging.info(f"原始特征数量: {len(original_features)}")
        logging.info(f"原始特征列表: {sorted(list(original_features))}")
        
        # 1. 基于缺失率选择
        if 'missing_rate' in self.selection_methods:
            logging.info("-" * 40)
            logging.info(f"步骤1: 基于缺失率选择 (阈值={self.missing_rate_threshold})")
            removed = self.select_by_missing_rate(X_copy, self.missing_rate_threshold)
            self.removed_features_['missing_rate'] = removed
            remaining_features -= set(removed)
            logging.info(f"移除的特征数量: {len(removed)}")
            if removed:
                logging.info(f"移除的特征: {sorted(removed)}")
            logging.info(f"剩余特征数量: {len(remaining_features)}")
            if remaining_features:
                X_copy = X_copy[list(remaining_features)]
        
        # 2. 基于IV值选择（仅分类任务）
        if 'iv' in self.selection_methods and self.task_type == 'classification':
            logging.info("-" * 40)
            logging.info(f"步骤2: 基于IV值选择 (阈值={self.iv_threshold})")
            if len(remaining_features) > 0:
                removed = self.select_by_iv(X_copy, y, self.iv_threshold)
                self.removed_features_['iv'] = removed
                remaining_features -= set(removed)
                logging.info(f"移除的特征数量: {len(removed)}")
                if removed:
                    logging.info(f"移除的特征: {sorted(removed)}")
                logging.info(f"剩余特征数量: {len(remaining_features)}")
                if remaining_features:
                    X_copy = X_copy[list(remaining_features)]
            else:
                logging.warning("无剩余特征可供选择")
        
        # 3. 基于相关性选择
        if 'correlation' in self.selection_methods and len(remaining_features) > 1:
            logging.info("-" * 40)
            logging.info(f"步骤3: 基于相关性选择 (阈值={self.correlation_threshold})")
            removed = self.select_by_correlation(X_copy, self.correlation_threshold)
            self.removed_features_['correlation'] = removed
            remaining_features -= set(removed)
            logging.info(f"移除的特征数量: {len(removed)}")
            if removed:
                logging.info(f"移除的特征: {sorted(removed)}")
            logging.info(f"剩余特征数量: {len(remaining_features)}")
            if remaining_features:
                X_copy = X_copy[list(remaining_features)]
        
        # 4. 基于方差选择
        if 'variance' in self.selection_methods:
            logging.info("-" * 40)
            logging.info(f"步骤4: 基于方差选择 (阈值={self.variance_threshold})")
            removed = self.select_by_variance(X_copy, self.variance_threshold)
            self.removed_features_['variance'] = removed
            remaining_features -= set(removed)
            logging.info(f"移除的特征数量: {len(removed)}")
            if removed:
                logging.info(f"移除的特征: {sorted(removed)}")
            logging.info(f"剩余特征数量: {len(remaining_features)}")
            if remaining_features:
                X_copy = X_copy[list(remaining_features)]
        
        # 5. 基于互信息选择
        if 'mutual_info' in self.selection_methods:
            logging.info("-" * 40)
            logging.info(f"步骤5: 基于互信息选择 (k={self.mutual_info_k}, 任务类型={self.task_type})")
            if len(remaining_features) > 0:
                k_value = min(self.mutual_info_k, len(remaining_features)) if isinstance(self.mutual_info_k, int) else min(20, len(remaining_features))
                selected = self.select_by_mutual_info(X_copy, y, k_value, self.task_type)
                removed = list(remaining_features - set(selected))
                self.removed_features_['mutual_info'] = removed
                remaining_features = set(selected)
                logging.info(f"选择的特征数量: {len(selected)}")
                if selected:
                    logging.info(f"选择的特征: {sorted(selected)}")
                logging.info(f"移除的特征数量: {len(removed)}")
                if removed:
                    logging.info(f"移除的特征: {sorted(removed)}")
                logging.info(f"剩余特征数量: {len(remaining_features)}")
                if remaining_features:
                    X_copy = X_copy[list(remaining_features)]
            else:
                logging.warning("无剩余特征可供选择")
        
        # 6. 基于特征重要性选择
        if 'importance' in self.selection_methods:
            logging.info("-" * 40)
            logging.info(f"步骤6: 基于特征重要性选择 (阈值={self.importance_threshold}, 模型类型={self.model_type})")
            if len(remaining_features) > 0:
                selected = self.select_by_importance(X_copy, y, self.importance_threshold, self.model_type, self.task_type)
                removed = list(remaining_features - set(selected))
                self.removed_features_['importance'] = removed
                remaining_features = set(selected)
                logging.info(f"选择的特征数量: {len(selected)}")
                if selected:
                    logging.info(f"选择的特征: {sorted(selected)}")
                logging.info(f"移除的特征数量: {len(removed)}")
                if removed:
                    logging.info(f"移除的特征: {sorted(removed)}")
                logging.info(f"剩余特征数量: {len(remaining_features)}")
                if remaining_features:
                    X_copy = X_copy[list(remaining_features)]
            else:
                logging.warning("无剩余特征可供选择")
        
        # 7. 基于F检验选择（回归任务）
        if 'f_regression' in self.selection_methods and self.task_type == 'regression':
            logging.info("-" * 40)
            logging.info(f"步骤7: 基于F检验回归选择 (k={self.f_regression_k})")
            if len(remaining_features) > 0:
                k_value = min(self.f_regression_k, len(remaining_features)) if isinstance(self.f_regression_k, int) else min(20, len(remaining_features))
                selected = self.select_by_f_regression(X_copy, y, k_value)
                removed = list(remaining_features - set(selected))
                self.removed_features_['f_regression'] = removed
                remaining_features = set(selected)
                logging.info(f"选择的特征数量: {len(selected)}")
                if selected:
                    logging.info(f"选择的特征: {sorted(selected)}")
                logging.info(f"移除的特征数量: {len(removed)}")
                if removed:
                    logging.info(f"移除的特征: {sorted(removed)}")
                logging.info(f"剩余特征数量: {len(remaining_features)}")
                if remaining_features:
                    X_copy = X_copy[list(remaining_features)]
            else:
                logging.warning("无剩余特征可供选择")
        
        # 8. 基于F检验选择（分类任务）
        if 'f_classif' in self.selection_methods and self.task_type == 'classification':
            logging.info("-" * 40)
            logging.info(f"步骤8: 基于F检验分类选择 (k={self.f_classif_k})")
            if len(remaining_features) > 0:
                k_value = min(self.f_classif_k, len(remaining_features)) if isinstance(self.f_classif_k, int) else min(20, len(remaining_features))
                selected = self.select_by_f_classif(X_copy, y, k_value)
                removed = list(remaining_features - set(selected))
                self.removed_features_['f_classif'] = removed
                remaining_features = set(selected)
                logging.info(f"选择的特征数量: {len(selected)}")
                if selected:
                    logging.info(f"选择的特征: {sorted(selected)}")
                logging.info(f"移除的特征数量: {len(removed)}")
                if removed:
                    logging.info(f"移除的特征: {sorted(removed)}")
                logging.info(f"剩余特征数量: {len(remaining_features)}")
                if remaining_features:
                    X_copy = X_copy[list(remaining_features)]
            else:
                logging.warning("无剩余特征可供选择")
        
        # 存储最终选择的特征
        self.selected_features_ = list(remaining_features)
        
        logging.info("=" * 50)
        logging.info("特征选择完成")
        logging.info(f"最终选择的特征数量: {len(self.selected_features_)}")
        if self.selected_features_:
            logging.info(f"最终选择的特征: {sorted(self.selected_features_)}")
        logging.info(f"总共移除的特征数量: {len(original_features) - len(self.selected_features_)}")
        
        # 打印移除特征的详细统计
        if self.removed_features_:
            logging.info("-" * 40)
            logging.info("移除特征详细统计")
            for step, removed in self.removed_features_.items():
                if removed:
                    logging.info(f"{step}: 移除了 {len(removed)} 个特征")
                    logging.info(f"具体移除的特征: {sorted(removed)}")
        
        return self


class FeatureGeneratorPipeline(BaseEstimator, TransformerMixin):
    """
    特征生成Pipeline - 符合scikit-learn规范的转换器
    整合多种特征生成方法，创建新的特征
    """
    
    def __init__(self, generate_polynomial=False, polynomial_degree=2,
                 polynomial_include_bias=False, generate_interaction=False,
                 interaction_pairs=None, generate_statistical=False,
                 statistical_group_cols=None, generate_binning=False,
                 bins_config=None, generate_aggregation=False,
                 aggregation_window_cols=None):
        """
        初始化特征生成Pipeline，接收具体的参数
        
        Args:
            generate_polynomial: 是否生成多项式特征
            polynomial_degree: 多项式次数
            polynomial_include_bias: 是否包含偏置项
            generate_interaction: 是否生成交互特征
            interaction_pairs: 交互特征对
            generate_statistical: 是否生成统计特征
            statistical_group_cols: 统计特征分组列
            generate_binning: 是否生成分箱特征
            bins_config: 分箱配置
            generate_aggregation: 是否生成聚合特征
            aggregation_window_cols: 聚合窗口列
        """
        self.generate_polynomial = generate_polynomial
        self.polynomial_degree = polynomial_degree
        self.polynomial_include_bias = polynomial_include_bias
        self.generate_interaction = generate_interaction
        self.interaction_pairs = interaction_pairs
        self.generate_statistical = generate_statistical
        self.statistical_group_cols = statistical_group_cols
        self.generate_binning = generate_binning
        self.bins_config = bins_config
        self.generate_aggregation = generate_aggregation
        self.aggregation_window_cols = aggregation_window_cols
        
        self.generated_features_ = []
        self.feature_types_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        在训练数据上学习特征生成规则
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Returns:
            self: 返回自身实例
        """
        # 特征生成不需要学习过程，但为了符合sklearn规范，返回self
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用配置生成新特征
        
        Args:
            X: 输入特征数据
            
        Returns:
            包含原始特征和新生成特征的DataFrame
        """
        X_copy = X.copy()
        original_features = list(X_copy.columns)
        
        # 记录生成的特征
        self.generated_features_ = []
        self.feature_types_ = {}
        
        # 1. 生成多项式特征
        if self.generate_polynomial:
            try:
                from sklearn.preprocessing import PolynomialFeatures
                numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=self.polynomial_include_bias)
                    poly_features = poly.fit_transform(X_copy[numeric_cols])
                    feature_names = poly.get_feature_names_out(numeric_cols)
                    
                    # 只保留非原始特征的多项式特征
                    new_features = [col for col in feature_names if col not in original_features]
                    if new_features:
                        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X_copy.index)
                        X_copy = pd.concat([X_copy, poly_df[new_features]], axis=1)
                        self.generated_features_.extend(new_features)
                        self.feature_types_.update({f: 'polynomial' for f in new_features})
            except Exception as e:
                logging.warning(f"多项式特征生成失败: {e}")
        
        # 2. 生成交互特征
        if self.generate_interaction:
            try:
                numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
                interactions = []
                
                if self.interaction_pairs is None:
                    # 创建所有可能的二阶交互
                    for i in range(len(numeric_cols)):
                        for j in range(i + 1, len(numeric_cols)):
                            col1, col2 = numeric_cols[i], numeric_cols[j]
                            interaction_name = f"{col1}_x_{col2}"
                            interaction = pd.DataFrame({interaction_name: X_copy[col1] * X_copy[col2]}, index=X_copy.index)
                            interactions.append(interaction)
                else:
                    # 创建指定的交互特征
                    for col1, col2 in self.interaction_pairs:
                        if col1 in numeric_cols and col2 in numeric_cols:
                            interaction_name = f"{col1}_x_{col2}"
                            interaction = pd.DataFrame({interaction_name: X_copy[col1] * X_copy[col2]}, index=X_copy.index)
                            interactions.append(interaction)
                
                if interactions:
                    interaction_df = pd.concat(interactions, axis=1)
                    X_copy = pd.concat([X_copy, interaction_df], axis=1)
                    new_features = list(interaction_df.columns)
                    self.generated_features_.extend(new_features)
                    self.feature_types_.update({f: 'interaction' for f in new_features})
            except Exception as e:
                logging.warning(f"交互特征生成失败: {e}")
    
        return X_copy
    
    def get_generated_features(self) -> List[str]:
        """获取生成的特征列表"""
        return self.generated_features_
    
    def get_feature_types(self) -> Dict[str, str]:
        """获取特征类型映射"""
        return self.feature_types_
    
    def get_generation_summary(self) -> pd.DataFrame:
        """获取特征生成摘要"""
        if not self.generated_features_:
            return pd.DataFrame(columns=['feature', 'type'])
            
        return pd.DataFrame([
            {'feature': feature, 'type': self.feature_types_.get(feature, 'unknown')}
            for feature in self.generated_features_
        ])