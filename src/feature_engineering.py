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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """特征选择器 - 支持分类和回归任务"""
    
    @staticmethod
    def select_by_missing_rate(data: pd.DataFrame, threshold: float = 0.3) -> List[str]:
        """基于缺失率选择特征"""
        missing_rates = data.isnull().sum() / len(data)
        return missing_rates[missing_rates > threshold].index.tolist()
    
    @staticmethod
    def select_by_iv(data: pd.DataFrame, target: pd.Series, threshold: float = 0.1) -> List[str]:
        """基于IV值选择特征（仅适用于分类任务）"""
        if len(target.unique()) > 2:  # 回归任务跳过IV计算
            return []
            
        iv_values = {}
        for col in data.columns:
            try:
                iv = FeatureSelector._calculate_iv(data[col], target)
                iv_values[col] = iv
            except:
                iv_values[col] = 0
        
        return [col for col, iv in iv_values.items() if iv < threshold]
    
    @staticmethod
    def select_by_correlation(data: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """基于相关性选择特征"""
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        return to_drop
    
    @staticmethod
    def select_by_variance(data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """基于方差选择特征"""
        variances = data.var()
        return variances[variances < threshold].index.tolist()
    
    @staticmethod
    def select_by_chi2(data: pd.DataFrame, target: pd.Series, k: int = 20, task_type: str = 'classification') -> List[str]:
        """基于卡方检验选择特征（仅分类任务）"""
        if task_type != 'classification' or len(target.unique()) > 2:
            return []
            
        try:
            selector = SelectKBest(score_func=chi2, k=min(k, len(data.columns)))
            selector.fit(data, target)
            selected_indices = selector.get_support(indices=True)
            return [data.columns[i] for i in selected_indices]
        except:
            return data.columns.tolist()
    
    @staticmethod
    def select_by_mutual_info(data: pd.DataFrame, target: pd.Series, k: int = 20, task_type: str = 'classification') -> List[str]:
        """基于互信息选择特征"""
        try:
            if task_type == 'classification' and len(target.unique()) <= 2:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
                
            selector = SelectKBest(score_func=score_func, k=min(k, len(data.columns)))
            selector.fit(data, target)
            selected_indices = selector.get_support(indices=True)
            return [data.columns[i] for i in selected_indices]
        except:
            return data.columns.tolist()
    
    @staticmethod
    def select_by_importance(data: pd.DataFrame, target: pd.Series, threshold: float = 0.01, 
                           model_type: str = 'random_forest', task_type: str = 'classification') -> List[str]:
        """基于特征重要性选择特征"""
        try:
            if task_type == 'classification':
                if model_type == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = LogisticRegression(random_state=42)
            else:
                if model_type == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = Ridge(random_state=42)
            
            model.fit(data, target)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return data.columns.tolist()
            
            feature_importance = pd.DataFrame({
                'feature': data.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
            return selected_features if selected_features else data.columns.tolist()
            
        except:
            return data.columns.tolist()
    
    @staticmethod
    def select_by_f_regression(data: pd.DataFrame, target: pd.Series, k: int = 20) -> List[str]:
        """基于F检验选择与回归目标最相关的K个特征"""
        from sklearn.feature_selection import f_regression, SelectKBest
        
        try:
            # 确保k不超过特征数量
            k = min(k, data.shape[1])
            
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(data, target)
            selected_indices = selector.get_support(indices=True)
            return data.columns[selected_indices].tolist()
        except Exception as e:
            logging.warning(f"F回归特征选择失败: {e}")
            return data.columns.tolist()
    
    @staticmethod
    def select_by_f_classif(data: pd.DataFrame, target: pd.Series, k: int = 20) -> List[str]:
        """基于F检验选择与分类目标最相关的K个特征"""
        from sklearn.feature_selection import f_classif, SelectKBest
        
        try:
            k = min(k, data.shape[1])
            
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(data, target)
            selected_indices = selector.get_support(indices=True)
            return data.columns[selected_indices].tolist()
        except Exception as e:
            logging.warning(f"F分类特征选择失败: {e}")
            return data.columns.tolist()
    
    @staticmethod
    def _calculate_iv(feature: pd.Series, target: pd.Series) -> float:
        """计算IV值（仅分类任务）"""
        df = pd.DataFrame({'feature': feature, 'target': target})
        df = df.dropna()
        
        if len(df) == 0:
            return 0
        
        # 分箱处理
        try:
            df['bin'] = pd.qcut(df['feature'], q=10, duplicates='drop')
        except:
            df['bin'] = pd.cut(df['feature'], bins=5, duplicates='drop')
        
        # 计算WOE和IV
        bin_group = df.groupby('bin')['target'].agg(['count', 'sum'])
        bin_group.columns = ['total', 'bad']
        bin_group['good'] = bin_group['total'] - bin_group['bad']
        
        total_bad = bin_group['bad'].sum()
        total_good = bin_group['good'].sum()
        
        bin_group['bad_rate'] = bin_group['bad'] / total_bad
        bin_group['good_rate'] = bin_group['good'] / total_good
        
        bin_group['woe'] = np.log((bin_group['good_rate'] + 0.0001) / (bin_group['bad_rate'] + 0.0001))
        bin_group['iv'] = (bin_group['good_rate'] - bin_group['bad_rate']) * bin_group['woe']
        
        return bin_group['iv'].sum()


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """重构的特征工程管道 - 符合scikit-learn规范，支持分类和回归"""
    
    def __init__(self, config):
        """
        初始化特征工程管道
        
        Args:
            config: UnifiedConfig实例，支持点分路径配置访问
        """
        self.config = config
        
        # 存储训练阶段学到的状态
        self.feature_names_ = None
        self.selected_features_ = None
        self.pca_model_ = None
        self.preprocessor_ = None
        self.task_type_ = None
        
        # 配置解析
        self.fe_config = self._parse_fe_config()
    
    def _parse_fe_config(self) -> Dict[str, Any]:
        """解析特征工程配置"""
        return {
            'handle_outliers': self.config.get('feature_engineering.handle_outliers', False),
            'outlier_method': self.config.get('feature_engineering.outlier_method', 'iqr'),
            'outlier_threshold': self.config.get('feature_engineering.outlier_threshold', 1.5),
            'transform_features': self.config.get('feature_engineering.transform_features', False),
            'transform_method': self.config.get('feature_engineering.transform_method', 'yeo-johnson'),
            'create_polynomial_features': self.config.get('feature_engineering.create_polynomial_features', False),
            'polynomial_degree': self.config.get('feature_engineering.polynomial_degree', 2),
            'create_interaction_features': self.config.get('feature_engineering.create_interaction_features', False),
            'interaction_pairs': self.config.get('feature_engineering.interaction_pairs', None),
            'create_statistical_features': self.config.get('feature_engineering.create_statistical_features', False),
            'statistical_group_cols': self.config.get('feature_engineering.statistical_group_cols', None),
            'create_binning_features': self.config.get('feature_engineering.create_binning_features', False),
            'bins_config': self.config.get('feature_engineering.bins_config', None),
            'use_chi2_selection': self.config.get('feature_engineering.use_chi2_selection', False),
            'chi2_k': self.config.get('feature_engineering.chi2_k', 20),
            'use_mutual_info_selection': self.config.get('feature_engineering.use_mutual_info_selection', False),
            'mutual_info_k': self.config.get('feature_engineering.mutual_info_k', 20),
            'use_importance_selection': self.config.get('feature_engineering.use_importance_selection', False),
            'importance_threshold': self.config.get('feature_engineering.importance_threshold', 0.01),
            'importance_model': self.config.get('feature_engineering.importance_model', 'random_forest'),
            'use_variance_selection': self.config.get('feature_engineering.use_variance_selection', False),
            'variance_threshold': self.config.get('feature_engineering.variance_threshold', 0.01),
            'scale_features': self.config.get('feature_engineering.scale_features', False),
            'scale_method': self.config.get('feature_engineering.scale_method', 'standard'),
            'use_pca': self.config.get('feature_engineering.use_pca', False),
            'pca_components': self.config.get('feature_engineering.pca_components', 0.95),
            'missing_rate_threshold': self.config.get('feature_engineering.missing_rate_threshold', 0.3),
            'iv_threshold': self.config.get('feature_engineering.iv_threshold', 0.1),
            'corr_threshold': self.config.get('feature_engineering.corr_threshold', 0.8),
            'use_f_regression_selection': self.config.get('feature_engineering.use_f_regression_selection', False),
            'f_regression_k': self.config.get('feature_engineering.f_regression_k', 20),
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """在训练数据上学习所有特征工程规则"""
        logging.info("开始拟合特征工程管道...")
        
        # 从配置文件获取任务类型
        self.task_type_ = self.config.get('modeling.task_type', 'classification')
        logging.info(f"配置文件指定任务类型: {self.task_type_}")
        
        X_processed = X.copy()
        
        # 0. 数据预处理
        logging.info("步骤0: 数据预处理...")
        self.preprocessor_ = DataPreprocessor(
            missing_strategy='median',
            outlier_method=self.fe_config['outlier_method'],
            outlier_threshold=self.fe_config['outlier_threshold'],
            transform_method=self.fe_config['transform_method'],
            scale_method=self.fe_config['scale_method']
        )
        
        # 只在训练集上学习预处理规则
        self.preprocessor_.fit(X_processed)
        X_processed = self.preprocessor_.transform(X_processed)
        
        # 存储原始特征名
        self.feature_names_ = X_processed.columns.tolist()
        
        # 1. 特征增加
        logging.info("步骤1: 特征增加...")
        if self.fe_config['create_polynomial_features']:
            degree = self.fe_config['polynomial_degree']
            poly_features = FeatureGenerator.create_polynomial_features(
                X_processed, degree=degree
            )
            X_processed = pd.concat([X_processed, poly_features], axis=1)
        
        if self.fe_config['create_interaction_features']:
            interaction_features = FeatureGenerator.create_interaction_features(
                X_processed, interaction_pairs=self.fe_config['interaction_pairs']
            )
            X_processed = pd.concat([X_processed, interaction_features], axis=1)
        
        if self.fe_config['create_statistical_features']:
            stat_features = FeatureGenerator.create_statistical_features(
                X_processed, group_cols=self.fe_config['statistical_group_cols']
            )
            X_processed = pd.concat([X_processed, stat_features], axis=1)
        
        if self.fe_config['create_binning_features']:
            bin_features = FeatureGenerator.create_binning_features(
                X_processed, bins_config=self.fe_config['bins_config']
            )
            X_processed = pd.concat([X_processed, bin_features], axis=1)
        
        logging.info(f"特征增加后特征数量: {len(X_processed.columns)}")
        
        # 2. 特征选择 - 多维度筛选
        logging.info("步骤2: 特征选择...")
        
        # 2.1 基于缺失率筛选
        missing_threshold = self.fe_config['missing_rate_threshold']
        missing_cols = FeatureSelector.select_by_missing_rate(X_processed, missing_threshold)
        X_processed = X_processed.drop(columns=missing_cols)
        
        # 2.2 基于IV值筛选（仅分类任务）
        if self.task_type_ == 'classification':
            iv_threshold = self.fe_config['iv_threshold']
            iv_cols = FeatureSelector.select_by_iv(X_processed, y, iv_threshold)
            X_processed = X_processed.drop(columns=iv_cols)
        
        # 2.3 基于相关性筛选
        corr_threshold = self.fe_config['corr_threshold']
        corr_cols = FeatureSelector.select_by_correlation(X_processed, corr_threshold)
        X_processed = X_processed.drop(columns=corr_cols)
        
        # 2.4 基于方差筛选
        if self.fe_config['use_variance_selection']:
            variance_threshold = self.fe_config['variance_threshold']
            var_cols = FeatureSelector.select_by_variance(X_processed, variance_threshold)
            X_processed = X_processed.drop(columns=var_cols)
        
        # 2.5 基于卡方检验选择（仅分类任务）
        if self.fe_config['use_chi2_selection'] and self.task_type_ == 'classification':
            chi2_features = FeatureSelector.select_by_chi2(
                X_processed, y, self.fe_config['chi2_k'], self.task_type_
            )
            X_processed = X_processed[chi2_features]
        
        # 2.6 基于互信息选择
        if self.fe_config['use_mutual_info_selection']:
            mi_features = FeatureSelector.select_by_mutual_info(
                X_processed, y, self.fe_config['mutual_info_k'], self.task_type_
            )
            X_processed = X_processed[mi_features]
        
        # 2.7 基于特征重要性选择
        if self.fe_config['use_importance_selection']:
            importance_features = FeatureSelector.select_by_importance(
                X_processed, y, self.fe_config['importance_threshold'],
                self.fe_config['importance_model'], self.task_type_
            )
            X_processed = X_processed[importance_features]
        
        # 2.8 基于特征-目标相关性选择
        if self.fe_config['use_f_regression_selection']:
            if self.task_type_ == 'regression':
                f_features = FeatureSelector.select_by_f_regression(
                    X_processed, y, self.fe_config['f_regression_k']
                )
            else:
                f_features = FeatureSelector.select_by_f_classif(
                    X_processed, y, self.fe_config['f_regression_k']
                )
            X_processed = X_processed[f_features]
        
        logging.info(f"特征选择后特征数量: {len(X_processed.columns)}")
        
        # 3. PCA降维
        if self.fe_config['use_pca']:
            logging.info("步骤3: PCA降维...")
            pca_components = self.fe_config['pca_components']
            if 0 < pca_components < 1:
                n_components = min(int(pca_components * len(X_processed.columns)), len(X_processed.columns))
            else:
                n_components = min(int(pca_components), len(X_processed.columns))
            
            self.pca_model_ = PCA(n_components=n_components, random_state=42)
            self.pca_model_.fit(X_processed)
            logging.info(f"PCA降维到 {n_components} 个主成分")
        
        # 存储最终选择的特征
        self.selected_features_ = X_processed.columns.tolist()
        logging.info(f"最终选择特征数量: {len(self.selected_features_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        if self.selected_features_ is None:
            raise ValueError("必须先调用fit方法")
        
        X_processed = X.copy()
        
        # 应用预处理
        X_processed = self.preprocessor_.transform(X_processed)
        
        # 1. 特征增加（使用训练阶段学到的规则）
        if self.fe_config['create_polynomial_features']:
            degree = self.fe_config['polynomial_degree']
            poly_features = FeatureGenerator.create_polynomial_features(
                X_processed, degree=degree
            )
            X_processed = pd.concat([X_processed, poly_features], axis=1)
        
        if self.fe_config['create_interaction_features']:
            interaction_features = FeatureGenerator.create_interaction_features(
                X_processed, interaction_pairs=self.fe_config['interaction_pairs']
            )
            X_processed = pd.concat([X_processed, interaction_features], axis=1)
        
        if self.fe_config['create_statistical_features']:
            stat_features = FeatureGenerator.create_statistical_features(
                X_processed, group_cols=self.fe_config['statistical_group_cols']
            )
            X_processed = pd.concat([X_processed, stat_features], axis=1)
        
        if self.fe_config['create_binning_features']:
            bin_features = FeatureGenerator.create_binning_features(
                X_processed, bins_config=self.fe_config['bins_config']
            )
            X_processed = pd.concat([X_processed, bin_features], axis=1)
        
        # 2. 应用特征选择（只选择训练阶段确定的特征）
        # 确保所有需要的特征都存在
        missing_features = [f for f in self.selected_features_ if f not in X_processed.columns]
        if missing_features:
            # 对于缺失的特征，用0填充
            for f in missing_features:
                X_processed[f] = 0
        
        X_processed = X_processed[self.selected_features_]
        
        # 3. 应用PCA降维
        if self.pca_model_ is not None:
            X_processed = pd.DataFrame(
                self.pca_model_.transform(X_processed),
                columns=[f"PC_{i+1}" for i in range(self.pca_model_.n_components_)],
                index=X_processed.index
            )
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换训练数据"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """获取输出特征名"""
        if self.pca_model_ is not None:
            return [f"PC_{i+1}" for i in range(self.pca_model_.n_components_)]
        else:
            return self.selected_features_ if self.selected_features_ else []


class FeatureGenerator:
    """特征生成器，负责创建新的特征"""
    
    @staticmethod
    def create_polynomial_features(data: pd.DataFrame, degree: int = 2, 
                                 include_bias: bool = False) -> pd.DataFrame:
        """
        创建多项式特征
        
        Args:
            data: 输入特征数据
            degree: 多项式次数
            include_bias: 是否包含偏差项
            
        Returns:
            包含多项式特征的DataFrame
        """
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(data)
        feature_names = poly.get_feature_names_out(data.columns)
        return pd.DataFrame(poly_features, columns=feature_names, index=data.index)
    
    @staticmethod
    def create_interaction_features(data: pd.DataFrame, interaction_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            data: 输入特征数据
            interaction_pairs: 需要创建交互的特征对，如果为None则创建所有可能的交互
            
        Returns:
            包含交互特征的DataFrame
        """
        data_copy = data.copy()
        
        # 只选择数值列进行交互特征创建
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:  # 需要至少2个数值列才能创建交互特征
            return pd.DataFrame(index=data_copy.index)  # 返回空的DataFrame但保持索引一致
            
        # 只使用数值列创建交互特征
        numeric_data = data_copy[numeric_cols]
        result_df = pd.DataFrame(index=data_copy.index)  # 预创建结果DataFrame
        
        if interaction_pairs is None:
            # 创建所有可能的二阶交互
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    interaction_name = f"{col1}_x_{col2}"
                    result_df[interaction_name] = numeric_data[col1] * numeric_data[col2]
        else:
            # 创建指定的交互特征，只处理数值列
            for col1, col2 in interaction_pairs:
                if col1 in numeric_cols and col2 in numeric_cols:
                    interaction_name = f"{col1}_x_{col2}"
                    result_df[interaction_name] = numeric_data[col1] * numeric_data[col2]
        
        return result_df
    
    @staticmethod
    def create_statistical_features(data: pd.DataFrame, group_cols: List[str] = None) -> pd.DataFrame:
        """
        创建统计特征（均值、标准差、偏度、峰度等）
        
        Args:
            data: 输入特征数据
            group_cols: 用于分组的列，如果为None则基于所有数值列创建统计特征
            
        Returns:
            包含统计特征的DataFrame
        """
        data_copy = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # 创建全局统计特征
        for col in numeric_cols:
            data_copy[f"{col}_log"] = np.log1p(np.abs(data[col]))
            data_copy[f"{col}_sqrt"] = np.sqrt(np.abs(data[col]))
            data_copy[f"{col}_square"] = data[col] ** 2
            
            # 标准化特征
            if data[col].std() > 0:
                data_copy[f"{col}_zscore"] = (data[col] - data[col].mean()) / data[col].std()
        
        # 创建分组统计特征
        if group_cols and all(col in data.columns for col in group_cols):
            for col in numeric_cols:
                if col not in group_cols:
                    group_stats = data.groupby(group_cols)[col].agg(['mean', 'std', 'min', 'max'])
                    group_stats.columns = [f"{col}_group_{stat}" for stat in group_stats.columns]
                    
                    # 合并统计特征
                    data_copy = data_copy.merge(
                        group_stats, 
                        left_on=group_cols, 
                        right_index=True,
                        how='left'
                    )
        
        return data_copy
    
    @staticmethod
    def create_binning_features(data: pd.DataFrame, bins_config: Dict[str, int] = None) -> pd.DataFrame:
        """
        创建分箱特征
        
        Args:
            data: 输入特征数据
            bins_config: 分箱配置，格式为{列名: 箱数}
            
        Returns:
            包含分箱特征的DataFrame
        """
        data_copy = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if bins_config is None:
            bins_config = {col: 5 for col in numeric_cols}
        
        for col, n_bins in bins_config.items():
            if col in numeric_cols:
                try:
                    # 使用等频分箱
                    data_copy[f"{col}_bin"] = pd.qcut(
                        data[col], 
                        q=n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                except ValueError:
                    # 如果等频分箱失败，使用等宽分箱
                    data_copy[f"{col}_bin"] = pd.cut(
                        data[col], 
                        bins=n_bins, 
                        labels=False
                    )
        
        return data_copy
    
    @staticmethod
    def create_aggregation_features(data: pd.DataFrame, window_cols: List[str] = None) -> pd.DataFrame:
        """
        创建聚合特征（滑动窗口统计）
        
        Args:
            data: 输入特征数据
            window_cols: 需要创建滑动窗口特征的列
            
        Returns:
            包含聚合特征的DataFrame
        """
        data_copy = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if window_cols is None:
            window_cols = numeric_cols
        
        for col in window_cols:
            if col in numeric_cols:
                # 创建滚动统计特征
                data_copy[f"{col}_rolling_mean_3"] = data[col].rolling(window=3, min_periods=1).mean()
                data_copy[f"{col}_rolling_std_3"] = data[col].rolling(window=3, min_periods=1).std()
                data_copy[f"{col}_rolling_min_3"] = data[col].rolling(window=3, min_periods=1).min()
                data_copy[f"{col}_rolling_max_3"] = data[col].rolling(window=3, min_periods=1).max()
                
                # 创建扩展窗口统计特征
                data_copy[f"{col}_expanding_mean"] = data[col].expanding(min_periods=1).mean()
                data_copy[f"{col}_expanding_std"] = data[col].expanding(min_periods=1).std()
        
        return data_copy


class FeatureReducer:
    """特征降维器"""
    
    @staticmethod
    def pca_reduction(data: pd.DataFrame, n_components: int = 0.95, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, PCA]:
        """
        PCA降维
        
        Args:
            data: 输入数据
            n_components: 保留的组件数量或解释的方差比例
            random_state: 随机种子
            
        Returns:
            降维后的数据和PCA模型
        """
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_data = pca.fit_transform(data)
        
        # 生成新的特征名
        feature_names = [f"PC_{i+1}" for i in range(reduced_data.shape[1])]
        
        return pd.DataFrame(reduced_data, columns=feature_names, index=data.index), pca
    
    @staticmethod
    def select_top_features_by_variance(data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        基于方差选择特征
        
        Args:
            data: 输入数据
            threshold: 方差阈值
            
        Returns:
            选择的特征列表
        """
        variances = data.var()
        return variances[variances > threshold].index.tolist()


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
    
    def __init__(self, missing_strategy: str = 'median', outlier_method: str = 'iqr', 
                 outlier_threshold: float = 1.5, transform_method: str = 'yeo-johnson',
                 scale_method: str = 'standard', encoding_method: str = 'ordinal'):
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.transform_method = transform_method
        self.scale_method = scale_method
        self.encoding_method = encoding_method  # 新增：编码策略选择
        
        # 存储训练阶段学到的转换器
        self.imputers_ = {}
        self.encoders_ = None
        self.scalers_ = {}
        self.outlier_bounds_ = {}
        self.transformers_ = {}
        self.categorical_cols_ = None  # 存储类别列名
        
    def fit(self, X: pd.DataFrame, y=None):
        """在训练数据上学习所有预处理规则"""
        X_copy = X.copy()
        
        # 1. 识别数值和类别列
        numeric_cols = X_copy.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X_copy.select_dtypes(exclude=np.number).columns.tolist()
        
        # 2. 学习缺失值填充规则
        if len(numeric_cols) > 0:
            from sklearn.impute import SimpleImputer
            self.imputers_['numeric'] = SimpleImputer(strategy=self.missing_strategy)
            self.imputers_['numeric'].fit(X_copy[numeric_cols])
        
        # 3. 学习类别编码规则（根据encoding_method选择编码器）
        if len(self.categorical_cols_) > 0:
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
            
            # 先填充缺失值
            X_cat_imputed = self.imputers_['categorical'].transform(X_copy[self.categorical_cols_])
            
            if self.encoding_method == 'onehot':
                # 独热编码
                self.encoders_ = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    drop='first'  # 避免虚拟变量陷阱
                )
            else:
                # 默认使用序号编码
                self.encoders_ = OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1  # 未知类别编码为-1
                )
            
            self.encoders_.fit(X_cat_imputed)
        
        # 4. 学习异常值处理边界
        for col in numeric_cols:
            if self.outlier_method == 'iqr':
                Q1 = X_copy[col].quantile(0.25)
                Q3 = X_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds_[col] = {
                    'lower': Q1 - self.outlier_threshold * IQR,
                    'upper': Q3 + self.outlier_threshold * IQR
                }
        
        # 5. 学习特征变换规则
        numeric_cols_for_transform = X_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_for_transform) > 0:
            from sklearn.preprocessing import PowerTransformer
            self.transformers_['power'] = PowerTransformer(method=self.transform_method)
            self.transformers_['power'].fit(X_copy[numeric_cols_for_transform])
        
        # 6. 学习特征缩放规则
        if len(numeric_cols) > 0:
            if self.scale_method == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scalers_['main'] = StandardScaler()
            elif self.scale_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scalers_['main'] = MinMaxScaler()
            elif self.scale_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scalers_['main'] = RobustScaler()
            
            self.scalers_['main'].fit(X_copy[numeric_cols])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用学到的规则转换数据"""
        X_copy = X.copy()
        
        # 1. 应用缺失值填充
        numeric_cols = X_copy.select_dtypes(include=np.number).columns
        
        if 'numeric' in self.imputers_ and len(numeric_cols) > 0:
            X_copy[numeric_cols] = self.imputers_['numeric'].transform(X_copy[numeric_cols])
        
        # 2. 应用类别编码（根据编码策略）
        if self.encoders_ is not None and len(self.categorical_cols_) > 0:
            X_cat_imputed = self.imputers_['categorical'].transform(X_copy[self.categorical_cols_])
            
            if self.encoding_method == 'onehot':
                # 独热编码会生成新的列名
                encoded_features = self.encoders_.transform(X_cat_imputed)
                feature_names = self.encoders_.get_feature_names_out(self.categorical_cols_)
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X_copy.index)
                
                # 删除原始类别列，添加编码后的列
                X_copy = X_copy.drop(columns=self.categorical_cols_)
                X_copy = pd.concat([X_copy, encoded_df], axis=1)
            else:
                # 序号编码保持原始列名
                X_copy[self.categorical_cols_] = self.encoders_.transform(X_cat_imputed)
        
        # 3. 应用异常值处理
        for col, bounds in self.outlier_bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(bounds['lower'], bounds['upper'])
        
        # 4. 应用特征变换
        if 'power' in self.transformers_ and len(numeric_cols) > 0:
            X_copy[numeric_cols] = self.transformers_['power'].transform(X_copy[numeric_cols])
        
        # 5. 应用特征缩放
        if 'main' in self.scalers_ and len(numeric_cols) > 0:
            X_copy[numeric_cols] = self.scalers_['main'].transform(X_copy[numeric_cols])
        
        return X_copy
