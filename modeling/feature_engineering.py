import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import settings
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class ChurnFeatureEngineer:
    """Handles feature engineering for churn prediction"""
    
    def __init__(self):
        self.feature_pipeline = None
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Set up the feature engineering pipeline"""
        numeric_features = ['recency', 'frequency', 'avg_order_value', 
                          'monetary_value', 'unique_categories']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_features = ['high_margin_ratio']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        self.feature_pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        try:
            features = self.feature_pipeline.fit_transform(df)
            feature_names = self._get_feature_names()
            return pd.DataFrame(features, columns=feature_names, index=df.index)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
            
    def _get_feature_names(self) -> list:
        """Get feature names after transformation"""
        numeric_features = self.feature_pipeline.named_transformers_['num'].get_feature_names_out()
        cat_features = self.feature_pipeline.named_transformers_['cat'].get_feature_names_out()
        return list(numeric_features) + list(cat_features)
    
    def save_pipeline(self, path: str):
        """Save the feature pipeline to disk"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.feature_pipeline, f)
            logger.info(f"Saved feature pipeline to {path}")
        except Exception as e:
            logger.error(f"Failed to save feature pipeline: {e}")
            raise
