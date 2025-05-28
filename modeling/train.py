import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import logging
import pickle
from pathlib import Path
from config import settings
import mlflow
from datetime import datetime

logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """Handles training and evaluation of churn prediction model"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.metrics = None
        
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train random forest classifier with hyperparameter tuning"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize MLflow
            mlflow.set_tracking_uri(settings.MLFLOW_URI)
            mlflow.set_experiment("customer_churn_prediction")
            
            with mlflow.start_run():
                # Hyperparameter grid
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', None]
                }
                
                # Grid search
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                
                # Evaluate
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                self.metrics = {
                    'roc_auc': roc_auc_score(y_test, y_proba),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                # Log to MLflow
                mlflow.log_params(self.best_params)
                mlflow.log_metrics({
                    'roc_auc': self.metrics['roc_auc'],
                    'precision': self.metrics['classification_report']['weighted avg']['precision'],
                    'recall': self.metrics['classification_report']['weighted avg']['recall'],
                    'f1': self.metrics['classification_report']['weighted avg']['f1-score']
                })
                mlflow.sklearn.log_model(self.model, "random_forest_churn_model")
                
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
            
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
