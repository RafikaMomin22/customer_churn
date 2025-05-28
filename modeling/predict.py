import pandas as pd
import logging
import pickle
from typing import Optional, Tuple
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Handles churn probability predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_pipeline = None
        
    def load_artifacts(self, model_path: str, pipeline_path: str) -> bool:
        """Load trained model and feature pipeline"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            with open(pipeline_path, 'rb') as f:
                self.feature_pipeline = pickle.load(f)
                
            logger.info("Successfully loaded model and feature pipeline")
            return True
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            return False
            
    def predict_churn(self, customer_data
