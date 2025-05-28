import pandas as pd
import pyodbc
from sqlalchemy import create_engine
from config import settings
import logging
from typing import Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ChurnDataExtractor:
    """Extracts customer data for churn prediction"""
    
    def __init__(self):
        self.engine = create_engine(
            f"mssql+pyodbc://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_SERVER}/{settings.DB_NAME}?driver={settings.DB_DRIVER}"
        )
        
    def _get_churn_window(self) -> Tuple[datetime, datetime]:
        """Calculate observation and churn windows"""
        end_date = datetime.now() - timedelta(days=settings.CHURN_DAYS_OFFSET)
        obs_window_start = end_date - timedelta(days=settings.OBSERVATION_WINDOW_DAYS)
        churn_window_start = end_date
        churn_window_end = end_date + timedelta(days=settings.CHURN_WINDOW_DAYS)
        return obs_window_start, churn_window_end
        
    def extract_customer_data(self) -> Optional[pd.DataFrame]:
        """
        Extract customer behavioral data and churn labels
        Returns DataFrame with features and churn label
        """
        obs_start, churn_end = self._get_churn_window()
        
        query = f"""
        WITH customer_metrics AS (
            SELECT 
                c.customer_id,
                DATEDIFF(day, MAX(t.transaction_date), '{obs_start}') AS recency,
                COUNT(DISTINCT t.transaction_id) AS frequency,
                AVG(t.order_value) AS avg_order_value,
                SUM(t.order_value) AS monetary_value,
                COUNT(DISTINCT p.category) AS unique_categories,
                AVG(CASE WHEN p.category = 'High Margin' THEN 1 ELSE 0 END) AS high_margin_ratio
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            LEFT JOIN transaction_items ti ON t.transaction_id = ti.transaction_id
            LEFT JOIN products p ON ti.product_id = p.product_id
            WHERE t.transaction_date BETWEEN DATEADD(day, -{settings.OBSERVATION_WINDOW_DAYS}, '{obs_start}') AND '{obs_start}'
            GROUP BY c.customer_id
        ),
        churn_labels AS (
            SELECT 
                customer_id,
                CASE WHEN COUNT(transaction_id) = 0 THEN 1 ELSE 0 END AS churned
            FROM transactions
            WHERE transaction_date BETWEEN '{churn_window_start}' AND '{churn_window_end}'
            GROUP BY customer_id
        )
        SELECT 
            cm.*,
            COALESCE(cl.churned, 1) AS churned  -- Assume churn if no recent transactions
        FROM customer_metrics cm
        LEFT JOIN churn_labels cl ON cm.customer_id = cl.customer_id
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Successfully extracted {len(df)} customer records")
            return df
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return None
