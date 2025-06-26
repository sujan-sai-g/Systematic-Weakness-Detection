# models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class BaseModel(ABC):
    """Simple base class for models that generate CSV predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('name', 'unknown')
        self.device = config.get('device', 'cuda:0')
        
    @abstractmethod
    def predict_to_csv(self, image_paths: List[str], output_path: str) -> pd.DataFrame:
        """
        Run inference on images and save results to CSV
        
        Args:
            image_paths: List of paths to images
            output_path: Path to save CSV file
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def get_csv_columns(self) -> List[str]:
        """Return the column names this model outputs in its CSV"""
        pass