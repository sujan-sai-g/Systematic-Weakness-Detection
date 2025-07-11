from typing import Dict, Any
from .base_model import BaseModel
from .imagenet21k_models import ImageNet21kModel
from .bdd_detection_models import BDD100kDetectionModel
# from .custom_model import CustomModel

class ModelFactory:
    """Simple factory for creating models that generate prediction CSVs"""
    
    _model_registry = {
        'imagenet21k': ImageNet21kModel,
        'bdd100k_detection': BDD100kDetectionModel,
        # 'custom': CustomModel,
    }
    
    @classmethod
    def create(cls, model_config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_config: Configuration dict with:
                - type: REQUIRED - 'imagenet21k', 'bdd100k_detection', or 'custom'
                - model_name: specific model (e.g., 'vit_base_patch16_224_miil_in21k')
                - device: 'cuda:0', 'cpu', etc.
                - **other model-specific args
        """
        model_type = model_config.get('type')
        
        if model_type is None:
            raise ValueError("Model type is required! Specify one of: " + 
                           ", ".join(cls._model_registry.keys()))
        
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: '{model_type}'. "
                           f"Available types: {', '.join(cls._model_registry.keys())}")
        
        return cls._model_registry[model_type](model_config)