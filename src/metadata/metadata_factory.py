from typing import Dict, Any
from .clip_embeddings import CLIPEmbeddingGenerator
from .metadata_generator import MetadataGenerator

class MetadataFactory:
    """Factory for creating metadata processing components"""
    
    @staticmethod
    def create_embedding_generator(config: Dict[str, Any]) -> CLIPEmbeddingGenerator:
        """Create CLIP embedding generator"""
        return CLIPEmbeddingGenerator(config)
    
    @staticmethod
    def create_metadata_generator(config: Dict[str, Any]) -> MetadataGenerator:
        """Create metadata generator"""
        return MetadataGenerator(config)