"""
CLIP embedding generation for images
"""

import glob
import random
import clip
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Optional

class CLIPEmbeddingGenerator:
    """Generate CLIP embeddings for images"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device(config.get('device', 'auto'))
        self.model_name = config.get('clip_model', 'ViT-L/14')
        self.seed = config.get('seed', 100)
        self.setup_reproducibility()
        self.load_clip_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == 'auto':
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def setup_reproducibility(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
    def load_clip_model(self):
        """Load CLIP model and preprocessing"""
        print(f"Loading CLIP model: {self.model_name}")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        print(f"Model loaded on device: {self.device}")
        
    def get_image_paths(self, image_folder: str, pattern: str = "*.png") -> List[str]:
        """Get sorted list of image paths"""
        search_pattern = str(Path(image_folder) / pattern)
        filenames = glob.glob(search_pattern)
        filenames.sort()
        print(f"Found {len(filenames)} images")
        return filenames
        
    def encode_single_image(self, image_path: str) -> np.ndarray:
        """Encode a single image to CLIP embedding"""
        with torch.no_grad():
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            return image_features.cpu().numpy()
    
    def generate_embeddings(self, image_paths: List[str], output_path: str, batch_size: int = 1) -> pd.DataFrame:       
        print(f"Generating CLIP embeddings for {len(image_paths)} images...")
        
        filename_list = []
        image_features_list = []
        
        for image_path in tqdm.tqdm(image_paths, desc="Generating embeddings"):
            try:
                # Extract filename without extension
                filename = Path(image_path).stem
                
                # Generate embedding
                image_features = self.encode_single_image(image_path)
                
                filename_list.append(filename)
                image_features_list.append(image_features)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame({
            'Filename': filename_list,
            'image_features_list': image_features_list
        })
        
        # Save to pickle
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(str(output_path))
        
        print(f"✓ Saved {len(df)} embeddings to {output_path}")
        return df
    
    def generate_embeddings_from_folder(self, image_folder: str, output_path: str, 
                                      pattern: str = "*.png") -> pd.DataFrame:
        """Generate embeddings for all images in a folder"""
        image_paths = self.get_image_paths(image_folder, pattern)
        return self.generate_embeddings(image_paths, output_path)