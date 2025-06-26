import argparse
import glob
import os
import urllib
import torch
import numpy as np
import pandas as pd
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src.utils.imagenet21k_helpers.semantic_loss import ImageNet21kSemanticSoftmax
import tqdm
from typing import List

from .base_model import BaseModel

class ImageNet21kModel(BaseModel):
    """Model wrapper for ImageNet21k models (like your current implementation)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_type = config.get('model_name', 'vit_base_patch16_224_miil_in21k')
        self.setup_model()
    
    def setup_model(self):
        """Initialize the ImageNet21k model and semantic processor"""
        # Download tree file if needed
        url = "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth"
        filename = "imagenet21k_miil_tree.pth"
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)
        
        # Setup semantic processor
        args = argparse.Namespace()
        args.tree_path = filename
        args.device = self.device
        self.semantic_processor = ImageNet21kSemanticSoftmax(args)
        
        # Load model
        self.model = timm.create_model(self.model_type, pretrained=True)
        self.model.eval()
        
        # Setup transforms
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
        print(f"Loaded {self.model_type} with {self.model.num_classes} classes")
    
    def predict_to_csv(self, image_paths: List[str], output_path: str) -> pd.DataFrame:
        """Run inference and save to CSV"""
        results_dict = {}
        
        for path in tqdm.tqdm(image_paths, desc=f"Running {self.model_name} inference"):
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(tensor)
                semantic_logit_list = self.semantic_processor.split_logits_to_semantic_logits(logits)
                
                # Use first hierarchy level
                logits_i = semantic_logit_list[0]
                probabilities = torch.nn.functional.softmax(logits_i[0], dim=0)
                top1_prob, top1_id = torch.topk(probabilities, 1)
                
                # Get class info
                top_class_number = self.semantic_processor.hierarchy_indices_list[0][top1_id[0]]
                top_class_name = self.semantic_processor.tree['class_list'][top_class_number]
                top_class_description = self.semantic_processor.tree['class_description'][top_class_name]
                
                # Store results (matches your format exactly)
                results_dict[path] = {
                    'pred_ImgNet21K_name': top_class_description,
                    'pred_name': 'person' if top_class_description == 'person' else 'not person',
                    'pred': 1 if top_class_description == 'person' else 0,
                    'Filename': path.split("/")[-1].split(".")[0],
                    'prob_notperson': 1 - np.round(probabilities[2].item(), 2),
                    'prob_person': np.round(probabilities[2].item(), 2)
                }
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(results_dict).T.reset_index(names='img_path')
        
        # Save CSV
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
        return df
    
    def get_csv_columns(self) -> List[str]:
        """Return columns this model outputs"""
        return [
            'img_path', 'pred_ImgNet21K_name', 'pred_name', 'pred', 
            'Filename', 'prob_notperson', 'prob_person'
        ]