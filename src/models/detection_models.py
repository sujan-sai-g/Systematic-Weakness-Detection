import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List
from .base_model import BaseModel

class BDD100kDetectionModel(BaseModel):
    """Model wrapper for BDD100k detection results (JSON format)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.predictions_json_path = config.get('predictions_json_path')
        self.gt_json_path = config.get('gt_json_path', None)
        self.target_category = config.get('target_category', 'pedestrian')
        
    def predict_to_csv(self, image_paths: List[str] = None, output_path: str = None) -> pd.DataFrame:
        """
        Convert BDD100k JSON predictions to CSV format
        Note: image_paths not used here since we read from JSON file directly
        """
        if not self.predictions_json_path or not output_path:
            raise ValueError("predictions_json_path and output_path are required")
            
        return self._read_pred_json_and_create_csv(output_path)
    
    def _read_pred_json_and_create_csv(self, output_path: str) -> pd.DataFrame:
        """Convert prediction JSON to CSV (matches your BDD100k implementation)"""
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        filename_list = []
        pred_id_list = []
        score_list = []
        category_list = []
        
        with open(self.predictions_json_path, "r") as file:
            json_data = file.read()
            data = json.loads(json_data)
            images = data['frames']
            
            for image in tqdm(images, desc="Processing BDD100k predictions"):
                filename = image['name']
                for id_anno in image['labels']:
                    if id_anno['category'] == self.target_category:
                        filename_list.append(filename.split(".")[0])
                        pred_id_list.append(id_anno['id'])
                        x1_list.append(id_anno['box2d']['x1'])
                        y1_list.append(id_anno['box2d']['y1'])
                        x2_list.append(id_anno['box2d']['x2'])
                        y2_list.append(id_anno['box2d']['y2'])
                        score_list.append(id_anno['score'])
                        category_list.append(id_anno['category'])
        
        df = pd.DataFrame({
            'Filename': filename_list,
            'pred_id': pred_id_list,
            'pred_bbox_x1': x1_list,
            'pred_bbox_y1': y1_list,
            'pred_bbox_x2': x2_list,
            'pred_bbox_y2': y2_list,
            'score': score_list,
            'category': category_list,
        })
        
        df.to_csv(output_path, index=False)
        print(f"Saved BDD100k predictions to {output_path}")
        return df
    
    def read_gt_and_create_csv(self, gt_output_path: str) -> pd.DataFrame:
        """Convert ground truth JSON to CSV"""
        if not self.gt_json_path:
            raise ValueError("gt_json_path required for ground truth processing")
            
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        filename_list = []
        pred_id_list = []
        category_list = []
        occluded_list = []
        
        with open(self.gt_json_path, "r") as file:
            json_data = file.read()
            data = json.loads(json_data)
            
            for image in tqdm(data, desc="Processing BDD100k ground truth"):
                filename = image['name']
                for id_anno in image['labels']:
                    if id_anno['category'] == self.target_category:
                        filename_list.append(filename.split(".")[0])
                        pred_id_list.append(id_anno['id'])
                        x1_list.append(id_anno['box2d']['x1'])
                        y1_list.append(id_anno['box2d']['y1'])
                        x2_list.append(id_anno['box2d']['x2'])
                        y2_list.append(id_anno['box2d']['y2'])
                        category_list.append(id_anno['category'])
                        occluded_list.append(id_anno['attributes']['occluded'])
        
        df = pd.DataFrame({
            'Filename': filename_list,
            'id': pred_id_list,
            'x1_list': x1_list,
            'y1_list': y1_list,
            'x2_list': x2_list,
            'y2_list': y2_list,
            'category': category_list,
            'occluded': occluded_list,
        })
        
        df.to_csv(gt_output_path, index=False)
        print(f"Saved BDD100k ground truth to {gt_output_path}")
        return df
    
    def get_csv_columns(self) -> List[str]:
        return [
            'Filename', 'pred_id', 'pred_bbox_x1', 'pred_bbox_y1', 
            'pred_bbox_x2', 'pred_bbox_y2', 'score', 'category'
        ]
        
        
class YOLODetectionModel(BaseModel):
    """Model wrapper for YOLO detection results (multiple JSON files)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.results_folder = config.get('results_folder')
        self.pedestrian_class_id = config.get('pedestrian_class_id', 0)
        self.image_prefix = config.get('image_prefix', 'night_')  # For ECP dataset
        
    def predict_to_csv(self, image_paths: List[str] = None, output_path: str = None) -> pd.DataFrame:
        """
        Convert YOLO JSON results to CSV format
        Note: image_paths not used here since we read from JSON files directly
        """
        if not self.results_folder or not output_path:
            raise ValueError("results_folder and output_path are required")
            
        return self._read_jsons_and_create_csv(output_path)
    
    def _read_jsons_and_create_csv(self, output_csv_path: str) -> pd.DataFrame:
        """Convert multiple JSON files to CSV"""
        # Lists to store the extracted data
        image_names = []
        box_ids = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        classes = []
        confidences = []
        folder_names = []
        
        # Recursively find all JSON files in the results folder
        json_files = list(Path(self.results_folder).rglob('*.json'))
        print(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for json_path in tqdm(json_files, desc="Processing YOLO predictions"):
            folder_name = json_path.parent.name
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract image name
                image_name = self.image_prefix + data.get('image_name', json_path.stem.replace('_result', ''))
                
                # Process each bounding box in the JSON
                if 'boxes' in data and len(data['boxes']) > 0:
                    for i, box in enumerate(data['boxes']):
                        # Only process pedestrian class
                        if ('classes' in data and i < len(data['classes']) and 
                            data['classes'][i] == self.pedestrian_class_id):
                            # Only process if the box has valid coordinates
                            if len(box) == 4:
                                image_names.append(image_name)
                                folder_names.append(folder_name)
                                box_ids.append(i)
                                x1_list.append(box[0])
                                y1_list.append(box[1])
                                x2_list.append(box[2])
                                y2_list.append(box[3])
                                
                                # Get confidence if available
                                if 'confidences' in data and i < len(data['confidences']):
                                    confidences.append(data['confidences'][i])
                                else:
                                    confidences.append(0.0)
                                    
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Folder': folder_names,
            'Filename': image_names,
            'box_id': box_ids,
            'x1': x1_list,
            'y1': y1_list,
            'x2': x2_list,
            'y2': y2_list,
            'confidence': confidences
        })
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"CSV file saved to {output_csv_path}")
        print(f"Total detection boxes: {len(df)}")
        print(f"Images processed: {df['Filename'].nunique()}")
        print(f"Folders processed: {df['Folder'].nunique()}")
        
        return df
    
    def get_csv_columns(self) -> List[str]:
        return [
            'Folder', 'Filename', 'box_id', 'x1', 'y1', 'x2', 'y2', 'confidence'
        ]