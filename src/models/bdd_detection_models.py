import json
import pandas as pd
import numpy as np
import tqdm
from typing import List, Dict, Any, Tuple
from .base_model import BaseModel


class BDD100kDetectionModel(BaseModel):
    """Model wrapper for BDD100k detection models with prediction matching"""
    
    def __init__(self, config):
        super().__init__(config)
        print(config)
        self.model_name = config.get('model_name', 'ConvNeXt-T')
        print(config.get('predictions_json'))
        self.path_to_preds = config.get('predictions_json')
        self.path_to_gt = config.get('labels_json_path')
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.gt_path = config.get('labels_csv_path')

        if not self.path_to_preds or not self.path_to_gt:
            raise ValueError("Both 'path_to_preds' and 'path_to_gt' must be specified in config")
    
    def setup_model(self):
        """Setup is handled via JSON file loading - no model initialization needed"""
        print(f"BDD100k Detection Model ({self.model_name}) ready for processing")
    
    def _read_predictions_json(self) -> pd.DataFrame:
        """Read predictions from JSON and convert to DataFrame"""
        data_lists = {
            'Filename': [],
            'pred_id': [],
            'pred_bbox_x1': [],
            'pred_bbox_y1': [],
            'pred_bbox_x2': [],
            'pred_bbox_y2': [],
            'score': [],
            'category': []
        }
        
        with open(self.path_to_preds, "r") as file:
            data = json.load(file)
            images = data['frames']
            
            for image in tqdm.tqdm(images, desc="Processing predictions"):
                filename = image['name']
                for annotation in image['labels']:
                    if annotation['category'] == "pedestrian":
                        data_lists['Filename'].append(filename.split(".")[0])
                        data_lists['pred_id'].append(annotation['id'])
                        data_lists['pred_bbox_x1'].append(annotation['box2d']['x1'])
                        data_lists['pred_bbox_y1'].append(annotation['box2d']['y1'])
                        data_lists['pred_bbox_x2'].append(annotation['box2d']['x2'])
                        data_lists['pred_bbox_y2'].append(annotation['box2d']['y2'])
                        data_lists['score'].append(annotation['score'])
                        data_lists['category'].append(annotation['category'])
        
        return pd.DataFrame(data_lists)
    
    def _read_ground_truth_json(self) -> pd.DataFrame:
        """Read ground truth from JSON and convert to DataFrame"""
        data_lists = {
            'Filename': [],
            'id': [],
            'x1_list': [],
            'y1_list': [],
            'x2_list': [],
            'y2_list': [],
            'category': [],
            'occluded': []
        }
        
        with open(self.path_to_gt, "r") as file:
            data = json.load(file)
            
            for image in tqdm.tqdm(data, desc="Processing ground truth"):
                filename = image['name']
                for annotation in image['labels']:
                    if annotation['category'] == "pedestrian":
                        data_lists['Filename'].append(filename.split(".")[0])
                        data_lists['id'].append(annotation['id'])
                        data_lists['x1_list'].append(annotation['box2d']['x1'])
                        data_lists['y1_list'].append(annotation['box2d']['y1'])
                        data_lists['x2_list'].append(annotation['box2d']['x2'])
                        data_lists['y2_list'].append(annotation['box2d']['y2'])
                        data_lists['category'].append(annotation['category'])
                        data_lists['occluded'].append(annotation['attributes']['occluded'])
        
        return pd.DataFrame(data_lists)
    
    def _instance_iou_image(self, detections: np.array, gt_info: np.array, 
                           gt_bbox_ids: List[str] = None) -> pd.DataFrame:
        """
        Calculate IoU between detections and ground truth boxes
        
        Args:
            detections: Array with shape [n_detections, 6] - [class, confidence, x1, y1, x2, y2]
            gt_info: Array with shape [n_gt, 4] - [x1, y1, x2, y2]
            gt_bbox_ids: List of ground truth bbox IDs
        
        Returns:
            DataFrame with matched predictions
        """
        if detections.shape[0] == 0:
            return pd.DataFrame(columns=self.get_csv_columns())
        
        orig_dets = detections[:, 0:6].astype(float).copy()
        detections = detections[:, 0:6].astype(float)
        
        # Extract bounding boxes and confidence scores
        det_bboxes = detections[:, 2:].astype(float)
        confidence = detections[:, 1].astype(np.float32)
        
        gt_bboxes = gt_info[:, :].astype(float)
        
        # Handle case where no ground truth exists
        if gt_bboxes.shape[0] == 0:
            gt_bboxes = np.asarray([[-1, -1, -1, -1]])
            gt_bbox_ids = ["-1"]
        
        if gt_bbox_ids is None:
            gt_bbox_ids = [str(x) for x in range(gt_bboxes.shape[0])]
        
        # Sort by confidence (descending)
        sorted_ind = np.argsort(-confidence)
        det_bboxes = det_bboxes[sorted_ind, :]
        confidence = confidence[sorted_ind]
        orig_dets = orig_dets[sorted_ind, :]
        
        n_det = det_bboxes.shape[0]
        detected_gt_labels = set()
        
        # Store results
        results = {
            'pred_bbox_x1': [],
            'pred_bbox_y1': [],
            'pred_bbox_x2': [],
            'pred_bbox_y2': [],
            'instance_id_match': [],
            'pred_confidence': [],
            'gt_iou_match': []
        }
        
        for det_id in range(n_det):
            bb = det_bboxes[det_id, :]
            bbox_actual = orig_dets[det_id, :]
            
            # Compute IoU between predicted bbox and all gt bboxes
            ixmin = np.maximum(gt_bboxes[:, 0], bb[0])
            iymin = np.maximum(gt_bboxes[:, 1], bb[1])
            ixmax = np.minimum(gt_bboxes[:, 2], bb[2])
            iymax = np.minimum(gt_bboxes[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                   (gt_bboxes[:, 3] - gt_bboxes[:, 1]) - inters)
            
            intersec_over_union = inters / uni
            ovmax = np.max(intersec_over_union)
            jmax = np.argmax(intersec_over_union)
            
            # Store bbox coordinates and confidence
            results['pred_bbox_x1'].append(bbox_actual[2])
            results['pred_bbox_y1'].append(bbox_actual[3])
            results['pred_bbox_x2'].append(bbox_actual[4])
            results['pred_bbox_y2'].append(bbox_actual[5])
            results['pred_confidence'].append(float(confidence[det_id]))
            
            gt_box_id = gt_bbox_ids[jmax]
            
            # Check if this is a valid match
            if gt_box_id not in detected_gt_labels and ovmax > self.iou_threshold:
                detected_gt_labels.add(gt_box_id)
                results['instance_id_match'].append(gt_box_id)
                results['gt_iou_match'].append(float(ovmax))
            else:
                results['instance_id_match'].append(np.nan)
                results['gt_iou_match'].append(np.nan)
        
        return pd.DataFrame(results)
    
    def _match_predictions(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Match predictions with ground truth using IoU"""
        pred_df['class'] = 1
        
        print(f"Number of predictions before matching: {len(pred_df)}")
        print(f"Number of ground truth annotations: {len(gt_df)}")
        
        unique_filenames = gt_df['Filename'].unique()
        df_list = []
        
        for filename in tqdm.tqdm(unique_filenames, desc="Matching predictions"):
            gt = gt_df[gt_df['Filename'] == filename]
            preds = pred_df[pred_df['Filename'] == filename]
            
            if len(preds) == 0:
                continue
                
            instance_ids = gt['id'].tolist()
            
            # Prepare data for IoU calculation
            gt_columns = ['x1_list', 'y1_list', 'x2_list', 'y2_list']
            pred_columns = ['class', 'score', 'pred_bbox_x1', 'pred_bbox_y1', 'pred_bbox_x2', 'pred_bbox_y2']
            
            gt_array = gt[gt_columns].to_numpy()
            preds_array = preds[pred_columns].to_numpy()
            
            # Match predictions to ground truth
            matched_preds = self._instance_iou_image(preds_array, gt_array, instance_ids)
            
            # Merge with original predictions
            merged = pd.merge(
                left=preds, 
                right=matched_preds, 
                how='outer',
                left_on=['pred_bbox_x1', 'pred_bbox_y1', 'pred_bbox_x2', 'pred_bbox_y2'],
                right_on=['pred_bbox_x1', 'pred_bbox_y1', 'pred_bbox_x2', 'pred_bbox_y2']
            )
            df_list.append(merged)
        
        if not df_list:
            return pd.DataFrame(columns=self.get_csv_columns())
            
        result_df = pd.concat(df_list, ignore_index=True)
        print(f"Number of predictions after matching: {len(result_df)}")
        
        return result_df
    
    def predict_to_csv(self, image_paths: List[str], output_path: str) -> pd.DataFrame:
        """
        Process BDD100k detection data and save matched predictions to CSV
        
        Args:
            image_paths: Not used for this model (data comes from JSON files)
            output_path: Path to save the output CSV
            
        Returns:
            DataFrame with matched predictions
        """
        print(f"Processing BDD100k detection data with {self.model_name}")
        
        # Read predictions and ground truth
        pred_df = self._read_predictions_json()
        gt_df = self._read_ground_truth_json()
        gt_df.to_csv(self.gt_path, index=False)
        # Match predictions with ground truth
        matched_df = self._match_predictions(gt_df, pred_df)
        
        # Clean up the DataFrame
        columns_to_drop = [col for col in ['Unnamed: 0', 'class', 'score', 'pred_id'] if col in matched_df.columns]
        if columns_to_drop:
            matched_df = matched_df.drop(columns_to_drop, axis=1)
        
        # Sort by filename and instance ID
        matched_df = matched_df.sort_values(by=['Filename', 'instance_id_match'])
        
        # Save to CSV
        matched_df.to_csv(output_path, index=False)
        print(f"Saved matched predictions to {output_path}")
        
        return matched_df
    
    def get_csv_columns(self) -> List[str]:
        """Return columns this model outputs"""
        return [
            'Filename', 'pred_bbox_x1', 'pred_bbox_y1', 'pred_bbox_x2', 'pred_bbox_y2',
            'category', 'instance_id_match', 'pred_confidence', 'gt_iou_match'
        ]