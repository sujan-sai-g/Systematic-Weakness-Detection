import os
from pathlib import Path

RESULTS_BASE = "/home/IAIS/sgannamane/results/tmlr_2025/bdd100k/"
DATA_BASE = "/data/share/image_datasets/BDD/bdd100k/"
PEDESTRIAN_CROPS_FOLDER = "/home/IAIS/sgannamane/data/cvpr/"
project_root = Path(__file__).parent.parent.parent

config = {
    "seed": 100,
    
    "experiment": {
        "name": "BDD100K_pedestrian_detection",
        "description": "SliceLine analysis for BDD100K pedestrian detection",
        "version": "1.0",
    },
    
    "include_instance_id": True,
    
    "model": {
        "type": "bdd100k_detection",
        "model_name": "ConvNeXt-T",  
        "device": "cuda:0",
        'confidence_threshold': 0.5,
        'iou_threshold': 0.5,
        "predictions_json": os.path.join(RESULTS_BASE, "ConvNeXt-T_preds.json"), # downloaded from https://github.com/SysCV/bdd100k-models/tree/main/det)
        "matched_predictions_csv": os.path.join(RESULTS_BASE, "ConvNeXt-T_preds_matched.csv"),   
        "labels_json_path": os.path.join(DATA_BASE, "labels/detection20/det_v2_val_release.json"),
        "labels_csv_path": os.path.join(RESULTS_BASE, "bdd100k_val_gt.csv")

    },
    
    "output": RESULTS_BASE,
    
    "dataset": {
        "name": "BDD100K",
        'category': 'pedestrian',
        "image_folder": os.path.join(DATA_BASE, "images/100k/val/"),
        "pedestrian_crops_folder": os.path.join(PEDESTRIAN_CROPS_FOLDER, "bdd100k_val_crops_square_clear_bg/"),
        # "labels_json_folder": os.path.join(DATA_BASE, "labels/detection20/det_v2_val_release.json"),
        "image_pattern": "*.png"
    },  
    
    
    "metadata": {
        "image_embeddings_path": os.path.join(RESULTS_BASE, "bdd_image_embeddings.pkl"),
        "ontology_path": "data/ODDs/",
        "prompts_path": "data/prompts/",
        'object': 'ad_person',  # Use existing pedestrian prompts and ontology
        "metadata_file": "bdd_metadata.csv"
    },
    
    "sliceline_input": {
        "precision_data_path": str(project_root / "data" / "precision_and_recall_csvs" / "precisions_and_recall_bdd100k.csv"),    
        "clip_data_path": os.path.join(RESULTS_BASE, "bdd100k_original_full_file.csv")
    },
    
    "debug": {
        "enabled": False,
        "max_images": 100
    },
    
    "sliceline": {
        "alpha": 0.95,           # Used by SliceLineR and SliceLineRplus classes
        "level": 2,              # Used by SliceLineR and SliceLineRplus classes  
        "top_k": 100,            # Used by top_slices() method
        "max_k_value": 100,      # Used for evaluation/plotting
        "err_column": "detection_type",  # Default error column name
        "inv_thresh": 0.1        # Used by NaivePPIestimator
    },
    
    "label_mapping": {
        "gender": "gender",
        "age": "age",
        "skin-color": "skin-color",
        "clothing-color": "clothing-color",
        "blurry": "blurry",
        "occluded": "occluded",
    }
}