import os

RESULTS_BASE = "/home/IAIS/sgannamane/results/tmlr_2025/celeba_results"
DATA_BASE = "/home/IAIS/sgannamane/data"

config = {
    "seed": 100,
    
    "experiment": {
        "name": "celeba_imagenet21k_slice_discovery",
        "description": "CelebA experiment with ImageNet21k model for slice discovery"
    },
    
    "model": {
        "type": "imagenet21k",
        "model_name": "vit_base_patch16_224_miil_in21k",  # or mobilenetv3_large_100_miil_in21k, tresnet_m_miil_in21k
        "device": "cuda:0"
    },
    
    "output": RESULTS_BASE,
    
    "dataset": {
        "name": "celeba",
        "image_folder": os.path.join(DATA_BASE, "clip_paper/celebAdataset/Img/img_align_celeba_png/"),
        "image_pattern": "*.png"
    },
    
    "predictions_csv": "celeba_predictions_vit.csv",
    
    "metadata": {
        "image_embeddings_path": os.path.join(RESULTS_BASE, "celebA_embedding_original.pkl"),
        "ontology_path": "data/ODDs/",
        "prompts_path": "data/prompts/",
        "object": "celebA_person",
        "metadata_file": "celebA_person_metadata_original.csv"
    },
    
    "sliceline_input": {
        "precision_data_path": os.path.join(RESULTS_BASE, "precisions_and_recall_celebA.csv"),
        "gt_data_path": os.path.join(RESULTS_BASE, "celeba_gt_metadata_with_performance.csv"),
        "clip_data_path": os.path.join(RESULTS_BASE, "celeba_clip_metadata_with_performance.csv")
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
        "Male": "gender",
        "Young": "age",
        "Pale_Skin": "skin-color",
        "Wearing_Hat": "wearing-hat",
        "Goatee": "goatee",
        "Beard": "beard",
        "Smiling": "smiling",
        "Bald": "bald",
        "Eyeglasses": "eye-glasses"
    }
}