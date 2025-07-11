# Explanation about experiments

This subfolder contains Python scripts that either run partly or fully systematic weakness detection.

## Files

### `run_full_celeba_experiment.py`
**Purpose**: To obtain systematic weaknesses of the ViT Model (DuT) on celebA dataset

**What happens when you run it**: 
- Creates CLIP embeddings for all celebA images
- Creates metadata for given ODD, prompts for all embeddings (very minor variations in metadata might occur due to [CLIP reproducability issues](https://github.com/openai/CLIP/issues/13). To get exact slice sizes as in paper, please contact me to get metadata csv for CLIP)
- Runs inference on ViT model (DuT) for all celebA images
- Identifies weak slices with CLIP (SWD-1), Corrected (SWD-2). The merged (SWD-3) result is just a combination of both of these by removing duplicates and ordering by slice_score.
- For corrected (SWD-2), the precision and recall values are taken for celebA from the data/precision_and_recall_csvs/precisions_and_recall_celebA.csv
- Additionally GT slices can be calculated if the GT metadata is provided as csv (path defined in config) -- please contact if you are interested in getting this csv
- The celeba_config.py contains all the relevant parameters and file paths. Dataset root path and a path to save the results should be provided. Intermediate csv files are stored here while building embeddings, metadata, inference results and finally the identified slices. 

**Usage**:
```bash
python run_full_celeba_experiment.py
```

### `run_full_bdd100k_experiment.py`
**Purpose**: To obtain systematic weaknesses of the Pedestrian detection Model (DuT) on BDD100k dataset

**What happens when you run it**: 
- Creates CLIP embeddings for all BDD100k val images. It must be noted that the pipeline starts with embeddings cropped images of pedestrians from autonomous driving scene. The crops must be generated beforehand using seperate tool. 
- Creates metadata for given ODD, prompts for all embeddings (very minor variations in metadata might occur due to [CLIP reproducability issues](https://github.com/openai/CLIP/issues/13). To get exact slice sizes as in paper, please contact me to get metadata csv for CLIP)
- Gets the inference values for ConvNeXT - [inference](https://github.com/SysCV/bdd100k-models/tree/main/det)
- Identifies weak slices with CLIP (SWD-1), Corrected (SWD-2). The merged (SWD-3) result is just a combination of both of these by removing duplicates and ordering by slice_score.
- For corrected (SWD-2), the precision and recall values are taken for celebA from the data/precision_and_recall_csvs/precisions_and_recall_celebA.csv- 
- The bdd100k_config.py contains all the relevant parameters and file paths. Dataset root path and a path to save the results should be provided. Intermediate csv files are stored here while building embeddings, metadata, inference results and finally the identified slices. 

**Usage**:
```bash
python run_full_bdd100k_experiment.py
```

