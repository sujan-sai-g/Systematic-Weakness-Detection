import sys
import os
import yaml
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.metadata.metadata_factory import MetadataFactory
from src.models import ModelFactory
from src.sliceline import run_three_approaches, run_two_approaches, clean_slice_results
import pandas as pd
import glob


class BDD100kPipeline:
    """
    Complete pipeline for BDD100k experiments including:
    1. CLIP embeddings generation
    2. Metadata generation
    3. Model inference
    4. SliceLine analysis
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = self.config['output']
        self._setup_output_directory()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from Python file"""
        config_dir = os.path.dirname(self.config_path)
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        import sys
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
        
        config_module = __import__(config_name)
        return config_module.config
        
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def _print_step_header(self, step_name: str, step_number: int, total_steps: int):
        """Print formatted step header"""
        print("\n" + "=" * 80)
        print(f"STEP {step_number}/{total_steps}: {step_name}")
        print("=" * 80)     
    
    
    def step1_generate_embeddings(self) -> str:
        """Step 1: Generate CLIP embeddings for images"""
        self._print_step_header("Generate CLIP Embeddings", 1, 4)
        
        embeddings_config = self.config.get('metadata', {})
        dataset_config = self.config.get('dataset', {})
        include_instance_id = self.config.get('include_instance_id', False)
        # Build embedding generator configuration
        generator_config = {
            'device': self.config['model'].get('device', 'cuda:0'),
            'clip_model': embeddings_config.get('clip_model', 'ViT-L/14'),
            'seed': embeddings_config.get('seed', 100),
            'include_instance_id': include_instance_id
        }
        
        # Set paths
        image_folder = dataset_config.get('pedestrian_crops_folder')
        image_pattern = dataset_config.get('image_pattern', '*.png')
        output_filename = embeddings_config.get('image_embeddings_path')
        output_path = os.path.join(output_filename)
        
        # Print configuration
        print(f"CLIP Model: {generator_config['clip_model']}")
        print(f"Image folder: {image_folder}")
        print(f"Image pattern: {image_pattern}")
        print(f"Output path: {output_path}")
        print(f"Device: {generator_config['device']}")
        
        # Validate required paths
        if not image_folder or not os.path.exists(image_folder):
            raise ValueError(f"Image folder does not exist: {image_folder}")
        
        # Skip if embeddings already exist
        if os.path.exists(output_path):
            print(f"✓ Embeddings already exist at {output_path}")
            embeddings_df = pd.read_pickle(output_path)
            print(f"✓ Loaded {len(embeddings_df)} existing embeddings")
            return output_path
        
        # Create embedding generator using factory
        print("Creating embedding generator...")
        generator = MetadataFactory.create_embedding_generator(generator_config)
        
        # Generate embeddings
        print("Starting embedding generation...")
        embeddings_df = generator.generate_embeddings_from_folder(
            image_folder=image_folder,
            output_path=output_path,
            pattern=image_pattern
        )
        
        print(f"✓ Generated embeddings for {len(embeddings_df)} images")
        print(f"✓ Saved to: {output_path}")
        
        return output_path
    
    def step2_generate_metadata(self, embeddings_path: str) -> str:
        """Step 2: Generate metadata using CLIP embeddings"""
        self._print_step_header("Generate Metadata", 2, 4)
        
        metadata_config = self.config.get('metadata', {})
        include_instance_id = self.config.get('include_instance_id', False)

        # Build metadata generator configuration
        generator_config = {
            'seed': self.config.get('seed', 100),
            'device': self.config.get('device', 'cuda:0'),
            'image_embeddings_path': embeddings_path,
            'ontology_path': metadata_config.get('ontology_path'),
            'prompts_path': metadata_config.get('prompts_path'),
            'object': metadata_config.get('object'),
            'output_path': os.path.join(self.output_dir, metadata_config.get('metadata_file')),
            'include_instance_id': include_instance_id

        }
        
        # Print configuration
        print(f"Object type: {generator_config['object']}")
        print(f"Image embeddings: {generator_config['image_embeddings_path']}")
        print(f"Ontology path: {generator_config['ontology_path']}")
        print(f"Prompts path: {generator_config['prompts_path']}")
        print(f"Output path: {generator_config['output_path']}")
        
        # Validate required fields
        required_fields = ['image_embeddings_path', 'ontology_path', 'prompts_path', 'object']
        for field in required_fields:
            if not generator_config.get(field):
                raise ValueError(f"Missing required field '{field}' in metadata configuration")
        
        # Skip if metadata already exists
        if os.path.exists(generator_config['output_path']):
            print(f"✓ Metadata already exists at {generator_config['output_path']}")
            metadata_df = pd.read_csv(generator_config['output_path'])
            print(f"✓ Loaded {len(metadata_df)} existing metadata records")
            return generator_config['output_path']
        
        # Create metadata generator using factory
        print("Creating metadata generator...")
        generator = MetadataFactory.create_metadata_generator(generator_config)
        
        # Generate metadata
        print("Starting metadata generation...")
        metadata_df = generator.run()
        
        print(f"✓ Generated metadata for {len(metadata_df)} images")
        print(f"✓ Columns: {list(metadata_df.columns)}")
        print(f"✓ Saved to: {generator_config['output_path']}")
        
        return generator_config['output_path']
    
    def step3_run_inference(self) -> str:
        """Step 3: Run model inference on images"""
        self._print_step_header("Run Model Inference", 3, 4)
        
        # Get image paths
        dataset_config = self.config['dataset']
        debug_config = self.config.get('debug', {})
        max_images = debug_config.get('max_images') if debug_config.get('enabled', False) else None
        
        # This step is not needed as we obtain the inference from the website https://github.com/SysCV/bdd100k-models/tree/main/det
        # image_paths = self._get_image_paths(max_images)
        
        # if not image_paths:
        #     raise ValueError("No images found!")
        image_paths = None
        
        # Create model
        print(f"Creating model: {self.config['model']['model_name']}")
        model = ModelFactory.create(self.config['model'])
        
        # Run inference
        output_csv = os.path.join(self.output_dir, self.config["model"]['matched_predictions_csv'])
        print(f"Output CSV: {output_csv}")
        
        # Skip if predictions already exist
        if os.path.exists(output_csv):
            print(f"✓ Matched Predictions already exist at {output_csv}")
            df = pd.read_csv(output_csv)
            print(f"✓ Loaded {len(df)} existing matched predictions")
            return output_csv
        
        print("Running inference...")
        df = model.predict_to_csv(image_paths, output_csv)
        
        print(f"✓ Generated matched predictions for {len(df)} images")
        print(f"✓ CSV saved to: {output_csv}")
        
        return output_csv
    
    def step3_5_merge_data(self, metadata_path: str, predictions_path: str) -> str:
        """Step 3.5: Merge metadata and predictions data"""                
        
        self._print_step_header("Merge Metadata and Predictions", 3.5, 4)
        
        # Load both datasets
        metadata_df = pd.read_csv(metadata_path)
        predictions_df = pd.read_csv(predictions_path)
        gt_df = pd.read_csv(self.config['model']['labels_csv_path'])
        
        # Merge on image path (adjust column names as needed)
        merged_df = pd.merge(metadata_df, predictions_df, left_on=['Filename', 'instance_id'], right_on=['Filename', 'instance_id_match'], how='outer')
        
        merged_df = pd.merge(merged_df, gt_df, left_on=['Filename', 'instance_id'], right_on=['Filename', 'id'], how='outer')

        # Sort and drop unnecessary columns safely
        merged_df = merged_df.sort_values(by=["Filename", "instance_id"])
               
        # Handle missing values - replace empty strings with pd.NA
        columns_to_clean = ["x1_list", "pred_bbox_x1", "gender"]
        for col in columns_to_clean:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].replace("", pd.NA)
        
        # Initialize detection_type column
        merged_df["detection_type"] = None
        
        # Classify detection types based on the logic from process_merged_data
        if "x1_list" in merged_df.columns and "gender" in merged_df.columns:
            # False Positive: missing ground truth annotations
            merged_df.loc[
                merged_df["x1_list"].isna() | merged_df["gender"].isna(), "detection_type"
            ] = "FP"
        
        if "pred_bbox_x1" in merged_df.columns:
            # False Negative: missing predictions
            merged_df.loc[merged_df["pred_bbox_x1"].isna(), "detection_type"] = "FN"
        
        # True Positive: everything else (has both ground truth and predictions)
        merged_df.loc[merged_df["detection_type"].isnull(), "detection_type"] = "TP"
        
        # Reset index and drop old index column
        merged_df.reset_index(drop=True, inplace=True)       
       
        # Save to clip_data_path
        clip_data_path = self.config['sliceline_input']['clip_data_path']
        merged_df.to_csv(clip_data_path, index=False)
        
        print(f"✓ Detection type distribution:")
        print(merged_df["detection_type"].value_counts())
        print(f"✓ Merged data saved to: {clip_data_path}")
        
        return clip_data_path
    
    def step4_sliceline_analysis(self) -> tuple:
        """Step 4: Run SliceLine analysis"""
        self._print_step_header("SliceLine Analysis", 4, 4)
        
        # Load and prepare data
        data_config = self.config['sliceline_input']
        
        print("Loading datasets...")
        df_clip = pd.read_csv(data_config['clip_data_path'])
        
        # Check if ground truth data exists
        gt_data_path = data_config.get('gt_data_path')
        if gt_data_path and os.path.exists(gt_data_path):
            print("Ground truth data found, loading...")
            df_gt = pd.read_csv(gt_data_path)
            has_ground_truth = True
            
            # Data preprocessing
            if 'No_Beard' in df_gt.columns:
                df_gt["Beard"] = df_gt["No_Beard"].apply(lambda x: 1 if x == 0 else 0)
            
            # Merge datasets
            df = pd.merge(df_clip, df_gt, on="img_path", how="inner", suffixes=(None, "_drop"))
            df = df.loc[:, ~df.columns.str.endswith("_drop")]
        else:
            print("No ground truth data found, using CLIP data only...")
            df = df_clip.copy()
            has_ground_truth = False
        df = df[df["detection_type"].isin(["TP", "FN"])]    
        df["detection_type"] = df["detection_type"].map({"FN": 1, "TP": 0})
        
        
        df["size"] = (df["x2_list"] - df["x1_list"]) * (df["y2_list"] - df["y1_list"])
                                           
        df = df[df["size"] > 3000] ### Ensure only large pedestrians remain
        df = df[~df['gender'].isna()]

        # Load precision data
        clip_precisions = pd.read_csv(data_config['precision_data_path'])
        
        print(f"✓ Loaded {len(df)} samples")
        
        # Calculate error rate
        error_rate = df["detection_type"].sum() / len(df["detection_type"])
        print(f"Overall error rate: {error_rate:.3f}")
        
        # Run SliceLine analysis
        sliceline_config = self.config['sliceline']
        label_map = self.config['label_mapping']
        
        df[list(label_map.values())] = (
            df[list(label_map.values())]
            .apply(pd.to_numeric, errors="coerce")
            .astype("int64")
        )
        
        # Extract column names
        gt_columns = list(label_map.keys())
        clip_columns = list(label_map.values())
        
        print(f"GT columns: {gt_columns}")
        print(f"CLIP columns: {clip_columns}")
        print(f"Level: {sliceline_config['level']}, Alpha: {sliceline_config['alpha']}")
        
        # Check if ground truth data is available AND columns exist in dataframe
        use_ground_truth = (has_ground_truth and 
                        len(gt_columns) > 0 and 
                        all(col in df.columns for col in gt_columns))
        
        if use_ground_truth:
            print("Running three approaches (with ground truth)...")
            # Run three approaches
            slices_gt, slices_clip, slices_corrected = run_three_approaches(
                df=df,
                gt_columns=gt_columns,
                clip_columns=clip_columns,
                clip_precisions=clip_precisions,
                label_map=label_map,
                level=sliceline_config['level'],
                alpha=sliceline_config['alpha'],
                top_k=sliceline_config['top_k'],
                err_col_name=sliceline_config['err_column'],
                inv_threshold=sliceline_config['inv_thresh']
            )
            
            # Clean results
            slices_gt_clean, slices_clip_clean, slices_corrected_clean = clean_slice_results(
                slices_gt, slices_clip, slices_corrected, error_rate
            )
            
            print(f"✓ Found {len(slices_gt_clean)} GT slices")
            print(f"✓ Found {len(slices_clip_clean)} CLIP slices")
            print(f"✓ Found {len(slices_corrected_clean)} corrected slices")
            
            # Save results
            output_files = {
                'gt_slices.csv': slices_gt_clean,
                'clip_slices.csv': slices_clip_clean,
                'corrected_slices.csv': slices_corrected_clean
            }
            
            print("Saving SliceLine results...")
            for filename, df_result in output_files.items():
                output_path = os.path.join(self.output_dir, filename)
                df_result.to_csv(output_path, index=False)
                print(f"✓ Saved {filename} with {len(df_result)} slices")
            
            return slices_gt_clean, slices_clip_clean, slices_corrected_clean
            
        else:
            print("Running two approaches (no ground truth available)...")
            # Run two approaches
            slices_clip, slices_corrected = run_two_approaches(
                df=df,
                clip_columns=clip_columns,
                clip_precisions=clip_precisions,
                label_map=label_map,
                level=sliceline_config['level'],
                alpha=sliceline_config['alpha'],
                top_k=sliceline_config['top_k'],
                err_col_name=sliceline_config['err_column'],
                inv_threshold=sliceline_config['inv_thresh']
            )
            
            # Clean results
            slices_clip_clean, slices_corrected_clean = clean_slice_results(
                None, slices_clip, slices_corrected, error_rate
            )
            
            print(f"✓ Found {len(slices_clip_clean)} CLIP slices")
            print(f"✓ Found {len(slices_corrected_clean)} corrected slices")
            
            # Save results
            output_files = {
                'clip_slices.csv': slices_clip_clean,
                'corrected_slices.csv': slices_corrected_clean
            }
            
            print("Saving SliceLine results...")
            for filename, df_result in output_files.items():
                output_path = os.path.join(self.output_dir, filename)
                df_result.to_csv(output_path, index=False)
                print(f"✓ Saved {filename} with {len(df_result)} slices")
            
            return None, slices_clip_clean, slices_corrected_clean
    
    def print_final_summary(self, slices_gt, slices_clip, slices_corrected):
        """Print final experiment summary"""
        print("\n" + "=" * 80)
        print("BDD100k EXPERIMENT SUMMARY")
        print("=" * 80)
        
        # Top slices summary
        approaches = []
        if slices_gt is not None:
            approaches.append(("Ground Truth", slices_gt))
        approaches.extend([("CLIP", slices_clip), ("Corrected", slices_corrected)])
        
        for name, df in approaches:
            print(f"\n{name} - Top 5 slices:")
            if len(df) > 0:
                top_5 = df.head(5)[['slice_score', 'slice_average_error', 'slice_size']]
                print(top_5.to_string(index=False))
            else:
                print("No slices found")
        
        print(f"\n✓ All results saved to: {self.output_dir}")
        print("=" * 80)
    
    def run(self, skip_existing: bool = True):
        """Run the complete BDD100k pipeline"""
        start_time = time.time()
        
        print("=" * 80)
        print(f"RUNNING BDD100k PIPELINE: {self.config['experiment']['name']}")
        print("=" * 80)
        print(f"Skip existing files: {skip_existing}")
        
        try:
            # # Step 1: Generate embeddings
            embeddings_path = self.step1_generate_embeddings()
            
            # Step 2: Generate metadata
            metadata_path = self.step2_generate_metadata(embeddings_path)
            
            # Step 3: Run inference
            predictions_path = self.step3_run_inference()
            
            # Step 3.5: Merge metadata and predictions
            self.step3_5_merge_data(metadata_path, predictions_path)
            
            # Step 4: SliceLine analysis
            slices_gt, slices_clip, slices_corrected = self.step4_sliceline_analysis()
            
            # Print final summary
            self.print_final_summary(slices_gt, slices_clip, slices_corrected)
            
            # Calculate total time
            total_time = time.time() - start_time
            print(f"\n✓ Pipeline completed successfully in {total_time/60:.1f} minutes!")
            
            return {
                'embeddings_path': embeddings_path,
                'metadata_path': metadata_path,
                'predictions_path': predictions_path,
                'slices_gt': slices_gt,
                'slices_clip': slices_clip,
                'slices_corrected': slices_corrected
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise


def main():
    """Main function to run the BDD100k pipeline"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "bdd100k_config.py") 
    try:
        # Create and run pipeline
        pipeline = BDD100kPipeline(config_path)
        results = pipeline.run(skip_existing=True)
        
        print("\n🎉 BDD100k experiment pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()