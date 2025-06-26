import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.metadata.metadata_factory import MetadataFactory
import pandas as pd


class GPTMetadataPipeline:
    """
    Simple pipeline for GPT-based metadata generation
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
    
    def _print_step_header(self, step_name: str):
        """Print formatted step header"""
        print("\n" + "=" * 80)
        print(f"GPT METADATA GENERATION: {step_name}")
        print("=" * 80)
    
    def generate_gpt_metadata(self) -> str:
        """Generate metadata using GPT"""
        self._print_step_header("Starting GPT Metadata Generation")
        
        # Get dataset configuration
        dataset_config = self.config['dataset']
        metadata_config = self.config.get('metadata', {})
        
        # Build GPT metadata generator configuration
        generator_config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'model_name': metadata_config.get('gpt_model', 'gpt-4o-2024-08-06'),
            'images_path': dataset_config.get('image_folder'),
            'output_path': os.path.join(self.output_dir, 'gpt_metadata.csv'),
            'batch_size': metadata_config.get('batch_size', 2)
        }
        
        # Print configuration
        print(f"GPT Model: {generator_config['model_name']}")
        print(f"Images folder: {generator_config['images_path']}")
        print(f"Output path: {generator_config['output_path']}")
        print(f"Batch size: {generator_config['batch_size'] or 'All images'}")
        
        # Validate required fields
        if not generator_config['openai_api_key']:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        if not generator_config['images_path'] or not os.path.exists(generator_config['images_path']):
            raise ValueError(f"Images folder does not exist: {generator_config['images_path']}")
        
        # Skip if metadata already exists
        if os.path.exists(generator_config['output_path']):
            print(f"✓ GPT metadata already exists at {generator_config['output_path']}")
            metadata_df = pd.read_csv(generator_config['output_path'])
            print(f"✓ Loaded {len(metadata_df)} existing metadata records")
            return generator_config['output_path']
        
        # Create GPT metadata generator using factory
        print("Creating GPT metadata generator...")
        generator = MetadataFactory.create_gpt_metadata_generator(generator_config)
        
        # Generate metadata
        print("Starting GPT metadata generation...")
        result_path = generator.run()
        
        # Load and display results
        metadata_df = pd.read_csv(result_path)
        print(f"✓ Generated metadata for {len(metadata_df)} images")
        print(f"✓ Columns: {list(metadata_df.columns)}")
        print(f"✓ Saved to: {result_path}")
        
        return result_path
    
    def run(self):
        """Run the GPT metadata generation pipeline"""
        start_time = time.time()
        
        print("=" * 80)
        print(f"RUNNING GPT METADATA PIPELINE: {self.config['experiment']['name']}")
        print("=" * 80)
        
        try:
            # Generate GPT metadata
            metadata_path = self.generate_gpt_metadata()
            
            # Calculate total time
            total_time = time.time() - start_time
            print(f"\n✓ Pipeline completed successfully in {total_time/60:.1f} minutes!")
            
            return metadata_path
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            raise


def main():
    """Main function to run the GPT metadata pipeline"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "celeba_config.py") 
    
    try:
        # Create and run pipeline
        pipeline = GPTMetadataPipeline(config_path)
        result = pipeline.run()
        
        print("\n🎉 GPT metadata generation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()