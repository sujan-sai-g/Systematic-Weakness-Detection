import os
import sys
import base64
import glob
import csv
import random
from typing import Dict, Any, List
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

# Add data directory to path for importing gpt_odd
data_dir = Path(__file__).parent.parent.parent / "data/ODDs/"
sys.path.append(str(data_dir))

from gpt_odd_celeba import ODDImage


class GPTMetadataGenerator:
    """GPT-based metadata generator for image annotation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("openai_api_key", "")
        self.model_name = config.get("model_name", "gpt-4o-2024-08-06")
        self.images_path = config["images_path"]
        self.output_path = config["output_path"]
        self.batch_size = config.get("batch_size", None)
        
        # Initialize OpenAI client
        if not self.api_key:
            raise ValueError("OpenAI API key is required in config['openai_api_key']")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Setup field mappings
        self.setup_field_mappings()
    
    def setup_field_mappings(self):
        """Setup field names and binary mappings"""
        self.fieldnames = [
            "filename",
            "Gender",
            "Age", 
            "Pale_Skin",
            "Wearing_hat",
            "Goatee",
            "Beard",
            "Smiling",
            "Eye_glasses",
            "Bald",
        ]
        
        self.binary_mapping = {
            "Gender": {"female": 0, "male": 1},
            "Age": {"adult": 0, "young": 1},
            "Pale_Skin": {"pale": 1, "not_pale": 0},
            "Wearing_hat": {"false": 0, "true": 1},
            "Goatee": {"false": 0, "true": 1},
            "Beard": {"false": 0, "true": 1},
            "Smiling": {"false": 0, "true": 1},
            "Eye_glasses": {"false": 0, "true": 1},
            "Bald": {"false": 0, "true": 1},
        }
    
    def encode_image(self, image_path: str) -> str:
        """Convert an image to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def process_single_image(self, image_path: str) -> tuple[Dict[str, Any], float]:
        """Process a single image and return metadata and cost"""
        base64_image = self.encode_image(image_path)
        
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at generating structured data about an image. You will be given an image and should extract data from it based on the given structure.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        }
                    ],
                },
            ],
            response_format=ODDImage,
        )
        
        assigned_odd = completion.choices[0].message.parsed
        
        # Calculate cost
        request_cost = (completion.usage.prompt_tokens * 2.5 / 1000000) + (
            completion.usage.completion_tokens * 10 / 1000000
        )
        
        # Prepare row data
        row = {"filename": os.path.basename(image_path)}
        
        # Get binary values for each attribute
        for attr_name in self.fieldnames[1:]:  # Skip 'filename'
            enum_value = getattr(assigned_odd, attr_name)
            text_value = enum_value.value.lower()
            row[attr_name] = self.binary_mapping[attr_name][text_value]
        
        return row, request_cost
    
    def run(self):
        """Generate metadata for all images and save to CSV"""
        # Get list of image filenames
        filenames = glob.glob(self.images_path + "*.png")
        if not filenames:
            filenames = glob.glob(self.images_path + "*.jpg")
        if not filenames:
            filenames = glob.glob(self.images_path + "*.jpeg")
        
        random.shuffle(filenames)
        
        # Limit batch size if specified
        if self.batch_size and self.batch_size < len(filenames):
            filenames = filenames[:self.batch_size]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Check if file exists to decide whether to write headers
        file_exists = os.path.isfile(self.output_path)
        
        total_cost = 0
        processed_count = 0
        
        # Open the CSV file in append mode
        with open(self.output_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            
            # Write header only if the file doesn't exist yet
            if not file_exists:
                writer.writeheader()
            
            # Process each image
            for image_path in tqdm(filenames, desc="Processing images"):
                try:
                    row, request_cost = self.process_single_image(image_path)
                    writer.writerow(row)
                    total_cost += request_cost
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images, cost so far: ${total_cost:.4f}")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        print(f"Processing complete. Total cost: ${total_cost:.4f}")
        print(f"Results saved to: {self.output_path}")
        
        return self.output_path