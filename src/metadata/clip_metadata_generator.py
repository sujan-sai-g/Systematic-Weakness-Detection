import os
import sys
import clip
import numpy as np
import pandas as pd
import torch
import tqdm
from typing import Dict, Any, List, Tuple, Optional
import importlib.util


class CLIPAnnotator:
    """CLIP-based annotation system for generating semantic scores"""

    def __init__(self, config: Dict[str, Any]):
        # Add seed setting
        seed = config.get('seed', 100)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_embeddings_path = config["image_embeddings_path"]
        self.image_embeddings_df = pd.read_pickle(self.image_embeddings_path)
        self.load_clip_model()
        self.convert_image_embeddings_to_features()

    def load_clip_model(self):
        """Load CLIP model and preprocessing"""
        model_name = "ViT-L/14"
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def convert_image_embeddings_to_features(self):
        """Convert image embeddings from dataframe to tensor features"""
        image_tensor = torch.tensor(self.image_embeddings_df["image_features_list"])
        self.image_features = torch.squeeze(image_tensor).to(self.device)
        print(f"Loaded image features shape: {self.image_features.shape}")

    def convert_text_embeddings_to_features(self, prompts: List[str]):
        """Convert text prompts to CLIP features"""
        text = clip.tokenize(prompts).to(self.device)
        self.text_features = self.model.encode_text(text)

    def calculate_similarity(self) -> torch.Tensor:
        """Calculate similarity between image and text features"""
        with torch.no_grad():
            # Normalize features
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * self.image_features @ self.text_features.T).softmax(
                dim=-1
            )
            self.logits = self.image_features @ self.text_features.T
        return similarity

    def get_scores(
        self, prompts: List[str], num_attributes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction scores, confidence, and entropy for given prompts

        Args:
            prompts: List of text prompts
            num_attributes: Number of attributes to predict

        Returns:
            Tuple of (predictions, confidence, entropy)
        """
        self.convert_text_embeddings_to_features(prompts)
        similarity = self.calculate_similarity()

        num_prompts_per_attribute = int((len(prompts) / num_attributes))
        # Calculate according to Gannamaneni et al. (2023)
        # TODO add paper link

        # Reshape similarity to group by attributes
        reshaped_similarity = similarity.reshape(
            -1, num_attributes, num_prompts_per_attribute
        )

        # Calculate mean similarity across prompts for each attribute
        mean_similarity = reshaped_similarity.mean(axis=-1)
        mean_similarity_softmax = torch.nn.functional.softmax(mean_similarity, dim=1)

        def normalize_by_sum(x):
            return x / np.sum(x, axis=1, keepdims=True)

        # Get predictions and confidence
        percentage_values_sum = normalize_by_sum(mean_similarity.cpu().numpy())
        preds = np.argmax(mean_similarity.cpu().numpy(), axis=1)
        conf = percentage_values_sum[np.arange(percentage_values_sum.shape[0]), preds]

        # Calculate entropy
        softmax_np = mean_similarity_softmax.cpu().numpy()
        entropy = -np.sum(np.log(softmax_np + 1e-8) * softmax_np, axis=1)

        return preds, conf, entropy


class MetadataGenerator:
    """
    Main metadata generator that creates CSV files with object metadata
    using CLIP-based semantic annotation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_path = config["output_path"]
        self.object_under_study = config["object"]
        self.ontology_path = config["ontology_path"]
        self.prompts_path = config.get("prompts_path", "data/prompts")
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load image embeddings
        self.image_embeddings_df = pd.read_pickle(config["image_embeddings_path"])
        print(f"Loaded {len(self.image_embeddings_df)} image embeddings")

        # Initialize CLIP annotator
        self.clip_annotator = CLIPAnnotator(config)

        # Setup ontology and prompts
        self.load_ontology()
        self.load_prompts()
        self.extract_semantic_dimensions_from_ontology()
        self.initialize_dataframe()

    def load_ontology(self):
        """Load ontology file based on object type"""
        ontology_files = {
            "tracks": "tracks_ontology.csv",
            "celebA_person": "celebA_ontology.csv",
            "ad_person": "pedestrian_ontology.csv",
            "dummy": "dummy_ontology.csv",
        }

        ontology_file = ontology_files.get(self.object_under_study)
        if not ontology_file:
            raise ValueError(f"Unknown object type: {self.object_under_study}")

        ontology_full_path = os.path.join(self.ontology_path, ontology_file)
        if not os.path.exists(ontology_full_path):
            raise FileNotFoundError(f"Ontology file not found: {ontology_full_path}")

        self.ontology = pd.read_csv(ontology_full_path)
        print(f"Loaded ontology from: {ontology_full_path}")

        # Calculate number of attributes per semantic dimension
        # Count non-empty values in each row (excluding the sem-dim column)
        attribute_columns = [col for col in self.ontology.columns if col != "sem-dim"]
        self.ontology["num_attributes"] = (
            self.ontology[attribute_columns].notna().sum(axis=1)
        )

        print(f"Ontology contains {len(self.ontology)} semantic dimensions")

    def load_prompts(self):
        """Dynamically load prompts module based on object type"""
        prompt_files = {
            "celebA_person": "prompts_celebA.py",
            "ad_person": "prompts_pedestrians.py",
        }

        prompt_file = prompt_files.get(self.object_under_study)
        if not prompt_file:
            raise ValueError(
                f"No prompts file defined for object type: {self.object_under_study}"
            )

        prompt_full_path = os.path.join(self.prompts_path, prompt_file)
        if not os.path.exists(prompt_full_path):
            raise FileNotFoundError(f"Prompts file not found: {prompt_full_path}")

        # Dynamically import the prompts module
        spec = importlib.util.spec_from_file_location(
            "prompts_module", prompt_full_path
        )
        prompts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts_module)

        if not hasattr(prompts_module, "prompts"):
            raise AttributeError(
                f"Prompts file {prompt_file} must contain a 'prompts' dictionary"
            )

        self.prompts = prompts_module.prompts
        print(f"Loaded prompts from: {prompt_full_path}")
        print(f"Available prompt categories: {list(self.prompts.keys())}")

    def extract_semantic_dimensions_from_ontology(self):
        """Extract semantic dimensions from ontology"""
        self.semantic_dimensions = self.ontology["sem-dim"].tolist()
        print(f"Semantic dimensions: {self.semantic_dimensions}")

    def generate_prompts_for_semantic_dimension(self, semantic_dim: str) -> List[str]:
        """
        Generate prompts for a given semantic dimension

        Args:
            semantic_dim: The semantic dimension to generate prompts for

        Returns:
            List of prompts for the semantic dimension
        """
        if semantic_dim not in self.prompts:
            raise KeyError(f"No prompts found for semantic dimension: {semantic_dim}")

        prompts_for_dim = self.prompts[semantic_dim]
        print(f"Generated {len(prompts_for_dim)} prompts for '{semantic_dim}'")

        return prompts_for_dim

    def initialize_dataframe(self):
        """Initialize the output dataframe with proper structure"""
        # Create columns for each semantic dimension
        col_names = {col: [] for col in self.semantic_dimensions}
        self.df = pd.DataFrame(col_names)

        # Add filename (and potentially instance_id if available)
        self.df["Filename"] = self.image_embeddings_df["Filename"]

        # Add instance_id if it exists in embeddings
        if "instance_id" in self.image_embeddings_df.columns:
            self.df["instance_id"] = self.image_embeddings_df["instance_id"]
            metadata_columns = ["Filename", "instance_id"]
        else:
            metadata_columns = ["Filename"]

        # Reorder columns to put metadata columns first
        semantic_columns = [
            col for col in self.df.columns if col not in metadata_columns
        ]
        self.df = self.df[metadata_columns + semantic_columns]

        print(
            f"Initialized dataframe with {len(self.df)} rows and columns: {list(self.df.columns)}"
        )

    def validate_prompts_and_ontology(
        self, semantic_dim: str, prompts_per_dim: List[str], num_attributes: int
    ):
        """Validate that prompts align with ontology expectations"""
        expected_prompts = (
            num_attributes * 5
        )  # Assuming roughly 5 prompts per attribute

        if len(prompts_per_dim) % num_attributes != 0:
            print(
                f"Warning: Number of prompts ({len(prompts_per_dim)}) for '{semantic_dim}' "
                f"is not evenly divisible by number of attributes ({num_attributes})"
            )

        prompts_per_attr = len(prompts_per_dim) // num_attributes
        print(
            f"'{semantic_dim}': {num_attributes} attributes, {prompts_per_attr} prompts per attribute"
        )

    def generate_metadata(self):
        """Main method to generate metadata for all semantic dimensions"""
        print(
            f"\nGenerating metadata for {len(self.semantic_dimensions)} semantic dimensions..."
        )

        for idx, semantic_dim in enumerate(
            tqdm.tqdm(self.semantic_dimensions, desc="Processing dimensions")
        ):
            try:
                # Get prompts for this semantic dimension
                prompts_per_dim = self.generate_prompts_for_semantic_dimension(
                    semantic_dim
                )
                num_attributes = self.ontology["num_attributes"].iloc[idx]

                # Validate alignment
                self.validate_prompts_and_ontology(
                    semantic_dim, prompts_per_dim, num_attributes
                )

                # Get scores from CLIP annotator
                scores, confidence, entropy = self.clip_annotator.get_scores(
                    prompts_per_dim, num_attributes
                )

                # Store results in dataframe
                self.df[semantic_dim] = scores
                self.df[f"{semantic_dim}_conf"] = confidence
                self.df[f"{semantic_dim}_entropy"] = entropy

                print(
                    f"Completed '{semantic_dim}': score range [{scores.min()}, {scores.max()}], "
                    f"avg confidence: {confidence.mean():.3f}"
                )

            except Exception as e:
                print(f"Error processing semantic dimension '{semantic_dim}': {e}")
                # Fill with default values to maintain dataframe structure
                self.df[semantic_dim] = -1
                self.df[f"{semantic_dim}_conf"] = 0.0
                self.df[f"{semantic_dim}_entropy"] = 0.0
                continue

        self.save_generated_metadata_file()

    def save_generated_metadata_file(self):
        """Save the generated metadata to CSV file"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"\nMetadata saved to: {self.output_path}")
        print(f"Final dataframe shape: {self.df.shape}")

    def run(self):
        """Convenience method to run the complete metadata generation pipeline"""
        print(f"Starting metadata generation for {self.object_under_study}")
        print(f"Device: {self.device}")
        print(f"Image embeddings: {len(self.image_embeddings_df)} samples")
        self.generate_metadata()
        
        return pd.read_csv(self.output_path)
