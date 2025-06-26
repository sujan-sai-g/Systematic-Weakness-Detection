# Detecting Systematic Weaknesses in Vision Models along Predefined Human-Understandable Dimensions

## Abstract
Slice discovery methods (SDMs) are prominent algorithms for finding systematic weaknesses in DNNs. They identify top-k semantically coherent slices/subsets of data where a DNN-under-test has low performance. For being directly useful, slices should be aligned with human-understandable and relevant dimensions, which, for example, are defined by safety and domain experts as part of the operational design domain (ODD). While SDMs can be applied effectively on structured data, their application on image data is complicated by the lack of semantic metadata. To address these issues, we present an algorithm that combines foundation models for zero-shot image classification to generate semantic metadata with methods for combinatorial search to find systematic weaknesses in images. In contrast to existing approaches, ours identifies weak slices that are in line with pre-defined human-understandable dimensions. As the algorithm includes foundation models, its intermediate and final results may not always be exact. Therefore, we include an approach to address the impact of noisy metadata. We validate our algorithm on both synthetic and real-world datasets, demonstrating its ability to recover human-understandable systematic weaknesses. Furthermore, using our approach, we identify systematic weaknesses of multiple pre-trained and publicly available state-of-the-art computer vision DNNs.

Slice Discovery, Vision Model Testing




## Paper Information
- **Journal**: TMLR 2025
- **Paper Link**: https://openreview.net/forum?id=yK9pvt4nBX
- **ArXiv**: https://arxiv.org/abs/2502.12360

## Quick Start
```bash
# Clone the repository
git clone [your-repo-url]
cd [repo-name]

# Install dependencies
pip install -e .
```

# Usage
## Run celebA experiment
- Creates CLIP embeddings for all celebA images
- Creates metadata for given ODD, prompts for all embeddings (very minor variations in metadata might occur due to [CLIP reproducability issues](https://github.com/openai/CLIP/issues/13). To get exact slice sizes as in paper, please contact me to get metadata csv for CLIP)
- Runs inference on ViT model (DuT) for all celebA images
- Identifies weak slices with CLIP (SWD-1), Corrected (SWD-2) and the merged SWD-3
- Additionally GT slices can be calculated if the GT metadata is provided as csv (path defined in config) -- please contact if you are interested in getting this csv

```bash
python experiments/run_full_celeba_experiment.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses
This project uses OpenAI's CLIP model. See [OpenAI CLIP License](https://github.com/openai/CLIP/blob/main/LICENSE) for details.

## Contact
For questions or issues, please open an issue on GitHub.

