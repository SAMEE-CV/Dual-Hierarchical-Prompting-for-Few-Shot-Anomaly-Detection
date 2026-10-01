# Dual-Prompt: Hierarchical Multimodal Prompting for Industrial Anomaly Detection

PyTorch implementation of **“Multimodal Representation Learning via Dual Hierarchical Soft Prompting for Few-Shot Industrial Anomaly Detection.”**

## Highlights

- Hierarchical soft prompts in the frozen CLIP text and vision transformers.
- Cross-modal coupling from language prompts to visual prompts.
- Few-shot training using only normal MVTec AD images.
- Multi-level normal feature memories for pixel-level localization.
- Image and pixel AUROC/AUPR evaluation with saved anomaly maps.

## Method

Normal and defective descriptions are encoded using prompted CLIP text features. Coupled visual prompts adapt early, middle, and late ViT blocks. During inference, semantic image-text similarity is combined with nearest-neighbor distances from normal patch-feature memories.

See [METHOD.md](docs/METHOD.md) for the essential implementation notes.

## Installation

Follow [INSTALL.md](docs/INSTALL.md).

## Data Preparation

Follow [DATASETS.md](docs/DATASETS.md) for the MVTec AD layout.

## Training and Evaluation

Follow [RUN.md](docs/RUN.md).

Quick example:

```bash
python -m dual_prompt train \
  --config configs/trainers/DualPrompt/vit_b16_4shot.yaml \
  --data-root /path/to/mvtec_anomaly_detection \
  --category bottle
```

## Repository Structure

```text
configs/trainers/DualPrompt/   Experiment configurations
docs/                          Installation, data, method, and run guides
scripts/dual_prompt/           MVTec launch scripts
src/dual_prompt/               Model, data, training, memory, and metrics
tests/                         Unit and end-to-end smoke tests
```

## Citation

Please cite the Dual-Prompt paper when using this project. Software citation metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

This implementation uses CLIP representations and the MVTec AD benchmark.

## License

Code is released under the [MIT License](LICENSE). MVTec AD and pretrained weights retain their own licenses.
