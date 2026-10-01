# Reproducibility protocol

## Evaluation unit

Train a separate prompt model and separate feature memories for each MVTec category. Use only the seeded few-shot subset of `train/good`. Do not use test images, anomaly labels, or masks for adaptation or hyperparameter selection.

## Seeded support selection

The loader sorts normal file paths and applies Python's local seeded sampler. The selected paths can be recovered from the checkpoint's category/configuration and the dataset version. For publication runs, archive the resulting `run_metadata.json`, package versions, and a text list of support paths if the dataset may be reorganized.

## Preprocessing

Images are converted to RGB, resized to a square with bicubic interpolation and antialiasing, converted to tensors, and normalized with the standard CLIP mean and standard deviation. Masks use nearest-neighbor resize and binary thresholding.

## Reported metrics

- Image AUROC and average precision over all category test images.
- Pixel AUROC and average precision over all resized test pixels.
- Global pixel Dice at the configured fixed threshold.

The manuscript reports AUROC and, for the carbon-fiber experiments, AUPR and Dice. This repository emits all five metrics for transparency. It does not choose a Dice threshold on the test set.

## Recommended experiment record

Record the Git commit, Python/PyTorch/OpenCLIP versions, CUDA version, GPU model, YAML configuration, random seed, MVTec release, category, sampled support paths, complete logs, checkpoint, and raw per-image prediction CSV.

## Result interpretation

The one-epoch command is an integration check, not a convergence claim. Comparisons against paper tables require the full declared protocol and should disclose the manuscript ambiguities listed in the README.

